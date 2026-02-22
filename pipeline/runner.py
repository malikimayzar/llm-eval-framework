import json
import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.model_client import OllamaClient, save_query_results, QueryOutput
from pipeline.prompt_templates import PromptTemplate, TEMPLATE_VERSION
from pipeline.config_loader import config
from evaluators.faithfulness import FaithfulnessEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("runner")
DATA_DIR    = PROJECT_ROOT / "data" / "dataset"
REPORTS_DIR = PROJECT_ROOT / config["pipeline"]["results_dir"]
CHECKPOINT_DIR = PROJECT_ROOT / config["pipeline"]["checkpoint_dir"]

DATASET_FILES = {
    "clean":      DATA_DIR / "clean_cases.json",
    "distractor": DATA_DIR / "distractor_cases.json",
    "paraphrase": DATA_DIR / "paraphrase_cases.json",
}

# dataset loader
def load_dataset(dataset_name: str, limit: int = None) -> list[dict]:
    if dataset_name == "all":
        cases = []
        for name, path in DATASET_FILES.items():
            if path.exists():
                loaded = _load_json_cases(path, name)
                cases.extend(loaded)
                logger.info(f"Loaded {len(loaded)} cases from {name}")
            else:
                logger.warning(f"Dataset file not found: {path}")
        return cases[:limit] if limit else cases

    path = DATASET_FILES.get(dataset_name)
    if not path:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Valid options: {list(DATASET_FILES.keys()) + ['all']}"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Pastikan file ada di: {DATA_DIR}"
        )

    cases = _load_json_cases(path, dataset_name)
    return cases[:limit] if limit else cases

def _load_json_cases(path: Path, dataset_name: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    required_fields = {"id", "context", "question"}
    for case in cases:
        missing = required_fields - set(case.keys())
        if missing:
            raise ValueError(
                f"Case '{case.get('id', 'unknown')}' di dataset '{dataset_name}' "
                f"missing required fields: {missing}"
            )
    return cases

# checkpoint helpers — resume capability
def _checkpoint_path(model: str, dataset: str, case_id: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace(":", "-")
    return CHECKPOINT_DIR / f"{safe_model}__{dataset}__{case_id}.json"

def _load_checkpoint(model: str, dataset: str, case_id: str) -> dict | None:
    path = _checkpoint_path(model, dataset, case_id)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Checkpoint loaded | case_id={case_id}")
        return data
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Corrupt checkpoint ignored | case_id={case_id} | {e}")
        return None

def _save_checkpoint(model: str, dataset: str, case_id: str, result: dict) -> None:
    path = _checkpoint_path(model, dataset, case_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.debug(f"Checkpoint saved | case_id={case_id} | path={path}")

def _clear_checkpoints(model: str, dataset: str, case_ids: list[str]) -> None:
    cleared = 0
    for case_id in case_ids:
        path = _checkpoint_path(model, dataset, case_id)
        if path.exists():
            path.unlink()
            cleared += 1
    if cleared:
        logger.info(f"Cleared {cleared} existing checkpoints (--no-resume)")

# retry-aware query — dengan exponential backoff
def _query_with_retry(
    client: OllamaClient,
    case: dict,
    case_type: str,
    max_retries: int = None,
    backoff_base: int = None,
) -> QueryOutput:
    max_retries  = max_retries  or config["inference"]["max_retries"]
    backoff_base = backoff_base or config["inference"]["retry_backoff_base"]
    case_id      = case["id"]
    last_error   = None

    for attempt in range(max_retries):
        try:
            result = _query_with_template(client, case, case_type)
            if result.success:
                if attempt > 0:
                    logger.info(f"Retry succeeded | case_id={case_id} | attempt={attempt + 1}")
                return result
            raise RuntimeError(result.error_message or "Query returned success=False")

        except Exception as e:
            last_error = e
            wait = backoff_base ** attempt  # 1s, 2s, 4s
            logger.warning(
                f"Query attempt {attempt + 1}/{max_retries} failed | "
                f"case_id={case_id} | error={e}"
            )
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
                
    logger.error(
        f"All {max_retries} attempts failed | case_id={case_id} | "
        f"last_error={last_error}"
    )
    return QueryOutput(
        case_id=case_id,
        model=client.model,
        question=case["question"],
        context_length_chars=len(case["context"]),
        answer="",
        latency_seconds=0.0,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        success=False,
        error_message=f"All {max_retries} retries failed. Last: {last_error}",
    )

# Main evaluation pipeline
def run_evaluation(
    model_name: str,
    dataset_name: str,
    limit: int = None,
    similarity_threshold: float = None,
    resume: bool = True,
    dry_run: bool = False,
) -> dict:
    similarity_threshold = similarity_threshold or config["thresholds"]["faithfulness_evidence"]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info(
        f"Starting evaluation | run_id={run_id} | model={model_name} | "
        f"dataset={dataset_name} | limit={limit} | resume={resume} | dry_run={dry_run}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = REPORTS_DIR / f"{model_name.replace(':', '-')}_{dataset_name}_{run_id}"

    # Load dataset
    logger.info("Loading dataset...")
    cases = load_dataset(dataset_name, limit=limit)
    logger.info(f"Loaded {len(cases)} cases")
    if not cases:
        logger.error("No cases to evaluate.")
        sys.exit(1)
        
    if dry_run:
        print(f"\n[DRY RUN] Dataset valid: {len(cases)} cases")
        print(f"[DRY RUN] Model: {model_name} | Threshold: {similarity_threshold}")
        for case in cases:
            print(f"  {case['id']} — {case['question'][:60]}")
        print("\n[DRY RUN] Selesai. Tidak ada inference yang dijalankan.")
        return {"dry_run": True, "cases": len(cases)}

    client = OllamaClient(model=model_name, temperature=0.0)
    faithfulness_evaluator = FaithfulnessEvaluator(
        similarity_threshold=similarity_threshold
    )

    # health checks
    logger.info("Running health checks...")
    if not client.health_check():
        logger.error(f"Model '{model_name}' tidak tersedia. Jalankan: ollama pull {model_name}")
        sys.exit(1)
    if not faithfulness_evaluator.health_check():
        logger.error("nomic-embed-text tidak tersedia. Jalankan: ollama pull nomic-embed-text")
        sys.exit(1)
    logger.info("All health checks passed.\n")
    
    case_ids = [c["id"] for c in cases]
    if not resume:
        _clear_checkpoints(model_name, dataset_name, case_ids)
        
    already_done = [cid for cid in case_ids if _load_checkpoint(model_name, dataset_name, cid)]
    if already_done and resume:
        logger.info(f"Resume mode: {len(already_done)}/{len(cases)} cases already done, skipping.")

    logger.info(
        f"Starting inference | {len(cases)} cases | model={model_name}\n"
        f"Estimasi waktu: ~4-5 menit per case di CPU. "
        f"Biarkan jalan sampai selesai atau Ctrl+C — progress tersimpan.\n"
    )

    query_results        = []
    faithfulness_results = []
    total     = len(cases)
    run_start = time.time()

    for i, case in enumerate(cases, 1):
        case_id   = case["id"]
        case_type = _detect_case_type(case)
        
        cached = _load_checkpoint(model_name, dataset_name, case_id)
        if cached and resume:
            logger.info(f"[{i}/{total}] {case_id} — SKIPPED (checkpoint exists)")
            query_results.append(_dict_to_query_output(cached.get("query_result", {})))
            faithfulness_results.append(_CheckpointFaithfulness(cached.get("faithfulness_result", {})))
            continue

        logger.info(f"[{i}/{total}] Running | case_id={case_id}")
        case_start = time.time()
        query_result = _query_with_retry(client, case, case_type)
        query_results.append(query_result)
        faith_result_dict = None

        if query_result.success and query_result.answer:
            faith_result = faithfulness_evaluator.evaluate(
                case=case,
                model_answer=query_result.answer,
                model_name=model_name,
            )
            faithfulness_results.append(faith_result)
            faith_result_dict = _faithfulness_to_dict(faith_result)
        else:
            logger.warning(
                f"Skipping faithfulness eval | case_id={case_id} | "
                f"reason={'query failed' if not query_result.success else 'empty answer'}"
            )
            
        checkpoint_data = {
            "case_id": case_id,
            "model": model_name,
            "dataset": dataset_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query_result": {
                "case_id": query_result.case_id,
                "success": query_result.success,
                "answer": query_result.answer,
                "latency_seconds": query_result.latency_seconds,
                "error_message": getattr(query_result, "error_message", None),
            },
            "faithfulness_result": faith_result_dict,
        }
        _save_checkpoint(model_name, dataset_name, case_id, checkpoint_data)
        
        elapsed   = round(time.time() - case_start, 1)
        remaining = total - i
        eta       = remaining * elapsed
        logger.info(
            f"Case done | elapsed={elapsed}s | "
            f"ETA={eta//60:.0f}m {eta%60:.0f}s remaining\n"
        )

    total_elapsed = round(time.time() - run_start, 1)

    # save raw query results
    query_output_path = str(output_prefix) + "_queries.json"
    save_query_results(query_results, query_output_path)

    # save faithfulness results
    real_faith_results = [r for r in faithfulness_results if not isinstance(r, _CheckpointFaithfulness)]
    faith_output_path = str(output_prefix) + "_faithfulness.json"
    if real_faith_results:
        faithfulness_evaluator.save_results(real_faith_results, faith_output_path)

    # build & save summary
    summary = _build_summary(
        run_id=run_id,
        model_name=model_name,
        dataset_name=dataset_name,
        cases=cases,
        query_results=query_results,
        faithfulness_results=faithfulness_results,
        total_elapsed=total_elapsed,
        similarity_threshold=similarity_threshold,
    )
    summary_path = str(output_prefix) + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _print_summary(summary, faithfulness_results)
    logger.info(f"Run complete | outputs saved to {REPORTS_DIR}")
    return summary

# checkpoint proxy — lightweight wrapper untuk cached results
class _CheckpointFaithfulness:
    def __init__(self, data: dict):
        self._data = data or {}
        
    @property
    def faithfulness_score(self) -> float:
        return self._data.get("faithfulness_score", 0.0)

    @property
    def has_failure(self) -> bool:
        return self._data.get("has_failure", False)

    @property
    def is_insufficient_context_response(self) -> bool:
        return self._data.get("is_insufficient_context_response", False)

    @property
    def case_id(self) -> str:
        return self._data.get("case_id", "unknown")

    @property
    def failure_cases(self) -> list:
        return self._data.get("failure_cases", [])

    @property
    def question(self) -> str:
        return self._data.get("question", "")


def _dict_to_query_output(d: dict) -> QueryOutput:
    return QueryOutput(
        case_id=d.get("case_id", "unknown"),
        model="",
        question="",
        context_length_chars=0,
        answer=d.get("answer", ""),
        latency_seconds=d.get("latency_seconds", 0.0),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        success=d.get("success", False),
        error_message=d.get("error_message"),
    )


def _faithfulness_to_dict(result) -> dict:
    return {
        "case_id": result.case_id,
        "question": result.question,
        "faithfulness_score": result.faithfulness_score,
        "has_failure": result.has_failure,
        "is_insufficient_context_response": result.is_insufficient_context_response,
        "evaluation_path": result.evaluation_path,
        "failure_cases": result.failure_cases,
        "supported_count": result.supported_count,
        "unsupported_count": result.unsupported_count,
        "total_claims": result.total_claims,
    }

# Query helpers
def _query_with_template(client: OllamaClient, case: dict, case_type: str) -> QueryOutput:
    import requests as _requests

    case_id = case["id"]

    if case_type == "distractor":
        distractor_type = case.get("distractor_type", "unknown")
        prompt = PromptTemplate.distractor_qa(
            context=case["context"],
            question=case["question"],
            distractor_type=distractor_type,
        )
    elif case_type == "paraphrase":
        prompt = PromptTemplate.consistency_qa(
            context=case["context"],
            question=case["question"],
            variant_label="original",
        )
    else:
        prompt = PromptTemplate.strict_qa(
            context=case["context"],
            question=case["question"],
        )

    payload = {
        "model": client.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": client.temperature,
            "num_predict": client.max_tokens,
        },
    }

    start = time.time()
    try:
        response = _requests.post(
            client._endpoint,
            json=payload,
            timeout=client.timeout_seconds,
        )
        response.raise_for_status()
        latency = round(time.time() - start, 3)
        raw     = response.json()
        answer  = raw.get("message", {}).get("content", "").strip()

        logger.info(
            f"Query OK | case_id={case_id} | latency={latency}s | "
            f"answer_length={len(answer)} chars"
        )
        return QueryOutput(
            case_id=case_id,
            model=client.model,
            question=case["question"],
            context_length_chars=len(case["context"]),
            answer=answer,
            latency_seconds=latency,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            success=bool(answer),
        )

    except Exception as e:
        latency = round(time.time() - start, 3)
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"Query FAILED | case_id={case_id} | {error_msg}")
        return QueryOutput(
            case_id=case_id,
            model=client.model,
            question=case["question"],
            context_length_chars=len(case["context"]),
            answer="",
            latency_seconds=latency,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            success=False,
            error_message=error_msg,
        )

def _detect_case_type(case: dict) -> str:
    case_id = case.get("id", "")
    if "distractor" in case_id:
        return "distractor"
    elif "paraphrase" in case_id:
        return "paraphrase"
    return "clean"

# summary helpers
def _build_summary(
    run_id, model_name, dataset_name, cases,
    query_results, faithfulness_results,
    total_elapsed, similarity_threshold,
) -> dict:
    success_queries = [r for r in query_results if getattr(r, "success", False)]
    insufficient = [
        r for r in faithfulness_results
        if r.is_insufficient_context_response
    ]
    failures    = [r for r in faithfulness_results if r.has_failure]
    faith_scores = [r.faithfulness_score for r in faithfulness_results]

    return {
        "run_id": run_id,
        "model": model_name,
        "dataset": dataset_name,
        "template_version": TEMPLATE_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "similarity_threshold": similarity_threshold,
            "embedding_model": config["models"]["embedding"],
        },
        "inference": {
            "total_cases": len(cases),
            "successful_queries": len(success_queries),
            "failed_queries": len(query_results) - len(success_queries),
            "total_elapsed_seconds": total_elapsed,
            "avg_latency_seconds": round(
                sum(getattr(r, "latency_seconds", 0) for r in query_results) / len(query_results), 1
            ) if query_results else 0,
        },
        "faithfulness": {
            "evaluated": len(faithfulness_results),
            "avg_score": round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else 0,
            "perfect_score_count": sum(1 for s in faith_scores if s == 1.0),
            "cases_with_failure": len(failures),
            "insufficient_context_responses": len(insufficient),
            "failure_rate": round(
                len(failures) / len(faithfulness_results), 4
            ) if faithfulness_results else 0,
        },
        "failure_case_ids": [r.case_id for r in failures],
    }

def _print_summary(summary: dict, faithfulness_results: list) -> None:
    f   = summary["faithfulness"]
    inf = summary["inference"]
    print("\n" + "=" * 60)
    print("EVALUATION RUN COMPLETE")
    print("=" * 60)
    print(f"Run ID      : {summary['run_id']}")
    print(f"Model       : {summary['model']}")
    print(f"Dataset     : {summary['dataset']}")
    print(f"Total time  : {inf['total_elapsed_seconds']:.0f}s "
          f"({inf['total_elapsed_seconds']/60:.1f} min)")
    print()
    print("INFERENCE")
    print(f"  Total cases      : {inf['total_cases']}")
    print(f"  Successful       : {inf['successful_queries']}")
    print(f"  Failed           : {inf['failed_queries']}")
    print(f"  Avg latency      : {inf['avg_latency_seconds']}s")
    print()
    print("FAITHFULNESS")
    print(f"  Evaluated        : {f['evaluated']}")
    print(f"  Avg score        : {f['avg_score']} ({f['avg_score']*100:.1f}%)")
    print(f"  Perfect (1.0)    : {f['perfect_score_count']}")
    print(f"  Cases w/ failure : {f['cases_with_failure']}")
    print(f"  Failure rate     : {f['failure_rate']*100:.1f}%")
    print(f"  INSUFFICIENT_CTX : {f['insufficient_context_responses']}")

    if summary["failure_case_ids"]:
        print()
        print("FAILURE CASES:")
        for case_id in summary["failure_case_ids"]:
            result = next(
                (r for r in faithfulness_results if r.case_id == case_id), None
            )
            if result:
                print(f"\n  {case_id} (score={result.faithfulness_score})")
                for fc in result.failure_cases:
                    print(f"     Claim    : {fc.get('unsupported_claim', '')[:80]}")
                    print(f"     Score    : {fc.get('similarity_score', 'N/A')}")
                    if fc.get("technical_conflict"):
                        print(f"     Technical conflict detected")
                    print(f"     Diagnosis: {fc.get('diagnosis', '')}")
    print("\n" + "=" * 60)

# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Framework — Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run — validasi dataset tanpa inference
  python pipeline/runner.py --model phi3:mini --dataset clean --dry-run

  # Development: 3 cases pertama
  python pipeline/runner.py --model phi3:mini --dataset clean --limit 3

  # Full eval dengan resume (default)
  python pipeline/runner.py --model mistral --dataset clean

  # Force re-run dari awal (ignore checkpoint)
  python pipeline/runner.py --model mistral --dataset clean --no-resume

  # Background process (progress tersimpan, bisa resume jika crash)
  nohup python pipeline/runner.py --model mistral --dataset clean > logs/run.log 2>&1 &
        """
    )
    parser.add_argument("--model", type=str, default="phi3:mini",
        help="Nama model Ollama (default: phi3:mini)")
    parser.add_argument("--dataset", type=str, default="clean",
        choices=["clean", "distractor", "paraphrase", "all"],
        help="Dataset yang dievaluasi (default: clean)")
    parser.add_argument("--limit", type=int, default=None,
        help="Batasi jumlah cases (default: semua)")
    parser.add_argument("--threshold", type=float, default=None,
        help=f"Similarity threshold (default: dari config.yaml = "
             f"{config['thresholds']['faithfulness_evidence']})")
    parser.add_argument("--no-resume", action="store_true",
        help="Ignore checkpoint, re-run semua cases dari awal")
    parser.add_argument("--dry-run", action="store_true",
        help="Validasi dataset dan config tanpa inference")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 60)
    print("LLM EVALUATION FRAMEWORK")
    print("=" * 60)
    print(f"Model     : {args.model}")
    print(f"Dataset   : {args.dataset}")
    print(f"Limit     : {args.limit or 'all cases'}")
    threshold_display = args.threshold or config["thresholds"]["faithfulness_evidence"]
    print(f"Threshold : {threshold_display}")
    print(f"Resume    : {not args.no_resume}")
    print(f"Dry run   : {args.dry_run}")
    print("=" * 60 + "\n")

    run_evaluation(
        model_name=args.model,
        dataset_name=args.dataset,
        limit=args.limit,
        similarity_threshold=args.threshold,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )