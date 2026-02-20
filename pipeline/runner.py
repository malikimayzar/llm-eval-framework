import json
import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.model_client import OllamaClient, save_query_results
from pipeline.prompt_templates import PromptTemplate, TEMPLATE_VERSION
from evaluators.faithfulness import FaithfulnessEvaluator

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("runner")

# Konstanta path
DATA_DIR    = PROJECT_ROOT / "data" / "dataset"
REPORTS_DIR = PROJECT_ROOT / "reports" / "results"

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

# run pipeline untuk satu dataset
def run_evaluation(
    model_name: str,
    dataset_name: str,
    limit: int = None,
    similarity_threshold: float = 0.75,
) -> dict:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info(
        f"Starting evaluation run | run_id={run_id} | "
        f"model={model_name} | dataset={dataset_name} | limit={limit}"
    )
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    output_prefix = REPORTS_DIR / f"{model_name.replace(':', '-')}_{dataset_name}_{run_id}"
    logger.info("Loading dataset...")
    cases = load_dataset(dataset_name, limit=limit)
    logger.info(f"Loaded {len(cases)} cases")

    if not cases:
        logger.error("No cases to evaluate. Check dataset files.")
        sys.exit(1)
        
    client = OllamaClient(model=model_name, temperature=0.0)
    faithfulness_evaluator = FaithfulnessEvaluator(
        similarity_threshold=similarity_threshold
    )
    
    logger.info("Running health checks...")
    if not client.health_check():
        logger.error(f"Model '{model_name}' tidak tersedia. Jalankan: ollama pull {model_name}")
        sys.exit(1)

    if not faithfulness_evaluator.health_check():
        logger.error("nomic-embed-text tidak tersedia. Jalankan: ollama pull nomic-embed-text")
        sys.exit(1)

    logger.info("All health checks passed.\n")
    logger.info(f"Starting inference | {len(cases)} cases | model={model_name}")
    logger.info("Estimasi waktu: ~4-5 menit per case di CPU. Biarkan jalan sampai selesai.\n")

    query_results = []
    faithfulness_results = []

    total = len(cases)
    run_start = time.time()

    for i, case in enumerate(cases, 1):
        case_start = time.time()
        logger.info(f"Progress: {i}/{total} | case_id={case['id']}")
        case_type = _detect_case_type(case)
        prompt = _build_prompt(case, case_type)

        query_result = client.query(
            context=case["context"],
            question=prompt,  
            case_id=case["id"],
        )
        query_result = _query_with_template(client, case, case_type)
        query_results.append(query_result)

        # Faithfulness evaluation (jika query berhasil)
        if query_result.success and query_result.answer:
            faith_result = faithfulness_evaluator.evaluate(
                case=case,
                model_answer=query_result.answer,
                model_name=model_name,
            )
            faithfulness_results.append(faith_result)
        else:
            logger.warning(
                f"Skipping faithfulness eval | case_id={case['id']} | "
                f"reason={'query failed' if not query_result.success else 'empty answer'}"
            )

        case_elapsed = round(time.time() - case_start, 1)
        remaining = total - i
        eta_seconds = remaining * case_elapsed
        logger.info(
            f"Case done | elapsed={case_elapsed}s | "
            f"ETA={eta_seconds//60:.0f}m {eta_seconds%60:.0f}s remaining\n"
        )

    total_elapsed = round(time.time() - run_start, 1)

    # save raw query results
    query_output_path = str(output_prefix) + "_queries.json"
    save_query_results(query_results, query_output_path)

    # save faithfulness results 
    faith_output_path = str(output_prefix) + "_faithfulness.json"
    if faithfulness_results:
        faithfulness_evaluator.save_results(faithfulness_results, faith_output_path)

    # build summary 
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

    # save summary 
    summary_path = str(output_prefix) + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _print_summary(summary, faithfulness_results)

    logger.info(f"Run complete | outputs saved to {REPORTS_DIR}")
    return summary


# helper: query dengan template yang benar
def _query_with_template(client: OllamaClient, case: dict, case_type: str):
    import requests
    import time as _time
    from pipeline.model_client import QueryOutput
    from datetime import datetime, timezone

    case_id = case["id"]

    # build prompt dari template
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

    start = _time.time()
    try:
        response = requests.post(
            client._endpoint,
            json=payload,
            timeout=client.timeout_seconds,
        )
        response.raise_for_status()
        latency = round(_time.time() - start, 3)
        raw = response.json()
        answer = raw.get("message", {}).get("content", "").strip()

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
            success=True,
        )

    except Exception as e:
        latency = round(_time.time() - start, 3)
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


def _build_prompt(case: dict, case_type: str) -> str:
    return case["question"]

# Helper: build summary dict
def _build_summary(
    run_id, model_name, dataset_name, cases,
    query_results, faithfulness_results,
    total_elapsed, similarity_threshold,
) -> dict:
    success_queries = [r for r in query_results if r.success]
    insufficient = [
        r for r in faithfulness_results
        if r.is_insufficient_context_response
    ]
    failures = [r for r in faithfulness_results if r.has_failure]
    faith_scores = [r.faithfulness_score for r in faithfulness_results]

    return {
        "run_id": run_id,
        "model": model_name,
        "dataset": dataset_name,
        "template_version": TEMPLATE_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "similarity_threshold": similarity_threshold,
            "embedding_model": "nomic-embed-text",
        },
        "inference": {
            "total_cases": len(cases),
            "successful_queries": len(success_queries),
            "failed_queries": len(query_results) - len(success_queries),
            "total_elapsed_seconds": total_elapsed,
            "avg_latency_seconds": round(
                sum(r.latency_seconds for r in query_results) / len(query_results), 1
            ) if query_results else 0,
        },
        "faithfulness": {
            "evaluated": len(faithfulness_results),
            "avg_score": round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else 0,
            "perfect_score_count": sum(1 for s in faith_scores if s == 1.0),
            "cases_with_failure": len(failures),
            "insufficient_context_responses": len(insufficient),
            "failure_rate": round(len(failures) / len(faithfulness_results), 4) if faithfulness_results else 0,
        },
        "failure_case_ids": [r.case_id for r in failures],
    }


# Helper: print summary ke terminal
def _print_summary(summary: dict, faithfulness_results: list) -> None:
    f = summary["faithfulness"]
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
                    print(f"     Claim    : {fc['unsupported_claim'][:80]}")
                    print(f"     Score    : {fc['similarity_score']}")
                    print(f"     Diagnosis: {fc['diagnosis']}")

    print("\n" + "=" * 60)
    
# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Framework — Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Development: 3 cases pertama, model ringan
  python pipeline/runner.py --model phi3:mini --dataset clean --limit 3

  # Eval final: semua clean cases, mistral
  python pipeline/runner.py --model mistral --dataset clean

  # Eval distractor cases
  python pipeline/runner.py --model mistral --dataset distractor

  # Eval semua dataset sekaligus
  python pipeline/runner.py --model mistral --dataset all
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="phi3:mini",
        help="Nama model Ollama (default: phi3:mini)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="clean",
        choices=["clean", "distractor", "paraphrase", "all"],
        help="Dataset yang akan dievaluasi (default: clean)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batasi jumlah cases (default: semua)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Similarity threshold untuk faithfulness (default: 0.75)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 60)
    print("LLM EVALUATION FRAMEWORK")
    print("=" * 60)
    print(f"Model     : {args.model}")
    print(f"Dataset   : {args.dataset}")
    print(f"Limit     : {args.limit or 'all cases'}")
    print(f"Threshold : {args.threshold}")
    print("=" * 60 + "\n")

    run_evaluation(
        model_name=args.model,
        dataset_name=args.dataset,
        limit=args.limit,
        similarity_threshold=args.threshold,
    )