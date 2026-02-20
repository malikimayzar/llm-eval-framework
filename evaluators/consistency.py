import json
import logging
import itertools
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from rouge_score import rouge_scorer

from evaluators.faithfulness import OllamaEmbedder, OLLAMA_BASE_URL, EMBEDDING_MODEL

logger = logging.getLogger("consistency")

DEFAULT_SEMANTIC_THRESHOLD = 0.72
DEFAULT_ROUGE_THRESHOLD = 0.40


@dataclass
class AnswerVariant:
    variant_id: str
    variant_label: str
    question: str
    answer: str
    model: str


@dataclass
class VariantPairComparison:
    pair_id: str
    variant_a: str
    variant_b: str
    answer_a: str
    answer_b: str
    semantic_similarity: float
    rouge_l_score: float
    is_consistent: bool
    inconsistency_type: Optional[str]


@dataclass
class ConsistencyResult:
    case_id: str
    model: str
    topic: str
    total_variants: int
    variants: list
    total_pairs: int
    comparisons: list
    avg_semantic_similarity: float
    avg_rouge_l: float
    consistency_score: float
    is_fully_consistent: bool
    inconsistent_pairs: list
    has_failure: bool
    semantic_threshold: float
    rouge_threshold: float
    embedding_model: str
    evaluator_version: str
    timestamp_utc: str


class ConsistencyEvaluator:
    VERSION = "1.0.0"

    def __init__(
        self,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
        rouge_threshold: float = DEFAULT_ROUGE_THRESHOLD,
        ollama_base_url: str = OLLAMA_BASE_URL,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.semantic_threshold = semantic_threshold
        self.rouge_threshold = rouge_threshold
        self.embedding_model = embedding_model
        self.embedder = OllamaEmbedder(model=embedding_model, base_url=ollama_base_url)
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        logger.info(
            f"ConsistencyEvaluator ready | "
            f"semantic_threshold={semantic_threshold} | "
            f"rouge_threshold={rouge_threshold} | "
            f"version={self.VERSION}"
        )

    def health_check(self) -> bool:
        return self.embedder.health_check()

    def evaluate_from_answers(
        self,
        case_id: str,
        topic: str,
        variants_with_answers: list,
        model_name: str = "unknown",
    ) -> ConsistencyResult:
        logger.info(f"Evaluating consistency | case_id={case_id} | variants={len(variants_with_answers)}")

        answer_variants = [
            AnswerVariant(
                variant_id=f"{case_id}_{v['label']}",
                variant_label=v["label"],
                question=v["question"],
                answer=v["answer"],
                model=model_name,
            )
            for v in variants_with_answers
        ]

        valid = [v for v in answer_variants if v.answer.strip()]
        if len(valid) < 2:
            logger.warning(f"Not enough valid variants | case_id={case_id}")
            return self._build_insufficient_result(case_id, topic, model_name, answer_variants)

        logger.info(f"Embedding {len(valid)} answers...")
        embeddings = self.embedder.embed_batch([v.answer for v in valid])

        comparisons = []
        for idx_a, idx_b in itertools.combinations(range(len(valid)), 2):
            va, vb = valid[idx_a], valid[idx_b]

            emb_a = embeddings[idx_a].reshape(1, -1)
            emb_b = embeddings[idx_b].reshape(1, -1)
            sem_sim = float(self._cosine_similarity(emb_a, emb_b)[0][0])

            rouge_result = self.rouge.score(va.answer, vb.answer)
            rouge_l = round(rouge_result["rougeL"].fmeasure, 4)

            sem_ok = sem_sim >= self.semantic_threshold
            rouge_ok = rouge_l >= self.rouge_threshold
            is_consistent = sem_ok and rouge_ok

            inconsistency_type = None
            if not is_consistent:
                if not sem_ok and not rouge_ok:
                    inconsistency_type = "both"
                elif not sem_ok:
                    inconsistency_type = "semantic"
                else:
                    inconsistency_type = "lexical"

            comparisons.append(VariantPairComparison(
                pair_id=f"{va.variant_label}_vs_{vb.variant_label}",
                variant_a=va.variant_label,
                variant_b=vb.variant_label,
                answer_a=va.answer,
                answer_b=vb.answer,
                semantic_similarity=round(sem_sim, 4),
                rouge_l_score=rouge_l,
                is_consistent=is_consistent,
                inconsistency_type=inconsistency_type,
            ))

        sem_scores = [c.semantic_similarity for c in comparisons]
        rouge_scores_list = [c.rouge_l_score for c in comparisons]
        inconsistent = [c for c in comparisons if not c.is_consistent]

        avg_sem = round(sum(sem_scores) / len(sem_scores), 4)
        avg_rouge = round(sum(rouge_scores_list) / len(rouge_scores_list), 4)
        consistency_score = round(0.7 * avg_sem + 0.3 * avg_rouge, 4)

        inconsistent_pairs_doc = [
            {
                "pair_id": c.pair_id,
                "semantic_similarity": c.semantic_similarity,
                "rouge_l_score": c.rouge_l_score,
                "inconsistency_type": c.inconsistency_type,
                "answer_a_preview": c.answer_a[:150],
                "answer_b_preview": c.answer_b[:150],
                "diagnosis": self._diagnose(c),
            }
            for c in inconsistent
        ]

        result = ConsistencyResult(
            case_id=case_id, model=model_name, topic=topic,
            total_variants=len(valid), variants=answer_variants,
            total_pairs=len(comparisons), comparisons=comparisons,
            avg_semantic_similarity=avg_sem, avg_rouge_l=avg_rouge,
            consistency_score=consistency_score,
            is_fully_consistent=len(inconsistent) == 0,
            inconsistent_pairs=inconsistent_pairs_doc,
            has_failure=len(inconsistent) > 0,
            semantic_threshold=self.semantic_threshold,
            rouge_threshold=self.rouge_threshold,
            embedding_model=self.embedding_model,
            evaluator_version=self.VERSION,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

        logger.log(
            logging.WARNING if result.has_failure else logging.INFO,
            f"Consistency result | case_id={case_id} | score={consistency_score} | "
            f"inconsistent={len(inconsistent)}/{len(comparisons)}"
        )
        return result

    def run_and_evaluate(
        self,
        cases: list,
        client,
        model_name: str,
    ) -> list:
        import requests as req
        import time
        from pipeline.prompt_templates import PromptTemplate

        results = []
        total = len(cases)

        for i, case in enumerate(cases, 1):
            case_id = case["id"]
            context = case["context"]
            variants = case["variants"]

            logger.info(f"Consistency progress: {i}/{total} | case_id={case_id}")

            variants_with_answers = []
            for variant in variants:
                label = variant["label"]
                question = variant["question"]
                prompt = PromptTemplate.consistency_qa(
                    context=context,
                    question=question,
                    variant_label=label,
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
                logger.info(f"  Querying variant '{label}'...")
                start = time.time()
                try:
                    response = req.post(
                        client._endpoint, json=payload,
                        timeout=client.timeout_seconds,
                    )
                    response.raise_for_status()
                    answer = response.json().get("message", {}).get("content", "").strip()
                    logger.info(f"  Variant '{label}' OK | latency={round(time.time()-start,1)}s")
                except Exception as e:
                    answer = ""
                    logger.error(f"  Variant '{label}' FAILED | {e}")

                variants_with_answers.append({
                    "label": label, "question": question, "answer": answer,
                })

            result = self.evaluate_from_answers(
                case_id=case_id,
                topic=case.get("topic", ""),
                variants_with_answers=variants_with_answers,
                model_name=model_name,
            )
            results.append(result)

        return results

    def save_results(self, results: list, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
        logger.info(f"Consistency results saved | path={output_path}")

    def print_summary(self, results: list) -> None:
        total = len(results)
        if not total:
            print("Tidak ada hasil.")
            return

        scores = [r.consistency_score for r in results]
        avg = sum(scores) / total
        failed = [r for r in results if r.has_failure]
        fully_ok = [r for r in results if r.is_fully_consistent]

        print("\n" + "=" * 60)
        print("CONSISTENCY EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total cases          : {total}")
        print(f"Avg consistency score: {avg:.4f} ({avg*100:.1f}%)")
        print(f"Fully consistent     : {len(fully_ok)}/{total}")
        print(f"Cases with failure   : {len(failed)}")
        print(f"Semantic threshold   : {results[0].semantic_threshold}")
        print(f"ROUGE-L threshold    : {results[0].rouge_threshold}")

        if failed:
            print("\nINCONSISTENCY CASES:")
            print("-" * 60)
            for r in failed:
                print(f"\n  case_id : {r.case_id}")
                print(f"  topic   : {r.topic}")
                print(f"  score   : {r.consistency_score}")
                for ip in r.inconsistent_pairs:
                    print(f"\n   {ip['pair_id']}")
                    print(f"     Type     : {ip['inconsistency_type']}")
                    print(f"     Semantic : {ip['semantic_similarity']}")
                    print(f"     ROUGE-L  : {ip['rouge_l_score']}")
                    print(f"     Diagnosis: {ip['diagnosis'][:100]}")
                    print(f"     Answer A : {ip['answer_a_preview'][:80]}")
                    print(f"     Answer B : {ip['answer_b_preview'][:80]}")

        print("\n" + "=" * 60)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(a_norm, b_norm.T)

    def _diagnose(self, c: VariantPairComparison) -> str:
        t = c.inconsistency_type
        if t == "both":
            return (
                f"Severe inconsistency: low semantic ({c.semantic_similarity}) "
                f"AND low lexical overlap ({c.rouge_l_score}). "
                f"Model produced fundamentally different answers."
            )
        elif t == "semantic":
            return (
                f"Semantic inconsistency (sem={c.semantic_similarity} < threshold). "
                f"Model changed meaning despite some word overlap."
            )
        elif t == "lexical":
            return (
                f"Lexical inconsistency (rouge={c.rouge_l_score} < threshold) "
                f"but semantically close ({c.semantic_similarity}). "
                f"Likely acceptable paraphrase — review manually."
            )
        return "Unknown inconsistency type."

    def _build_insufficient_result(self, case_id, topic, model_name, variants) -> ConsistencyResult:
        return ConsistencyResult(
            case_id=case_id, model=model_name, topic=topic,
            total_variants=len(variants), variants=variants,
            total_pairs=0, comparisons=[],
            avg_semantic_similarity=0.0, avg_rouge_l=0.0,
            consistency_score=0.0, is_fully_consistent=False,
            inconsistent_pairs=[{"pair_id": "N/A", "diagnosis": "Not enough valid variants (< 2)."}],
            has_failure=True,
            semantic_threshold=self.semantic_threshold,
            rouge_threshold=self.rouge_threshold,
            embedding_model=self.embedding_model,
            evaluator_version=self.VERSION,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

# smoke test
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    print("=" * 60)
    print("ConsistencyEvaluator — Smoke Test")
    print("Hanya butuh nomic-embed-text, tidak butuh LLM")
    print("=" * 60)

    evaluator = ConsistencyEvaluator(semantic_threshold=0.72, rouge_threshold=0.40)

    print("\n[1] Health check...")
    if not evaluator.health_check():
        print("nomic-embed-text tidak tersedia.")
        sys.exit(1)
    print("Siap.\n")

    print("[2] Consistent answers...")
    r1 = evaluator.evaluate_from_answers(
        case_id="smoke_001",
        topic="Query parameter required",
        variants_with_answers=[
            {"label": "original", "question": "How to make query param required?",
             "answer": "By not declaring any default value for the parameter."},
            {"label": "paraphrase_1", "question": "How to force query param mandatory?",
             "answer": "You can make it required by not providing a default value."},
            {"label": "paraphrase_2", "question": "How to make query param non-optional?",
             "answer": "Simply omit the default value declaration for that parameter."},
        ],
        model_name="smoke_test",
    )
    print(f"   Score: {r1.consistency_score} | Failure: {r1.has_failure}")

    print("\n[3] Inconsistent answers...")
    r2 = evaluator.evaluate_from_answers(
        case_id="smoke_002",
        topic="HTTP Methods",
        variants_with_answers=[
            {"label": "original", "question": "What HTTP method updates data?",
             "answer": "PUT is used to update data."},
            {"label": "paraphrase_1", "question": "Which method modifies existing data?",
             "answer": "PATCH is the correct HTTP method for partial updates and data modification."},
            {"label": "paraphrase_2", "question": "HTTP method for data modification?",
             "answer": "PUT is the standard method for updating data according to the documentation."},
        ],
        model_name="smoke_test",
    )
    print(f"   Score: {r2.consistency_score} | Failure: {r2.has_failure}")
    for ip in r2.inconsistent_pairs:
        print(f"   {ip['pair_id']}: {ip['diagnosis'][:80]}")

    evaluator.print_summary([r1, r2])