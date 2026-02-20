import json
import logging
import requests as req
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np

from evaluators.faithfulness import OllamaEmbedder, OLLAMA_BASE_URL, EMBEDDING_MODEL

logger = logging.getLogger("retrieval_dependency")

DEFAULT_CHANGE_THRESHOLD = 0.70

# Kondisi konteks yang dievaluasi
CONDITION_FULL    = "full_context"
CONDITION_REMOVED = "no_context"
CONDITION_WRONG   = "wrong_context"


# data classes
@dataclass
class ConditionResult:
    condition: str          
    context_used: str       
    question: str
    answer: str
    success: bool
    latency_seconds: float


@dataclass
class DependencyComparison:
    pair_id: str           
    condition_a: str
    condition_b: str
    answer_a: str
    answer_b: str
    semantic_similarity: float
    answer_changed: bool   
    dependency_flag: bool   
    diagnosis: str


@dataclass
class RetrievalDependencyResult:
    case_id: str
    model: str
    question: str
    conditions: list        
    comparisons: list      
    full_vs_no_context_similarity: float
    full_vs_wrong_context_similarity: float
    high_dependency: bool  
    has_failure: bool       
    failure_flags: list
    change_threshold: float
    embedding_model: str
    evaluator_version: str
    timestamp_utc: str

class RetrievalDependencyEvaluator:
    VERSION = "1.0.0"
    
    def __init__(
        self,
        change_threshold: float = DEFAULT_CHANGE_THRESHOLD,
        ollama_base_url: str = OLLAMA_BASE_URL,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.change_threshold = change_threshold
        self.embedding_model = embedding_model
        self.embedder = OllamaEmbedder(
            model=embedding_model,
            base_url=ollama_base_url,
        )
        logger.info(
            f"RetrievalDependencyEvaluator ready | "
            f"change_threshold={change_threshold} | "
            f"version={self.VERSION}"
        )

    def health_check(self) -> bool:
        return self.embedder.health_check()

    def evaluate_from_answers(
        self,
        case_id: str,
        question: str,
        full_answer: str,
        no_context_answer: str,
        wrong_answer: str,
        model_name: str = "unknown",
        full_context: str = "",
        wrong_context: str = "",
    ) -> RetrievalDependencyResult:
        logger.info(f"Evaluating retrieval dependency | case_id={case_id}")

        conditions = [
            ConditionResult(
                condition=CONDITION_FULL,
                context_used=full_context,
                question=question,
                answer=full_answer,
                success=bool(full_answer.strip()),
                latency_seconds=0.0,
            ),
            ConditionResult(
                condition=CONDITION_REMOVED,
                context_used="",
                question=question,
                answer=no_context_answer,
                success=bool(no_context_answer.strip()),
                latency_seconds=0.0,
            ),
            ConditionResult(
                condition=CONDITION_WRONG,
                context_used=wrong_context,
                question=question,
                answer=wrong_answer,
                success=bool(wrong_answer.strip()),
                latency_seconds=0.0,
            ),
        ]

        valid_answers = [full_answer, no_context_answer, wrong_answer]
        logger.info("Embedding 3 condition answers...")
        embeddings = self.embedder.embed_batch(valid_answers)

        emb_full    = embeddings[0].reshape(1, -1)
        emb_no_ctx  = embeddings[1].reshape(1, -1)
        emb_wrong   = embeddings[2].reshape(1, -1)

        sim_full_vs_no  = float(self._cosine_similarity(emb_full, emb_no_ctx)[0][0])
        sim_full_vs_wrong = float(self._cosine_similarity(emb_full, emb_wrong)[0][0])

        sim_full_vs_no    = round(sim_full_vs_no, 4)
        sim_full_vs_wrong = round(sim_full_vs_wrong, 4)

        comp_no = DependencyComparison(
            pair_id="full_vs_no_context",
            condition_a=CONDITION_FULL,
            condition_b=CONDITION_REMOVED,
            answer_a=full_answer,
            answer_b=no_context_answer,
            semantic_similarity=sim_full_vs_no,
            answer_changed=sim_full_vs_no < self.change_threshold,
            dependency_flag=sim_full_vs_no >= self.change_threshold,
            diagnosis=self._diagnose(
                "full_vs_no_context", sim_full_vs_no, self.change_threshold
            ),
        )

        comp_wrong = DependencyComparison(
            pair_id="full_vs_wrong_context",
            condition_a=CONDITION_FULL,
            condition_b=CONDITION_WRONG,
            answer_a=full_answer,
            answer_b=wrong_answer,
            semantic_similarity=sim_full_vs_wrong,
            answer_changed=sim_full_vs_wrong < self.change_threshold,
            dependency_flag=sim_full_vs_wrong >= self.change_threshold,
            diagnosis=self._diagnose(
                "full_vs_wrong_context", sim_full_vs_wrong, self.change_threshold
            ),
        )

        comparisons = [comp_no, comp_wrong]
        high_dependency = comp_no.answer_changed and comp_wrong.answer_changed
        
        failure_flags = []
        if comp_no.dependency_flag:
            failure_flags.append({
                "flag": "CONTEXT_INDEPENDENT_NO_CONTEXT",
                "pair": "full_vs_no_context",
                "similarity": sim_full_vs_no,
                "diagnosis": comp_no.diagnosis,
                "implication": (
                    "Model memberikan jawaban yang sangat mirip meski tidak ada konteks. "
                    "Ini mengindikasikan model menjawab dari training memory, bukan dari dokumen."
                ),
            })

        if comp_wrong.dependency_flag:
            failure_flags.append({
                "flag": "CONTEXT_INDEPENDENT_WRONG_CONTEXT",
                "pair": "full_vs_wrong_context",
                "similarity": sim_full_vs_wrong,
                "diagnosis": comp_wrong.diagnosis,
                "implication": (
                    "Model memberikan jawaban yang sangat mirip meski konteks salah. "
                    "Model tidak mendeteksi atau tidak peduli bahwa konteksnya sudah dimanipulasi."
                ),
            })

        result = RetrievalDependencyResult(
            case_id=case_id,
            model=model_name,
            question=question,
            conditions=conditions,
            comparisons=comparisons,
            full_vs_no_context_similarity=sim_full_vs_no,
            full_vs_wrong_context_similarity=sim_full_vs_wrong,
            high_dependency=high_dependency,
            has_failure=len(failure_flags) > 0,
            failure_flags=failure_flags,
            change_threshold=self.change_threshold,
            embedding_model=self.embedding_model,
            evaluator_version=self.VERSION,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

        logger.log(
            logging.WARNING if result.has_failure else logging.INFO,
            f"Retrieval dependency | case_id={case_id} | "
            f"high_dependency={high_dependency} | "
            f"flags={len(failure_flags)}"
        )
        return result

    def run_and_evaluate(
        self,
        cases: list,
        client,
        model_name: str,
        distractor_map: dict = None,
    ) -> list:
        from pipeline.prompt_templates import PromptTemplate

        results = []
        total = len(cases)

        for i, case in enumerate(cases, 1):
            case_id = case["id"]
            context = case["context"]
            question = case["question"]

            logger.info(
                f"Retrieval dependency progress: {i}/{total} | case_id={case_id}"
            )

            if distractor_map and case_id in distractor_map:
                wrong_context = distractor_map[case_id]
            else:
                wrong_context = self._shuffle_context(context)
                
            conditions_config = [
                (CONDITION_FULL,    context,      "full context"),
                (CONDITION_REMOVED, "",           "no context"),
                (CONDITION_WRONG,   wrong_context,"wrong context"),
            ]

            condition_answers = {}
            condition_contexts = {
                CONDITION_FULL: context,
                CONDITION_REMOVED: "",
                CONDITION_WRONG: wrong_context,
            }

            for condition, ctx, label in conditions_config:
                if ctx:
                    prompt = PromptTemplate.strict_qa(
                        context=ctx,
                        question=question,
                    )
                else:
                    prompt = f"""You are a Question Answering system.

Answer the following question based only on your knowledge.
If you are not certain, say so explicitly.

QUESTION:
{question}

ANSWER:"""

                payload = {
                    "model": client.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": client.temperature,
                        "num_predict": client.max_tokens,
                    },
                }

                logger.info(f"  Querying [{label}] | case_id={case_id}")
                start = time.time()
                try:
                    response = req.post(
                        client._endpoint,
                        json=payload,
                        timeout=client.timeout_seconds,
                    )
                    response.raise_for_status()
                    answer = response.json().get("message", {}).get("content", "").strip()
                    latency = round(time.time() - start, 1)
                    logger.info(f"  [{label}] OK | latency={latency}s")
                except Exception as e:
                    answer = ""
                    logger.error(f"  [{label}] FAILED | {e}")

                condition_answers[condition] = answer

            # Evaluasi
            result = self.evaluate_from_answers(
                case_id=case_id,
                question=question,
                full_answer=condition_answers.get(CONDITION_FULL, ""),
                no_context_answer=condition_answers.get(CONDITION_REMOVED, ""),
                wrong_answer=condition_answers.get(CONDITION_WRONG, ""),
                model_name=model_name,
                full_context=context,
                wrong_context=wrong_context,
            )
            results.append(result)

        return results

    def save_results(self, results: list, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
        logger.info(f"Retrieval dependency results saved | path={output_path}")

    def print_summary(self, results: list) -> None:
        total = len(results)
        if not total:
            print("Tidak ada hasil.")
            return

        high_dep = [r for r in results if r.high_dependency]
        failed = [r for r in results if r.has_failure]

        sim_no_ctx = [r.full_vs_no_context_similarity for r in results]
        sim_wrong  = [r.full_vs_wrong_context_similarity for r in results]

        print("\n" + "=" * 60)
        print("RETRIEVAL DEPENDENCY SUMMARY")
        print("=" * 60)
        print(f"Total cases evaluated    : {total}")
        print(f"High dependency (good)   : {len(high_dep)}/{total}")
        print(f"Dependency failures      : {len(failed)}")
        print(f"Avg sim full vs no-ctx   : {sum(sim_no_ctx)/len(sim_no_ctx):.4f}")
        print(f"Avg sim full vs wrong-ctx: {sum(sim_wrong)/len(sim_wrong):.4f}")
        print(f"Change threshold         : {results[0].change_threshold}")

        if failed:
            print("\nFAILURE FLAGS:")
            print("-" * 60)
            for r in failed:
                print(f"\n  case_id : {r.case_id}")
                print(f"  question: {r.question[:70]}")
                for flag in r.failure_flags:
                    print(f"\n  🚩 {flag['flag']}")
                    print(f"     Similarity : {flag['similarity']}")
                    print(f"     Diagnosis  : {flag['diagnosis'][:100]}")
                    print(f"     Implication: {flag['implication'][:100]}")

        print("\n" + "=" * 60)

    # private helpers 
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(a_norm, b_norm.T)

    def _diagnose(self, pair: str, similarity: float, threshold: float) -> str:
        if similarity >= threshold:
            return (
                f"HIGH similarity ({similarity}) between {pair}. "
                f"Model answer barely changed despite context modification. "
                f"Strong indicator of training memory usage over context retrieval."
            )
        elif similarity >= threshold - 0.15:
            return (
                f"MODERATE similarity ({similarity}) between {pair}. "
                f"Some context dependency detected but not conclusive. "
                f"Manual review recommended."
            )
        else:
            return (
                f"LOW similarity ({similarity}) between {pair}. "
                f"Model answer changed significantly — good context dependency."
            )

    def _shuffle_context(self, context: str) -> str:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', context.strip())
        if len(sentences) <= 1:
            return context + " Note: This information may not be accurate."
        shuffled = list(reversed(sentences))
        return " ".join(shuffled)
    
# smoke test 
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    print("=" * 60)
    print("RetrievalDependencyEvaluator — Smoke Test")
    print("Hanya butuh nomic-embed-text, tidak butuh LLM")
    print("=" * 60)

    evaluator = RetrievalDependencyEvaluator(change_threshold=0.70)

    print("\n[1] Health check...")
    if not evaluator.health_check():
        print("nomic-embed-text tidak tersedia.")
        sys.exit(1)
    print("Siap.\n")

    print("[2] Context-dependent model (expected behavior)...")
    r1 = evaluator.evaluate_from_answers(
        case_id="dep_smoke_001",
        question="What HTTP method is used to update data?",
        full_answer="PUT is used to update data according to the documentation.",
        no_context_answer="I am not certain without context, but typically PUT or PATCH is used.",
        wrong_answer="POST is used to update data as stated in the provided context.",
        model_name="smoke_test",
        full_context="POST to create data, GET to read data, PUT to update data, DELETE to delete data.",
        wrong_context="POST to create data, GET to read data, POST to update data, DELETE to delete data.",
    )
    print(f"High dependency: {r1.high_dependency} | Failure: {r1.has_failure}")
    print(f"full_vs_no_ctx similarity  : {r1.full_vs_no_context_similarity}")
    print(f"full_vs_wrong_ctx similarity: {r1.full_vs_wrong_context_similarity}")

    print("\n[3] Context-independent model (red flag)...")
    r2 = evaluator.evaluate_from_answers(
        case_id="dep_smoke_002",
        question="What HTTP method is used to update data?",
        full_answer="PUT is used to update data.",
        no_context_answer="PUT is used to update data in REST APIs.",
        wrong_answer="PUT is the standard method for updating resources.",
        model_name="smoke_test",
    )
    print(f"High dependency: {r2.high_dependency} | Failure: {r2.has_failure}")
    print(f"full_vs_no_ctx similarity  : {r2.full_vs_no_context_similarity}")
    print(f"full_vs_wrong_ctx similarity: {r2.full_vs_wrong_context_similarity}")
    if r2.failure_flags:
        for flag in r2.failure_flags:
            print(f"{flag['flag']}: {flag['diagnosis'][:80]}")

    evaluator.print_summary([r1, r2])