import json
import logging
import re
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from rouge_score import rouge_scorer

from evaluators.faithfulness import OllamaEmbedder, OLLAMA_BASE_URL, EMBEDDING_MODEL

logger = logging.getLogger("sensitivity")

DEFAULT_ROBUSTNESS_THRESHOLD = 0.85

# Tipe perturbasi yang tersedia
PERTURBATION_TYPO      = "typo"
PERTURBATION_LOWERCASE = "lowercase"
PERTURBATION_NOISE     = "context_noise"
PERTURBATION_REORDER   = "question_reorder"

# Perturbation functions
def perturb_typo(question: str) -> str:
    keyboard_neighbors = {
        'a': 's', 'b': 'v', 'c': 'x', 'd': 's', 'e': 'r', 'f': 'd',
        'g': 'f', 'h': 'g', 'i': 'u', 'j': 'h', 'k': 'j', 'l': 'k',
        'm': 'n', 'n': 'm', 'o': 'i', 'p': 'o', 'q': 'w', 'r': 'e',
        's': 'a', 't': 'r', 'u': 'y', 'v': 'b', 'w': 'q', 'x': 'z',
        'y': 'u', 'z': 'x',
    }
    
    candidates = [
        i for i, c in enumerate(question)
        if c.lower() in keyboard_neighbors and c != ' '
    ]
    if not candidates:
        return question

    mid_candidates = [i for i in candidates if 3 <= i <= len(question) - 4]
    pos = random.choice(mid_candidates if mid_candidates else candidates)

    char = question[pos]
    replacement = keyboard_neighbors.get(char.lower(), char)
    if char.isupper():
        replacement = replacement.upper()
    return question[:pos] + replacement + question[pos+1:]


def perturb_lowercase(question: str) -> str:
    return question.lower()

def perturb_context_noise(context: str) -> str:
    noise_sentences = [
        "Note: This documentation was last updated recently.",
        "See also the official documentation for more details.",
        "Additional configuration options may vary by version.",
        "Please refer to the changelog for recent updates.",
    ]
    noise = random.choice(noise_sentences)
    return context.strip() + " " + noise


def perturb_question_reorder(question: str) -> str:
    rewrites = [
        ("What ", "Can you tell me what "),
        ("How ", "Could you explain how "),
        ("Why ", "What is the reason why "),
        ("Which ", "Can you tell me which "),
        ("When ", "At what point "),
    ]
    for original, replacement in rewrites:
        if question.startswith(original):
            return replacement + question[len(original):]
    return "Please answer: " + question

# Data classes
@dataclass
class PerturbedVariant:
    variant_id: str
    perturbation_type: str
    original_question: str
    perturbed_question: str
    original_context: str
    perturbed_context: str
    answer: str
    model: str
    success: bool


@dataclass
class SensitivityComparison:
    pair_id: str
    perturbation_type: str
    answer_original: str
    answer_perturbed: str
    semantic_similarity: float
    rouge_l_score: float
    is_robust: bool
    sensitivity_type: Optional[str] 
    diagnosis: str


@dataclass
class SensitivityResult:
    case_id: str
    model: str
    original_question: str
    original_answer: str

    total_perturbations: int
    variants: list              
    comparisons: list          

    avg_semantic_similarity: float
    avg_rouge_l: float
    robustness_score: float     

    is_fully_robust: bool
    has_failure: bool
    sensitive_perturbations: list   

    robustness_threshold: float
    embedding_model: str
    evaluator_version: str
    timestamp_utc: str

# Sensitivity Evaluator
class SensitivityEvaluator:
    VERSION = "1.0.0"

    PERTURBATIONS = [
        PERTURBATION_TYPO,
        PERTURBATION_LOWERCASE,
        PERTURBATION_NOISE,
        PERTURBATION_REORDER,
    ]

    def __init__(
        self,
        robustness_threshold: float = DEFAULT_ROBUSTNESS_THRESHOLD,
        ollama_base_url: str = OLLAMA_BASE_URL,
        embedding_model: str = EMBEDDING_MODEL,
        random_seed: int = 42,
    ):
        self.robustness_threshold = robustness_threshold
        self.embedding_model = embedding_model
        self.embedder = OllamaEmbedder(
            model=embedding_model,
            base_url=ollama_base_url,
        )
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        random.seed(random_seed)

        logger.info(
            f"SensitivityEvaluator ready | "
            f"robustness_threshold={robustness_threshold} | "
            f"perturbations={self.PERTURBATIONS} | "
            f"version={self.VERSION}"
        )

    def health_check(self) -> bool:
        return self.embedder.health_check()

    def apply_perturbation(
        self,
        question: str,
        context: str,
        perturbation_type: str,
    ) -> tuple:
        if perturbation_type == PERTURBATION_TYPO:
            return perturb_typo(question), context

        elif perturbation_type == PERTURBATION_LOWERCASE:
            return perturb_lowercase(question), context

        elif perturbation_type == PERTURBATION_NOISE:
            return question, perturb_context_noise(context)

        elif perturbation_type == PERTURBATION_REORDER:
            return perturb_question_reorder(question), context

        else:
            logger.warning(f"Unknown perturbation type: {perturbation_type}")
            return question, context

    def evaluate_from_answers(
        self,
        case_id: str,
        original_question: str,
        original_context: str,
        original_answer: str,
        perturbed_variants: list,
        model_name: str = "unknown",
    ) -> SensitivityResult:
        logger.info(
            f"Evaluating sensitivity | case_id={case_id} | "
            f"perturbations={len(perturbed_variants)}"
        )

        variants = []
        for v in perturbed_variants:
            variants.append(PerturbedVariant(
                variant_id=f"{case_id}_{v['perturbation_type']}",
                perturbation_type=v["perturbation_type"],
                original_question=original_question,
                perturbed_question=v["perturbed_question"],
                original_context=original_context,
                perturbed_context=v["perturbed_context"],
                answer=v["answer"],
                model=model_name,
                success=bool(v["answer"].strip()),
            ))

        valid_variants = [v for v in variants if v.success]
        if not valid_variants:
            return self._build_no_variants_result(
                case_id, model_name, original_question, original_answer, variants
            )

        all_answers = [original_answer] + [v.answer for v in valid_variants]
        logger.info(f"Embedding {len(all_answers)} answers...")
        embeddings = self.embedder.embed_batch(all_answers)

        emb_original = embeddings[0].reshape(1, -1)
        
        comparisons = []
        for i, variant in enumerate(valid_variants):
            emb_perturbed = embeddings[i + 1].reshape(1, -1)
            sem_sim = float(self._cosine_similarity(emb_original, emb_perturbed)[0][0])

            rouge_result = self.rouge.score(original_answer, variant.answer)
            rouge_l = round(rouge_result["rougeL"].fmeasure, 4)

            sem_ok   = sem_sim >= self.robustness_threshold
            rouge_ok = rouge_l >= 0.40  
            is_robust = sem_ok  

            sensitivity_type = None
            if not is_robust:
                if not sem_ok and not rouge_ok:
                    sensitivity_type = "both"
                elif not sem_ok:
                    sensitivity_type = "semantic"
                else:
                    sensitivity_type = "lexical"

            comparison = SensitivityComparison(
                pair_id=f"original_vs_{variant.perturbation_type}",
                perturbation_type=variant.perturbation_type,
                answer_original=original_answer,
                answer_perturbed=variant.answer,
                semantic_similarity=round(sem_sim, 4),
                rouge_l_score=rouge_l,
                is_robust=is_robust,
                sensitivity_type=sensitivity_type,
                diagnosis=self._diagnose(variant.perturbation_type, sem_sim, rouge_l, is_robust),
            )
            comparisons.append(comparison)

        # Scoring
        sem_scores   = [c.semantic_similarity for c in comparisons]
        rouge_scores = [c.rouge_l_score for c in comparisons]
        sensitive    = [c for c in comparisons if not c.is_robust]

        avg_sem   = round(sum(sem_scores) / len(sem_scores), 4)
        avg_rouge = round(sum(rouge_scores) / len(rouge_scores), 4)
        robustness_score = avg_sem

        sensitive_doc = [
            {
                "perturbation_type": c.perturbation_type,
                "semantic_similarity": c.semantic_similarity,
                "rouge_l_score": c.rouge_l_score,
                "sensitivity_type": c.sensitivity_type,
                "answer_original_preview": c.answer_original[:100],
                "answer_perturbed_preview": c.answer_perturbed[:100],
                "diagnosis": c.diagnosis,
            }
            for c in sensitive
        ]

        result = SensitivityResult(
            case_id=case_id,
            model=model_name,
            original_question=original_question,
            original_answer=original_answer,
            total_perturbations=len(valid_variants),
            variants=variants,
            comparisons=comparisons,
            avg_semantic_similarity=avg_sem,
            avg_rouge_l=avg_rouge,
            robustness_score=robustness_score,
            is_fully_robust=len(sensitive) == 0,
            has_failure=len(sensitive) > 0,
            sensitive_perturbations=sensitive_doc,
            robustness_threshold=self.robustness_threshold,
            embedding_model=self.embedding_model,
            evaluator_version=self.VERSION,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

        logger.log(
            logging.WARNING if result.has_failure else logging.INFO,
            f"Sensitivity result | case_id={case_id} | "
            f"robustness={robustness_score:.4f} | "
            f"sensitive={len(sensitive)}/{len(comparisons)}"
        )
        return result

    def run_and_evaluate(
        self,
        cases: list,
        client,
        model_name: str,
        perturbation_types: list = None,
    ) -> list:
        from pipeline.prompt_templates import PromptTemplate

        if perturbation_types is None:
            perturbation_types = self.PERTURBATIONS

        results = []
        total = len(cases)

        for i, case in enumerate(cases, 1):
            case_id  = case["id"]
            context  = case["context"]
            question = case["question"]

            logger.info(f"Sensitivity progress: {i}/{total} | case_id={case_id}")

            original_prompt = PromptTemplate.strict_qa(context, question)
            original_answer = self._query(client, original_prompt, f"{case_id}_original")

            if not original_answer:
                logger.warning(f"Original query failed | case_id={case_id}")
                continue

            perturbed_variants = []
            for ptype in perturbation_types:
                p_question, p_context = self.apply_perturbation(question, context, ptype)
                p_prompt = PromptTemplate.strict_qa(p_context, p_question)
                p_answer = self._query(client, p_prompt, f"{case_id}_{ptype}")

                perturbed_variants.append({
                    "perturbation_type": ptype,
                    "perturbed_question": p_question,
                    "perturbed_context": p_context,
                    "answer": p_answer,
                })

            result = self.evaluate_from_answers(
                case_id=case_id,
                original_question=question,
                original_context=context,
                original_answer=original_answer,
                perturbed_variants=perturbed_variants,
                model_name=model_name,
            )
            results.append(result)
        return results

    def save_results(self, results: list, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
        logger.info(f"Sensitivity results saved | path={output_path}")

    def print_summary(self, results: list) -> None:
        total = len(results)
        if not total:
            print("Tidak ada hasil.")
            return

        scores    = [r.robustness_score for r in results]
        avg       = sum(scores) / total
        failed    = [r for r in results if r.has_failure]
        fully_ok  = [r for r in results if r.is_fully_robust]
        ptype_counts = {}
        for r in results:
            for sp in r.sensitive_perturbations:
                pt = sp["perturbation_type"]
                ptype_counts[pt] = ptype_counts.get(pt, 0) + 1

        print("\n" + "=" * 60)
        print("SENSITIVITY EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total cases          : {total}")
        print(f"Avg robustness score : {avg:.4f} ({avg*100:.1f}%)")
        print(f"Fully robust         : {len(fully_ok)}/{total}")
        print(f"Cases with failure   : {len(failed)}")
        print(f"Robustness threshold : {results[0].robustness_threshold}")

        if ptype_counts:
            print("\nSensitive perturbation types:")
            for pt, count in sorted(ptype_counts.items(), key=lambda x: -x[1]):
                print(f"  {pt:<25}: {count} case(s)")

        if failed:
            print("\nSENSITIVE CASES:")
            print("-" * 60)
            for r in failed:
                print(f"\n  case_id : {r.case_id}")
                print(f"  score   : {r.robustness_score}")
                for sp in r.sensitive_perturbations:
                    print(f"\n  ⚠ {sp['perturbation_type']}")
                    print(f"    Semantic : {sp['semantic_similarity']}")
                    print(f"    Diagnosis: {sp['diagnosis'][:100]}")
                    print(f"    Original : {sp['answer_original_preview'][:70]}")
                    print(f"    Perturbed: {sp['answer_perturbed_preview'][:70]}")

        print("\n" + "=" * 60)

    def _query(self, client, prompt: str, label: str) -> str:
        import requests as req
        import time

        payload = {
            "model": client.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": client.temperature,
                "num_predict": client.max_tokens,
            },
        }
        logger.info(f"  Querying [{label}]...")
        start = time.time()
        try:
            response = req.post(
                client._endpoint, json=payload,
                timeout=client.timeout_seconds,
            )
            response.raise_for_status()
            answer = response.json().get("message", {}).get("content", "").strip()
            logger.info(f"  [{label}] OK | latency={round(time.time()-start,1)}s")
            return answer
        except Exception as e:
            logger.error(f"  [{label}] FAILED | {e}")
            return ""

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(a_norm, b_norm.T)

    def _diagnose(
        self, ptype: str, sem_sim: float, rouge_l: float, is_robust: bool
    ) -> str:
        if is_robust:
            return (
                f"ROBUST: Model tidak terpengaruh perturbasi '{ptype}' "
                f"(semantic={sem_sim:.4f} >= threshold={self.robustness_threshold})."
            )

        ptype_desc = {
            PERTURBATION_TYPO:      "typo satu huruf",
            PERTURBATION_LOWERCASE: "semua huruf kecil",
            PERTURBATION_NOISE:     "noise di konteks",
            PERTURBATION_REORDER:   "reorder pertanyaan",
        }.get(ptype, ptype)

        if sem_sim < 0.60:
            return (
                f"HIGHLY SENSITIVE: Jawaban berubah signifikan akibat {ptype_desc} "
                f"(semantic={sem_sim:.4f}). Model sangat tidak robust."
            )
        else:
            return (
                f"SENSITIVE: Jawaban sedikit berubah akibat {ptype_desc} "
                f"(semantic={sem_sim:.4f} < threshold={self.robustness_threshold}). "
                f"Review manual disarankan."
            )

    def _build_no_variants_result(
        self, case_id, model_name, question, original_answer, variants
    ) -> SensitivityResult:
        return SensitivityResult(
            case_id=case_id, model=model_name,
            original_question=question, original_answer=original_answer,
            total_perturbations=0, variants=variants, comparisons=[],
            avg_semantic_similarity=0.0, avg_rouge_l=0.0,
            robustness_score=0.0, is_fully_robust=False, has_failure=True,
            sensitive_perturbations=[{"diagnosis": "No valid perturbed variants."}],
            robustness_threshold=self.robustness_threshold,
            embedding_model=self.embedding_model,
            evaluator_version=self.VERSION,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

# Smoke test
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    print("=" * 60)
    print("SensitivityEvaluator — Smoke Test")
    print("Hanya butuh nomic-embed-text, tidak butuh LLM")
    print("=" * 60)

    evaluator = SensitivityEvaluator(robustness_threshold=0.85)

    print("\n[1] Health check...")
    if not evaluator.health_check():
        print("nomic-embed-text tidak tersedia.")
        sys.exit(1)
    print("Siap.\n")

    print("[2] Test perturbation functions...")
    q = "What does FastAPI use for data validation?"
    ctx = "FastAPI uses Pydantic for data validation and supports async operations."

    print(f"   Original  : {q}")
    print(f"   Typo      : {perturb_typo(q)}")
    print(f"   Lowercase : {perturb_lowercase(q)}")
    print(f"   Reorder   : {perturb_question_reorder(q)}")
    print(f"   Ctx noise : {perturb_context_noise(ctx)[-60:]}")

    print("\n[3] Robust model test...")
    r1 = evaluator.evaluate_from_answers(
        case_id="sensitivity_smoke_001",
        original_question=q,
        original_context=ctx,
        original_answer="FastAPI uses Pydantic for data validation.",
        perturbed_variants=[
            {"perturbation_type": PERTURBATION_TYPO,
             "perturbed_question": perturb_typo(q),
             "perturbed_context": ctx,
             "answer": "FastAPI uses Pydantic for data validation."},
            {"perturbation_type": PERTURBATION_LOWERCASE,
             "perturbed_question": perturb_lowercase(q),
             "perturbed_context": ctx,
             "answer": "FastAPI utilizes Pydantic for validating data."},
        ],
        model_name="smoke_test",
    )
    print(f"   Robustness: {r1.robustness_score} | Failure: {r1.has_failure}")

    print("\n[4] Sensitive model test...")
    r2 = evaluator.evaluate_from_answers(
        case_id="sensitivity_smoke_002",
        original_question=q,
        original_context=ctx,
        original_answer="FastAPI uses Pydantic for data validation.",
        perturbed_variants=[
            {"perturbation_type": PERTURBATION_TYPO,
             "perturbed_question": perturb_typo(q),
             "perturbed_context": ctx,
             "answer": "I cannot determine the answer from the given context."},
            {"perturbation_type": PERTURBATION_NOISE,
             "perturbed_question": q,
             "perturbed_context": perturb_context_noise(ctx),
             "answer": "Marshmallow is used for serialization in FastAPI projects."},
        ],
        model_name="smoke_test",
    )
    print(f"   Robustness: {r2.robustness_score} | Failure: {r2.has_failure}")
    for sp in r2.sensitive_perturbations:
        print(f"{sp['perturbation_type']}: {sp['diagnosis'][:80]}")

    evaluator.print_summary([r1, r2])