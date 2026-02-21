import re
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import requests

logger = logging.getLogger("faithfulness")

# Konstanta
DEFAULT_SIMILARITY_THRESHOLD = 0.75
INSUFFICIENT_VALIDATION_THRESHOLD = 0.65
INSUFFICIENT_VALIDATION_THRESHOLD = 0.65
MIN_CLAIM_LENGTH_WORDS = 2
INSUFFICIENT_CONTEXT_SIGNAL = "INSUFFICIENT_CONTEXT"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"


# Data classes
@dataclass
class Claim:
    claim_id: str
    text: str
    source_answer: str


@dataclass
class EvidenceMatch:
    claim_id: str
    claim_text: str
    best_evidence_span: str
    similarity_score: float
    classification: str       
    threshold_used: float


@dataclass
class FaithfulnessResult:
    case_id: str
    model: str
    question: str
    answer: str
    context_preview: str

    total_claims: int
    claims: list
    evidence_matches: list

    supported_count: int
    unsupported_count: int
    skipped_count: int
    faithfulness_score: float

    is_insufficient_context_response: bool
    has_failure: bool
    failure_cases: list

    similarity_threshold: float
    embedding_model: str
    evaluator_version: str
    timestamp_utc: str


# Ollama Embedder
class OllamaEmbedder:
    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._endpoint = f"{self.base_url}/api/embeddings"
        logger.info(f"OllamaEmbedder initialized | model={self.model}")

    def embed(self, text: str) -> np.ndarray:
        response = requests.post(
            self._endpoint,
            json={"model": self.model, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)

    def embed_batch(self, texts: list) -> np.ndarray:
        embeddings = []
        for i, text in enumerate(texts):
            logger.debug(f"Embedding {i+1}/{len(texts)}")
            embeddings.append(self.embed(text))
        return np.stack(embeddings)

    def health_check(self) -> bool:
        try:
            self.embed("health check")
            logger.info(f"Embedder health check OK | model={self.model}")
            return True
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Run: ollama serve")
            return False
        except Exception as e:
            logger.error(f"Embedder health check FAILED | {e}")
            return False


# Claim Extractor
class ClaimExtractor:
    def extract(self, answer: str, case_id: str) -> list:
        if not answer or not answer.strip():
            return []

        raw_sentences = self._split_into_sentences(answer.strip())
        candidates = []

        for sentence in raw_sentences:
            sentence = sentence.strip()
            if self._is_filler(sentence):
                continue

            expanded = self._expand_enumeration(sentence)
            if expanded:
                candidates.extend(expanded)
            else:
                candidates.append(sentence)

        claims = []
        for i, text in enumerate(candidates, 1):
            if len(text.split()) < MIN_CLAIM_LENGTH_WORDS:
                logger.debug(f"Skipping short claim ({len(text.split())} words): '{text}'")
                continue
            claims.append(Claim(
                claim_id=f"{case_id}_claim_{i:02d}",
                text=text,
                source_answer=answer,
            ))

        logger.info(f"Extracted {len(claims)} claims | case_id={case_id}")
        return claims

    def _split_into_sentences(self, text: str) -> list:
        text = re.sub(r'\n\s*[-•*]\s+', '\n', text)
        text = re.sub(r'\n\s*\d+\.\s+', '\n', text)

        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        result = []
        for s in sentences:
            parts = [p.strip() for p in s.split('\n') if p.strip()]
            result.extend(parts)
        return result

    def _expand_enumeration(self, sentence: str) -> list:
        list_triggers = [
            r'(includes?)\s+',
            r'(contains?)\s+',
            r'(supports?)\s+',
            r'(provides?)\s+',
            r'(consists? of)\s+',
            r'(such as)\s+',
        ]

        for trigger_pattern in list_triggers:
            pattern = rf'^(.+?\s+{trigger_pattern})(.+)$'
            match = re.match(pattern, sentence, re.IGNORECASE)
            if not match:
                continue

            prefix = match.group(1).strip()         
            items_str = match.group(3).strip()       

            items_str = items_str.rstrip('.')

            raw_items = [i.strip() for i in items_str.split(',')]
            cleaned_items = []
            for item in raw_items:
                item = re.sub(r'^and\s+', '', item.strip())
                if item:
                    cleaned_items.append(item)

            if len(cleaned_items) < 2:
                return []

            expanded = []
            for item in cleaned_items:
                claim_text = f"{prefix} {item}."
                expanded.append(claim_text)

            logger.debug(
                f"Expanded enumeration into {len(expanded)} claims: "
                f"{[c[:50] for c in expanded]}"
            )
            return expanded
        return []

    def _is_filler(self, sentence: str) -> bool:
        filler_patterns = [
            r'^based on (the |this )?(context|document|information|text)',
            r'^according to (the |this )?(context|document)',
            r'^the (context|document|provided text) (states?|says?|mentions?)',
            r'^sure[,.]',
            r'^of course[,.]',
            r'^here is',
            r'^to answer',
        ]
        s = sentence.lower()
        return any(re.match(p, s) for p in filler_patterns)


# Evidence Matcher
class EvidenceMatcher:
    def __init__(
        self,
        embedder: OllamaEmbedder,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    def match(self, claims: list, context: str) -> list:
        if not claims:
            return []

        context_spans = self._split_context(context)
        if not context_spans:
            logger.warning("Context produced no valid spans")
            return [
                EvidenceMatch(
                    claim_id=c.claim_id, claim_text=c.text,
                    best_evidence_span="", similarity_score=0.0,
                    classification="UNSUPPORTED",
                    threshold_used=self.similarity_threshold,
                )
                for c in claims
            ]

        logger.info(
            f"Embedding {len(claims)} claims + {len(context_spans)} spans "
            f"via {self.embedder.model}..."
        )
        claim_embs = self.embedder.embed_batch([c.text for c in claims])
        span_embs = self.embedder.embed_batch(context_spans)

        sim_matrix = self._cosine_similarity(claim_embs, span_embs)

        results = []
        for i, claim in enumerate(claims):
            scores = sim_matrix[i]
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            classification = "SUPPORTED" if best_score >= self.similarity_threshold else "UNSUPPORTED"

            results.append(EvidenceMatch(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                best_evidence_span=context_spans[best_idx],
                similarity_score=round(best_score, 4),
                classification=classification,
                threshold_used=self.similarity_threshold,
            ))

            logger.debug(
                f"{claim.claim_id} | {classification} | "
                f"score={best_score:.4f}"
            )

        return results

    def _split_context(self, context: str) -> list:
        spans = re.split(r'(?<=[.!?])\s+', context.strip())
        return [s.strip() for s in spans if len(s.strip().split()) >= 3]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(a_norm, b_norm.T)


# Faithfulness Evaluator 
class FaithfulnessEvaluator:
    VERSION = "1.1.0" 
    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        ollama_base_url: str = OLLAMA_BASE_URL,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.extractor = ClaimExtractor()
        self.embedder = OllamaEmbedder(model=embedding_model, base_url=ollama_base_url)
        self.matcher = EvidenceMatcher(
            embedder=self.embedder,
            similarity_threshold=similarity_threshold,
        )
        logger.info(
            f"FaithfulnessEvaluator ready | "
            f"threshold={similarity_threshold} | "
            f"embedding={embedding_model} | "
            f"version={self.VERSION}"
        )

    def health_check(self) -> bool:
        return self.embedder.health_check()

    def evaluate(
        self,
        case: dict,
        model_answer: str,
        model_name: str = "unknown",
    ) -> FaithfulnessResult:
        case_id = case["id"]
        context = case["context"]
        question = case["question"]

        logger.info(f"Evaluating | case_id={case_id}")

        if INSUFFICIENT_CONTEXT_SIGNAL in model_answer.upper():
            logger.info(f"Model returned INSUFFICIENT_CONTEXT | case_id={case_id}")
            ground_truth = case.get("ground_truth") or case.get("ground_truth_from_context")
            if ground_truth:
                gt_embedding = self.embedder.embed(ground_truth)
                context_spans = self.matcher._split_context(context)
                if context_spans:
                    span_embeddings = self.embedder.embed_batch(context_spans)
                    sims = self.matcher._cosine_similarity(
                        gt_embedding.reshape(1, -1), span_embeddings
                    )[0]
                    max_sim = float(sims.max())
                    logger.info(
                        f"INSUFFICIENT_CONTEXT validation | "
                        f"ground_truth↔context similarity={max_sim:.4f} | "
                        f"threshold={INSUFFICIENT_VALIDATION_THRESHOLD}"
                    )
                    if max_sim >= INSUFFICIENT_VALIDATION_THRESHOLD:
                        logger.warning(
                            f"FALSE INSUFFICIENT_CONTEXT detected | case_id={case_id} | "
                            f"Evidence exists in context (sim={max_sim:.4f}) but model claimed insufficient."
                        )
                        return self._build_false_insufficient_result(
                            case_id, model_name, question, model_answer,
                            context, max_sim
                        )

            return self._build_insufficient_result(
                case_id, model_name, question, model_answer, context
            )

        # Claim Extraction
        claims = self.extractor.extract(answer=model_answer, case_id=case_id)
        if not claims:
            return self._build_empty_result(
                case_id, model_name, question, model_answer, context
            )

        # evidence matching
        evidence_matches = self.matcher.match(claims=claims, context=context)

        # scoring
        supported = [m for m in evidence_matches if m.classification == "SUPPORTED"]
        unsupported = [m for m in evidence_matches if m.classification == "UNSUPPORTED"]
        skipped = [m for m in evidence_matches if m.classification == "SKIPPED"]

        evaluable = len(claims) - len(skipped)
        score = len(supported) / evaluable if evaluable > 0 else 0.0

        # failure case documentation
        failure_cases = [
            {
                "claim_id": m.claim_id,
                "unsupported_claim": m.claim_text,
                "best_match_in_context": m.best_evidence_span,
                "similarity_score": m.similarity_score,
                "threshold": m.threshold_used,
                "diagnosis": self._diagnose_failure(m),
            }
            for m in unsupported
        ]

        result = FaithfulnessResult(
            case_id=case_id,
            model=model_name,
            question=question,
            answer=model_answer,
            context_preview=context[:200] + "..." if len(context) > 200 else context,
            total_claims=len(claims),
            claims=claims,
            evidence_matches=evidence_matches,
            supported_count=len(supported),
            unsupported_count=len(unsupported),
            skipped_count=len(skipped),
            faithfulness_score=round(score, 4),
            is_insufficient_context_response=False,
            has_failure=len(unsupported) > 0,
            failure_cases=failure_cases,
            similarity_threshold=self.similarity_threshold,
            embedding_model=self.embedding_model,
            evaluator_version=self.VERSION,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

        logger.log(
            logging.WARNING if result.has_failure else logging.INFO,
            f"Result | case_id={case_id} | score={score:.4f} | "
            f"supported={len(supported)} | unsupported={len(unsupported)}"
        )

        return result

    def evaluate_batch(
        self,
        cases: list,
        model_answers: list,
        model_name: str = "unknown",
    ) -> list:
        assert len(cases) == len(model_answers), (
            f"Jumlah cases ({len(cases)}) != model_answers ({len(model_answers)})"
        )
        return [
            self.evaluate(case=c, model_answer=a, model_name=model_name)
            for c, a in zip(cases, model_answers)
        ]

    def save_results(self, results: list, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved | path={output_path} | count={len(results)}")

    def print_summary(self, results: list) -> None:
        total = len(results)
        if not total:
            print("Tidak ada hasil.")
            return

        scores = [r.faithfulness_score for r in results]
        avg = sum(scores) / total
        failed = [r for r in results if r.has_failure]

        print("\n" + "=" * 60)
        print("FAITHFULNESS EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total cases       : {total}")
        print(f"Average score     : {avg:.4f} ({avg*100:.1f}%)")
        print(f"Perfect (1.0)     : {sum(1 for s in scores if s == 1.0)}")
        print(f"Cases with failure: {len(failed)}")
        print(f"Threshold         : {results[0].similarity_threshold}")
        print(f"Embedding model   : {results[0].embedding_model}")

        if failed:
            print("\nFAILURE CASES:")
            print("-" * 60)
            for r in failed:
                print(f"\n  case_id : {r.case_id}")
                print(f"  score   : {r.faithfulness_score}")
                print(f"  question: {r.question[:80]}")
                for fc in r.failure_cases:
                    print(f"\n  Unsupported claim:")
                    print(f"     {fc['unsupported_claim'][:100]}")
                    print(f"  Best match (score={fc['similarity_score']}):")
                    print(f"     {fc['best_match_in_context'][:100]}")
                    print(f"  Diagnosis: {fc['diagnosis']}")

        print("\n" + "=" * 60)

    # private helpers
    def _diagnose_failure(self, match: EvidenceMatch) -> str:
        score = match.similarity_score
        if score >= 0.60:
            return (
                f"Semantically close (score={score}) but below threshold. "
                "Possible paraphrase or partial evidence — review manually."
            )
        elif score >= 0.40:
            return (
                f"Weakly related (score={score}). "
                "Claim may be drawn from training data outside context."
            )
        else:
            return (
                f"No relevant evidence found (score={score}). "
                "High likelihood of hallucination or prior knowledge injection."
            )

    def _build_false_insufficient_result(
        self, case_id, model_name, question, answer, context, evidence_sim
    ) -> FaithfulnessResult:
        return FaithfulnessResult(
            case_id=case_id, model=model_name, question=question,
            answer=answer, context_preview=context[:200],
            total_claims=0, claims=[], evidence_matches=[],
            supported_count=0, unsupported_count=1, skipped_count=0,
            faithfulness_score=0.0,
            is_insufficient_context_response=True,
            has_failure=True,
            failure_cases=[{
                "claim_id": f"{case_id}_false_insufficient",
                "unsupported_claim": "Model claimed INSUFFICIENT_CONTEXT",
                "best_match_in_context": f"Evidence exists (similarity={evidence_sim:.4f})",
                "similarity_score": evidence_sim,
                "threshold": self.similarity_threshold,
                "diagnosis": (
                    f"FALSE INSUFFICIENT_CONTEXT: Model refused to answer despite evidence "
                    f"in context (similarity={evidence_sim:.4f} >= threshold={self.similarity_threshold}). "
                    f"Model over-triggered escape hatch or failed to read context correctly."
                ),
            }],
            similarity_threshold=self.similarity_threshold,
            embedding_model=self.embedding_model,
            evaluator_version=self.VERSION,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

    def _build_insufficient_result(
        self, case_id, model_name, question, answer, context
    ) -> FaithfulnessResult:
        return FaithfulnessResult(
            case_id=case_id, model=model_name, question=question,
            answer=answer, context_preview=context[:200],
            total_claims=0, claims=[], evidence_matches=[],
            supported_count=0, unsupported_count=0, skipped_count=0,
            faithfulness_score=1.0,        
            is_insufficient_context_response=True,
            has_failure=False, failure_cases=[],
            similarity_threshold=self.similarity_threshold,
            embedding_model=self.embedding_model,
            evaluator_version=self.VERSION,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

    def _build_empty_result(
        self, case_id, model_name, question, answer, context
    ) -> FaithfulnessResult:
        return FaithfulnessResult(
            case_id=case_id, model=model_name, question=question,
            answer=answer, context_preview=context[:200],
            total_claims=0, claims=[], evidence_matches=[],
            supported_count=0, unsupported_count=0, skipped_count=0,
            faithfulness_score=0.0,
            is_insufficient_context_response=False,
            has_failure=True,
            failure_cases=[{
                "claim_id": f"{case_id}_claim_none",
                "unsupported_claim": "No extractable claims from answer",
                "best_match_in_context": "",
                "similarity_score": 0.0,
                "threshold": self.similarity_threshold,
                "diagnosis": "Model returned an empty or unparseable answer.",
            }],
            similarity_threshold=self.similarity_threshold,
            embedding_model=self.embedding_model,
            evaluator_version=self.VERSION,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )


# smoke test
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    print("=" * 60)
    print("FaithfulnessEvaluator — Smoke Test")
    print(f"Embedding: {EMBEDDING_MODEL} via Ollama")
    print("=" * 60)

    evaluator = FaithfulnessEvaluator(similarity_threshold=0.75)

    print("\n[1] Health check...")
    if not evaluator.health_check():
        print("\n Ollama atau nomic-embed-text tidak siap.")
        print("   Pastikan Ollama berjalan  : ollama serve")
        print("   Pastikan model tersedia   : ollama list")
        sys.exit(1)

    print(" Siap.\n")

    test_case = {
        "id": "smoke_001",
        "context": (
            "FastAPI generates a schema with all your API using the OpenAPI standard. "
            "This schema definition includes your API paths, the possible parameters they take."
        ),
        "question": "What does the OpenAPI schema generated by FastAPI include?",
    }

    print("[2] Faithful answer...")
    r1 = evaluator.evaluate(
        test_case,
        "The OpenAPI schema includes your API paths and the possible parameters they take.",
        model_name="smoke_test",
    )
    print(f"   Score: {r1.faithfulness_score} | Failure: {r1.has_failure}")

    print("\n[3] Hallucinated answer...")
    r2 = evaluator.evaluate(
        test_case,
        (
            "The OpenAPI schema includes your API paths, parameters, "
            "authentication methods, response schemas, and rate limiting configuration."
        ),
        model_name="smoke_test",
    )
    print(f"   Score: {r2.faithfulness_score} | Failure: {r2.has_failure}")
    for fc in r2.failure_cases:
        print(f"   {fc['unsupported_claim'][:80]}")
        print(f"      → {fc['diagnosis']}")

    evaluator.print_summary([r1, r2])