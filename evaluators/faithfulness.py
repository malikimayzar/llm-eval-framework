"""
evaluators/faithfulness.py
──────────────────────────
Faithfulness evaluator v1.2.0

Changelog dari v1.1.0:
- [FIX] Short answer path (_evaluate_short_answer) sekarang terhubung ke evaluate()
  Sebelumnya: method ada tapi tidak pernah dipanggil → clean_008, clean_009 selalu
  masuk claim extraction path dan dapat score 0 (blind spot terkonfirmasi)
- [NEW] Technical conflict detection (_has_technical_conflict)
  PUT vs PATCH similarity = 0.8337 tapi factually berbeda → sekarang di-override ke 0.0
  Konfigurasi via config.yaml: technical_term_groups
- [NEW] Config-driven thresholds via pipeline.config_loader
  Tidak ada lagi magic numbers tersebar di source code
- [FIX] Removed duplicate INSUFFICIENT_VALIDATION_THRESHOLD assignment (typo di v1.1.0)
"""

import re
from difflib import SequenceMatcher
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import requests

from pipeline.config_loader import config

# ---------------------------------------------------------------------------
# Short answer helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _char_similarity(a: str, b: str) -> float:
    """Character-level similarity via SequenceMatcher (0-1)."""
    return SequenceMatcher(None, a, b).ratio()


def _is_short_answer(answer: str) -> bool:
    """Cek apakah jawaban termasuk kategori singkat (di bawah threshold dari config)."""
    threshold = config["thresholds"]["short_answer_max_words"]
    return len(answer.strip().split()) < threshold


logger = logging.getLogger("faithfulness")

# ---------------------------------------------------------------------------
# Konstanta — nilai default, akan di-override oleh config jika tersedia
# ---------------------------------------------------------------------------
DEFAULT_SIMILARITY_THRESHOLD    = config["thresholds"]["faithfulness_evidence"]
INSUFFICIENT_VALIDATION_THRESHOLD = config["thresholds"]["insufficient_ctx_validation"]
MIN_CLAIM_LENGTH_WORDS          = 2
SHORT_ANSWER_WORD_THRESHOLD     = config["thresholds"]["short_answer_max_words"]
SHORT_ANSWER_CHAR_SIM_THRESHOLD = config["thresholds"]["short_answer_char_sim"]
INSUFFICIENT_CONTEXT_SIGNAL     = "INSUFFICIENT_CONTEXT"
OLLAMA_BASE_URL                 = "http://localhost:11434"
EMBEDDING_MODEL                 = config["models"]["embedding"]

# Technical term groups dari config — untuk conflict detection
_TECHNICAL_TERM_GROUPS = [
    set(group) for group in config.get("technical_term_groups", [])
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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
    technical_conflict: bool = False   # [NEW] flag jika conflict terdeteksi


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
    evaluation_path: str = "claim_extraction"


# ---------------------------------------------------------------------------
# Ollama Embedder
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Claim Extractor
# ---------------------------------------------------------------------------

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
            items_str = match.group(3).strip().rstrip('.')
            raw_items = [i.strip() for i in items_str.split(',')]
            cleaned_items = []
            for item in raw_items:
                item = re.sub(r'^and\s+', '', item.strip())
                if item:
                    cleaned_items.append(item)

            if len(cleaned_items) < 2:
                return []

            expanded = [f"{prefix} {item}." for item in cleaned_items]
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


# ---------------------------------------------------------------------------
# Evidence Matcher — dengan technical conflict detection
# ---------------------------------------------------------------------------

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
        span_embs  = self.embedder.embed_batch(context_spans)
        sim_matrix = self._cosine_similarity(claim_embs, span_embs)

        results = []
        for i, claim in enumerate(claims):
            scores = sim_matrix[i].copy()

            # ── [NEW] TECHNICAL CONFLICT CHECK ──────────────────────────────
            # Sebelum pilih best span, nol-kan span yang conflicting.
            # Ini mencegah PUT vs PATCH (similarity 0.83) dianggap SUPPORTED.
            conflict_detected = False
            for j, span in enumerate(context_spans):
                if _has_technical_conflict(claim.text, span):
                    logger.debug(
                        f"Technical conflict | claim='{claim.text[:50]}' "
                        f"span='{span[:50]}' | score zeroed from {scores[j]:.4f}"
                    )
                    scores[j] = 0.0
                    conflict_detected = True
            # ────────────────────────────────────────────────────────────────

            best_idx   = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            classification = "SUPPORTED" if best_score >= self.similarity_threshold else "UNSUPPORTED"

            results.append(EvidenceMatch(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                best_evidence_span=context_spans[best_idx],
                similarity_score=round(best_score, 4),
                classification=classification,
                threshold_used=self.similarity_threshold,
                technical_conflict=conflict_detected,
            ))

            logger.debug(
                f"{claim.claim_id} | {classification} | "
                f"score={best_score:.4f} | conflict={conflict_detected}"
            )

        return results

    def _split_context(self, context: str) -> list:
        spans = re.split(r'(?<=[.!?])\s+', context.strip())
        return [s.strip() for s in spans if len(s.strip().split()) >= 3]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(a_norm, b_norm.T)


# ---------------------------------------------------------------------------
# [NEW] Technical conflict detection — module-level function
# ---------------------------------------------------------------------------

def _has_technical_conflict(claim: str, evidence: str) -> bool:
    """
    Return True jika claim dan evidence menggunakan technical term yang berbeda
    dari grup yang sama.

    Contoh yang benar di-catch:
        claim="use PUT method" + evidence="use PATCH method"
        → group HTTP methods → claim_terms={'put'}, evidence_terms={'patch'} → conflict

    Contoh yang tidak di-catch (intentional):
        claim="POST or PUT" + evidence="POST or PUT" → terms sama → no conflict
        claim="send a request" + evidence="use POST" → claim tidak punya HTTP term → no conflict

    Config: technical_term_groups di config.yaml
    """
    claim_lower    = set(re.findall(r'\b\w+\b', claim.lower()))
    evidence_lower = set(re.findall(r'\b\w+\b', evidence.lower()))

    for term_group in _TECHNICAL_TERM_GROUPS:
        claim_hits    = claim_lower & term_group
        evidence_hits = evidence_lower & term_group

        # Conflict: keduanya punya term dari grup yang sama, tapi term-nya berbeda
        if claim_hits and evidence_hits and claim_hits != evidence_hits:
            logger.debug(
                f"Technical conflict detected | "
                f"claim_terms={claim_hits} vs evidence_terms={evidence_hits}"
            )
            return True
    return False


# ---------------------------------------------------------------------------
# Faithfulness Evaluator
# ---------------------------------------------------------------------------

class FaithfulnessEvaluator:
    VERSION = "1.2.1"

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        ollama_base_url: str = OLLAMA_BASE_URL,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.extractor = ClaimExtractor()
        self.embedder  = OllamaEmbedder(model=embedding_model, base_url=ollama_base_url)
        self.matcher   = EvidenceMatcher(
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

    # ── [FIX] Short answer path — sekarang benar-benar dipanggil dari evaluate() ──

    def _evaluate_short_answer(
        self,
        case: dict,
        answer: str,
        model_name: str,
    ) -> "FaithfulnessResult":
        """
        Path alternatif untuk jawaban singkat (< SHORT_ANSWER_WORD_THRESHOLD kata).

        v1.1.0: Method ini ada tapi tidak pernah dipanggil dari evaluate() — blind spot.
        v1.2.0: [FIXED] Sekarang dipanggil di awal evaluate() sebelum claim extraction.

        Logika scoring:
            1. Exact match vs ground_truth          → score 1.0
            2. Substring match (satu ada di lain)   → score 0.9
            3. Char similarity >= threshold          → score = char_sim
            4. Tidak ada match + tidak grounded      → score = char_sim * 0.5
        """
        case_id      = case["id"]
        context      = case["context"]
        question     = case["question"]
        ground_truth = case.get("ground_truth", "")

        norm_answer = _normalize(answer)
        norm_truth  = _normalize(ground_truth)
        norm_ctx    = _normalize(context)

        # Tahap 1: Exact match
        if norm_answer == norm_truth:
            match_type = "exact"
            base_score = 1.0
            matched    = True
        # Tahap 2: Substring match
        elif norm_answer in norm_truth or norm_truth in norm_answer:
            match_type = "substring"
            base_score = 0.9
            matched    = True
        # Tahap 3: Character similarity
        else:
            char_sim   = _char_similarity(norm_answer, norm_truth)
            matched    = char_sim >= SHORT_ANSWER_CHAR_SIM_THRESHOLD
            match_type = "char_similarity" if matched else "no_match"
            base_score = round(char_sim, 4)

        grounded = norm_answer in norm_ctx
        score    = base_score if (grounded or matched) else round(base_score * 0.5, 4)
        has_fail = not matched

        diagnosis = (
            f"Short answer ({len(answer.split())} words). "
            f"Match: {match_type}. Grounded in context: {grounded}. Score: {score}."
        )

        failure_cases = []
        if has_fail:
            failure_cases.append({
                "claim_id": f"{case_id}_short_answer",
                "unsupported_claim": answer,
                "best_match_in_context": ground_truth,
                "similarity_score": base_score,
                "threshold": SHORT_ANSWER_CHAR_SIM_THRESHOLD,
                "diagnosis": diagnosis,
            })

        logger.log(
            logging.WARNING if has_fail else logging.INFO,
            f"Short answer eval | case_id={case_id} | "
            f"match={match_type} | score={score} | grounded={grounded}"
        )

        return FaithfulnessResult(
            case_id=case_id,
            model=model_name,
            question=question,
            answer=answer,
            context_preview=context[:200] + "..." if len(context) > 200 else context,
            total_claims=1,
            claims=[],
            evidence_matches=[],
            supported_count=1 if matched else 0,
            unsupported_count=0 if matched else 1,
            skipped_count=0,
            faithfulness_score=score,
            has_failure=has_fail,
            failure_cases=failure_cases,
            is_insufficient_context_response=False,
            evaluation_path="short_answer_exact_match",
            embedding_model=self.embedding_model,
            similarity_threshold=self.similarity_threshold,
            evaluator_version=self.VERSION,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

    def evaluate(
        self,
        case: dict,
        model_answer: str,
        model_name: str = "unknown",
    ) -> FaithfulnessResult:
        case_id  = case["id"]
        context  = case["context"]
        question = case["question"]

        logger.info(f"Evaluating | case_id={case_id}")

        # ── BRANCH 1: EMPTY ANSWER ────────────────────────────────────────────
        # Harus dicek pertama — string kosong lolos ke short_answer path dan
        # substring match dengan ground_truth → score 0.9 (false positive).
        if not model_answer or not model_answer.strip():
            return self._build_empty_result(case_id, model_name, question, model_answer, context)
        # ─────────────────────────────────────────────────────────────────────

        # ── BRANCH 2: INSUFFICIENT CONTEXT SIGNAL ────────────────────────────
        # Harus dicek sebelum short_answer — "INSUFFICIENT_CONTEXT" adalah
        # 1 kata sehingga akan masuk short_answer path jika tidak dicek duluan.
        if INSUFFICIENT_CONTEXT_SIGNAL in model_answer.upper():
            logger.info(f"Model returned INSUFFICIENT_CONTEXT | case_id={case_id}")
            ground_truth = case.get("ground_truth") or case.get("ground_truth_from_context")
            if ground_truth:
                gt_embedding   = self.embedder.embed(ground_truth)
                context_spans  = self.matcher._split_context(context)
                if context_spans:
                    span_embeddings = self.embedder.embed_batch(context_spans)
                    sims    = self.matcher._cosine_similarity(
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
                            f"Evidence exists (sim={max_sim:.4f}) but model claimed insufficient."
                        )
                        return self._build_false_insufficient_result(
                            case_id, model_name, question, model_answer,
                            context, max_sim
                        )

            return self._build_insufficient_result(
                case_id, model_name, question, model_answer, context
            )
        # ─────────────────────────────────────────────────────────────────────

        # ── BRANCH 3: SHORT ANSWER PATH ───────────────────────────────────────
        # Dicek setelah INSUFFICIENT_CONTEXT agar signal keyword tidak salah path.
        # clean_008 ('PUT') dan clean_009 ('/files/...') masuk sini.
        if _is_short_answer(model_answer):
            logger.info(
                f"Short answer detected ({len(model_answer.split())} words) | "
                f"case_id={case_id} → short_answer path"
            )
            return self._evaluate_short_answer(case, model_answer, model_name)
        # ─────────────────────────────────────────────────────────────────────

        # ── BRANCH 4 (NORMAL): Claim Extraction → Evidence Matching ──────────
        claims = self.extractor.extract(answer=model_answer, case_id=case_id)
        if not claims:
            return self._build_empty_result(
                case_id, model_name, question, model_answer, context
            )

        # Evidence matching (sekarang include technical conflict detection di dalam matcher)
        evidence_matches = self.matcher.match(claims=claims, context=context)

        supported   = [m for m in evidence_matches if m.classification == "SUPPORTED"]
        unsupported = [m for m in evidence_matches if m.classification == "UNSUPPORTED"]
        skipped     = [m for m in evidence_matches if m.classification == "SKIPPED"]

        evaluable = len(claims) - len(skipped)
        score     = len(supported) / evaluable if evaluable > 0 else 0.0

        failure_cases = [
            {
                "claim_id": m.claim_id,
                "unsupported_claim": m.claim_text,
                "best_match_in_context": m.best_evidence_span,
                "similarity_score": m.similarity_score,
                "threshold": m.threshold_used,
                "technical_conflict": m.technical_conflict,
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

        scores  = [r.faithfulness_score for r in results]
        avg     = sum(scores) / total
        failed  = [r for r in results if r.has_failure]
        by_path = {}
        for r in results:
            by_path[r.evaluation_path] = by_path.get(r.evaluation_path, 0) + 1

        print("\n" + "=" * 60)
        print("FAITHFULNESS EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total cases       : {total}")
        print(f"Average score     : {avg:.4f} ({avg*100:.1f}%)")
        print(f"Perfect (1.0)     : {sum(1 for s in scores if s == 1.0)}")
        print(f"Cases with failure: {len(failed)}")
        print(f"Threshold         : {results[0].similarity_threshold}")
        print(f"Embedding model   : {results[0].embedding_model}")
        print(f"Evaluation paths  : {by_path}")

        if failed:
            print("\nFAILURE CASES:")
            print("-" * 60)
            for r in failed:
                print(f"\n  case_id : {r.case_id}")
                print(f"  score   : {r.faithfulness_score}")
                print(f"  path    : {r.evaluation_path}")
                print(f"  question: {r.question[:80]}")
                for fc in r.failure_cases:
                    print(f"\n  Unsupported claim:")
                    print(f"     {fc['unsupported_claim'][:100]}")
                    print(f"  Best match (score={fc['similarity_score']}):")
                    print(f"     {fc.get('best_match_in_context', '')[:100]}")
                    if fc.get("technical_conflict"):
                        print(f"  ⚠ Technical conflict detected")
                    print(f"  Diagnosis: {fc['diagnosis']}")

        print("\n" + "=" * 60)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _diagnose_failure(self, match: EvidenceMatch) -> str:
        if match.technical_conflict:
            return (
                f"Technical term conflict detected (score={match.similarity_score}). "
                "Claim and evidence use different terms from the same technical group "
                "(e.g. PUT vs PATCH). Semantically similar but factually incorrect."
            )
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
                "technical_conflict": False,
                "diagnosis": (
                    f"FALSE INSUFFICIENT_CONTEXT: Model refused to answer despite evidence "
                    f"in context (similarity={evidence_sim:.4f} >= threshold={INSUFFICIENT_VALIDATION_THRESHOLD}). "
                    f"Model over-triggered escape hatch or failed to read context correctly."
                ),
            }],
            similarity_threshold=self.similarity_threshold,
            embedding_model=self.embedding_model,
            evaluator_version=self.VERSION,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            evaluation_path="false_insufficient_context",
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
            evaluation_path="insufficient_context",
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
                "technical_conflict": False,
                "diagnosis": "Model returned an empty or unparseable answer.",
            }],
            similarity_threshold=self.similarity_threshold,
            embedding_model=self.embedding_model,
            evaluator_version=self.VERSION,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            evaluation_path="empty_answer",
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    print("=" * 60)
    print(f"FaithfulnessEvaluator — Smoke Test (v{FaithfulnessEvaluator.VERSION})")
    print(f"Embedding : {EMBEDDING_MODEL} via Ollama")
    print(f"Threshold : {DEFAULT_SIMILARITY_THRESHOLD}")
    print("=" * 60)

    evaluator = FaithfulnessEvaluator()

    print("\n[1] Health check...")
    if not evaluator.health_check():
        print("\n  Ollama atau nomic-embed-text tidak siap.")
        print("  Pastikan Ollama berjalan : ollama serve")
        print("  Pastikan model tersedia  : ollama list")
        sys.exit(1)
    print("  Siap.\n")

    base_case = {
        "id": "smoke_001",
        "context": (
            "FastAPI generates a schema with all your API using the OpenAPI standard. "
            "This schema definition includes your API paths, the possible parameters they take. "
            "To update a resource, you should use the PATCH method, not PUT."
        ),
        "question": "What does the OpenAPI schema generated by FastAPI include?",
        "ground_truth": "API paths and the possible parameters",
    }

    print("[2] Normal answer (faithful)...")
    r1 = evaluator.evaluate(
        base_case,
        "The OpenAPI schema includes your API paths and the possible parameters they take.",
        model_name="smoke_test",
    )
    print(f"   Score: {r1.faithfulness_score} | Path: {r1.evaluation_path} | Failure: {r1.has_failure}")

    print("\n[3] Short answer — exact match ('PUT')...")
    short_case = {**base_case, "id": "smoke_002", "ground_truth": "PATCH"}
    r2 = evaluator.evaluate(short_case, "PUT", model_name="smoke_test")
    print(f"   Score: {r2.faithfulness_score} | Path: {r2.evaluation_path}")
    print(f"   Expected: low score (wrong HTTP method). Got: {r2.faithfulness_score}")

    print("\n[4] Short answer — correct match...")
    r3 = evaluator.evaluate(short_case, "PATCH", model_name="smoke_test")
    print(f"   Score: {r3.faithfulness_score} | Path: {r3.evaluation_path}")
    print(f"   Expected: 1.0. Got: {r3.faithfulness_score}")

    print("\n[5] Technical conflict — PUT vs PATCH in long answer...")
    r4 = evaluator.evaluate(
        base_case,
        "To update a resource, you should use the PUT method.",
        model_name="smoke_test",
    )
    print(f"   Score: {r4.faithfulness_score} | Path: {r4.evaluation_path}")
    print(f"   Expected: 0.0 (PUT vs PATCH conflict). Got: {r4.faithfulness_score}")
    if r4.failure_cases:
        print(f"   Diagnosis: {r4.failure_cases[0]['diagnosis']}")

    print("\n[6] Hallucinated answer...")
    r5 = evaluator.evaluate(
        base_case,
        (
            "The OpenAPI schema includes your API paths, parameters, "
            "authentication methods, response schemas, and rate limiting configuration."
        ),
        model_name="smoke_test",
    )
    print(f"   Score: {r5.faithfulness_score} | Failure: {r5.has_failure}")

    evaluator.print_summary([r1, r2, r3, r4, r5])