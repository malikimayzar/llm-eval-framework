import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import asdict

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.faithfulness import (
    ClaimExtractor,
    EvidenceMatcher,
    FaithfulnessEvaluator,
    OllamaEmbedder,
    Claim,
    EvidenceMatch,
    FaithfulnessResult,
    DEFAULT_SIMILARITY_THRESHOLD,
    INSUFFICIENT_VALIDATION_THRESHOLD,
)

# Fixtures
@pytest.fixture
def sample_case():
    return {
        "id": "test_001",
        "topic": "Query Parameters",
        "context": (
            "When you declare other function parameters that are not part of the path parameters, "
            "they are automatically interpreted as query parameters. "
            "As they are part of the URL, they are naturally strings. "
            "But when you declare them with Python types, they are converted to that type and validated against it."
        ),
        "question": "What happens to query parameters declared with Python types in FastAPI?",
        "ground_truth": "They are converted to that type and validated against it.",
        "evidence_span": "when you declare them with Python types, they are converted to that type and validated against it",
    }


@pytest.fixture
def mock_embedder():
    
    embedder = MagicMock(spec=OllamaEmbedder)

    def fake_embed(text: str) -> np.ndarray:
        np.random.seed(hash(text[:50]) % (2**32))
        base = np.random.rand(768).astype(np.float32)
        return base / np.linalg.norm(base)

    def fake_embed_batch(texts: list) -> np.ndarray:
        return np.stack([fake_embed(t) for t in texts])

    embedder.embed.side_effect = fake_embed
    embedder.embed_batch.side_effect = fake_embed_batch
    embedder.health_check.return_value = True
    embedder.model = "nomic-embed-text-mock"
    return embedder

@pytest.fixture
def claim_extractor():
    return ClaimExtractor()

# Test: ClaimExtractor
class TestClaimExtractor:
    def test_extract_single_sentence(self, claim_extractor):
        answer = "FastAPI uses Pydantic for data validation."
        claims = claim_extractor.extract(answer, "test_001")
        assert len(claims) >= 1
        assert all(isinstance(c, Claim) for c in claims)

    def test_extract_multiple_sentences(self, claim_extractor):
        answer = (
            "FastAPI uses Pydantic for data validation. "
            "It also supports async operations out of the box. "
            "The framework generates OpenAPI documentation automatically."
        )
        claims = claim_extractor.extract(answer, "test_002")
        assert len(claims) >= 2

    def test_skip_short_sentences(self, claim_extractor):
        answer = "Yes. FastAPI uses Pydantic for data validation and type checking."
        claims = claim_extractor.extract(answer, "test_003")
        for claim in claims:
            assert len(claim.text.split()) >= 4

    def test_skip_filler_sentences(self, claim_extractor):
        answer = (
            "Based on the context provided, FastAPI uses Pydantic for validation. "
            "According to the document, it supports async operations."
        )
        claims = claim_extractor.extract(answer, "test_004")
        for claim in claims:
            text_lower = claim.text.lower()
            assert not text_lower.startswith("based on")
            assert not text_lower.startswith("according to")

    def test_expand_enumeration(self, claim_extractor):
        answer = "FastAPI supports async, dependency injection, and type hints."
        claims = claim_extractor.extract(answer, "test_005")
        assert len(claims) >= 2

    def test_empty_answer(self, claim_extractor):
        assert claim_extractor.extract("", "test_006") == []
        assert claim_extractor.extract("   ", "test_007") == []

    def test_claim_id_format(self, claim_extractor):
        answer = "FastAPI uses Pydantic for data validation and type checking."
        claims = claim_extractor.extract(answer, "clean_001")
        for claim in claims:
            assert claim.claim_id.startswith("clean_001_claim_")

    def test_source_answer_preserved(self, claim_extractor):
        answer = "FastAPI uses Pydantic for data validation."
        claims = claim_extractor.extract(answer, "test_008")
        for claim in claims:
            assert claim.source_answer == answer
            
# Test: EvidenceMatcher
class TestEvidenceMatcher:
    def test_supported_claim(self, mock_embedder):
        def identical_embed(text):
            np.random.seed(42)
            v = np.random.rand(768).astype(np.float32)
            return v / np.linalg.norm(v)

        mock_embedder.embed.side_effect = identical_embed
        mock_embedder.embed_batch.side_effect = lambda texts: np.stack(
            [identical_embed(t) for t in texts]
        )

        matcher = EvidenceMatcher(
            embedder=mock_embedder,
            similarity_threshold=0.75,
        )

        claims = [
            Claim(
                claim_id="test_claim_01",
                text="FastAPI converts query parameters to their declared Python type.",
                source_answer="FastAPI converts query parameters to their declared Python type.",
            )
        ]

        context = "FastAPI converts query parameters to their declared Python type and validates them."
        matches = matcher.match(claims, context)

        assert len(matches) == 1
        assert matches[0].classification == "SUPPORTED"
        assert matches[0].similarity_score >= 0.75

    def test_unsupported_claim(self, mock_embedder):
        call_count = [0]

        def divergent_embed(text):
            call_count[0] += 1
            np.random.seed(call_count[0] * 100)
            v = np.random.rand(768).astype(np.float32)
            return v / np.linalg.norm(v)

        mock_embedder.embed_batch.side_effect = lambda texts: np.stack(
            [divergent_embed(t) for t in texts]
        )

        matcher = EvidenceMatcher(
            embedder=mock_embedder,
            similarity_threshold=0.75,
        )

        claims = [
            Claim(
                claim_id="test_claim_02",
                text="FastAPI supports automatic caching of database queries.",
                source_answer="FastAPI supports automatic caching of database queries.",
            )
        ]

        context = "FastAPI uses Pydantic for data validation and supports async operations."
        matches = matcher.match(claims, context)

        assert len(matches) == 1
        assert matches[0].classification in ["SUPPORTED", "UNSUPPORTED"]
        assert isinstance(matches[0].similarity_score, float)
        assert 0.0 <= matches[0].similarity_score <= 1.0

    def test_empty_claims(self, mock_embedder):
        matcher = EvidenceMatcher(embedder=mock_embedder)
        result = matcher.match([], "some context here.")
        assert result == []

    def test_evidence_match_structure(self, mock_embedder):
        matcher = EvidenceMatcher(embedder=mock_embedder, similarity_threshold=0.75)
        claims = [Claim("c1", "FastAPI is a web framework.", "FastAPI is a web framework.")]
        matches = matcher.match(claims, "FastAPI is a modern web framework for Python.")

        assert len(matches) == 1
        match = matches[0]
        assert hasattr(match, "claim_id")
        assert hasattr(match, "claim_text")
        assert hasattr(match, "best_evidence_span")
        assert hasattr(match, "similarity_score")
        assert hasattr(match, "classification")
        assert hasattr(match, "threshold_used")
        assert match.threshold_used == 0.75



# Test: FaithfulnessEvaluator
class TestFaithfulnessEvaluator:
    def _make_evaluator(self, mock_embedder, threshold=0.75):
        evaluator = FaithfulnessEvaluator.__new__(FaithfulnessEvaluator)
        evaluator.similarity_threshold = threshold
        evaluator.embedding_model = "nomic-embed-text-mock"
        evaluator.extractor = ClaimExtractor()
        evaluator.embedder = mock_embedder

        from evaluators.faithfulness import EvidenceMatcher
        evaluator.matcher = EvidenceMatcher(
            embedder=mock_embedder,
            similarity_threshold=threshold,
        )
        return evaluator

    def test_insufficient_context_response(self, sample_case, mock_embedder):
        evaluator = self._make_evaluator(mock_embedder)

        def high_sim_embed(text):
            np.random.seed(42)  
            v = np.random.rand(768).astype(np.float32)
            return v / np.linalg.norm(v)

        mock_embedder.embed.side_effect = high_sim_embed
        mock_embedder.embed_batch.side_effect = lambda texts: np.stack(
            [high_sim_embed(t) for t in texts]
        )

        result = evaluator.evaluate(
            case=sample_case,
            model_answer="INSUFFICIENT_CONTEXT",
            model_name="test_model",
        )

        assert isinstance(result, FaithfulnessResult)
        assert result.is_insufficient_context_response is True
        assert result.has_failure is True
        assert result.faithfulness_score == 0.0

    def test_empty_answer(self, sample_case, mock_embedder):
        evaluator = self._make_evaluator(mock_embedder)
        result = evaluator.evaluate(
            case=sample_case,
            model_answer="",
            model_name="test_model",
        )
        assert result.faithfulness_score == 0.0
        assert result.has_failure is True

    def test_result_structure(self, sample_case, mock_embedder):
        evaluator = self._make_evaluator(mock_embedder)
        result = evaluator.evaluate(
            case=sample_case,
            model_answer="Query parameters are converted to their declared Python type and validated.",
            model_name="test_model",
        )
        
        assert hasattr(result, "case_id")
        assert hasattr(result, "model")
        assert hasattr(result, "faithfulness_score")
        assert hasattr(result, "has_failure")
        assert hasattr(result, "failure_cases")
        assert hasattr(result, "total_claims")
        assert hasattr(result, "supported_count")
        assert hasattr(result, "unsupported_count")
        
        assert isinstance(result.faithfulness_score, float)
        assert isinstance(result.has_failure, bool)
        assert isinstance(result.failure_cases, list)
        assert 0.0 <= result.faithfulness_score <= 1.0

        assert result.case_id == "test_001"
        assert result.model == "test_model"

    def test_score_consistency(self, sample_case, mock_embedder):
        evaluator = self._make_evaluator(mock_embedder)
        result = evaluator.evaluate(
            case=sample_case,
            model_answer="Query parameters are converted to their declared Python type and validated.",
            model_name="test_model",
        )

        if result.total_claims > 0:
            evaluable = result.total_claims - result.skipped_count
            if evaluable > 0:
                expected_score = result.supported_count / evaluable
                assert abs(result.faithfulness_score - expected_score) < 0.001

    def test_failure_case_documentation(self, sample_case, mock_embedder):
        call_count = [0]

        def divergent(text):
            call_count[0] += 1
            np.random.seed(call_count[0] * 999)
            v = np.random.rand(768).astype(np.float32)
            return v / np.linalg.norm(v)

        mock_embedder.embed_batch.side_effect = lambda texts: np.stack(
            [divergent(t) for t in texts]
        )
        mock_embedder.embed.side_effect = divergent

        evaluator = self._make_evaluator(mock_embedder)
        result = evaluator.evaluate(
            case=sample_case,
            model_answer=(
                "Query parameters support caching, rate limiting, "
                "and automatic database connection pooling."
            ),
            model_name="test_model",
        )

        if result.has_failure:
            for fc in result.failure_cases:
                assert "claim_id" in fc
                assert "unsupported_claim" in fc
                assert "similarity_score" in fc
                assert "diagnosis" in fc
                assert isinstance(fc["similarity_score"], float)
                assert isinstance(fc["diagnosis"], str)
                assert len(fc["diagnosis"]) > 0

    def test_batch_evaluate(self, sample_case, mock_embedder):
        evaluator = self._make_evaluator(mock_embedder)
        cases = [sample_case, sample_case]
        answers = [
            "Query parameters are converted to their Python type.",
            "FastAPI validates query parameters automatically.",
        ]

        results = evaluator.evaluate_batch(
            cases=cases,
            model_answers=answers,
            model_name="test_model",
        )

        assert len(results) == 2
        assert all(isinstance(r, FaithfulnessResult) for r in results)

    def test_batch_length_mismatch(self, sample_case, mock_embedder):
        evaluator = self._make_evaluator(mock_embedder)
        with pytest.raises(AssertionError):
            evaluator.evaluate_batch(
                cases=[sample_case, sample_case],
                model_answers=["only one answer"],
            )

    def test_save_results(self, sample_case, mock_embedder, tmp_path):
        import json

        evaluator = self._make_evaluator(mock_embedder)
        result = evaluator.evaluate(
            case=sample_case,
            model_answer="Query parameters are converted to their declared Python type.",
            model_name="test_model",
        )

        output_path = tmp_path / "test_results.json"
        evaluator.save_results([result], str(output_path))

        assert output_path.exists()

        with open(output_path) as f:
            loaded = json.load(f)

        assert isinstance(loaded, list)
        assert len(loaded) == 1
        assert loaded[0]["case_id"] == "test_001"
        assert "faithfulness_score" in loaded[0]

# Test: Diagnosis messages
class TestDiagnosis:
    def test_diagnosis_high_score(self, mock_embedder):
        matcher = EvidenceMatcher(embedder=mock_embedder, similarity_threshold=0.75)
        match = EvidenceMatch(
            claim_id="c1",
            claim_text="test",
            best_evidence_span="test span",
            similarity_score=0.65,
            classification="UNSUPPORTED",
            threshold_used=0.75,
        )
        evaluator = FaithfulnessEvaluator.__new__(FaithfulnessEvaluator)
        evaluator.similarity_threshold = 0.75
        diagnosis = evaluator._diagnose_failure(match)
        assert "close" in diagnosis.lower() or "below threshold" in diagnosis.lower()

    def test_diagnosis_low_score(self, mock_embedder):
        match = EvidenceMatch(
            claim_id="c1",
            claim_text="test",
            best_evidence_span="test span",
            similarity_score=0.20,
            classification="UNSUPPORTED",
            threshold_used=0.75,
        )
        evaluator = FaithfulnessEvaluator.__new__(FaithfulnessEvaluator)
        evaluator.similarity_threshold = 0.75
        diagnosis = evaluator._diagnose_failure(match)
        assert "hallucination" in diagnosis.lower() or "no relevant" in diagnosis.lower()



# Test: Constants
class TestConstants:
    def test_default_threshold_range(self):
        assert 0.5 <= DEFAULT_SIMILARITY_THRESHOLD <= 1.0

    def test_insufficient_validation_threshold_lower(self):
        assert INSUFFICIENT_VALIDATION_THRESHOLD < DEFAULT_SIMILARITY_THRESHOLD

    def test_threshold_values(self):
        assert DEFAULT_SIMILARITY_THRESHOLD == 0.75
        assert INSUFFICIENT_VALIDATION_THRESHOLD == 0.65

# Integration-style test
class TestIntegration:
    def test_full_pipeline_mock(self, sample_case, mock_embedder):
        evaluator = FaithfulnessEvaluator.__new__(FaithfulnessEvaluator)
        evaluator.similarity_threshold = 0.75
        evaluator.embedding_model = "mock"
        evaluator.extractor = ClaimExtractor()
        evaluator.embedder = mock_embedder
        
        from evaluators.faithfulness import EvidenceMatcher
        evaluator.matcher = EvidenceMatcher(
            embedder=mock_embedder,
            similarity_threshold=0.75,
        )
        
        answer = (
            "Query parameters declared with Python types are converted to that type"
            "and validated against it."
        )
        result = evaluator.evaluate(
            case=sample_case,
            model_answer=answer,
            model_name="integration_test",
        )
        
        assert result is not None
        assert result.case_id == sample_case["id"]
        assert 0.0 <= result.faithfulness_score <= 1.0
        assert result.total_claims >= 1
        assert result.supported_count + result.unsupported_count + result.skipped_count == result.total_claims
        result_dict = asdict(result)
        assert isinstance(result_dict, dict)
        assert "faithfulness_score" in result_dict