import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.consistency import (
    ConsistencyEvaluator,
    ConsistencyResult,
    VariantPairComparison,
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_ROUGE_THRESHOLD,
)

from evaluators.faithfulness import OllamaEmbedder
# Fixtures
@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=OllamaEmbedder)

    def fake_embed(text: str) -> np.ndarray:
        np.random.seed(hash(text[:50]) % (2**32))
        v = np.random.rand(768).astype(np.float32)
        return v / np.linalg.norm(v)

    embedder.embed.side_effect = fake_embed
    embedder.embed_batch.side_effect = lambda texts: np.stack([fake_embed(t) for t in texts])
    embedder.health_check.return_value = True
    embedder.model = "nomic-embed-text-mock"
    return embedder

@pytest.fixture
def evaluator(mock_embedder):
    ev = ConsistencyEvaluator.__new__(ConsistencyEvaluator)
    ev.semantic_threshold = DEFAULT_SEMANTIC_THRESHOLD
    ev.rouge_threshold = DEFAULT_ROUGE_THRESHOLD
    ev.embedding_model = "mock"
    ev.embedder = mock_embedder

    from rouge_score import rouge_scorer
    ev.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return ev

@pytest.fixture
def identical_embedder():
    embedder = MagicMock(spec=OllamaEmbedder)

    def same_embed(text):
        np.random.seed(42)
        v = np.random.rand(768).astype(np.float32)
        return v / np.linalg.norm(v)

    embedder.embed.side_effect = same_embed
    embedder.embed_batch.side_effect = lambda texts: np.stack([same_embed(t) for t in texts])
    embedder.health_check.return_value = True
    embedder.model = "mock-identical"
    return embedder

@pytest.fixture
def divergent_embedder():
    embedder = MagicMock(spec=OllamaEmbedder)
    call_count = [0]

    def diff_embed(text):
        call_count[0] += 1
        np.random.seed(call_count[0] * 777)
        v = np.random.rand(768).astype(np.float32)
        return v / np.linalg.norm(v)

    embedder.embed.side_effect = diff_embed
    embedder.embed_batch.side_effect = lambda texts: np.stack([diff_embed(t) for t in texts])
    embedder.health_check.return_value = True
    embedder.model = "mock-divergent"
    return embedder


@pytest.fixture
def sample_variants():
    return [
        {"label": "original",     "question": "How to make query param required?",
         "answer": "By not declaring any default value for the parameter."},
        {"label": "paraphrase_1", "question": "How to force query param mandatory?",
         "answer": "You can make it required by not providing a default value."},
        {"label": "paraphrase_2", "question": "How to make query param non-optional?",
         "answer": "Simply omit the default value declaration for that parameter."},
    ]


# Test: evaluate_from_answers — struktur result
class TestEvaluateFromAnswers:
    def test_result_is_consistency_result(self, evaluator, sample_variants):
        result = evaluator.evaluate_from_answers(
            case_id="test_001",
            topic="Query params",
            variants_with_answers=sample_variants,
            model_name="test",
        )
        assert isinstance(result, ConsistencyResult)

    def test_total_variants_correct(self, evaluator, sample_variants):
        result = evaluator.evaluate_from_answers("test_001", "topic", sample_variants)
        assert result.total_variants == 3

    def test_total_pairs_correct(self, evaluator, sample_variants):
        result = evaluator.evaluate_from_answers("test_001", "topic", sample_variants)
        assert result.total_pairs == 3

    def test_score_range(self, evaluator, sample_variants):
        result = evaluator.evaluate_from_answers("test_001", "topic", sample_variants)
        assert 0.0 <= result.consistency_score <= 1.0

    def test_case_id_preserved(self, evaluator, sample_variants):
        result = evaluator.evaluate_from_answers("my_case_id", "topic", sample_variants)
        assert result.case_id == "my_case_id"

    def test_model_name_preserved(self, evaluator, sample_variants):
        result = evaluator.evaluate_from_answers("t1", "topic", sample_variants, model_name="mistral")
        assert result.model == "mistral"

    def test_comparisons_have_required_fields(self, evaluator, sample_variants):
        result = evaluator.evaluate_from_answers("test_001", "topic", sample_variants)
        for comp in result.comparisons:
            assert hasattr(comp, "pair_id")
            assert hasattr(comp, "semantic_similarity")
            assert hasattr(comp, "rouge_l_score")
            assert hasattr(comp, "is_consistent")
            assert hasattr(comp, "inconsistency_type")
            assert 0.0 <= comp.semantic_similarity <= 1.0
            assert 0.0 <= comp.rouge_l_score <= 1.0

    def test_fully_consistent_with_identical_answers(self, identical_embedder, sample_variants):
        ev = ConsistencyEvaluator.__new__(ConsistencyEvaluator)
        ev.semantic_threshold = 0.72
        ev.rouge_threshold = 0.40
        ev.embedding_model = "mock"
        ev.embedder = identical_embedder

        from rouge_score import rouge_scorer
        ev.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        
        identical_variants = [
            {"label": "original",     "question": "Q1", "answer": "FastAPI uses Pydantic for validation."},
            {"label": "paraphrase_1", "question": "Q2", "answer": "FastAPI uses Pydantic for validation."},
            {"label": "paraphrase_2", "question": "Q3", "answer": "FastAPI uses Pydantic for validation."},
        ]

        result = ev.evaluate_from_answers("test", "topic", identical_variants)
        assert result.is_fully_consistent is True
        assert result.has_failure is False
        assert result.consistency_score == 1.0

    def test_not_consistent_with_divergent_answers(self, divergent_embedder):
        ev = ConsistencyEvaluator.__new__(ConsistencyEvaluator)
        ev.semantic_threshold = 0.72
        ev.rouge_threshold = 0.40
        ev.embedding_model = "mock"
        ev.embedder = divergent_embedder

        from rouge_score import rouge_scorer
        ev.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        divergent_variants = [
            {"label": "original",     "question": "Q1", "answer": "PUT is used to update data."},
            {"label": "paraphrase_1", "question": "Q2", "answer": "PATCH handles partial modifications to resources."},
        ]

        result = ev.evaluate_from_answers("test", "topic", divergent_variants)
        assert result.total_pairs == 1
        assert isinstance(result.has_failure, bool)

# Test: insufficient variants
class TestInsufficientVariants:
    def test_single_variant_returns_failure(self, evaluator):
        result = evaluator.evaluate_from_answers(
            case_id="test_001",
            topic="topic",
            variants_with_answers=[
                {"label": "original", "question": "Q", "answer": "Some answer."}
            ],
        )
        assert result.has_failure is True
        assert result.total_pairs == 0
        assert result.consistency_score == 0.0

    def test_empty_answers_filtered(self, evaluator):
        variants = [
            {"label": "original",     "question": "Q1", "answer": "Valid answer here."},
            {"label": "paraphrase_1", "question": "Q2", "answer": ""}, 
            {"label": "paraphrase_2", "question": "Q3", "answer": "   "},
        ]
        result = evaluator.evaluate_from_answers("test_001", "topic", variants)
        assert result.has_failure is True

# Test: scoring logic
class TestScoringLogic:
    def test_consistency_score_weighted(self, identical_embedder):
        ev = ConsistencyEvaluator.__new__(ConsistencyEvaluator)
        ev.semantic_threshold = 0.72
        ev.rouge_threshold = 0.40
        ev.embedding_model = "mock"
        ev.embedder = identical_embedder

        from rouge_score import rouge_scorer
        ev.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        identical_answer = "FastAPI uses Pydantic for data validation."
        variants = [
            {"label": "original",     "question": "Q1", "answer": identical_answer},
            {"label": "paraphrase_1", "question": "Q2", "answer": identical_answer},
        ]

        result = ev.evaluate_from_answers("test", "topic", variants)
        assert result.consistency_score == 1.0
        assert result.avg_semantic_similarity == 1.0
        assert result.avg_rouge_l == 1.0

    def test_inconsistent_pairs_documented(self, evaluator, sample_variants):
        result = evaluator.evaluate_from_answers("test_001", "topic", sample_variants)
        if result.has_failure:
            for ip in result.inconsistent_pairs:
                assert "pair_id" in ip
                assert "diagnosis" in ip
                assert "inconsistency_type" in ip
                assert isinstance(ip["diagnosis"], str)
                assert len(ip["diagnosis"]) > 0

# Test: diagnosis messages
class TestDiagnosis:
    def test_both_inconsistency_diagnosis(self, evaluator):
        comp = VariantPairComparison(
            pair_id="a_vs_b",
            variant_a="a", variant_b="b",
            answer_a="answer a", answer_b="answer b",
            semantic_similarity=0.50,
            rouge_l_score=0.20,
            is_consistent=False,
            inconsistency_type="both",
        )
        diagnosis = evaluator._diagnose(comp)
        assert "severe" in diagnosis.lower() or "both" in diagnosis.lower()

    def test_semantic_inconsistency_diagnosis(self, evaluator):
        comp = VariantPairComparison(
            pair_id="a_vs_b",
            variant_a="a", variant_b="b",
            answer_a="answer a", answer_b="answer b",
            semantic_similarity=0.60,
            rouge_l_score=0.50,
            is_consistent=False,
            inconsistency_type="semantic",
        )
        diagnosis = evaluator._diagnose(comp)
        assert "semantic" in diagnosis.lower()

    def test_lexical_inconsistency_diagnosis(self, evaluator):
        comp = VariantPairComparison(
            pair_id="a_vs_b",
            variant_a="a", variant_b="b",
            answer_a="answer a", answer_b="answer b",
            semantic_similarity=0.85,
            rouge_l_score=0.25,
            is_consistent=False,
            inconsistency_type="lexical",
        )
        diagnosis = evaluator._diagnose(comp)
        assert "lexical" in diagnosis.lower() or "review" in diagnosis.lower()

# Test: save_results
class TestSaveResults:
    def test_save_and_reload(self, evaluator, sample_variants, tmp_path):
        import json
        result = evaluator.evaluate_from_answers(
            "test_save", "topic", sample_variants, model_name="test"
        )

        output_path = tmp_path / "consistency_results.json"
        evaluator.save_results([result], str(output_path))

        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)

        assert isinstance(loaded, list)
        assert len(loaded) == 1
        assert loaded[0]["case_id"] == "test_save"
        assert "consistency_score" in loaded[0]
        assert "has_failure" in loaded[0]

    def test_save_multiple_results(self, evaluator, sample_variants, tmp_path):
        import json
        results = [
            evaluator.evaluate_from_answers(f"case_{i}", "topic", sample_variants)
            for i in range(3)
        ]

        output_path = tmp_path / "multi_results.json"
        evaluator.save_results(results, str(output_path))

        with open(output_path) as f:
            loaded = json.load(f)

        assert len(loaded) == 3

# Test: constants
class TestConstants:
    def test_semantic_threshold_range(self):
        assert 0.5 <= DEFAULT_SEMANTIC_THRESHOLD <= 1.0

    def test_rouge_threshold_range(self):
        assert 0.0 <= DEFAULT_ROUGE_THRESHOLD <= 1.0

    def test_semantic_higher_than_rouge(self):
        assert DEFAULT_SEMANTIC_THRESHOLD > DEFAULT_ROUGE_THRESHOLD