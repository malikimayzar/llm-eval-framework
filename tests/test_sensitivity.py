import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.sensitivity import (
    SensitivityEvaluator,
    SensitivityResult,
    perturb_typo,
    perturb_lowercase,
    perturb_context_noise,
    perturb_question_reorder,
    PERTURBATION_TYPO,
    PERTURBATION_LOWERCASE,
    PERTURBATION_NOISE,
    PERTURBATION_REORDER,
    DEFAULT_ROBUSTNESS_THRESHOLD,
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
    embedder.model = "mock"
    return embedder


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
        np.random.seed(call_count[0] * 999)
        v = np.random.rand(768).astype(np.float32)
        return v / np.linalg.norm(v)

    embedder.embed.side_effect = diff_embed
    embedder.embed_batch.side_effect = lambda texts: np.stack([diff_embed(t) for t in texts])
    embedder.health_check.return_value = True
    embedder.model = "mock-divergent"
    return embedder

@pytest.fixture
def evaluator(mock_embedder):
    ev = SensitivityEvaluator.__new__(SensitivityEvaluator)
    ev.robustness_threshold = DEFAULT_ROBUSTNESS_THRESHOLD
    ev.embedding_model = "mock"
    ev.embedder = mock_embedder

    from rouge_score import rouge_scorer
    ev.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    import random
    random.seed(42)
    return ev


@pytest.fixture
def sample_question():
    return "What does FastAPI use for data validation?"

@pytest.fixture
def sample_context():
    return "FastAPI uses Pydantic for data validation and supports async operations."

@pytest.fixture
def sample_perturbed_variants(sample_question, sample_context):
    return [
        {
            "perturbation_type": PERTURBATION_TYPO,
            "perturbed_question": perturb_typo(sample_question),
            "perturbed_context": sample_context,
            "answer": "FastAPI uses Pydantic for data validation.",
        },
        {
            "perturbation_type": PERTURBATION_LOWERCASE,
            "perturbed_question": perturb_lowercase(sample_question),
            "perturbed_context": sample_context,
            "answer": "FastAPI utilizes Pydantic for validating data.",
        },
    ]

# Test: Perturbation functions
class TestPerturbationFunctions:
    def test_typo_changes_one_character(self, sample_question):
        perturbed = perturb_typo(sample_question)
        assert len(perturbed) == len(sample_question)
        diffs = sum(1 for a, b in zip(sample_question, perturbed) if a != b)
        assert diffs == 1

    def test_typo_preserves_length(self, sample_question):
        perturbed = perturb_typo(sample_question)
        assert len(perturbed) == len(sample_question)

    def test_lowercase_converts_all(self, sample_question):
        perturbed = perturb_lowercase(sample_question)
        assert perturbed == sample_question.lower()
        assert perturbed.islower() or not any(c.isalpha() for c in perturbed)

    def test_lowercase_preserves_content(self, sample_question):
        perturbed = perturb_lowercase(sample_question)
        assert perturbed.replace(" ", "") == sample_question.lower().replace(" ", "")

    def test_context_noise_adds_sentence(self, sample_context):
        perturbed = perturb_context_noise(sample_context)
        assert len(perturbed) > len(sample_context)
        assert sample_context in perturbed

    def test_context_noise_original_preserved(self, sample_context):
        perturbed = perturb_context_noise(sample_context)
        assert perturbed.startswith(sample_context.strip())

    def test_question_reorder_changes_question(self, sample_question):
        perturbed = perturb_question_reorder(sample_question)
        assert perturbed != sample_question

    def test_question_reorder_preserves_core(self, sample_question):
        perturbed = perturb_question_reorder(sample_question)
        assert "FastAPI" in perturbed or "fastapi" in perturbed.lower()

    def test_typo_short_question(self):
        result = perturb_typo("Hi?")
        assert isinstance(result, str)

    def test_reorder_fallback(self):
        q = "Does FastAPI support async operations?"
        result = perturb_question_reorder(q)
        assert isinstance(result, str)
        assert len(result) > 0

# Test: apply_perturbation
class TestApplyPerturbation:
    def test_typo_affects_question(self, evaluator, sample_question, sample_context):
        pq, pc = evaluator.apply_perturbation(sample_question, sample_context, PERTURBATION_TYPO)
        assert pq != sample_question
        assert pc == sample_context

    def test_lowercase_affects_question(self, evaluator, sample_question, sample_context):
        pq, pc = evaluator.apply_perturbation(sample_question, sample_context, PERTURBATION_LOWERCASE)
        assert pq == sample_question.lower()
        assert pc == sample_context

    def test_noise_affects_context(self, evaluator, sample_question, sample_context):
        pq, pc = evaluator.apply_perturbation(sample_question, sample_context, PERTURBATION_NOISE)
        assert pq == sample_question
        assert pc != sample_context
        assert len(pc) > len(sample_context)

    def test_reorder_affects_question(self, evaluator, sample_question, sample_context):
        pq, pc = evaluator.apply_perturbation(sample_question, sample_context, PERTURBATION_REORDER)
        assert pq != sample_question
        assert pc == sample_context

    def test_unknown_perturbation_returns_original(self, evaluator, sample_question, sample_context):
        pq, pc = evaluator.apply_perturbation(sample_question, sample_context, "unknown_type")
        assert pq == sample_question
        assert pc == sample_context

# Test: evaluate_from_answers
class TestEvaluateFromAnswers:
    def test_returns_sensitivity_result(self, evaluator, sample_question,
                                        sample_context, sample_perturbed_variants):
        result = evaluator.evaluate_from_answers(
            case_id="test_001",
            original_question=sample_question,
            original_context=sample_context,
            original_answer="FastAPI uses Pydantic for data validation.",
            perturbed_variants=sample_perturbed_variants,
            model_name="test",
        )
        assert isinstance(result, SensitivityResult)

    def test_case_id_preserved(self, evaluator, sample_question,
                                sample_context, sample_perturbed_variants):
        result = evaluator.evaluate_from_answers(
            case_id="my_case",
            original_question=sample_question,
            original_context=sample_context,
            original_answer="FastAPI uses Pydantic.",
            perturbed_variants=sample_perturbed_variants,
        )
        assert result.case_id == "my_case"

    def test_total_perturbations_correct(self, evaluator, sample_question,
                                          sample_context, sample_perturbed_variants):
        result = evaluator.evaluate_from_answers(
            case_id="test",
            original_question=sample_question,
            original_context=sample_context,
            original_answer="FastAPI uses Pydantic.",
            perturbed_variants=sample_perturbed_variants,
        )
        assert result.total_perturbations == len(sample_perturbed_variants)

    def test_robustness_score_range(self, evaluator, sample_question,
                                     sample_context, sample_perturbed_variants):
        result = evaluator.evaluate_from_answers(
            case_id="test",
            original_question=sample_question,
            original_context=sample_context,
            original_answer="FastAPI uses Pydantic.",
            perturbed_variants=sample_perturbed_variants,
        )
        assert 0.0 <= result.robustness_score <= 1.0

    def test_fully_robust_with_identical_answers(self, identical_embedder,
                                                   sample_question, sample_context):
        ev = SensitivityEvaluator.__new__(SensitivityEvaluator)
        ev.robustness_threshold = 0.85
        ev.embedding_model = "mock"
        ev.embedder = identical_embedder
        from rouge_score import rouge_scorer
        ev.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        identical_answer = "FastAPI uses Pydantic for data validation."
        variants = [
            {"perturbation_type": PERTURBATION_TYPO,
             "perturbed_question": perturb_typo(sample_question),
             "perturbed_context": sample_context,
             "answer": identical_answer},
            {"perturbation_type": PERTURBATION_LOWERCASE,
             "perturbed_question": perturb_lowercase(sample_question),
             "perturbed_context": sample_context,
             "answer": identical_answer},
        ]

        result = ev.evaluate_from_answers(
            case_id="test", original_question=sample_question,
            original_context=sample_context, original_answer=identical_answer,
            perturbed_variants=variants,
        )
        assert result.is_fully_robust is True
        assert result.has_failure is False
        assert result.robustness_score == 1.0

    def test_sensitive_with_divergent_answers(self, divergent_embedder,
                                               sample_question, sample_context):
        ev = SensitivityEvaluator.__new__(SensitivityEvaluator)
        ev.robustness_threshold = 0.85
        ev.embedding_model = "mock"
        ev.embedder = divergent_embedder
        from rouge_score import rouge_scorer
        ev.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        variants = [
            {"perturbation_type": PERTURBATION_TYPO,
             "perturbed_question": perturb_typo(sample_question),
             "perturbed_context": sample_context,
             "answer": "I cannot determine the answer from the context."},
        ]

        result = ev.evaluate_from_answers(
            case_id="test", original_question=sample_question,
            original_context=sample_context,
            original_answer="FastAPI uses Pydantic for data validation.",
            perturbed_variants=variants,
        )
        assert result.has_failure is True
        assert len(result.sensitive_perturbations) >= 1

    def test_empty_perturbed_answer_filtered(self, evaluator, sample_question, sample_context):
        variants = [
            {"perturbation_type": PERTURBATION_TYPO,
             "perturbed_question": perturb_typo(sample_question),
             "perturbed_context": sample_context,
             "answer": ""}, 
            {"perturbation_type": PERTURBATION_LOWERCASE,
             "perturbed_question": perturb_lowercase(sample_question),
             "perturbed_context": sample_context,
             "answer": "   "},  
        ]
        result = evaluator.evaluate_from_answers(
            case_id="test", original_question=sample_question,
            original_context=sample_context,
            original_answer="FastAPI uses Pydantic.",
            perturbed_variants=variants,
        )
        assert result.has_failure is True
        assert result.total_perturbations == 0

    def test_comparisons_structure(self, evaluator, sample_question,
                                    sample_context, sample_perturbed_variants):
        result = evaluator.evaluate_from_answers(
            case_id="test", original_question=sample_question,
            original_context=sample_context,
            original_answer="FastAPI uses Pydantic.",
            perturbed_variants=sample_perturbed_variants,
        )
        for comp in result.comparisons:
            assert hasattr(comp, "pair_id")
            assert hasattr(comp, "perturbation_type")
            assert hasattr(comp, "semantic_similarity")
            assert hasattr(comp, "rouge_l_score")
            assert hasattr(comp, "is_robust")
            assert hasattr(comp, "diagnosis")
            assert 0.0 <= comp.semantic_similarity <= 1.0
            assert 0.0 <= comp.rouge_l_score <= 1.0
            assert isinstance(comp.diagnosis, str)
            assert len(comp.diagnosis) > 0

    def test_sensitive_perturbations_documented(self, divergent_embedder,
                                                  sample_question, sample_context):
        ev = SensitivityEvaluator.__new__(SensitivityEvaluator)
        ev.robustness_threshold = 0.85
        ev.embedding_model = "mock"
        ev.embedder = divergent_embedder
        from rouge_score import rouge_scorer
        ev.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        variants = [
            {"perturbation_type": PERTURBATION_TYPO,
             "perturbed_question": perturb_typo(sample_question),
             "perturbed_context": sample_context,
             "answer": "Different answer entirely."},
        ]
        result = ev.evaluate_from_answers(
            case_id="test", original_question=sample_question,
            original_context=sample_context,
            original_answer="FastAPI uses Pydantic for data validation.",
            perturbed_variants=variants,
        )
        if result.has_failure:
            for sp in result.sensitive_perturbations:
                assert "perturbation_type" in sp
                assert "semantic_similarity" in sp
                assert "diagnosis" in sp

# Test: diagnosis messages
class TestDiagnosis:
    def test_robust_diagnosis(self, evaluator):
        diagnosis = evaluator._diagnose(PERTURBATION_TYPO, 0.95, 0.80, True)
        assert "robust" in diagnosis.lower()

    def test_highly_sensitive_diagnosis(self, evaluator):
        diagnosis = evaluator._diagnose(PERTURBATION_TYPO, 0.40, 0.10, False)
        assert "highly sensitive" in diagnosis.lower() or "signifikan" in diagnosis.lower()

    def test_sensitive_diagnosis(self, evaluator):
        diagnosis = evaluator._diagnose(PERTURBATION_LOWERCASE, 0.78, 0.50, False)
        assert "sensitive" in diagnosis.lower()

    def test_diagnosis_mentions_perturbation_type(self, evaluator):
        for ptype in [PERTURBATION_TYPO, PERTURBATION_LOWERCASE,
                      PERTURBATION_NOISE, PERTURBATION_REORDER]:
            diagnosis = evaluator._diagnose(ptype, 0.70, 0.30, False)
            assert isinstance(diagnosis, str)
            assert len(diagnosis) > 10

# Test: save_results
class TestSaveResults:
    def test_save_and_reload(self, evaluator, sample_question, sample_context,
                              sample_perturbed_variants, tmp_path):
        import json
        result = evaluator.evaluate_from_answers(
            case_id="save_test", original_question=sample_question,
            original_context=sample_context,
            original_answer="FastAPI uses Pydantic.",
            perturbed_variants=sample_perturbed_variants,
            model_name="test",
        )
        output_path = tmp_path / "sensitivity_results.json"
        evaluator.save_results([result], str(output_path))

        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)

        assert isinstance(loaded, list)
        assert len(loaded) == 1
        assert loaded[0]["case_id"] == "save_test"
        assert "robustness_score" in loaded[0]
        assert "has_failure" in loaded[0]
        assert "sensitive_perturbations" in loaded[0]

# Test: constants
class TestConstants:
    def test_robustness_threshold_range(self):
        assert 0.5 <= DEFAULT_ROBUSTNESS_THRESHOLD <= 1.0

    def test_robustness_threshold_higher_than_consistency(self):
        from evaluators.consistency import DEFAULT_SEMANTIC_THRESHOLD
        assert DEFAULT_ROBUSTNESS_THRESHOLD >= DEFAULT_SEMANTIC_THRESHOLD

    def test_perturbation_types_defined(self):
        assert PERTURBATION_TYPO == "typo"
        assert PERTURBATION_LOWERCASE == "lowercase"
        assert PERTURBATION_NOISE == "context_noise"
        assert PERTURBATION_REORDER == "question_reorder"

    def test_evaluator_has_all_perturbations(self):
        assert PERTURBATION_TYPO in SensitivityEvaluator.PERTURBATIONS
        assert PERTURBATION_LOWERCASE in SensitivityEvaluator.PERTURBATIONS
        assert PERTURBATION_NOISE in SensitivityEvaluator.PERTURBATIONS
        assert PERTURBATION_REORDER in SensitivityEvaluator.PERTURBATIONS