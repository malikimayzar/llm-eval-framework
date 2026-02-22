import sys
import os
import json
import pytest
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.faithfulness import (
    ClaimExtractor,
    EvidenceMatcher,
    FaithfulnessEvaluator,
    OllamaEmbedder,
    _has_technical_conflict,
    _is_short_answer,
    DEFAULT_SIMILARITY_THRESHOLD,
    SHORT_ANSWER_WORD_THRESHOLD,
)

# Shared fixtures
@pytest.fixture
def base_case():
    return {
        "id": "test_001",
        "context": (
            "To update a resource partially, use the PATCH method. "
            "To replace a resource entirely, use the PUT method. "
            "FastAPI supports both HTTP methods for different use cases."
        ),
        "question": "Which HTTP method should you use to partially update a resource?",
        "ground_truth": "PATCH",
    }

@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=OllamaEmbedder)

    def fake_embed(text: str) -> np.ndarray:
        np.random.seed(hash(text[:50]) % (2**32))
        v = np.random.rand(768).astype(np.float32)
        return v / np.linalg.norm(v)

    def fake_embed_batch(texts: list) -> np.ndarray:
        return np.stack([fake_embed(t) for t in texts])

    embedder.embed.side_effect = fake_embed
    embedder.embed_batch.side_effect = fake_embed_batch
    embedder.health_check.return_value = True
    embedder.model = "nomic-embed-text-mock"
    return embedder

def _make_evaluator(mock_embedder, threshold=0.75):
    evaluator = FaithfulnessEvaluator.__new__(FaithfulnessEvaluator)
    evaluator.similarity_threshold = threshold
    evaluator.embedding_model = "nomic-embed-text-mock"
    evaluator.extractor = ClaimExtractor()
    evaluator.embedder = mock_embedder
    evaluator.matcher = EvidenceMatcher(
        embedder=mock_embedder,
        similarity_threshold=threshold,
    )
    return evaluator


# TestTechnicalConflict
class TestTechnicalConflict:
    def test_http_method_conflict_put_vs_patch(self):
        assert _has_technical_conflict(
            "use the PUT method to update",
            "use the PATCH method to update"
        ) is True

    def test_http_method_conflict_post_vs_put(self):
        assert _has_technical_conflict(
            "send a POST request",
            "send a PUT request"
        ) is True

    def test_http_method_conflict_get_vs_delete(self):
        assert _has_technical_conflict(
            "use GET to retrieve the resource",
            "use DELETE to remove the resource"
        ) is True

    def test_no_conflict_same_http_method(self):
        assert _has_technical_conflict(
            "use PATCH to partially update",
            "PATCH is used for partial updates"
        ) is False

    def test_no_conflict_claim_has_no_http_term(self):
        assert _has_technical_conflict(
            "send a request to the endpoint",
            "use POST to create a resource"
        ) is False

    def test_no_conflict_evidence_has_no_http_term(self):
        """Evidence tidak punya HTTP term → tidak conflict."""
        assert _has_technical_conflict(
            "use PUT to update",
            "the endpoint accepts requests"
        ) is False

    def test_status_code_conflict_200_vs_201(self):
        assert _has_technical_conflict(
            "returns status code 200 on success",
            "returns status code 201 on creation"
        ) is True

    def test_status_code_conflict_404_vs_422(self):
        assert _has_technical_conflict(
            "returns 404 when not found",
            "returns 422 for validation errors"
        ) is True

    def test_no_conflict_same_status_code(self):
        assert _has_technical_conflict(
            "returns 200 OK",
            "the response is 200 on success"
        ) is False

    def test_boolean_conflict_true_vs_false(self):
        assert _has_technical_conflict(
            "the parameter is required true",
            "the parameter is required false"
        ) is True

    def test_no_conflict_same_boolean(self):
        assert _has_technical_conflict(
            "optional is set to true",
            "the flag is true by default"
        ) is False
        
    def test_empty_strings(self):
        assert _has_technical_conflict("", "") is False

    def test_conflict_is_case_insensitive(self):
        assert _has_technical_conflict(
            "use PUT to update",
            "use patch for partial update"
        ) is True

    def test_no_false_positive_on_partial_word(self):
        r"""
        'putative' mengandung 'put' tapi bukan sebagai word boundary.
        _has_technical_conflict menggunakan re.findall(r'\b\w+\b') jadi
        hanya match whole words — 'putative' tidak match 'put'.
        """
        assert _has_technical_conflict(
            "the putative answer is correct",
            "use patch to update"
        ) is False

    def test_multiple_term_groups_no_cross_contamination(self):
        result = _has_technical_conflict(
            "use PATCH and expect 200",
            "use PUT and expect 200"
        )
        assert result is True 

    def test_returns_bool(self):
        result = _has_technical_conflict("use PUT", "use PATCH")
        assert isinstance(result, bool)
        
# TestBranchOrdering
class TestBranchOrdering:
    def test_empty_string_goes_to_empty_path(self, base_case, mock_embedder):
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "", "test")
        assert result.faithfulness_score == 0.0
        assert result.has_failure is True
        assert result.evaluation_path in ("empty_answer", "claim_extraction")

    def test_whitespace_only_goes_to_empty_path(self, base_case, mock_embedder):
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "   ", "test")
        assert result.faithfulness_score == 0.0
        assert result.has_failure is True

    def test_empty_answer_does_not_substring_match_ground_truth(self, base_case, mock_embedder):
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "", "test")
        assert result.faithfulness_score != 0.9
        assert result.faithfulness_score == 0.0
        
    def test_insufficient_context_signal_not_treated_as_short_answer(self, base_case, mock_embedder):
        def high_sim(text):
            np.random.seed(42)
            v = np.random.rand(768).astype(np.float32)
            return v / np.linalg.norm(v)

        mock_embedder.embed.side_effect = high_sim
        mock_embedder.embed_batch.side_effect = lambda texts: np.stack([high_sim(t) for t in texts])

        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "INSUFFICIENT_CONTEXT", "test")
        
        assert result.is_insufficient_context_response is True
        assert result.evaluation_path != "short_answer_exact_match"

    def test_insufficient_context_in_longer_answer(self, base_case, mock_embedder):
        def low_sim(text):
            np.random.seed(hash(text[:20]) % (2**32))
            v = np.random.rand(768).astype(np.float32)
            return v / np.linalg.norm(v)

        mock_embedder.embed.side_effect = low_sim
        mock_embedder.embed_batch.side_effect = lambda texts: np.stack([low_sim(t) for t in texts])

        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(
            base_case,
            "Based on the context, INSUFFICIENT_CONTEXT to answer this.",
            "test"
        )
        assert result.is_insufficient_context_response is True
        
    def test_short_answer_uses_exact_match_path(self, base_case, mock_embedder):
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "PATCH", "test")
        assert result.evaluation_path == "short_answer_exact_match"

    def test_short_answer_comes_after_insufficient_check(self, base_case, mock_embedder):
        def high_sim(text):
            np.random.seed(42)
            v = np.random.rand(768).astype(np.float32)
            return v / np.linalg.norm(v)

        mock_embedder.embed.side_effect = high_sim
        mock_embedder.embed_batch.side_effect = lambda texts: np.stack([high_sim(t) for t in texts])

        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "INSUFFICIENT_CONTEXT", "test")

        assert result.evaluation_path != "short_answer_exact_match"
        assert result.is_insufficient_context_response is True

    def test_long_answer_uses_claim_extraction_path(self, base_case, mock_embedder):
        evaluator = _make_evaluator(mock_embedder)
        long_answer = (
            "To partially update a resource in FastAPI, you should use the PATCH method. "
            "This allows you to send only the fields that need to be changed. "
            "The PUT method, on the other hand, replaces the entire resource."
        )
        result = evaluator.evaluate(base_case, long_answer, "test")
        assert result.evaluation_path not in ("short_answer_exact_match", "empty_answer")
        assert result.total_claims >= 1


# TestShortAnswerPath
class TestShortAnswerPath:
    def test_is_short_answer_true_for_single_word(self):
        assert _is_short_answer("PATCH") is True

    def test_is_short_answer_true_for_four_words(self):
        assert _is_short_answer("use the PATCH method") is True

    def test_is_short_answer_false_for_long_answer(self):
        assert _is_short_answer(
            "FastAPI supports both PUT and PATCH methods for different use cases."
        ) is False

    def test_is_short_answer_boundary(self):
        four_words = "use the PATCH method"
        five_words  = "use the PATCH HTTP method"
        assert len(four_words.split()) == 4
        assert len(five_words.split()) == 5
        assert _is_short_answer(four_words) is True
        assert _is_short_answer(five_words) is False

    def test_exact_match_scores_1_0(self, base_case, mock_embedder):
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "PATCH", "test")
        assert result.faithfulness_score == 1.0
        assert result.has_failure is False
        assert result.evaluation_path == "short_answer_exact_match"

    def test_exact_match_case_insensitive(self, base_case, mock_embedder):
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "patch", "test")
        assert result.faithfulness_score == 1.0

    def test_substring_match_scores_0_9(self, mock_embedder):
        case = {
            "id": "test_sub",
            "context": "The endpoint returns a JSON response.",
            "question": "What does the endpoint return?",
            "ground_truth": "a JSON response object",
        }
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(case, "JSON response", "test")
        assert result.faithfulness_score == 0.9
        assert result.evaluation_path == "short_answer_exact_match"

    def test_wrong_short_answer_has_failure(self, base_case, mock_embedder):
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "PUT", "test")
        assert result.has_failure is True
        assert result.faithfulness_score < 0.9
        assert result.evaluation_path == "short_answer_exact_match"

    def test_short_answer_result_has_correct_structure(self, base_case, mock_embedder):
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "PATCH", "test")

        assert result.case_id == "test_001"
        assert result.total_claims == 1
        assert result.is_insufficient_context_response is False
        assert isinstance(result.faithfulness_score, float)
        assert 0.0 <= result.faithfulness_score <= 1.0

    def test_short_answer_failure_has_diagnosis(self, base_case, mock_embedder):
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "PUT", "test")

        if result.has_failure:
            assert len(result.failure_cases) > 0
            fc = result.failure_cases[0]
            assert "diagnosis" in fc
            assert len(fc["diagnosis"]) > 0


# TestConfigLoader
class TestConfigLoader:
    def test_config_loads_without_error(self):
        from pipeline.config_loader import config
        assert config is not None
        assert isinstance(config, dict)

    def test_thresholds_section_exists(self):
        from pipeline.config_loader import config
        assert "thresholds" in config

    def test_faithfulness_threshold_matches_constant(self):
        from pipeline.config_loader import config
        assert config["thresholds"]["faithfulness_evidence"] == DEFAULT_SIMILARITY_THRESHOLD

    def test_short_answer_max_words_matches_constant(self):
        from pipeline.config_loader import config
        assert config["thresholds"]["short_answer_max_words"] == SHORT_ANSWER_WORD_THRESHOLD

    def test_technical_term_groups_exist_and_nonempty(self):
        from pipeline.config_loader import config
        groups = config.get("technical_term_groups", [])
        assert isinstance(groups, list)
        assert len(groups) > 0

    def test_http_methods_in_technical_groups(self):
        from pipeline.config_loader import config
        groups = config.get("technical_term_groups", [])
        all_terms = [term for group in groups for term in group]
        assert "put" in all_terms
        assert "patch" in all_terms
        assert "post" in all_terms

    def test_inference_config_exists(self):
        from pipeline.config_loader import config
        assert "inference" in config
        assert "max_retries" in config["inference"]
        assert config["inference"]["max_retries"] > 0

    def test_threshold_values_are_in_valid_range(self):
        from pipeline.config_loader import config
        thresholds = config["thresholds"]
        for key, value in thresholds.items():
            if isinstance(value, (int, float)) and "words" not in key:
                assert 0.0 < value <= 1.0, f"Threshold '{key}' = {value} out of range (0, 1]"

    def test_insufficient_threshold_lower_than_faithfulness(self):
        from pipeline.config_loader import config
        faith = config["thresholds"]["faithfulness_evidence"]
        insuf = config["thresholds"]["insufficient_ctx_validation"]
        assert insuf < faith, (
            f"insufficient_ctx_validation ({insuf}) should be < "
            f"faithfulness_evidence ({faith})"
        )


# TestCheckpointHelpers
class TestCheckpointHelpers:
    def _get_helpers(self, tmp_path):
        import pipeline.runner as runner_module
        
        original = runner_module.CHECKPOINT_DIR
        runner_module.CHECKPOINT_DIR = tmp_path
        return runner_module, original

    def test_checkpoint_path_format(self, tmp_path):
        import pipeline.runner as runner_module
        runner_module.CHECKPOINT_DIR = tmp_path

        path = runner_module._checkpoint_path("mistral", "clean", "clean_001")
        assert "mistral" in path.name
        assert "clean" in path.name
        assert "clean_001" in path.name
        assert path.suffix == ".json"

    def test_checkpoint_path_sanitizes_colon(self, tmp_path):
        import pipeline.runner as runner_module
        runner_module.CHECKPOINT_DIR = tmp_path

        path = runner_module._checkpoint_path("phi3:mini", "clean", "clean_001")
        assert ":" not in path.name

    def test_save_and_load_checkpoint(self, tmp_path):
        import pipeline.runner as runner_module
        runner_module.CHECKPOINT_DIR = tmp_path

        data = {
            "case_id": "clean_001",
            "faithfulness_result": {"faithfulness_score": 0.75, "has_failure": False},
            "query_result": {"success": True, "answer": "PATCH"},
        }

        runner_module._save_checkpoint("mistral", "clean", "clean_001", data)
        loaded = runner_module._load_checkpoint("mistral", "clean", "clean_001")

        assert loaded is not None
        assert loaded["case_id"] == "clean_001"
        assert loaded["faithfulness_result"]["faithfulness_score"] == 0.75

    def test_load_nonexistent_checkpoint_returns_none(self, tmp_path):
        import pipeline.runner as runner_module
        runner_module.CHECKPOINT_DIR = tmp_path

        result = runner_module._load_checkpoint("mistral", "clean", "nonexistent_case")
        assert result is None

    def test_load_corrupt_checkpoint_returns_none(self, tmp_path):
        import pipeline.runner as runner_module
        runner_module.CHECKPOINT_DIR = tmp_path

        corrupt_path = runner_module._checkpoint_path("mistral", "clean", "corrupt_case")
        corrupt_path.parent.mkdir(parents=True, exist_ok=True)
        corrupt_path.write_text("{ invalid json !!!")

        result = runner_module._load_checkpoint("mistral", "clean", "corrupt_case")
        assert result is None

    def test_clear_checkpoints_removes_files(self, tmp_path):
        import pipeline.runner as runner_module
        runner_module.CHECKPOINT_DIR = tmp_path
        
        for case_id in ["clean_001", "clean_002", "clean_003"]:
            runner_module._save_checkpoint("mistral", "clean", case_id, {"case_id": case_id})

        for case_id in ["clean_001", "clean_002", "clean_003"]:
            assert runner_module._checkpoint_path("mistral", "clean", case_id).exists()
            
        runner_module._clear_checkpoints("mistral", "clean", ["clean_001", "clean_002", "clean_003"])
        
        for case_id in ["clean_001", "clean_002", "clean_003"]:
            assert not runner_module._checkpoint_path("mistral", "clean", case_id).exists()

    def test_clear_checkpoints_only_removes_specified_cases(self, tmp_path):
        import pipeline.runner as runner_module
        runner_module.CHECKPOINT_DIR = tmp_path

        for case_id in ["clean_001", "clean_002", "clean_003"]:
            runner_module._save_checkpoint("mistral", "clean", case_id, {"case_id": case_id})
            
        runner_module._clear_checkpoints("mistral", "clean", ["clean_001"])

        assert not runner_module._checkpoint_path("mistral", "clean", "clean_001").exists()
        assert runner_module._checkpoint_path("mistral", "clean", "clean_002").exists()
        assert runner_module._checkpoint_path("mistral", "clean", "clean_003").exists()

    def test_checkpoint_persists_correct_data_types(self, tmp_path):
        import pipeline.runner as runner_module
        runner_module.CHECKPOINT_DIR = tmp_path

        data = {
            "case_id": "clean_001",
            "faithfulness_result": {
                "faithfulness_score": 0.6667,
                "has_failure": True,
                "total_claims": 3,
                "failure_cases": [{"claim_id": "c1", "diagnosis": "test"}],
            },
        }
        
        runner_module._save_checkpoint("mistral", "clean", "clean_001", data)
        loaded = runner_module._load_checkpoint("mistral", "clean", "clean_001")

        fr = loaded["faithfulness_result"]
        assert isinstance(fr["faithfulness_score"], float)
        assert isinstance(fr["has_failure"], bool)
        assert isinstance(fr["total_claims"], int)
        assert isinstance(fr["failure_cases"], list)

    def test_multiple_models_have_separate_checkpoints(self, tmp_path):
        import pipeline.runner as runner_module
        runner_module.CHECKPOINT_DIR = tmp_path

        runner_module._save_checkpoint("mistral", "clean", "clean_001",
                                       {"model": "mistral", "score": 0.8})
        runner_module._save_checkpoint("phi3:mini", "clean", "clean_001",
                                       {"model": "phi3:mini", "score": 0.6})

        mistral_cp = runner_module._load_checkpoint("mistral", "clean", "clean_001")
        phi3_cp    = runner_module._load_checkpoint("phi3:mini", "clean", "clean_001")

        assert mistral_cp["model"] == "mistral"
        assert phi3_cp["model"] == "phi3:mini"
        assert mistral_cp["score"] != phi3_cp["score"]