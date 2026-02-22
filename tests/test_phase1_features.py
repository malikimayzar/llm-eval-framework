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


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
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
    """Helper: buat FaithfulnessEvaluator dengan mock embedder."""
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


# ---------------------------------------------------------------------------
# 1. TestTechnicalConflict
# ---------------------------------------------------------------------------

class TestTechnicalConflict:
    def test_http_method_conflict_put_vs_patch(self):
        """PUT di claim, PATCH di evidence → conflict."""
        assert _has_technical_conflict(
            "use the PUT method to update",
            "use the PATCH method to update"
        ) is True

    def test_http_method_conflict_post_vs_put(self):
        """POST vs PUT → conflict."""
        assert _has_technical_conflict(
            "send a POST request",
            "send a PUT request"
        ) is True

    def test_http_method_conflict_get_vs_delete(self):
        """GET vs DELETE → conflict."""
        assert _has_technical_conflict(
            "use GET to retrieve the resource",
            "use DELETE to remove the resource"
        ) is True

    def test_no_conflict_same_http_method(self):
        """PATCH di keduanya → tidak conflict."""
        assert _has_technical_conflict(
            "use PATCH to partially update",
            "PATCH is used for partial updates"
        ) is False

    def test_no_conflict_claim_has_no_http_term(self):
        """Claim tidak punya HTTP term → tidak conflict meskipun evidence punya."""
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

    # ── HTTP Status Codes ─────────────────────────────────────────────────

    def test_status_code_conflict_200_vs_201(self):
        """200 vs 201 → conflict (keduanya dalam grup status codes)."""
        assert _has_technical_conflict(
            "returns status code 200 on success",
            "returns status code 201 on creation"
        ) is True

    def test_status_code_conflict_404_vs_422(self):
        """404 vs 422 → conflict."""
        assert _has_technical_conflict(
            "returns 404 when not found",
            "returns 422 for validation errors"
        ) is True

    def test_no_conflict_same_status_code(self):
        """200 di keduanya → tidak conflict."""
        assert _has_technical_conflict(
            "returns 200 OK",
            "the response is 200 on success"
        ) is False

    # ── Boolean Values ────────────────────────────────────────────────────

    def test_boolean_conflict_true_vs_false(self):
        """true vs false → conflict."""
        assert _has_technical_conflict(
            "the parameter is required true",
            "the parameter is required false"
        ) is True

    def test_no_conflict_same_boolean(self):
        """true di keduanya → tidak conflict."""
        assert _has_technical_conflict(
            "optional is set to true",
            "the flag is true by default"
        ) is False

    # ── Edge cases ────────────────────────────────────────────────────────

    def test_empty_strings(self):
        """Empty string di kedua sisi → tidak conflict."""
        assert _has_technical_conflict("", "") is False

    def test_conflict_is_case_insensitive(self):
        """PUT dan patch (mixed case) → conflict."""
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
        """
        HTTP method conflict tidak triggered oleh status code yang sama.
        Claim: PATCH + 200, Evidence: PUT + 200
        → conflict karena PATCH vs PUT, bukan karena 200 vs 200.
        """
        result = _has_technical_conflict(
            "use PATCH and expect 200",
            "use PUT and expect 200"
        )
        assert result is True  # karena PATCH vs PUT

    def test_returns_bool(self):
        """Return value harus bool, bukan truthy/falsy."""
        result = _has_technical_conflict("use PUT", "use PATCH")
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 2. TestBranchOrdering
# ---------------------------------------------------------------------------

class TestBranchOrdering:
    """
    Tests untuk memastikan urutan branch di evaluate() benar:
        Branch 1: Empty answer      → score 0.0, has_failure=True
        Branch 2: INSUFFICIENT_CTX  → insufficient path
        Branch 3: Short answer      → exact match path
        Branch 4: Normal            → claim extraction path

    Root cause dari 2 test failure sebelumnya adalah urutan yang salah.
    Tests ini memastikan regresi tidak terjadi lagi.
    """

    # ── Branch 1: Empty answer ────────────────────────────────────────────

    def test_empty_string_goes_to_empty_path(self, base_case, mock_embedder):
        """String kosong → score 0.0, bukan masuk short answer path."""
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "", "test")
        assert result.faithfulness_score == 0.0
        assert result.has_failure is True
        assert result.evaluation_path in ("empty_answer", "claim_extraction")

    def test_whitespace_only_goes_to_empty_path(self, base_case, mock_embedder):
        """Whitespace-only string → diperlakukan sama dengan empty."""
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "   ", "test")
        assert result.faithfulness_score == 0.0
        assert result.has_failure is True

    def test_empty_answer_does_not_substring_match_ground_truth(self, base_case, mock_embedder):
        """
        Bug yang sudah difix: empty string adalah substring dari apapun,
        sehingga bisa dapat score 0.9 kalau masuk short_answer path.
        Test ini memastikan bug tidak muncul lagi.
        """
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "", "test")
        assert result.faithfulness_score != 0.9
        assert result.faithfulness_score == 0.0

    # ── Branch 2: INSUFFICIENT_CONTEXT ───────────────────────────────────

    def test_insufficient_context_signal_not_treated_as_short_answer(self, base_case, mock_embedder):
        """
        'INSUFFICIENT_CONTEXT' adalah 1 kata → tanpa branch ordering yang benar,
        akan masuk short_answer path dan tidak diproses sebagai signal.
        Test ini memastikan signal selalu diproses dengan benar.
        """
        def high_sim(text):
            np.random.seed(42)
            v = np.random.rand(768).astype(np.float32)
            return v / np.linalg.norm(v)

        mock_embedder.embed.side_effect = high_sim
        mock_embedder.embed_batch.side_effect = lambda texts: np.stack([high_sim(t) for t in texts])

        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "INSUFFICIENT_CONTEXT", "test")

        # Harus masuk insufficient context path, bukan short_answer path
        assert result.is_insufficient_context_response is True
        assert result.evaluation_path != "short_answer_exact_match"

    def test_insufficient_context_in_longer_answer(self, base_case, mock_embedder):
        """Signal bisa ada di tengah kalimat yang lebih panjang."""
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

    # ── Branch 3: Short answer ────────────────────────────────────────────

    def test_short_answer_uses_exact_match_path(self, base_case, mock_embedder):
        """Jawaban singkat harus masuk short_answer_exact_match path."""
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "PATCH", "test")
        assert result.evaluation_path == "short_answer_exact_match"

    def test_short_answer_comes_after_insufficient_check(self, base_case, mock_embedder):
        """
        Short answer path hanya untuk jawaban yang BUKAN INSUFFICIENT_CONTEXT signal.
        'INSUFFICIENT_CONTEXT' (1 kata) harus tetap masuk Branch 2.
        """
        def high_sim(text):
            np.random.seed(42)
            v = np.random.rand(768).astype(np.float32)
            return v / np.linalg.norm(v)

        mock_embedder.embed.side_effect = high_sim
        mock_embedder.embed_batch.side_effect = lambda texts: np.stack([high_sim(t) for t in texts])

        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "INSUFFICIENT_CONTEXT", "test")

        # Tidak boleh masuk short answer path
        assert result.evaluation_path != "short_answer_exact_match"
        assert result.is_insufficient_context_response is True

    # ── Branch 4: Normal ──────────────────────────────────────────────────

    def test_long_answer_uses_claim_extraction_path(self, base_case, mock_embedder):
        """Jawaban panjang harus masuk claim extraction (normal) path."""
        evaluator = _make_evaluator(mock_embedder)
        long_answer = (
            "To partially update a resource in FastAPI, you should use the PATCH method. "
            "This allows you to send only the fields that need to be changed. "
            "The PUT method, on the other hand, replaces the entire resource."
        )
        result = evaluator.evaluate(base_case, long_answer, "test")
        assert result.evaluation_path not in ("short_answer_exact_match", "empty_answer")
        assert result.total_claims >= 1


# ---------------------------------------------------------------------------
# 3. TestShortAnswerPath
# ---------------------------------------------------------------------------

class TestShortAnswerPath:
    """
    Tests untuk _evaluate_short_answer() dan _is_short_answer().
    Fokus pada logika scoring: exact → substring → char_sim.
    """

    def test_is_short_answer_true_for_single_word(self):
        assert _is_short_answer("PATCH") is True

    def test_is_short_answer_true_for_four_words(self):
        assert _is_short_answer("use the PATCH method") is True

    def test_is_short_answer_false_for_long_answer(self):
        assert _is_short_answer(
            "FastAPI supports both PUT and PATCH methods for different use cases."
        ) is False

    def test_is_short_answer_boundary(self):
        """Tepat di boundary SHORT_ANSWER_WORD_THRESHOLD."""
        # Threshold adalah 5, jadi 4 kata = short (< 5), 5 kata = bukan short (>= 5)
        four_words = "use the PATCH method"
        five_words  = "use the PATCH HTTP method"
        assert len(four_words.split()) == 4
        assert len(five_words.split()) == 5
        assert _is_short_answer(four_words) is True
        assert _is_short_answer(five_words) is False

    def test_exact_match_scores_1_0(self, base_case, mock_embedder):
        """Jawaban exact sama dengan ground_truth ('PATCH') → score 1.0."""
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "PATCH", "test")
        assert result.faithfulness_score == 1.0
        assert result.has_failure is False
        assert result.evaluation_path == "short_answer_exact_match"

    def test_exact_match_case_insensitive(self, base_case, mock_embedder):
        """'patch' (lowercase) vs ground_truth 'PATCH' → exact match setelah normalize."""
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "patch", "test")
        assert result.faithfulness_score == 1.0

    def test_substring_match_scores_0_9(self, mock_embedder):
        """Answer ada di dalam ground_truth → score 0.9."""
        case = {
            "id": "test_sub",
            "context": "The endpoint returns a JSON response.",
            "question": "What does the endpoint return?",
            "ground_truth": "a JSON response object",
        }
        evaluator = _make_evaluator(mock_embedder)
        # "JSON response" adalah substring dari "a JSON response object"
        result = evaluator.evaluate(case, "JSON response", "test")
        assert result.faithfulness_score == 0.9
        assert result.evaluation_path == "short_answer_exact_match"

    def test_wrong_short_answer_has_failure(self, base_case, mock_embedder):
        """'PUT' saat ground_truth 'PATCH' → has_failure=True, score rendah."""
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "PUT", "test")
        assert result.has_failure is True
        assert result.faithfulness_score < 0.9
        assert result.evaluation_path == "short_answer_exact_match"

    def test_short_answer_result_has_correct_structure(self, base_case, mock_embedder):
        """FaithfulnessResult dari short_answer path harus punya semua field."""
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "PATCH", "test")

        assert result.case_id == "test_001"
        assert result.total_claims == 1
        assert result.is_insufficient_context_response is False
        assert isinstance(result.faithfulness_score, float)
        assert 0.0 <= result.faithfulness_score <= 1.0

    def test_short_answer_failure_has_diagnosis(self, base_case, mock_embedder):
        """Failure case dari short answer path harus punya diagnosis."""
        evaluator = _make_evaluator(mock_embedder)
        result = evaluator.evaluate(base_case, "PUT", "test")

        if result.has_failure:
            assert len(result.failure_cases) > 0
            fc = result.failure_cases[0]
            assert "diagnosis" in fc
            assert len(fc["diagnosis"]) > 0


# ---------------------------------------------------------------------------
# 4. TestConfigLoader
# ---------------------------------------------------------------------------

class TestConfigLoader:
    """
    Tests untuk memastikan config.yaml terbaca dengan benar
    dan nilai-nilainya masuk akal (sanity check).
    """

    def test_config_loads_without_error(self):
        from pipeline.config_loader import config
        assert config is not None
        assert isinstance(config, dict)

    def test_thresholds_section_exists(self):
        from pipeline.config_loader import config
        assert "thresholds" in config

    def test_faithfulness_threshold_matches_constant(self):
        """Nilai di config harus sama dengan DEFAULT_SIMILARITY_THRESHOLD."""
        from pipeline.config_loader import config
        assert config["thresholds"]["faithfulness_evidence"] == DEFAULT_SIMILARITY_THRESHOLD

    def test_short_answer_max_words_matches_constant(self):
        """Nilai di config harus sama dengan SHORT_ANSWER_WORD_THRESHOLD."""
        from pipeline.config_loader import config
        assert config["thresholds"]["short_answer_max_words"] == SHORT_ANSWER_WORD_THRESHOLD

    def test_technical_term_groups_exist_and_nonempty(self):
        from pipeline.config_loader import config
        groups = config.get("technical_term_groups", [])
        assert isinstance(groups, list)
        assert len(groups) > 0

    def test_http_methods_in_technical_groups(self):
        """HTTP methods harus ada di salah satu group."""
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


# ---------------------------------------------------------------------------
# 5. TestCheckpointHelpers
# ---------------------------------------------------------------------------

class TestCheckpointHelpers:
    """
    Tests untuk resume/checkpoint di runner.py.
    Semua filesystem ops menggunakan tmp_path supaya tidak pollute repo.
    """

    def _get_helpers(self, tmp_path):
        """Import checkpoint helpers dan patch CHECKPOINT_DIR ke tmp_path."""
        import pipeline.runner as runner_module

        # Patch CHECKPOINT_DIR ke tmp_path untuk isolasi
        original = runner_module.CHECKPOINT_DIR
        runner_module.CHECKPOINT_DIR = tmp_path
        return runner_module, original

    def test_checkpoint_path_format(self, tmp_path):
        """Checkpoint path harus include model, dataset, dan case_id."""
        import pipeline.runner as runner_module
        runner_module.CHECKPOINT_DIR = tmp_path

        path = runner_module._checkpoint_path("mistral", "clean", "clean_001")
        assert "mistral" in path.name
        assert "clean" in path.name
        assert "clean_001" in path.name
        assert path.suffix == ".json"

    def test_checkpoint_path_sanitizes_colon(self, tmp_path):
        """phi3:mini harus di-sanitize jadi phi3-mini di nama file."""
        import pipeline.runner as runner_module
        runner_module.CHECKPOINT_DIR = tmp_path

        path = runner_module._checkpoint_path("phi3:mini", "clean", "clean_001")
        assert ":" not in path.name

    def test_save_and_load_checkpoint(self, tmp_path):
        """Simpan lalu load checkpoint — data harus identik."""
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
        """Load checkpoint yang tidak ada → None, bukan exception."""
        import pipeline.runner as runner_module
        runner_module.CHECKPOINT_DIR = tmp_path

        result = runner_module._load_checkpoint("mistral", "clean", "nonexistent_case")
        assert result is None

    def test_load_corrupt_checkpoint_returns_none(self, tmp_path):
        """Load checkpoint yang corrupt (invalid JSON) → None, bukan crash."""
        import pipeline.runner as runner_module
        runner_module.CHECKPOINT_DIR = tmp_path

        # Tulis file JSON yang rusak
        corrupt_path = runner_module._checkpoint_path("mistral", "clean", "corrupt_case")
        corrupt_path.parent.mkdir(parents=True, exist_ok=True)
        corrupt_path.write_text("{ invalid json !!!")

        result = runner_module._load_checkpoint("mistral", "clean", "corrupt_case")
        assert result is None

    def test_clear_checkpoints_removes_files(self, tmp_path):
        """_clear_checkpoints harus hapus file yang ada."""
        import pipeline.runner as runner_module
        runner_module.CHECKPOINT_DIR = tmp_path

        # Simpan beberapa checkpoint
        for case_id in ["clean_001", "clean_002", "clean_003"]:
            runner_module._save_checkpoint("mistral", "clean", case_id, {"case_id": case_id})

        # Verifikasi file ada
        for case_id in ["clean_001", "clean_002", "clean_003"]:
            assert runner_module._checkpoint_path("mistral", "clean", case_id).exists()

        # Clear
        runner_module._clear_checkpoints("mistral", "clean", ["clean_001", "clean_002", "clean_003"])

        # Verifikasi file sudah terhapus
        for case_id in ["clean_001", "clean_002", "clean_003"]:
            assert not runner_module._checkpoint_path("mistral", "clean", case_id).exists()

    def test_clear_checkpoints_only_removes_specified_cases(self, tmp_path):
        """Clear hanya hapus case yang dispesifikasi, bukan semua checkpoint."""
        import pipeline.runner as runner_module
        runner_module.CHECKPOINT_DIR = tmp_path

        for case_id in ["clean_001", "clean_002", "clean_003"]:
            runner_module._save_checkpoint("mistral", "clean", case_id, {"case_id": case_id})

        # Hanya clear clean_001
        runner_module._clear_checkpoints("mistral", "clean", ["clean_001"])

        assert not runner_module._checkpoint_path("mistral", "clean", "clean_001").exists()
        assert runner_module._checkpoint_path("mistral", "clean", "clean_002").exists()
        assert runner_module._checkpoint_path("mistral", "clean", "clean_003").exists()

    def test_checkpoint_persists_correct_data_types(self, tmp_path):
        """Data types harus preserved setelah save/load (JSON round-trip)."""
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
        """Model berbeda tidak saling overwrite checkpoint."""
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