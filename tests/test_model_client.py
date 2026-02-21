import sys
import os
import json
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.model_client import OllamaClient, QueryOutput

# Fixtures
@pytest.fixture
def client():
    return OllamaClient(
        model="test-model",
        temperature=0.0,
        max_tokens=512,
        timeout_seconds=30,
        base_url="http://localhost:11434",
    )


@pytest.fixture
def mock_successful_response():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "model": "test-model",
        "message": {
            "role": "assistant",
            "content": "FastAPI uses Pydantic for data validation.",
        },
        "done": True,
        "total_duration": 5000000000,
    }
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture
def mock_insufficient_response():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "model": "test-model",
        "message": {
            "role": "assistant",
            "content": "INSUFFICIENT_CONTEXT",
        },
        "done": True,
    }
    response.raise_for_status = MagicMock()
    return response

# Test: OllamaClient initialization
class TestOllamaClientInit:
    def test_default_model(self):
        client = OllamaClient(model="mistral")
        assert client.model == "mistral"

    def test_custom_temperature(self):
        client = OllamaClient(model="mistral", temperature=0.5)
        assert client.temperature == 0.5

    def test_endpoint_format(self):
        client = OllamaClient(model="mistral", base_url="http://localhost:11434")
        assert "11434" in client._endpoint
        assert "api" in client._endpoint

    def test_timeout_stored(self):
        client = OllamaClient(model="mistral", timeout_seconds=300)
        assert client.timeout_seconds == 300

    def test_base_url_stored(self):
        client = OllamaClient(model="mistral", base_url="http://localhost:11434")
        assert client.base_url == "http://localhost:11434"


# Test: health_check
class TestHealthCheck:
    def test_health_check_success(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "test-model"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = client.health_check()
        assert result is True

    def test_health_check_model_not_found(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "other-model"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = client.health_check()
        assert result is False

    def test_health_check_connection_error(self, client):
        import requests
        with patch("requests.get", side_effect=requests.ConnectionError("refused")):
            result = client.health_check()
        assert result is False

    def test_health_check_timeout(self, client):
        import requests
        with patch("requests.get", side_effect=requests.Timeout("timeout")):
            result = client.health_check()
        assert result is False

# Test: query
class TestQuery:
    def test_successful_query_returns_queryoutput(self, client, mock_successful_response):
        with patch("requests.post", return_value=mock_successful_response):
            result = client.query(
                context="FastAPI uses Pydantic for data validation.",
                question="What does FastAPI use for validation?",
                case_id="test_001",
            )
        assert isinstance(result, QueryOutput)

    def test_successful_query_answer_extracted(self, client, mock_successful_response):
        with patch("requests.post", return_value=mock_successful_response):
            result = client.query(
                context="FastAPI uses Pydantic for data validation.",
                question="What does FastAPI use for validation?",
                case_id="test_001",
            )
        assert result.answer == "FastAPI uses Pydantic for data validation."

    def test_successful_query_success_true(self, client, mock_successful_response):
        with patch("requests.post", return_value=mock_successful_response):
            result = client.query(
                context="Some context.",
                question="Some question?",
                case_id="test_002",
            )
        assert result.success is True
        assert result.error_message is None

    def test_case_id_preserved(self, client, mock_successful_response):
        with patch("requests.post", return_value=mock_successful_response):
            result = client.query(
                context="context",
                question="question",
                case_id="my_case_id",
            )
        assert result.case_id == "my_case_id"

    def test_model_name_in_result(self, client, mock_successful_response):
        with patch("requests.post", return_value=mock_successful_response):
            result = client.query("context", "question", case_id="t1")
        assert result.model == "test-model"

    def test_latency_recorded(self, client, mock_successful_response):
        with patch("requests.post", return_value=mock_successful_response):
            result = client.query("context", "question", case_id="t1")
        assert isinstance(result.latency_seconds, float)
        assert result.latency_seconds >= 0.0

    def test_context_length_recorded(self, client, mock_successful_response):
        context = "FastAPI uses Pydantic for data validation."
        with patch("requests.post", return_value=mock_successful_response):
            result = client.query(context=context, question="question", case_id="t1")
        assert result.context_length_chars == len(context)

    def test_timeout_returns_failed_result(self, client):
        import requests
        with patch("requests.post", side_effect=requests.Timeout("timed out")):
            result = client.query("context", "question", case_id="timeout_test")
        assert result.success is False
        assert result.error_message is not None
        assert "timeout" in result.error_message.lower() or "Timeout" in result.error_message

    def test_connection_error_returns_failed_result(self, client):
        import requests
        with patch("requests.post", side_effect=requests.ConnectionError("refused")):
            result = client.query("context", "question", case_id="conn_test")
        assert result.success is False
        assert result.error_message is not None

    def test_insufficient_context_answer(self, client, mock_insufficient_response):
        with patch("requests.post", return_value=mock_insufficient_response):
            result = client.query("context", "question", case_id="insuf_test")
        assert result.success is True
        assert "INSUFFICIENT_CONTEXT" in result.answer

    def test_timestamp_in_result(self, client, mock_successful_response):
        with patch("requests.post", return_value=mock_successful_response):
            result = client.query("context", "question", case_id="t1")
        assert result.timestamp_utc is not None
        assert "T" in result.timestamp_utc 

# Test: QueryOutput dataclass
class TestQueryOutput:
    def test_queryoutput_serializable(self, client, mock_successful_response):
        with patch("requests.post", return_value=mock_successful_response):
            result = client.query("context", "question", case_id="t1")
        result_dict = asdict(result)
        assert isinstance(result_dict, dict)
        assert "case_id" in result_dict
        assert "answer" in result_dict
        assert "success" in result_dict

    def test_failed_queryoutput_has_error_message(self):
        result = QueryOutput(
            case_id="test",
            model="test-model",
            question="q",
            context_length_chars=10,
            answer="",
            latency_seconds=5.0,
            timestamp_utc="2026-02-20T00:00:00+00:00",
            success=False,
            error_message="Request timeout after 300s.",
        )
        assert result.success is False
        assert result.error_message == "Request timeout after 300s."

    def test_successful_queryoutput_no_error(self):
        result = QueryOutput(
            case_id="test",
            model="test-model",
            question="q",
            context_length_chars=10,
            answer="Some answer.",
            latency_seconds=2.5,
            timestamp_utc="2026-02-20T00:00:00+00:00",
            success=True,
            error_message=None,
        )
        assert result.success is True
        assert result.error_message is None

# Test: save_query_results
class TestSaveQueryResults:
    def test_save_and_reload(self, client, mock_successful_response, tmp_path):
        from pipeline.model_client import save_query_results

        with patch("requests.post", return_value=mock_successful_response):
            result = client.query("context", "question", case_id="save_test")

        output_path = tmp_path / "query_results.json"
        save_query_results([result], str(output_path))

        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)

        assert isinstance(loaded, list)
        assert len(loaded) == 1
        assert loaded[0]["case_id"] == "save_test"
        assert "answer" in loaded[0]
        assert "success" in loaded[0]

    def test_save_multiple_results(self, client, mock_successful_response, tmp_path):
        from pipeline.model_client import save_query_results

        results = []
        for i in range(3):
            with patch("requests.post", return_value=mock_successful_response):
                r = client.query("context", "question", case_id=f"case_{i}")
            results.append(r)

        output_path = tmp_path / "multi_query.json"
        save_query_results(results, str(output_path))

        with open(output_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 3