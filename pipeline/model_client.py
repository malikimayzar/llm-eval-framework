import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime, timezone

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("model_client")


# data classes
@dataclass
class QueryInput:
    context: str          
    question: str          
    case_id: str           
    model: str             
    temperature: float     
    max_tokens: int       


@dataclass
class QueryOutput:
    case_id: str
    model: str
    question: str
    context_length_chars: int
    answer: str
    latency_seconds: float
    timestamp_utc: str
    success: bool
    error_message: Optional[str] = None
    raw_response: Optional[dict] = field(default=None, repr=False)


# Prompt template

STRICT_QA_PROMPT = """You are a document-grounded Question Answering system.

RULES (non-negotiable):
1. Answer ONLY based on the context provided below.
2. If the answer is not present in the context, respond exactly with: "INSUFFICIENT_CONTEXT"
3. Do not use prior knowledge, training data, or assumptions.
4. Be precise. Do not add information that is not explicitly stated in the context.
5. Keep your answer concise and direct.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


# client
class OllamaClient:
    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_seconds: int = 900,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self._endpoint = f"{self.base_url}/api/chat"

        logger.info(
            f"OllamaClient initialized | model={self.model} | "
            f"temperature={self.temperature} | endpoint={self._endpoint}"
        )

    def health_check(self) -> bool:
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )
            response.raise_for_status()
            available_models = [m["name"] for m in response.json().get("models", [])]
            
            model_available = any(
                self.model in m for m in available_models
            )

            if model_available:
                logger.info(f"Health check OK | model '{self.model}' available")
            else:
                logger.warning(
                    f"Health check WARNING | model '{self.model}' not found. "
                    f"Available: {available_models}. "
                    f"Run: ollama pull {self.model}"
                )

            return model_available

        except requests.exceptions.ConnectionError:
            logger.error(
                "Health check FAILED | Cannot connect to Ollama. "
                "Make sure Ollama is running: ollama serve"
            )
            return False
        except Exception as e:
            logger.error(f"Health check FAILED | Unexpected error: {e}")
            return False

    def query(
        self,
        context: str,
        question: str,
        case_id: str,
    ) -> QueryOutput:
        prompt = STRICT_QA_PROMPT.format(
            context=context.strip(),
            question=question.strip(),
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        logger.info(f"Querying | case_id={case_id} | model={self.model}")
        start_time = time.time()

        try:
            response = requests.post(
                self._endpoint,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()

            latency = round(time.time() - start_time, 3)
            raw = response.json()
            
            answer = raw.get("message", {}).get("content", "").strip()

            logger.info(
                f"Query OK | case_id={case_id} | "
                f"latency={latency}s | answer_length={len(answer)} chars"
            )

            return QueryOutput(
                case_id=case_id,
                model=self.model,
                question=question,
                context_length_chars=len(context),
                answer=answer,
                latency_seconds=latency,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                success=True,
                raw_response=raw,
            )

        except requests.exceptions.Timeout:
            latency = round(time.time() - start_time, 3)
            error_msg = (
                f"Request timeout after {self.timeout_seconds}s. "
                f"CPU inference is slow — consider increasing timeout_seconds."
            )
            logger.error(f"Query TIMEOUT | case_id={case_id} | {error_msg}")
            return QueryOutput(
                case_id=case_id,
                model=self.model,
                question=question,
                context_length_chars=len(context),
                answer="",
                latency_seconds=latency,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                success=False,
                error_message=error_msg,
            )

        except requests.exceptions.ConnectionError:
            error_msg = (
                "Cannot connect to Ollama server. "
                "Run 'ollama serve' first."
            )
            logger.error(f"Query FAILED | case_id={case_id} | {error_msg}")
            return QueryOutput(
                case_id=case_id,
                model=self.model,
                question=question,
                context_length_chars=len(context),
                answer="",
                latency_seconds=0.0,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                success=False,
                error_message=error_msg,
            )

        except Exception as e:
            latency = round(time.time() - start_time, 3)
            error_msg = f"Unexpected error: {type(e).__name__}: {e}"
            logger.error(f"Query FAILED | case_id={case_id} | {error_msg}")
            return QueryOutput(
                case_id=case_id,
                model=self.model,
                question=question,
                context_length_chars=len(context),
                answer="",
                latency_seconds=latency,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                success=False,
                error_message=error_msg,
            )

    def batch_query(
        self,
        cases: list[dict],
        delay_between_calls: float = 1.0,
    ) -> list[QueryOutput]:
        results = []
        total = len(cases)

        logger.info(f"Starting batch query | total={total} cases | model={self.model}")

        for i, case in enumerate(cases, 1):
            logger.info(f"Progress: {i}/{total} | case_id={case['id']}")

            output = self.query(
                context=case["context"],
                question=case["question"],
                case_id=case["id"],
            )
            results.append(output)

            if i < total:
                time.sleep(delay_between_calls)

        success_count = sum(1 for r in results if r.success)
        logger.info(
            f"Batch complete | success={success_count}/{total} | "
            f"failed={total - success_count}/{total}"
        )

        return results

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
        }


# output helper
def save_query_results(
    results: list[QueryOutput],
    output_path: str,
) -> None:
    serializable = []
    for r in results:
        d = asdict(r)
        d.pop("raw_response", None)  
        serializable.append(d)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved | path={output_path} | count={len(results)}")

# quick smoke test 
if __name__ == "__main__":
    import sys
    DEV_MODEL  = "phi3:mini"
    EVAL_MODEL = "mistral"

    print("=" * 60)
    print("OllamaClient — Smoke Test")
    print(f"Dev model  : {DEV_MODEL}")
    print(f"Eval model : {EVAL_MODEL}")
    print("=" * 60)

    client = OllamaClient(model=DEV_MODEL, temperature=0.0)

    print("\n[1] Health check...")
    is_healthy = client.health_check()

    if not is_healthy:
        print(f"\n Model '{DEV_MODEL}' tidak tersedia.")
        print(f"   Jalankan: ollama pull {DEV_MODEL}")
        sys.exit(1)

    print("Ollama siap.\n")

    print("[2] Single query test (dev model)...")
    result = client.query(
        context="FastAPI schema includes API paths and parameters.",
        question="What does FastAPI schema include?",
        case_id="smoke_test_001",
    )

    if result.success:
        print(f"\n Query berhasil")
        print(f"   Model   : {result.model}")
        print(f"   Latency : {result.latency_seconds}s")
        print(f"   Answer  : {result.answer[:150]}")
    else:
        print(f"\n❌ Query gagal: {result.error_message}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Smoke test selesai.")
    print(f"Untuk eval final, ganti model ke '{EVAL_MODEL}':")
    print(f"  client = OllamaClient(model='{EVAL_MODEL}')")
    print("=" * 60)