from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import uuid

from evaluators.faithfulness import FaithfulnessEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Eval Service", version="1.0.0")

# Init evaluator sekali saat startup — berat kalau tiap request
evaluator = None

@app.on_event("startup")
def startup():
    global evaluator
    logger.info("Initializing FaithfulnessEvaluator...")
    evaluator = FaithfulnessEvaluator()
    logger.info("FaithfulnessEvaluator ready")

class EvalRequest(BaseModel):
    answer: str
    context: str
    question: str

class EvalResponse(BaseModel):
    faithfulness_score: float
    failure_mode: str
    has_failure: bool
    supported_claims: int
    total_claims: int

@app.get("/health")
def health():
    return {"status": "ok", "service": "llm-eval-framework"}

@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    if not req.answer.strip():
        raise HTTPException(status_code=400, detail="answer cannot be empty")
    if not req.context.strip():
        raise HTTPException(status_code=400, detail="context cannot be empty")

    case = {
        "id": str(uuid.uuid4()),
        "context": req.context,
        "question": req.question,
    }

    try:
        result = evaluator.evaluate(case=case, model_answer=req.answer)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"evaluation failed: {e}")

    failure_mode = "none"
    if result.has_failure:
        failure_mode = "unsupported_claims"
    if result.is_insufficient_context_response:
        failure_mode = "insufficient_context"

    return EvalResponse(
        faithfulness_score=result.faithfulness_score,
        failure_mode=failure_mode,
        has_failure=result.has_failure,
        supported_claims=result.supported_count,
        total_claims=result.total_claims,
    )
