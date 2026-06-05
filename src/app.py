"""
API Entry Point

Provides a FastAPI server exposing static and streaming RAG endpoints.

The implementation supports:
    - Exposing a static RAG endpoint for simple evaluations
    - Exposing a dynamic RAG endpoint using Server-Sent Events (SSE) for streaming updates
    - Handling concurrent pipeline executions

Key classes / functions:
    - EvaluateRequest: Pydantic model for evaluation requests.
    - EvaluateResponse: Pydantic model for evaluation responses.
    - RunRequest: Pydantic model for dynamic run requests.
    - health_check: Lightweight health check endpoint.
    - evaluate_endpoint: Static evaluation endpoint.
    - run_endpoint: Streaming endpoint using SSE.

Version:
    - 05-Jun-2026 (Version 1.0): Initial implementation and documentation.
"""
import asyncio
import contextlib
import json
from typing import Any, AsyncGenerator, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from src.utils.logger import CustomLogger
logger = CustomLogger.setup_task_logger(__name__, output_dir="outputs/logs")
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.pipeline import run_rag_dynamic, run_rag_static

load_dotenv()

app = FastAPI(
    title="MMU-RAG TTD-DR Implementation",
    description="API server exposing static and streaming RAG endpoints.",
)


class EvaluateRequest(BaseModel):
    """
    Pydantic model for an evaluation request.

    Args:
        query (str): The search query to evaluate.
        iid (str): The unique identifier for the request.
    """
    query: str
    iid: str


class EvaluateResponse(BaseModel):
    """
    Pydantic model for an evaluation response.

    Args:
        query_id (str): The unique identifier matching the request.
        generated_response (str): The final generated RAG report.
    """
    query_id: str
    generated_response: str


class RunRequest(BaseModel):
    """
    Pydantic model for a streaming RAG request.

    Args:
        question (str): The user question or query to be processed.
    """
    question: str


@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Lightweight health check endpoint.

    Returns a simple JSON payload indicating that the server is up and running.

    Returns:
        dict[str, str]: A dictionary containing the status.

    Example:
        >>> health_check()
        {'status': 'ok'}
    """
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_endpoint(payload: EvaluateRequest) -> EvaluateResponse:
    """
    Static evaluation endpoint returning a single JSON response.

    Executes the static RAG pipeline using a separate thread to avoid blocking the event loop.

    Args:
        payload (EvaluateRequest): The evaluation request containing the query.

    Returns:
        EvaluateResponse: The evaluation response containing the generated report.

    Raises:
        HTTPException: If the static evaluation fails.

    Example:
        >>> # Depends on the FastAPI test client.
    """
    try:
        generated_response = await asyncio.to_thread(run_rag_static, payload.query)
    except Exception as e:
        logger.exception(f"Static evaluation failed {e}")
        raise HTTPException(status_code=500, detail="Static evaluation failed")

    return EvaluateResponse(query_id=payload.iid, generated_response=generated_response)


@app.post("/run")
async def run_endpoint(payload: RunRequest) -> EventSourceResponse:
    """
    Streaming endpoint that emits SSE updates for the research pipeline.

    Executes the dynamic RAG pipeline in a background thread, bridging its updates
    via an asyncio queue to the FastAPI event stream.

    Args:
        payload (RunRequest): The run request containing the question.

    Returns:
        EventSourceResponse: An SSE streaming response yielding JSON strings.

    Example:
        >>> # Typical usage involves connecting with an EventSource client.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    def pipeline_callback(update: Dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, update)

    def run_pipeline() -> None:
        try:
            run_rag_dynamic(payload.question, pipeline_callback)
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive: upstream services may fail
            logger.exception("Dynamic pipeline failed")
            error_payload = {"error": str(exc), "complete": True}
            loop.call_soon_threadsafe(queue.put_nowait, error_payload)

    pipeline_task = asyncio.create_task(asyncio.to_thread(run_pipeline))

    async def event_publisher() -> AsyncGenerator[Dict[str, str], None]:
        try:
            while True:
                update = await queue.get()
                payload = json.dumps(update)
                yield {"data": payload}
                if update.get("complete") is True:
                    break
        except asyncio.CancelledError:
            pipeline_task.cancel()
            raise
        finally:
            with contextlib.suppress(Exception):
                await pipeline_task

    return EventSourceResponse(event_publisher(), media_type="text/event-stream")


__all__ = ["app"]
