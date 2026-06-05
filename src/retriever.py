"""
Retriever Module

Handles document retrieval and reranking for the Test-Time-Diffusion framework.

The implementation supports:
    - Pluggable retrieval backends (Tavily Search API, FineWeb API)
    - Document chunking using the internal chunker
    - Cross-encoder reranking using a vLLM-hosted reranker model

Key classes / functions:
    - retrieve_tavily: Retrieves documents using the Tavily Search API.
    - retrieve_fineweb: Retrieves documents using the FineWeb Search API.
    - retrieve_documents: Dispatcher to route queries to the selected backend.
    - retrieve: Main retrieval pipeline including fetching, chunking, and reranking.

Version:
    - 05-Jun-2026 (Version 1.0): Initial implementation and documentation.
"""
import os
import json
import base64
import requests
from typing import List, Dict, Any
from src.utils.logger import CustomLogger
logger = CustomLogger.setup_task_logger(__name__, output_dir="outputs/logs")
from src.utils.chunker import chunk_document

from vllm import LLM

# Item 7: model_name now matches the actually-loaded model so there is no silent mismatch.
# The old declaration `model_name = "BAAI/bge-reranker-v2-m3"` has been removed.
from src.config import config

RERANKER_MODEL = config.retrieval.reranker_model
RETRIEVAL_BACKEND = config.retrieval.backend

_reranker_instance = None

def get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        from vllm import LLM
        _reranker_instance = LLM(
            model=RERANKER_MODEL,
            task="score",
            max_model_len=1024,
            gpu_memory_utilization=0.1,
            enforce_eager=True,
        )
    return _reranker_instance


# ---------------------------------------------------------------------------
# Item 7: The unused call_rerank_api() helper has been removed.
# It hard-coded the query "what is panda?" and would mislead anyone who tried
# to reuse it. If an API-based reranker is needed in the future, it should be
# added as a properly parameterised function.
# ---------------------------------------------------------------------------


def retrieve_tavily(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve documents from the Tavily Search API.

    Requires TAVILY_API_KEY to be set in the environment or .env file.
    Uses basic search depth (1 credit per request) to stay within the free-tier budget.

    Args:
        query (str): The search query string.
        top_k (int): The maximum number of documents to return. Default is 5.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing 'url', 'content', and 'title' keys.

    Raises:
        ImportError: If the tavily-python package is not installed.
        ValueError: If the TAVILY_API_KEY environment variable is not set.

    Example:
        >>> docs = retrieve_tavily("What is quantum computing?", top_k=2)
        >>> len(docs) <= 2
        True
    """
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise ImportError(
            "tavily-python is not installed. Run: pip install tavily-python"
        ) from exc

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set.")

    client = TavilyClient(api_key=api_key)
    response = client.search(query, max_results=top_k, search_depth="basic")

    documents = []
    for result in response.get("results", []):
        documents.append(
            {
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "title": result.get("title", ""),
            }
        )
    return documents


def retrieve_fineweb(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve documents from the FineWeb Search API.

    Serves as the original competition backend, now used as a fallback.

    Args:
        query (str): The search query string.
        top_k (int): The maximum number of documents to return. Default is 5.

    Returns:
        List[Dict[str, Any]]: A list of document dictionaries containing 'url' and 'content' keys.

    Raises:
        ValueError: If the FINEWEB_API_KEY environment variable is not set.
        Exception: On network errors or unexpected API responses.

    Example:
        >>> docs = retrieve_fineweb("Artificial Intelligence", top_k=2)
    """
    api_key = os.getenv("FINEWEB_API_KEY")
    if not api_key:
        raise ValueError("FINEWEB_API_KEY environment variable not set")

    base_url = "https://clueweb22.us/fineweb/search"

    headers = {"x-api-key": api_key, "Content-Type": "application/json"}

    params = {"query": query, "k": top_k}

    try:
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(
                f"FineWeb API error {response.status_code}: {response.text}"
            )

        result = response.json()

        documents = []
        for doc in result.get("results", []):
            try:
                decoded_data = base64.b64decode(doc).decode("utf-8")
                document = json.loads(decoded_data)
                documents.append(document)

            except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
                logger.error(f"Error processing document: {e}")
                continue

        return documents

    except requests.RequestException as e:
        raise Exception(f"Network error connecting to FineWeb API: {e}")


def retrieve_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Pluggable retrieval dispatcher.

    Routes the search query to the backend specified by the RETRIEVAL_BACKEND
    environment variable (e.g., 'tavily' or 'fineweb').

    Args:
        query (str): The search query string.
        top_k (int): The maximum number of documents to return. Default is 5.

    Returns:
        List[Dict[str, Any]]: A list of retrieved document dictionaries.

    Raises:
        ValueError: If an unknown retrieval backend is specified.

    Example:
        >>> docs = retrieve_documents("Machine Learning", top_k=3)
    """
    if RETRIEVAL_BACKEND == "tavily":
        return retrieve_tavily(query, top_k)
    elif RETRIEVAL_BACKEND == "fineweb":
        return retrieve_fineweb(query, top_k)
    else:
        raise ValueError(
            f"Unknown RETRIEVAL_BACKEND '{RETRIEVAL_BACKEND}'. "
            "Valid options: 'tavily', 'fineweb'."
        )


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Main retrieval pipeline including fetching, chunking, and reranking.

    Fetches documents from the active backend, normalizes their schema, chunks them,
    and reranks the chunks using a cross-encoder model to return the most relevant segments.

    Args:
        query (str): The search query string.
        top_k (int): The maximum number of chunks to return after reranking. Default is 5.

    Returns:
        List[Dict[str, Any]]: A list of the top reranked document chunks.

    Example:
        >>> best_chunks = retrieve("Deep learning breakthroughs", top_k=5)
        >>> len(best_chunks) <= 5
        True
    """
    try:
        logger.debug(f"searching q={query!r} via backend={RETRIEVAL_BACKEND!r}")
        docs = retrieve_documents(query, top_k)
        logger.debug(f"retrieved {len(docs)} documents")

        # Normalise document schema: Tavily returns 'content'; FineWeb returns 'text'.
        # Downstream code (chunker, pipeline) uses 'content' as the primary text field.
        for doc in docs:
            if "text" in doc and "content" not in doc:
                doc["content"] = doc["text"]

        chunks = []
        for doc in docs:
            doc_chunks = chunk_document(doc)
            chunks.extend(doc_chunks)

        texts = [chunk["text"] for chunk in chunks]

        try:
            outputs = get_reranker().score(query, texts)
            scores = [output.outputs.score for output in outputs]

            # Item 6: Fixed sort order — reverse=True so the highest-scoring
            # (most relevant) chunks are selected first. The original code used
            # reverse=False which returned the LEAST relevant chunks.
            idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            reordered = [chunks[i] for i in idxs]
            logger.debug(f"top reranked chunk: {reordered[0] if reordered else 'none'}")
            return reordered[:top_k]

        except Exception as e:
            if "out of memory" in str(e).lower():
                import torch
                torch.cuda.empty_cache()
                logger.error("CUDA OOM during reranking. Cleared cache. Returning un-reranked chunks.")
            else:
                logger.error(f"Reranking failed: {e}. Returning un-reranked chunks.")
            return chunks[:top_k]

    except Exception as e:
        logger.error(f"Error in retrieve: {e}")
        return []
