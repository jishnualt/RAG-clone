"""LangGraph orchestration for query answering."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue

from .search import HybridSearchRetriever
from .utils import (
    HYBRID_SEARCH_TOP_K,
    ask_question,
    get_accessible_document_ids,
    get_qdrant_vector_store,
)

logger = logging.getLogger(__name__)


class QueryState(TypedDict, total=False):
    """Mutable state passed between LangGraph nodes."""

    question: str
    mode: str
    metadata_filters: Dict[str, Any]
    assistant_instructions: Optional[str]
    thread_history: List[Dict[str, Any]]
    vector_store_id: str
    collection_name: str
    user: Any
    allow_human_review: bool

    # Retrieval options
    top_k: int
    rerank_model: Optional[str]
    use_mmr: bool

    # Retrieval outputs
    retrieval_results: List[Any]
    retrieval_time_ms: float
    total_chunks_retrieved: int
    retrieval_sources: List[str]
    fallback_reason: Optional[str]

    # Final response payload
    response: Dict[str, Any]
    human_review_required: bool


def _convert_docs_for_custom_context(docs: List[Any]) -> List[Dict[str, Any]]:
    """Convert LangChain documents to the structure expected by ``ask_question``.

    This version *prepends* a clear document id header to the page_content when available.
    This ensures downstream code (and the LLM) receives the document id alongside the chunk text
    even if the downstream `ask_question` implementation ignores metadata dictionaries.
    """

    converted: List[Dict[str, Any]] = []
    for doc in docs:
        page_content = getattr(doc, "page_content", "")
        if not page_content:
            continue

        # Extract metadata in a defensive way (metadata may be dict-like or an object)
        metadata = getattr(doc, "metadata", {}) or {}
        doc_id = None
        try:
            if isinstance(metadata, dict):
                doc_id = metadata.get("document_id") or metadata.get("doc_id") or metadata.get("documentId")
                doc_name = metadata.get("document_name") or metadata.get("document_title") or ("doc_name")
            else:
                doc_id = getattr(metadata, "document_id", None) or getattr(metadata, "doc_id", None)
                doc_name = getattr(metadata, "document_name", None) or getattr(metadata, "doc_name", None)
        except Exception:
            doc_id = None
            doc_name = None

        # If a document id exists, prepend it to the page content so the LLM sees the source
        if doc_id is not None:
            safe_id = str(doc_id)
            safe_name = doc_name
            page_content_with_id = f"[document_id={safe_id}]\n, [document_name={safe_name}]\n" + page_content
            logger.info(f"Retrieved chunk from document_id={safe_id}, "
                        f"content_preview={page_content[:80]!r}")
        else:
            page_content_with_id = page_content

        payload: Dict[str, Any] = {"page_content": page_content_with_id}
        if metadata:
            # Keep original metadata intact for callers that use it
            payload["metadata"] = metadata
        converted.append({"payload": payload})
    return converted


def determine_mode(state: QueryState) -> Dict[str, Any]:
    mode = (state.get("mode") or "document").lower()
    return {"mode": mode}


def route_mode(state: QueryState) -> str:
    mode = state.get("mode", "document")
    if mode in {"normal", "web"}:
        return mode
    return "document"


def retrieve_docs(state: QueryState) -> Dict[str, Any]:
    question = state["question"]
    user = state["user"]
    vector_store_id = state["vector_store_id"]
    metadata_filters = state.get("metadata_filters") or {}
    fallback_reason: Optional[str] = None
    retrieval_results: List[Any] = []
    retrieval_sources: List[str] = []
    retrieval_time_ms = 0.0
    top_k: int = int(state.get("top_k") or HYBRID_SEARCH_TOP_K)
    rerank_model: Optional[str] = state.get("rerank_model")
    use_mmr: bool = bool(state.get("use_mmr") or False)

    try:
        vector_store = get_qdrant_vector_store(user, state["collection_name"])
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("LangGraph: failed to load vector store: %s", exc, exc_info=True)
        fallback_reason = "vector_store_error"
        return {
            "retrieval_results": retrieval_results,
            "retrieval_time_ms": retrieval_time_ms,
            "total_chunks_retrieved": 0,
            "retrieval_sources": retrieval_sources,
            "fallback_reason": fallback_reason,
        }

    accessible_doc_ids = get_accessible_document_ids(user, vector_store_id)
    if not accessible_doc_ids:
        logger.info(
            "LangGraph: no accessible documents for user %s in vector store %s",
            getattr(user, "id", "unknown"),
            vector_store_id,
        )
        fallback_reason = "no_accessible_documents"
        return {
            "retrieval_results": retrieval_results,
            "retrieval_time_ms": retrieval_time_ms,
            "total_chunks_retrieved": 0,
            "retrieval_sources": retrieval_sources,
            "fallback_reason": fallback_reason,
        }

    accessible_doc_ids_str = [str(doc_id) for doc_id in accessible_doc_ids]
    # initialize with allowed doc ids; will refine to actual retrieved doc ids if available
    retrieval_sources = accessible_doc_ids_str

    must_conditions = [
        FieldCondition(
            key="metadata.tenant_id",
            match=MatchValue(value=str(user.tenant.id)),
        ),
        FieldCondition(
            key="metadata.document_id",
            match=MatchAny(any=accessible_doc_ids_str),
        ),
    ]
    for key, value in metadata_filters.items():
        must_conditions.append(
            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
        )

    search_filter = Filter(must=must_conditions)
    start_time = time.time()
    try:
        # Use the new HybridSearchRetriever for standardization
        retriever = HybridSearchRetriever(
            vector_store=vector_store,
            search_filter=search_filter,
            k=top_k,
            rerank_model=rerank_model,
            use_mmr=use_mmr
        )
        retrieval_results = retriever.invoke(question)
    except Exception as exc:  # pragma: no cover - retrieval should rarely fail
        logger.warning(
            "LangGraph: retrieval failed for question '%s': %s",
            question,
            exc,
            exc_info=True,
        )
        fallback_reason = "retrieval_error"
        retrieval_results = []
    else:
        retrieval_time_ms = (time.time() - start_time) * 1000
        logger.info(
            "LangGraph: retrieved %s chunks in %.2f ms",
            len(retrieval_results),
            retrieval_time_ms,
        )
        # refine sources to the unique document_ids present in the retrieved chunks
        try:
            doc_ids = []
            for d in retrieval_results:
                meta = getattr(d, "metadata", None)
                if meta and "document_id" in meta:
                    doc_ids.append(str(meta["document_id"]))
            if doc_ids:
                retrieval_sources = sorted(list({*doc_ids}))
        except Exception:
            pass
        if not retrieval_results:
            fallback_reason = "no_relevant_chunks"

    return {
        "retrieval_results": retrieval_results,
        "retrieval_time_ms": retrieval_time_ms,
        "total_chunks_retrieved": len(retrieval_results),
        "retrieval_sources": retrieval_sources,
        "fallback_reason": fallback_reason,
    }


def route_retrieval(state: QueryState) -> str:
    if state.get("retrieval_results"):
        return "answer"
    return "fallback"


def llm_answer(state: QueryState) -> Dict[str, Any]:
    mode = state.get("mode", "document")
    metadata_filters = state.get("metadata_filters")
    thread_history = state.get("thread_history") or []
    instructions = state.get("assistant_instructions")

    if mode in {"normal", "web"}:
        response = ask_question(
            question=state["question"],
            vector_store_id=state["vector_store_id"],
            user=state["user"],
            collection_name=state["collection_name"],
            assistant_instructions=instructions,
            thread_history=thread_history,
            mode=mode,
            metadata_filters=metadata_filters,
        )
        return {
            "response": response,
            "human_review_required": False,
            "fallback_reason": None,
        }

    docs = state.get("retrieval_results") or []
    converted_docs = _convert_docs_for_custom_context(docs)
    response = ask_question(
        question=state["question"],
        vector_store_id=state["vector_store_id"],
        user=state["user"],
        collection_name=state["collection_name"],
        assistant_instructions=instructions,
        thread_history=thread_history,
        metadata_filters=metadata_filters,
        documents=converted_docs,
    )

    retrieval_time_ms = state.get("retrieval_time_ms")
    if retrieval_time_ms is not None:
        response["retrieval_time_ms"] = round(retrieval_time_ms, 2)
    response["total_chunks_retrieved"] = state.get(
        "total_chunks_retrieved", len(converted_docs)
    )
    sources = state.get("retrieval_sources")
    if sources:
        response["sources"] = sources
    # used_document_ids is already extracted from LLM response in ask_question()

    return {
        "response": response,
        "human_review_required": False,
        "fallback_reason": None,
    }


def fallback_or_review(state: QueryState) -> Dict[str, Any]:
    metadata_filters = state.get("metadata_filters")
    thread_history = state.get("thread_history") or []
    instructions = state.get("assistant_instructions")
    fallback_reason = state.get("fallback_reason") or "no_retrieval"

    response = ask_question(
        question=state["question"],
        vector_store_id=state["vector_store_id"],
        user=state["user"],
        collection_name=state["collection_name"],
        assistant_instructions=instructions,
        thread_history=thread_history,
        metadata_filters=metadata_filters,
        documents=[],
    )

    retrieval_time_ms = state.get("retrieval_time_ms")
    if retrieval_time_ms is not None:
        response["retrieval_time_ms"] = round(retrieval_time_ms, 2)
    response["total_chunks_retrieved"] = state.get("total_chunks_retrieved", 0)
    response.setdefault("sources", [])
    response.setdefault("used_document_ids", [])
    response["fallback_reason"] = fallback_reason

    human_review_required = state.get("allow_human_review", False)
    if human_review_required:
        response["human_review_required"] = True

    return {
        "response": response,
        "human_review_required": human_review_required,
        "fallback_reason": fallback_reason,
    }


_COMPILED_QUERY_GRAPH: Optional[Any] = None


def _build_query_graph() -> Any:
    graph = StateGraph(QueryState)
    graph.add_node("determine_mode", determine_mode)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("llm_answer", llm_answer)
    graph.add_node("fallback_or_review", fallback_or_review)

    graph.set_entry_point("determine_mode")
    graph.add_conditional_edges(
        "determine_mode",
        route_mode,
        {
            "normal": "llm_answer",
            "web": "llm_answer",
            "document": "retrieve_docs",
        },
    )
    graph.add_conditional_edges(
        "retrieve_docs",
        route_retrieval,
        {
            "answer": "llm_answer",
            "fallback": "fallback_or_review",
        },
    )
    graph.add_edge("llm_answer", END)
    graph.add_edge("fallback_or_review", END)
    return graph.compile()


def _get_query_graph() -> Any:
    global _COMPILED_QUERY_GRAPH
    if _COMPILED_QUERY_GRAPH is None:
        _COMPILED_QUERY_GRAPH = _build_query_graph()
    return _COMPILED_QUERY_GRAPH


def run_query_graph(
    *,
    question: str,
    user: Any,
    vector_store_id: str,
    collection_name: str,
    assistant_instructions: Optional[str],
    thread_history: Optional[List[Dict[str, Any]]],
    mode: str = "document",
    metadata_filters: Optional[Dict[str, Any]] = None,
    allow_human_review: bool = False,
    top_k: Optional[int] = None,
    rerank_model: Optional[str] = None,
    use_mmr: bool = False,
) -> Dict[str, Any]:
    """Execute the LangGraph query flow and return the final response."""

    initial_state: QueryState = {
        "question": question,
        "user": user,
        "vector_store_id": vector_store_id,
        "collection_name": collection_name,
        "assistant_instructions": assistant_instructions,
        "thread_history": list(thread_history or []),
        "mode": mode,
        "metadata_filters": metadata_filters or {},
        "allow_human_review": allow_human_review,
        "top_k": int(top_k) if top_k is not None else HYBRID_SEARCH_TOP_K,
        "rerank_model": rerank_model,
        "use_mmr": bool(use_mmr),
    }

    graph = _get_query_graph()
    final_state = graph.invoke(initial_state)
    response = final_state.get("response")
    if response is None:
        logger.warning("LangGraph: no response generated for question '%s'", question)
        return {"answer": "No answer generated.", "sources": []}

    if final_state.get("fallback_reason") and "fallback_reason" not in response:
        response["fallback_reason"] = final_state["fallback_reason"]
    if final_state.get("human_review_required") and not response.get("human_review_required"):
        response["human_review_required"] = True

    return response
