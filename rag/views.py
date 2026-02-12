from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import ValidationError, AuthenticationFailed, NotAuthenticated
from knox.models import AuthToken
from django.shortcuts import get_object_or_404
from django.db import connections, transaction
from django.http import Http404
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.utils.decorators import method_decorator
from .models import (
    Tenant,
    Document,
    Collection,
    VectorStore,
    Assistant,
    Thread,
    Message,
    Run,
    DocumentAlert,
    DocumentAccess,
    LLMProviderConfig,
    User as UserProfile,
    MessageFeedback,
    Conversation,
    ConversationMessage,
    ResponseRecord,
)
from .serializers import *
from .utils import (
    debug_large_file_processing, generate_summary_from_text, is_large_file_by_row_count, monitor_memory_usage, process_file, extract_text_from_file, insert_document_to_vectorstore,
    generate_caption_from_text,
    ask_question, process_large_file_streaming, scroll_all_points_by_vector_store_db_id, delete_points_by_document_db_id,
    enrich_document, detect_alerts, get_authenticated_user, get_qdrant_vector_store, get_llm_config, initialize_qdrant_collection, get_qdrant_client,
    delete_qdrant_collection, ensure_default_assistant, sync_user_login_state, infer_provider_from_dimension, get_provider_embedding_dimension,
    insert_large_file_to_vectorstore_from_path,
)
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
from .search import HybridSearchRetriever
from .langgraph_flow import run_query_graph
from collections import defaultdict
from .utils import (
    extract_document_ids_from_response,
    extract_used_document_ids,
    get_accessible_document_ids,
    get_all_accessible_document_ids,
    infer_provider_from_model,
    normalize_provider_name,
    remove_document_id_citations,
    validate_model_availability,
)
from .llm_providers import (
    OpenAIProvider, OllamaProvider,
    LLMProviderError, LLMRateLimitError, LLMQuotaError,
    LLMAuthenticationError, LLMInvalidRequestError, LLMServiceUnavailableError,
    VectorStoreConnectionError
)
from .permissions import LLMReadyPermission
import os
import threading
from datetime import datetime
from django.contrib.auth import get_user_model
from pathlib import Path
import logging
from django.core.exceptions import ValidationError as DjangoValidationError
import tiktoken
import time
import traceback
import requests
from typing import Optional, List
from django.utils import timezone
import json
import uuid
from django.conf import settings
from .trace import trace_span
from qdrant_client.http.exceptions import UnexpectedResponse
from .resilience import call_with_resilience, close_connections_before_io

logger = logging.getLogger(__name__) # Use Django's logger
User = get_user_model()

PROHIBITED_TERMS = {"drop database", "shutdown"} # Consider moving to settings or a constants file
RATE_LIMIT = 10  # queries per minute
MAX_TOKEN_LIMIT = 100000
_rate_tracker = {}

# Common Redoc/Swagger parameters
AUTH_TOKEN_PARAMETER = openapi.Parameter(
    'token', openapi.IN_PATH, description="Authentication token issued at login.", type=openapi.TYPE_STRING
)

# Swagger/Redoc tags for grouping endpoints
AUTH_TAG = "Authentication"
TENANT_TAG = "Tenants"
USER_TAG = "Users"
PROVIDER_TAG = "LLM Providers"
VECTOR_STORE_TAG = "Vector Store"
DOCUMENT_TAG = "Documents"
ASSISTANT_TAG = "Assistants"
THREAD_TAG = "Threads"
MESSAGE_TAG = "Messages"
RUN_TAG = "Runs"
ACCESS_TAG = "Document Access"
ALERT_TAG = "Document Alerts"
CONVERSATION_TAG = "Conversations"
RESPONSE_TAG = "Responses"


def tokenized_auto_schema(**kwargs):
    """Apply the auth token parameter to swagger/Redoc docs."""

    manual_parameters = kwargs.pop("manual_parameters", [])
    return swagger_auto_schema(manual_parameters=[AUTH_TOKEN_PARAMETER, *manual_parameters], **kwargs)

DEFAULT_ASSISTANT_PROMPT_BASE = (
    "You are RAGitify, a retrieval-augmented assistant. Provide concise, direct answers grounded "
    "in the supplied documents and conversation history. Format responses using Markdown "
    "(headings, bullet points, bold text, code blocks) for clarity and readability. "
    "If the provided information is insufficient, say so and request the missing details instead of guessing."
)

DEFAULT_ASSISTANT_PROMPT_NO_CONTEXT = (
    f"{DEFAULT_ASSISTANT_PROMPT_BASE} When no context is available, use the available tools to gather "
    "what you need before responding."
)

DEFAULT_ASSISTANT_PROMPT_WITH_CONTEXT = (
    f"{DEFAULT_ASSISTANT_PROMPT_BASE} Prioritize the provided context and list all matching instances when "
    "multiple results exist. When you reference information from documents, cite the document ID using the format [document_id=<id>] where <id> matches the document_id from the context headers. "
    "Only call tools if the supplied context and history do not contain the necessary details."
)

JSON_OUTPUT_INSTRUCTION = (
    "\n\nYou must respond in JSON format with this structure:\n"
    '{\n'
    '  "answer": "<your answer text>",\n'
    '  "used_document_ids": ["doc_123", "doc_456"]\n'
    '}\n'
    "Only include document_ids in the 'used_document_ids' array that you actually used from the provided context. "
    "If you did not use any documents, return an empty array: []. "
    "Each document_id should match the [document_id=...] markers in the context."
)

# Appended to custom instructions when no explicit formatting is specified
MARKDOWN_FORMAT_INSTRUCTION = (
    "\n\nFormat responses using Markdown (headings, bullet points, bold text, code blocks) "
    "for clarity and readability. This is the default RAGitify styling."
)

# Keywords that indicate user has specified their own formatting preferences
FORMAT_KEYWORDS = ['markdown', 'json', 'plain text', 'html', 'xml', 'format', 'structured', 'bullet', 'numbered']


def build_assistant_system_prompt(instructions: str | None, *, has_context: bool, require_json: bool = False) -> str:
    """
    Build the system prompt for the assistant, ensuring consistent Markdown formatting.
    
    If custom instructions are provided without explicit formatting preferences,
    appends the default Markdown formatting instruction.
    """
    if instructions:
        # Check if user has specified formatting preferences
        instructions_lower = instructions.lower()
        has_format_preference = any(keyword in instructions_lower for keyword in FORMAT_KEYWORDS)
        
        if has_format_preference:
            # User specified formatting, use their instructions as-is
            base_prompt = instructions
        else:
            # No formatting specified, append Markdown instruction
            base_prompt = instructions + MARKDOWN_FORMAT_INSTRUCTION
    else:
        # No custom instructions, use defaults (which already include Markdown instruction)
        base_prompt = DEFAULT_ASSISTANT_PROMPT_WITH_CONTEXT if has_context else DEFAULT_ASSISTANT_PROMPT_NO_CONTEXT
    
    if require_json and has_context:
        return base_prompt + JSON_OUTPUT_INSTRUCTION
    
    return base_prompt




class RunCancelled(Exception):
    """Raised when a run is cancelled during processing."""


def get_user_active_collection(user):
    collection = getattr(user, "active_collection", None)
    if not collection:
        raise ValidationError(
            {"error": "User has no active collection. Create one first.", "code": "ACTIVE_COLLECTION_REQUIRED"}
        )
    return collection


def _async_ingest(
    document_id: int,
    text: Optional[str],
    file_ext: str,
    user_id: int,
    vector_store_id: str,
    collection_name: str,
    caption=None,
    chunk_strategy: str = "auto",
    source_path: Optional[str] = None,
    is_large_file: bool = False,
):
    """Background task to insert document into the vector store and update status."""
    start_time = time.time()
    doc = None
    with trace_span("ingest_document", document_id=document_id):
        try:
            monitor_memory_usage("START_ASYNC_INGEST", str(document_id))
            user = User.objects.get(id=user_id)
            doc = Document.objects.get(id=document_id)
            if is_large_file and source_path:
                insert_large_file_to_vectorstore_from_path(
                    file_path=source_path,
                    file_ext=file_ext,
                    document_id=str(doc.id),
                    document_name=doc.title or f"Document_{doc.id}",
                    user=user,
                    collection_name=collection_name,
                    vector_store_id=str(vector_store_id),
                )
            else:
                insert_document_to_vectorstore(
                    text=text or "",
                    file_type="file",
                    file_ext=file_ext,
                    document_id=str(doc.id),
                    document_name=doc.title or f"Document_{doc.id}",
                    user=user,
                    collection_name=collection_name,
                    vector_store_id=str(vector_store_id),
                    # caption=caption,
                    # chunk_strategy=chunk_strategy,
                )
            monitor_memory_usage("AFTER_VECTOR_INSERT", str(document_id))
            qdrant_vector_store_instance = get_qdrant_vector_store(user, collection_name)
            qdrant_client = qdrant_vector_store_instance.client
            verification_filter = Filter(must=[
                FieldCondition(key="metadata.document_id", match=MatchValue(value=str(doc.id))),
                FieldCondition(key="metadata.vector_store_id", match=MatchValue(value=str(vector_store_id)))
            ])
            count_response = qdrant_client.count(collection_name=collection_name, count_filter=verification_filter, exact=True)
            if not count_response or count_response.count == 0:
                raise RuntimeError("Qdrant verification failed: no points found")
            with transaction.atomic():
                doc.status = 'completed'
                doc.save(update_fields=['status'])
        except Exception as e:
            logger.error(f"Async ingestion failed for document {document_id}: {e}", exc_info=True)
            if doc:
                with transaction.atomic():
                    doc.status = 'failed'
                    doc.save(update_fields=['status'])
            raise
        finally:
            duration = time.time() - start_time
            logger.info(f"Async ingestion task finished for document {document_id} in {duration:.2f}s")


def _async_ingest_wrapper(document_id, user_id, vector_store_id, collection_name,
                          tmp_path, file_name, caption=None, chunk_strategy: str = "auto"):
    """Wrapper to call _async_ingest and ensure cleanup happens."""
    try:
        logger.info(f"Async wrapper started for document ID: {document_id} ('{file_name}').")
        
        # Fetch user and doc
        user = User.objects.get(id=user_id)
        doc = Document.objects.get(id=document_id)
        file_ext = Path(file_name).suffix.lower()
        
        # 1. Text extraction (Moved from view to background)
        is_large_file = False
        if file_ext in ['.csv', '.xlsx', '.xls']:
            try:
                is_large_file = is_large_file_by_row_count(tmp_path, file_ext)
            except Exception as e:
                logger.warning(f"Could not check if file is large: {e}")
                is_large_file = False

        if is_large_file:
            logger.info(f"Processing large file in background: '{file_name}'")
            extracted_text, summary = process_large_file_streaming(tmp_path, file_ext, file_name, user)
        else:
            extracted_text = extract_text_from_file(tmp_path, file_name, user)
            summary = generate_summary_from_text(extracted_text, user)

        if not extracted_text.strip():
            raise RuntimeError(f"No text could be extracted from the file '{file_name}'.")

        # 1.5. Generate caption for images (Moved from view)
        if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif']:
            try:
                caption = generate_caption_from_text(extracted_text, user)
                logger.info(f"Generated caption for image '{file_name}'")
            except Exception as e:
                logger.warning(f"Failed to generate caption for image '{file_name}': {e}")
                caption = None

        # 2. Update Document object with extracted content and summary
        with transaction.atomic():
            doc.content = extracted_text
            doc.summary = summary
            doc.status = 'processing'
            doc.save(update_fields=['content', 'summary', 'status'])
        
        # 3. Perform vector ingestion
        _async_ingest(
            document_id,
            extracted_text,
            file_ext,
            user_id,
            vector_store_id,
            collection_name,
            caption,
            chunk_strategy,
            source_path=tmp_path if is_large_file else None,
            is_large_file=is_large_file,
        )
        logger.info(f"Async ingestion completed successfully for document ID: {document_id} ('{file_name}').")
    except Exception as e:
        logger.error(f"Async ingestion failed for document ID: {document_id} ('{file_name}'): {e}", exc_info=True)
        try:
            with transaction.atomic():
                doc = Document.objects.get(id=document_id)
                doc.status = 'failed'
                doc.save(update_fields=['status'])
                logger.info(f"Updated document ID {document_id} status to 'failed'.")
        except Document.DoesNotExist:
            logger.warning(f"Document ID {document_id} not found to update status to 'failed'.")
        except Exception as save_e:
            logger.error(f"Failed to update document ID {document_id} status to 'failed': {save_e}")
    finally:
        if tmp_path and Path(tmp_path).exists():
            cleanup_attempts = 0
            max_attempts = 3
            while cleanup_attempts < max_attempts:
                try:
                    import gc, time as _time
                    gc.collect()
                    _time.sleep(0.5)
                    Path(tmp_path).unlink(missing_ok=True)
                    logger.debug(f"Successfully cleaned up temporary file: {tmp_path}")
                    break
                except (PermissionError, OSError) as e:
                    cleanup_attempts += 1
                    if cleanup_attempts >= max_attempts:
                        logger.warning(f"Could not clean up temporary file '{tmp_path}' after {max_attempts} attempts: {e}")
                        logger.info(f"File will be cleaned up by system temp directory maintenance: {tmp_path}")
                        break
                    else:
                        logger.debug(f"Cleanup attempt {cleanup_attempts} failed, retrying: {e}")
                        _time.sleep(1.0)
                except Exception as e_unlink:
                    logger.error(f"Unexpected error cleaning up temporary file '{tmp_path}': {e_unlink}")
                    break
        connections.close_all()



def process_run(run: Run, queries: List[str], vector_store_id_for_run: str, mode: str = "document", metadata_filters: Optional[dict] = None):
        """
        Handles a Run for multiple assistant behaviors and modes.
        Modes:
            - "document": perform document RAG search (default)
            - "normal": converse with the LLM without document retrieval
            - "web": perform a web search and use results for the answer
        Uses ask_question for basic modes and direct LLM calls for tool scenarios.
        """
        logger.info(f"Starting process_run for Run {run.id}")
        def raise_if_cancelled(current_run: Run) -> None:
            current_run.refresh_from_db(fields=['status'])
            if current_run.status == Run.STATUS_CANCELLED:
                logger.info(f"Run {current_run.id}: Cancellation detected. Exiting processing loop.")
                raise RunCancelled()
        # logger.info(f"Run {run.id}: Starting processing. User: {run.thread.user.id}, Assistant: {run.assistant.id}, VectorStore: {vector_store_id_for_run}")
        thread: Optional[Thread] = None
        try:
                raise_if_cancelled(run)
                run.status = Run.STATUS_IN_PROGRESS
                run.save(update_fields=['status'])

                thread = run.thread
                assistant = run.assistant
                messages = thread.messages.order_by('created_at')
                thread_history = [{"role": msg.role, "content": msg.content} for msg in messages]
                logger.debug(f"Run {run.id}: Fetched {len(thread_history)} messages for history.")

                run_user = thread.user
                active_collection = get_user_active_collection(run_user)
                active_collection_name = active_collection.qdrant_collection_name
                llm_config = get_llm_config(run_user.id)
                provider_instance = llm_config["provider_instance"]
                chat_model = assistant.model or llm_config.get("chat_model")
                api_key = llm_config.get("api_key")

                resolved_provider = normalize_provider_name(llm_config.get("provider_name"))
                model_provider = infer_provider_from_model(chat_model, resolved_provider)
                if model_provider != resolved_provider:
                    raise RuntimeError(
                        f"Assistant model '{chat_model}' is for {model_provider}, but user is configured for {resolved_provider}."
                    )

                has_tools = bool(assistant.tools)
                logger.debug(f"Run {run.id}: has_tools={has_tools}")
                has_instructions = bool(assistant.instructions and assistant.instructions.strip())
                logger.debug(f"Run {run.id}: has_instructions={has_instructions}")

                # Handle tool outputs if present (from SubmitToolOutputsAPIView)
                raise_if_cancelled(run)
                tool_outputs = run.tool_outputs or []
                logger.debug(f"Run {run.id}: tool_outputs count={len(tool_outputs)}")
                if tool_outputs:
                    if run.required_action and run.required_action.get('type') == 'submit_tool_outputs':
                        tool_calls = run.required_action.get('submit_tool_outputs', {}).get('tool_calls', [])
                        if tool_calls:
                            assistant_message = {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": call["id"],
                                        "type": call["type"],
                                        "function": {
                                            "name": call["function"]["name"],
                                            "arguments": call["function"]["arguments"]
                                        }
                                    } for call in tool_calls
                                ]
                            }
                            logger.debug(f"Run {run.id}: appending assistant message with tool_calls to history")
                            thread_history.append(assistant_message)
                            logger.info(f"Run {run.id}: Added assistant message with tool_calls.")
                        else:
                            logger.warning(f"Run {run.id}: No tool_calls in required_action. Clearing tool_outputs.")
                            run.tool_outputs = None
                            run.save(update_fields=['tool_outputs'])
                    else:
                        logger.warning(f"Run {run.id}: No valid required_action. Clearing tool_outputs.")
                        run.tool_outputs = None
                        run.save(update_fields=['tool_outputs'])

                    # Process tool outputs, aggregating JSON if possible
                    combined_outputs_text = ""
                    for output in tool_outputs:
                        raise_if_cancelled(run)
                        output_content = output["output"]
                        if not isinstance(output_content, str):
                            output_content = str(output_content)
                            logger.warning(f"Run {run.id}: Tool output for tool_call_id {output['tool_call_id']} was not a string; converted.")
                        try:
                            # Aggregate JSON by name to preserve all instances
                            output_json = json.loads(output_content)
                            if isinstance(output_json, list):
                                aggregated = {}
                                for item in output_json:
                                    name = item.get('name')  # Adjust key if needed
                                    if name:
                                        if name not in aggregated:
                                            aggregated[name] = []
                                        aggregated[name].append(item)
                                output_content = json.dumps(aggregated)
                                logger.info(f"Run {run.id}: Aggregated tool output by name for tool_call_id {output['tool_call_id']}.")
                        except json.JSONDecodeError:
                            pass  # Not JSON, use as-is
                        # Summarize if too long
                        if len(output_content) > MAX_TOKEN_LIMIT // 10:
                            summary_prompt = f"Summarize the following tool output, preserving all instances of names and their details: {output_content[:MAX_TOKEN_LIMIT]}"
                            output_summary = provider_instance.get_chat_completion(
                                [{"role": "user", "content": summary_prompt}],
                                model=chat_model,
                                api_key=api_key
                            )
                            output_content = output_summary.content
                            logger.info(f"Run {run.id}: Summarized tool output for tool_call_id {output['tool_call_id']} to length {len(output_content)}.")
                        thread_history.append({
                            "role": "tool",
                            "content": output_content,
                            "tool_call_id": output["tool_call_id"]
                        })
                        # Build combined assistant-visible text for persistence/reuse
                        combined_outputs_text += f"\n\n[tool_output id={output['tool_call_id']}]\n{output_content}"
                    run.tool_outputs = None
                    run.required_action = None
                    run.save(update_fields=['tool_outputs', 'required_action'])

                    # Persist a synthesized assistant message capturing tool outputs for future reuse
                    try:
                        if combined_outputs_text.strip():
                            Message.objects.create(
                                thread=thread,
                                role='assistant',
                                content=combined_outputs_text.strip(),
                                user=None,
                                metadata={"used_document_ids": []}
                            )
                            logger.info(f"Run {run.id}: Persisted assistant message with prior tool outputs for reuse.")
                    except Exception as persist_e:
                        logger.warning(f"Run {run.id}: Failed to persist assistant message for tool outputs: {persist_e}")

                for query_content in queries:
                    raise_if_cancelled(run)
                    logger.info(f"Run {run.id}: Processing query (first 100 chars): '{query_content[:100]}...'")

                    # Initialize filters for the query
                    local_filters = metadata_filters.copy() if metadata_filters else {}

                    if mode in ("normal", "web"):
                        instructions = (
                            assistant.instructions
                            or run.assistant_intro
                            or "You are a helpful assistant."
                        )
                        raise_if_cancelled(run)
                        result = run_query_graph(
                            question=query_content,
                            user=run_user,
                            vector_store_id=str(vector_store_id_for_run),
                            collection_name=active_collection_name,
                            assistant_instructions=instructions,
                            thread_history=thread_history,
                            mode=mode,
                            metadata_filters=local_filters,
                        )
                        raise_if_cancelled(run)
                        # Extract used_document_ids from result
                        used_document_ids = result.get("used_document_ids", [])
                        # Get answer (should already be cleaned by ask_question, but ensure it's clean)
                        answer_content = result.get("answer", "No answer generated.")
                        # Remove any remaining citations just in case
                        cleaned_answer = remove_document_id_citations(answer_content)
                        message_metadata = {"used_document_ids": used_document_ids}
                        logger.info(f"Run {run.id}: Storing message with metadata: {message_metadata}")
                        Message.objects.create(
                            thread=thread,
                            role='assistant',
                            content=cleaned_answer,
                            user=None,
                            metadata=message_metadata
                        )
                        if result.get("human_review_required"):
                            logger.info(
                                "Run %s: human review requested for query '%s'", run.id, query_content
                            )
                        continue

                    if not has_tools:
                        if not has_instructions:
                            instructions = (
                                "You are a helpful assistant. Always prioritize using the provided document context and conversation history. "
                                "List all matching instances if multiple exist for a query. If no context or history helps, state what is missing."
                            )
                        else:
                            instructions = assistant.instructions
                        raise_if_cancelled(run)
                        result = run_query_graph(
                            question=query_content,
                            user=run_user,
                            vector_store_id=str(vector_store_id_for_run),
                            collection_name=active_collection_name,
                            assistant_instructions=instructions,
                            thread_history=thread_history,
                            mode="document",
                            metadata_filters=local_filters,
                        )
                        logger.debug(
                            f"Run {run.id}: Graph answer preview: '{result.get('answer', '')[:100]}...'. Sources: {result.get('sources')}"
                        )
                        if result.get("fallback_reason"):
                            logger.info(
                                "Run %s: fallback path triggered (%s)",
                                run.id,
                                result["fallback_reason"],
                            )
                        raise_if_cancelled(run)
                        # Extract used_document_ids from result
                        used_document_ids = result.get("used_document_ids", [])
                        # Get answer (should already be cleaned by ask_question, but ensure it's clean)
                        answer_content = result.get("answer", "No answer generated.")
                        # Remove any remaining citations just in case
                        cleaned_answer = remove_document_id_citations(answer_content)
                        message_metadata = {"used_document_ids": used_document_ids}
                        logger.info(f"Run {run.id}: Storing message with metadata: {message_metadata}")
                        Message.objects.create(
                            thread=thread,
                            role='assistant',
                            content=cleaned_answer,
                            user=None,
                            metadata=message_metadata
                        )
                        if result.get("human_review_required"):
                            logger.info(
                                "Run %s: human review requested for query '%s'", run.id, query_content
                            )
                        continue

                    # Perform RAG search for tool-enabled flows
                    augmented_context = ""
                    available_doc_ids = []  # Initialize to empty list for validation
                    if thread.vector_store and query_content:
                        try:
                            collection_name = active_collection_name
                            vector_store = get_qdrant_vector_store(run_user, collection_name)
                            # Determine accessible documents for this user and vector store
                            accessible_doc_ids = get_accessible_document_ids(run_user, str(vector_store_id_for_run))
                            if not accessible_doc_ids:
                                logger.info(f"Run {run.id}: No accessible documents found for user {run_user.id} in vector store {vector_store_id_for_run}.")
                                search_results = []
                            else:
                                # Ensure IDs are strings to match stored metadata
                                accessible_doc_ids_str = [str(doc_id) for doc_id in accessible_doc_ids]
                                must_conditions = [
                                    FieldCondition(key="metadata.tenant_id", match=MatchValue(value=str(run_user.tenant.id))),
                                    FieldCondition(key="metadata.document_id", match=MatchAny(any=accessible_doc_ids_str)),
                                ]
                                for key, value in local_filters.items():
                                    must_conditions.append(FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value)))
                                search_filter = Filter(must=must_conditions)
                                # Use the standardized HybridSearchRetriever
                                retriever = HybridSearchRetriever(
                                    vector_store=vector_store,
                                    search_filter=search_filter,
                                    k=25
                                )
                                search_results = retriever.invoke(query_content)
                            
                            # Build augmented_context with explicit document id and document name headers to help the LLM cite sources.
                            augmented_parts = []
                            for doc in search_results:
                                page_content = getattr(doc, "page_content", "") or ""
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
                                header = f"[document_id={doc_id}]\n, [document_name={doc_name}]\n" if doc_id is not None else ""
                                augmented_parts.append(header + page_content)
                                if doc_id is not None:
                                    logger.info(f"Search result includes document_id={doc_id}, "
                                                f"content_preview={page_content[:80]!r}")
                                else:
                                    logger.info(f"Search result with no document_id, "
                                                f"content_preview={page_content[:80]!r}")
                            augmented_context = "\n\n".join(augmented_parts)
                            # Track available document IDs for validation (will extract from response later)
                            available_doc_ids = extract_used_document_ids(search_results)
                            logger.info(f"Run {run.id}: Retrieved {len(search_results)} relevant documents.")
                        except Exception as e:
                            logger.warning(f"Run {run.id}: Document search failed: {e}")
                            augmented_context = ""
                            available_doc_ids = []

                    # Pre-scan history for tool outputs; include persisted outputs regardless of keyword match
                    history_tool_context = ""
                    # Include up to last 2 persisted assistant messages containing tool outputs
                    persisted_tool_msgs = [m for m in thread_history if m.get("role") == "assistant" and "[tool_output" in str(m.get("content", ""))]
                    for m in persisted_tool_msgs[-2:]:
                        content_str = str(m.get("content", ""))
                        history_tool_context += "\nPrevious tool output: " + content_str[:MAX_TOKEN_LIMIT // 10]
                    # Also include any recent explicit tool role messages (up to last 2)
                    tool_role_msgs = [m for m in thread_history if m.get("role") == "tool"]
                    for m in tool_role_msgs[-2:]:
                        content_str = str(m.get("content", ""))
                        history_tool_context += "\nPrevious tool output: " + content_str[:MAX_TOKEN_LIMIT // 10]
                    if history_tool_context:
                        logger.info(f"Run {run.id}: Appended previous tool outputs from history.")

                    # 3: Tools and instructions
                    # If no context, trigger tool call
                    if not augmented_context and not history_tool_context:
                        system_prompt = build_assistant_system_prompt(
                            assistant.instructions,
                            has_context=False,
                        )
                        api_tools = [tool for tool in (assistant.tools or []) if tool.get("type") in ["function", "custom"]]
                        llm_messages = [
                            {"role": "system", "content": system_prompt}
                        ]
                        llm_messages.extend(thread_history)
                        llm_messages.append({"role": "user", "content": query_content})
                        raise_if_cancelled(run)
                        connections.close_all()
                        response = provider_instance.get_chat_completion(
                            llm_messages,
                            model=chat_model,
                            api_key=api_key,
                            tools=api_tools
                        )
                        if response.tool_calls:
                            tool_calls = response.tool_calls
                            custom_tool_calls = [tc for tc in tool_calls if tc.get("type", "function") == "function"]
                            if custom_tool_calls:
                                raise_if_cancelled(run)
                                run.required_action = {
                                    "type": "submit_tool_outputs",
                                    "submit_tool_outputs": {
                                        "tool_calls": [
                                            {
                                                "id": tc["id"],
                                                "type": tc["type"],
                                                "function": {
                                                    "name": tc["function"]["name"],
                                                    "arguments": tc["function"]["arguments"]
                                                }
                                            } for tc in custom_tool_calls
                                        ]
                                    }
                                }
                                run.status = Run.STATUS_REQUIRES_ACTION
                                run.save(update_fields=['required_action', 'status'])
                                logger.info(f"Run {run.id}: Set to requires_action due to no context and available tools.")
                                return
                        # If no tool calls, return no content
                        raise_if_cancelled(run)
                        Message.objects.create(
                            thread=thread,
                            role='assistant',
                            content="No content available to answer the query, and no tools were called.",
                            user=None
                        )
                        logger.info(f"Run {run.id}: No content or tool calls triggered.")
                        continue

                    # Context or history available, proceed with LLM
                    system_prompt = build_assistant_system_prompt(
                        assistant.instructions,
                        has_context=True,
                    )
                    # Add instruction to cite document IDs when using context
                    citation_instruction = "\n\nIMPORTANT: When you reference information from the document context, cite the document ID using the format [document_id=<id>] where <id> matches the document_id from the context headers."
                    api_tools = [tool for tool in (assistant.tools or []) if tool.get("type") in ["function", "custom"]]
                    llm_messages = [
                        {
                            "role": "system",
                            "content": system_prompt + citation_instruction + "\nRelevant document context: " + (augmented_context or "No relevant documents found.") + history_tool_context
                        }
                    ]
                    llm_messages.extend(thread_history)
                    llm_messages.append({"role": "user", "content": query_content})
                    raise_if_cancelled(run)
                    connections.close_all()
                    response = provider_instance.get_chat_completion(
                        llm_messages,
                        model=chat_model,
                        api_key=api_key,
                        tools=api_tools
                    )
                    if response.tool_calls:
                        tool_calls = response.tool_calls
                        custom_tool_calls = [tc for tc in tool_calls if tc.get("type", "function") == "function"]
                        if custom_tool_calls:
                            raise_if_cancelled(run)
                            run.required_action = {
                                "type": "submit_tool_outputs",
                                "submit_tool_outputs": {
                                    "tool_calls": [
                                        {
                                            "id": tc["id"],
                                            "type": tc["type"],
                                            "function": {
                                                "name": tc["function"]["name"],
                                                "arguments": tc["function"]["arguments"]
                                            }
                                        } for tc in custom_tool_calls
                                    ]
                                }
                            }
                            run.status = Run.STATUS_REQUIRES_ACTION
                            run.save(update_fields=['required_action', 'status'])
                            logger.info(f"Run {run.id}: Set to requires_action for tool calls.")
                            return
                    
                    assistant_response = response.content
                    raise_if_cancelled(run)
                    # Extract document IDs that the LLM actually referenced in its response
                    available_for_validation = available_doc_ids if 'available_doc_ids' in locals() else None
                    logger.info(f"Run {run.id}: Extracting document IDs from assistant response. Available doc IDs: {available_for_validation}")
                    logger.debug(f"Run {run.id}: Assistant response preview (first 500 chars): {str(assistant_response)[:500]!r}")
                    used_document_ids = extract_document_ids_from_response(
                        str(assistant_response), 
                        available_for_validation,
                        tenant=run_user.tenant,
                        user=run_user
                    )
                    logger.info(f"Run {run.id}: Extracted {len(used_document_ids)} used document IDs: {used_document_ids}")
                    # Remove document ID citations from content for clean display
                    cleaned_content = remove_document_id_citations(str(assistant_response))
                    logger.info(f"Run {run.id}: Cleaned content (removed citations). Original length: {len(str(assistant_response))}, Cleaned length: {len(cleaned_content)}")
                    message_metadata = {"used_document_ids": used_document_ids}
                    logger.info(f"Run {run.id}: Storing message with metadata: {message_metadata} (type: {type(message_metadata)}, used_document_ids type: {type(used_document_ids)})")
                    message_obj = Message.objects.create(
                        thread=thread,
                        role='assistant',
                        content=cleaned_content,
                        user=None,
                        metadata=message_metadata
                    )
                    # Verify what was actually saved
                    message_obj.refresh_from_db()
                    logger.info(f"Run {run.id}: Message saved with ID {message_obj.id}. Metadata after save: {message_obj.metadata} (type: {type(message_obj.metadata)})")

                try:
                    with transaction.atomic():
                        run.refresh_from_db(fields=['status'])
                        if run.status == Run.STATUS_CANCELLED:
                            logger.info(f"Run {run.id}: Skipping completion update because run was cancelled.")
                        else:
                            run.status = Run.STATUS_COMPLETED
                            run.completed_at = timezone.now()
                            run.save(update_fields=['status', 'completed_at'])
                            logger.info(f"Run {run.id} completed successfully.")
                except Exception as save_e:
                    logger.error(f"Run {run.id}: Failed to save completed status: {save_e}")

        except RunCancelled:
            logger.info(f"Run {run.id}: Processing halted due to cancellation.")
            return
        except (ValidationError, DjangoValidationError) as e:
            logger.error(f"Run {run.id}: Processing failed due to validation error: {e}", exc_info=True)
            try:
                run.status = Run.STATUS_FAILED
                run.metadata['error_message'] = str(e)
                run.completed_at = timezone.now()
                run.save(update_fields=['status', 'metadata', 'completed_at'])
                if thread:
                    Message.objects.create(thread=thread, role='assistant', content=str(e), user=None, metadata={"used_document_ids": []})
            except Exception as save_e:
                logger.error(f"Run {run.id}: Failed to save failed status: {save_e}")
        except LLMProviderError as e:
            logger.error(f"Run {run.id}: LLM provider error: {e}", exc_info=True)
            try:
                run.status = Run.STATUS_FAILED
                run.metadata['error_message'] = str(e)
                run.completed_at = timezone.now()
                run.save(update_fields=['status', 'metadata', 'completed_at'])
                if thread:
                    Message.objects.create(thread=thread, role='assistant', content=str(e), user=None, metadata={"used_document_ids": []})
            except Exception as save_e:
                logger.error(f"Run {run.id}: Failed to save failed status: {save_e}")
        except RuntimeError as e:
            logger.error(f"Run {run.id}: Processing failed due to runtime error: {e}", exc_info=True)
            try:
                run.status = Run.STATUS_FAILED
                run.metadata['error_message'] = str(e)
                run.completed_at = timezone.now()
                run.save(update_fields=['status', 'metadata', 'completed_at'])
                if thread:
                    Message.objects.create(thread=thread, role='assistant', content=str(e), user=None, metadata={"used_document_ids": []})
            except Exception as save_e:
                logger.error(f"Run {run.id}: Failed to save failed status: {save_e}")
        except Exception as e:
            logger.error(f"Run {run.id}: Unexpected error in process_run: {e}", exc_info=True)
            try:
                run.status = Run.STATUS_FAILED
                run.metadata['error_message'] = f"Unexpected error: {str(e)}"
                run.completed_at = timezone.now()
                run.save(update_fields=['status', 'metadata', 'completed_at'])
                if thread:
                    Message.objects.create(thread=thread, role='assistant', content="Run processing failed due to an unexpected server error.", user=None, metadata={"used_document_ids": []})
            except Exception as save_e:
                logger.error(f"Run {run.id}: Failed to save failed status: {save_e}")
        finally:
            connections.close_all()

class TokenAuthenticatedMixin:
    """Mixin for token-based authentication."""
    permission_classes = [LLMReadyPermission]
    
    def initial(self, request, *args, **kwargs):
        try:
            authenticated_user = get_authenticated_user(kwargs.get('token'))
            request.user = authenticated_user
            self.user = authenticated_user
            logger.debug(f"TokenAuthenticatedMixin: User {self.user.username} (ID: {self.user.id}) authenticated successfully for request to '{request.path}'.")
            
            super().initial(request, *args, **kwargs)
            
            try:
                if getattr(self.user, "llm_configured", False):
                    ensure_default_assistant(self.user)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to ensure default assistant for user %s: %s",
                    getattr(self.user, "id", None),
                    exc,
                    exc_info=True,
                )
        except (Http404, DjangoValidationError, ValueError) as e:
            logger.warning(f"Authentication failed for request to '{request.path}': {e}")
            raise AuthenticationFailed(detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error during authentication for request to '{request.path}': {e}", exc_info=True)
            raise AuthenticationFailed(detail="Authentication failed due to an unexpected error.")

# Tenant CRUD
@method_decorator(
    name='post',
    decorator=swagger_auto_schema(
        operation_description="Create a tenant record.",
        tags=[TENANT_TAG],
        request_body=TenantSerializer,
        responses={status.HTTP_201_CREATED: TenantSerializer},
    ),
)
class TenantCreateAPIView(generics.CreateAPIView):
    serializer_class = TenantSerializer
    queryset = Tenant.objects.all()


@method_decorator(
    name='get',
    decorator=swagger_auto_schema(
        operation_description="List all tenants.",
        tags=[TENANT_TAG],
        responses={status.HTTP_200_OK: TenantSerializer(many=True)},
    ),
)
class TenantListAPIView(generics.ListAPIView):
    serializer_class = TenantSerializer
    queryset = Tenant.objects.all()


@method_decorator(
    name='get',
    decorator=swagger_auto_schema(
        operation_description="Retrieve tenant details by ID.",
        tags=[TENANT_TAG],
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Tenant ID", type=openapi.TYPE_INTEGER)],
    ),
)
@method_decorator(
    name='put',
    decorator=swagger_auto_schema(
        operation_description="Update a tenant by ID.",
        tags=[TENANT_TAG],
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Tenant ID", type=openapi.TYPE_INTEGER)],
        request_body=TenantSerializer,
        responses={status.HTTP_200_OK: TenantSerializer},
    ),
)
@method_decorator(
    name='delete',
    decorator=swagger_auto_schema(
        operation_description="Delete a tenant by ID.",
        tags=[TENANT_TAG],
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Tenant ID", type=openapi.TYPE_INTEGER)],
    ),
)
class TenantRetrieveUpdateDestroyAPIView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = TenantSerializer
    queryset = Tenant.objects.all()
    lookup_field = 'id'


@method_decorator(
    name='post',
    decorator=swagger_auto_schema(
        operation_description="Register a user, tenant, and optional LLM provider setup.",
        tags=[AUTH_TAG],
        request_body=RegisterSerializer,
        responses={status.HTTP_201_CREATED: RegisterResponseSerializer},
    ),
)
class RegisterAPI(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        active_collection = getattr(user, "active_collection", None)
        ready = bool(
            getattr(user, "llm_configured", False)
            and getattr(user, "active_collection_ready", False)
            and active_collection
        )
        response_payload = {
            "user": UserSerializer(user).data,
            "tenant": {"id": user.tenant.id, "name": user.tenant.name} if user.tenant else None,
            "llm_setup_required": not ready,
        }
        if active_collection:
            response_payload["active_collection"] = {
                "id": str(active_collection.id),
                "name": active_collection.name,
                "qdrant_collection_name": active_collection.qdrant_collection_name,
            }
        headers = self.get_success_headers(serializer.data)
        return Response(response_payload, status=status.HTTP_201_CREATED, headers=headers)


class LLMSetupAPIView(TokenAuthenticatedMixin, APIView):
    serializer_class = LLMSetupSerializer

    @swagger_auto_schema(
        operation_description="Provision an LLM provider and collection for the authenticated user.",
        manual_parameters=[AUTH_TOKEN_PARAMETER],
        tags=[AUTH_TAG],
        request_body=LLMSetupSerializer,
        responses={status.HTTP_201_CREATED: LLMSetupResponseSerializer},
    )
    def post(self, request, token):
        serializer = self.serializer_class(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        if getattr(self.user, "llm_configured", False) and getattr(self.user, "active_collection", None):
            collection = self.user.active_collection
            return Response(
                {
                    "error": "LLM already configured.",
                    "code": "LLM_ALREADY_CONFIGURED",
                    "active_collection": {
                        "id": str(collection.id),
                        "name": collection.name,
                        "qdrant_collection_name": collection.qdrant_collection_name,
                    },
                },
                status=status.HTTP_409_CONFLICT,
            )

        provider = serializer.validated_data["llm_provider"]
        collection_name = serializer.validated_data.get("collection_name") or "default_collection"
        normalized_provider = normalize_provider_name(provider)
        embedding_dimension = get_provider_embedding_dimension(provider)
        if embedding_dimension is None:
            return Response(
                {"error": "Unsupported provider.", "code": "UNSUPPORTED_PROVIDER"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        tenant = self.user.tenant
        if not tenant:
            return Response(
                {"error": "User has no associated tenant.", "code": "TENANT_REQUIRED"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if Collection.objects.filter(tenant=tenant, name=collection_name).exists():
            return Response(
                {
                    "error": "Collection name already exists in this tenant. Choose a different collection_name.",
                    "code": "COLLECTION_NAME_TAKEN",
                },
                status=status.HTTP_409_CONFLICT,
            )

        collection = None
        try:
            collection = Collection.objects.create(
                tenant=tenant,
                name=collection_name,
                owner=self.user,
                is_active=True,
                embedding_dimension=embedding_dimension,
                provider=normalized_provider,
            )
            initialize_qdrant_collection(collection.qdrant_collection_name, embedding_dimension)
            self.user.active_collection = collection
            self.user.active_collection_ready = True
            self.user.llm_configured = True
            self.user.selected_llm_provider = normalized_provider
            self.user.is_setup = True
            self.user.save(
                update_fields=[
                    "active_collection",
                    "active_collection_ready",
                    "llm_configured",
                    "selected_llm_provider",
                    "is_setup",
                ]
            )
        except ValueError as e:
            if collection:
                collection.delete()
            return Response(
                {"error": str(e), "code": "QDRANT_DIMENSION_MISMATCH"},
                status=status.HTTP_409_CONFLICT,
            )
        except Exception as exc:
            if collection:
                collection.delete()
            logger.error("Failed to configure LLM for user %s: %s", self.user.id, exc, exc_info=True)
            return Response(
                {"error": "Failed to configure LLM/collection.", "code": "LLM_SETUP_FAILED"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return Response(
            {
                "message": "LLM configured and collection created.",
                "active_collection": {
                    "id": str(collection.id),
                    "name": collection.name,
                    "qdrant_collection_name": collection.qdrant_collection_name,
                    "embedding_dimension": embedding_dimension,
                    "provider": normalized_provider,
                },
            },
            status=status.HTTP_201_CREATED,
        )


class UserStatusAPIView(TokenAuthenticatedMixin, APIView):
    @swagger_auto_schema(
        operation_description="Return onboarding and LLM readiness state for the authenticated user.",
        manual_parameters=[AUTH_TOKEN_PARAMETER],
        tags=[AUTH_TAG],
        responses={status.HTTP_200_OK: UserStatusResponseSerializer},
    )
    def get(self, request, token):
        collection = getattr(self.user, "active_collection", None)
        ready = bool(
            getattr(self.user, "llm_configured", False)
            and getattr(self.user, "active_collection_ready", False)
            and collection
        )
        return Response(
            {
                "llm_configured": getattr(self.user, "llm_configured", False),
                "collection_ready": getattr(self.user, "active_collection_ready", False),
                "active_provider": getattr(self.user, "selected_llm_provider", None),
                "ready": ready,
                "active_collection": {
                    "id": str(collection.id),
                    "name": collection.name,
                    "qdrant_collection_name": collection.qdrant_collection_name,
                }
                if collection
                else None,
            },
            status=status.HTTP_200_OK,
        )


@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        operation_description="List users for the authenticated tenant.",
        tags=[USER_TAG],
        responses={status.HTTP_200_OK: UserSerializer(many=True)},
    ),
)
class UserListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = UserSerializer
    def get_queryset(self):
        return User.objects.filter(tenant=self.user.tenant)


@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        operation_description="Retrieve a user within the tenant.",
        tags=[USER_TAG],
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="User ID", type=openapi.TYPE_INTEGER)],
    ),
)
@method_decorator(
    name='put',
    decorator=tokenized_auto_schema(
        operation_description="Update a user within the tenant.",
        tags=[USER_TAG],
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="User ID", type=openapi.TYPE_INTEGER)],
        request_body=UserSerializer,
        responses={status.HTTP_200_OK: UserSerializer},
    ),
)
@method_decorator(
    name='delete',
    decorator=tokenized_auto_schema(
        operation_description="Delete a user within the tenant.",
        tags=[USER_TAG],
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="User ID", type=openapi.TYPE_INTEGER)],
    ),
)
class UserRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = UserSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return User.objects.filter(tenant=self.user.tenant)


class UserFullDeleteAPIView(TokenAuthenticatedMixin, APIView):
    """Delete a user and all associated data, including their Qdrant footprint."""

    def delete(self, request, token=None, user_id: Optional[int] = None):
        target_user = self.user
        if user_id and user_id != target_user.id:
            logger.warning(
                "User %s attempted to delete user %s without permission", target_user.id, user_id
            )
            return Response(
                {"error": "You can only delete your own account."},
                status=status.HTTP_403_FORBIDDEN,
            )

        tenant = target_user.tenant
        tenant_id = tenant.id if tenant else None
        collection = getattr(target_user, "active_collection", None)
        collection_name = collection.qdrant_collection_name if collection else None
        username = target_user.username
        user_id_value = target_user.id

        try:
            with transaction.atomic():
                if collection:
                    collection.delete()
                target_user.delete()
                if tenant and not tenant.users.exists():
                    tenant.delete()
        except Exception as exc:
            logger.error(
                "Failed to delete user %s and related records: %s", target_user.id, exc, exc_info=True
            )
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        if collection_name and tenant_id:
            try:
                delete_qdrant_collection(collection_name)
            except Exception as exc:
                logger.error(
                    "User %s removed from DB but Qdrant cleanup failed for collection '%s': %s",
                    user_id_value,
                    collection_name,
                    exc,
                    exc_info=True,
                )
                return Response(
                    {
                        "error": (
                            f"User removed from database, but vector cleanup failed for collection "
                            f"'{collection_name}': {exc}"
                        )
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        return Response(
            {"message": f"User '{username}' and all associated data have been deleted."},
            status=status.HTTP_200_OK,
        )

    @staticmethod
    def _delete_user_vectors(collection_name: str, tenant_id: int, user_id: int) -> None:
        close_connections_before_io("Qdrant user vector deletion")
        client = get_qdrant_client()
        selector = Filter(
            must=[
                FieldCondition(key="metadata.tenant_id", match=MatchValue(value=str(tenant_id))),
                FieldCondition(key="metadata.user_id", match=MatchValue(value=str(user_id))),
            ]
        )
        try:
            call_with_resilience(
                lambda: client.delete(collection_name=collection_name, points_selector=selector, wait=True),
                service="qdrant_delete_user_vectors",
                exceptions=(Exception,),
            )
            logger.info(
                "Deleted Qdrant points for user %s in collection '%s'", user_id, collection_name
            )
        except UnexpectedResponse as exc:
            if getattr(exc, "status_code", None) == 404:
                logger.info(
                    "Collection '%s' not found while deleting vectors for user %s", collection_name, user_id
                )
                return
            raise RuntimeError(
                f"Failed to delete vectors for user {user_id} in collection '{collection_name}': "
                f"{exc.content.decode() if exc.content else str(exc)}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to delete vectors for user {user_id} in collection '{collection_name}': {exc}"
            ) from exc

# Authentication
class LoginView(generics.CreateAPIView):
    serializer_class = LoginSerializer

    @swagger_auto_schema(
        operation_description="Authenticate and return a session token.",
        tags=[AUTH_TAG],
        request_body=LoginSerializer,
        responses={status.HTTP_201_CREATED: LoginResponseSerializer},
    )
    def post(self, request, format=None):
        try:
            serializer = self.serializer_class(data=request.data)
            serializer.is_valid(raise_exception=True)
            user = serializer.validated_data['user']
            if not hasattr(user, 'tenant') or user.tenant is None:
                ##logger.error(f"User {user.username} has no associated tenant")
                return Response({"error": "User has no associated tenant"}, status=status.HTTP_403_FORBIDDEN)
            sync_state = {}
            try:
                sync_state = sync_user_login_state(user)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to sync login state for user %s: %s", user.id, exc, exc_info=True)

            token_instance, token = AuthToken.objects.create(user)
            ##Logger.info(f"User {user.username} logged in successfully")
            active_collection = getattr(user, "active_collection", None)
            ready = bool(
                getattr(user, "llm_configured", False)
                and getattr(user, "active_collection_ready", False)
                and active_collection
            )
            response_payload = {
                'token': token_instance.token_key,
                'user': UserSerializer(user).data,
                'tenant': {"id": user.tenant.id, "name": user.tenant.name} if user.tenant else None,
                "llm_setup_required": not ready,
            }
            if active_collection:
                response_payload["active_collection"] = {
                    "id": str(active_collection.id),
                    "name": active_collection.name,
                    "qdrant_collection_name": active_collection.qdrant_collection_name,
                }
            if sync_state.get("warnings"):
                response_payload["warnings"] = sync_state["warnings"]
            return Response(response_payload, status=status.HTTP_201_CREATED)
        except ValidationError as e:
            ##logger.error(f"Login failed: {e}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class LogoutView(TokenAuthenticatedMixin, APIView):
    @swagger_auto_schema(
        operation_description="Invalidate the provided authentication token.",
        manual_parameters=[AUTH_TOKEN_PARAMETER],
        tags=[AUTH_TAG],
        responses={status.HTTP_200_OK: LogoutResponseSerializer},
    )
    def post(self, request, token=None, format=None):
        try:
            auth_token = get_object_or_404(AuthToken, token_key=token)
            auth_token.delete()
            return Response({'message': 'Logged out successfully.'}, status=status.HTTP_200_OK)
        except Exception as e:
            ##logger.error(f"Error logging out: {e}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class ProtectedView(TokenAuthenticatedMixin, APIView):
    @swagger_auto_schema(
        operation_description="Validate a token and fetch the basic user profile.",
        manual_parameters=[AUTH_TOKEN_PARAMETER],
        tags=[AUTH_TAG],
    )
    def get(self, request, token):
        return Response({
            'user': UserProfile.objects.filter(id=self.user.id).values('username', 'email', 'first_name', 'last_name', 'tenant').first(),
            'token': token,
        }, status=status.HTTP_200_OK)

# LLM Provider Config CRUD
@method_decorator(
    name='post',
    decorator=tokenized_auto_schema(
        operation_description="Create an LLM provider configuration for the authenticated user.",
        tags=[PROVIDER_TAG],
        request_body=LLMProviderConfigSerializer,
        responses={status.HTTP_201_CREATED: LLMProviderConfigSerializer},
    ),
)
class LLMProviderConfigCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = LLMProviderConfigSerializer

    def perform_create(self, serializer):
        serializer.save(user=self.user)
        logger.info(f"LLMProviderConfig created for user '{self.user.username}' (ID: {self.user.id}).")


class LLMProviderConfigListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = LLMProviderConfigSerializer
    def get_queryset(self):
        return LLMProviderConfig.objects.filter(user=self.user).order_by('-created_at')


@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        operation_description="List LLM provider configurations for the authenticated user.",
        tags=[PROVIDER_TAG],
        responses={status.HTTP_200_OK: LLMProviderConfigSerializer(many=True)},
    ),
)
class LLMProviderConfigRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = LLMProviderConfigSerializer
    lookup_field = 'id'

    @swagger_auto_schema(
        operation_description="Retrieve a provider configuration by ID.",
        tags=[PROVIDER_TAG],
        manual_parameters=[AUTH_TOKEN_PARAMETER, openapi.Parameter('id', openapi.IN_PATH, description="Config ID", type=openapi.TYPE_INTEGER)],
        responses={status.HTTP_200_OK: LLMProviderConfigSerializer},
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Update a provider configuration by ID.",
        tags=[PROVIDER_TAG],
        manual_parameters=[AUTH_TOKEN_PARAMETER, openapi.Parameter('id', openapi.IN_PATH, description="Config ID", type=openapi.TYPE_INTEGER)],
        request_body=LLMProviderConfigSerializer,
        responses={status.HTTP_200_OK: LLMProviderConfigSerializer},
    )
    def put(self, request, *args, **kwargs):
        return super().put(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Delete a provider configuration by ID.",
        tags=[PROVIDER_TAG],
        manual_parameters=[AUTH_TOKEN_PARAMETER, openapi.Parameter('id', openapi.IN_PATH, description="Config ID", type=openapi.TYPE_INTEGER)],
    )
    def delete(self, request, *args, **kwargs):
        return super().delete(request, *args, **kwargs)

    def get_queryset(self):
        return LLMProviderConfig.objects.filter(user=self.user)


# Backward-compatible OpenAIKey CRUD using the unified LLM provider configuration
@method_decorator(
    name='post',
    decorator=tokenized_auto_schema(
        tags=[PROVIDER_TAG],
        operation_description="Create an OpenAI provider configuration (legacy route).",
        request_body=LLMProviderConfigSerializer,
        responses={status.HTTP_201_CREATED: LLMProviderConfigSerializer},
    ),
)
class OpenAIKeyCreateAPIView(LLMProviderConfigCreateAPIView):
    """
    Legacy endpoint compatible with former OpenAIKey routes.
    It defaults provider selection to OpenAI to mirror historical behavior.
    """

    def perform_create(self, serializer):
        serializer.save(user=self.user, provider='OpenAI')


@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[PROVIDER_TAG],
        operation_description="List OpenAI provider configurations (legacy route).",
        responses={status.HTTP_200_OK: LLMProviderConfigSerializer(many=True)},
    ),
)
class OpenAIKeyListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = LLMProviderConfigSerializer

    def get_queryset(self):
        return (
            LLMProviderConfig.objects.filter(user=self.user, provider='OpenAI')
            .order_by('-created_at')
        )


@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[PROVIDER_TAG],
        operation_description="Retrieve an OpenAI provider configuration (legacy route).",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Config ID", type=openapi.TYPE_INTEGER)],
        responses={status.HTTP_200_OK: LLMProviderConfigSerializer},
    ),
)
@method_decorator(
    name='put',
    decorator=tokenized_auto_schema(
        tags=[PROVIDER_TAG],
        operation_description="Update an OpenAI provider configuration (legacy route).",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Config ID", type=openapi.TYPE_INTEGER)],
        request_body=LLMProviderConfigSerializer,
        responses={status.HTTP_200_OK: LLMProviderConfigSerializer},
    ),
)
@method_decorator(
    name='delete',
    decorator=tokenized_auto_schema(
        tags=[PROVIDER_TAG],
        operation_description="Delete an OpenAI provider configuration (legacy route).",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Config ID", type=openapi.TYPE_INTEGER)],
    ),
)
class OpenAIKeyRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = LLMProviderConfigSerializer
    lookup_field = 'id'

    def get_queryset(self):
        return LLMProviderConfig.objects.filter(user=self.user, provider='OpenAI')

# VectorStore CRUD
@method_decorator(
    name='post',
    decorator=tokenized_auto_schema(
        tags=[VECTOR_STORE_TAG],
        operation_description="Create a vector store for the active collection.",
        request_body=VectorStoreSerializer,
        responses={status.HTTP_201_CREATED: VectorStoreSerializer},
    ),
)
class VectorStoreCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = VectorStoreSerializer

    def perform_create(self, serializer):
        # Provider validation can be part of the serializer or model validation logic.
        # For example, if 'OpenAI' provider is chosen, an active configuration for the user should exist.
        # This check seems specific, might be better in serializer's validate method.
        # However, keeping it here if it's a quick check based on existing utils.

        # provider = serializer.validated_data.get('provider')
        # logger.info("Provider:", provider)
        # if provider.lower() == 'openai':
        #     # Check if user has a valid, active LLM provider configuration
        #     if not LLMProviderConfig.objects.filter(user=self.user, provider='openai', is_active=True, is_valid=True).exists():
        #         logger.warning(f"User {self.user.id} tried to create OpenAI VectorStore but has no active/valid configuration.")
        #         raise ValidationError("You do not have an active and valid OpenAI API key configured. Please add one before creating an OpenAI vector store.")

        active_collection = get_user_active_collection(self.user)
        instance = serializer.save(tenant=self.user.tenant, user=self.user, collection=active_collection)
        logger.info(f"VectorStore '{instance.name}' (ID: {instance.id}) created by user '{self.user.username}' (ID: {self.user.id}).")

@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[VECTOR_STORE_TAG],
        operation_description="List vector stores for the active collection.",
        responses={status.HTTP_200_OK: VectorStoreSerializer(many=True)},
    ),
)
class VectorStoreListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = VectorStoreSerializer
    def get_queryset(self):
        active_collection = get_user_active_collection(self.user)
        return VectorStore.objects.filter(tenant=self.user.tenant, user=self.user, collection=active_collection).order_by('-created_at')

@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[VECTOR_STORE_TAG],
        operation_description="Retrieve a vector store by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Vector store ID", type=openapi.TYPE_STRING)],
        responses={status.HTTP_200_OK: VectorStoreSerializer},
    ),
)
@method_decorator(
    name='put',
    decorator=tokenized_auto_schema(
        tags=[VECTOR_STORE_TAG],
        operation_description="Update a vector store by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Vector store ID", type=openapi.TYPE_STRING)],
        request_body=VectorStoreSerializer,
        responses={status.HTTP_200_OK: VectorStoreSerializer},
    ),
)
@method_decorator(
    name='delete',
    decorator=tokenized_auto_schema(
        tags=[VECTOR_STORE_TAG],
        operation_description="Delete a vector store by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Vector store ID", type=openapi.TYPE_STRING)],
    ),
)
class VectorStoreRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = VectorStoreSerializer
    lookup_field = 'id'
    def get_queryset(self):
        active_collection = get_user_active_collection(self.user)
        return VectorStore.objects.filter(tenant=self.user.tenant, user=self.user, collection=active_collection)

# Document CRUD
@method_decorator(
    name='post',
    decorator=tokenized_auto_schema(
        tags=[DOCUMENT_TAG],
        operation_description="Ingest a document into the active collection's vector store.",
        request_body=IngestDocumentSerializer,
        responses={status.HTTP_202_ACCEPTED: openapi.Response(description="Ingestion request accepted and processing asynchronously.")},
    ),
)
class IngestAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = IngestDocumentSerializer

    def post(self, request, token): # `token` is from URL, `self.user` is from TokenAuthenticatedMixin
        logger.info(f"IngestAPIView: Received ingestion request from user ID: {self.user.id}, tenant ID: {self.user.tenant.id if self.user.tenant else 'N/A'}")
        tmp_path: Optional[str] = None # Ensure tmp_path is defined for the finally block
        document_id_for_cleanup = None # Track document ID for potential cleanup on failure

        try:
            # Build a shallow, mutable payload without deep-copying uploaded files (which
            # can raise pickling errors when using QueryDict.copy on TemporaryUploadedFile).
            data = {key: request.data.get(key) for key in request.data}
            s3_url_raw = data.get('s3_file_url')
            if s3_url_raw:
                try:
                    from requests.utils import requote_uri
                    s3_url_normalized = requote_uri(s3_url_raw)
                    if s3_url_normalized != s3_url_raw:
                        logger.info(f"Normalized s3_file_url for ingestion: '{s3_url_raw}' -> '{s3_url_normalized}'")
                        data['s3_file_url'] = s3_url_normalized
                except Exception as norm_e:
                    logger.warning(f"Failed to normalize s3_file_url '{s3_url_raw}': {norm_e}")

            serializer = self.serializer_class(data=data)
            if not serializer.is_valid():
                logger.warning(f"Ingestion request failed validation for user ID {self.user.id}. Errors: {serializer.errors}")
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            validated_data = serializer.validated_data
            uploaded_file = validated_data.get('file')
            s3_file_url = validated_data.get('s3_file_url')
            vector_store_id = validated_data['vector_store_id'] # This is required by serializer

            # Tenant check (already implicitly handled by TokenAuthenticatedMixin if user.tenant is required)
            if not self.user.tenant: # Should not happen if mixin enforces tenant association
                logger.error(f"Critical: User {self.user.username} (ID: {self.user.id}) has no associated tenant during ingestion. This should be caught by authentication.")
                return Response({"error": "User has no associated tenant. Cannot process request."}, status=status.HTTP_403_FORBIDDEN)

            active_collection = get_user_active_collection(self.user)
            logger.debug(f"Fetching VectorStore ID: {vector_store_id} for user {self.user.id}, tenant {self.user.tenant.id}")
            vector_store = get_object_or_404(VectorStore, id=vector_store_id, tenant=self.user.tenant, user=self.user, collection=active_collection)
            if vector_store.collection and vector_store.collection_id != active_collection.id:
                return Response(
                    {"error": "Vector store does not belong to the active collection.", "code": "VECTOR_STORE_COLLECTION_MISMATCH"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if vector_store.collection_id is None:
                vector_store.collection = active_collection
                vector_store.save(update_fields=["collection"])
            logger.info(f"Using VectorStore '{vector_store.name}' (ID: {vector_store.id}) for ingestion.")

            collection_name = active_collection.qdrant_collection_name
            collection_exists = False
            expected_dimension = None
            try:
                llm_config = get_llm_config(self.user.id)
                expected_dimension = llm_config.get("embedding_dimension")
            except Exception as e:
                logger.warning("Falling back to settings dimension for collection '%s': %s", collection_name, e)

            try:
                close_connections_before_io("Qdrant collection check")
                call_with_resilience(
                    lambda: get_qdrant_client().get_collection(collection_name=collection_name),
                    service="qdrant_get_collection_view",
                    exceptions=(Exception,),
                )
                collection_exists = True
            except UnexpectedResponse as e:
                if getattr(e, "status_code", None) == 404:
                    initialize_qdrant_collection(
                        collection_name,
                        expected_dimension or getattr(settings, 'QDRANT_VECTOR_DIMENSION', 1536),
                    )
                    collection_exists = True
                else:
                    logger.warning(
                        "Unexpected response while checking collection '%s' for user %s: %s",
                        collection_name,
                        self.user.id,
                        e,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to verify collection '%s' for user %s: %s",
                    collection_name,
                    self.user.id,
                    e,
                )

            if collection_exists and not getattr(self.user, "active_collection_ready", False):
                self.user.active_collection_ready = True
                self.user.is_setup = True
                self.user.active_collection = active_collection
                self.user.llm_configured = True
                if expected_dimension:
                    inferred_provider = infer_provider_from_dimension(expected_dimension)
                    if inferred_provider:
                        self.user.selected_llm_provider = inferred_provider
                self.user.save(update_fields=['active_collection_ready', 'is_setup', 'selected_llm_provider', 'active_collection', 'llm_configured'])

            if collection_exists and not self.user.is_setup:
                default_provider = getattr(settings, 'DEFAULT_LLM_PROVIDER', 'OpenAI')
                if (
                    self.user.selected_llm_provider != default_provider
                    or self.user.is_setup
                ):
                    self.user.selected_llm_provider = default_provider
                self.user.is_setup = False
                self.user.save(update_fields=['selected_llm_provider', 'is_setup'])

            # File processing (downloads or uses uploaded file, saves to tmp_path)
            tmp_path, file_name = process_file(
                file=uploaded_file,
                s3_file_url=s3_file_url,
                file_name=validated_data.get('title') # Allow optional title override
            )
            logger.info(f"File '{file_name}' processed and saved to temporary path: {tmp_path}")

            file_ext = Path(file_name).suffix.lower()
            caption = None # Initialize caption to None as it's generated in background
            thread_started = False # Initialize thread_started flag

            # Create Document object in DB within a transaction
            # (Now status is 'queued', and content/summary will be filled in background)
            with transaction.atomic():
                document = Document(
                    tenant=self.user.tenant,
                    vector_store=vector_store,
                    title=file_name,
                    content="", # Will be filled in background
                    summary="", # Will be filled in background
                    user=self.user,
                    status='queued'
                )
                document.save()
                document_id_for_cleanup = document.id
                logger.info(f"Document object created with ID: {document.id} for file '{file_name}', status: {document.status}.")
            try:
                # Start background ingestion task using a thread
                # (Wrapper now handles extraction, summarization and vector ingestion)
                async_thread = threading.Thread(
                    target=_async_ingest_wrapper,
                    args=(
                        document.id,
                        self.user.id,
                        vector_store.id,
                        active_collection.qdrant_collection_name,
                        tmp_path,
                        file_name,
                        caption,
                        getattr(settings, 'CHUNK_STRATEGY', 'auto'),
                    ),
                    name=f"AsyncIngestThread-Doc{document.id}"
                )
                async_thread.daemon = False
                async_thread.start()
                thread_started = True
                logger.info(f"Async ingestion thread started for document ID: {document.id} ('{file_name}').")
            except Exception as thread_e:
                logger.error(f"Failed to start async thread for document {document.id}: {thread_e}")
                raise RuntimeError(f"Failed to initiate background processing: {thread_e}")

            return Response({
                "message": "File ingestion process started successfully. Processing in background.",
                "file_name": file_name,
                "document_id": document.id,
                "vector_store_id": vector_store.id,
                "status": document.status
            }, status=status.HTTP_202_ACCEPTED)
        
        except DjangoValidationError as e: # Catch Django's validation errors (e.g. from model full_clean or explicit raises)
            logger.warning(f"DjangoValidationError during ingestion for user {self.user.id}: {e.message_dict if hasattr(e, 'message_dict') else e}", exc_info=True)
            # Cleanup DB record if created but ingestion failed
            if document_id_for_cleanup:
                 try:
                     Document.objects.filter(id=document_id_for_cleanup).delete()
                     logger.info(f"Cleaned up failed document record ID: {document_id_for_cleanup}")
                 except Exception as cleanup_e:
                     logger.error(f"Error cleaning up failed document record ID {document_id_for_cleanup}: {cleanup_e}")
            return Response({"error": e.message_dict if hasattr(e, 'message_dict') else str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except ValidationError as e: # Catch DRF's validation errors (from serializer, or explicit raises)
            logger.warning(f"DRF ValidationError during ingestion for user {self.user.id}: {e.detail}", exc_info=True)
            # Cleanup DB record if created but ingestion failed
            if document_id_for_cleanup:
                 try:
                     Document.objects.filter(id=document_id_for_cleanup).delete()
                     logger.info(f"Cleaned up failed document record ID: {document_id_for_cleanup}")
                 except Exception as cleanup_e:
                     logger.error(f"Error cleaning up failed document record ID {document_id_for_cleanup}: {cleanup_e}")
            return Response({"error": e.detail}, status=status.HTTP_400_BAD_REQUEST)
        except RuntimeError as e: # Catch custom RuntimeErrors from utility functions
            logger.error(f"Runtime error during ingestion for user {self.user.id}: {e}", exc_info=True)
            # Cleanup DB record if created but ingestion failed
            if document_id_for_cleanup:
                try:
                     Document.objects.filter(id=document_id_for_cleanup).delete()
                     logger.info(f"Cleaned up failed document record ID: {document_id_for_cleanup}")
                except Exception as cleanup_e:
                     logger.error(f"Error cleaning up failed document record ID {document_id_for_cleanup}: {cleanup_e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e: # Catch any other unexpected errors
            logger.critical(f"Unexpected critical error during ingestion for user {self.user.id}: {e}", exc_info=True)
            # Cleanup DB record if created but ingestion failed
            if document_id_for_cleanup:
                 try:
                     Document.objects.filter(id=document_id_for_cleanup).delete()
                     logger.info(f"Cleaned up failed document record ID: {document_id_for_cleanup}")
                 except Exception as cleanup_e:
                     logger.error(f"Error cleaning up failed document record ID {document_id_for_cleanup}: {cleanup_e}")
            return Response({"error": "An unexpected server error occurred during file ingestion."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            # If the thread didn't start successfully, we must clean up the tmp file here
            if not thread_started and tmp_path and Path(tmp_path).exists():
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                    logger.info(f"Cleaned up temporary file after thread start failure: {tmp_path}")
                except Exception as cleanup_e:
                    logger.error(f"Failed to cleanup {tmp_path} after failure: {cleanup_e}")


@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[DOCUMENT_TAG],
        operation_description="List documents for the authenticated user and collection.",
        responses={status.HTTP_200_OK: DocumentSerializer(many=True)},
    ),
)
class DocumentListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = DocumentSerializer
    def get_queryset(self):
        vector_store_id = self.request.query_params.get('vector_store_id')
        queryset = Document.objects.filter(tenant=self.user.tenant, user=self.user)
        if vector_store_id:
            active_collection = get_user_active_collection(self.user)
            vector_store = get_object_or_404(VectorStore, id=vector_store_id, tenant=self.user.tenant, user=self.user, collection=active_collection)
            queryset = queryset.filter(vector_store=vector_store).order_by('-uploaded_at')
        return queryset

@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[DOCUMENT_TAG],
        operation_description="List document metadata entries.",
        responses={status.HTTP_200_OK: DocumentSerializer(many=True)},
    ),
)
class DocumentMetadataListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = DocumentSerializer
    def get_queryset(self):
        return Document.objects.filter(tenant=self.user.tenant, user=self.user).only('id', 'title', 'uploaded_at').order_by('-uploaded_at')

@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[DOCUMENT_TAG],
        operation_description="Retrieve a document by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Document ID", type=openapi.TYPE_STRING)],
        responses={status.HTTP_200_OK: DocumentSerializer},
    ),
)
@method_decorator(
    name='put',
    decorator=tokenized_auto_schema(
        tags=[DOCUMENT_TAG],
        operation_description="Update a document by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Document ID", type=openapi.TYPE_STRING)],
        request_body=DocumentSerializer,
        responses={status.HTTP_200_OK: DocumentSerializer},
    ),
)
@method_decorator(
    name='delete',
    decorator=tokenized_auto_schema(
        tags=[DOCUMENT_TAG],
        operation_description="Delete a document by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Document ID", type=openapi.TYPE_STRING)],
    ),
)
class DocumentRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = DocumentSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return Document.objects.filter(tenant=self.user.tenant, user=self.user)

    def perform_destroy(self, instance):
        # Use the renamed function and parameter
        active_collection = get_user_active_collection(self.user)
        delete_points_by_document_db_id(document_db_id=str(instance.id), user=self.user, collection_name=active_collection.qdrant_collection_name)
        instance.delete()
        logger.info(f"Document {instance.id} and its associated Qdrant points deleted by user {self.user.id}.")

class DocumentStatusAPIView(TokenAuthenticatedMixin, APIView):
    @swagger_auto_schema(
        operation_description="Retrieve processing status for a document.",
        manual_parameters=[
            AUTH_TOKEN_PARAMETER,
            openapi.Parameter('document_id', openapi.IN_PATH, description="Document ID", type=openapi.TYPE_STRING),
        ],
        tags=[DOCUMENT_TAG],
    )
    def get(self, request, token, document_id):
        try:
            document = get_object_or_404(Document, id=document_id, tenant=self.user.tenant, user=self.user) # Ensure user owns doc or has access
            logger.debug(f"Checking status for document ID: {document_id}, User: {self.user.id}")

            # The verification logic using client.count() is more robust than search with empty query
            active_collection = get_user_active_collection(self.user)
            qdrant_vector_store_instance = get_qdrant_vector_store(self.user, active_collection.qdrant_collection_name)
            qdrant_client = qdrant_vector_store_instance.client

            verification_filter = Filter(must=[
                FieldCondition(key="metadata.document_id", match=MatchValue(value=str(document_id))),
                FieldCondition(key="metadata.vector_store_id", match=MatchValue(value=str(document.vector_store.id))),
                FieldCondition(key="metadata.tenant_id", match=MatchValue(value=str(self.user.tenant.id)))
            ])

            count_response = qdrant_client.count(
                collection_name=active_collection.qdrant_collection_name,
                count_filter=verification_filter,  # Updated from filter to count_filter
                exact=True
            )

            is_verified_in_qdrant = count_response and count_response.count > 0

            # Update status based on Qdrant check AND existing DB status
            # If it's already 'completed' or 'failed' in DB, Qdrant check is a secondary verification.
            # If it's 'queued' or 'processing', Qdrant check helps determine if it's now 'completed'.
            current_db_status = document.status
            final_status = current_db_status

            if current_db_status in ['queued', 'processing']:
                if is_verified_in_qdrant:
                    final_status = 'completed' # Use model choices for status
                else:
                    # If still queued/processing but not in Qdrant, it might still be genuinely pending.
                    # Avoid marking as 'failed' prematurely unless sufficient time has passed.
                    # For a status API, we typically report current state.
                    final_status = 'processing' # Or keep as is
                    logger.info(f"Document {document_id} is in status '{current_db_status}' and not yet fully verified in Qdrant ({count_response.count} points found).")

            elif current_db_status == 'completed' and not is_verified_in_qdrant:
                logger.warning(f"Document {document_id} is 'COMPLETED' in DB but not found or has 0 points in Qdrant. Potential inconsistency.")
                # Optionally, update status_message or trigger a re-verification task.
                # For now, we report the DB status but log the inconsistency.
                final_status = 'error' # Indicate an issue
                # document.status_message = "Inconsistency: Marked completed but not found in vector store."

            if document.status != final_status:
                try:
                    with transaction.atomic():
                        document.status = final_status
                        document.save(update_fields=['status'])
                        logger.debug(f"Updated document {document_id} status from '{current_db_status}' to '{final_status}'")
                except Exception as save_e:
                    logger.error(f"Failed to update document {document_id} status to '{final_status}': {save_e}")
                    # Continue with the old status for response
                    final_status = current_db_status

            logger.info(f"Document {document_id} status reported as: {final_status}. Qdrant points: {count_response.count if count_response else 'N/A'}.")
            return Response({
                "document_id": document_id,
                "status": final_status,
                "qdrant_points": count_response.count if count_response else 0
            }, status=status.HTTP_200_OK)
        except Document.DoesNotExist:
            logger.warning(f"Document status check failed: Document ID {document_id} not found for user {self.user.id}.", exc_info=True)
            return Response({"error": f"Document with ID {document_id} not found."}, status=status.HTTP_404_NOT_FOUND)
        except (RuntimeError, ValueError) as e_qdrant: # Errors from get_qdrant_vector_store
            logger.error(f"Error accessing Qdrant for document {document_id} status check: {e_qdrant}", exc_info=True)
            return Response({"error": f"Failed to check document status due to vector store error: {str(e_qdrant)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            logger.error(f"Unexpected error checking document {document_id} status for user {self.user.id}: {e}", exc_info=True)
            return Response({"error": f"An unexpected error occurred while checking document status: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Assistant CRUD
@method_decorator(
    name='post',
    decorator=tokenized_auto_schema(
        tags=[ASSISTANT_TAG],
        operation_description="Create an assistant for the authenticated user.",
        request_body=AssistantSerializer,
        responses={status.HTTP_201_CREATED: AssistantSerializer},
    ),
)
class AssistantCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = AssistantSerializer
    def perform_create(self, serializer):
        vector_store_id = serializer.validated_data.get('vector_store_id') # This is now an ID, not an object
        vector_store_instance = None
        if vector_store_id:
            active_collection = get_user_active_collection(self.user)
            # Ensure the VectorStore ID provided belongs to the user's tenant
            vector_store_instance = get_object_or_404(VectorStore, id=vector_store_id, tenant=self.user.tenant, user=self.user, collection=active_collection)

        # Save with the actual VectorStore instance
        instance = serializer.save(tenant=self.user.tenant, vector_store=vector_store_instance, creator=self.user)
        logger.info(f"Assistant '{instance.name}' (ID: {instance.id}) created by user '{self.user.username}' (ID: {self.user.id}). VectorStore ID: {vector_store_id if vector_store_instance else 'None'}.")

@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[ASSISTANT_TAG],
        operation_description="List assistants for the authenticated user.",
        responses={status.HTTP_200_OK: AssistantSerializer(many=True)},
    ),
)
class AssistantListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = AssistantSerializer
    def get_queryset(self):
        return Assistant.objects.filter(tenant=self.user.tenant, creator=self.user).order_by('-created_at')

@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[ASSISTANT_TAG],
        operation_description="Retrieve an assistant by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Assistant ID", type=openapi.TYPE_STRING)],
        responses={status.HTTP_200_OK: AssistantSerializer},
    ),
)
@method_decorator(
    name='put',
    decorator=tokenized_auto_schema(
        tags=[ASSISTANT_TAG],
        operation_description="Update an assistant by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Assistant ID", type=openapi.TYPE_STRING)],
        request_body=AssistantSerializer,
        responses={status.HTTP_200_OK: AssistantSerializer},
    ),
)
@method_decorator(
    name='delete',
    decorator=tokenized_auto_schema(
        tags=[ASSISTANT_TAG],
        operation_description="Delete an assistant by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Assistant ID", type=openapi.TYPE_STRING)],
    ),
)
class AssistantRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = AssistantSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return Assistant.objects.filter(tenant=self.user.tenant, creator=self.user)

@method_decorator(
    name='post',
    decorator=tokenized_auto_schema(
        tags=[THREAD_TAG],
        operation_description="Create a thread.",
        request_body=ThreadSerializer,
        responses={status.HTTP_201_CREATED: ThreadSerializer},
    ),
)
class ThreadCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = ThreadSerializer
    def perform_create(self, serializer):
        vector_store_id = serializer.validated_data.get('vector_store_id')  # Optional now
        vector_store_instance = None
        if vector_store_id:
            # Ensure the VectorStore ID provided belongs to the user's tenant
            active_collection = get_user_active_collection(self.user)
            vector_store_instance = get_object_or_404(VectorStore, id=vector_store_id, tenant=self.user.tenant, user=self.user, collection=active_collection)

        instance = serializer.save(tenant=self.user.tenant, vector_store=vector_store_instance, user=self.user)
        logger.info(f"Thread (ID: {instance.id}) created by user '{self.user.username}' (ID: {self.user.id})" + 
                   (f" for VectorStore ID: {vector_store_id}." if vector_store_id else " without vector store."))

@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[THREAD_TAG],
        operation_description="List threads.",
        responses={status.HTTP_200_OK: ThreadSerializer(many=True)},
    ),
)
class ThreadListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = ThreadSerializer
    def get_queryset(self):
        return Thread.objects.filter(tenant=self.user.tenant, user=self.user).order_by('-created_at')

@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[THREAD_TAG],
        operation_description="Retrieve a thread by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Thread ID", type=openapi.TYPE_STRING)],
        responses={status.HTTP_200_OK: ThreadSerializer},
    ),
)
@method_decorator(
    name='put',
    decorator=tokenized_auto_schema(
        tags=[THREAD_TAG],
        operation_description="Update a thread by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Thread ID", type=openapi.TYPE_STRING)],
        request_body=ThreadSerializer,
        responses={status.HTTP_200_OK: ThreadSerializer},
    ),
)
@method_decorator(
    name='delete',
    decorator=tokenized_auto_schema(
        tags=[THREAD_TAG],
        operation_description="Delete a thread by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Thread ID", type=openapi.TYPE_STRING)],
    ),
)
class ThreadRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = ThreadSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return Thread.objects.filter(tenant=self.user.tenant, user=self.user)

class ThreadMessagesAPIView(TokenAuthenticatedMixin, APIView):
    @swagger_auto_schema(
        operation_description="List messages for a specific thread.",
        manual_parameters=[
            AUTH_TOKEN_PARAMETER,
            openapi.Parameter('thread_id', openapi.IN_PATH, description="Thread ID", type=openapi.TYPE_STRING),
        ],
        tags=[THREAD_TAG],
        responses={status.HTTP_200_OK: MessageSerializer(many=True)},
    )
    def get(self, request, token, thread_id):
        try:
            # Ensure the thread belongs to the requesting user and their tenant
            thread = get_object_or_404(Thread, id=thread_id, tenant=self.user.tenant, user=self.user)
            messages = thread.messages.order_by('created_at') # Fetch related messages
            serializer = MessageSerializer(messages, many=True)
            logger.debug(f"Retrieved {len(messages)} messages for thread ID: {thread_id}, user: {self.user.id}")
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Thread.DoesNotExist: # Should be caught by get_object_or_404, but as an example
            logger.warning(f"Thread messages retrieval failed: Thread ID {thread_id} not found for user {self.user.id}.", exc_info=True)
            return Response({"error": f"Thread with ID {thread_id} not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error retrieving messages for thread ID {thread_id}, user {self.user.id}: {e}", exc_info=True)
            return Response({"error": "An unexpected error occurred while retrieving messages."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@method_decorator(
    name='post',
    decorator=tokenized_auto_schema(
        tags=[MESSAGE_TAG],
        operation_description="Create a message in a thread.",
        request_body=MessageSerializer,
        responses={status.HTTP_201_CREATED: MessageSerializer},
    ),
)
class MessageCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = MessageSerializer
    def perform_create(self, serializer):
        thread_id = serializer.validated_data['thread_id'] # This is an ID from request data
        # Ensure the thread belongs to the requesting user and their tenant before adding a message
        thread_instance = get_object_or_404(Thread, id=thread_id, tenant=self.user.tenant, user=self.user)

        # Set role to 'user' by default for messages created via this endpoint
        instance = serializer.save(thread=thread_instance, role='user', user=self.user)
        
        # Generate thread title if this is the first user message and thread doesn't have a title
        if not thread_instance.title:
            try:
                # Check if this is the first user message in the thread
                user_messages_count = thread_instance.messages.filter(role='user').count()
                if user_messages_count == 1:  # This is the first user message
                    from .utils import generate_thread_title
                    title = generate_thread_title(instance.content, self.user)
                    thread_instance.title = title
                    thread_instance.save(update_fields=['title'])
                    logger.info(f"Generated thread title '{title}' for thread {thread_id} based on first user message.")
            except Exception as e:
                logger.warning(f"Failed to generate title for thread {thread_id}: {e}")
                # Set a fallback title
                thread_instance.title = "New Conversation"
                thread_instance.save(update_fields=['title'])
        
        logger.info(f"Message (ID: {instance.id}) created by user '{self.user.username}' (ID: {self.user.id}) in thread ID: {thread_id}.")

@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[MESSAGE_TAG],
        operation_description="List messages for the authenticated user.",
        responses={status.HTTP_200_OK: MessageSerializer(many=True)},
    ),
)
class MessageListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = MessageSerializer
    def get_queryset(self):
        thread_id = self.request.query_params.get('thread_id')
        queryset = Message.objects.filter(user=self.user, thread__tenant=self.user.tenant)
        if thread_id:
            thread = get_object_or_404(Thread, id=thread_id, tenant=self.user.tenant, user=self.user)
            queryset = queryset.filter(thread=thread)
        return queryset

@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[MESSAGE_TAG],
        operation_description="Retrieve a message by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Message ID", type=openapi.TYPE_INTEGER)],
        responses={status.HTTP_200_OK: MessageSerializer},
    ),
)
@method_decorator(
    name='put',
    decorator=tokenized_auto_schema(
        tags=[MESSAGE_TAG],
        operation_description="Update a message by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Message ID", type=openapi.TYPE_INTEGER)],
        request_body=MessageSerializer,
        responses={status.HTTP_200_OK: MessageSerializer},
    ),
)
@method_decorator(
    name='delete',
    decorator=tokenized_auto_schema(
        tags=[MESSAGE_TAG],
        operation_description="Delete a message by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Message ID", type=openapi.TYPE_INTEGER)],
    ),
)
class MessageRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = MessageSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return Message.objects.filter(thread__tenant=self.user.tenant, user=self.user)


@method_decorator(
    name='post',
    decorator=tokenized_auto_schema(
        tags=[MESSAGE_TAG],
        operation_description="Create or update feedback for a message.",
        request_body=MessageFeedbackSerializer,
        responses={
            status.HTTP_200_OK: MessageFeedbackSerializer,
            status.HTTP_201_CREATED: MessageFeedbackSerializer,
        },
    ),
)
class MessageFeedbackCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = MessageFeedbackSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        was_created = getattr(serializer, '_created', True)
        headers = self.get_success_headers(serializer.data)
        status_code = status.HTTP_201_CREATED if was_created else status.HTTP_200_OK
        return Response(serializer.data, status=status_code, headers=headers)


@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[MESSAGE_TAG],
        operation_description="List feedback for messages.",
        responses={status.HTTP_200_OK: MessageFeedbackSerializer(many=True)},
    ),
)
class MessageFeedbackListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = MessageFeedbackSerializer

    def get_queryset(self):
        queryset = MessageFeedback.objects.filter(user=self.user, message__thread__tenant=self.user.tenant)
        message_id = self.request.query_params.get('message_id')
        thread_id = self.request.query_params.get('thread_id')
        if message_id:
            queryset = queryset.filter(message_id=message_id)
        if thread_id:
            queryset = queryset.filter(message__thread_id=thread_id)
        return queryset.order_by('-created_at')

@method_decorator(
    name='post',
    decorator=tokenized_auto_schema(
        tags=[RUN_TAG],
        operation_description="Create a run for a thread.",
        request_body=RunSerializer,
        responses={status.HTTP_201_CREATED: RunSerializer},
    ),
)
class RunCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = RunSerializer

    def perform_create(self, serializer):
        try:
            thread_id = serializer.validated_data['thread_id']
            assistant_id = serializer.validated_data.get('assistant_id')
            thread = get_object_or_404(Thread, id=thread_id, tenant=self.user.tenant)
            if assistant_id:
                assistant = get_object_or_404(Assistant, id=assistant_id, tenant=self.user.tenant)
            else:
                assistant = (
                    Assistant.objects.filter(tenant=self.user.tenant, is_default=True)
                    .order_by('-created_at')
                    .first()
                )
                if not assistant:
                    raise ValidationError(
                        {"assistant_id": "No assistant specified and no default assistant is configured."}
                    )

            try:
                llm_config = get_llm_config(self.user.id)
            except Exception as exc:
                raise ValidationError({"assistant_id": f"LLM configuration error: {exc}"})

            user_provider = normalize_provider_name(
                getattr(self.user, "selected_llm_provider", getattr(settings, "DEFAULT_LLM_PROVIDER", "OpenAI"))
            )
            assistant_provider = infer_provider_from_model(assistant.model, user_provider)
            if assistant_provider != user_provider:
                raise ValidationError(
                    {
                        "assistant_id": (
                            f"User default LLM is {user_provider}; assistant model '{assistant.model}' cannot be used."
                        )
                    }
                )

            try:
                validate_model_availability(
                    provider=user_provider,
                    model=assistant.model,
                    api_key=llm_config.get("api_key"),
                    base_url=llm_config.get("base_url"),
                )
            except ValueError as exc:
                raise ValidationError({"assistant_id": str(exc)})

            request_data = serializer.context['request'].data

            message_id = serializer.validated_data.pop('message_id', None)
            source_run_id = serializer.validated_data.pop('source_run_id', None)

            messages = thread.messages.order_by('created_at')
            user_message_objs = [msg for msg in messages if msg.role == 'user']

            if not user_message_objs and not message_id and not request_data.get('queries'):
                raise ValidationError({"error": "No user messages found in the thread."})

            source_message = None
            rerun_of = None

            if message_id is not None:
                message = get_object_or_404(Message, id=message_id, thread=thread)
                if message.role != 'user':
                    raise ValidationError({"message_id": "Only user messages can be rerun."})
                if message.thread.user != self.user:
                    raise ValidationError({"message_id": "You can only rerun your own messages."})
                source_message = message

            if source_run_id is not None:
                rerun_of = get_object_or_404(Run, id=source_run_id, thread=thread)
                if rerun_of.assistant != assistant:
                    raise ValidationError({"source_run_id": "Run must belong to the same assistant."})
                if rerun_of.thread.user != self.user:
                    raise ValidationError({"source_run_id": "You can only rerun runs from your own threads."})
                if source_message is None and rerun_of.source_message is not None:
                    source_message = rerun_of.source_message

            queries = None
            if source_message is not None:
                queries = [source_message.content]

            requested_queries = request_data.get('queries')
            if requested_queries and source_message is None:
                if not isinstance(requested_queries, list):
                    requested_queries = [requested_queries]
                queries = requested_queries

            if queries is None:
                if user_message_objs:
                    source_message = source_message or user_message_objs[-1]
                    queries = [source_message.content]
                else:
                    raise ValidationError({"error": "No user messages found in the thread."})

            sanitized_queries = []
            for q in queries:
                if not isinstance(q, str):
                    raise ValidationError({"error": "Invalid query content."})
                if len(q) > 3500 or any(term in q.lower() for term in PROHIBITED_TERMS):
                    raise ValidationError({"error": "Invalid query content or Query too long."})
                timestamps = _rate_tracker.get(self.user.id, [])
                timestamps = [t for t in timestamps if (datetime.now() - t).seconds < 60]
                if len(timestamps) >= RATE_LIMIT:
                    raise ValidationError({"error": "Rate limit exceeded."})
                timestamps.append(datetime.now())
                _rate_tracker[self.user.id] = timestamps
                sanitized_queries.append(q)
            queries = sanitized_queries

            vector_store = thread.vector_store
            # Set mode to "normal" if thread has no vector_store
            if not vector_store:
                run_mode = "normal"
                logger.info(f"Run for thread {thread_id}, assistant {assistant_id} will use mode: {run_mode} (no vector store)")
            else:
                # Get mode from serializer validated_data or request_data, default to "document"
                run_mode = serializer.validated_data.get('mode') or request_data.get('mode') or "document"
                logger.info(f"Run for thread {thread_id}, assistant {assistant_id} will use vector_store_id: {vector_store.id} (Name: {vector_store.name}), mode: {run_mode}")

            run = serializer.save(
                thread=thread,
                assistant=assistant,
                status=Run.STATUS_QUEUED,
                source_message=source_message,
                rerun_of=rerun_of,
                mode=run_mode,
            )

            if vector_store:
                logger.info(f"Run {run.id} created and queued for processing. Vector Store ID: {vector_store.id}")
                metadata_filters = serializer.context['request'].data.get('filters')
                threading.Thread(target=process_run, args=(run, queries, vector_store.id, run_mode, metadata_filters)).start()
            else:
                logger.info(f"Run {run.id} created and queued for processing. Mode: {run_mode} (no vector store)")
                # For normal mode without vector store, we still need to process the run
                # Use empty string or None for vector_store_id - process_run should handle this
                metadata_filters = serializer.context['request'].data.get('filters')
                threading.Thread(target=process_run, args=(run, queries, "", run_mode, metadata_filters)).start()
            ##Logger.info(f"Started background processing for run {run.id}")
        except ValidationError as e:
            logger.error(f"Run creation validation error for user {self.user.id}, thread {thread_id}, assistant {assistant_id}: {e.detail if hasattr(e, 'detail') else e}", exc_info=True)
            raise # Re-raise the validation error to be handled by DRF
        except Thread.DoesNotExist as e: # More specific than just Exception for GetObjectOr404 errors
            logger.warning(f"Run creation failed for user {self.user.id}: Could not find Thread. Error: {e}", exc_info=True)
            raise ValidationError(f"Could not find the specified thread: {str(e)}") # Return 400
        except Assistant.DoesNotExist as e:
            logger.warning(f"Run creation failed for user {self.user.id}: Could not find Assistant. Error: {e}", exc_info=True)
            raise ValidationError(f"Could not find the specified assistant: {str(e)}") # Return 400
        except Exception as e:
            logger.critical(f"Unexpected critical error creating run for user {self.user.id}, thread {thread_id}, assistant {assistant_id}: {e}", exc_info=True)
            # For truly unexpected errors, a 500 might be more appropriate than ValidationError
            # However, to keep client handling simple, ValidationError can be used if preferred.
            # For now, let's make it a generic validation error for the client.
            raise ValidationError({"error": f"An unexpected server error occurred while creating the run: {str(e)}"})


@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[RUN_TAG],
        operation_description="List runs for the authenticated user.",
        responses={status.HTTP_200_OK: RunSerializer(many=True)},
    ),
)
class RunListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = RunSerializer
    def get_queryset(self):
        thread_id = self.request.query_params.get('thread_id')
        queryset = Run.objects.filter(thread__tenant=self.user.tenant)
        if thread_id:
            thread = get_object_or_404(Thread, id=thread_id, tenant=self.user.tenant)
            queryset = queryset.filter(thread=thread)
        return queryset

@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[RUN_TAG],
        operation_description="Retrieve a run by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Run ID", type=openapi.TYPE_STRING)],
        responses={status.HTTP_200_OK: RunSerializer},
    ),
)
@method_decorator(
    name='put',
    decorator=tokenized_auto_schema(
        tags=[RUN_TAG],
        operation_description="Update a run by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Run ID", type=openapi.TYPE_STRING)],
        request_body=RunSerializer,
        responses={status.HTTP_200_OK: RunSerializer},
    ),
)
@method_decorator(
    name='delete',
    decorator=tokenized_auto_schema(
        tags=[RUN_TAG],
        operation_description="Delete a run by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Run ID", type=openapi.TYPE_STRING)],
    ),
)
class RunRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = RunSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return Run.objects.filter(thread__tenant=self.user.tenant)


class RunCancelAPIView(TokenAuthenticatedMixin, APIView):
    @swagger_auto_schema(
        operation_description="Cancel a run if it is still pending or in progress.",
        manual_parameters=[
            AUTH_TOKEN_PARAMETER,
            openapi.Parameter('run_id', openapi.IN_PATH, description="Run ID", type=openapi.TYPE_STRING),
        ],
        tags=[RUN_TAG],
        responses={status.HTTP_200_OK: RunSerializer},
    )
    def post(self, request, token=None, run_id=None):
        run = get_object_or_404(Run, id=run_id, thread__tenant=self.user.tenant)
        if run.thread.user != self.user:
            return Response({"error": "You can only cancel runs from your own threads."}, status=status.HTTP_403_FORBIDDEN)

        if run.status in Run.TERMINAL_STATUSES:
            return Response({"error": "Run is already finished and cannot be cancelled."}, status=status.HTTP_400_BAD_REQUEST)

        with transaction.atomic():
            run.status = Run.STATUS_CANCELLED
            now = timezone.now()
            run.cancelled_at = now
            run.completed_at = now
            run.required_action = None
            run.tool_outputs = None
            run.save(update_fields=['status', 'cancelled_at', 'completed_at', 'required_action', 'tool_outputs'])

        logger.info(f"Run {run.id} cancelled by user {self.user.id}.")
        return Response({"message": "Run cancelled successfully."}, status=status.HTTP_200_OK)


class RunRerunAPIView(RunCreateAPIView):
    @swagger_auto_schema(
        operation_description="Rerun a completed or failed run.",
        manual_parameters=[
            AUTH_TOKEN_PARAMETER,
            openapi.Parameter('run_id', openapi.IN_PATH, description="Run ID", type=openapi.TYPE_STRING),
        ],
        tags=[RUN_TAG],
        request_body=RunSerializer,
        responses={status.HTTP_201_CREATED: RunSerializer},
    )
    def post(self, request, *args, **kwargs):
        run_id = kwargs.get('run_id')
        original_run = get_object_or_404(Run, id=run_id, thread__tenant=self.user.tenant)
        if original_run.thread.user != self.user:
            return Response({"error": "You can only rerun runs from your own threads."}, status=status.HTTP_403_FORBIDDEN)

        data = request.data.copy()
        data['thread_id'] = str(original_run.thread.id)
        data['assistant_id'] = str(original_run.assistant.id)
        data.setdefault('mode', original_run.mode)
        data['source_run_id'] = original_run.id

        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


class SubmitToolOutputsAPIView(TokenAuthenticatedMixin, APIView):
    @swagger_auto_schema(
        operation_description="Submit tool outputs for a pending run step.",
        manual_parameters=[
            AUTH_TOKEN_PARAMETER,
            openapi.Parameter('run_id', openapi.IN_PATH, description="Run ID", type=openapi.TYPE_STRING),
        ],
        tags=[RUN_TAG],
        request_body=SubmitToolOutputsSerializer,
        responses={status.HTTP_200_OK: RunSerializer},
    )
    def post(self, request, run_id, token=None):
        run = get_object_or_404(Run, id=run_id, thread__tenant=self.user.tenant)
        if run.status != Run.STATUS_REQUIRES_ACTION:
            return Response({"error": "Run is not in requires_action state."}, status=status.HTTP_400_BAD_REQUEST)

        serializer = SubmitToolOutputsSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        tool_outputs = serializer.validated_data['tool_outputs']

        # Validate tool_outputs match required tool_calls
        required_calls = run.required_action.get('submit_tool_outputs', {}).get('tool_calls', [])
        required_ids = {call['id'] for call in required_calls}
        submitted_ids = {output['tool_call_id'] for output in tool_outputs}
        if submitted_ids != required_ids:
            return Response({"error": "Submitted tool outputs do not match required tool calls."}, status=status.HTTP_400_BAD_REQUEST)

        # Save tool_outputs and resume run
        run.tool_outputs = tool_outputs
        run.status = Run.STATUS_IN_PROGRESS
        run.save(update_fields=['tool_outputs', 'status'])

        # Get the last user query for process_run
        messages = run.thread.messages.order_by('created_at')
        user_messages = [msg.content for msg in messages if msg.role == 'user']
        queries = [user_messages[-1]] if user_messages else ["Continue processing with submitted tool outputs"]

        # Determine vector store ID
        vector_store_id = None

        if run.thread.vector_store:
            vector_store_id = str(run.thread.vector_store.id)
            logger.info(f"Run {run.id}: Using vector store associated with thread: {vector_store_id}.")
        else:
            logger.error(f"Run {run.id}: No vector store associated with thread or tenant.")
            run.status = 'failed'
            run.completed_at = timezone.now()
            run.save(update_fields=['status', 'completed_at'])
            Message.objects.create(
                thread=run.thread,
                role='assistant',
                content="Run processing failed: No vector store available for this thread or tenant.",
                user=None,
                metadata={"used_document_ids": []}
            )
            return Response({"error": "No vector store available for this thread or tenant."}, status=status.HTTP_400_BAD_REQUEST)

        threading.Thread(target=process_run, args=(run, queries, vector_store_id, run.mode, None)).start()
        return Response({"message": "Tool outputs submitted successfully. Run resuming."}, status=status.HTTP_200_OK)
    

@method_decorator(
    name='post',
    decorator=tokenized_auto_schema(
        tags=[ALERT_TAG],
        operation_description="Create a document alert.",
        request_body=DocumentAlertSerializer,
        responses={status.HTTP_201_CREATED: DocumentAlertSerializer},
    ),
)
class DocumentAlertCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = DocumentAlertSerializer
    def perform_create(self, serializer):
        document_id = serializer.validated_data['document'].id # Assuming document is validated to an instance by serializer
        # Ensure the document belongs to the user's tenant. User creating alert must be the same as self.user.
        document_instance = get_object_or_404(Document, id=document_id, tenant=self.user.tenant, user=self.user)

        instance = serializer.save(document=document_instance, tenant=self.user.tenant, user=self.user)
        logger.info(f"DocumentAlert (ID: {instance.id}) created by user '{self.user.username}' (ID: {self.user.id}) for Document ID: {document_id}.")

@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[ALERT_TAG],
        operation_description="List document alerts.",
        responses={status.HTTP_200_OK: DocumentAlertSerializer(many=True)},
    ),
)
class DocumentAlertListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = DocumentAlertSerializer
    def get_queryset(self):
        document_id = self.request.query_params.get('document_id')
        queryset = DocumentAlert.objects.filter(user=self.user, document__tenant=self.user.tenant)
        if document_id:
            document = get_object_or_404(Document, id=document_id, tenant=self.user.tenant, user=self.user)
            queryset = queryset.filter(document=document)
        return queryset

@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[ALERT_TAG],
        operation_description="Retrieve a document alert by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Alert ID", type=openapi.TYPE_INTEGER)],
        responses={status.HTTP_200_OK: DocumentAlertSerializer},
    ),
)
@method_decorator(
    name='put',
    decorator=tokenized_auto_schema(
        tags=[ALERT_TAG],
        operation_description="Update a document alert by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Alert ID", type=openapi.TYPE_INTEGER)],
        request_body=DocumentAlertSerializer,
        responses={status.HTTP_200_OK: DocumentAlertSerializer},
    ),
)
@method_decorator(
    name='delete',
    decorator=tokenized_auto_schema(
        tags=[ALERT_TAG],
        operation_description="Delete a document alert by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Alert ID", type=openapi.TYPE_INTEGER)],
    ),
)
class DocumentAlertRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = DocumentAlertSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return DocumentAlert.objects.filter(document__tenant=self.user.tenant, user=self.user)

@method_decorator(
    name='post',
    decorator=tokenized_auto_schema(
        tags=[ACCESS_TAG],
        operation_description="Grant or update document access.",
        request_body=DocumentAccessSerializer,
        responses={status.HTTP_201_CREATED: DocumentAccessSerializer},
    ),
)
class DocumentAccessCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = DocumentAccessSerializer

    def perform_create(self, serializer):
        try:
            serializer.is_valid(raise_exception=True)
            # Serializer validation should ensure these are valid instances and belong to the user/tenant
            vector_store_instance = serializer.validated_data['vector_store']
            document_instances = serializer.validated_data['documents'] # This should be a list of Document instances

            # Ensure granted_by is the authenticated user
            granted_by_user = self.user
            # Ensure all documents and the vector store belong to the user's tenant
            if vector_store_instance.tenant != self.user.tenant:
                logger.warning(f"User {self.user.id} trying to grant access to VectorStore {vector_store_instance.id} not in their tenant.")
                raise ValidationError("VectorStore does not belong to your tenant.")

            for doc_instance in document_instances:
                if doc_instance.tenant != self.user.tenant:
                    logger.warning(f"User {self.user.id} trying to grant access for Document {doc_instance.id} not in their tenant.")
                    raise ValidationError(f"Document '{doc_instance.title}' does not belong to your tenant.")

            created_access_details = []
            for doc_instance in document_instances:
                # Create DocumentAccess, ensuring it's also linked to the tenant
                access, created = DocumentAccess.objects.update_or_create(
                    document=doc_instance,
                    vector_store=vector_store_instance,
                    defaults={'granted_by': granted_by_user}
                )
                status_log = "created" if created else "updated"
                logger.info(f"DocumentAccess {status_log} for Document ID {doc_instance.id} to VectorStore ID {vector_store_instance.id} by User ID {granted_by_user.id}.")
                created_access_details.append({
                    "document_access_id": access.id,
                    "document_id": str(doc_instance.id),
                    "vector_store_id": str(vector_store_instance.id),
                    "status": status_log
                })

            return Response({
                "message": f"Access link processed for {len(created_access_details)} documents in vector store '{vector_store_instance.name}'.",
                "access_details": created_access_details
            }, status=status.HTTP_201_CREATED if any(d['status'] == 'created' for d in created_access_details) else status.HTTP_200_OK)

        except ValidationError as e: # Catch DRF validation errors
            logger.warning(f"Validation error during DocumentAccess creation for user {self.user.id}: {e.detail}", exc_info=True)
            raise # Re-raise for DRF to handle
        except Exception as e: # Catch other unexpected errors
            logger.error(f"Unexpected error granting document access for user {self.user.id}: {e}", exc_info=True)
            # Use DRF's ValidationError to return a 400 to the client for consistency
            raise ValidationError({"detail": f"An unexpected error occurred: {str(e)}"})


@method_decorator(
    name='put',
    decorator=tokenized_auto_schema(
        tags=[ACCESS_TAG],
        operation_description="Remove document access entries for a user and vector store.",
        request_body=DocumentAccessRemoveSerializer,
        responses={status.HTTP_200_OK: DocumentAccessRemoveSerializer},
    ),
)
class DocumentAccessRemoveAPIView(TokenAuthenticatedMixin, generics.UpdateAPIView): # Should likely be a DELETE or custom action
    serializer_class = DocumentAccessRemoveSerializer

    def put(self, request, *args, **kwargs):
        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            # Assuming serializer validates that vector_store and document_ids are provided and are of correct type
            vector_store_instance = serializer.validated_data['vector_store']
            document_ids_to_remove = serializer.validated_data['document_ids'] # Changed from valid_document_ids for clarity

            # Ensure vector_store belongs to the user's tenant
            if vector_store_instance.tenant != self.user.tenant:
                logger.warning(f"User {self.user.id} trying to remove access from VectorStore {vector_store_instance.id} not in their tenant.")
                raise ValidationError("VectorStore does not belong to your tenant.")

            # Filter DocumentAccess records for deletion
            # This assumes DocumentAccess links a Document to a VectorStore for a specific User (grantee)
            # and was granted by someone (granted_by).
            # If removal means "this user can no longer access these documents via this vector store":
            access_records_to_delete = DocumentAccess.objects.filter(
                vector_store=vector_store_instance,
                document_id__in=document_ids_to_remove, # document_id__in expects a list of IDs
                user=self.user, # Current user is the one whose access is being revoked
                tenant=self.user.tenant # Ensure tenant consistency
            )

            deleted_count, _ = access_records_to_delete.delete() # delete() returns (total, dict_of_deletions_per_type)

            if deleted_count == 0:
                logger.info(f"No DocumentAccess records found to remove for User {self.user.id}, VectorStore {vector_store_instance.id}, and specified document IDs.")
                return Response({
                    "message": "No matching access records found to remove for the specified documents in this vector store for your user."
                }, status=status.HTTP_200_OK) # Or 404 if no records found is an error

            logger.info(f"Removed {deleted_count} DocumentAccess record(s) for User {self.user.id} from VectorStore '{vector_store_instance.name}' (ID: {vector_store_instance.id}).")
            return Response({
                "message": f"Successfully removed access for {deleted_count} document(s) from the vector store for your user.",
                "removed_document_ids_count": deleted_count # Consider returning actual IDs if needed, but count is simpler
            }, status=status.HTTP_200_OK)

        except ValidationError as e: # Catch DRF validation errors
            logger.warning(f"Validation error during DocumentAccess removal for user {self.user.id}: {e.detail}", exc_info=True)
            raise # Re-raise for DRF to handle
        except Exception as e: # Catch other unexpected errors
            logger.error(f"Unexpected error removing document access for user {self.user.id}: {e}", exc_info=True)
            raise ValidationError({"detail": f"An unexpected error occurred: {str(e)}"})


@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[ACCESS_TAG],
        operation_description="List document access grants created by the user.",
        responses={status.HTTP_200_OK: DocumentAccessListSerializer(many=True)},
    ),
)
class DocumentAccessListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = DocumentAccessListSerializer

    def get_queryset(self):
        logger.debug(f"User {self.user.id} listing DocumentAccess records they granted.")
        return DocumentAccess.objects.filter(granted_by=self.user).order_by('-updated_at')


@method_decorator(
    name='get',
    decorator=tokenized_auto_schema(
        tags=[ACCESS_TAG],
        operation_description="Retrieve a document access grant by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Access ID", type=openapi.TYPE_INTEGER)],
        responses={status.HTTP_200_OK: DocumentAccessSerializer},
    ),
)
@method_decorator(
    name='put',
    decorator=tokenized_auto_schema(
        tags=[ACCESS_TAG],
        operation_description="Update a document access grant by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Access ID", type=openapi.TYPE_INTEGER)],
        request_body=DocumentAccessSerializer,
        responses={status.HTTP_200_OK: DocumentAccessSerializer},
    ),
)
@method_decorator(
    name='delete',
    decorator=tokenized_auto_schema(
        tags=[ACCESS_TAG],
        operation_description="Delete a document access grant by ID.",
        manual_parameters=[openapi.Parameter('id', openapi.IN_PATH, description="Access ID", type=openapi.TYPE_INTEGER)],
    ),
)
class DocumentAccessRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = DocumentAccessSerializer # Might need adjustment for update
    lookup_field = 'id'

    def get_queryset(self):
        # This allows management of DocumentAccess objects granted *by* the current user.
        logger.debug(f"User {self.user.id} accessing DocumentAccess records they granted via retrieve/update/destroy.")
        # return DocumentAccess.objects.filter(granted_by=self.user, tenant=self.user.tenant)
        return DocumentAccess.objects.filter(granted_by=self.user, document__tenant=self.user.tenant)

    def perform_destroy(self, instance):
        logger.info(f"User {self.user.id} deleting DocumentAccess record ID: {instance.id} (Doc: {instance.document_id}, VS: {instance.vector_store_id}).")
        super().perform_destroy(instance)

    def perform_update(self, serializer):
        instance = serializer.save()
        logger.info(f"User {self.user.id} updated DocumentAccess record ID: {instance.id}.")


# OpenAI Responses API Endpoints

class ConversationCreateAPIView(TokenAuthenticatedMixin, APIView):
    """Create a new conversation without requiring vector_store."""

    @swagger_auto_schema(
        operation_description="Create a conversation with an optional title.",
        manual_parameters=[
            openapi.Parameter(
                'token', openapi.IN_PATH, description="Authentication token", type=openapi.TYPE_STRING
            ),
        ],
        tags=[CONVERSATION_TAG],
        request_body=ConversationCreateRequestSerializer,
        responses={status.HTTP_201_CREATED: ConversationCreateResponseSerializer},
    )
    def post(self, request, token):
        try:
            title = request.data.get("title")
            # Create conversation
            conversation = Conversation.objects.create(
                tenant=self.user.tenant,
                user=self.user,
                title=title,
            )
            logger.info(f"Conversation (ID: {conversation.id}) created by user '{self.user.username}' (ID: {self.user.id})")
            return Response({
                "id": conversation.id,
                "created_at": conversation.created_at.isoformat(),
                "title": conversation.title,
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Error creating conversation for user {self.user.id}: {e}", exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ConversationRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    """Retrieve, update, or delete a conversation."""

    serializer_class = ConversationSerializer
    lookup_field = 'id'

    @swagger_auto_schema(
        operation_description="Retrieve a specific conversation by ID.",
        manual_parameters=[
            openapi.Parameter('token', openapi.IN_PATH, description="Authentication token", type=openapi.TYPE_STRING),
            openapi.Parameter('id', openapi.IN_PATH, description="Conversation ID", type=openapi.TYPE_STRING),
        ],
        tags=[CONVERSATION_TAG],
    )
    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Update an existing conversation.",
        manual_parameters=[
            openapi.Parameter('token', openapi.IN_PATH, description="Authentication token", type=openapi.TYPE_STRING),
            openapi.Parameter('id', openapi.IN_PATH, description="Conversation ID", type=openapi.TYPE_STRING),
        ],
        tags=[CONVERSATION_TAG],
        request_body=ConversationSerializer,
    )
    def put(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)

    @swagger_auto_schema(auto_schema=None)
    def patch(self, request, *args, **kwargs):
        return self.partial_update(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Delete a conversation.",
        manual_parameters=[
            openapi.Parameter('token', openapi.IN_PATH, description="Authentication token", type=openapi.TYPE_STRING),
            openapi.Parameter('id', openapi.IN_PATH, description="Conversation ID", type=openapi.TYPE_STRING),
        ],
        tags=[CONVERSATION_TAG],
    )
    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)

    def get_queryset(self):
        return Conversation.objects.filter(tenant=self.user.tenant, user=self.user)

    def perform_destroy(self, instance):
        logger.info(f"User {self.user.id} deleting conversation {instance.id}")
        super().perform_destroy(instance)


class ResponsesAPIView(TokenAuthenticatedMixin, APIView):
    """Generate responses using LLM providers with optional conversation storage."""

    @swagger_auto_schema(
        operation_description=(
            "Generate a chat response using the configured LLM provider. The request supports optional metadata, "
            "document tools, and conversation persistence so Redoc displays all accepted fields."
        ),
        manual_parameters=[
            openapi.Parameter(
                'token', openapi.IN_PATH, description="Authentication token", type=openapi.TYPE_STRING
            ),
        ],
        tags=[RESPONSE_TAG],
        request_body=ResponsesCreateSerializer,
        responses={status.HTTP_200_OK: ResponsesOutputSerializer},
    )
    def post(self, request, token):
        try:
            serializer = ResponsesCreateSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
            validated_data = serializer.validated_data
            model = validated_data['model']
            conversation_id = validated_data.get('conversation')
            instructions = validated_data.get('instructions', '')
            input_data = validated_data['input']
            tools = validated_data.get('tools')
            meta_data = validated_data.get('metadata', {})
            logger.debug(f"Request Metadata: {meta_data}")
            response_record = None
            
            # Extract user messages from input with metadata
            user_messages = []
            user_messages_metadata = []
            for item in input_data:
                if item.get('role') == 'user':
                    content = item.get('content', [])
                    item_metadata = item.get('metadata', {})
                    # Handle content as array or string
                    if isinstance(content, list):
                        text_content = ""
                        for content_item in content:
                            if isinstance(content_item, dict):
                                if content_item.get('type') == 'input_text':
                                    text_content += content_item.get('text', '')
                                elif content_item.get('type') == 'text':
                                    text_content += content_item.get('text', '')
                            elif isinstance(content_item, str):
                                text_content += content_item
                        user_messages.append(text_content)
                        user_messages_metadata.append(item_metadata)
                    elif isinstance(content, str):
                        user_messages.append(content)
                        user_messages_metadata.append(item_metadata)
            
            if not user_messages:
                return Response({"error": "No user messages found in input"}, status=status.HTTP_400_BAD_REQUEST)
            
            # Validate tools if present
            vector_store_ids = []
            if tools:
                active_collection = get_user_active_collection(self.user)
                for tool in tools:
                    if tool.get('type') != 'document':
                        return Response(
                            {"error": f"Invalid tool type '{tool.get('type')}'. Only 'document' type is allowed."},
                            status=status.HTTP_400_BAD_REQUEST
                        )
                    tool_vector_store_ids = tool.get('vector_store_ids', [])
                    if not tool_vector_store_ids:
                        return Response(
                            {"error": "Tool of type 'document' must include 'vector_store_ids' field."},
                            status=status.HTTP_400_BAD_REQUEST
                        )
                    # Validate vector_store_ids belong to user in bulk
                    valid_vector_stores = VectorStore.objects.filter(
                        id__in=tool_vector_store_ids,
                        tenant=self.user.tenant,
                        user=self.user,
                        collection=active_collection
                    ).values_list('id', flat=True)
                    
                    # Convert to UUID or string depending on your system's ID type
                    # Assuming string/UUID comparison is fine here
                    valid_vs_ids = set(str(vs_id) for vs_id in valid_vector_stores)
                    for vs_id in tool_vector_store_ids:
                        if str(vs_id) not in valid_vs_ids:
                            return Response(
                                {"error": f"Vector store '{vs_id}' not found or does not belong to user."},
                                status=status.HTTP_400_BAD_REQUEST
                            )
                        vector_store_ids.append(vs_id)
            
            # Get LLM configuration
            llm_config = get_llm_config(self.user.id)
            provider_name = normalize_provider_name(llm_config.get("provider_name"))
            provider_instance = llm_config["provider_instance"]
            chat_model = model or llm_config.get("chat_model")
            api_key = llm_config.get("api_key")
            base_url = llm_config.get("base_url")

            if not chat_model:
                return Response(
                    {"error": "LLM model is required and was not provided or configured."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Ensure the requested model aligns with the user's configured LLM provider
            model_provider = infer_provider_from_model(chat_model, provider_name)
            if model_provider != provider_name:
                return Response(
                    {
                        "error": (
                            f"User default LLM is {provider_name}; model '{chat_model}' cannot be used with a different provider."
                        )
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if not provider_instance:
                if provider_name == "OpenAI":
                    provider_instance = OpenAIProvider()
                    api_key = getattr(settings, "OPENAI_API_KEY", None)
                    chat_model = chat_model or getattr(settings, "DEFAULT_OPENAI_MODEL", "gpt-4.1")
                elif provider_name == "Ollama":
                    base_url = base_url or getattr(
                        settings, "OLLAMA_API_URL", getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
                    )
                    provider_instance = OllamaProvider(base_url=base_url)
                    api_key = None

            if not provider_instance:
                return Response(
                    {"error": "LLM provider not configured."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            try:
                validate_model_availability(
                    provider=provider_name,
                    model=chat_model,
                    api_key=api_key,
                    base_url=base_url,
                )
            except ValueError as exc:
                return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

            # Prepare messages for LLM
            messages = []
            
            # Get conversation history if conversation_id provided
            conversation = None
            conversation_history = []
            if conversation_id:
                try:
                    conversation = Conversation.objects.get(id=conversation_id, tenant=self.user.tenant, user=self.user)
                    existing_messages = conversation.messages.order_by('created_at')
                    conversation_history = [{"role": msg.role, "content": msg.content} for msg in existing_messages]
                    messages.extend(conversation_history)
                except Conversation.DoesNotExist:
                    return Response(
                        {"error": f"Conversation '{conversation_id}' not found."},
                        status=status.HTTP_404_NOT_FOUND
                    )
            
            # Add user messages
            for user_msg in user_messages:
                messages.append({"role": "user", "content": user_msg})

            if conversation and not conversation.title and user_messages:
                conversation.title = user_messages[0][:255]
                conversation.save(update_fields=["title", "updated_at"])

            # Persist response record for tracking
            response_record = ResponseRecord.objects.create(
                conversation=conversation,
                tenant=self.user.tenant,
                user=self.user,
                model=chat_model,
                instructions=instructions or "",
                input_messages=input_data,
                metadata=meta_data,
                status=ResponseRecord.STATUS_IN_PROGRESS,
            )
            
            # Perform Qdrant search if tools provided
            documents = None
            if tools and vector_store_ids:
                try:
                    active_collection = get_user_active_collection(self.user)
                    collection_name = active_collection.qdrant_collection_name
                    
                    # Combine all vector_store_ids for search with consolidated query
                    all_accessible_doc_ids = get_all_accessible_document_ids(self.user, vector_store_ids)
                    
                    if all_accessible_doc_ids:
                        accessible_doc_ids_str = [str(doc_id) for doc_id in set(all_accessible_doc_ids)]
                        vector_store_ids_str = [str(vs_id) for vs_id in vector_store_ids]
                        must_conditions = [
                            FieldCondition(key="metadata.tenant_id", match=MatchValue(value=str(self.user.tenant.id))),
                            FieldCondition(key="metadata.document_id", match=MatchAny(any=accessible_doc_ids_str)),
                        ]
                        
                        search_filter = Filter(must=must_conditions)
                        vector_store = get_qdrant_vector_store(self.user, collection_name)
                        
                        # Use the last user message for search
                        search_query = user_messages[-1] if user_messages else ""
                        # Use the standardized HybridSearchRetriever
                        retriever = HybridSearchRetriever(
                            vector_store=vector_store,
                            search_filter=search_filter,
                            k=25
                        )
                        search_results = retriever.invoke(search_query)
                        documents = search_results
                except VectorStoreConnectionError as vse:
                    logger.error(f"Vector store connection error: {vse}")
                    if response_record:
                        response_record.status = ResponseRecord.STATUS_FAILED
                        response_record.error_message = str(vse)
                        response_record.save(update_fields=["status", "error_message", "updated_at"])
                    return Response(
                        {"error": f"Vector store unavailable: {str(vse)}"},
                        status=status.HTTP_503_SERVICE_UNAVAILABLE
                    )
                except Exception as e:
                    logger.error(f"Error performing Qdrant search: {e}", exc_info=True)
                    # Continue without documents if search fails genericly (or validation error)
            
            # Generate response using LLM
            response_start_time = response_record.created_at if response_record else timezone.now()
            # Track available document IDs for validation
            available_doc_ids = []
            try:
                has_context = bool(documents)
                if documents:
                    # Extract available document IDs from documents that will be included in context
                    available_doc_ids = extract_used_document_ids(documents)
                    logger.info(f"ResponsesAPIView: Extracted {len(available_doc_ids)} available document IDs from retrieved documents: {available_doc_ids}")
                    # Build context from documents with document_id headers
                    context_parts = []
                    for doc in documents:
                        # Extract page_content and metadata
                        if hasattr(doc, 'page_content'):
                            page_content = str(doc.page_content)
                            metadata = getattr(doc, 'metadata', {}) or {}
                        elif isinstance(doc, dict):
                            page_content = doc.get('page_content', '') or doc.get('payload', {}).get('page_content', '')
                            metadata = doc.get('metadata', {}) or doc.get('payload', {}).get('metadata', {}) or {}
                        else:
                            page_content = str(doc) if doc else ''
                            metadata = {}

                        if page_content and page_content.strip():
                            # Extract document_id and document_name from metadata
                            doc_id = None
                            doc_name = None
                            if isinstance(metadata, dict):
                                doc_id = metadata.get("document_id") or metadata.get("doc_id") or metadata.get("documentId")
                                doc_name = metadata.get("document_name") or metadata.get("document_title") or "Unknown"
                            
                            # Prepend document_id header if available
                            if doc_id:
                                formatted_content = f"[document_id={doc_id}]\n[document_name={doc_name}]\n{page_content.strip()}"
                            else:
                                formatted_content = page_content.strip()
                            
                            context_parts.append(formatted_content)

                    if context_parts:
                        context = "\n\n---\n\n".join(context_parts)
                        # Add context to last user message
                        if messages and messages[-1]['role'] == 'user':
                            messages[-1]['content'] = f"Context:\n{context}\n\nQuestion: {messages[-1]['content']}"
                        logger.info(f"Added {len(context_parts)} document chunks to context (total length: {len(context)} chars)")
                    else:
                        logger.warning("No page_content found in retrieved documents")

                # Build system prompt with JSON instructions if documents are provided
                from .utils import _get_json_format_for_provider
                json_format_kwargs = _get_json_format_for_provider(provider_name) if has_context else {}
                
                resolved_instructions = build_assistant_system_prompt(
                    instructions, 
                    has_context=has_context, 
                    require_json=has_context
                )
                if resolved_instructions:
                    messages.insert(0, {"role": "system", "content": resolved_instructions})

                connections.close_all()
                llm_response = provider_instance.get_chat_completion(
                    messages=messages,
                    model=chat_model,
                    api_key=api_key,
                    **json_format_kwargs
                )
                
                # Extract answer text
                from .utils import _extract_answer_text, _extract_used_document_ids
                answer_text = _extract_answer_text(llm_response)
                logger.info(f"ResponsesAPIView: Extracted answer text (length: {len(answer_text) if answer_text else 0})")
                
                # Extract used_document_ids from LLM response (handles JSON and regex fallback)
                # Pass tenant and user for validation to filter out deleted document IDs
                used_document_ids = _extract_used_document_ids(llm_response, answer_text, tenant=self.user.tenant, user=self.user) if has_context else []
                logger.info(f"ResponsesAPIView: Extracted {len(used_document_ids)} used document IDs: {used_document_ids}")
                
                # If answer is JSON, extract the answer text from it
                import json
                try:
                    if isinstance(answer_text, str) and answer_text.strip().startswith('{'):
                        parsed_json = json.loads(answer_text)
                        if isinstance(parsed_json, dict) and "answer" in parsed_json:
                            answer_text = parsed_json.get("answer", answer_text)
                except (json.JSONDecodeError, ValueError):
                    pass  # Not JSON, use answer_text as is
                
                # Remove document ID citations from content for clean display
                from .utils import remove_document_id_citations
                cleaned_answer_text = remove_document_id_citations(answer_text)
                logger.info(f"ResponsesAPIView: Cleaned answer text (removed citations). Original length: {len(answer_text)}, Cleaned length: {len(cleaned_answer_text)}")

            except LLMAuthenticationError as e:
                logger.error(f"LLM authentication error: {e}")
                if response_record:
                    response_record.status = ResponseRecord.STATUS_FAILED
                    response_record.error_message = str(e)
                    response_record.save(update_fields=["status", "error_message", "updated_at"])
                return Response({"error": str(e)}, status=status.HTTP_401_UNAUTHORIZED)
            except (LLMRateLimitError, LLMQuotaError) as e:
                logger.warning(f"LLM rate limit/quota error: {e}")
                if response_record:
                    response_record.status = ResponseRecord.STATUS_FAILED
                    response_record.error_message = str(e)
                    response_record.save(update_fields=["status", "error_message", "updated_at"])
                return Response({"error": str(e)}, status=status.HTTP_429_TOO_MANY_REQUESTS)
            except LLMInvalidRequestError as e:
                logger.warning(f"LLM invalid request: {e}")
                if response_record:
                    response_record.status = ResponseRecord.STATUS_FAILED
                    response_record.error_message = str(e)
                    response_record.save(update_fields=["status", "error_message", "updated_at"])
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
            except LLMServiceUnavailableError as e:
                logger.error(f"LLM service unavailable: {e}")
                if response_record:
                    response_record.status = ResponseRecord.STATUS_FAILED
                    response_record.error_message = str(e)
                    response_record.save(update_fields=["status", "error_message", "updated_at"])
                return Response({"error": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            except LLMProviderError as e:
                logger.error(f"LLM provider error: {e}")
                if response_record:
                    response_record.status = ResponseRecord.STATUS_FAILED
                    response_record.error_message = str(e)
                    response_record.save(update_fields=["status", "error_message", "updated_at"])
                return Response({"error": str(e)}, status=getattr(e, 'status_code', 500))
            except ValueError as e:
                logger.warning(f"LLM request rejected: {e}")
                if response_record:
                    response_record.status = ResponseRecord.STATUS_FAILED
                    response_record.error_message = str(e)
                    response_record.save(update_fields=["status", "error_message", "updated_at"])
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_400_BAD_REQUEST
                )
            except RuntimeError as e:
                logger.error(f"LLM provider error (RuntimeError): {e}", exc_info=True)
                if response_record:
                    response_record.status = ResponseRecord.STATUS_FAILED
                    response_record.error_message = str(e)
                    response_record.save(update_fields=["status", "error_message", "updated_at"])
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_502_BAD_GATEWAY
                )
            except Exception as e:
                logger.error(f"Error generating LLM response (Exception): {e}", exc_info=True)
                if response_record:
                    response_record.status = ResponseRecord.STATUS_FAILED
                    response_record.error_message = str(e)
                    response_record.save(update_fields=["status", "error_message", "updated_at"])
                return Response(
                    {"error": f"Failed to generate response: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            response_end_time = timezone.now()

            # Store messages if conversation_id provided
            assistant_message_id = None
            if conversation_id and conversation:
                try:
                    # Store user messages with metadata
                    for idx, user_msg in enumerate(user_messages):
                        ConversationMessage.objects.create(
                            conversation=conversation,
                            role='user',
                            content=user_msg,
                            user=self.user,
                            metadata=user_messages_metadata[idx] if idx < len(user_messages_metadata) else {}
                        )
                    
                    # Store assistant response with used_document_ids in metadata (use cleaned content)
                    assistant_metadata = dict(meta_data) if meta_data else {}
                    assistant_metadata["used_document_ids"] = used_document_ids
                    logger.info(f"ResponsesAPIView: Storing ConversationMessage with metadata: {assistant_metadata} (type: {type(assistant_metadata)}, used_document_ids type: {type(used_document_ids)})")
                    assistant_msg = ConversationMessage.objects.create(
                        conversation=conversation,
                        role='assistant',
                        content=cleaned_answer_text,
                        user=None,
                        metadata=assistant_metadata
                    )
                    # Verify what was actually saved
                    assistant_msg.refresh_from_db()
                    logger.info(f"ResponsesAPIView: ConversationMessage saved with ID {assistant_msg.id}. Metadata after save: {assistant_msg.metadata} (type: {type(assistant_msg.metadata)})")
                    assistant_message_id = assistant_msg.id
                except Exception as e:
                    logger.error(f"Error storing messages: {e}", exc_info=True)
            
            # Format response
            # Add used_document_ids to metadata for response
            response_metadata = {}
            response_metadata["used_document_ids"] = used_document_ids
            logger.info(f"ResponsesAPIView: Final response metadata: {response_metadata}")
            response_data = {
                "id": response_record.id if response_record else None,
                "created_at": response_start_time.isoformat(),
                "status": ResponseRecord.STATUS_COMPLETED,
                "completed_at": response_end_time.isoformat(),
                "instructions": instructions,
                "model": chat_model,
                "output": [{
                    "message_id": str(assistant_message_id) if assistant_message_id else None,
                    "type": "message",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": cleaned_answer_text
                    }],
                    "role": "assistant",
                    "metadata": response_metadata,
                }],
                "user_id": str(self.user.id),
                "metadata": meta_data
            }

            if response_record:
                response_record.output = response_data.get("output", [])
                response_record.metadata = meta_data
                #To store both user metadata and assistant's response metadata
                # response_record.metadata = {**meta_data, **response_metadata}
                response_record.status = ResponseRecord.STATUS_COMPLETED
                response_record.error_message = ""
                logger.info(f"ResponsesAPIView: Saving ResponseRecord with metadata: {response_metadata}")
                response_record.save(update_fields=["output", "metadata", "status", "error_message", "updated_at"])

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Unexpected error in ResponsesAPIView: {e}", exc_info=True)
            if 'response_record' in locals() and response_record:
                response_record.status = ResponseRecord.STATUS_FAILED
                response_record.error_message = str(e)
                response_record.save(update_fields=["status", "error_message", "updated_at"])
            return Response(
                {"error": f"An unexpected error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        finally:
            connections.close_all()


class ResponseRetrieveDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveDestroyAPIView):
    """Retrieve or delete a stored response record."""

    serializer_class = ResponseRecordSerializer
    lookup_field = 'id'

    @swagger_auto_schema(
        operation_description="Retrieve a specific response by ID.",
        manual_parameters=[
            openapi.Parameter('token', openapi.IN_PATH, description="Authentication token", type=openapi.TYPE_STRING),
            openapi.Parameter('id', openapi.IN_PATH, description="Response ID", type=openapi.TYPE_STRING),
        ],
        tags=[RESPONSE_TAG],
    )
    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Delete a specific response by ID.",
        manual_parameters=[
            openapi.Parameter('token', openapi.IN_PATH, description="Authentication token", type=openapi.TYPE_STRING),
            openapi.Parameter('id', openapi.IN_PATH, description="Response ID", type=openapi.TYPE_STRING),
        ],
        tags=[RESPONSE_TAG],
    )
    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)

    def get_queryset(self):
        return ResponseRecord.objects.filter(tenant=self.user.tenant, user=self.user)


class ResponseCancelAPIView(TokenAuthenticatedMixin, APIView):
    """Cancel an in-progress response."""

    @swagger_auto_schema(
        operation_description="Cancel an ongoing response generation.",
        manual_parameters=[
            openapi.Parameter('token', openapi.IN_PATH, description="Authentication token", type=openapi.TYPE_STRING),
            openapi.Parameter('response_id', openapi.IN_PATH, description="Response ID", type=openapi.TYPE_STRING),
        ],
        tags=[RESPONSE_TAG],
    )
    def post(self, request, token, response_id):
        try:
            response_record = ResponseRecord.objects.get(id=response_id, tenant=self.user.tenant, user=self.user)
        except ResponseRecord.DoesNotExist:
            return Response({"error": f"Response '{response_id}' not found."}, status=status.HTTP_404_NOT_FOUND)

        if response_record.status in [ResponseRecord.STATUS_COMPLETED, ResponseRecord.STATUS_CANCELLED]:
            return Response({"message": f"Response is already {response_record.status}."}, status=status.HTTP_200_OK)

        response_record.status = ResponseRecord.STATUS_CANCELLED
        response_record.save(update_fields=["status", "updated_at"])
        return Response({"message": "Response cancelled."}, status=status.HTTP_200_OK)


class ConversationItemsAPIView(TokenAuthenticatedMixin, APIView):
    """Retrieve all messages from a conversation."""

    @swagger_auto_schema(
        operation_description="List all messages in a conversation with their metadata and timestamps.",
        manual_parameters=[
            openapi.Parameter(
                'conversation_id', openapi.IN_PATH, description="Conversation identifier", type=openapi.TYPE_STRING
            ),
            openapi.Parameter(
                'token', openapi.IN_PATH, description="Authentication token", type=openapi.TYPE_STRING
            ),
        ],
        tags=[CONVERSATION_TAG],
        responses={status.HTTP_200_OK: ConversationItemsSerializer(many=True)},
    )
    def get(self, request, conversation_id, token):
        try:
            # Get conversation
            try:
                conversation = Conversation.objects.get(id=conversation_id, tenant=self.user.tenant, user=self.user)
            except Conversation.DoesNotExist:
                return Response(
                    {"error": f"Conversation '{conversation_id}' not found."},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Get all messages ordered by created_at
            messages = conversation.messages.order_by('created_at')
            serializer = ConversationItemsSerializer(messages, many=True)
            
            logger.debug(f"Retrieved {len(messages)} messages for conversation ID: {conversation_id}, user: {self.user.id}")
            return Response(serializer.data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Unexpected error retrieving conversation items: {e}", exc_info=True)
            return Response(
                {"error": "An unexpected error occurred while retrieving conversation items."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
