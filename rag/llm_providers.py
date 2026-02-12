import abc
from typing import List, Dict, Any, Optional, Union
from django.conf import settings
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LangChainOpenAIEmbeddings
try:
    from langchain_ollama import ChatOllama
except ImportError:
    # Fallback to community if ollama package not yet installed as promised
    from langchain_community.chat_models import ChatOllama
import openai
import requests
import json
import logging
import backoff # Import backoff library for retries
import uuid
import math
from .resilience import call_with_resilience, close_connections_before_io


logger = logging.getLogger(__name__) # Initialize module-level logger

# --- Custom Exceptions ---

class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    def __init__(self, message, status_code=500, raw_response=None):
        super().__init__(message)
        self.status_code = status_code
        self.raw_response = raw_response

class LLMRateLimitError(LLMProviderError):
    """Exception raised when the provider rate limit is exceeded."""
    def __init__(self, message="Rate limit exceeded", raw_response=None):
        super().__init__(message, status_code=429, raw_response=raw_response)

class LLMQuotaError(LLMProviderError):
    """Exception raised when the provider quota is exceeded."""
    def __init__(self, message="Quota exceeded", raw_response=None):
        super().__init__(message, status_code=429, raw_response=raw_response)

class LLMAuthenticationError(LLMProviderError):
    """Exception raised when authentication with the provider fails."""
    def __init__(self, message="Authentication failed"):
        super().__init__(message, status_code=401)

class LLMInvalidRequestError(LLMProviderError):
    """Exception raised when the request sent to the provider is invalid."""
    def __init__(self, message="Invalid request"):
        super().__init__(message, status_code=400)

class LLMServiceUnavailableError(LLMProviderError):
    """Exception raised when the provider service is unavailable."""
    def __init__(self, message="Service unavailable"):
        super().__init__(message, status_code=503)

# --- Vector Store Exceptions ---

class VectorStoreError(Exception):
    """Base exception for vector store errors."""
    def __init__(self, message, status_code=500):
        super().__init__(message)
        self.status_code = status_code

class VectorStoreConnectionError(VectorStoreError):
    """Exception raised when the vector store is unavailable (e.g. connection refused)."""
    def __init__(self, message="Vector store connection failed"):
        super().__init__(message, status_code=503)



# --- Configuration ---
OLLAMA_REQUEST_TIMEOUT = getattr(settings, 'OLLAMA_REQUEST_TIMEOUT', 180) # Default to 180 seconds
# Ollama /api/embeddings does not officially support batching. Process individually.
OLLAMA_MAX_RETRIES = getattr(settings, 'OLLAMA_MAX_RETRIES', 3)
DEFAULT_FALLBACK_DIMENSION = 1536 # Default dimension if detection fails

# --- Known Model Dimensions (Add more as needed) ---
# Attempt to fetch dynamically, but have fallbacks
KNOWN_OLLAMA_EMBEDDING_DIMENSIONS = {
    "bge-m3": 1024,
    "bge-m3:latest": 1024,
    "mxbai-embed-large": 1024,
    "mxbai-embed-large:latest": 1024,
    # Add other common Ollama embedding models and their dimensions
}

KNOWN_OPENAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-large": 1536,
    "text-embedding-3-small": 1536,
    # It's good practice to keep a trailing comma for easier additions.
}

# Defaults for controllable OpenAI chat completion fields that improve output
# quality and response determinism. These can be overridden through Django
# settings or per-request keyword arguments.
OPENAI_CHAT_COMPLETION_DEFAULTS = getattr(settings, "OPENAI_CHAT_COMPLETION_DEFAULTS", {
    "temperature": 0.3,
    "top_p": 1,
    # "max_tokens": 8000,
    # "max_completion_tokens": 8000,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    # response_format is optional; set in settings to enforce JSON mode, etc.
})

# Defaults for Ollama chat completion options. Ollama expects generation
# controls to be nested under an `options` key.
OLLAMA_CHAT_OPTIONS_DEFAULTS = getattr(settings, "OLLAMA_CHAT_OPTIONS_DEFAULTS", {
    "temperature": 0.3,
    "top_p": 0.9,
    # "num_predict": 1024,
    "repeat_penalty": 1.1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
})

OLLAMA_RESPONSE_FORMAT = getattr(settings, "OLLAMA_RESPONSE_FORMAT", None)

# --- Helper Functions ---

@backoff.on_exception(backoff.expo,
                      (requests.exceptions.RequestException, LLMServiceUnavailableError),
                      max_tries=OLLAMA_MAX_RETRIES,
                      jitter=backoff.full_jitter)
def make_ollama_request(method: str, url: str, payload: Dict = None, stream: bool = False):
    """Make HTTP request to Ollama API with exponential backoff."""
    try:
        close_connections_before_io("Ollama API request")
        response = call_with_resilience(
            lambda: requests.request(
                method,
                url,
                json=payload,
                stream=stream,
                timeout=OLLAMA_REQUEST_TIMEOUT,
            ),
            service="ollama_http_request",
            exceptions=(requests.exceptions.RequestException,),
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        try:
            error_data = e.response.json()
            message = error_data.get('error', error_data.get('message', e.response.text))
        except:
            message = e.response.text or str(e)
        
        if status_code == 429:
            raise LLMRateLimitError(message)
        elif status_code in (401, 403):
            raise LLMAuthenticationError(message)
        elif status_code == 400:
            raise LLMInvalidRequestError(message)
        elif status_code >= 500:
            raise LLMServiceUnavailableError(message)
        else:
            raise LLMProviderError(message, status_code=status_code)
    except requests.exceptions.RequestException as e:
        raise LLMServiceUnavailableError(str(e))

# --- Base Provider Class ---

class BaseLLMProvider(abc.ABC):
    """Base abstract class for LLM providers."""

    @abc.abstractmethod
    def validate_credentials(self, api_key: str = None, model: str = None, **kwargs) -> bool:
        """Validate API key and other settings."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_chat_completion(self, messages: List[Dict[str, Any]], model: str, api_key: str = None, tools: List[Dict] = None, **kwargs) -> Any:
        """Get chat completion from the LLM, with optional tools."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_embeddings(self, texts: List[str], model: str, api_key: str = None, **kwargs) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_embedding_dimension(self, model: str, api_key: str = None, **kwargs) -> int:
        """Return the embedding dimension for a given model."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_multimodal(self, model: str, api_key: str = None, **kwargs) -> bool:
        """Check if the specified chat model supports multimodal inputs."""
        raise NotImplementedError

    def get_provider_name(self) -> str:
        """Returns the name of the provider (e.g., 'OpenAI', 'Ollama')."""
        return self.__class__.__name__.replace("Provider", "")


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _estimate_message_tokens(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(item) for item in content)
        total += _estimate_tokens(str(content))
    return total


def enforce_token_limits(messages: List[Dict[str, Any]], provider: str) -> List[Dict[str, Any]]:
    max_prompt_tokens = getattr(settings, "LLM_MAX_PROMPT_TOKENS", 12000)
    max_context_tokens = getattr(settings, "LLM_MAX_CONTEXT_TOKENS", 16000)
    max_allowed = min(max_prompt_tokens, max_context_tokens)

    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    other_messages = [msg for msg in messages if msg.get("role") != "system"]
    trimmed_messages = system_messages + other_messages

    while other_messages and _estimate_message_tokens(trimmed_messages) > max_allowed:
        removed = other_messages.pop(0)
        trimmed_messages = system_messages + other_messages
        logger.warning(
            "Trimmed oldest message for %s token limits: role=%s",
            provider,
            removed.get("role"),
        )

    total_tokens = _estimate_message_tokens(trimmed_messages)
    if total_tokens > max_allowed and trimmed_messages:
        last_msg = trimmed_messages[-1]
        content = str(last_msg.get("content", ""))
        allowed_chars = max(0, int(max_allowed * 4))
        if len(content) > allowed_chars:
            last_msg = {**last_msg, "content": content[:allowed_chars]}
            trimmed_messages[-1] = last_msg
            logger.warning(
                "Truncated message content for %s to meet token limits (approx %s tokens).",
                provider,
                max_allowed,
            )

    return trimmed_messages

# --- Normalization ---

class NormalizedChatResponse:
    """Standardized response from any LLM provider."""
    def __init__(self, content: Optional[str], role: str, tool_calls: List[Dict[str, Any]] = None, finish_reason: str = "stop", model: str = None, id: str = None):
        self.content = content
        self.role = role
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.model = model
        self.id = id

    @classmethod
    def from_langchain(cls, message: BaseMessage, model: str = None) -> 'NormalizedChatResponse':
        """Create a NormalizedChatResponse from a LangChain message."""
        content = message.content
        role = "assistant"
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, ToolMessage):
            role = "tool"
        
        tool_calls = []
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.get('id'),
                    "type": "function",
                    "function": {
                        "name": tc.get('name'),
                        "arguments": json.dumps(tc.get('args')) if isinstance(tc.get('args'), dict) else tc.get('args')
                    }
                })
        
        # Metadata extraction
        metadata = getattr(message, 'response_metadata', {}) or {}
        additional = getattr(message, 'additional_kwargs', {}) or {}
        
        finish_reason = additional.get('finish_reason') or metadata.get('finish_reason') or 'stop'
        msg_id = message.id if hasattr(message, 'id') else additional.get('id') or metadata.get('id')
        
        return cls(
            content=content,
            role=role,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            model=model,
            id=msg_id
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "role": self.role,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
            "model": self.model,
            "id": self.id,
        }

    def __str__(self):
        return self.content or ""

def _requires_completion_tokens(model: str) -> bool:
    """Check if model requires max_completion_tokens instead of max_tokens."""
    return model.lower().startswith(("o1", "gpt-4.1", "gpt-5"))

def _is_reasoning_model(model: str) -> bool:
    """Check if model is a reasoning model (o1, etc) with restricted params."""
    return model.lower().startswith(("o1", "gpt-5"))

def _uses_responses_api(model: str) -> bool:
    """Check if model uses the NEW OpenAI Responses API (GPT-5+)."""
    return model.lower().startswith("gpt-5")

# --- OpenAI Provider ---

class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI API."""

    def __init__(self):
        self.client: Optional[openai.OpenAI] = None
        self.chat_config: Dict[str, Any] = {}

    def _initialize_client(self, api_key: str) -> None:
        """Initialize OpenAI client with provided API key."""
        if not api_key:
            # This case should ideally be caught before calling, but as a safeguard:
            logger.error("OpenAI client initialization attempted without API key.")
            raise ValueError("API key is required for OpenAI provider but was not provided.")
        if self.client is None or self.client.api_key != api_key:
            logger.debug("Initializing OpenAI client.")
            self.client = openai.OpenAI(api_key=api_key)
        else:
            logger.debug("OpenAI client already initialized with the provided API key.")

    def validate_credentials(self, api_key: str, model: str = None, **kwargs) -> bool:
        """Validate OpenAI API credentials by attempting to list models."""
        logger.debug(f"Validating OpenAI credentials. API key provided: {bool(api_key)}")
        if not api_key:
            logger.warning("OpenAI API key not provided for validation.")
            return False
        try:
            self._initialize_client(api_key)
            logger.debug("Attempting to list OpenAI models for API key validation.")
            self.client.models.list() # Lightweight call to check API key validity
            logger.info("OpenAI API key validated successfully.")
            return True
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI API key validation failed: AuthenticationError - {e.body.get('message') if e.body else str(e)}", exc_info=True)
            return False
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API key validation failed: APIConnectionError - Could not connect to OpenAI. {str(e)}", exc_info=True)
            return False
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI API key validation failed due to rate limit: {e.body.get('message') if e.body else str(e)}", exc_info=True)
            return False # Or True, depending on if this is considered "valid but temporarily unavailable"
        except Exception as e: # Catch other potential OpenAI errors or general issues
            logger.error(f"OpenAI API key validation failed due to an unexpected error: {str(e)}", exc_info=True)
            return False

    def get_chat_completion(self, messages: List[Dict[Union[str, Any], Any]], model: str, api_key: str = None, tools: List[Dict] = None, **kwargs) -> NormalizedChatResponse:
        """Get chat completion from OpenAI using LangChain's ChatOpenAI."""
        effective_api_key = api_key or getattr(settings, 'OPENAI_API_KEY', None)
        if not effective_api_key:
            raise ValueError("OpenAI API key is required.")
        
        effective_model = model or getattr(settings, 'DEFAULT_OPENAI_MODEL', 'gpt-4.1')
        messages = enforce_token_limits(messages, "openai")
        max_output_tokens_limit = getattr(settings, "LLM_MAX_OUTPUT_TOKENS", 12000)
        
        # Convert dict messages to LangChain messages
        lc_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            elif role == 'user':
                lc_messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                # Handle potential tool calls in history
                tcs = msg.get('tool_calls')
                if tcs:
                    lc_messages.append(AIMessage(content=content, tool_calls=[
                        {"id": tc['id'], "name": tc['function']['name'], "args": json.loads(tc['function']['arguments'])}
                        for tc in tcs
                    ]))
                else:
                    lc_messages.append(AIMessage(content=content))
            elif role == 'tool':
                lc_messages.append(ToolMessage(content=content, tool_call_id=msg.get('tool_call_id')))

        # Specialized logic for GPT-5 Responses API if model requires it
        if _uses_responses_api(effective_model):
            self._initialize_client(effective_api_key)
            # Map parameters for Responses API
            chat_params = {**OPENAI_CHAT_COMPLETION_DEFAULTS, **self.chat_config, **kwargs}
            responses_kwargs = {
                "model": effective_model,
                "input": messages,
                "reasoning": chat_params.get("reasoning", {"effort": "low"})
            }
            max_output_tokens = chat_params.get("max_completion_tokens") or chat_params.get("max_tokens")
            if max_output_tokens_limit and max_output_tokens_limit > 0:
                if max_output_tokens:
                    responses_kwargs["max_output_tokens"] = min(max_output_tokens, max_output_tokens_limit)
                else:
                    responses_kwargs["max_output_tokens"] = max_output_tokens_limit
            elif max_output_tokens:
                responses_kwargs["max_output_tokens"] = max_output_tokens
            
            try:
                close_connections_before_io("OpenAI Responses API call")
                response = call_with_resilience(
                    lambda: self.client.responses.create(**responses_kwargs),
                    service="openai_responses",
                    exceptions=(Exception,),
                )
                logger.info(f"OpenAI GPT-5 response object received. Type: {type(response)}")
                # Log a preview of the response object attributes for debugging
                try:
                    attrs = [a for a in dir(response) if not a.startswith('_')]
                    logger.debug(f"GPT-5 response attributes: {attrs}")
                    if hasattr(response, 'output'):
                         logger.debug(f"GPT-5 response.output type: {type(response.output)}")
                except:
                    pass

                content = None
                
                # Priority 1: Direct output_text convenience property
                if hasattr(response, 'output_text'):
                    content = response.output_text
                
                # Priority 2: Standard structured output
                if not content:
                    output = getattr(response, 'output', [])
                    if isinstance(output, list) and output:
                        # Strategy: Try to find text in the first message's content parts
                        first_msg = output[0]
                        # Some versions might have a 'text' property directly on the first message
                        if hasattr(first_msg, 'text'):
                            content = getattr(first_msg, 'text')
                            if hasattr(content, 'value'): # Handle Text(value='...')
                                content = content.value
                        
                        if not content:
                            content_parts = getattr(first_msg, 'content', [])
                            if isinstance(content_parts, list):
                                for part in content_parts:
                                    part_type = getattr(part, 'type', None)
                                    if part_type == 'text':
                                        text_obj = getattr(part, 'text', None)
                                        if hasattr(text_obj, 'value'):
                                            content = text_obj.value
                                        else:
                                            content = str(text_obj) if text_obj else None
                                        if content:
                                            break
                
                # Final Priority: Aggressive search in model dump for any text
                if not content and response:
                    try:
                        # Try to dump to dict if it's a Pydantic model/OpenAI object
                        res_dict = {}
                        if hasattr(response, 'model_dump'):
                            res_dict = response.model_dump()
                        elif hasattr(response, 'dict'):
                            res_dict = response.dict()
                        
                        if res_dict:
                            # Look for 'text' or 'value' in nested structures
                            def find_text(obj, seen=None):
                                if seen is None: seen = set()
                                if id(obj) in seen: return None
                                seen.add(id(obj))
                                
                                if isinstance(obj, str) and len(obj) > 0:
                                    return obj
                                if isinstance(obj, dict):
                                    # Prioritize common keys
                                    for k in ['text', 'value', 'content']:
                                        found = find_text(obj.get(k), seen)
                                        if found: return found
                                    for k, v in obj.items():
                                        if k not in ['text', 'value', 'content']:
                                            found = find_text(v, seen)
                                            if found: return found
                                if isinstance(obj, list):
                                    for item in obj:
                                        found = find_text(item, seen)
                                        if found: return found
                                return None
                            
                            content = find_text(res_dict)
                    except Exception as dump_e:
                        logger.debug(f"Deep extraction failed: {dump_e}")

                if not content and response:
                    logger.warning(f"Failed specialized parsing for GPT-5 response, falling back to string conversion. Response type: {type(response)}")
                    content = str(response)

                return NormalizedChatResponse(
                    content=content,
                    role="assistant",
                    model=effective_model,
                    id=getattr(response, 'id', None)
                )
            except Exception as e:
                logger.warning(f"GPT-5 Responses API call failed, falling back to ChatOpenAI: {e}")
                # Fall through to ChatOpenAI

        try:
            # Prepare config
            chat_params = {**OPENAI_CHAT_COMPLETION_DEFAULTS, **self.chat_config, **kwargs}
            if max_output_tokens_limit and max_output_tokens_limit > 0:
                if "max_tokens" not in chat_params and "max_completion_tokens" not in chat_params:
                    chat_params["max_tokens"] = max_output_tokens_limit
                elif "max_tokens" in chat_params:
                    chat_params["max_tokens"] = min(chat_params["max_tokens"], max_output_tokens_limit)
            
            # Handle max_tokens vs max_completion_tokens
            if _requires_completion_tokens(effective_model):
                chat_params["max_completion_tokens"] = chat_params.pop("max_tokens", chat_params.get("max_completion_tokens"))
            
            # Remove params that reasoning models don't support
            if _is_reasoning_model(effective_model):
                for field in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
                    chat_params.pop(field, None)

            # Move response_format to model_kwargs to avoid LangChain warnings 
            # while keeping it functional for the underlying API
            model_kwargs = chat_params.pop("model_kwargs", {})
            if "response_format" in chat_params:
                model_kwargs["response_format"] = chat_params.pop("response_format")

            llm = ChatOpenAI(
                api_key=effective_api_key,
                model=effective_model,
                model_kwargs=model_kwargs,
                **chat_params
            )
            
            if tools:
                llm = llm.bind_tools(tools)
            
            close_connections_before_io("OpenAI chat completion")
            response_msg = call_with_resilience(
                lambda: llm.invoke(lc_messages),
                service="openai_chat_completion",
                exceptions=(Exception,),
            )
            logger.info(f"OpenAI chat completion via LangChain successful for model: {effective_model}.")
            
            return NormalizedChatResponse.from_langchain(response_msg, model=effective_model)

        except openai.AuthenticationError as e:
            raise LLMAuthenticationError(str(e))
        except openai.RateLimitError as e:
            msg = str(e)
            if "quota" in msg.lower():
                raise LLMQuotaError(msg)
            raise LLMRateLimitError(msg)
        except Exception as e:
            raise LLMProviderError(str(e))


    def get_embeddings(self, texts: List[str], model: str, api_key: str = None, **kwargs) -> List[List[float]]:
        """Get embeddings from OpenAI using LangChain's OpenAIEmbeddings."""
        effective_api_key = api_key or getattr(settings, 'OPENAI_API_KEY', None)
        if not effective_api_key:
            raise ValueError("OpenAI API key is required.")
        
        effective_model = model or getattr(settings, 'DEFAULT_OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')
        
        try:
            embeddings_model = LangChainOpenAIEmbeddings(
                api_key=effective_api_key,
                model=effective_model,
                **kwargs
            )
            close_connections_before_io("OpenAI embeddings")
            return call_with_resilience(
                lambda: embeddings_model.embed_documents(texts),
                service="openai_embeddings",
                exceptions=(Exception,),
            )
        except Exception as e:
            raise LLMProviderError(str(e))

    def get_embedding_dimension(self, model: str, api_key: str = None, **kwargs) -> int:
        effective_api_key = api_key or getattr(settings, 'OPENAI_API_KEY', None)
        embedding_model_name = model or getattr(settings, 'DEFAULT_OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')
        dimension = KNOWN_OPENAI_EMBEDDING_DIMENSIONS.get(embedding_model_name)
        if dimension:
            logger.info(f"Using known embedding dimension for OpenAI model '{embedding_model_name}': {dimension}")
            return dimension

        logger.warning(f"Embedding dimension for OpenAI model '{embedding_model_name}' not known. Attempting to infer with a test embedding.")
        try:
            test_embedding = self.get_embeddings(["test"], model=embedding_model_name, api_key=effective_api_key)[0]
            dimension = len(test_embedding)
            KNOWN_OPENAI_EMBEDDING_DIMENSIONS[embedding_model_name] = dimension
            logger.info(f"Inferred dimension: {dimension}")
            return dimension
        except Exception as e:
            raise LLMProviderError(f"Could not determine embedding dimension for model '{embedding_model_name}': {str(e)}")

    def is_multimodal(self, model: str, api_key: str = None, **kwargs) -> bool:
        # Broadened check for multimodal models
        if not model: return False
        model_lc = model.lower()
        multimodal_indicators = ['gpt-4o', 'gpt-4.1', 'gpt-4-vision', 'gpt-5', 'o1-']
        return any(ind in model_lc for ind in multimodal_indicators)

# --- Ollama Provider ---

class OllamaProvider(BaseLLMProvider):
    """Provider for Ollama API."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or getattr(settings, 'OLLAMA_API_URL', getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434'))
        self._model_details_cache = {}  # Cache for model details to avoid repeated /api/show calls
        self.chat_config: Dict[str, Any] = {}

    def _get_model_details(self, model: str) -> Dict:
        """Fetch model details from Ollama /api/show."""
        if model in self._model_details_cache:
            return self._model_details_cache[model]
        try:
            response = make_ollama_request('POST', f'{self.base_url}/api/show', {"name": model})
            details = response.json()
            self._model_details_cache[model] = details
            return details
        except Exception as e:
            logger.error(f"Failed to fetch details for Ollama model '{model}': {e}")
            return {}

    def validate_credentials(self, api_key: str = None, model: str = None, **kwargs) -> bool:
        """Validate Ollama server availability and model (no API key needed)."""
        logger.debug("Validating Ollama credentials (server availability and model).")
        try:
            # Check server health with a simple /api/tags call
            response = make_ollama_request('GET', f'{self.base_url}/api/tags')
            if response.status_code != 200:
                logger.error(f"Ollama server validation failed: {response.text}")
                return False

            # If model specified, check if it's available
            if model:
                models_data = response.json().get('models', [])
                if not any(m['name'] == model for m in models_data):
                    logger.warning(f"Ollama model '{model}' not found in available models.")
                    return False

            logger.info("Ollama server validated successfully.")
            return True
        except Exception as e:
            logger.error(f"Ollama validation failed: {e}")
            return False

    def get_chat_completion(self, messages: List[Dict[str, Any]], model: str, api_key: str = None, tools: List[Dict] = None, **kwargs) -> NormalizedChatResponse:
        """Get chat completion from Ollama using LangChain's ChatOllama."""
        effective_model = model or getattr(settings, 'DEFAULT_OLLAMA_MODEL', 'llama3.1:latest')
        logger.debug(f"Requesting Ollama chat completion via LangChain with model: {effective_model}")
        messages = enforce_token_limits(messages, "ollama")
        max_output_tokens_limit = getattr(settings, "LLM_MAX_OUTPUT_TOKENS", 12000)

        # Convert dict messages to LangChain messages
        lc_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content') or "" # Content cannot be None for LangChain messages
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            elif role == 'user':
                lc_messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                tcs = msg.get('tool_calls')
                if tcs:
                    lc_messages.append(AIMessage(content=content, tool_calls=[
                        {"id": tc.get('id'), "name": tc['function']['name'], "args": json.loads(tc['function']['arguments']) if isinstance(tc['function']['arguments'], str) else tc['function']['arguments']}
                        for tc in tcs
                    ]))
                else:
                    lc_messages.append(AIMessage(content=content))
            elif role == 'tool':
                lc_messages.append(ToolMessage(content=content, tool_call_id=msg.get('tool_call_id')))

        try:
            # Prepare options
            chat_params = {**OLLAMA_CHAT_OPTIONS_DEFAULTS, **(self.chat_config or {}), **kwargs}
            if max_output_tokens_limit and max_output_tokens_limit > 0:
                if "num_predict" not in chat_params:
                    chat_params["num_predict"] = max_output_tokens_limit
                else:
                    chat_params["num_predict"] = min(chat_params["num_predict"], max_output_tokens_limit)
            
            # Map Ollama format
            format_type = chat_params.pop("format", None)

            llm = ChatOllama(
                base_url=self.base_url,
                model=effective_model,
                format=format_type,
                **{k: v for k, v in chat_params.items() if v is not None}
            )
            
            if tools:
                # Standard LangChain tool binding
                llm = llm.bind_tools(tools)
            
            close_connections_before_io("Ollama chat completion")
            response_msg = call_with_resilience(
                lambda: llm.invoke(lc_messages),
                service="ollama_chat_completion",
                exceptions=(Exception,),
            )
            logger.info(f"Ollama chat completion via LangChain successful for model: {effective_model}.")
            
            return NormalizedChatResponse.from_langchain(response_msg, model=effective_model)

        except Exception as e:
            logger.error(f"Ollama chat completion failed for model {effective_model}: {e}")
            raise LLMProviderError(str(e))

    def get_embeddings(self, texts: List[str], model: str, api_key: str = None, **kwargs) -> List[List[float]]:
        """Get embeddings from Ollama using standard request format."""
        embedding_model_name = model or getattr(settings, 'DEFAULT_OLLAMA_EMBEDDING_MODEL', 'bge-m3')
        logger.debug(f"Requesting Ollama embeddings for {len(texts)} texts with model: {embedding_model_name}")

        try:
            try:
                from langchain_ollama import OllamaEmbeddings as LangChainOllamaEmbeddings
            except ImportError:
                from langchain_community.embeddings import OllamaEmbeddings as LangChainOllamaEmbeddings
            
            embeddings_model = LangChainOllamaEmbeddings(
                base_url=self.base_url,
                model=embedding_model_name,
                **kwargs
            )
            close_connections_before_io("Ollama embeddings")
            return call_with_resilience(
                lambda: embeddings_model.embed_documents(texts),
                service="ollama_embeddings",
                exceptions=(Exception,),
            )
        except Exception as e:
            logger.error(f"Ollama embedding failed for model {embedding_model_name}: {e}")
            raise LLMProviderError(str(e))

    def get_embedding_dimension(self, model: str = None, api_key: str = None, **kwargs) -> int:
        """
        Return embedding dimension for Ollama model.
        1. Prioritize KNOWN_OLLAMA_EMBEDDING_DIMENSIONS.
        2. If not known, attempt to infer with a test embedding.
        3. If inference fails, raise ValueError.
        """
        embedding_model_name = model or getattr(settings, 'DEFAULT_OLLAMA_EMBEDDING_MODEL', 'bge-m3')

        dimension = KNOWN_OLLAMA_EMBEDDING_DIMENSIONS.get(embedding_model_name)
        if dimension:
            logger.info(f"Using known embedding dimension for Ollama model '{embedding_model_name}': {dimension}")
            return dimension

        logger.warning(f"Embedding dimension for Ollama model '{embedding_model_name}' is not in known dimensions. Attempting to infer with a test embedding.")
        try:
            # Validate model availability before attempting to get embeddings for dimension inference
            # This also warms up the _model_details_cache if not already populated.
            if not self.validate_credentials(model=embedding_model_name):
                 raise LLMProviderError(f"Ollama model '{embedding_model_name}' is not available or server is unreachable.")

            test_embedding_data = self.get_embeddings(texts=["test"], model=embedding_model_name) # get_embeddings uses its own default or the one passed

            if test_embedding_data and test_embedding_data[0] and isinstance(test_embedding_data[0], list) and len(test_embedding_data[0]) > 0:
                dimension = len(test_embedding_data[0])
                logger.info(f"Successfully inferred embedding dimension for Ollama model '{embedding_model_name}': {dimension}. Caching this value.")
                KNOWN_OLLAMA_EMBEDDING_DIMENSIONS[embedding_model_name] = dimension
                return dimension
            else:
                raise LLMProviderError(f"Ollama model '{embedding_model_name}' returned empty or invalid embedding for dimension inference.")
        except Exception as e:
            raise LLMProviderError(str(e))


    def is_multimodal(self, model: str, api_key: str = None, **kwargs) -> bool:
        """
        Check if the specified Ollama chat model supports multimodal inputs by inspecting its details.
        Ollama's /api/show endpoint provides model details. Heuristics are used as there isn't a direct boolean flag.
        Common indicators:
        - 'families' array containing 'clip' (typical for vision encoders).
        - Presence of 'mmproj' (multimodal projector) in the model's parameter list string.
        This method might need updates if Ollama changes its model information structure.
        """
        logger.debug(f"Checking multimodal capability for Ollama model: '{model}'")
        details = self._get_model_details(model)

        if not details: # If fetching details failed or model not found
            logger.warning(f"Could not fetch details for Ollama model '{model}'. Assuming not multimodal for safety.")
            return False

        model_families = details.get('details', {}).get('families', [])
        # Parameters are often a long string of text in Ollama's /api/show.
        # We convert it to lowercase for case-insensitive search.
        model_parameters_str = str(details.get('parameters', '')).lower()

        is_multi = False
        if model_families and any('clip' in str(f).lower() for f in model_families):
            is_multi = True
            logger.debug(f"Model '{model}' identified as potentially multimodal based on 'clip' in families: {model_families}")
        elif 'mmproj' in model_parameters_str: # Check for multimodal projector parameters
            is_multi = True
            logger.debug(f"Model '{model}' identified as potentially multimodal based on 'mmproj' in parameters.")

        logger.info(f"Multimodal check for Ollama model '{model}': {is_multi}. (Families: {model_families}, Inspected params snippet: '{model_parameters_str[:200]}...')")
        return is_multi

# --- Langchain Embeddings Wrappers ---

class OllamaEmbeddings(Embeddings):
    """Expose Ollama embeddings via LangChain's Embeddings API."""

    def __init__(self, model: str = None):
        self.provider = OllamaProvider()
        self.model = model or getattr(settings, 'DEFAULT_OLLAMA_EMBEDDING_MODEL', 'bge-m3')
        try:
            from langchain_ollama import OllamaEmbeddings as LangChainOllamaEmbeddings
        except ImportError:
            from langchain_community.embeddings import OllamaEmbeddings as LangChainOllamaEmbeddings
        
        self.lc_embeddings = LangChainOllamaEmbeddings(
            base_url=self.provider.base_url,
            model=self.model
        )
        logger.info(f"Initialized OllamaEmbeddings with model: {self.model}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents using LangChain's OllamaEmbeddings."""
        return self.lc_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using LangChain's OllamaEmbeddings."""
        return self.lc_embeddings.embed_query(text)

    def get_dimension(self) -> int:
        """Returns the embedding dimension of the model."""
        logger.debug(f"OllamaEmbeddings: Getting dimension for model {self.model}")
        try:
            return self.provider.get_embedding_dimension(model=self.model)
        except ValueError as e:
            logger.error(f"OllamaEmbeddings: Could not get dimension for model {self.model}: {e}", exc_info=True)
            raise # Re-raise to indicate failure in getting dimension

# (Keep OpenAIEmbeddings wrapper if needed, or remove if using langchain_openai directly)
# from langchain_openai import OpenAIEmbeddings # Already imported in utils.py
