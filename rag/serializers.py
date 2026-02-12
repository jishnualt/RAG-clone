# rag/serializers.py

from rest_framework import serializers, status
from .models import (
    Tenant,
    User,
    Collection,
    VectorStore,
    Document,
    Assistant,
    Thread,
    Message,
    Run,
    DocumentAccess,
    LLMProviderConfig,
    DocumentAlert,
    MessageFeedback,
    Conversation,
    ConversationMessage,
    ResponseRecord,
)
from django.shortcuts import get_object_or_404
from django.contrib.auth import authenticate
from django.db import IntegrityError, transaction
import requests
from project.settings import DEFAULT_OPENAI_MODEL, DEFAULT_OLLAMA_MODEL
from django.conf import settings
import logging
from .utils import (
    normalize_provider_name,
    infer_provider_from_model,
    get_provider_embedding_dimension,
    get_llm_config,
    validate_model_availability,
)
from .llm_providers import LLMProviderError, LLMServiceUnavailableError

logger = logging.getLogger(__name__)


def validate_metadata_map(value):
    """Validate metadata dictionary per OpenAI-like constraints.
    
    Allows string values for most keys, but allows lists/arrays for 'used_document_ids' key.
    """
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise serializers.ValidationError("metadata must be an object with string keys and values.")
    if len(value) > 16:
        raise serializers.ValidationError("metadata can include at most 16 entries.")
    validated = {}
    for key, val in value.items():
        if not isinstance(key, str):
            raise serializers.ValidationError("metadata keys must be strings.")
        if len(key) > 64:
            raise serializers.ValidationError(f"metadata key '{key}' exceeds 64 characters.")
        
        # Special handling for used_document_ids - allow list/array
        if key == "used_document_ids":
            if isinstance(val, list):
                # Validate list contains only strings
                for item in val:
                    if not isinstance(item, str):
                        raise serializers.ValidationError(f"metadata value for key '{key}' must be a list of strings.")
                validated[key] = val
            else:
                raise serializers.ValidationError(f"metadata value for key '{key}' must be a list.")
        else:
            # For other keys, require string values
            if not isinstance(val, str):
                raise serializers.ValidationError(f"metadata value for key '{key}' must be a string.")
            if len(val) > 512:
                raise serializers.ValidationError(f"metadata value for key '{key}' exceeds 512 characters.")
            validated[key] = val
    return validated

class TenantSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tenant
        fields = ['id', 'name']

    def validate_name(self, value):
        if not value.strip():
            raise serializers.ValidationError("Tenant name cannot be empty.")
        return value

class RegisterSerializer(serializers.ModelSerializer):
    first_name = serializers.CharField(required=False, allow_blank=True, help_text="Optional first name for the user")
    last_name = serializers.CharField(required=False, allow_blank=True, help_text="Optional last name for the user")
    password = serializers.CharField(write_only=True, help_text="Plain-text password used to create the account")
    # tenant = serializers.PrimaryKeyRelatedField(queryset=Tenant.objects.all(), required=False)
    tenant_name = serializers.CharField(
        write_only=True, required=False, help_text="Tenant to create or join. Required when registering."
    )
    collection_name = serializers.CharField(
        write_only=True, required=False, help_text="Name of the default collection to create for the tenant"
    )
    llm_provider = serializers.ChoiceField(
        choices=[("openai", "openai"), ("ollama", "ollama")],
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Preferred LLM provider to provision during sign-up."
    )

    language = serializers.CharField(required=False, allow_blank=True, help_text="UI language preference")

    class Meta:
        model = User
        fields = (
            'id',
            'first_name',
            'last_name',
            'email',
            'password',
            'tenant_name',
            'collection_name',
            'llm_provider',
            'language',
        )

    def validate(self, data):
        # tenant = data.get('tenant')
        tenant_name = data.get('tenant_name')
        if not tenant_name:
            raise serializers.ValidationError("'tenant_name' must be provided.")

        email = (data.get("email") or "").strip().lower()
        if not email:
            raise serializers.ValidationError({"email": "Email is required."})
        if User.objects.filter(username=email).exists() or User.objects.filter(email=email).exists():
            conflict = serializers.ValidationError(
                {
                    "error": "A user with this email already exists.",
                    "code": "USER_ALREADY_EXISTS",
                    "email": ["A user with this email already exists."],
                }
            )
            conflict.status_code = status.HTTP_409_CONFLICT
            raise conflict
        data["email"] = email
        return data

    def create(self, validated_data):
        tenant_name = validated_data.get('tenant_name')
        collection_name = validated_data.get('collection_name', 'default_collection')
        requested_provider = validated_data.get("llm_provider")

        if not tenant_name:
            error = serializers.ValidationError(
                {"error": "Tenant name is required.", "code": "TENANT_REQUIRED"}
            )
            error.status_code = status.HTTP_400_BAD_REQUEST
            raise error
        if not collection_name:
            error = serializers.ValidationError(
                {"error": "Collection name is required.", "code": "COLLECTION_REQUIRED"}
            )
            error.status_code = status.HTTP_400_BAD_REQUEST
            raise error

        tenant, _ = Tenant.objects.get_or_create(name=tenant_name)

        try:
            with transaction.atomic():
                user = User.objects.create_user(
                    first_name=validated_data.get('first_name', ''),
                    last_name=validated_data.get('last_name', ''),
                    username=validated_data['email'],
                    email=validated_data['email'],
                    password=validated_data['password'],
                    tenant=tenant,
                    language=validated_data.get('language', 'en'),
                    selected_llm_provider=normalize_provider_name(requested_provider) if requested_provider else None,
                    is_setup=False,
                    active_collection_ready=False,
                    llm_configured=False,
                )
        except IntegrityError:
            conflict = serializers.ValidationError(
                {
                    "error": "A user with this email already exists.",
                    "code": "USER_ALREADY_EXISTS",
                    "email": ["A user with this email already exists."],
                }
            )
            conflict.status_code = status.HTTP_409_CONFLICT
            raise conflict

        # If no provider selected, return without creating a collection
        if not requested_provider:
            return user

        from .models import Collection
        normalized_provider = normalize_provider_name(requested_provider)
        embedding_dimension = get_provider_embedding_dimension(requested_provider) or get_provider_embedding_dimension(normalized_provider)
        if embedding_dimension is None:
            error = serializers.ValidationError(
                {"error": "Unsupported provider.", "code": "UNSUPPORTED_PROVIDER"}
            )
            error.status_code = status.HTTP_400_BAD_REQUEST
            raise error
        if Collection.objects.filter(tenant=tenant, name=collection_name).exists():
            conflict = serializers.ValidationError(
                {
                    "error": "Collection name already exists in this tenant. Choose a different collection_name.",
                    "code": "COLLECTION_NAME_TAKEN",
                }
            )
            conflict.status_code = status.HTTP_409_CONFLICT
            raise conflict

        try:
            collection = Collection.objects.create(
                tenant=tenant,
                name=collection_name,
                owner=user,
                is_active=True,
                embedding_dimension=embedding_dimension,
                provider=normalized_provider,
            )
        except Exception as exc:
            logger.error("Failed to create collection for tenant %s: %s", tenant.id, exc)
            raise

        try:
            from .utils import initialize_qdrant_collection

            initialize_qdrant_collection(
                collection.qdrant_collection_name,
                embedding_dimension,
            )
            user.active_collection = collection
            user.active_collection_ready = True
            user.llm_configured = True
            user.is_setup = True
            user.save(update_fields=["active_collection", "active_collection_ready", "llm_configured", "is_setup"])
        except ValueError as e:
            collection.delete()
            logger.warning(
                "Dimension mismatch initializing Qdrant collection '%s': %s",
                collection.qdrant_collection_name,
                e,
            )
            mismatch = serializers.ValidationError(
                {"error": str(e), "code": "QDRANT_DIMENSION_MISMATCH"}
            )
            mismatch.status_code = status.HTTP_409_CONFLICT
            raise mismatch
        except Exception as e:
            collection.delete()
            logger.warning(
                "Failed to initialize Qdrant collection '%s' during registration: %s",
                collection.qdrant_collection_name,
                e,
            )
            unavailable = serializers.ValidationError(
                {"error": "Qdrant unavailable during collection initialization.", "code": "QDRANT_UNAVAILABLE"}
            )
            unavailable.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            raise unavailable

        return user


class LLMSetupSerializer(serializers.Serializer):
    llm_provider = serializers.ChoiceField(choices=[("openai", "openai"), ("ollama", "ollama")])
    collection_name = serializers.CharField(required=False, allow_blank=True, default="default_collection")

    def validate_collection_name(self, value):
        if not value or not value.strip():
            raise serializers.ValidationError("Collection name cannot be empty.")
        return value.strip()

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = User
        fields = (
            'id',
            'username',
            'email',
            'tenant',
            'password',
            'first_name',
            'last_name',
            'language',
            'is_setup',
            'llm_configured',
            'active_collection',
            'active_collection_ready',
            'selected_llm_provider',
        )
        read_only_fields = ('id',)

    def validate_username(self, value):
        if not value.strip():
            raise serializers.ValidationError("Username cannot be empty.")
        return value

    def update(self, instance, validated_data):
        password = validated_data.pop('password', None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if password:
            instance.set_password(password)
        instance.save()
        return instance

class LoginSerializer(serializers.Serializer):
    email = serializers.CharField(help_text="Email address used as the username")
    password = serializers.CharField(help_text="Plain-text password for authentication")

    def validate(self, attrs):
        email = attrs.get('email')
        password = attrs.get('password')

        if email and password:
            user = authenticate(request=self.context.get('request'), username=email, password=password)

            if not user:
                msg = 'Unable to log in with provided credentials.'
                raise serializers.ValidationError(msg, code='authorization')
        else:
            msg = 'Must include "email" and "password".'
            raise serializers.ValidationError(msg, code='authorization')

        attrs['user'] = user
        return attrs


class CollectionSummarySerializer(serializers.Serializer):
    id = serializers.CharField(read_only=True, help_text="Collection identifier")
    name = serializers.CharField(read_only=True, help_text="Collection display name")
    qdrant_collection_name = serializers.CharField(
        read_only=True, help_text="Physical Qdrant collection backing the logical collection"
    )


class RegisterResponseSerializer(serializers.Serializer):
    user = UserSerializer(read_only=True, help_text="Created user profile")
    tenant = TenantSerializer(read_only=True, allow_null=True, help_text="Tenant associated with the user")
    llm_setup_required = serializers.BooleanField(
        help_text="Whether the user still needs to configure an LLM provider and collection"
    )
    active_collection = CollectionSummarySerializer(
        read_only=True, required=False, help_text="Provisioned collection that became active during registration"
    )


class LoginResponseSerializer(serializers.Serializer):
    token = serializers.CharField(help_text="Session token to be supplied as the path parameter for authenticated APIs")
    user = UserSerializer(read_only=True, help_text="Authenticated user profile")
    tenant = TenantSerializer(read_only=True, allow_null=True, help_text="Tenant associated with the user")
    llm_setup_required = serializers.BooleanField(help_text="Flag indicating if LLM setup must still be completed")
    active_collection = CollectionSummarySerializer(
        read_only=True, required=False, help_text="Active collection context returned for the user"
    )
    warnings = serializers.ListField(
        child=serializers.CharField(), required=False, help_text="Any warnings emitted while syncing login state"
    )


class LogoutResponseSerializer(serializers.Serializer):
    message = serializers.CharField(help_text="Confirmation message indicating logout success")


class LLMSetupResponseSerializer(serializers.Serializer):
    tenant_id = serializers.CharField(help_text="Tenant identifier")
    collection_id = serializers.CharField(help_text="Newly created collection identifier")
    collection_name = serializers.CharField(help_text="Human friendly name for the collection")
    qdrant_collection_name = serializers.CharField(help_text="Physical Qdrant collection backing the created collection")


class UserStatusResponseSerializer(serializers.Serializer):
    is_setup = serializers.BooleanField(help_text="Indicates whether initial onboarding steps have completed")
    llm_configured = serializers.BooleanField(help_text="True when an LLM provider has been configured for the user")
    active_collection_ready = serializers.BooleanField(
        help_text="True when the active collection exists and is initialized in the vector store"
    )

    def validate(self, attrs):
        email = attrs.get('email')
        password = attrs.get('password')
        if not email or not password:
            raise serializers.ValidationError("Must include 'email' and 'password'.", code='authorization')
        user = User.objects.filter(email=email).first()
        if not user or not user.check_password(password):
            raise serializers.ValidationError("Invalid credentials.", code='authorization')
        attrs['user'] = user
        return attrs

class VectorStoreSerializer(serializers.ModelSerializer):
    user = serializers.ReadOnlyField(source='user.username')
    metadata = serializers.JSONField(required=False, default=dict)

    class Meta:
        model = VectorStore
        fields = ['id', 'name', 'metadata', 'user', 'collection', 'created_at']
        read_only_fields = ['id', 'user', 'collection', 'created_at']

    def validate_name(self, value):
        if not value.strip():
            raise serializers.ValidationError("Vector store name cannot be empty.")
        return value

    def validate_metadata(self, value):
        return validate_metadata_map(value)

class IngestDocumentSerializer(serializers.Serializer):
    file = serializers.FileField(required=False)
    s3_file_url = serializers.URLField(required=False)
    vector_store_id = serializers.CharField(required=True)

    def validate(self, data):
        if not data.get('file') and not data.get('s3_file_url'):
            raise serializers.ValidationError("Either 'file' or 's3_file_url' must be provided.")
        if data.get('file') and data.get('s3_file_url'):
            raise serializers.ValidationError("Provide only one of 'file' or 's3_file_url'.")
        return data

class DocumentSerializer(serializers.ModelSerializer):
    user = serializers.ReadOnlyField(source='user.username')
    metadata = serializers.JSONField(required=False, default=dict)

    class Meta:
        model = Document
        fields = ['id', 'title', 'vector_store', 'user', 'uploaded_at', 'status', 'metadata']
        read_only_fields = ['id', 'user', 'uploaded_at']

    def validate_title(self, value):
        if not value.strip():
            raise serializers.ValidationError("Document title cannot be empty.")
        return value

    def validate_metadata(self, value):
        return validate_metadata_map(value)

class AssistantSerializer(serializers.ModelSerializer):
    vector_store_id = serializers.CharField(write_only=True, required=False)
    instructions = serializers.CharField(required=False, allow_blank=True)
    model = serializers.CharField(required=False, allow_blank=False)
    tools = serializers.JSONField(required=False, default=list)
    metadata = serializers.JSONField(required=False, default=dict)
    is_default = serializers.BooleanField(required=False, default=False)

    class Meta:
        model = Assistant
        fields = [
            'id',
            'name',
            'vector_store_id',
            'instructions',
            'model',
            'tools',
            'metadata',
            'is_default',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def validate_name(self, value):
        if not value.strip():
            raise serializers.ValidationError("Assistant name cannot be empty.")
        return value

    def validate(self, data):
        request = self.context.get("request")
        user = getattr(request, "user", None)
        try:
            llm_config = get_llm_config(user.id if user else None)
        except Exception as exc:
            raise serializers.ValidationError({"model": f"LLM configuration error: {exc}"})

        default_provider = normalize_provider_name(
            llm_config.get("provider_name")
            or getattr(user, "selected_llm_provider", getattr(settings, "DEFAULT_LLM_PROVIDER", "OpenAI"))
        )

        provided_model = data.get("model")
        if not provided_model or not str(provided_model).strip():
            fallback_model = DEFAULT_OLLAMA_MODEL if default_provider == "Ollama" else DEFAULT_OPENAI_MODEL
            if not fallback_model:
                raise serializers.ValidationError({"model": "Model is required for the selected provider."})
            data["model"] = fallback_model
        model_value = data["model"]
        model_provider = infer_provider_from_model(model_value, default_provider)

        if model_provider != default_provider:
            raise serializers.ValidationError(
                {
                    "model": (
                        f"User default LLM is {default_provider}; cannot configure model "
                        f"'{model_value}' for {model_provider}."
                    )
                }
            )

        try:
            validate_model_availability(
                provider=default_provider,
                model=model_value,
                api_key=llm_config.get("api_key"),
                base_url=llm_config.get("base_url"),
            )
        except (ValueError, LLMProviderError, LLMServiceUnavailableError) as exc:
            # Catch provider errors (e.g., authenticatiom, connection) and return as validation errors
            raise serializers.ValidationError({"model": str(exc)})

        return data

    def validate_metadata(self, value):
        return validate_metadata_map(value)

    def validate_is_default(self, value):
        return bool(value)

    def _get_vector_store(self, vector_store_id):
        if vector_store_id is None:
            return None
        return get_object_or_404(VectorStore, id=vector_store_id)

    def _ensure_default_presence(self, instance, is_default_requested):
        """Prevent unsetting the last default assistant for a tenant."""
        if instance and instance.is_default and not is_default_requested:
            has_other_default = (
                Assistant.objects.filter(tenant=instance.tenant, is_default=True)
                .exclude(pk=instance.pk)
                .exists()
            )
            if not has_other_default:
                raise serializers.ValidationError(
                    {"is_default": "At least one default assistant is required for the tenant."}
                )

    def validate_tools(self, tools):
        if not isinstance(tools, list):
            raise serializers.ValidationError("Tools must be a list.")
        for tool in tools:
            if not isinstance(tool, dict) or "type" not in tool:
                raise serializers.ValidationError("Each tool must be a dict with 'type' key.")
            tool_type = tool["type"]
            if tool_type == "file_search":
                if "file_search" in tool and isinstance(tool["file_search"], dict):
                    ranking_options = tool["file_search"].get("ranking_options", {})
                    if "ranker" not in ranking_options or "score_threshold" not in ranking_options:
                        raise serializers.ValidationError("file_search tool must have valid ranking_options.")
                else:
                    raise serializers.ValidationError("file_search tool must include 'file_search' dict.")
            elif tool_type == "function":
                if "function" not in tool or not isinstance(tool["function"], dict):
                    raise serializers.ValidationError("Function tool must have 'function' dict.")
                func = tool["function"]
                if "name" not in func or "description" not in func or "parameters" not in func:
                    raise serializers.ValidationError("Function must have 'name', 'description', and 'parameters'.")
                params = func["parameters"]
                if not isinstance(params, dict) or "type" not in params or params["type"] != "object":
                    raise serializers.ValidationError("Parameters must be an object with type 'object'.")
                if "properties" not in params or not isinstance(params["properties"], dict):
                    raise serializers.ValidationError("Parameters must have 'properties' dict.")
                for prop_name, prop in params["properties"].items():
                    if "type" not in prop or "description" not in prop:
                        raise serializers.ValidationError(f"Property '{prop_name}' must have 'type' and 'description'.")
                if "required" in params and not isinstance(params["required"], list):
                    raise serializers.ValidationError("'required' must be a list.")
                if "strict" in func and not isinstance(func["strict"], bool):
                    raise serializers.ValidationError("'strict' must be a boolean.")
            else:
                raise serializers.ValidationError(f"Unsupported tool type: {tool_type}")
        return tools

    def create(self, validated_data):
        vector_store_id = validated_data.pop("vector_store_id", None)
        vector_store = self._get_vector_store(vector_store_id)
        if vector_store:
            validated_data["vector_store"] = vector_store
        return super().create(validated_data)

    def update(self, instance, validated_data):
        vector_store_id = validated_data.pop("vector_store_id", None)
        if vector_store_id is not None:
            validated_data["vector_store"] = self._get_vector_store(vector_store_id)

        # Prevent removing the only default assistant within the tenant.
        is_default_requested = validated_data.get("is_default", instance.is_default)
        self._ensure_default_presence(instance, is_default_requested)

        updated_instance = super().update(instance, validated_data)

        # Ensure only one default remains per tenant after updates.
        if updated_instance.is_default:
            Assistant.objects.filter(tenant=updated_instance.tenant, is_default=True).exclude(pk=updated_instance.pk).update(
                is_default=False
            )

        return updated_instance

class ThreadSerializer(serializers.ModelSerializer):
    vector_store_id = serializers.CharField(write_only=True, required=False, allow_null=True)
    vector_store_id_read = serializers.SerializerMethodField(read_only=True)
    metadata = serializers.JSONField(required=False, default=dict)

    class Meta:
        model = Thread
        fields = ['id', 'title', 'metadata', 'created_at', 'vector_store_id', 'vector_store_id_read']
        read_only_fields = ['id', 'created_at', 'vector_store_id_read']

    def get_vector_store_id_read(self, obj):
        return str(obj.vector_store.id) if obj.vector_store else None
    def validate_vector_store_id(self, value):
        if value is None:
            return value
        try:
            VectorStore.objects.get(id=value)
        except VectorStore.DoesNotExist:
            raise serializers.ValidationError("Invalid vector_store_id.")
        return value

    def validate_title(self, value):
        """Validate the title field"""
        if value and len(value.strip()) < 1:
            raise serializers.ValidationError("Title cannot be empty.")
        if value and len(value) > 255:
            raise serializers.ValidationError("Title cannot exceed 255 characters.")
        return value.strip() if value else value

    def validate_metadata(self, value):
        return validate_metadata_map(value)

class MessageSerializer(serializers.ModelSerializer):
    thread_id = serializers.CharField(write_only=True)

    user = serializers.ReadOnlyField(source='user.username')
    metadata = serializers.JSONField(required=False, default=dict)

    class Meta:
        model = Message
        fields = ['id', 'thread_id', 'user', 'role', 'content', 'metadata', 'created_at']
        read_only_fields = ['id', 'user', 'role', 'created_at']

    def validate_content(self, value):
        if not value.strip():
            raise serializers.ValidationError("Message content cannot be empty.")
        return value

    def validate_thread_id(self, value):
        try:
            Thread.objects.get(id=value)
        except Thread.DoesNotExist:
            raise serializers.ValidationError("Invalid thread_id.")
        return value

    def validate_metadata(self, value):
        # Only validate on input (write), not on output (read)
        # This allows lists for used_document_ids
        return validate_metadata_map(value) if value else {}
    
    def to_representation(self, instance):
        """Ensure metadata is returned as-is from database, preserving used_document_ids list."""
        data = super().to_representation(instance)
        # Metadata from database should already be correct, but ensure it's preserved
        if hasattr(instance, 'metadata') and instance.metadata:
            data['metadata'] = instance.metadata
        return data

class RunModeSerializer(serializers.Serializer):
    mode = serializers.ChoiceField(choices=["document", "normal", "web"], default="document")

class RunSerializer(serializers.ModelSerializer):
    thread_id = serializers.CharField(write_only=True)
    assistant_id = serializers.CharField(write_only=True, required=False, allow_blank=True, allow_null=True)
    message_id = serializers.IntegerField(write_only=True, required=False)
    source_run_id = serializers.CharField(write_only=True, required=False)
    required_action = serializers.JSONField(read_only=True)
    tool_outputs = serializers.JSONField(write_only=True, required=False)
    mode = serializers.CharField(required=False)
    metadata = serializers.JSONField(required=False, default=dict)
    source_message_id = serializers.IntegerField(read_only=True, source='source_message.id')
    rerun_of_id = serializers.CharField(read_only=True, source='rerun_of.id')

    class Meta:
        model = Run
        fields = [
            'id',
            'thread_id',
            'status',
            'assistant_id',
            'mode',
            'metadata',
            'required_action',
            'tool_outputs',
            'created_at',
            'completed_at',
            'cancelled_at',
            'message_id',
            'source_run_id',
            'source_message_id',
            'rerun_of_id',
        ]
        read_only_fields = ['id', 'created_at', 'completed_at', 'cancelled_at', 'source_message_id', 'rerun_of_id']

    def validate_thread_id(self, value):
        try:
            Thread.objects.get(id=value)
        except Thread.DoesNotExist:
            raise serializers.ValidationError("Invalid thread_id.")
        return value

    def validate_assistant_id(self, value):
        if value in (None, ""):
            return None
        try:
            Assistant.objects.get(id=value)
        except Assistant.DoesNotExist:
            raise serializers.ValidationError("Invalid assistant_id.")
        return value

    def validate_mode(self, value):
        data = {"mode": value or "document"}
        serializer = RunModeSerializer(data=data)
        serializer.is_valid(raise_exception=True)
        return serializer.validated_data["mode"]

    def validate_metadata(self, value):
        return validate_metadata_map(value)

class SubmitToolOutputsSerializer(serializers.Serializer):
    tool_outputs = serializers.ListField(
        child=serializers.DictField(),
        required=True,
        help_text="List of {'tool_call_id': str, 'output': str}"
    )

    def validate_tool_outputs(self, value):
        if not value:
            raise serializers.ValidationError("tool_outputs cannot be empty.")
        for output in value:
            if "tool_call_id" not in output or "output" not in output:
                raise serializers.ValidationError("Each tool output must have 'tool_call_id' and 'output'.")
        return value

class DocumentAccessSerializer(serializers.ModelSerializer):
    document_ids = serializers.ListField(
        child=serializers.CharField(),
        write_only=True,
        help_text="List of document IDs to grant access to"
    )
    vector_store_id = serializers.CharField(write_only=True)
    granted_by = serializers.PrimaryKeyRelatedField(read_only=True)

    class Meta:
        model = DocumentAccess
        fields = ['id', 'document_ids', 'vector_store_id', 'granted_by', 'granted_at']
        read_only_fields = ['id', 'granted_at']

    def validate(self, data):
        vector_store_id = data.get('vector_store_id')
        document_ids = data.get('document_ids')
        user = self.context['request'].user

        # Validate vector_store_id
        if not vector_store_id:
            raise serializers.ValidationError({"vector_store_id": "This field is required."})
        vector_store = get_object_or_404(VectorStore, id=vector_store_id, tenant=user.tenant)

        # Validate document_ids
        if not document_ids:
            raise serializers.ValidationError({"document_ids": "At least one document ID is required."})
        invalid_ids = []
        valid_documents = []
        for doc_id in document_ids:
            try:
                doc = Document.objects.get(id=doc_id, tenant=user.tenant)
                valid_documents.append(doc)
            except Document.DoesNotExist:
                invalid_ids.append(doc_id)
        if invalid_ids:
            raise serializers.ValidationError({
                "document_ids": f"Invalid document IDs: {', '.join(invalid_ids)}"
            })

        # Check for existing access to prevent duplicates
        existing_access = DocumentAccess.objects.filter(
            vector_store=vector_store,
            document__in=valid_documents
        ).values_list('document__id', flat=True)
        existing_ids = [str(doc_id) for doc_id in existing_access]
        duplicate_ids = [doc_id for doc_id in document_ids if doc_id in existing_ids]
        if duplicate_ids:
            raise serializers.ValidationError({
                "document_ids": f"Access already granted for document IDs: {', '.join(duplicate_ids)}"
            })

        data['vector_store'] = vector_store
        data['documents'] = valid_documents
        return data
    
class DocumentAccessListSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentAccess
        fields = '__all__'

class DocumentAccessRemoveSerializer(serializers.Serializer):
    document_ids = serializers.ListField(
        child=serializers.CharField(),
        write_only=True,
        help_text="List of document IDs to remove access from"
    )
    vector_store_id = serializers.CharField(write_only=True)

    def validate(self, data):
        vector_store_id = data.get('vector_store_id')
        document_ids = data.get('document_ids')
        user = self.context['request'].user

        # Validate vector_store_id
        if not vector_store_id:
            raise serializers.ValidationError({"vector_store_id": "This field is required."})
        vector_store = get_object_or_404(VectorStore, id=vector_store_id, tenant=user.tenant)

        # Validate document_ids
        if not document_ids:
            raise serializers.ValidationError({"document_ids": "At least one document ID is required."})
        invalid_ids = []
        valid_document_ids = []
        for doc_id in document_ids:
            if Document.objects.filter(id=doc_id, tenant=user.tenant).exists():
                valid_document_ids.append(doc_id)
            else:
                invalid_ids.append(doc_id)
        if invalid_ids:
            raise serializers.ValidationError({
                "document_ids": f"Invalid document IDs: {', '.join(invalid_ids)}"
            })

        data['vector_store'] = vector_store
        data['valid_document_ids'] = valid_document_ids
        return data


class MessageFeedbackSerializer(serializers.ModelSerializer):
    message_id = serializers.IntegerField(write_only=True)
    message = serializers.PrimaryKeyRelatedField(read_only=True)
    rating = serializers.ChoiceField(choices=MessageFeedback.RATING_CHOICES)

    class Meta:
        model = MessageFeedback
        fields = ['id', 'message', 'message_id', 'rating', 'created_at', 'updated_at']
        read_only_fields = ['id', 'message', 'created_at', 'updated_at']

    def create(self, validated_data):
        message_id = validated_data.pop('message_id')
        request = self.context.get('request')
        user = getattr(request, 'user', None)
        if user is None or not user.is_authenticated:
            raise serializers.ValidationError({"message_id": "Authentication required to submit feedback."})

        try:
            message = Message.objects.select_related('thread__tenant', 'thread__user').get(id=message_id)
        except Message.DoesNotExist:
            raise serializers.ValidationError({"message_id": "Message not found."})

        if message.thread.tenant != user.tenant or message.thread.user != user:
            raise serializers.ValidationError({"message_id": "You can only rate messages in your own threads."})

        if message.role != 'assistant':
            raise serializers.ValidationError({"message_id": "Feedback can only be provided on assistant messages."})

        feedback, created = MessageFeedback.objects.update_or_create(
            message=message,
            user=user,
            defaults={'rating': validated_data['rating']}
        )
        self._created = created
        return feedback

    def to_representation(self, instance):
        data = super().to_representation(instance)
        data['message_id'] = instance.message_id
        return data

class LLMProviderConfigSerializer(serializers.ModelSerializer):
    api_key = serializers.CharField(allow_blank=True, required=False)

    class Meta:
        model = LLMProviderConfig
        fields = ['id', 'api_key', 'name', 'model', 'provider', 'chat_config', 'base_url', 'is_valid', 'is_active']
        read_only_fields = ['id', 'is_valid']

    def validate(self, data):
        provider = data.get('provider') or getattr(getattr(self, 'instance', None), 'provider', None)
        api_key = data.get('api_key')
        model = data.get('model')

        # Enforce alignment with the user's default/provider selection.
        request = self.context.get('request')
        user = getattr(request, "user", None)
        if user:
            default_provider = getattr(user, "selected_llm_provider", getattr(settings, "DEFAULT_LLM_PROVIDER", "OpenAI"))
            normalized_default = normalize_provider_name(default_provider)
            normalized_requested = normalize_provider_name(provider or normalized_default)
            provider = normalized_requested
            data['provider'] = normalized_requested

            provider_locked = bool(getattr(user, "is_setup", False) and getattr(user, "active_collection_ready", False))
            if provider_locked and provider and normalized_requested != normalized_default:
                raise serializers.ValidationError(
                    {
                        "provider": (
                            f"Default LLM provider is {normalized_default}; cannot configure {provider} "
                            "while collection is already created."
                        )
                    }
                )

        # Set default model if not provided
        if provider == 'OpenAI':
            if not model:
                model = DEFAULT_OPENAI_MODEL
            data['model'] = model

            if not api_key:
                raise serializers.ValidationError({"api_key": "API key is required for OpenAI provider."})

            from .llm_providers import OpenAIProvider
            provider_inst = OpenAIProvider()
            if provider_inst.validate_credentials(api_key=api_key):
                data['is_valid'] = True
            else:
                raise serializers.ValidationError({"api_key": "Invalid OpenAI API key or provider is unreachable."})


        elif provider == 'Ollama':
            if not model:
                model = DEFAULT_OLLAMA_MODEL
            data['model'] = model
            data['base_url'] = data.get('base_url') or getattr(settings, 'OLLAMA_API_URL', getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434'))
            # Assuming Ollama doesn't require API key validation here
            data['is_valid'] = True

        else:
            raise serializers.ValidationError({"provider": f"Unsupported provider: {provider}"})

        return data

    def create(self, validated_data):
        user = self.context['request'].user
        is_active = validated_data.get('is_active', False)
        if not LLMProviderConfig.objects.filter(user=user).exists():
            is_active = True
            validated_data['is_active'] = True
        if is_active:
            LLMProviderConfig.objects.filter(user=user).update(is_active=False)
        instance = super().create(validated_data)
        user.selected_llm_provider = instance.provider
        user.is_setup = True
        user.save(update_fields=['selected_llm_provider', 'is_setup'])
        return instance

    def update(self, instance, validated_data):
        user = self.context['request'].user
        updated_instance = super().update(instance, validated_data)

        if validated_data.get('is_active'):
            LLMProviderConfig.objects.filter(user=user).exclude(id=updated_instance.id).update(is_active=False)
            user.selected_llm_provider = updated_instance.provider
            user.is_setup = True
            user.save(update_fields=['selected_llm_provider', 'is_setup'])

        return updated_instance

class DocumentAlertSerializer(serializers.ModelSerializer):
    user = serializers.ReadOnlyField(source='user.username')

    class Meta:
        model = DocumentAlert
        fields = ['id', 'document', 'user', 'keyword', 'snippet', 'created_at']
        read_only_fields = ['id', 'user', 'created_at']

    def validate_keyword(self, value):
        if not value.strip():
            raise serializers.ValidationError("Keyword cannot be empty.")
        return value


class ResponsesCreateSerializer(serializers.Serializer):
    """Serializer for Responses API request body."""

    model = serializers.CharField(
        required=True,
        help_text="Fully-qualified model name to use for the chat completion (e.g. gpt-4.1).",
    )
    conversation = serializers.CharField(
        required=False,
        allow_null=True,
        help_text="Existing conversation ID to append messages to. If omitted, no conversation history is stored.",
    )
    instructions = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="Optional system prompt that overrides the default assistant instructions.",
    )
    input = serializers.ListField(
        required=True,
        child=serializers.DictField(),
        help_text=(
            "Array of chat messages. Each item must include 'role', 'content', and can include 'metadata'. "
            "Content supports OpenAI-style message content blocks."
        ),
    )
    tools = serializers.ListField(
        required=False,
        allow_null=True,
        child=serializers.DictField(),
        help_text="Optional document tools. Only type 'document' with 'vector_store_ids' is supported.",
    )
    metadata = serializers.JSONField(
        required=False,
        allow_null=True,
        default=dict,
        help_text="Arbitrary key/value metadata echoed back on stored and returned assistant messages.",
    )

    def validate_metadata(self, value):
        """Validate metadata - optional, can be any JSON object."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise serializers.ValidationError("metadata must be an object.")
        return value

    def validate_tools(self, value):
        """Validate tools array - only 'document' type allowed."""
        if not value:
            return value
        for tool in value:
            tool_type = tool.get('type')
            if tool_type != 'document':
                raise serializers.ValidationError(
                    f"Invalid tool type '{tool_type}'. Only 'document' type is allowed."
                )
            if 'vector_store_ids' not in tool:
                raise serializers.ValidationError(
                    "Tool of type 'document' must include 'vector_store_ids' field."
                )
        return value

    def validate_input(self, value):
        """Validate input array structure."""
        if not value:
            raise serializers.ValidationError("Input array cannot be empty.")
        for item in value:
            if 'role' not in item or 'content' not in item:
                raise serializers.ValidationError(
                    "Each input item must have 'role' and 'content' fields."
                )
        return value


class ConversationItemsSerializer(serializers.ModelSerializer):
    """Serializer for conversation items (messages)."""
    class Meta:
        model = ConversationMessage
        fields = ['id', 'role', 'content', 'created_at', 'metadata']
        read_only_fields = ['id', 'created_at']
    
    def to_representation(self, instance):
        """Ensure metadata is returned as-is from database, preserving used_document_ids list."""
        data = super().to_representation(instance)
        # Metadata from database should already be correct, but ensure it's preserved
        if hasattr(instance, 'metadata') and instance.metadata:
            data['metadata'] = instance.metadata
        return data


class ConversationSerializer(serializers.ModelSerializer):
    """Serializer for Conversation retrieve/update operations."""

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']


class ConversationCreateRequestSerializer(serializers.Serializer):
    """Request body for creating a conversation."""

    title = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="Optional title for the conversation. If omitted, the first user message becomes the title when provided.",
    )


class ConversationCreateResponseSerializer(serializers.Serializer):
    """Response payload returned when a conversation is created."""

    id = serializers.CharField(help_text="Conversation identifier.")
    created_at = serializers.DateTimeField(help_text="ISO-8601 timestamp when the conversation was created.")
    title = serializers.CharField(
        allow_blank=True,
        required=False,
        help_text="The conversation title. May be empty if not supplied or inferred yet.",
    )


class ResponsesContentSerializer(serializers.Serializer):
    """Individual content block within a response message."""

    type = serializers.CharField(help_text="Content kind, such as 'output_text'.")
    text = serializers.CharField(help_text="Rendered assistant text for this block.")


class ResponsesMessageSerializer(serializers.Serializer):
    """Assistant message representation in Responses API output."""

    message_id = serializers.CharField(
        allow_null=True,
        help_text="ConversationMessage ID when persisted, otherwise null.",
    )
    type = serializers.CharField(help_text="Fixed value 'message' for message outputs.")
    status = serializers.CharField(help_text="Completion status, typically 'completed'.")
    role = serializers.CharField(help_text="Role of the message author, e.g. 'assistant'.")
    metadata = serializers.JSONField(
        required=False,
        help_text="Metadata stored alongside the assistant message.",
    )
    content = ResponsesContentSerializer(
        many=True,
        help_text="Ordered content blocks returned by the assistant.",
    )


class ResponsesOutputSerializer(serializers.Serializer):
    """Serializer documenting the Responses API output envelope."""

    id = serializers.CharField(help_text="Server-generated response identifier.")
    created_at = serializers.DateTimeField(help_text="ISO-8601 timestamp when response generation started.")
    status = serializers.CharField(help_text="Overall response status, such as 'completed'.")
    completed_at = serializers.DateTimeField(help_text="ISO-8601 timestamp when generation finished.")
    instructions = serializers.CharField(
        allow_blank=True,
        required=False,
        help_text="System instructions applied to the run.",
    )
    model = serializers.CharField(help_text="Model used for the LLM call.")
    output = ResponsesMessageSerializer(
        many=True,
        help_text="Assistant messages returned for the request.",
    )
    sources = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_empty=True,
        help_text="Document IDs actually used by the LLM to generate the response (LLM-attributed sources).",
    )
    user_id = serializers.CharField(help_text="User identifier associated with the response.")
    metadata = serializers.JSONField(help_text="Metadata echoed from the request and stored with the assistant message.")


class ResponseRecordSerializer(serializers.ModelSerializer):
    """Serializer for retrieving stored response records."""

    class Meta:
        model = ResponseRecord
        fields = [
            'id', 'conversation', 'model', 'instructions', 'input_messages', 'output',
            'metadata', 'status', 'error_message', 'created_at', 'updated_at'
        ]
        read_only_fields = fields
