# rag/models.py


from django.db import models
from django.contrib.auth.models import AbstractUser
import uuid
from django.core.exceptions import ValidationError
from project import settings
import requests
from django.utils import timezone
from django.db.models import Q
from django.utils.text import slugify

# Define generator functions at module level
def generate_prefixed_uuid_doc():
    return f"doc_{uuid.uuid4()}"

def generate_prefixed_uuid_vs():
    return f"vs_{uuid.uuid4()}"

def generate_prefixed_uuid_run():
    return f"run_{uuid.uuid4()}"

def generate_prefixed_uuid_thread():
    return f"thread_{uuid.uuid4()}"

def generate_prefixed_uuid_assistant():
    return f"assistant_{uuid.uuid4()}"

def generate_prefixed_uuid_response():
    return f"resp_{uuid.uuid4().hex[:32]}"

def generate_prefixed_uuid_conversation():
    return f"conv_{uuid.uuid4().hex}"

class Tenant(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name


def generate_qdrant_collection_name(tenant_id: int, name: str) -> str:
    slug = slugify(name) or "collection"
    return f"t_{tenant_id}__c_{slug}__{uuid.uuid4().hex[:8]}"


class Collection(models.Model):
    PROVIDER_CHOICES = [
        ("OpenAI", "OpenAI"),
        ("Ollama", "Ollama"),
        ("Claude", "Claude"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="collections")
    name = models.CharField(max_length=255)
    owner = models.ForeignKey(
        "User", on_delete=models.SET_NULL, related_name="collections", null=True, blank=True
    )
    is_active = models.BooleanField(default=True, db_index=True)
    qdrant_collection_name = models.CharField(max_length=255, unique=True)
    embedding_dimension = models.IntegerField(null=True, blank=True)
    provider = models.CharField(max_length=50, choices=PROVIDER_CHOICES, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["tenant", "name"],
                name="unique_collection_name_per_tenant",
            ),
            models.UniqueConstraint(
                fields=["owner"],
                condition=models.Q(is_active=True),
                name="unique_active_collection_per_user",
            ),
        ]
        indexes = [
            models.Index(fields=["tenant", "is_active"], name="collection_tenant_active_idx"),
        ]

    def save(self, *args, **kwargs):
        if not self.qdrant_collection_name and self.tenant_id:
            self.qdrant_collection_name = generate_qdrant_collection_name(
                self.tenant_id, self.name
            )
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} ({self.qdrant_collection_name})"

class User(AbstractUser):
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="users")
    language = models.CharField(max_length=10, default="en", blank=True)
    is_setup = models.BooleanField(default=False)
    llm_configured = models.BooleanField(
        default=False,
        help_text="Indicates whether the user has configured an LLM and collection.",
        db_index=True,
    )
    active_collection = models.ForeignKey(
        Collection, on_delete=models.SET_NULL, related_name="active_users", null=True, blank=True
    )
    active_collection_ready = models.BooleanField(
        default=False,
        help_text="Indicates whether the active collection has been provisioned for this user.",
        db_index=True,
    )
    selected_llm_provider = models.CharField(
        max_length=50,
        choices=[("OpenAI", "OpenAI"), ("Ollama", "Ollama"), ("Claude", "Claude")],
        null=True,
        blank=True,
        default=None,
        help_text="The LLM provider currently selected for this user.",
        db_index=True,
    )

class LLMProviderConfig(models.Model):
    PROVIDER_CHOICES = [
        ('OpenAI', 'OpenAI'),
        ('Ollama', 'Ollama'),
        ('Claude', 'Claude'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="llm_provider_configs")
    name = models.CharField(max_length=255, blank=True, null=True, help_text="Optional name for the credential or endpoint")
    model = models.CharField(max_length=50, default="gpt-4.1", help_text="Preferred chat model to use with this configuration")
    api_key = models.CharField(max_length=255)
    provider = models.CharField(max_length=50, choices=PROVIDER_CHOICES, default='OpenAI', db_index=True)
    base_url = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Optional base URL for provider requests (e.g., Ollama host).",
    )
    chat_config = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional generation controls to override provider defaults (e.g., temperature, max_tokens).",
    )
    is_valid = models.BooleanField(default=False, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=False, help_text="Is this configuration currently active for the user?", db_index=True)
    deactivated_at = models.DateTimeField(null=True, blank=True, help_text="Timestamp when the configuration was last deactivated.")

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['user', 'is_active'], condition=models.Q(is_active=True), name='unique_active_key_per_user')
        ]
        indexes = [
            models.Index(fields=["user", "provider"], name="llm_config_user_provider_idx"),
            models.Index(fields=["user", "is_valid", "is_active"], name="llm_config_user_state_idx"),
        ]

    def __str__(self):
        return f"{self.provider} configuration for {self.user.username}"

    def clean(self):
        if not self.api_key and self.provider == 'OpenAI':
            raise ValidationError("API key cannot be empty for OpenAI provider.")
        # The is_valid field will be set by the serializer after provider validation.
        # For now, the clean method can simply pass or perform very basic checks if needed.
        pass

    def save(self, *args, **kwargs):
        if self.pk is not None:
            original_instance = LLMProviderConfig.objects.get(pk=self.pk)
            if not original_instance.is_active and self.is_active:
                # Configuration is being activated
                LLMProviderConfig.objects.filter(user=self.user, is_active=True).exclude(pk=self.pk).update(is_active=False, deactivated_at=timezone.now())
            elif original_instance.is_active and not self.is_active:
                # Configuration is being deactivated
                self.deactivated_at = timezone.now()
        elif self.is_active:
            # New configuration is being created as active
            LLMProviderConfig.objects.filter(user=self.user, is_active=True).update(is_active=False, deactivated_at=timezone.now())

        super().save(*args, **kwargs)

class VectorStore(models.Model):
    id = models.CharField(primary_key=True, max_length=50, default=generate_prefixed_uuid_vs, editable=False)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="vector_stores")
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="owned_vector_stores", null=True, blank=True)
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE, related_name="vector_stores", null=True, blank=True)
    name = models.CharField(max_length=255)
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional key/value metadata for this vector store (max 16 entries).",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["tenant", "user", "collection"], name="vs_tenant_user_col_idx"),
            models.Index(fields=["tenant", "collection"], name="vector_store_tenant_col_idx"),
        ]

    def __str__(self):
        return f"VectorStore {self.id} - {self.name}"

class Document(models.Model):
    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    id = models.CharField(primary_key=True, max_length=50, default=generate_prefixed_uuid_doc, editable=False)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="documents")
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="owned_documents", null=True, blank=True)
    vector_store = models.ForeignKey(VectorStore, on_delete=models.CASCADE, related_name="documents")
    title = models.CharField(max_length=255)
    content = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True, db_index=True)
    summary = models.TextField(blank=True, null=True)
    keywords = models.JSONField(blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued', db_index=True)  # New field
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional key/value metadata for this document (max 16 entries).",
    )

    class Meta:
        indexes = [
            models.Index(fields=["tenant", "user", "uploaded_at"], name="doc_tenant_user_upload_idx"),
            models.Index(fields=["tenant", "vector_store"], name="document_tenant_store_idx"),
            models.Index(fields=["user", "vector_store"], name="document_user_store_idx"),
            models.Index(fields=["status", "uploaded_at"], name="document_status_uploaded_idx"),
        ]

    def __str__(self):
        return self.title

class Assistant(models.Model):
    id = models.CharField(primary_key=True, max_length=50, default=generate_prefixed_uuid_assistant, editable=False)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="assistants")
    name = models.CharField(max_length=255)
    vector_store = models.ForeignKey(VectorStore, on_delete=models.CASCADE, related_name="assistants", null=True, blank=True)
    instructions = models.TextField(
        null=True,
        blank=True,
        help_text="Custom instructions for the assistant, used as the system prompt during runs."
    )
    model = models.CharField(max_length=100, help_text="LLM model to use for this assistant.")
    tools = models.JSONField(default=list, blank=True, help_text="List of tools (JSON schemas) for the assistant.")
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional key/value metadata for this assistant (max 16 entries).",
    )
    is_default = models.BooleanField(
        default=False,
        help_text="Marks this assistant as the tenant's default when no assistant is specified for a run.",
    )
    creator = models.ForeignKey(User, on_delete=models.CASCADE, related_name="created_assistants", null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["tenant", "is_default"],
                condition=models.Q(is_default=True),
                name="unique_default_assistant_per_tenant",
            )
        ]
        indexes = [
            models.Index(fields=["tenant", "is_default"], name="assistant_tenant_default_idx"),
        ]

    def save(self, *args, **kwargs):
        if self.pk is None and not self.is_default:
            if not Assistant.objects.filter(tenant=self.tenant).exists():
                self.is_default = True

        if self.is_default:
            Assistant.objects.filter(tenant=self.tenant, is_default=True).exclude(pk=self.pk).update(is_default=False)

        super().save(*args, **kwargs)


    def __str__(self):
        return f"Assistant {self.id} - {self.name}"

class Thread(models.Model):
    id = models.CharField(primary_key=True, max_length=50, default=generate_prefixed_uuid_thread, editable=False)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="threads")
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="owned_threads", null=True, blank=True)
    vector_store = models.ForeignKey(VectorStore, on_delete=models.CASCADE, related_name="threads", null=True, blank=True)
    title = models.CharField(max_length=255, blank=True, null=True, help_text="Auto-generated title based on first user message")
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional key/value metadata for this thread (max 16 entries).",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["tenant", "user", "created_at"], name="thread_tenant_user_created_idx"),
        ]

    def __str__(self):
        return f"Thread {self.id} - {self.title or 'Untitled'}"

class Conversation(models.Model):
    """Model for OpenAI-style conversations (separate from Thread)."""
    id = models.CharField(primary_key=True, max_length=50, default=generate_prefixed_uuid_conversation, editable=False)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="conversations")
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="owned_conversations", null=True, blank=True)
    title = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Optional title for the conversation, typically derived from the first user message.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["tenant", "user", "created_at"], name="conv_tenant_user_created_idx"),
        ]

    def __str__(self):
        return f"Conversation {self.id}"

class ConversationMessage(models.Model):
    """Messages for conversations (separate from Thread messages)."""
    id = models.AutoField(primary_key=True)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="owned_conversation_messages", null=True, blank=True)
    role = models.CharField(max_length=20)
    content = models.TextField()
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional key/value metadata for this message (max 16 entries).",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["conversation", "created_at"], name="conversation_msg_created_idx"),
            models.Index(fields=["user", "created_at"], name="convmsg_user_created_idx"),
        ]

    def __str__(self):
        return f"ConversationMessage in {self.conversation.id} by {self.role}"


class ResponseRecord(models.Model):
    STATUS_IN_PROGRESS = "in_progress"
    STATUS_COMPLETED = "completed"
    STATUS_CANCELLED = "cancelled"
    STATUS_FAILED = "failed"

    STATUS_CHOICES = [
        (STATUS_IN_PROGRESS, "In Progress"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_CANCELLED, "Cancelled"),
        (STATUS_FAILED, "Failed"),
    ]

    id = models.CharField(
        primary_key=True,
        max_length=50,
        default=generate_prefixed_uuid_response,
        editable=False,
    )
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name="responses",
        null=True,
        blank=True,
    )
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="responses")
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="owned_responses", null=True, blank=True)
    model = models.CharField(max_length=100)
    instructions = models.TextField(blank=True)
    input_messages = models.JSONField(default=list, blank=True)
    output = models.JSONField(default=list, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_IN_PROGRESS, db_index=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"ResponseRecord {self.id} ({self.status})"

    class Meta:
        indexes = [
            models.Index(fields=["tenant", "user", "created_at"], name="resp_tenant_user_created_idx"),
            models.Index(fields=["status", "created_at"], name="response_status_created_idx"),
        ]

class Message(models.Model):
    id = models.AutoField(primary_key=True)
    thread = models.ForeignKey(Thread, on_delete=models.CASCADE, related_name="messages")
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="owned_messages", null=True, blank=True)
    role = models.CharField(max_length=20)
    content = models.TextField()
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional key/value metadata for this message (max 16 entries).",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["thread", "created_at"], name="message_thread_created_idx"),
            models.Index(fields=["user", "created_at"], name="message_user_created_idx"),
        ]

    def __str__(self):
        return f"Message in {self.thread.id} by {self.role}"


class MessageFeedback(models.Model):
    RATING_GOOD = "good"
    RATING_BAD = "bad"
    RATING_CHOICES = [
        (RATING_GOOD, "Good"),
        (RATING_BAD, "Bad"),
    ]

    id = models.AutoField(primary_key=True)
    message = models.ForeignKey(Message, on_delete=models.CASCADE, related_name="feedback")
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="feedback")
    rating = models.CharField(max_length=10, choices=RATING_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("message", "user")

    def __str__(self):
        return f"Feedback by {self.user.username} on message {self.message.id}: {self.rating}"


class Run(models.Model):
    id = models.CharField(primary_key=True, max_length=50, default=generate_prefixed_uuid_run, editable=False)
    thread = models.ForeignKey(Thread, on_delete=models.CASCADE, related_name="runs")
    assistant = models.ForeignKey(Assistant, on_delete=models.CASCADE, related_name="runs")

    RUN_MODE_CHOICES = [
        ("document", "Document"),
        ("normal", "Normal"),
        ("web", "Web"),
    ]

    STATUS_QUEUED = "queued"
    STATUS_IN_PROGRESS = "in_progress"
    STATUS_REQUIRES_ACTION = "requires_action"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"
    STATUS_CHOICES = [
        (STATUS_QUEUED, "Queued"),
        (STATUS_IN_PROGRESS, "In Progress"),
        (STATUS_REQUIRES_ACTION, "Requires Action"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_FAILED, "Failed"),
        (STATUS_CANCELLED, "Cancelled"),
    ]
    TERMINAL_STATUSES = {STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED}

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_QUEUED, db_index=True)
    mode = models.CharField(max_length=20, choices=RUN_MODE_CHOICES, default="document")
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional key/value metadata for this run (max 16 entries).",
    )
    required_action = models.JSONField(null=True, blank=True, help_text="Details for required actions like tool calls.")
    tool_outputs = models.JSONField(null=True, blank=True, help_text="Submitted tool outputs for resumption.")
    source_message = models.ForeignKey(
        Message,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="triggered_runs",
        help_text="Original user message that triggered this run, when available.",
    )
    rerun_of = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="reruns",
        help_text="Reference to the run that this run is rerunning, if any.",
    )
    cancelled_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Run {self.id} in {self.thread.id}"

    class Meta:
        indexes = [
            models.Index(fields=["thread", "created_at"], name="run_thread_created_idx"),
            models.Index(fields=["status", "created_at"], name="run_status_created_idx"),
        ]

    @property
    def assistant_intro(self) -> str:
        if self.mode == "normal":
            return (
                "Act as an I'm RAGitify Assistant, here to assist you by answering your questions. "
                "Let me know how I can help!"
            )
        if self.mode == "web":
            return (
                "I'm the RAGitify Assistant, here to help you with your questions using web search results."
            )
        return ""

class DocumentAlert(models.Model):
    id = models.AutoField(primary_key=True)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='alerts')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="owned_document_alerts", null=True, blank=True)
    keyword = models.CharField(max_length=255)
    snippet = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Alert on {self.document.title}: {self.keyword}"

    class Meta:
        indexes = [
            models.Index(fields=["document", "created_at"], name="doc_alert_document_created_idx"),
            models.Index(fields=["user", "created_at"], name="doc_alert_user_created_idx"),
        ]

class DocumentAccess(models.Model):
    id = models.AutoField(primary_key=True)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="access")
    vector_store = models.ForeignKey(VectorStore, on_delete=models.CASCADE, related_name="access")
    granted_by = models.ForeignKey(User, on_delete=models.CASCADE)
    granted_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('document', 'vector_store')
        indexes = [
            models.Index(fields=["vector_store", "document"], name="doc_access_store_doc_idx"),
        ]
