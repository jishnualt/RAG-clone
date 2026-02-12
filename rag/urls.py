#  rag/urls.py

from django.urls import path
from .views import (
    RegisterAPI, LoginView, LogoutView, ProtectedView,
    TenantCreateAPIView, TenantListAPIView, TenantRetrieveUpdateDestroyAPIView,
    UserListAPIView, UserRetrieveUpdateDestroyAPIView, UserFullDeleteAPIView,
    UserStatusAPIView,
    VectorStoreCreateAPIView, VectorStoreListAPIView, VectorStoreRetrieveUpdateDestroyAPIView,
    IngestAPIView, DocumentListAPIView, DocumentRetrieveUpdateDestroyAPIView,
    AssistantCreateAPIView, AssistantListAPIView, AssistantRetrieveUpdateDestroyAPIView,
    ThreadCreateAPIView, ThreadListAPIView, ThreadRetrieveUpdateDestroyAPIView, ThreadMessagesAPIView,
    MessageCreateAPIView, MessageListAPIView, MessageRetrieveUpdateDestroyAPIView,
    RunCreateAPIView, RunListAPIView, RunRetrieveUpdateDestroyAPIView, RunCancelAPIView, RunRerunAPIView,
    DocumentAccessCreateAPIView, DocumentAccessListAPIView, DocumentAccessRetrieveUpdateDestroyAPIView,
    DocumentAlertCreateAPIView, DocumentAlertListAPIView, DocumentAlertRetrieveUpdateDestroyAPIView,
    LLMProviderConfigCreateAPIView, LLMProviderConfigListAPIView, LLMProviderConfigRetrieveUpdateDestroyAPIView,
    OpenAIKeyCreateAPIView, OpenAIKeyListAPIView, OpenAIKeyRetrieveUpdateDestroyAPIView,
    DocumentAccessRemoveAPIView,DocumentStatusAPIView,
    SubmitToolOutputsAPIView, MessageFeedbackCreateAPIView, MessageFeedbackListAPIView,
    LLMSetupAPIView,
    ConversationCreateAPIView, ConversationRetrieveUpdateDestroyAPIView, ResponsesAPIView, ConversationItemsAPIView,
    ResponseRetrieveDestroyAPIView, ResponseCancelAPIView,
)
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

# Create the schema view as usual
schema_view = get_schema_view(
    openapi.Info(
        title="RAG API",
        default_version='v1',
        description="API documentation for the Retrieval-Augmented Generation (RAG) project",
        # terms_of_service="https://www.example.com/terms/",
        contact=openapi.Contact(email="pakshay@stratapps.com"),
        # license=openapi.License(name="MIT License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    # Swagger UI and schema endpoints
    # path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('api-documentation/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    # path('swagger.json', schema_view.without_ui(cache_timeout=0), name='schema-json'),

    # Authentication
    path('register/', RegisterAPI.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/<str:token>/', LogoutView.as_view(), name='logout'),
    path('llm/setup/<str:token>/', LLMSetupAPIView.as_view(), name='llm-setup'),
    path('me/status/<str:token>/', UserStatusAPIView.as_view(), name='me-status'),
    path('protected/<str:token>/', ProtectedView.as_view(), name='protected'),

    # Tenant
    path('tenant/', TenantCreateAPIView.as_view(), name='tenant-create'),
    path('tenant/list/', TenantListAPIView.as_view(), name='tenant-list'),
    path('tenant/<int:id>/', TenantRetrieveUpdateDestroyAPIView.as_view(), name='tenant-detail'),

    # User
    path('user/<str:token>/list/', UserListAPIView.as_view(), name='user-list'),
    path('user/<str:token>/<int:id>/', UserRetrieveUpdateDestroyAPIView.as_view(), name='user-detail'),
    path('user/<str:token>/purge/', UserFullDeleteAPIView.as_view(), name='user-full-delete'),

    # Vector Store
    path('vector-store/<str:token>/', VectorStoreCreateAPIView.as_view(), name='vector-store-create'),
    path('vector-store/<str:token>/list/', VectorStoreListAPIView.as_view(), name='vector-store-list'),
    path('vector-store/<str:token>/<str:id>/', VectorStoreRetrieveUpdateDestroyAPIView.as_view(), name='vector-store-detail'),

    # Document
    path('document/<str:token>/ingest/', IngestAPIView.as_view(), name='document-ingest'),
    path('document/<str:token>/list/', DocumentListAPIView.as_view(), name='document-list'),
    path('document/<str:token>/<str:id>/', DocumentRetrieveUpdateDestroyAPIView.as_view(), name='document-detail'),
    path('document/<str:token>/<str:document_id>/status/', DocumentStatusAPIView.as_view(), name='document-status'),

    # Assistant
    path('assistant/<str:token>/', AssistantCreateAPIView.as_view(), name='assistant-create'),
    path('assistant/<str:token>/list/', AssistantListAPIView.as_view(), name='assistant-list'),
    path('assistant/<str:token>/<str:id>/', AssistantRetrieveUpdateDestroyAPIView.as_view(), name='assistant-detail'),

    # Thread
    path('thread/<str:token>/', ThreadCreateAPIView.as_view(), name='thread-create'),
    path('thread/<str:token>/list/', ThreadListAPIView.as_view(), name='thread-list'),
    path('thread/<str:token>/<str:id>/', ThreadRetrieveUpdateDestroyAPIView.as_view(), name='thread-detail'),
    path('thread/<str:token>/<str:thread_id>/messages/', ThreadMessagesAPIView.as_view(), name='thread-messages'),

    # Message
    path('message/<str:token>/', MessageCreateAPIView.as_view(), name='message-create'),
    path('message/<str:token>/list/', MessageListAPIView.as_view(), name='message-list'),
    path('message/<str:token>/<int:id>/', MessageRetrieveUpdateDestroyAPIView.as_view(), name='message-detail'),
    path('message-feedback/<str:token>/', MessageFeedbackCreateAPIView.as_view(), name='message-feedback-create'),
    path('message-feedback/<str:token>/list/', MessageFeedbackListAPIView.as_view(), name='message-feedback-list'),

    # Run
    path('run/<str:token>/', RunCreateAPIView.as_view(), name='run-create'),
    path('run/<str:token>/list/', RunListAPIView.as_view(), name='run-list'),
    path('run/<str:token>/<str:id>/', RunRetrieveUpdateDestroyAPIView.as_view(), name='run-detail'),
    path('run/<str:token>/<str:run_id>/cancel/', RunCancelAPIView.as_view(), name='run-cancel'),
    path('run/<str:token>/<str:run_id>/rerun/', RunRerunAPIView.as_view(), name='run-rerun'),
    path('run/<str:token>/<str:run_id>/submit-tool-outputs/', SubmitToolOutputsAPIView.as_view(), name='submit-tool-outputs'),

    # Document Access
    path('document-access/<str:token>/', DocumentAccessCreateAPIView.as_view(), name='document-access-create'),
    path('document-access/<str:token>/list/', DocumentAccessListAPIView.as_view(), name='document-access-list'),
    path('document-access/<str:token>/<int:id>/', DocumentAccessRetrieveUpdateDestroyAPIView.as_view(), name='document-access-detail'),

    # Document Access Remove
    path('document-access/remove/<str:token>/', DocumentAccessRemoveAPIView.as_view(), name='document-access-remove'),

    # Document Alert
    path('document-alert/<str:token>/', DocumentAlertCreateAPIView.as_view(), name='document-alert-create'),
    path('document-alert/<str:token>/list/', DocumentAlertListAPIView.as_view(), name='document-alert-list'),
    path('document-alert/<str:token>/<int:id>/', DocumentAlertRetrieveUpdateDestroyAPIView.as_view(), name='document-alert-detail'),

    # LLM Provider Configuration
    path('llm-config/<str:token>/', LLMProviderConfigCreateAPIView.as_view(), name='llm-config-create'),
    path('llm-config/<str:token>/list/', LLMProviderConfigListAPIView.as_view(), name='llm-config-list'),
    path('llm-config/<str:token>/<int:id>/', LLMProviderConfigRetrieveUpdateDestroyAPIView.as_view(), name='llm-config-detail'),

    # Backward-compatible OpenAIKey endpoints
    path('openai-key/<str:token>/', OpenAIKeyCreateAPIView.as_view(), name='openai-key-create'),
    path('openai-key/<str:token>/list/', OpenAIKeyListAPIView.as_view(), name='openai-key-list'),
    path('openai-key/<str:token>/<int:id>/', OpenAIKeyRetrieveUpdateDestroyAPIView.as_view(), name='openai-key-detail'),

    # Conversation Endpoints
    path('conversation/generate/<str:token>/', ConversationCreateAPIView.as_view(), name='conversation-generate'),
    path('conversation/<str:token>/<str:id>/', ConversationRetrieveUpdateDestroyAPIView.as_view(), name='conversation-detail'),
    path('conversation/<str:conversation_id>/items/<str:token>/', ConversationItemsAPIView.as_view(), name='conversation-items'),

    # Response Endpoints
    path('response/chat/<str:token>/', ResponsesAPIView.as_view(), name='response-chat'),
    path('response/<str:token>/<str:id>/', ResponseRetrieveDestroyAPIView.as_view(), name='response-detail'),
    path('response/<str:token>/<str:response_id>/cancel/', ResponseCancelAPIView.as_view(), name='response-cancel'),
]
