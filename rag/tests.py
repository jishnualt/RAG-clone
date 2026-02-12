from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from knox.models import AuthToken
from django.utils import timezone

from .models import (
    Tenant,
    User,
    Collection,
    VectorStore,
    Assistant,
    Thread,
    Message,
    Run,
    MessageFeedback,
    Conversation,
    ConversationMessage,
)
from .views import (
    DEFAULT_ASSISTANT_PROMPT_NO_CONTEXT,
)
from .llm_providers import NormalizedChatResponse


class RunFeatureTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="Acme")
        self.user = User.objects.create_user(username="alice", password="password", tenant=self.tenant)
        self.collection = Collection.objects.create(
            tenant=self.tenant, name="acme_collection", owner=self.user
        )
        self.user.active_collection = self.collection
        self.user.active_collection_ready = True
        self.user.llm_configured = True
        self.user.save(update_fields=["active_collection", "active_collection_ready", "llm_configured"])
        self.vector_store = VectorStore.objects.create(
            tenant=self.tenant, user=self.user, name="Primary Store", collection=self.collection
        )
        self.assistant = Assistant.objects.create(tenant=self.tenant, name="Helper", vector_store=self.vector_store)
        self.thread = Thread.objects.create(tenant=self.tenant, user=self.user, vector_store=self.vector_store)
        self.user_message = Message.objects.create(thread=self.thread, user=self.user, role='user', content="What is the status?")
        token_instance, _ = AuthToken.objects.create(self.user)
        self.token_key = token_instance.token_key
        self.client = APIClient()

    def _run_create_url(self):
        return reverse('run-create', kwargs={'token': self.token_key})

    def _run_cancel_url(self, run_id):
        return reverse('run-cancel', kwargs={'token': self.token_key, 'run_id': run_id})

    def _run_rerun_url(self, run_id):
        return reverse('run-rerun', kwargs={'token': self.token_key, 'run_id': run_id})

    def _message_feedback_url(self):
        return reverse('message-feedback-create', kwargs={'token': self.token_key})

    def _message_feedback_list_url(self):
        return reverse('message-feedback-list', kwargs={'token': self.token_key})

    @patch('rag.views.threading.Thread')
    def test_run_creation_with_message_id_uses_source_message(self, mock_thread):
        mock_thread.return_value.start = MagicMock()

        payload = {
            'thread_id': str(self.thread.id),
            'assistant_id': str(self.assistant.id),
            'message_id': self.user_message.id,
        }
        response = self.client.post(self._run_create_url(), payload, format='json')
        self.assertEqual(response.status_code, 201)

        run = Run.objects.latest('created_at')
        self.assertEqual(run.source_message, self.user_message)
        self.assertIsNone(run.rerun_of)

        thread_args = mock_thread.call_args.kwargs['args']
        self.assertEqual(thread_args[1], [self.user_message.content])

    @patch('rag.views.threading.Thread')
    def test_run_rerun_endpoint_creates_child_run(self, mock_thread):
        mock_thread.return_value.start = MagicMock()

        original_run = Run.objects.create(
            thread=self.thread,
            assistant=self.assistant,
            status=Run.STATUS_COMPLETED,
            source_message=self.user_message,
        )

        response = self.client.post(self._run_rerun_url(original_run.id), {}, format='json')
        self.assertEqual(response.status_code, 201)

        new_run = Run.objects.exclude(id=original_run.id).latest('created_at')
        self.assertEqual(new_run.rerun_of, original_run)
        self.assertEqual(new_run.source_message, self.user_message)

        thread_args = mock_thread.call_args.kwargs['args']
        self.assertEqual(thread_args[1], [self.user_message.content])

    def test_run_cancel_sets_status_and_timestamps(self):
        run = Run.objects.create(
            thread=self.thread,
            assistant=self.assistant,
            status=Run.STATUS_IN_PROGRESS,
            source_message=self.user_message,
            required_action={'type': 'submit_tool_outputs'},
        )

        response = self.client.post(self._run_cancel_url(run.id), {}, format='json')
        self.assertEqual(response.status_code, 200)

        run.refresh_from_db()
        self.assertEqual(run.status, Run.STATUS_CANCELLED)
        self.assertIsNotNone(run.cancelled_at)
        self.assertIsNotNone(run.completed_at)
        self.assertIsNone(run.required_action)

    def test_cancelling_completed_run_returns_bad_request(self):
        run = Run.objects.create(
            thread=self.thread,
            assistant=self.assistant,
            status=Run.STATUS_COMPLETED,
            completed_at=timezone.now(),
            source_message=self.user_message,
        )

        response = self.client.post(self._run_cancel_url(run.id), {}, format='json')
        self.assertEqual(response.status_code, 400)

    def test_feedback_submission_and_update(self):
        assistant_message = Message.objects.create(thread=self.thread, role='assistant', content="All good.")

        create_resp = self.client.post(
            self._message_feedback_url(),
            {'message_id': assistant_message.id, 'rating': MessageFeedback.RATING_GOOD},
            format='json'
        )
        self.assertEqual(create_resp.status_code, 201)
        feedback = MessageFeedback.objects.get(message=assistant_message, user=self.user)
        self.assertEqual(feedback.rating, MessageFeedback.RATING_GOOD)

        update_resp = self.client.post(
            self._message_feedback_url(),
            {'message_id': assistant_message.id, 'rating': MessageFeedback.RATING_BAD},
            format='json'
        )
        self.assertEqual(update_resp.status_code, 200)
        feedback.refresh_from_db()
        self.assertEqual(feedback.rating, MessageFeedback.RATING_BAD)

        list_resp = self.client.get(self._message_feedback_list_url(), {'thread_id': self.thread.id}, format='json')
        self.assertEqual(list_resp.status_code, 200)
        self.assertEqual(len(list_resp.data), 1)
        self.assertEqual(list_resp.data[0]['rating'], MessageFeedback.RATING_BAD)

    def test_feedback_rejects_non_assistant_message(self):
        response = self.client.post(
            self._message_feedback_url(),
            {'message_id': self.user_message.id, 'rating': MessageFeedback.RATING_GOOD},
            format='json'
        )
        self.assertEqual(response.status_code, 400)


class ConversationResponsesFlowTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="FlowCorp")
        self.user = User.objects.create_user(
            username="flow-user", password="password", tenant=self.tenant
        )
        self.collection = Collection.objects.create(
            tenant=self.tenant, name="flow_collection", owner=self.user
        )
        self.vector_store = VectorStore.objects.create(
            tenant=self.tenant, user=self.user, name="Flow VS", collection=self.collection
        )
        self.user.active_collection = self.collection
        self.user.active_collection_ready = True
        self.user.llm_configured = True
        self.user.save(update_fields=["active_collection", "active_collection_ready", "llm_configured"])

        token_instance, _ = AuthToken.objects.create(self.user)
        self.token_key = token_instance.token_key
        self.client = APIClient()

    def _conversation_create_url(self):
        return reverse('conversation-generate', kwargs={'token': self.token_key})

    def _response_chat_url(self):
        return reverse('response-chat', kwargs={'token': self.token_key})

    def _conversation_items_url(self, conversation_id):
        return reverse('conversation-items', kwargs={'conversation_id': conversation_id, 'token': self.token_key})

    @patch('rag.views.get_llm_config')
    def test_conversation_response_flow_persists_messages_and_metadata(self, mock_llm_config):
        provider = MagicMock()
        provider_response = NormalizedChatResponse(content="Hello there!", role="assistant")
        provider.get_chat_completion.return_value = provider_response
        mock_llm_config.return_value = {
            "provider_instance": provider,
            "chat_model": "gpt-4o-mini",
            "api_key": "test-key",
            "provider_name": "OpenAI",
        }

        create_response = self.client.post(self._conversation_create_url(), {}, format='json')
        self.assertEqual(create_response.status_code, 201)
        conversation_id = create_response.data['id']

        payload = {
            "model": "gpt-4o-mini",
            "conversation": str(conversation_id),
            "instructions": "Be concise.",
            "input": [
                {
                    "role": "user",
                    "content": "How are you?",
                    "metadata": {"turn": 1},
                }
            ],
            "metadata": {"request_id": "abc-123"},
        }

        response = self.client.post(self._response_chat_url(), payload, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['model'], "gpt-4o-mini")

        messages = ConversationMessage.objects.filter(conversation_id=conversation_id).order_by('created_at')
        self.assertEqual(messages.count(), 2)
        self.assertEqual(messages[0].role, 'user')
        self.assertEqual(messages[0].content, "How are you?")
        self.assertEqual(messages[0].metadata, {"turn": 1})
        self.assertEqual(messages[1].role, 'assistant')
        self.assertEqual(messages[1].content, "Hello there!")
        self.assertEqual(messages[1].metadata, {"request_id": "abc-123"})

        output_message = response.data['output'][0]
        self.assertEqual(output_message['role'], 'assistant')
        self.assertEqual(output_message['content'][0]['text'], "Hello there!")
        self.assertEqual(output_message['message_id'], str(messages[1].id))
        self.assertEqual(output_message['metadata'], {"request_id": "abc-123"})

        provider.get_chat_completion.assert_called_once()
        called_messages = provider.get_chat_completion.call_args.kwargs['messages']
        self.assertEqual(called_messages[-1]['content'], "How are you?")

        items_response = self.client.get(self._conversation_items_url(conversation_id), format='json')
        self.assertEqual(items_response.status_code, 200)
        self.assertEqual(len(items_response.data), 2)
        self.assertEqual(items_response.data[0]['content'], "How are you?")
        self.assertEqual(items_response.data[1]['content'], "Hello there!")

    @patch('rag.views.get_qdrant_vector_store')
    @patch('rag.views.get_accessible_document_ids')
    @patch('rag.views.hybrid_search_all_files')
    @patch('rag.views.get_llm_config')
    def test_response_flow_adds_context_with_document_tool(self, mock_llm_config, mock_hybrid_search, mock_accessible_docs, mock_get_vector_store):
        provider = MagicMock()
        provider.get_chat_completion.return_value = NormalizedChatResponse(content="Contextual answer", role="assistant")
        mock_llm_config.return_value = {
            "provider_instance": provider,
            "chat_model": "gpt-4o-mini",
            "api_key": "test-key",
            "provider_name": "OpenAI",
        }

        mock_accessible_docs.return_value = [123]
        mock_hybrid_search.return_value = [SimpleNamespace(page_content="Relevant doc chunk.")]
        mock_get_vector_store.return_value = SimpleNamespace()

        conversation_resp = self.client.post(self._conversation_create_url(), {}, format='json')
        conversation_id = conversation_resp.data['id']
        conversation = Conversation.objects.get(id=conversation_id)

        payload = {
            "model": "gpt-4o-mini",
            "conversation": str(conversation_id),
            "input": [
                {
                    "role": "user",
                    "content": "What is in the document?",
                }
            ],
            "tools": [
                {
                    "type": "document",
                    "vector_store_ids": [str(self.vector_store.id)],
                }
            ],
        }

        response = self.client.post(self._response_chat_url(), payload, format='json')

        self.assertEqual(response.status_code, 200)
        conversation.refresh_from_db()
        self.assertEqual(conversation.title, "What is in the document?")
        mock_hybrid_search.assert_called_once()
        mock_accessible_docs.assert_called_once_with(self.user, str(self.vector_store.id))

        called_messages = provider.get_chat_completion.call_args.kwargs['messages']
        self.assertTrue(called_messages[-1]['content'].startswith("Context:"))
        self.assertIn("Question: What is in the document?", called_messages[-1]['content'])

    @patch('rag.views.get_qdrant_vector_store')
    @patch('rag.views.get_accessible_document_ids')
    @patch('rag.views.hybrid_search_all_files')
    @patch('rag.views.get_llm_config')
    def test_response_flow_does_not_filter_by_vector_store_id(self, mock_llm_config, mock_hybrid_search, mock_accessible_docs, mock_get_vector_store):
        provider = MagicMock()
        provider.get_chat_completion.return_value = NormalizedChatResponse(content="Contextual answer", role="assistant")
        mock_llm_config.return_value = {
            "provider_instance": provider,
            "chat_model": "gpt-4o-mini",
            "api_key": "test-key",
            "provider_name": "OpenAI",
        }

        mock_accessible_docs.return_value = [123]
        mock_hybrid_search.return_value = [SimpleNamespace(page_content="Relevant doc chunk.")]
        mock_get_vector_store.return_value = SimpleNamespace()

        conversation_resp = self.client.post(self._conversation_create_url(), {}, format='json')
        conversation_id = conversation_resp.data['id']

        payload = {
            "model": "gpt-4o-mini",
            "conversation": str(conversation_id),
            "input": [
                {
                    "role": "user",
                    "content": "What is in the document?",
                }
            ],
            "tools": [
                {
                    "type": "document",
                    "vector_store_ids": [str(self.vector_store.id)],
                }
            ],
        }

        response = self.client.post(self._response_chat_url(), payload, format='json')
        self.assertEqual(response.status_code, 200)
        # Ensure hybrid search was called and the search_filter does not include vector_store_id condition
        self.assertTrue(mock_hybrid_search.called)
        _, kwargs = mock_hybrid_search.call_args
        search_filter = kwargs.get('search_filter')
        # Convert to string to inspect; ensure no vector_store_id key appears in the textual representation
        self.assertIsNotNone(search_filter)
        self.assertNotIn('vector_store_id', str(search_filter))


        messages = ConversationMessage.objects.filter(conversation_id=conversation_id).order_by('created_at')
        self.assertEqual(messages.count(), 2)
        self.assertEqual(messages[0].role, 'user')
        self.assertEqual(messages[1].role, 'assistant')

    @patch('rag.views.get_llm_config')
    def test_response_flow_supports_ollama_provider(self, mock_llm_config):
        provider = MagicMock()
        provider.get_chat_completion.return_value = NormalizedChatResponse(content="Hello from Ollama", role="assistant")
        mock_llm_config.return_value = {
            "provider_instance": provider,
            "chat_model": "llama3",
            "api_key": None,
            "base_url": "http://localhost:11434",
            "provider_name": "Ollama",
        }

        conversation_resp = self.client.post(self._conversation_create_url(), {}, format='json')
        conversation_id = conversation_resp.data['id']

        payload = {
            "model": "llama3",
            "conversation": str(conversation_id),
            "instructions": "Stay brief",
            "input": [
                {
                    "role": "user",
                    "content": "Ping",
                }
            ],
        }

        response = self.client.post(self._response_chat_url(), payload, format='json')

        self.assertEqual(response.status_code, 200)
        provider.get_chat_completion.assert_called_once()
        call_kwargs = provider.get_chat_completion.call_args.kwargs
        self.assertEqual(call_kwargs['model'], "llama3")
        self.assertIsNone(call_kwargs['api_key'])

        messages = call_kwargs['messages']
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[0]['content'], "Stay brief")
        self.assertEqual(messages[-1]['content'], "Ping")

        stored_messages = ConversationMessage.objects.filter(conversation_id=conversation_id).order_by('created_at')
        self.assertEqual(stored_messages.count(), 2)
        self.assertEqual(stored_messages[1].content, "Hello from Ollama")

    @patch('rag.views.get_llm_config')
    def test_response_flow_uses_default_system_prompt_without_instructions(self, mock_llm_config):
        provider = MagicMock()
        provider.get_chat_completion.return_value = NormalizedChatResponse(content="Hi", role="assistant")
        mock_llm_config.return_value = {
            "provider_instance": provider,
            "chat_model": "gpt-4o-mini",
            "api_key": "key",
            "provider_name": "OpenAI",
        }

        payload = {
            "model": "gpt-4o-mini",
            "input": [
                {"role": "user", "content": "Hello"}
            ],
        }

        response = self.client.post(self._response_chat_url(), payload, format='json')

        self.assertEqual(response.status_code, 200)
        provider.get_chat_completion.assert_called_once()
        called_messages = provider.get_chat_completion.call_args.kwargs['messages']
        self.assertEqual(called_messages[0]['role'], 'system')
        self.assertEqual(called_messages[0]['content'], DEFAULT_ASSISTANT_PROMPT_NO_CONTEXT)
        self.assertEqual(called_messages[-1]['content'], 'Hello')

    @patch('rag.views.OllamaProvider')
    @patch('rag.views.get_llm_config')
    def test_response_flow_reconfigures_provider_for_requested_model(self, mock_llm_config, mock_ollama_provider):
        openai_provider = MagicMock()
        mock_llm_config.return_value = {
            "provider_instance": openai_provider,
            "chat_model": "gpt-4o-mini",
            "api_key": "key",
            "provider_name": "OpenAI",
            "base_url": "http://localhost:11434",
        }
        ollama_provider = MagicMock()
        ollama_provider.get_chat_completion.return_value = NormalizedChatResponse(content="Ollama response", role="assistant")
        mock_ollama_provider.return_value = ollama_provider

        payload = {
            "model": "llama3",
            "input": [
                {"role": "user", "content": "Switch providers"}
            ],
        }

        response = self.client.post(self._response_chat_url(), payload, format='json')

        self.assertEqual(response.status_code, 200)
        mock_ollama_provider.assert_called_once_with(base_url="http://localhost:11434")
        ollama_provider.get_chat_completion.assert_called_once()
        self.assertIsNone(ollama_provider.get_chat_completion.call_args.kwargs['api_key'])
        self.assertEqual(ollama_provider.get_chat_completion.call_args.kwargs['model'], 'llama3')

    def test_conversation_create_accepts_title(self):
        payload = {"title": "My Custom Conversation"}
        create_response = self.client.post(self._conversation_create_url(), payload, format='json')
        self.assertEqual(create_response.status_code, 201)

        conversation_id = create_response.data['id']
        conversation = Conversation.objects.get(id=conversation_id)

        self.assertEqual(conversation.title, "My Custom Conversation")
        self.assertEqual(create_response.data['title'], "My Custom Conversation")


class ThreadMessageApiTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="ThreadCo")
        self.user = User.objects.create_user(username="threader", password="password", tenant=self.tenant)
        self.collection = Collection.objects.create(tenant=self.tenant, name="thread_collection", owner=self.user)
        self.user.active_collection = self.collection
        self.user.active_collection_ready = True
        self.user.llm_configured = True
        self.user.save(update_fields=["active_collection", "active_collection_ready", "llm_configured"])
        self.vector_store = VectorStore.objects.create(
            tenant=self.tenant, user=self.user, name="Thread Store", collection=self.collection
        )
        token_instance, _ = AuthToken.objects.create(self.user)
        self.token_key = token_instance.token_key
        self.client = APIClient()

    def _thread_create_url(self):
        return reverse('thread-create', kwargs={'token': self.token_key})

    def _message_create_url(self):
        return reverse('message-create', kwargs={'token': self.token_key})

    def _thread_messages_url(self, thread_id):
        return reverse('thread-messages', kwargs={'token': self.token_key, 'thread_id': thread_id})

    def _message_list_url(self):
        return reverse('message-list', kwargs={'token': self.token_key})

    def test_thread_and_message_endpoints_store_and_filter_messages(self):
        thread_payload = {
            "title": "Support thread",
            "vector_store_id": str(self.vector_store.id),
            "metadata": {"topic": "support"},
        }
        thread_response = self.client.post(self._thread_create_url(), thread_payload, format='json')
        self.assertEqual(thread_response.status_code, 201)
        thread_id = thread_response.data['id']

        message_payload = {
            "thread_id": thread_id,
            "content": "I need help with my account.",
            "metadata": {"category": "account"},
        }
        message_response = self.client.post(self._message_create_url(), message_payload, format='json')
        self.assertEqual(message_response.status_code, 201)

        message_obj = Message.objects.get(thread_id=thread_id)
        self.assertEqual(message_obj.role, 'user')
        self.assertEqual(message_obj.content, "I need help with my account.")
        self.assertEqual(message_obj.metadata, {"category": "account"})

        thread_messages = self.client.get(self._thread_messages_url(thread_id), format='json')
        self.assertEqual(thread_messages.status_code, 200)
        self.assertEqual(len(thread_messages.data), 1)
        self.assertEqual(thread_messages.data[0]['content'], "I need help with my account.")

        filtered_messages = self.client.get(self._message_list_url(), {"thread_id": thread_id}, format='json')
        self.assertEqual(filtered_messages.status_code, 200)
        self.assertEqual(len(filtered_messages.data), 1)
        self.assertEqual(filtered_messages.data[0]['content'], "I need help with my account.")
