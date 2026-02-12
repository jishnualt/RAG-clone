from django.test import TestCase
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from rag.search import rerank_documents, _get_cross_encoder
from rag import utils
from rag.llm_providers import NormalizedChatResponse


class RerankerCacheTests(TestCase):
    def test_cross_encoder_cached(self):
        _get_cross_encoder.cache_clear()
        docs = [SimpleNamespace(page_content="a"), SimpleNamespace(page_content="b")]
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.9, 0.1]
        with patch('rag.search.CrossEncoder', return_value=fake_model) as mock_ce:
            rerank_documents("q", docs, "test-model", top_k=2)
            rerank_documents("q", docs, "test-model", top_k=1)
            self.assertEqual(mock_ce.call_count, 1)


class AskQuestionFallbackTests(TestCase):
    def test_missing_prompt_added_when_no_context(self):
        user = SimpleNamespace(id=1, tenant=SimpleNamespace(id=1))
        fake_response = NormalizedChatResponse(content="none", role="assistant")
        fake_provider = MagicMock()
        fake_provider.get_chat_completion.return_value = fake_response
        llm_conf = {
            "provider_instance": fake_provider,
            "provider_name": "openai",
            "chat_model": "gpt",
            "api_key": "key",
        }
        with patch.object(utils, 'get_llm_config', return_value=llm_conf), \
             patch.object(utils, 'get_qdrant_vector_store'):
            utils.ask_question(
                question="q",
                vector_store_id="vs",
                user=user,
                collection_name="col",
                documents=[],
            )
        args, kwargs = fake_provider.get_chat_completion.call_args
        messages = kwargs["messages"]
        self.assertIn("provided documents do not contain", messages[1]["content"].lower())
