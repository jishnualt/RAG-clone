"""
Hybrid search module for RAG: combines BM25 lexical and vector similarity search.
Production-ready, scalable, robust, and explainable.
"""
import logging
import os
from typing import List, Optional, Tuple, Any
from rank_bm25 import BM25Okapi
import numpy as np
import hashlib
import re
import spacy
from sentence_transformers import CrossEncoder
from functools import lru_cache
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from .trace import trace_span

logger = logging.getLogger(__name__)

class HybridSearchRetriever(BaseRetriever):
    """
    Standard LangChain retriever that uses our custom hybrid search logic.
    """
    vector_store: Any
    search_filter: Optional[Any] = None
    k: int = 15
    alpha: Optional[float] = None
    bm25_corpus_limit: Optional[int] = None
    rerank_model: Optional[str] = None
    use_mmr: bool = False

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Implementation of the standard LangChain retriever interface.
        """
        return hybrid_search_all_files(
            question=query,
            vector_store=self.vector_store,
            search_filter=self.search_filter,
            k=self.k,
            alpha=self.alpha,
            bm25_corpus_limit=self.bm25_corpus_limit,
            rerank_model=self.rerank_model,
            use_mmr=self.use_mmr
        )

# Optional sparse encoder (e.g., SPLADE via fastembed)
try:
    from fastembed import SparseTextEmbedding

    _sparse_encoder = SparseTextEmbedding(
        model_name=os.getenv("SPARSE_EMBED_MODEL", "splade_en_v1")
    )
    HAS_SPARSE_ENCODER = True
except Exception:
    HAS_SPARSE_ENCODER = False
    _sparse_encoder = None


@lru_cache(maxsize=2)
def _get_cross_encoder(model_name: str) -> CrossEncoder:
    """Cache CrossEncoder instances by model name."""
    return CrossEncoder(model_name)

# --- spaCy Tokenizer Setup ---
# Try to load a full English model for best tokenization; fallback to blank with warning.
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    logger.info("spaCy 'en_core_web_sm' loaded for tokenization.")
except Exception:
    nlp = spacy.blank("en")
    logger.warning("spaCy 'en_core_web_sm' not found. Using blank model for tokenization. Tokenization quality may be reduced.")

# Regex to capture full e-mail addresses as single tokens
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def spacy_tokenize(text: str) -> List[str]:
    """Tokenize text while preserving email addresses as single tokens."""
    emails = EMAIL_REGEX.findall(text)
    # Remove emails from text to avoid fragmented spaCy tokens
    text_no_emails = EMAIL_REGEX.sub(" ", text)

    if nlp:
        tokens = [t.text.lower() for t in nlp(text_no_emails) if t.text.strip() and not t.is_space]
    else:
        logger.warning("spaCy model unavailable. Falling back to regex tokenization.")
        tokens = re.findall(r"\b\w+\b", text_no_emails.lower())

    tokens.extend([e.lower() for e in emails])
    return tokens


def get_sparse_scores(query: str, docs: List[Any]) -> Optional[np.ndarray]:
    """Compute sparse vector scores using optional fastembed model."""
    if not HAS_SPARSE_ENCODER:
        return None
    try:
        q_vec = next(_sparse_encoder.embed([query], show_progress=False))
        doc_vecs = list(
            _sparse_encoder.embed([d.page_content for d in docs], show_progress=False)
        )
        q_map = dict(zip(q_vec.indices, q_vec.values))
        scores = []
        for vec in doc_vecs:
            score = sum(q_map.get(i, 0.0) * v for i, v in zip(vec.indices, vec.values))
            scores.append(score)
        return np.array(scores)
    except Exception as e:
        logger.warning(f"Sparse scoring failed: {e}")
        return None


def hybrid_search_all_files(
    question: str,
    vector_store: Any,
    search_filter: Any,
    k: int = 15,
    alpha: Optional[float] = None,
    bm25_corpus_limit: Optional[int] = None,
    explain: bool = False,
    rerank_model: Optional[str] = None,
    use_mmr: bool = False,
) -> List[Any]:
    """
    Hybrid search using BM25 and vector similarity. Returns top-k results.
    Args:
        question: The user query.
        vector_store: The vector store instance.
        search_filter: The filter to apply.
        k: Number of results to return.
        alpha: Weight for vector score (0-1). If None, uses env or default.
        bm25_corpus_limit: Max number of chunks to use for BM25. If None, uses env or default.
        explain: If True, returns (doc, score, method) tuples for explainability.
    Returns:
        List of top-k document chunks (optionally with scores/methods).
    """
    # Configurable parameters
    alpha = float(os.getenv("HYBRID_SEARCH_ALPHA", alpha if alpha is not None else 0.3))
    bm25_corpus_limit = int(os.getenv("BM25_CORPUS_LIMIT", bm25_corpus_limit if bm25_corpus_limit is not None else 5000))

    try:
        with trace_span("hybrid_search", question=question):
            # 1. Get filtered chunks (limit corpus size for BM25)
            try:
                filter_results = vector_store.similarity_search(query=question, k=bm25_corpus_limit, filter=search_filter)
                filtered_chunks = [doc for doc in filter_results]
            except Exception as e:
                logger.warning(f"Filter retrieval failed: {e}. Using all.")
                filter_results = vector_store.similarity_search(query=question, k=bm25_corpus_limit)
                filtered_chunks = [doc for doc in filter_results]

            if not filtered_chunks:
                logger.warning("No chunks found for hybrid search corpus.")
                return []

            all_chunks = [str(doc.page_content) for doc in filtered_chunks]
            tokenized_corpus = [spacy_tokenize(chunk) for chunk in all_chunks]

            # 2. BM25 Search
            tokenized_query = spacy_tokenize(question)
            if not tokenized_query:
                logger.warning("Query tokenization yielded no tokens. Returning vector-only results.")
                return [doc for doc, _ in vector_store.similarity_search_with_score(question, k=k, filter=search_filter)]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = np.array(bm25.get_scores(tokenized_query))
            lexical_scores = bm25_scores.copy()

            # Optional: Boost if query looks like a name
            if len(tokenized_query) == 2:
                for i, tokens in enumerate(tokenized_corpus):
                    if tokenized_query[0] in tokens and tokenized_query[1] in tokens:
                        lexical_scores[i] *= 1.5

            sparse_scores = get_sparse_scores(question, filtered_chunks)
            if sparse_scores is not None:
                lexical_scores += sparse_scores

            # 3. Vector Search (over the same filter)
            vector_results = vector_store.similarity_search_with_score(question, k=k, filter=search_filter)

            # 4. Build metadata with hash
            chunk_metadata = []
            for doc in filtered_chunks:
                content = str(doc.page_content)
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                chunk_metadata.append({
                    'chunk_obj': doc,
                    'content_hash': content_hash
                })

            # 5. Combine
            combined_docs = combine_bm25_vector_results(
                lexical_scores=lexical_scores,
                chunk_metadata=chunk_metadata,
                vector_results=vector_results,
                alpha=alpha,
                explain=explain,
                top_k=k
            )

            docs_only = [doc for doc, *_ in combined_docs] if explain else combined_docs

            if use_mmr:
                try:
                    docs_only = apply_mmr(vector_store, question, docs_only, k)
                except Exception as mmr_e:
                    logger.warning(f"MMR step failed: {mmr_e}")

            model_name = rerank_model or os.getenv("RERANKER_MODEL")
            if model_name:
                try:
                    docs_only = rerank_documents(question, docs_only, model_name, k)
                except Exception as rerank_e:
                    logger.warning(f"Rerank step failed: {rerank_e}")

            logger.info(f"Hybrid search returned {len(docs_only)} results (requested top {k})")
            if len(docs_only) < k:
                logger.warning(f"Hybrid search returned fewer results ({len(docs_only)}) than requested ({k}).")

            return docs_only if (use_mmr or model_name) and explain else (docs_only if not explain else combined_docs)

    except Exception as e:
        logger.error(f"Hybrid search failed: {e} | Query: {question} | Filter: {search_filter}", exc_info=True)
        # Always return a list of docs only
        try:
            fallback = vector_store.similarity_search_with_score(question, k=k, filter=search_filter)
            return [doc for doc, _ in fallback]
        except Exception as fallback_e:
            logger.critical(f"Vector fallback also failed: {fallback_e}")
            return []


def combine_bm25_vector_results(
    lexical_scores: np.ndarray,
    chunk_metadata: List[dict],
    vector_results: List[Tuple[Any, float]],
    alpha: float,
    explain: bool = False,
    top_k: int = 15
) -> List[Any]:
    """
    Combine lexical (BM25 and optional sparse) and vector search results with weighted scoring using content hash.
    Deduplicate results and optionally return explainability info.
    Returns top_k results only.
    """
    try:
        # --- Handle lexical scores (BM25 +/- sparse) ---
        if isinstance(lexical_scores, np.ndarray) and len(lexical_scores) > 0:
            max_bm25 = float(np.max(lexical_scores)) or 1.0
            normalized_bm25 = (lexical_scores / max_bm25).tolist()
        elif isinstance(lexical_scores, list) and len(lexical_scores) > 0:
            max_bm25 = max(lexical_scores) or 1.0
            normalized_bm25 = [score / max_bm25 for score in lexical_scores]
        else:
            normalized_bm25 = [0.0] * len(chunk_metadata)

        # --- Build vector score map using content hash ---
        vector_score_map = {}
        for doc, score in vector_results:
            try:
                # If using distance, convert to similarity. If using cosine, adjust here.
                if isinstance(score, np.ndarray):
                    if score.size == 0:
                        score_value = 0.0
                    else:
                        score_value = float(np.ravel(score)[0])
                elif hasattr(score, '__len__') and not isinstance(score, str):
                    score_value = float(score[0]) if len(score) > 0 else 0.0
                else:
                    score_value = float(score)
                # If you use cosine similarity (higher is better), use: similarity = score_value
                similarity = 1 / (1 + score_value) if score_value >= 0 else 0.0
                content = str(doc.page_content)
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                vector_score_map[content_hash] = similarity
            except Exception as e:
                logger.warning(f"Failed to process vector score {repr(score)}: {e}. Using default similarity.")
                content = str(doc.page_content)
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                vector_score_map[content_hash] = 0.5

        # --- Combine scores ---
        combined_results = []
        for i, (bm25_score, metadata) in enumerate(zip(normalized_bm25, chunk_metadata)):
            chunk_obj = metadata['chunk_obj']
            try:
                content = str(chunk_obj.page_content)
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            except Exception:
                content_hash = "unknown"
            vector_score = vector_score_map.get(content_hash, 0.0)
            combined_score = alpha * vector_score + (1 - alpha) * bm25_score
            method = (
                "hybrid"
                if bm25_score > 0 and vector_score > 0 else
                "bm25" if bm25_score > 0 else
                "vector" if vector_score > 0 else
                "none"
            )
            combined_results.append((chunk_obj, combined_score, method) if explain else (chunk_obj, combined_score))

        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate by content hash (keep highest score), stop at top_k
        seen_hashes = set()
        deduped = []
        for tup in combined_results:
            doc = tup[0]
            content = str(doc.page_content)
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            if content_hash not in seen_hashes:
                deduped.append(tup if explain else doc)
                seen_hashes.add(content_hash)
                if len(deduped) >= top_k:
                    break

        return deduped

    except Exception as e:
        logger.error(f"Unexpected error in combine_bm25_vector_results: {e}", exc_info=True)
        logger.info("Falling back to vector-only results.")
        if explain:
            return [(doc, score, "vector") for doc, score in vector_results][:top_k]
        return [doc for doc, score in vector_results][:top_k]


def rerank_documents(question: str, docs: List[Any], model_name: str, top_k: int) -> List[Any]:
    """Rerank documents using a cross-encoder model."""
    if not docs:
        return []
    cross_encoder = _get_cross_encoder(model_name)
    pairs = [[question, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


def apply_mmr(vector_store: Any, question: str, docs: List[Any], k: int, lambda_mult: float = 0.5) -> List[Any]:
    """Apply Max Marginal Relevance to diversify documents."""
    if not docs:
        return []
    embed_fn = vector_store.embedding_function
    query_emb = embed_fn.embed_query(question)
    doc_embs = embed_fn.embed_documents([doc.page_content for doc in docs])
    selected = maximal_marginal_relevance(query_emb, doc_embs, k=k, lambda_mult=lambda_mult)
    return [docs[i] for i in selected]
