"""
Caching utilities for SBERT model and corpus embeddings.
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Global cache for SBERT model and corpus embeddings to avoid reloading/recomputing
_sbert_model_cache = None
_corpus_embeddings_cache = {}


def get_sbert_model():
    """Get cached SBERT model or load it if not cached."""
    global _sbert_model_cache
    if _sbert_model_cache is None:
        print("Loading SBERT model (cached for reuse)...")
        _sbert_model_cache = SentenceTransformer("LazarusNLP/all-indo-e5-small-v4")
    return _sbert_model_cache


def get_corpus_embeddings(corpus: List[tuple], corpus_id: str = None) -> Dict[str, Any]:
    """Get cached corpus embeddings or compute them if not cached."""
    global _corpus_embeddings_cache
    
    # Create a unique identifier for this corpus
    if corpus_id is None:
        # Use hash of corpus content as identifier
        corpus_content = str(sorted(corpus))
        corpus_id = str(hash(corpus_content))
    
    if corpus_id not in _corpus_embeddings_cache:
        print(f"Computing and caching embeddings for corpus (ID: {corpus_id[:8]}...) with {len(corpus)} examples...")
        sbert_model = get_sbert_model()
        source_texts = [pair[0] for pair in corpus]
        embeddings = sbert_model.encode(source_texts, show_progress_bar=True)
        _corpus_embeddings_cache[corpus_id] = {
            'embeddings': embeddings,
            'source_texts': source_texts,
            'corpus': corpus
        }
        print(f"✅ Corpus embeddings cached successfully!")
    else:
        print(f"Using cached embeddings for corpus (ID: {corpus_id[:8]}...)")
    
    return _corpus_embeddings_cache[corpus_id]


def clear_cache():
    """Clear all cached models and embeddings."""
    global _sbert_model_cache, _corpus_embeddings_cache
    _sbert_model_cache = None
    _corpus_embeddings_cache = {}
    print("Cache cleared.")
