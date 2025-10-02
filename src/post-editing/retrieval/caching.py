"""
Caching utilities for SBERT model and corpus embeddings.
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Global cache for SBERT models and corpus embeddings to avoid reloading/recomputing
_sbert_model_cache = {}
_corpus_embeddings_cache = {}


def get_sbert_model(model_name: str = "LazarusNLP/all-indo-e5-small-v4"):
    """Get cached SBERT model or load it if not cached."""
    global _sbert_model_cache
    if model_name not in _sbert_model_cache:
        print(f"Loading SBERT model '{model_name}' (cached for reuse)...")
        _sbert_model_cache[model_name] = SentenceTransformer(model_name)
    return _sbert_model_cache[model_name]


def get_corpus_embeddings(corpus: List[tuple], corpus_id: str = None, model_name: str = "LazarusNLP/all-indo-e5-small-v4") -> Dict[str, Any]:
    """Get cached corpus embeddings or compute them if not cached."""
    global _corpus_embeddings_cache
    
    # Create a unique identifier for this corpus (include model name in the ID)
    if corpus_id is None:
        # Use hash of corpus content and model name as identifier
        corpus_content = str(sorted(corpus)) + model_name
        corpus_id = str(hash(corpus_content))
    else:
        # Include model name in the corpus_id to separate caches for different models
        corpus_id = f"{corpus_id}_{model_name}"
    
    if corpus_id not in _corpus_embeddings_cache:
        print(f"Computing and caching embeddings for corpus (ID: {corpus_id[:8]}...) with {len(corpus)} examples using model '{model_name}'...")
        sbert_model = get_sbert_model(model_name)
        source_texts = [pair[0] for pair in corpus]
        embeddings = sbert_model.encode(source_texts, show_progress_bar=True)
        _corpus_embeddings_cache[corpus_id] = {
            'embeddings': embeddings,
            'source_texts': source_texts,
            'corpus': corpus,
            'model_name': model_name
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
