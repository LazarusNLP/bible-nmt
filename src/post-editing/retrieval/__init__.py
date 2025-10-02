"""
Retrieval and similarity modules for few-shot example selection.
"""

from .vectorizers import BM25Retriever, TFIDFRetriever, SBERTRetriever, CHRFRAGRetriever, WordBasedParallelRetriever
from .few_shot import FewShotSelector
from .glossary import GlossarySelector
from .caching import get_sbert_model, get_corpus_embeddings

__all__ = [
    'BM25Retriever',
    'TFIDFRetriever', 
    'SBERTRetriever',
    'CHRFRAGRetriever',
    'WordBasedParallelRetriever',
    'FewShotSelector',
    'GlossarySelector',
    'get_sbert_model',
    'get_corpus_embeddings'
]
