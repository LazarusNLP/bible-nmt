"""
Few-shot example selection utilities.
"""

from typing import List, Tuple, Dict, Any
from core.constants import SUPPORTED_VECTORIZERS
from .vectorizers import BM25Retriever, TFIDFRetriever, SBERTRetriever, CHRFRAGRetriever, WordBasedLCSRetriever


class FewShotSelector:
    """Unified interface for few-shot example selection."""
    
    def __init__(self, vectorizer_type: str = "bm25", corpus_id: str = None):
        """
        Initialize few-shot selector.
        
        Args:
            vectorizer_type: Type of vectorizer ("bm25", "tfidf", "sbert", "chrf_rag", "word_lcs", "full")
            corpus_id: Optional corpus ID for SBERT caching
        """
        # Add "full" to supported vectorizers for this class
        supported_types = SUPPORTED_VECTORIZERS + ["full"]
        if vectorizer_type not in supported_types:
            raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}. "
                           f"Supported types: {supported_types}")
        
        self.vectorizer_type = vectorizer_type
        self.corpus_id = corpus_id
        
        # Initialize retriever based on type (skip for "full" mode)
        if vectorizer_type == "full":
            self.retriever = None  # No retriever needed for full mode
        elif vectorizer_type == "bm25":
            self.retriever = BM25Retriever()
        elif vectorizer_type == "tfidf":
            self.retriever = TFIDFRetriever()
        elif vectorizer_type == "sbert":
            self.retriever = SBERTRetriever(corpus_id)
        elif vectorizer_type == "chrf_rag":
            self.retriever = CHRFRAGRetriever()
        elif vectorizer_type == "word_lcs":
            self.retriever = WordBasedLCSRetriever()
    
    def get_examples(self, query: str, corpus: List[Tuple], k: int) -> List[Tuple]:
        """
        Get top-k similar examples from corpus.
        
        Args:
            query: Query text to find similar examples for
            corpus: List of (source, target) text pairs
            k: Number of examples to retrieve
            
        Returns:
            List of top-k most similar (source, target) pairs (or all if vectorizer_type is "full")
        """
        if self.vectorizer_type == "full":
            # Return all examples in corpus (ignore k parameter)
            return corpus
        else:
            return self.retriever.get_similar_examples(query, corpus, k)
    
    def get_examples_for_rows(self, rows: List[Any], corpus: List[Tuple], k: int) -> List[List[Tuple]]:
        """
        Get few-shot examples for multiple rows efficiently.
        
        Args:
            rows: List of row objects with src_text attribute
            corpus: List of (source, target) text pairs
            k: Number of examples per row
            
        Returns:
            List of example lists, one per row
        """
        if self.vectorizer_type == "full":
            print(f"Using FULL mode: providing all {len(corpus)} corpus examples for each of {len(rows)} rows...")
            # Return the same full corpus for every row
            return [corpus for _ in rows]
        else:
            print(f"Selecting few-shot examples for {len(rows)} rows using {self.vectorizer_type}...")
            
            examples_list = []
            for row in rows:
                examples = self.get_examples(row.src_text, corpus, k)
                examples_list.append(examples)
            
            return examples_list
