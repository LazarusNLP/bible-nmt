"""
Few-shot example selection utilities.
"""

from typing import List, Tuple, Dict, Any
from core.constants import SUPPORTED_VECTORIZERS
from .vectorizers import BM25Retriever, TFIDFRetriever, SBERTRetriever, CHRFRAGRetriever, WordBasedParallelRetriever


class FewShotSelector:
    """Unified interface for few-shot example selection."""
    
    def __init__(self, vectorizer_type: str = "bm25", corpus_id: str = None, top_n_per_word: int = 3):
        """
        Initialize few-shot selector.
        
        Args:
            vectorizer_type: Type of vectorizer ("bm25", "tfidf", "sbert", "chrf_rag", "word_parallel", "full")
            corpus_id: Optional corpus ID for SBERT caching
            top_n_per_word: For word_parallel mode, number of top matches per word (hyperparameter)
        """
        # Add "full" to supported vectorizers for this class
        supported_types = SUPPORTED_VECTORIZERS + ["full"]
        if vectorizer_type not in supported_types:
            raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}. "
                           f"Supported types: {supported_types}")
        
        self.vectorizer_type = vectorizer_type
        self.corpus_id = corpus_id
        self.top_n_per_word = top_n_per_word
        
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
        elif vectorizer_type == "word_parallel":
            self.retriever = WordBasedParallelRetriever(top_n_per_word=top_n_per_word)
        elif vectorizer_type == "all_mpnet":
            # Use SBERT retriever with all-mpnet-base-v2 model
            self.retriever = SBERTRetriever(
                corpus_id=corpus_id, 
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
        elif vectorizer_type == "bge":
            # Use SBERT retriever with BGE-large-en-v1.5 model (SOTA for English retrieval)
            self.retriever = SBERTRetriever(
                corpus_id=corpus_id, 
                model_name="BAAI/bge-large-en-v1.5"
            )
    
    def get_examples(self, query: str, corpus: List[Tuple], k: int) -> List[Tuple]:
        """
        Get top-k similar examples from corpus.
        
        Args:
            query: Query text to find similar examples for
            corpus: List of (source, target) text pairs
            k: Number of examples to retrieve (use -1 for word_parallel mode to get all word-matched examples)
            
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
        Get few-shot examples for multiple rows efficiently using batch processing when available.
        
        Args:
            rows: List of row objects with src_text attribute
            corpus: List of (source, target) text pairs
            k: Number of examples per row (use -1 for word_parallel mode to get all word-matched examples)
            
        Returns:
            List of example lists, one per row
        """
        if self.vectorizer_type == "full":
            print(f"Using FULL mode: providing all {len(corpus)} corpus examples for each of {len(rows)} rows...")
            # Return the same full corpus for every row
            return [corpus for _ in rows]
        
        # Check if retriever supports optimized batch processing
        if hasattr(self.retriever, 'get_similar_examples_batch'):
            print(f"🚀 Using OPTIMIZED BATCH processing for {len(rows)} rows with {self.vectorizer_type} vectorizer...")
            queries = [row.src_text for row in rows]
            return self.retriever.get_similar_examples_batch(queries, corpus, k)
        else:
            # Fallback to individual processing with progress tracking
            if self.vectorizer_type == "word_parallel" and k == -1:
                print(f"Using WORD_PARALLEL mode (individual processing): providing all word-matched examples for each of {len(rows)} rows...")
            else:
                print(f"Selecting few-shot examples for {len(rows)} rows using {self.vectorizer_type} (individual processing)...")
            
            examples_list = []
            from tqdm import tqdm
            
            for row in tqdm(rows, desc=f"Vectorizing with {self.vectorizer_type}"):
                examples = self.get_examples(row.src_text, corpus, k)
                examples_list.append(examples)
            
            return examples_list
