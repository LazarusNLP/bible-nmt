"""
Base classes for retrieval systems.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseRetriever(ABC):
    """Abstract base class for similarity-based retrievers."""
    
    @abstractmethod
    def get_similar_examples(self, query: str, corpus: List[Tuple], k: int) -> List[Tuple]:
        """
        Get top-k similar examples from corpus.
        
        Args:
            query: Query text to find similar examples for
            corpus: List of (source, target) text pairs
            k: Number of examples to retrieve
            
        Returns:
            List of top-k most similar (source, target) pairs
        """
        pass
