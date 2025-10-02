"""
Base classes for LLM model interfaces.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseLLM(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def generate(self, messages: List[dict], translation_mode: bool = False) -> str:
        """
        Generate a single response from messages.
        
        Args:
            messages: List of chat messages with 'role' and 'content' keys
            translation_mode: If True, use translation mode instead of post-editing
            
        Returns:
            Generated text response
        """
        pass
    
    def generate_batch(self, messages_list: List[List[dict]]) -> List[str]:
        """
        Generate responses for a batch of message lists.
        
        Default implementation uses individual calls.
        Override for more efficient batch processing.
        
        Args:
            messages_list: List of message lists
            
        Returns:
            List of generated responses
        """
        return [self.generate(messages, False) for messages in messages_list]
    
    @property
    def supports_batch(self) -> bool:
        """Whether this model supports efficient batch processing."""
        return False
    
    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return getattr(self, '_model_name', 'unknown')
