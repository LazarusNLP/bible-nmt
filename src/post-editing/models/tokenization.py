"""
Token counting utilities for different models.
"""

from typing import List, Optional


class TokenCounter:
    """Handles token counting for various models."""
    
    def __init__(self, tokenizer=None):
        """Initialize with optional tokenizer."""
        self.tokenizer = tokenizer
    
    def count_tokens_in_messages(self, messages: List[dict]) -> int:
        """Count total tokens in a list of chat messages."""
        if self.tokenizer is not None:
            # Use proper tokenizer if available, but handle large inputs gracefully
            total_tokens = 0
            for message in messages:
                try:
                    # Tokenize the content of each message
                    tokens = self.tokenizer.encode(message['content'], 
                                                 add_special_tokens=False,
                                                 truncation=False,  # Don't truncate for counting
                                                 max_length=None)   # No max length limit
                    total_tokens += len(tokens)
                except Exception as e:
                    # If tokenizer fails (e.g., context too long), fall back to word count
                    print(f"Tokenizer failed for long input, using word approximation: {e}")
                    words = len(message['content'].split())
                    total_tokens += int(words * 1.3)  # Rough token approximation
            
            # Add tokens for special chat formatting (approximate)
            # Most chat models add extra tokens for role indicators, special tokens, etc.
            formatting_tokens = len(messages) * 3  # Rough estimate for role tokens and separators
            return total_tokens + formatting_tokens
        else:
            # Fallback to word-based approximation
            total_words = 0
            for message in messages:
                # Simple word count approximation
                words = len(message['content'].split())
                total_words += words
            
            # Rough approximation: 1.3 tokens per word + formatting tokens
            formatting_tokens = len(messages) * 3  # Rough estimate for role tokens and separators
            return int(total_words * 1.3) + formatting_tokens
    
    @staticmethod
    def create_tokenizer(model_name: str = None):
        """Create a tokenizer for token counting."""
        # For API models (when model_name is None), don't use any tokenizer
        if model_name is None:
            print("No tokenizer needed for API models - using word-based approximation")
            return TokenCounter()  # Return tokenizer without any transformer tokenizer
        
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"Initialized tokenizer: {model_name}")
            return TokenCounter(tokenizer)
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            print("Token counting will use word-based approximation.")
            return TokenCounter()
