"""
Constants and schemas used across the post-editing pipeline.
"""

# JSON schema for structured output
POST_EDIT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "post_edited_text": {
            "type": "string",
            "description": "The corrected and improved translation in the target language"
        }
    },
    "required": ["post_edited_text"],
    "additionalProperties": False
}

# Default model configurations
DEFAULT_MODEL_CONFIGS = {
    "gpt": {
        "model_name": "gpt-4o",
        "max_tokens": 4096,  # Increased for longer outputs
        "temperature": 0.0
    },
    "gemini": {
        "model_name": "gemini-2.5-flash", 
        "max_tokens": 8192,  # Gemini 2.5 Flash supports up to 8192 output tokens
        "temperature": 0.0
    },
    "vllm": {
        "max_model_len": 6000,  # Increased context window for large prompts
        "gpu_memory_utilization": 0.95,
        "temperature": 0.0,
        "max_tokens": 512
    }
}

# Supported vectorizer types
SUPPORTED_VECTORIZERS = ["bm25", "tfidf", "sbert", "chrf_rag", "word_lcs", "full"]

# Supported model types  
SUPPORTED_MODEL_TYPES = ["gpt", "gemini", "vllm"]

# Supported few-shot modes
SUPPORTED_FEW_SHOT_MODES = ["parallel", "glossary", "both"]

# Supported glossary modes
SUPPORTED_GLOSSARY_MODES = ["smart", "full"]
