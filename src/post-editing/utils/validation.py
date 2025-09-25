"""
Validation utilities for arguments and inputs.
"""

import os
from pathlib import Path
from typing import List, Optional

from core.constants import SUPPORTED_VECTORIZERS, SUPPORTED_MODEL_TYPES


class ArgumentValidator:
    """Validates command line arguments and configuration."""
    
    @staticmethod
    def validate_model_type(model_type: str) -> bool:
        """Validate model type is supported."""
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {SUPPORTED_MODEL_TYPES}")
        return True
    
    @staticmethod
    def validate_vectorizer(vectorizer: str) -> bool:
        """Validate vectorizer type is supported."""
        if vectorizer not in SUPPORTED_VECTORIZERS:
            raise ValueError(f"Unsupported vectorizer: {vectorizer}. "
                           f"Supported types: {SUPPORTED_VECTORIZERS}")
        return True
    
    @staticmethod
    def validate_file_exists(filepath: str, file_type: str = "file") -> bool:
        """Validate that a file exists."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{file_type.capitalize()} not found: {filepath}")
        return True
    
    @staticmethod
    def validate_csv_columns(csv_path: str, required_columns: List[str]) -> bool:
        """Validate CSV file has required columns."""
        import csv
        
        ArgumentValidator.validate_file_exists(csv_path, "CSV file")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            missing_columns = [col for col in required_columns if col not in reader.fieldnames]
            
            if missing_columns:
                raise ValueError(f"CSV file missing required columns: {missing_columns}. "
                               f"Found columns: {reader.fieldnames}")
        return True
    
    @staticmethod
    def validate_output_directory(output_dir: str) -> bool:
        """Validate output directory exists or can be created."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            return True
        except Exception as e:
            raise PermissionError(f"Cannot create output directory {output_dir}: {e}")
    
    @staticmethod
    def validate_environment_variables(model_type: str) -> bool:
        """Validate required environment variables are set."""
        required_vars = {
            "gpt": ["OPENAI_API_KEY"],
            "gemini": ["GEMINI_API_KEY"],
            "vllm": []  # No API keys required for vLLM
        }
        
        missing_vars = []
        for var in required_vars.get(model_type, []):
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables for {model_type}: {missing_vars}")
        
        return True
    
    @staticmethod
    def validate_prompt_key(prompt_key: str) -> bool:
        """Validate prompt key exists."""
        # Import here to avoid circular imports
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from prompt import get_prompt
        
        try:
            get_prompt(prompt_key)
            return True
        except KeyError as e:
            raise ValueError(f"Invalid prompt key: {e}")
    
    @staticmethod
    def validate_all_arguments(args) -> bool:
        """Validate all arguments at once."""
        # Validate model type
        ArgumentValidator.validate_model_type(args.model_type)
        
        # Validate CSV file and columns
        ArgumentValidator.validate_csv_columns(
            args.csv_path, 
            ['source_text', 'target_text', 'pred_target_text']
        )
        
        # Validate few-shot corpus if provided
        if args.few_shot_corpus_path:
            ArgumentValidator.validate_file_exists(args.few_shot_corpus_path, "Few-shot corpus")
        
        # Validate vectorizer
        ArgumentValidator.validate_vectorizer(args.vectorizer)
        
        # Validate output directory
        ArgumentValidator.validate_output_directory(args.output_dir)
        
        # Validate environment variables
        ArgumentValidator.validate_environment_variables(args.model_type)
        
        # Validate prompt key
        ArgumentValidator.validate_prompt_key(args.prompt)
        
        print("✅ All arguments validated successfully")
        return True
