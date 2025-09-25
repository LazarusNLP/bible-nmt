"""
Configuration management for post-editing pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from core.constants import DEFAULT_MODEL_CONFIGS, SUPPORTED_MODEL_TYPES, SUPPORTED_VECTORIZERS, SUPPORTED_FEW_SHOT_MODES, SUPPORTED_GLOSSARY_MODES


@dataclass
class PostEditingConfig:
    """Configuration class for post-editing pipeline."""
    
    # Model configuration
    model_type: str = "vllm"
    model_name: str = "microsoft/DialoGPT-medium"
    
    # Data configuration
    csv_path: str = ""
    few_shot_corpus_path: Optional[str] = None
    glossary_path: Optional[str] = None
    src_lang: str = "en"
    tgt_lang: str = "id"
    src_lang_name: str = "English"
    tgt_lang_name: str = "Indonesian"
    max_samples: Optional[int] = None
    
    # Processing configuration
    prompt: str = "default"
    num_few_shot: int = 5
    vectorizer: str = "bm25"
    few_shot_mode: str = "parallel"
    glossary_mode: str = "smart"
    max_glossary_entries: Optional[int] = None
    batch_size: Optional[int] = None
    num_workers: int = 8
    batch_timeout: int = 7200
    
    # Output configuration
    output_dir: str = "./results"
    debug: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration values."""
        if self.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        if self.vectorizer not in SUPPORTED_VECTORIZERS:
            raise ValueError(f"Unsupported vectorizer: {self.vectorizer}")
        
        if self.few_shot_mode not in SUPPORTED_FEW_SHOT_MODES:
            raise ValueError(f"Unsupported few-shot mode: {self.few_shot_mode}")
        
        if self.glossary_mode not in SUPPORTED_GLOSSARY_MODES:
            raise ValueError(f"Unsupported glossary mode: {self.glossary_mode}")
        
        if not self.csv_path:
            raise ValueError("csv_path is required")
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        if self.few_shot_corpus_path and not os.path.exists(self.few_shot_corpus_path):
            raise FileNotFoundError(f"Few-shot corpus not found: {self.few_shot_corpus_path}")
        
        if self.glossary_path and not os.path.exists(self.glossary_path):
            raise FileNotFoundError(f"Glossary file not found: {self.glossary_path}")
        
        # Validate few-shot mode requirements
        if self.few_shot_mode == "parallel" and not self.few_shot_corpus_path:
            raise ValueError("few_shot_corpus_path is required when few_shot_mode is 'parallel'")
        
        if self.few_shot_mode == "glossary" and not self.glossary_path:
            raise ValueError("glossary_path is required when few_shot_mode is 'glossary'")
        
        if self.few_shot_mode == "both" and (not self.few_shot_corpus_path or not self.glossary_path):
            raise ValueError("Both few_shot_corpus_path and glossary_path are required when few_shot_mode is 'both'")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        base_config = DEFAULT_MODEL_CONFIGS.get(self.model_type, {}).copy()
        base_config["model_name"] = self.model_name
        return base_config
    
    def get_output_filepath(self) -> str:
        """Generate output filepath."""
        from core.data_models import Row
        return Row.get_savepath(self.output_dir, self.model_name, self.csv_path, 
                               self.src_lang, self.tgt_lang)
    
    def get_corpus_id(self) -> Optional[str]:
        """Generate corpus ID for caching."""
        if self.few_shot_corpus_path:
            return f"corpus_{self.few_shot_corpus_path}_{self.vectorizer}"
        return None
    
    @classmethod
    def from_args(cls, args) -> 'PostEditingConfig':
        """Create configuration from command line arguments."""
        return cls(
            model_type=getattr(args, 'model_type', 'vllm'),
            model_name=getattr(args, 'model_name', 'microsoft/DialoGPT-medium'),
            csv_path=args.csv_path,
            few_shot_corpus_path=getattr(args, 'few_shot_corpus_path', None),
            glossary_path=getattr(args, 'glossary_path', None),
            src_lang=args.src,
            tgt_lang=args.tgt,
            src_lang_name=args.src_lang_name,
            tgt_lang_name=args.tgt_lang_name,
            max_samples=getattr(args, 'max_samples', None),
            prompt=getattr(args, 'prompt', 'default'),
            num_few_shot=getattr(args, 'num_few_shot', 5),
            vectorizer=getattr(args, 'vectorizer', 'bm25'),
            few_shot_mode=getattr(args, 'few_shot_mode', 'parallel'),
            glossary_mode=getattr(args, 'glossary_mode', 'smart'),
            max_glossary_entries=getattr(args, 'max_glossary_entries', None),
            batch_size=getattr(args, 'batch_size', None),
            num_workers=getattr(args, 'num_workers', 8),
            batch_timeout=getattr(args, 'batch_timeout', 7200),
            output_dir=args.output_dir,
            debug=getattr(args, 'debug', False),
        )
    
    def print_summary(self):
        """Print configuration summary."""
        print("="*50)
        print("POST-EDITING CONFIGURATION")
        print("="*50)
        print(f"Model Type: {self.model_type}")
        print(f"Model Name: {self.model_name}")
        print(f"CSV Path: {self.csv_path}")
        print(f"Few-shot Mode: {self.few_shot_mode}")
        print(f"Few-shot Corpus: {self.few_shot_corpus_path or 'None'}")
        print(f"Glossary Path: {self.glossary_path or 'None'}")
        print(f"Glossary Mode: {self.glossary_mode}")
        print(f"Source Language: {self.src_lang_name} ({self.src_lang})")
        print(f"Target Language: {self.tgt_lang_name} ({self.tgt_lang})")
        print(f"Max Samples: {self.max_samples or 'All'}")
        print(f"Prompt: {self.prompt}")
        print(f"Vectorizer: {self.vectorizer}")
        print(f"Few-shot Examples: {self.num_few_shot}")
        print(f"Max Glossary Entries: {self.max_glossary_entries or 'All available'}")
        print(f"Batch Size: {self.batch_size or 'Auto'}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Debug Mode: {self.debug}")
        print("="*50)
