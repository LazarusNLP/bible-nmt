#!/usr/bin/env python3
"""
Unified post-editing script supporting multiple model types and retrieval methods.

This script provides a unified interface for post-editing machine translation output
using various language models (GPT, Gemini, vLLM) and retrieval methods (BM25, TF-IDF, SBERT).

Usage:
    python run_post_editing_fixed.py --model_type vllm --model_name path/to/model ...
    python run_post_editing_fixed.py --model_type gpt --model_name gpt-4o ...
    python run_post_editing_fixed.py --model_type gemini --model_name gemini-2.5-flash ...
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Add current directory to Python path to enable absolute imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.data_io import DataHandler
    from core.constants import DEFAULT_MODEL_CONFIGS, SUPPORTED_MODEL_TYPES, SUPPORTED_VECTORIZERS, SUPPORTED_FEW_SHOT_MODES, SUPPORTED_GLOSSARY_MODES
    from models.base import BaseLLM
    from models.api_models import GPTModel, GeminiModel
    from models.vllm_models import VLLMModel
    from models.tokenization import TokenCounter
    from retrieval.few_shot import FewShotSelector
    from retrieval.glossary import GlossarySelector
    from utils.processing import ProcessingEngine
    from utils.validation import ArgumentValidator
    from utils.logging import setup_logging
    from prompt import list_available_prompts
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the post-editing directory")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified post-editing script for machine translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using vLLM with local model
  python run_post_editing_fixed.py --model_type vllm --model_name microsoft/DialoGPT-medium \\
    --csv_path data.csv --src en --tgt id --src_lang_name English --tgt_lang_name Indonesian \\
    --output_dir ./results

  # Using GPT API with few-shot examples
  python run_post_editing_fixed.py --model_type gpt --model_name gpt-4o \\
    --csv_path data.csv --few_shot_corpus_path corpus.csv --vectorizer sbert \\
    --src en --tgt id --src_lang_name English --tgt_lang_name Indonesian \\
    --output_dir ./results
        """
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--model_type", type=str, required=True, 
                           choices=["gpt", "gemini", "vllm"],
                           help="Type of model to use")
    model_group.add_argument("--model_name", type=str, required=True,
                           help="Model name or path (e.g., 'gpt-4o', 'gemini-2.5-flash', 'path/to/vllm/model')")
    
    # Data configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument("--csv_path", type=str, required=True,
                          help="Path to CSV file with source_text, target_text, and pred_target_text columns")
    data_group.add_argument("--few_shot_corpus_path", type=str,
                          help="Path to few-shot corpus file (CSV with source_text/target_text columns or text file)")
    data_group.add_argument("--glossary_path", type=str,
                          help="Path to glossary file (CSV with source_word, target_word, and optional pos columns)")
    data_group.add_argument("--src", type=str, required=True,
                          help="Source language alpha-2 code (e.g., 'en')")
    data_group.add_argument("--tgt", type=str, required=True,
                          help="Target language alpha-2 code (e.g., 'id')")
    data_group.add_argument("--src_lang_name", type=str, required=True,
                          help="Source language name (e.g., 'English')")
    data_group.add_argument("--tgt_lang_name", type=str, required=True,
                          help="Target language name (e.g., 'Indonesian')")
    data_group.add_argument("--max_samples", type=int,
                          help="Maximum number of samples to process (optional)")
    
    # Processing configuration
    proc_group = parser.add_argument_group('Processing Configuration')
    proc_group.add_argument("--prompt", type=str, default="default",
                          help=f"Prompt template key. Options: {', '.join(list_available_prompts())}")
    proc_group.add_argument("--num_few_shot", type=int, default=5,
                          help="Number of few-shot examples to use")
    proc_group.add_argument("--vectorizer", type=str, default="bm25", 
                          choices=["bm25", "tfidf", "sbert", "chrf_rag", "word_lcs", "full"],
                          help="Similarity method for few-shot selection ('full' uses entire corpus)")
    proc_group.add_argument("--few_shot_mode", type=str, default="parallel",
                          choices=["parallel", "glossary", "both"],
                          help="Few-shot mode: 'parallel' (examples), 'glossary' (terminology), or 'both'")
    proc_group.add_argument("--glossary_mode", type=str, default="smart",
                          choices=["smart", "full"],
                          help="Glossary mode: 'smart' (intelligent matching) or 'full' (entire glossary)")
    proc_group.add_argument("--max_glossary_entries", type=int, default=None,
                          help="Maximum number of glossary entries per input (default: use all available)")
    proc_group.add_argument("--batch_size", type=int,
                          help="Process messages in batches (optional)")
    proc_group.add_argument("--num_workers", type=int, default=8,
                          help="Number of worker threads for API calls")
    proc_group.add_argument("--batch_timeout", type=int, default=7200,
                          help="Timeout for batch jobs in seconds (Gemini only)")
    proc_group.add_argument("--sequential", action="store_true",
                          help="Process requests sequentially without concurrency (disables batch processing and threading)")
    
    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument("--output_dir", type=str, required=True,
                            help="Output directory for results")
    output_group.add_argument("--debug", action="store_true",
                            help="Enable debug mode")
    output_group.add_argument("--prompt-only", action="store_true",
                            help="Print final LLM prompt for first sample and exit (for debugging)")
    output_group.add_argument("--log_file", type=str,
                            help="Optional log file path")
    
    return parser.parse_args()


def create_model(model_type: str, model_name: str):
    """Create model instance based on type and name."""
    if model_type == "gpt":
        return GPTModel(model_name)
    elif model_type == "gemini":
        return GeminiModel(model_name)
    elif model_type == "vllm":
        return VLLMModel(model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def create_few_shot_selector(vectorizer: str, few_shot_corpus_path: str, few_shot_mode: str) -> FewShotSelector:
    """Create few-shot selector if corpus is provided and mode requires it."""
    if few_shot_corpus_path and few_shot_mode in ["parallel", "both"]:
        corpus_id = f"corpus_{few_shot_corpus_path}_{vectorizer}"
        return FewShotSelector(vectorizer, corpus_id)
    return None


def create_glossary_selector(glossary_path: str, few_shot_mode: str, glossary_mode: str = "smart") -> GlossarySelector:
    """Create glossary selector if glossary is provided and mode requires it."""
    if glossary_path and few_shot_mode in ["glossary", "both"]:
        # Load glossary
        data_handler = DataHandler()
        glossary = data_handler.load_glossary(glossary_path)
        return GlossarySelector(glossary, mode=glossary_mode)
    return None


def get_output_filepath(output_dir: str, model_name: str, csv_path: str, src: str, tgt: str) -> str:
    """Generate output filepath."""
    csv_name = csv_path.split('/')[-1].replace('.csv', '')
    return f"{output_dir}/{model_name.split('/')[-1]}_{csv_name}_{src}_{tgt}.csv"


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(
        level="DEBUG" if args.debug else "INFO",
        log_file=getattr(args, 'log_file', None)
    )
    
    try:
        logger.info("Starting post-editing pipeline...")
        
        # Print configuration summary
        print("="*50)
        print("POST-EDITING CONFIGURATION")
        print("="*50)
        print(f"Model Type: {args.model_type}")
        print(f"Model Name: {args.model_name}")
        print(f"CSV Path: {args.csv_path}")
        print(f"Few-shot Mode: {args.few_shot_mode}")
        print(f"Few-shot Corpus: {args.few_shot_corpus_path or 'None'}")
        print(f"Glossary Path: {args.glossary_path or 'None'}")
        print(f"Glossary Mode: {args.glossary_mode}")
        print(f"Source Language: {args.src_lang_name} ({args.src})")
        print(f"Target Language: {args.tgt_lang_name} ({args.tgt})")
        print(f"Max Samples: {args.max_samples or 'All'}")
        print(f"Prompt: {args.prompt}")
        print(f"Vectorizer: {args.vectorizer}")
        print(f"Few-shot Examples: {args.num_few_shot}")
        print(f"Max Glossary Entries: {args.max_glossary_entries or 'All available'}")
        print(f"Batch Size: {args.batch_size or 'Auto'}")
        print(f"Output Directory: {args.output_dir}")
        print(f"Debug Mode: {args.debug}")
        print(f"Sequential Mode: {getattr(args, 'sequential', False)}")
        print(f"Prompt-Only Mode: {getattr(args, 'prompt_only', False)}")
        print("="*50)
        
        # Basic validation
        if args.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        
        if args.vectorizer not in SUPPORTED_VECTORIZERS:
            raise ValueError(f"Unsupported vectorizer: {args.vectorizer}")
        
        if args.few_shot_mode not in SUPPORTED_FEW_SHOT_MODES:
            raise ValueError(f"Unsupported few-shot mode: {args.few_shot_mode}")
        
        # Validate few-shot mode requirements
        if args.few_shot_mode == "parallel" and not args.few_shot_corpus_path:
            raise ValueError("--few_shot_corpus_path is required when --few_shot_mode is 'parallel'")
        
        if args.few_shot_mode == "glossary" and not args.glossary_path:
            raise ValueError("--glossary_path is required when --few_shot_mode is 'glossary'")
        
        if args.few_shot_mode == "both" and (not args.few_shot_corpus_path or not args.glossary_path):
            raise ValueError("Both --few_shot_corpus_path and --glossary_path are required when --few_shot_mode is 'both'")
        
        # Load data
        data_handler = DataHandler()
        rows = data_handler.load_from_csv(
            args.csv_path, args.src, args.tgt,
            args.src_lang_name, args.tgt_lang_name, args.max_samples
        )
        
        # Load few-shot corpus if provided and mode requires it
        few_shot_corpus = None
        if args.few_shot_corpus_path and args.few_shot_mode in ["parallel", "both"]:
            few_shot_corpus = data_handler.load_few_shot_corpus(args.few_shot_corpus_path)
        
        # Create model
        logger.info(f"Initializing {args.model_type} model...")
        llm = create_model(args.model_type, args.model_name)
        
        # Create few-shot selector
        few_shot_selector = create_few_shot_selector(args.vectorizer, args.few_shot_corpus_path, args.few_shot_mode)
        
        # Create glossary selector
        glossary_selector = create_glossary_selector(args.glossary_path, args.few_shot_mode, args.glossary_mode)
        
        # Create token counter (only for vLLM models, API models don't need local tokenization)
        token_counter = TokenCounter.create_tokenizer(
            args.model_name if args.model_type == "vllm" else None
        )
        
        # Create processing engine
        processor = ProcessingEngine(llm, few_shot_selector, glossary_selector, token_counter)
        
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        output_filepath = get_output_filepath(args.output_dir, args.model_name, args.csv_path, 
                                            args.src, args.tgt)
        
        # Optimize batch size for async models
        optimized_batch_size = args.batch_size
        sequential_mode = getattr(args, 'sequential', False)
        
        if sequential_mode:
            # Force sequential processing
            optimized_batch_size = 1  # Process one at a time
            print(f"Sequential mode enabled: Processing requests one by one without concurrency")
        elif args.model_type == "gemini" and not args.batch_size:
            # For Gemini async processing, disable outer batching to avoid event loop issues
            # Process all rows in one go with internal async sub-batching
            optimized_batch_size = None  # No outer batching
            print(f"Optimized for Gemini: No outer batching, using internal async sub-batching only")
        
        # Process rows
        results = processor.process_rows(
            rows=rows,
            output_filepath=output_filepath,
            prompt_key=args.prompt,
            few_shot_corpus=few_shot_corpus,
            num_few_shot=args.num_few_shot,
            max_glossary_entries=args.max_glossary_entries,
            batch_size=optimized_batch_size,
            num_workers=args.num_workers,
            debug=args.debug,
            prompt_only=getattr(args, 'prompt_only', False),
            sequential=sequential_mode
        )
        
        logger.info("Post-editing pipeline completed successfully!")
        
        # Print final results
        if results:
            print("\n" + "="*60)
            print("FINAL RESULTS SUMMARY")
            print("="*60)
            
            if "original_mt_metrics" in results:
                print("\nOriginal MT Metrics:")
                for metric, score in results["original_mt_metrics"].items():
                    print(f"  {metric.upper()}: {score:.2f}")
            
            if "post_edited_metrics" in results:
                print("\nPost-Edited Metrics:")
                for metric, score in results["post_edited_metrics"].items():
                    print(f"  {metric.upper()}: {score:.2f}")
            
            if "improvements" in results:
                print("\nImprovements:")
                for metric, improvement in results["improvements"].items():
                    sign = "+" if improvement >= 0 else ""
                    print(f"  {metric.upper()}: {sign}{improvement:.2f}")
            
            print("="*60)
        
    except Exception as e:
        logger.error(f"Error in post-editing pipeline: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
