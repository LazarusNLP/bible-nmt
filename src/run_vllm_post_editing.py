# Set multiprocessing start method to 'spawn' BEFORE any other imports to avoid CUDA initialization issues
import multiprocessing
import os
from pathlib import Path
import warnings

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['DISABLE_XFORMERS'] = '1'
os.environ['FLASH_ATTENTION_SKIP_CUDA_BUILD'] = '1'
os.environ['ENFORCE_EAGER'] = '1'

# Suppress specific warnings related to resource cleanup
os.environ['NCCL_SUPPRESS_WARN'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'  # Reduce NCCL verbosity

# Fix device configuration issues for vLLM
import torch
if torch.cuda.is_available():
    # Ensure CUDA device is properly configured
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    print(f"CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
else:
    print("CUDA not available, will attempt to use CPU (not recommended for large models)")

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        print("No .env file found, proceeding without loading environment variables")
except ImportError:
    print("python-dotenv not installed, skipping .env file loading")

import csv
import json
import argparse
from functools import cached_property
from typing import List, Optional
from pydantic import BaseModel, Field
from sacrebleu import corpus_chrf
import evaluate
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer

from prompt import get_prompt, list_available_prompts


# Global cache for SBERT model and corpus embeddings to avoid reloading/recomputing
_sbert_model_cache = None
_corpus_embeddings_cache = {}

def get_sbert_model():
    """Get cached SBERT model or load it if not cached"""
    global _sbert_model_cache
    if _sbert_model_cache is None:
        print("Loading SBERT model (cached for reuse)...")
        _sbert_model_cache = SentenceTransformer("LazarusNLP/all-indo-e5-small-v4")
    return _sbert_model_cache

def get_corpus_embeddings(corpus: List[tuple], corpus_id: str = None):
    """Get cached corpus embeddings or compute them if not cached"""
    global _corpus_embeddings_cache
    
    # Create a unique identifier for this corpus
    if corpus_id is None:
        # Use hash of corpus content as identifier
        corpus_content = str(sorted(corpus))
        corpus_id = str(hash(corpus_content))
    
    if corpus_id not in _corpus_embeddings_cache:
        print(f"Computing and caching embeddings for corpus (ID: {corpus_id[:8]}...) with {len(corpus)} examples...")
        sbert_model = get_sbert_model()
        source_texts = [pair[0] for pair in corpus]
        embeddings = sbert_model.encode(source_texts, show_progress_bar=True)
        _corpus_embeddings_cache[corpus_id] = {
            'embeddings': embeddings,
            'source_texts': source_texts,
            'corpus': corpus
        }
        print(f"✅ Corpus embeddings cached successfully!")
    else:
        print(f"Using cached embeddings for corpus (ID: {corpus_id[:8]}...)")
    
    return _corpus_embeddings_cache[corpus_id]


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


def count_tokens_in_messages(messages: List[dict], tokenizer) -> int:
    """Count total tokens in a list of chat messages"""
    total_tokens = 0
    for message in messages:
        # Tokenize the content of each message
        tokens = tokenizer.encode(message['content'], add_special_tokens=False)
        total_tokens += len(tokens)
    
    # Add tokens for special chat formatting (approximate)
    # Most chat models add extra tokens for role indicators, special tokens, etc.
    formatting_tokens = len(messages) * 3  # Rough estimate for role tokens and separators
    return total_tokens + formatting_tokens


def parse_json_response(response_text: str) -> str:
    """Safely parse JSON response from vLLM and extract post_edited_text"""
    try:
        # Clean the response text first
        cleaned_text = response_text.strip()
        
        # With guided JSON generation, response should be pure JSON
        response_json = json.loads(cleaned_text)
        
        if isinstance(response_json, dict) and 'post_edited_text' in response_json:
            extracted_text = response_json['post_edited_text'].strip()
            
            # Additional validation: ensure it's not just echoing the source
            if extracted_text and len(extracted_text) > 0:
                return extracted_text
            else:
                print(f"Warning: Empty post_edited_text in JSON response")
                return "[EXTRACTION_FAILED]"
        else:
            print(f"Warning: JSON response missing 'post_edited_text' field: {response_json}")
            return "[INVALID_JSON_STRUCTURE]"
                
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON response: {e}")
        print(f"Raw response (first 200 chars): {response_text[:200]}...")
        
        # With guided JSON, this should rarely happen, but provide fallback
        # Look for JSON-like content in case there's extra text
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            try:
                json_part = cleaned_text[start_idx:end_idx]
                response_json = json.loads(json_part)
                if isinstance(response_json, dict) and 'post_edited_text' in response_json:
                    return response_json['post_edited_text'].strip()
            except json.JSONDecodeError:
                pass
                
        return "[JSON_PARSE_FAILED]"


def parse_args():
    parser = argparse.ArgumentParser(description="Post-edit machine translation using a VLLM model with comprehensive evaluation metrics")
    parser.add_argument("--model", type=str, required=True, help="VLLM model name or path")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file containing source_text, target_text, and pred_target_text columns")
    parser.add_argument("--few_shot_corpus_path", type=str, required=False, help="Path to few-shot corpus file (txt or csv with 'source_text' and 'target_text' columns) (optional)")
    parser.add_argument("--src", type=str, required=True, help="alpha-2 source language code")
    parser.add_argument("--tgt", type=str, required=True, help="alpha-2 target language code")
    parser.add_argument("--src_lang_name", type=str, required=True, help="Source language name (e.g., 'Indonesian')")
    parser.add_argument("--tgt_lang_name", type=str, required=True, help="Target language name (e.g., 'Dhao')")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--prompt", type=str, default="default", 
                       help=f"Prompt template key. Available options: {', '.join(list_available_prompts())}")
    parser.add_argument("--num_few_shot", type=int, default=5, help="Number of few-shot examples to use")
    parser.add_argument("--vectorizer", type=str, default="bm25", choices=["bm25", "tfidf", "sbert"], help="Similarity method for few-shot selection")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process from CSV (optional, processes all if not specified)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to print created messages during inference")
    parser.add_argument("--batch_size", type=int, default=None, help="Process messages in batches of this size to reduce memory usage (optional, processes all at once if not specified)")
    return parser.parse_args()


class Row(BaseModel):
    src_text: str
    tgt_text: str
    pred_tgt_text: str = Field(description="Original NLLB model prediction from input CSV")
    post_edited_tgt_txt: Optional[str] = None
    src_lang: str = Field(description="Src lang alpha-2 code")
    tgt_lang: str = Field(description="Tgt lang alpha-2 code")
    src_lang_name: str = Field(description="Source language name")
    tgt_lang_name: str = Field(description="Target language name")

    @classmethod
    def get_savepath(cls, output_dir, model, csv_path, src, tgt):
        csv_name = csv_path.split('/')[-1].replace('.csv', '')
        return f"{output_dir}/{model.split('/')[-1]}_{csv_name}_{src}_{tgt}.csv"

    @classmethod
    def load_from_savepath(cls, filepath) -> List['Row']:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [cls.model_validate(row) for row in reader]
    
    @classmethod
    def save_to_savepath(cls, rows: List['Row'], filepath):
        print(f"Saving {len(rows)} rows to {filepath}")
        os.makedirs(Path(filepath).parent, exist_ok=True)
        with open(file=filepath, mode="w", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cls.model_fields.keys(), 
                                  quoting=csv.QUOTE_MINIMAL, quotechar='"')
            writer.writeheader()
            for row in rows:
                writer.writerow(row.model_dump())
    
    # Language names are now provided directly as fields, no need for cached_property

    @classmethod
    def load_from_files(cls, source_path, target_path, pred_path, src_lang, tgt_lang, src_lang_name, tgt_lang_name) -> List['Row']:
        """Load data from source, target, and prediction text files"""
        print(f"Loading data from {source_path}, {target_path}, and {pred_path}")
        
        with open(source_path, 'r', encoding='utf-8') as src_file:
            src_lines = [line.strip() for line in src_file.readlines()]
        
        with open(target_path, 'r', encoding='utf-8') as tgt_file:
            tgt_lines = [line.strip() for line in tgt_file.readlines()]
            
        with open(pred_path, 'r', encoding='utf-8') as pred_file:
            pred_lines = [line.strip() for line in pred_file.readlines()]
        
        assert len(src_lines) == len(tgt_lines), f"Source file has {len(src_lines)} lines but target file has {len(tgt_lines)} lines"
        assert len(src_lines) == len(pred_lines), f"Source file has {len(src_lines)} lines but prediction file has {len(pred_lines)} lines"
        
        rows = []
        for src_text, tgt_text, pred_text in zip(src_lines, tgt_lines, pred_lines):
                if src_text.strip() and tgt_text.strip() and pred_text.strip():  # Skip empty lines
                    rows.append(cls(
                        src_text=src_text,
                        tgt_text=tgt_text,
                        pred_tgt_text=pred_text,
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        src_lang_name=src_lang_name,
                        tgt_lang_name=tgt_lang_name
                    ))
        
        print(f"Loaded {len(rows)} sentence pairs")
        return rows

    @classmethod
    def load_from_csv(cls, csv_path, src_lang, tgt_lang, src_lang_name, tgt_lang_name, max_samples=None) -> List['Row']:
        """Load data from CSV file containing source_text, target_text, and pred_target_text columns"""
        print(f"Loading data from {csv_path}")
        if max_samples:
            print(f"Limiting to first {max_samples} samples")
        
        rows = []
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row_data in enumerate(reader):
                # Check if we've reached the maximum number of samples
                if max_samples and len(rows) >= max_samples:
                    break
                    
                # Check for required columns
                if 'source_text' not in row_data or 'target_text' not in row_data or 'pred_target_text' not in row_data:
                    raise ValueError(f"CSV file must contain 'source_text', 'target_text', and 'pred_target_text' columns")
                
                src_text = row_data['source_text'].strip()
                tgt_text = row_data['target_text'].strip()
                pred_tgt_text = row_data['pred_target_text'].strip()
                
                # Skip empty lines
                if src_text and tgt_text and pred_tgt_text:
                    rows.append(cls(
                        src_text=src_text,
                        tgt_text=tgt_text,
                        pred_tgt_text=pred_tgt_text,
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        src_lang_name=src_lang_name,
                        tgt_lang_name=tgt_lang_name
                    ))
        
        print(f"Loaded {len(rows)} sentence pairs")
        return rows

    
    def get_prompt(self) -> str:
        return f"Translate the following text to {self.tgt_lang}:\n\n{self.src_text}\n\nTranslation:"
    
    def get_messages(self, prompt_key: str = "default", few_shot_examples: List[tuple] = None) -> List[dict]:
        # Get the prompt template
        prompt_template = get_prompt(prompt_key)
        
        # Build the prompt content using the template
        system_content = prompt_template["system"]
        
        # Use the user_template from prompt with proper variable substitution
        base_user_content = prompt_template["user_template"].format(
            src_lang_name=self.src_lang_name,
            src_text=self.src_text,
            tgt_lang_name=self.tgt_lang_name,
            pred_text=self.pred_tgt_text
        )
        
        # Add few-shot examples if provided
        if few_shot_examples:
            examples_text = f"\n\nHere are some examples of verified correct translations:\n"
            for src_ex, tgt_ex in few_shot_examples:
                examples_text += f"{self.src_lang_name}: {src_ex}\n{self.tgt_lang_name}: {tgt_ex}\n\n"
            user_content = base_user_content + examples_text
        else:
            user_content = base_user_content
        
        # Return messages with system and user content
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        return messages

    @classmethod
    def _compute_metrics_for_predictions(cls, predictions: List[str], references: List[str]) -> dict:
        """Helper method to compute metrics for given predictions and references"""
        # Postprocess text (strip whitespace)
        cleaned_preds = [pred.strip() for pred in predictions]
        cleaned_labels = [[label.strip()] for label in references]
        
        # Load evaluation metrics
        chrf = evaluate.load("chrf")
        sacrebleu = evaluate.load("sacrebleu")
        spbleu = evaluate.load("sacrebleu")
        
        # Calculate BLEU score using NLTK (matching run_evaluation.py)
        tokenized_preds = [word_tokenize(pred.lower().strip()) for pred in cleaned_preds]
        tokenized_refs = [[word_tokenize(ref[0].lower().strip())] for ref in cleaned_labels]
        bleu_score = corpus_bleu(tokenized_refs, tokenized_preds)
        
        # Calculate SacreBLEU scores
        sacrebleu_result = sacrebleu.compute(predictions=cleaned_preds, references=cleaned_labels)
        spbleu_result = spbleu.compute(predictions=cleaned_preds, references=cleaned_labels, tokenize="flores200")
        
        # Calculate chrF variants
        chrf_result = chrf.compute(predictions=cleaned_preds, references=cleaned_labels)
        chrf3_result = chrf.compute(predictions=cleaned_preds, references=cleaned_labels, beta=3)
        chrf_plus_result = chrf.compute(predictions=cleaned_preds, references=cleaned_labels, word_order=2)
        
        # Compile results
        eval_result = {
            "bleu": bleu_score * 100,  # Convert to percentage like run_evaluation.py
            "sacrebleu": sacrebleu_result["score"],
            "spbleu": spbleu_result["score"],
            "chrf": chrf_result["score"],
            "chrf3": chrf3_result["score"],
            "chrf++": chrf_plus_result["score"],
        }
        eval_result = {k: round(v, 4) for k, v in eval_result.items()}
        return eval_result

    @classmethod
    def calculate_metrics(cls, rows: List['Row']) -> dict:
        """Calculate all evaluation metrics including original MT and post-edited results with improvements"""
        # make sure all the rows have post_edited_tgt_txt
        assert all(r.post_edited_tgt_txt is not None for r in rows), "Some rows do not have post_edited_tgt_txt"
        
        # Extract texts
        original_predictions = [r.pred_tgt_text for r in rows]  # Original MT output
        post_edited_predictions = [r.post_edited_tgt_txt for r in rows]  # Post-edited output
        references = [r.tgt_text for r in rows]  # Ground truth
        
        # Calculate metrics for original MT output
        print("Calculating metrics for original MT output...")
        original_metrics = cls._compute_metrics_for_predictions(original_predictions, references)
        
        # Calculate metrics for post-edited output
        print("Calculating metrics for post-edited output...")
        post_edited_metrics = cls._compute_metrics_for_predictions(post_edited_predictions, references)
        
        # Calculate improvements (delta)
        improvements = {}
        for metric in original_metrics.keys():
            improvement = post_edited_metrics[metric] - original_metrics[metric]
            improvements[metric] = round(improvement, 4)
        
        # Compile final results
        final_results = {
            "original_mt_metrics": original_metrics,
            "post_edited_metrics": post_edited_metrics,
            "improvements": improvements
        }
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS SUMMARY")
        print("="*50)
        print("\nOriginal MT Metrics:")
        for metric, score in original_metrics.items():
            print(f"  {metric.upper()}: {score:.2f}")
        
        print("\nPost-Edited Metrics:")
        for metric, score in post_edited_metrics.items():
            print(f"  {metric.upper()}: {score:.2f}")
        
        print("\nImprovements (Post-Edited - Original):")
        for metric, improvement in improvements.items():
            sign = "+" if improvement >= 0 else ""
            print(f"  {metric.upper()}: {sign}{improvement:.2f}")
        print("="*50)
        
        return final_results


def load_few_shot_corpus(corpus_path: str) -> List[tuple]:
    """Load few-shot corpus from text file or CSV file with parallel source/target text"""
    print(f"Loading few-shot corpus from {corpus_path}")
    
    # Check if file is CSV or TXT based on extension
    if corpus_path.lower().endswith('.csv'):
        corpus = []
        with open(corpus_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            required_columns = ['source_text', 'target_text']
            missing_columns = [col for col in required_columns if col not in reader.fieldnames]
            if missing_columns:
                raise ValueError(f"CSV file must contain {required_columns} columns. Missing: {missing_columns}. Found columns: {reader.fieldnames}")
            
            for row in reader:
                src_text = row['source_text'].strip()
                tgt_text = row['target_text'].strip()
                if src_text and tgt_text:  # Skip empty lines
                    corpus.append((src_text, tgt_text))
        print(f"Loaded {len(corpus)} parallel sentence pairs from CSV for few-shot corpus")
    else:
        # Default behavior for text files - treat each line as source text, duplicate as target
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            corpus = [(line, line) for line in lines]  # Use same text for both source and target
        print(f"Loaded {len(corpus)} sentences from text file for few-shot corpus (duplicated as source-target pairs)")
    
    return corpus


def get_few_shot_examples(query: str, corpus: List[tuple], k: int, vectorizer_type: str = "bm25", corpus_id: str = None) -> List[tuple]:
    """Get top-k similar examples from parallel corpus using specified vectorizer"""
    
    # Extract source texts for similarity computation
    source_texts = [pair[0] for pair in corpus]
    
    if vectorizer_type == "tfidf":
        print(f"Getting {k} few-shot examples using TF-IDF vectorizer")
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(source_texts)
        query_vector = vectorizer.transform([query])
        cosine_similarities = (tfidf_matrix * query_vector.T).toarray().flatten()
        top_k_indices = cosine_similarities.argsort()[-k:][::-1]
        return [corpus[i] for i in top_k_indices]
    
    elif vectorizer_type == "bm25":
        print(f"Getting {k} few-shot examples using BM25 vectorizer")
        tokenized_corpus = [doc.split() for doc in source_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        top_k_indices = scores.argsort()[-k:][::-1]
        return [corpus[i] for i in top_k_indices]
    
    elif vectorizer_type == "sbert":
        print(f"Getting {k} few-shot examples using SBERT vectorizer (with caching)")
        
        # Get cached corpus embeddings (this will compute and cache if not already cached)
        corpus_data = get_corpus_embeddings(corpus, corpus_id)
        embeddings = corpus_data['embeddings']
        cached_corpus = corpus_data['corpus']
        
        # Compute query embedding only
        sbert_model = get_sbert_model()
        query_embedding = sbert_model.encode([query], show_progress_bar=False)  # (1, dim)
        
        # Compute similarities using cached corpus embeddings
        cosine_similarities = embeddings @ query_embedding.T  # (n_samples, 1)
        top_k_indices = cosine_similarities[:, 0].argsort()[-k:][::-1]
        return [cached_corpus[i] for i in top_k_indices]
    
    else:
        raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")


def find_unprocessed_rows(all_rows: List[Row], output_filepath: str) -> List[Row]:
    """Find rows that haven't been processed yet by comparing with existing TSV file"""
    if not os.path.exists(output_filepath):
        print("No existing output file found. Processing all rows.")
        return all_rows
    
    try:
        existing_rows = Row.load_from_savepath(output_filepath)
        print(f"Found existing output file with {len(existing_rows)} completed rows")
        
        # Create a set of completed row identifiers (using src_text as unique identifier)
        completed_src_texts = {row.src_text for row in existing_rows if row.post_edited_tgt_txt}
        
        # Find unprocessed rows
        unprocessed_rows = [row for row in all_rows if row.src_text not in completed_src_texts]
        print(f"Found {len(unprocessed_rows)} unprocessed rows out of {len(all_rows)} total rows")
        
        return unprocessed_rows
    except Exception as e:
        print(f"Error reading existing file: {e}. Processing all rows.")
        return all_rows


def append_row_to_csv(row: Row, filepath: str):
    """Append a single row to the CSV file"""
    file_exists = os.path.exists(filepath)
    
    with open(filepath, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.model_fields.keys(), 
                              quoting=csv.QUOTE_MINIMAL, quotechar='"')
        
        # Write header only if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row.model_dump())


def main():
    args = parse_args()
    corpus_info = f"Few-shot corpus: {args.few_shot_corpus_path}" if args.few_shot_corpus_path else "Few-shot corpus: None"
    samples_info = f"Max samples: {args.max_samples}" if args.max_samples else "Max samples: All"
    print(f"Model: {args.model}; CSV: {args.csv_path}; {corpus_info}; {samples_info}; Source Lang: {args.src}; Target Lang: {args.tgt}; Prompt: {args.prompt}")

    # Validate prompt key
    try:
        get_prompt(args.prompt)
    except KeyError as e:
        print(f"Error: {e}")
        return

    # Load the data from CSV file
    rows = Row.load_from_csv(args.csv_path, args.src, args.tgt, args.src_lang_name, args.tgt_lang_name, args.max_samples)

    # Load few-shot corpus if provided (examples will be calculated only for unprocessed rows)
    few_shot_corpus = None
    if args.few_shot_corpus_path:
        few_shot_corpus = load_few_shot_corpus(args.few_shot_corpus_path)
        print("Few-shot corpus loaded (examples will be calculated for unprocessed rows only)")
    else:
        print("No few-shot corpus provided, using prompts without few-shot examples")

    # Setup output file path and directory
    os.makedirs(args.output_dir, exist_ok=True)
    filepath = Row.get_savepath(args.output_dir, args.model, args.csv_path, args.src, args.tgt)
    
    # Find unprocessed rows (for resumption capability)
    unprocessed_rows = find_unprocessed_rows(rows, filepath)
    
    if not unprocessed_rows:
        print("All rows have already been processed!")
    else:
        # Get few-shot examples for unprocessed rows only
        unprocessed_few_shot_list = []
        if few_shot_corpus is not None:
            print(f"Selecting few-shot examples for {len(unprocessed_rows)} unprocessed rows...")
            # Create a corpus ID based on the corpus path for consistent caching
            corpus_id = f"corpus_{args.few_shot_corpus_path}_{args.vectorizer}" if args.few_shot_corpus_path else None
            
            for row in unprocessed_rows:
                examples = get_few_shot_examples(row.src_text, few_shot_corpus, args.num_few_shot, args.vectorizer, corpus_id)
                unprocessed_few_shot_list.append(examples)
        else:
            unprocessed_few_shot_list = [None for _ in unprocessed_rows]
        
        # Generate sample message for logging
        sample_message = unprocessed_rows[0].get_messages(args.prompt, unprocessed_few_shot_list[0])
        if args.debug:
            print(f"\n--- DEBUG: Sample message structure ---")
            for i, msg in enumerate(sample_message):
                print(f"Message {i+1} ({msg['role']}):")
                print(f"{msg['content']}")
                print("---")
        else:
            print(f"Sample message generated (use --debug to see full content)")
        
        # Initialize vLLM with single GPU and tensor parallel size 1 to avoid multiprocessing issues
        print(f"Initializing VLLM model: {args.model}")
        try:
            model = LLM(
                args.model, 
                max_model_len=6000,
                gpu_memory_utilization=0.95,
                disable_log_stats=True,      # Reduce memory overhead
                tensor_parallel_size=1,      # Explicitly set to 1 to avoid device issues
                enable_chunked_prefill=True, # More memory efficient
                enforce_eager=True,          # Avoid graph compilation issues
            )
            print("✅ vLLM model initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing vLLM model: {e}")
            print("\n🔧 Troubleshooting suggestions:")
            print("1. Check CUDA_VISIBLE_DEVICES environment variable")
            print("2. Verify GPU availability with: nvidia-smi")
            print("3. Try setting CUDA_VISIBLE_DEVICES=0 before running")
            print("4. Check if the model name/path is correct")
            print("5. Ensure sufficient GPU memory is available")
            
            # Try to provide more specific error information
            if "Device string must not be empty" in str(e):
                print("\n🎯 Specific fix for 'Device string must not be empty' error:")
                print("   - This usually indicates a CUDA device detection issue")
                print("   - Try: export CUDA_VISIBLE_DEVICES=0")
                print("   - Or run: CUDA_VISIBLE_DEVICES=0 python your_script.py")
                print("   - Check if CUDA is properly installed and accessible")
            
            raise e
        
        # Initialize tokenizer for token counting
        print(f"Initializing tokenizer for token counting: {args.model}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {args.model}: {e}")
            print("Token counting will be disabled.")
            tokenizer = None
        
        # Let vLLM handle batching internally for maximum efficiency
        print(f"Processing {len(unprocessed_rows)} unprocessed rows...")
        successful_count = 0
        skipped_count = 0
        
        try:
            # Determine batch processing strategy
            if args.batch_size and len(unprocessed_rows) > args.batch_size:
                print(f"Processing {len(unprocessed_rows)} rows in batches of {args.batch_size} to reduce memory usage...")
                
                # Process in batches
                total_token_counts = []
                all_translations = []
                
                for batch_start in range(0, len(unprocessed_rows), args.batch_size):
                    batch_end = min(batch_start + args.batch_size, len(unprocessed_rows))
                    batch_rows = unprocessed_rows[batch_start:batch_end]
                    batch_few_shot = unprocessed_few_shot_list[batch_start:batch_end]
                    
                    print(f"Processing batch {batch_start//args.batch_size + 1}/{(len(unprocessed_rows) + args.batch_size - 1)//args.batch_size} "
                          f"(rows {batch_start+1}-{batch_end})...")
                    
                    # Generate messages for this batch only
                    batch_messages = [row.get_messages(args.prompt, few_shot_examples) 
                                    for row, few_shot_examples in zip(batch_rows, batch_few_shot)]
                    
                    if args.debug and batch_start == 0:  # Only show debug for first batch
                        print(f"\n--- DEBUG: First batch with {len(batch_messages)} messages ---")
                        for i, msg_list in enumerate(batch_messages[:2]):  # Show first 2 messages
                            print(f"\nMessage {i+1}:")
                            for j, msg in enumerate(msg_list):
                                print(f"  Role: {msg['role']}")
                                print(f"  Content: {msg['content'][:200]}...")  # Truncate for readability
                        print("--- End debug messages ---\n")
                    
                    # Count tokens for this batch
                    if tokenizer is not None:
                        batch_token_counts = [count_tokens_in_messages(msg_list, tokenizer) for msg_list in batch_messages]
                        total_token_counts.extend(batch_token_counts)
                        batch_avg = sum(batch_token_counts) / len(batch_token_counts)
                        print(f"  Batch token average: {batch_avg:.1f}")
                    
                    # Generate translations for this batch with guided JSON decoding
                    guided_decoding_params = GuidedDecodingParams(json=POST_EDIT_JSON_SCHEMA)
                    batch_outputs = model.chat(batch_messages, sampling_params=SamplingParams(
                        max_tokens=2048,
                        temperature=0.0,
                        guided_decoding=guided_decoding_params,
                    ))
                    
                    batch_translations = [parse_json_response(out.outputs[0].text) for out in batch_outputs]
                    all_translations.extend(batch_translations)
                    
                    # Process and save this batch immediately (memory efficient)
                    for row, translation in zip(batch_rows, batch_translations):
                        if not translation or translation.strip() == "" or translation.startswith("["):
                            print(f"Skipping row: VLLM returned invalid content '{translation}' for source: {row.src_text[:50]}...")
                            skipped_count += 1
                            continue
                        
                        row.post_edited_tgt_txt = translation
                        append_row_to_csv(row, filepath)
                        successful_count += 1
                    
                    # Clear batch data to free memory
                    del batch_messages, batch_outputs, batch_translations
                
                # Print overall statistics
                if tokenizer is not None and total_token_counts:
                    avg_tokens = sum(total_token_counts) / len(total_token_counts)
                    max_tokens = max(total_token_counts)
                    min_tokens = min(total_token_counts)
                    print(f"\nOverall token statistics - Average: {avg_tokens:.1f}, Min: {min_tokens}, Max: {max_tokens}")
                
                print("\n" + "="*50)
                print("BATCH PROCESSING COMPLETE")
                print(f"Processed {len(all_translations)} translations in batches")
                print("="*50 + "\n")
                
            else:
                # Original all-at-once processing for smaller datasets
                print("Generating messages for all unprocessed rows...")
                messages = [row.get_messages(args.prompt, few_shot_examples) 
                           for row, few_shot_examples in zip(unprocessed_rows, unprocessed_few_shot_list)]
                
                if args.debug:
                    print(f"\n--- DEBUG: Generated {len(messages)} messages for batch processing ---")
                    for i, msg_list in enumerate(messages[:3]):  # Show first 3 messages
                        print(f"\nMessage set {i+1}:")
                        for j, msg in enumerate(msg_list):
                            print(f"  Role: {msg['role']}")
                            print(f"  Content: {msg['content']}")
                    if len(messages) > 3:
                        print(f"  ... and {len(messages) - 3} more message sets")
                    print("--- End debug messages ---\n")
                
                # Count and print token lengths for all messages
                if tokenizer is not None:
                    print("Counting tokens for all messages...")
                    token_counts = [count_tokens_in_messages(msg_list, tokenizer) for msg_list in messages]
                    avg_tokens = sum(token_counts) / len(token_counts)
                    max_tokens = max(token_counts)
                    min_tokens = min(token_counts)
                    print(f"Token statistics - Average: {avg_tokens:.1f}, Min: {min_tokens}, Max: {max_tokens}")
                    if args.debug:
                        print("First 10 message token counts:", token_counts[:10])
                
                # Generate all translations at once using vLLM's internal batching with guided JSON decoding
                print("Generating translations using vLLM...")
                guided_decoding_params = GuidedDecodingParams(json=POST_EDIT_JSON_SCHEMA)
                outputs = model.chat(messages, sampling_params=SamplingParams(
                    max_tokens=512,
                    temperature=0.0,
                    guided_decoding=guided_decoding_params,
                ))
                print("\n" + "="*50)
                print("POST EDITING RESULT")
                
                translations = [parse_json_response(out.outputs[0].text) for out in outputs]
                
                # Print the actual text translations instead of raw output objects
                print("Generated translations:")
                for i, translation in enumerate(translations[:5]):  # Show first 5 translations
                    print(f"  Translation {i+1}: {translation}")
                if len(translations) > 5:
                    print(f"  ... and {len(translations) - 5} more translations")
                print("="*50 + "\n")
                
                # Process each translation and save incrementally
                print("Processing and saving results...")
                for row, translation in tqdm(zip(unprocessed_rows, translations), desc="Saving results"):
                    # Skip rows where VLLM returned empty or invalid response
                    if not translation or translation.strip() == "" or translation.startswith("["):
                        print(f"Skipping row: VLLM returned invalid content '{translation}' for source: {row.src_text[:50]}...")
                        skipped_count += 1
                        continue
                    
                    # Assign prediction and save immediately
                    row.post_edited_tgt_txt = translation
                    append_row_to_csv(row, filepath)
                    successful_count += 1
                
        except Exception as e:
            print(f"Error during processing: {e}")
            print("Attempting to process remaining rows individually...")
            # Fallback: process individually if batch processing fails
            for i, (row, few_shot_examples) in enumerate(zip(unprocessed_rows, unprocessed_few_shot_list)):
                try:
                    messages = [row.get_messages(args.prompt, few_shot_examples)]
                    
                    # Count tokens for this individual message
                    if tokenizer is not None:
                        token_count = count_tokens_in_messages(messages[0], tokenizer)
                        print(f"Row {i+1} token count: {token_count}")
                    
                    if args.debug:
                        print(f"\n--- DEBUG: Individual processing row {i+1} ---")
                        for j, msg in enumerate(messages[0]):
                            print(f"Message {j+1} ({msg['role']}):")
                            print(f"{msg['content']}")
                        print("--- End debug message ---\n")
                    
                    guided_decoding_params = GuidedDecodingParams(json=POST_EDIT_JSON_SCHEMA)
                    outputs = model.chat(messages, sampling_params=SamplingParams(
                        max_tokens=512,
                        temperature=0.0,
                        guided_decoding=guided_decoding_params,
                    ))
                    translation = parse_json_response(outputs[0].outputs[0].text)
                    
                    if translation and translation.strip() and not translation.startswith("["):
                        row.post_edited_tgt_txt = translation
                        append_row_to_csv(row, filepath)
                        successful_count += 1
                    else:
                        print(f"Skipping individual row: Invalid translation '{translation}'")
                        skipped_count += 1
                        
                except Exception as row_error:
                    print(f"Error processing row {i}: {row_error}")
                    skipped_count += 1
        
        print(f"Processing complete: {successful_count} successful, {skipped_count} skipped")
    
    # Load all completed results for metrics calculation (only non-empty predictions)
    if os.path.exists(filepath):
        completed_rows = Row.load_from_savepath(filepath)
        # Filter out rows with empty predictions for metrics calculation
        valid_rows = [row for row in completed_rows if row.post_edited_tgt_txt and row.post_edited_tgt_txt.strip()]
        print(f"Calculating metrics for {len(valid_rows)} valid completed rows")
        metrics = Row.calculate_metrics(valid_rows)
    else:
        print("No completed rows found. Cannot calculate metrics.")
        metrics = {}
    
    # Save metrics to JSON file
    metrics_filepath = filepath.replace('.csv', '_metrics.json')
    with open(metrics_filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_filepath}")


if __name__ == "__main__":
    import atexit
    import torch.distributed as dist
    
    def cleanup_resources():
        """Clean up distributed resources to avoid warnings"""
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass  # Ignore cleanup errors
    
    # Register cleanup function to run at exit
    atexit.register(cleanup_resources)
    
    try:
        main()
    finally:
        cleanup_resources()
