# Set multiprocessing start method to 'spawn' BEFORE any other imports to avoid CUDA initialization issues
import multiprocessing
import os
from pathlib import Path

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
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm

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


def count_tokens_in_messages(messages: List[dict], tokenizer=None) -> int:
    """Count total tokens in a list of chat messages"""
    if tokenizer is not None:
        # Use proper tokenizer if available
        total_tokens = 0
        for message in messages:
            # Tokenize the content of each message
            tokens = tokenizer.encode(message['content'], add_special_tokens=False)
            total_tokens += len(tokens)
        
        # Add tokens for special chat formatting (approximate)
        # Most chat models add extra tokens for role indicators, special tokens, etc.
        formatting_tokens = len(messages) * 3  # Rough estimate for role tokens and separators
        return total_tokens + formatting_tokens
    else:
        # Fallback to word-based approximation for API version
        total_words = 0
        for message in messages:
            # Simple word count approximation
            words = len(message['content'].split())
            total_words += words
        
        # Rough approximation: 1.3 tokens per word + formatting tokens
        formatting_tokens = len(messages) * 3  # Rough estimate for role tokens and separators
        return int(total_words * 1.3) + formatting_tokens


class GPT:
    def __init__(self, model: str = "gpt-5"):
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, messages: List[dict]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=512,  # Controls max output tokens - increase if needed
        )
        content = completion.choices[0].message.content
        if content is None:
            print(f"Warning: API returned None content for model {self.model}")
            return ""  # Return empty string instead of crashing
        return content.strip()


class Gemini:
    """
    Gemini API client with official Batch API support.
    
    Features:
    - Uses official Google GenAI client library instead of OpenAI-compatible endpoint
    - Supports Batch API with 50% cost reduction compared to standard API
    - Automatic selection between inline batch (<15MB) and file-based batch (up to 2GB)
    - Fallback to OpenAI-compatible endpoint if google.genai not available
    - Asynchronous batch processing with configurable timeout
    
    Reference: https://ai.google.dev/gemini-api/docs/batch-api
    """
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        # Import Google GenAI client
        try:
            from google import genai
            self.genai = genai
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            self.use_batch_api = True
        except ImportError:
            print("Warning: google.genai not installed. Falling back to OpenAI-compatible endpoint.")
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self.use_batch_api = False

    def generate(self, messages: List[dict]) -> str:
        """Generate single response - fallback for individual requests"""
        if self.use_batch_api:
            # Convert messages to Gemini format
            gemini_request = self._convert_messages_to_gemini_request(messages)
            
            # For single requests, use inline batch processing
            inline_batch_job = self.client.batches.create(
                model=f"models/{self.model}",
                src=[gemini_request],
                config={
                    'display_name': f"single-request-{hash(str(messages))}",
                },
            )
            
            # Wait for completion and get result
            completed_job = self._wait_for_batch_completion(inline_batch_job.name)
            
            if completed_job and completed_job.dest and completed_job.dest.inlined_responses:
                response = completed_job.dest.inlined_responses[0]
                if response.response and hasattr(response.response, 'candidates'):
                    return response.response.candidates[0].content.parts[0].text.strip()
                elif response.error:
                    print(f"Gemini API error: {response.error}")
                    return ""
            return ""
        else:
            # Fallback to OpenAI-compatible endpoint
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            content = completion.choices[0].message.content
            if content is None:
                print(f"Warning: API returned None content for model {self.model}")
                return ""
            return content.strip()

    def generate_batch(self, messages_list: List[List[dict]], batch_name: str = None, timeout: int = 7200) -> List[str]:
        """Generate responses for a batch of messages using Gemini Batch API"""
        if not self.use_batch_api:
            # Fallback to individual requests
            return [self.generate(messages) for messages in messages_list]
        
        if not batch_name:
            batch_name = f"batch-{hash(str(messages_list))}"
        
        # Convert all messages to Gemini format
        gemini_requests = [self._convert_messages_to_gemini_request(messages) for messages in messages_list]
        
        # Determine batch method based on size
        total_size_estimate = sum(len(str(req)) for req in gemini_requests)
        
        if total_size_estimate < 15 * 1024 * 1024:  # Less than 15MB, use inline
            print(f"Using inline batch processing for {len(gemini_requests)} requests")
            return self._process_inline_batch(gemini_requests, batch_name, timeout)
        else:
            print(f"Using file-based batch processing for {len(gemini_requests)} requests")
            return self._process_file_batch(gemini_requests, batch_name, timeout)

    def _convert_messages_to_gemini_request(self, messages: List[dict]) -> dict:
        """Convert OpenAI-style messages to Gemini GenerateContentRequest format"""
        contents = []
        system_instruction = None
        
        for message in messages:
            if message['role'] == 'system':
                system_instruction = message['content']
            elif message['role'] == 'user':
                contents.append({
                    'parts': [{'text': message['content']}],
                    'role': 'user'
                })
            elif message['role'] == 'assistant':
                contents.append({
                    'parts': [{'text': message['content']}],
                    'role': 'model'
                })
        
        request = {
            'contents': contents,
            'generation_config': {
                'temperature': 0.0,
                'max_output_tokens': 512,
            }
        }
        
        if system_instruction:
            request['system_instruction'] = {
                'parts': [{'text': system_instruction}]
            }
        
        return request

    def _process_inline_batch(self, gemini_requests: List[dict], batch_name: str, timeout: int) -> List[str]:
        """Process batch using inline requests"""
        inline_batch_job = self.client.batches.create(
            model=f"models/{self.model}",
            src=gemini_requests,
            config={
                'display_name': batch_name,
            },
        )
        
        print(f"Created inline batch job: {inline_batch_job.name}")
        
        # Wait for completion
        completed_job = self._wait_for_batch_completion(inline_batch_job.name, timeout)
        
        if not completed_job:
            return [""] * len(gemini_requests)
        
        # Extract results
        results = []
        if completed_job.dest and completed_job.dest.inlined_responses:
            for response in completed_job.dest.inlined_responses:
                if response.response and hasattr(response.response, 'candidates'):
                    text = response.response.candidates[0].content.parts[0].text.strip()
                    results.append(text)
                elif response.error:
                    print(f"Error in batch response: {response.error}")
                    results.append("")
                else:
                    results.append("")
        
        return results

    def _process_file_batch(self, gemini_requests: List[dict], batch_name: str, timeout: int) -> List[str]:
        """Process batch using file upload method"""
        import json
        import tempfile
        
        # Create JSONL file
        jsonl_content = []
        for i, request in enumerate(gemini_requests):
            jsonl_content.append(json.dumps({
                "key": f"request-{i}",
                "request": request
            }))
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('\n'.join(jsonl_content))
            temp_file_path = f.name
        
        try:
            # Upload file
            uploaded_file = self.client.files.upload(
                file=temp_file_path,
                config=self.genai.types.UploadFileConfig(
                    display_name=f'{batch_name}-input',
                    mime_type='application/jsonl'
                )
            )
            
            print(f"Uploaded batch file: {uploaded_file.name}")
            
            # Create batch job
            batch_job = self.client.batches.create(
                model=f"models/{self.model}",
                src=uploaded_file,
                config={
                    'display_name': batch_name,
                }
            )
            
            print(f"Created file-based batch job: {batch_job.name}")
            
            # Wait for completion
            completed_job = self._wait_for_batch_completion(batch_job.name, timeout)
            
            if not completed_job:
                return [""] * len(gemini_requests)
            
            # Download and parse results
            if completed_job.dest and completed_job.dest.responses_file:
                response_file = self.client.files.get(completed_job.dest.responses_file)
                file_content = self.client.files.download(response_file.name)
                
                # Parse JSONL response
                results = [""] * len(gemini_requests)
                for line in file_content.strip().split('\n'):
                    if line.strip():
                        response_data = json.loads(line)
                        if 'key' in response_data:
                            request_idx = int(response_data['key'].split('-')[1])
                            if 'response' in response_data and 'candidates' in response_data['response']:
                                text = response_data['response']['candidates'][0]['content']['parts'][0]['text']
                                results[request_idx] = text.strip()
                            elif 'error' in response_data:
                                print(f"Error in request {request_idx}: {response_data['error']}")
                
                return results
            
        finally:
            # Clean up temporary file
            import os
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        return [""] * len(gemini_requests)

    def _wait_for_batch_completion(self, batch_name: str, max_wait_time: int = 7200):
        """Wait for batch job to complete with timeout (default 2 hours)"""
        import time
        
        start_time = time.time()
        print(f"Waiting for batch job {batch_name} to complete...")
        
        while time.time() - start_time < max_wait_time:
            try:
                batch_job = self.client.batches.get(batch_name)
                
                if hasattr(batch_job, 'done') and batch_job.done:
                    if hasattr(batch_job, 'metadata') and batch_job.metadata.state == 'JOB_STATE_SUCCEEDED':
                        print(f"Batch job completed successfully!")
                        return batch_job
                    else:
                        print(f"Batch job failed with state: {batch_job.metadata.state}")
                        if hasattr(batch_job, 'error'):
                            print(f"Error: {batch_job.error}")
                        return None
                
                # Wait before checking again
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Error checking batch status: {e}")
                time.sleep(30)
        
        print(f"Batch job timed out after {max_wait_time} seconds")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Post-edit machine translation using an LLM API with comprehensive evaluation metrics")
    parser.add_argument("--model", type=str, required=True, choices=["gemini", "gpt"], help="Model type: gemini or gpt")
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
    parser.add_argument("--vectorizer", type=str, default="bm25", choices=["bm25", "tfidf", "sbert", "chrf_rag"], help="Similarity method for few-shot selection")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker threads for API calls")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process from CSV (optional, processes all if not specified)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to print created messages during inference")
    parser.add_argument("--batch_size", type=int, default=None, help="Process messages in batches of this size to reduce memory usage (optional, processes all at once if not specified)")
    parser.add_argument("--batch_timeout", type=int, default=7200, help="Maximum time to wait for Gemini batch jobs to complete in seconds (default: 7200 = 2 hours)")
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


def generate_translation(llm, messages: List[dict]) -> str:
    """Generate translation using the LLM API"""
    return llm.generate(messages)


def find_unprocessed_rows(all_rows: List[Row], output_filepath: str) -> List[Row]:
    """Find rows that haven't been processed yet by comparing with existing CSV file"""
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

    # Initialize the appropriate model
    if args.model == "gemini":
        llm = Gemini(model="gemini-2.5-flash")  # Using Gemini 2.5 Flash with Batch API support
    elif args.model == "gpt":
        llm = GPT(model="gpt-5")  # Using GPT-4o as the current available model
    else:
        raise ValueError(f"Unsupported model: {args.model}")

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
        
        # Initialize tokenizer for token counting if possible
        tokenizer = None
        try:
            # Try to import and initialize transformers tokenizer
            from transformers import AutoTokenizer
            # Use a generic tokenizer for rough estimation
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            print("Initialized tokenizer for token counting")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            print("Token counting will use word-based approximation.")
        
        # Process rows with progress saving
        print(f"Processing {len(unprocessed_rows)} unprocessed rows...")
        successful_count = 0
        skipped_count = 0
        
        try:
            # Determine batch processing strategy
            if args.batch_size and len(unprocessed_rows) > args.batch_size:
                print(f"Processing {len(unprocessed_rows)} rows in batches of {args.batch_size} to manage API rate limits...")
                
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
                    
                    # Generate translations for this batch
                    if args.model == "gemini" and hasattr(llm, 'generate_batch') and llm.use_batch_api:
                        # Use Gemini Batch API for this batch
                        batch_name = f"batch-{batch_start//args.batch_size + 1}-{args.src}-{args.tgt}"
                        batch_translations = llm.generate_batch(batch_messages, batch_name, args.batch_timeout)
                    else:
                        # Use threading for parallel API calls
                        def process_single_message(msg_list):
                            try:
                                return generate_translation(llm, msg_list)
                            except Exception as e:
                                print(f"Error in API call: {e}")
                                return ""
                        
                        batch_translations = thread_map(process_single_message, batch_messages, 
                                                       max_workers=args.num_workers, 
                                                       desc=f"Batch {batch_start//args.batch_size + 1}")
                    all_translations.extend(batch_translations)
                    
                    # Process and save this batch immediately
                    for row, translation in zip(batch_rows, batch_translations):
                        if not translation or translation.strip() == "":
                            print(f"Skipping row: API returned empty content for source: {row.src_text[:50]}...")
                            skipped_count += 1
                            continue
                        
                        row.post_edited_tgt_txt = translation
                        append_row_to_csv(row, filepath)
                        successful_count += 1
                    
                    # Clear batch data to free memory
                    del batch_messages, batch_translations
                
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
                # Original all-at-once processing for smaller datasets using threading
                print("Generating messages for all unprocessed rows...")
                messages = [row.get_messages(args.prompt, few_shot_examples) 
                           for row, few_shot_examples in zip(unprocessed_rows, unprocessed_few_shot_list)]
                
                if args.debug:
                    print(f"\n--- DEBUG: Generated {len(messages)} messages for processing ---")
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
                
                # Generate all translations - use batch API for Gemini, threading for others
                print("Generating translations using API calls...")
                
                if args.model == "gemini" and hasattr(llm, 'generate_batch') and llm.use_batch_api:
                    # Use Gemini Batch API for significant cost savings (50% reduction)
                    print("Using Gemini Batch API for cost-effective processing...")
                    batch_name = f"post-editing-{args.src}-{args.tgt}-{len(messages)}"
                    translations = llm.generate_batch(messages, batch_name, args.batch_timeout)
                else:
                    # Use threading for parallel API calls (GPT or fallback)
                    def process_single_message(msg_list):
                        try:
                            return generate_translation(llm, msg_list)
                        except Exception as e:
                            print(f"Error in API call: {e}")
                            return ""
                    
                    translations = thread_map(process_single_message, messages, 
                                            max_workers=args.num_workers, 
                                            desc="Generating translations")
                
                print("\n" + "="*50)
                print("POST EDITING RESULT")
                
                # Print the actual text translations
                print("Generated translations:")
                for i, translation in enumerate(translations[:5]):  # Show first 5 translations
                    print(f"  Translation {i+1}: {translation}")
                if len(translations) > 5:
                    print(f"  ... and {len(translations) - 5} more translations")
                print("="*50 + "\n")
                
                # Process each translation and save incrementally
                print("Processing and saving results...")
                for row, translation in tqdm(zip(unprocessed_rows, translations), desc="Saving results"):
                    # Skip rows where API returned empty string
                    if not translation or translation.strip() == "":
                        print(f"Skipping row: API returned empty content for source: {row.src_text[:50]}...")
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
                    messages = row.get_messages(args.prompt, few_shot_examples)
                    
                    # Count tokens for this individual message
                    if tokenizer is not None:
                        token_count = count_tokens_in_messages(messages, tokenizer)
                        print(f"Row {i+1} token count: {token_count}")
                    
                    if args.debug:
                        print(f"\n--- DEBUG: Individual processing row {i+1} ---")
                        for j, msg in enumerate(messages):
                            print(f"Message {j+1} ({msg['role']}):")
                            print(f"{msg['content']}")
                        print("--- End debug message ---\n")
                    
                    translation = generate_translation(llm, messages)
                    
                    if translation and translation.strip():
                        row.post_edited_tgt_txt = translation
                        append_row_to_csv(row, filepath)
                        successful_count += 1
                    else:
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
    main()
