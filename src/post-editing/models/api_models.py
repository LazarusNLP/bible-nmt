"""
API-based model implementations (GPT, Gemini).
"""

import os
import asyncio
import time
from typing import List
from pydantic import BaseModel
from .base import BaseLLM
from core.constants import DEFAULT_MODEL_CONFIGS


class PostEditedOutput(BaseModel):
    """Pydantic model for structured post-editing output."""
    post_edited_text: str


class TranslatedOutput(BaseModel):
    """Pydantic model for structured translation output."""
    translated_text: str


class GPTModel(BaseLLM):
    """OpenAI GPT model interface."""
    
    def __init__(self, model: str = "gpt-4o"):
        """Initialize GPT model."""
        self._model_name = model
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, messages: List[dict], translation_mode: bool = False) -> str:
        """Generate single response using OpenAI API."""
        gpt_defaults = DEFAULT_MODEL_CONFIGS.get('gpt', {})
        completion = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=gpt_defaults.get('temperature', 0.0),
            max_tokens=gpt_defaults.get('max_tokens', 4096),
        )
        content = completion.choices[0].message.content
        if content is None:
            print(f"Warning: API returned None content for model {self._model_name}")
            return ""  # Return empty string instead of crashing
        return content.strip()
    
    def generate_batch(self, messages_list: List[List[dict]], sequential=False) -> List[str]:
        """Generate responses for a batch of message lists (GPT doesn't have native batch processing)."""
        # GPT doesn't have native batch processing, so this always processes sequentially
        if sequential:
            print(f"Generating {len(messages_list)} responses using GPT SEQUENTIAL processing...")
        else:
            print(f"Generating {len(messages_list)} responses using GPT (no native batch support - will process sequentially)...")
        
        results = []
        for i, messages in enumerate(messages_list):
            try:
                result = self.generate(messages)
                results.append(result)
                if sequential:
                    print(f"  Processed GPT request {i+1}/{len(messages_list)}")
            except Exception as e:
                print(f"  Error in GPT request {i+1}: {e}")
                results.append("")
        return results
    
    @property
    def supports_batch(self) -> bool:
        """GPT supports sequential batch processing."""
        return True


class GeminiModel(BaseLLM):
    """
    Gemini API client using official Google GenAI library with async batch processing support.
    
    This version uses the official Google GenAI client to access
    all Gemini models including the latest 2.5 series, with async support for parallel requests.
    """
    
    def __init__(self, model: str = "gemini-2.5-flash", batch_size: int = None, delay_between_batches: float = 0.0):
        """Initialize Gemini model."""
        self._model_name = model
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self._cleanup_performed = False
        try:
            from google import genai
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            print(f"✓ Using official Gemini API with model: {model} (async batch processing enabled)")
        except ImportError:
            print("Error: google-genai package not found. Installing...")
            import subprocess
            subprocess.run(["pip", "install", "google-genai"], check=True)
            from google import genai
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            print(f"✓ Installed google-genai and initialized with model: {model} (async batch processing enabled)")
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        if not self._cleanup_performed:
            # Try to clean up synchronously
            try:
                import gc
                gc.collect()
                self._cleanup_performed = True
            except Exception:
                pass  # Ignore cleanup errors in destructor

    def _combine_messages(self, messages: List[dict]) -> str:
        """Combine system and user messages into a single prompt."""
        combined_prompt = ""
        
        for message in messages:
            if message['role'] == 'system':
                combined_prompt += f"SYSTEM INSTRUCTION: {message['content']}\n\n"
            elif message['role'] == 'user':
                combined_prompt += f"USER: {message['content']}"
            elif message['role'] == 'assistant':
                combined_prompt += f"ASSISTANT: {message['content']}\n\n"
        
        return combined_prompt

    def generate(self, messages: List[dict], translation_mode: bool = False) -> str:
        """Generate single response using official Gemini API with structured output."""
        try:
            gemini_defaults = DEFAULT_MODEL_CONFIGS.get('gemini', {})
            combined_prompt = self._combine_messages(messages)
            
            # Choose appropriate schema based on mode
            response_schema = TranslatedOutput if translation_mode else PostEditedOutput
            
            # Use structured output with response_schema and increased generation config
            response = self.client.models.generate_content(  
                model=f"models/{self._model_name}",
                contents=[{
                    'parts': [{'text': combined_prompt}],
                    'role': 'user'
                }],
                config={
                    "response_mime_type": "application/json", 
                    "response_schema": response_schema,
                    "max_output_tokens": gemini_defaults.get('max_tokens', 8192),
                    "temperature": gemini_defaults.get('temperature', 0.0),
                    "seed": 42,  # Fixed seed for reproducibility
                }
            )
            
            return self._parse_response(response, translation_mode)
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return ""

    async def _generate_async(self, messages: List[dict]) -> str:
        """Generate single response asynchronously using the new SDK."""
        try:
            gemini_defaults = DEFAULT_MODEL_CONFIGS.get('gemini', {})
            combined_prompt = self._combine_messages(messages)
            
            # Use the async client from the new SDK
            response = await self.client.aio.models.generate_content(
                model=f"models/{self._model_name}",
                contents=[{
                    'parts': [{'text': combined_prompt}],
                    'role': 'user'
                }],
                config={
                    "response_mime_type": "application/json", 
                    "response_schema": PostEditedOutput,
                    "max_output_tokens": gemini_defaults.get('max_tokens', 8192),
                    "temperature": gemini_defaults.get('temperature', 0.0),
                    "seed": 42,  # Fixed seed for reproducibility
                }
            )
            
            return self._parse_response(response)
            
        except Exception as e:
            print(f"Error in async Gemini API call: {e}")
            return ""

    def _parse_response(self, response, translation_mode: bool = False) -> str:
        """Parse response from Gemini API with robust error handling."""
        if response and hasattr(response, 'parsed') and response.parsed:
            # Use the structured parsed response - this is the preferred path
            if translation_mode:
                return response.parsed.translated_text.strip()
            else:
                return response.parsed.post_edited_text.strip()
        elif response and hasattr(response, 'candidates') and response.candidates:
            # Fallback to manual JSON parsing if parsed response not available
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                if candidate.content.parts:
                    raw_text = candidate.content.parts[0].text.strip()
                    
                    # Handle cases where LLM returns reasoning text + JSON
                    field_name = 'translated_text' if translation_mode else 'post_edited_text'
                    json_pattern = f'{{"{field_name}"'
                    
                    if json_pattern in raw_text:
                        # Extract JSON from mixed content
                        json_start = raw_text.find(json_pattern)
                        if json_start != -1:
                            json_part = raw_text[json_start:]
                            # Find the end of the JSON object
                            brace_count = 0
                            json_end = -1
                            for i, char in enumerate(json_part):
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        json_end = i + 1
                                        break
                            
                            if json_end != -1:
                                json_text = json_part[:json_end]
                                try:
                                    import json
                                    json_response = json.loads(json_text)
                                    result = json_response.get(field_name, '').strip()
                                    if result:
                                        print(f"⚠️ Warning: Extracted JSON from mixed content (reasoning + JSON)")
                                        return result
                                except json.JSONDecodeError as e:
                                    print(f"Warning: Failed to parse extracted JSON: {e}")
                    
                    # Try to parse the entire response as JSON
                    try:
                        import json
                        json_response = json.loads(raw_text)
                        return json_response.get(field_name, '').strip()
                    except json.JSONDecodeError:
                        # If it's not JSON, check if it's just plain text (direct translation)
                        if len(raw_text) < 1000 and not raw_text.startswith(('I need', 'The user', 'Let me', 'Here is')):
                            print(f"⚠️ Warning: Got plain text instead of JSON, using as-is: '{raw_text[:50]}...'")
                            return raw_text
                        else:
                            print(f"Warning: Failed to parse response as JSON and content looks like reasoning text")
                            print(f"Raw response: {raw_text[:200]}...")
                            return ""
        
        print(f"Warning: No content returned from {self._model_name}")
        return ""

    def generate_batch(self, messages_list: List[List[dict]], rows_data=None, output_filepath=None, sequential=False, translation_mode=False) -> List[str]:
        """Generate responses for a batch of message lists using Gemini's native Batch API."""
        if sequential:
            # Force sequential processing
            print(f"Generating {len(messages_list)} responses using Gemini SEQUENTIAL processing...")
            results = []
            for i, messages in enumerate(messages_list):
                try:
                    result = self.generate(messages, translation_mode)
                    results.append(result)
                    print(f"  Processed Gemini request {i+1}/{len(messages_list)}")
                except Exception as e:
                    print(f"  Error in Gemini request {i+1}: {e}")
                    results.append("")
            return results
        
        print(f"Generating {len(messages_list)} responses using Gemini native Batch API...")
        
        # Use simple concurrent processing with threading for immediate results
        return self._generate_batch_threaded(messages_list, rows_data, output_filepath, translation_mode)

    def _generate_batch_native(self, messages_list: List[List[dict]], rows_data=None, output_filepath=None) -> List[str]:
        """Generate responses using Gemini's native Batch API."""
        import json
        import tempfile
        import os
        import uuid
        
        # Step 1: Create JSONL file with batch requests
        batch_requests = []
        gemini_defaults = DEFAULT_MODEL_CONFIGS.get('gemini', {})
        
        for i, messages in enumerate(messages_list):
            combined_prompt = self._combine_messages(messages)
            
            # Create a batch request following Gemini's format
            request = {
                "custom_id": f"request_{i}",
                "method": "POST", 
                "url": "/v1beta/models/gemini-2.5-flash:generateContent",
                "body": {
                    "contents": [{
                        "parts": [{"text": combined_prompt}],
                        "role": "user"
                    }],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": {
                            "type": "object",
                            "properties": {
                                "post_edited_text": {"type": "string"}
                            },
                            "required": ["post_edited_text"]
                        },
                        "maxOutputTokens": gemini_defaults.get('max_tokens', 8192),
                        "temperature": gemini_defaults.get('temperature', 0.0),
                        "seed": 42
                    }
                }
            }
            batch_requests.append(request)
        
        # Step 2: Write requests to JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
            jsonl_path = f.name
        
        try:
            print(f"Created batch file with {len(batch_requests)} requests: {jsonl_path}")
            
            # Step 3: Upload file and create batch job
            print("Uploading batch file...")
            
            # Upload the file
            uploaded_file = self.client.files.upload(path=jsonl_path)
            print(f"File uploaded with URI: {uploaded_file.uri}")
            
            # Step 4: Create batch job
            print("Creating batch job...")
            batch_job = self.client.batches.create(
                input_file_id=uploaded_file.name,
                endpoint="/v1beta/models/gemini-2.5-flash:generateContent",
                completion_window="24h"
            )
            
            print(f"Batch job created: {batch_job.id}")
            print("Job will complete within 24 hours. For immediate results, use sequential=True")
            
            # Step 5: For now, return placeholder results and inform user
            # In a real implementation, you'd poll for completion and retrieve results
            results = ["BATCH_JOB_PENDING"] * len(messages_list)
            
            # Save batch job info for later retrieval
            batch_info = {
                "batch_id": batch_job.id,
                "status": "pending",
                "total_requests": len(messages_list),
                "created_at": time.time()
            }
            
            if output_filepath:
                batch_info_path = output_filepath.replace('.csv', '_batch_info.json')
                with open(batch_info_path, 'w') as f:
                    json.dump(batch_info, f, indent=2)
                print(f"Batch info saved to: {batch_info_path}")
            
            return results
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(jsonl_path)
            except:
                pass
    
    def retrieve_batch_results(self, batch_id: str) -> List[str]:
        """Retrieve results from a completed batch job."""
        try:
            batch_job = self.client.batches.retrieve(batch_id)
            
            if batch_job.status != "completed":
                print(f"Batch job {batch_id} status: {batch_job.status}")
                return []
            
            # Download and parse results
            if batch_job.output_file_id:
                output_file = self.client.files.content(batch_job.output_file_id)
                results = []
                
                for line in output_file.text.strip().split('\n'):
                    if line:
                        result_data = json.loads(line)
                        if result_data.get('response', {}).get('body', {}).get('candidates'):
                            candidate = result_data['response']['body']['candidates'][0]
                            if candidate.get('content', {}).get('parts'):
                                try:
                                    content = json.loads(candidate['content']['parts'][0]['text'])
                                    results.append(content.get('post_edited_text', ''))
                                except:
                                    results.append('')
                            else:
                                results.append('')
                        else:
                            results.append('')
                
                print(f"Retrieved {len(results)} results from batch job {batch_id}")
                return results
            else:
                print(f"No output file for batch job {batch_id}")
                return []
                
        except Exception as e:
            print(f"Error retrieving batch results: {e}")
            return []
    
    def _generate_batch_threaded(self, messages_list: List[List[dict]], rows_data=None, output_filepath=None, translation_mode=False) -> List[str]:
        """Process requests in batches with incremental saving after each batch."""
        import concurrent.futures
        
        # Use batch_size for both batch splitting and max workers
        batch_size = self.batch_size if self.batch_size else 5
        max_workers = min(batch_size, 10)  # Don't create too many threads
        
        print(f"Processing {len(messages_list)} requests in batches of {batch_size} with {max_workers} concurrent threads...")
        print(f"Incremental saving enabled - results saved after each batch")
        
        def process_single(args):
            messages, index = args
            try:
                result = self.generate(messages, translation_mode)
                return result
            except Exception as e:
                print(f"  Error in request {index + 1}: {e}")
                return ""
        
        # Process in batches
        all_results = []
        total_success = 0
        
        for batch_start in range(0, len(messages_list), batch_size):
            batch_end = min(batch_start + batch_size, len(messages_list))
            batch_messages = messages_list[batch_start:batch_end]
            batch_rows = rows_data[batch_start:batch_end] if rows_data else None
            
            batch_num = (batch_start // batch_size) + 1
            total_batches = (len(messages_list) + batch_size - 1) // batch_size
            
            print(f"\nProcessing batch {batch_num}/{total_batches} (samples {batch_start+1}-{batch_end})...")
            
            # Create arguments for this batch
            request_args = [(messages, batch_start + i) for i, messages in enumerate(batch_messages)]
            
            # Process this batch concurrently
            batch_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks for this batch
                future_to_index = {executor.submit(process_single, args): i for i, args in enumerate(request_args)}
                
                # Collect results in order for this batch
                batch_results = [None] * len(batch_messages)
                batch_success = 0
                
                for future in concurrent.futures.as_completed(future_to_index):
                    local_index = future_to_index[future]
                    try:
                        result = future.result()
                        batch_results[local_index] = result if result else ""
                        if result and not result.startswith("["):
                            batch_success += 1
                    except Exception as e:
                        print(f"Exception in batch request {local_index + 1}: {e}")
                        batch_results[local_index] = ""
            
            # Update rows_data for this batch and calculate metrics
            # Always process all rows - use fallback for failed translations
            all_batch_rows = []
            batch_success_count = 0
            batch_fallback_count = 0
            
            if batch_rows:
                for i, result in enumerate(batch_results):
                    if result and not result.startswith("["):
                        # Valid translation - use it
                        batch_rows[i].post_edited_tgt_txt = result
                        batch_success_count += 1
                        
                        # Calculate metrics
                        try:
                            from core.metrics import MetricsCalculator
                            individual_metrics = MetricsCalculator.calculate_individual_metrics(
                                original_text=batch_rows[i].pred_tgt_text,
                                post_edited_text=result,
                                reference_text=batch_rows[i].tgt_text
                            )
                            batch_rows[i].spbleu_improvement = individual_metrics["improvements"]["spbleu"]
                            batch_rows[i].chrf3_improvement = individual_metrics["improvements"]["chrf3"]
                            batch_rows[i].chrfpp_improvement = individual_metrics["improvements"]["chrfpp"]
                        except Exception as e:
                            print(f"Warning: Could not calculate metrics for row {batch_start + i + 1}: {e}")
                            batch_rows[i].spbleu_improvement = None
                            batch_rows[i].chrf3_improvement = None
                            batch_rows[i].chrfpp_improvement = None
                    else:
                        # Failed/invalid translation - use original pred_text as fallback
                        batch_rows[i].post_edited_tgt_txt = batch_rows[i].pred_tgt_text
                        batch_fallback_count += 1
                        print(f"  ⚠️ Row {batch_start + i + 1}: Using original pred_text as fallback (API returned: '{result}')")
                        
                        # For fallback cases, metrics improvements should be 0 (no improvement)
                        batch_rows[i].spbleu_improvement = 0.0
                        batch_rows[i].chrf3_improvement = 0.0
                        batch_rows[i].chrfpp_improvement = 0.0
                    
                    # Always add the row (either successful or fallback)
                    all_batch_rows.append(batch_rows[i])
            
            # Save this batch to CSV immediately (all rows, not just successful ones)
            if all_batch_rows and output_filepath:
                try:
                    from core.data_io import DataHandler
                    DataHandler.batch_append_to_csv(all_batch_rows, output_filepath)
                    print(f"  ✅ Batch {batch_num}: {len(all_batch_rows)} rows saved to CSV ({batch_success_count} successful, {batch_fallback_count} fallback)")
                except Exception as e:
                    print(f"  ⚠️ Warning: Could not save batch {batch_num} to CSV: {e}")
            
            # Add to overall results
            all_results.extend(batch_results)
            total_success += batch_success_count
            
            print(f"  Batch {batch_num} complete: {batch_success_count}/{len(batch_results)} successful, {batch_fallback_count} fallback")
        
        print(f"\nAll batches complete: {total_success}/{len(all_results)} total successful")
        if output_filepath:
            print(f"Results incrementally saved to: {output_filepath}")
        
        return all_results
    
    @property
    def supports_batch(self) -> bool:
        """Gemini supports efficient batch processing."""
        return True
