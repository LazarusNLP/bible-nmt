"""
API-based model implementations (GPT, Gemini).
"""

import os
import asyncio
import time
from typing import List
from pydantic import BaseModel
from .base import BaseLLM


class PostEditedOutput(BaseModel):
    """Pydantic model for structured post-editing output."""
    post_edited_text: str


class GPTModel(BaseLLM):
    """OpenAI GPT model interface."""
    
    def __init__(self, model: str = "gpt-4o"):
        """Initialize GPT model."""
        self._model_name = model
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, messages: List[dict]) -> str:
        """Generate single response using OpenAI API."""
        completion = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=4096,  # Increased for longer outputs
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
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        """Initialize Gemini model."""
        self._model_name = model
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

    def generate(self, messages: List[dict]) -> str:
        """Generate single response using official Gemini API with structured output."""
        try:
            combined_prompt = self._combine_messages(messages)
            
            # Use structured output with response_schema and increased generation config
            response = self.client.models.generate_content(  
                model=f"models/{self._model_name}",
                contents=[{
                    'parts': [{'text': combined_prompt}],
                    'role': 'user'
                }],
                config={
                    "response_mime_type": "application/json", 
                    "response_schema": PostEditedOutput,
                    "max_output_tokens": 8192,  # Increased output token limit
                }
            )
            
            return self._parse_response(response)
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return ""

    async def _generate_async(self, messages: List[dict]) -> str:
        """Generate single response asynchronously using the new SDK."""
        try:
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
                    "max_output_tokens": 8192,
                }
            )
            
            return self._parse_response(response)
            
        except Exception as e:
            print(f"Error in async Gemini API call: {e}")
            return ""

    def _parse_response(self, response) -> str:
        """Parse response from Gemini API."""
        if response and hasattr(response, 'parsed') and response.parsed:
            # Use the structured parsed response
            return response.parsed.post_edited_text.strip()
        elif response and hasattr(response, 'candidates') and response.candidates:
            # Fallback to manual JSON parsing if parsed response not available
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                if candidate.content.parts:
                    import json
                    try:
                        json_response = json.loads(candidate.content.parts[0].text.strip())
                        return json_response.get('post_edited_text', '').strip()
                    except json.JSONDecodeError:
                        print(f"Warning: Failed to parse JSON response from {self._model_name}")
                        return candidate.content.parts[0].text.strip()
        
        print(f"Warning: No content returned from {self._model_name}")
        return ""

    def generate_batch(self, messages_list: List[List[dict]], rows_data=None, output_filepath=None, sequential=False) -> List[str]:
        """Generate responses for a batch of message lists using async processing with incremental saving."""
        if sequential:
            # Force sequential processing without async batching
            print(f"Generating {len(messages_list)} responses using Gemini SEQUENTIAL processing...")
            results = []
            for i, messages in enumerate(messages_list):
                try:
                    result = self.generate(messages)
                    results.append(result)
                    print(f"  Processed Gemini request {i+1}/{len(messages_list)}")
                except Exception as e:
                    print(f"  Error in Gemini request {i+1}: {e}")
                    results.append("")
            return results
        
        print(f"Generating {len(messages_list)} responses using Gemini async batch processing...")
        
        # Try to use existing event loop, or create one if needed
        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
            print("Using existing event loop")
            # We're in an async context, but we need to run from sync context
            # Use asyncio.run_coroutine_threadsafe or similar approach
            import concurrent.futures
            import threading
            
            def run_async():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        self._generate_batch_async(messages_list, rows_data, output_filepath)
                    )
                finally:
                    # Ensure the loop is properly closed to prevent resource leaks
                    try:
                        # Cancel any remaining tasks
                        pending = asyncio.all_tasks(new_loop)
                        for task in pending:
                            task.cancel()
                        
                        # Close the loop properly
                        new_loop.close()
                        print("✅ Event loop cleaned up successfully")
                    except Exception as cleanup_error:
                        print(f"⚠️ Loop cleanup warning (normal): {cleanup_error}")
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async)
                return future.result()
                
        except RuntimeError:
            # No running loop, we can create our own
            print("Creating new event loop")
            try:
                return asyncio.run(self._generate_batch_async(messages_list, rows_data, output_filepath))
            finally:
                # Additional cleanup after asyncio.run() completes
                import gc
                gc.collect()
                print("✅ Async processing cleanup completed")

    async def _generate_batch_async(self, messages_list: List[List[dict]], rows_data=None, output_filepath=None) -> List[str]:
        """Async batch processing implementation with token-aware rate limiting, retry logic, and incremental saving."""
        # Import here to avoid circular imports
        from core.data_io import DataHandler
        
        try:
            # Create async tasks for all requests
            tasks = [self._generate_async_with_retry(messages) for messages in messages_list]
            
            # Rate limiting configuration for Gemini API
            # Being much more conservative to prevent event loop issues and rate limits
            max_concurrent = min(1, len(tasks))  # Very small batches - only 10 concurrent
            delay_between_batches = 0  # 30 second delay between batches to prevent rate limits
            
            results = []
            batch_size = max_concurrent
            
            print(f"Processing {len(tasks)} requests with CONSERVATIVE rate limiting...")
            print(f"Max concurrent: {max_concurrent}, Delay between batches: {delay_between_batches}s")
            if output_filepath:
                print(f"Incremental saving enabled: Results will be saved after each batch to {output_filepath}")
            print(f"This prevents quota exhaustion and preserves progress if interrupted")
            
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(tasks) + batch_size - 1) // batch_size
                
                print(f"Processing async batch {batch_num}/{total_batches} ({len(batch_tasks)} requests)...")
                batch_start_time = time.time()
                
                # Execute batch with gather and progress tracking
                try:
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    batch_duration = time.time() - batch_start_time
                    print(f"  Batch {batch_num} completed in {batch_duration:.1f}s")
                except Exception as e:
                    print(f"Error in batch {batch_num}: {e}")
                    # Create empty results for failed batch
                    batch_results = ["" for _ in batch_tasks]
                
                # Clean up any pending tasks after each batch
                await asyncio.sleep(0.1)  # Brief pause for cleanup
            
            # Handle any exceptions and collect results
            batch_successful = 0
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"Error in request {i + j + 1}: {result}")
                    results.append("")
                else:
                    result_text = result if result else ""
                    results.append(result_text)
                    if result_text and not result_text.startswith("["):
                        batch_successful += 1
            
            # INCREMENTAL SAVING: Save this batch's results immediately
            if output_filepath and rows_data and batch_successful > 0:
                print(f"  Saving {batch_successful} successful results from batch {batch_num}...")
                try:
                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            continue
                        
                        result_text = result if result else ""
                        if result_text and not result_text.startswith("["):
                            # Get the corresponding row and update it
                            row_index = i + j
                            if row_index < len(rows_data):
                                row = rows_data[row_index]
                                row.post_edited_tgt_txt = result_text
                                DataHandler.append_row_to_csv(row, output_filepath)
                    
                    print(f"  ✅ Batch {batch_num} results saved to CSV")
                except Exception as save_error:
                    print(f"  ⚠️ Error saving batch {batch_num}: {save_error}")
            
            # Add delay between batches to respect rate limits (except for last batch)
            if i + batch_size < len(tasks):
                print(f"  Waiting {delay_between_batches}s before next batch to respect rate limits...")
                await asyncio.sleep(delay_between_batches)
        
            success_count = sum(1 for r in results if r and not r.startswith("["))
            print(f"Async batch processing complete: {success_count}/{len(results)} successful")
            if output_filepath:
                print(f"All results have been incrementally saved to: {output_filepath}")
            
            return results
            
        finally:
            # Explicit cleanup to close HTTP sessions and prevent warnings
            await self._cleanup_async_resources()
    
    async def _cleanup_async_resources(self):
        """Clean up async resources to prevent unclosed session warnings."""
        if self._cleanup_performed:
            return
            
        try:
            # Give time for any pending operations to complete
            await asyncio.sleep(0.5)
            
            # Try to clean up the client's internal HTTP sessions
            if hasattr(self.client, 'aio') and hasattr(self.client.aio, '_client_session'):
                if hasattr(self.client.aio._client_session, 'close'):
                    await self.client.aio._client_session.close()
                    print("✅ Cleaned up async HTTP session")
            
            # Alternative cleanup approach - force garbage collection
            import gc
            gc.collect()
            self._cleanup_performed = True
            
        except Exception as cleanup_error:
            # Don't let cleanup errors affect the main process
            print(f"⚠️ Note: Cleanup completed with minor warnings (normal): {cleanup_error}")
        
        # Final pause to ensure cleanup completion
        await asyncio.sleep(0.1)

    async def _generate_async_with_retry(self, messages: List[dict], max_retries: int = 3) -> str:
        """Generate with exponential backoff retry for rate limit errors."""
        for attempt in range(max_retries + 1):
            try:
                return await self._generate_async(messages)
            except Exception as e:
                error_str = str(e)
                # Handle various error types
                if "Event loop is closed" in error_str or "different loop" in error_str:
                    print(f"Event loop error (attempt {attempt + 1}): {e}")
                    if attempt < max_retries:
                        await asyncio.sleep(2)  # Brief pause before retry
                        continue
                    else:
                        return ""
                elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries:
                        # Extract retry delay from error message if available
                        retry_delay = 60  # Default 60 seconds
                        if "retry in" in error_str.lower():
                            import re
                            match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_str.lower())
                            if match:
                                retry_delay = float(match.group(1))
                        
                        print(f"Rate limit hit, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries + 1})")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        print(f"Max retries exceeded for rate limit error: {e}")
                        return ""
                else:
                    print(f"Non-rate-limit error: {e}")
                    return ""
        return ""
    
    @property
    def supports_batch(self) -> bool:
        """Gemini supports efficient async batch processing."""
        return True
