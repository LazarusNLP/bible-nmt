"""
vLLM-based model implementation.
"""

import json
import os
from typing import List
from .base import BaseLLM


def parse_json_response(response_text: str) -> str:
    """Safely parse JSON response from vLLM and extract post_edited_text."""
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


class VLLMModel(BaseLLM):
    """vLLM model interface with guided JSON decoding."""
    
    def __init__(self, model_path: str, **kwargs):
        """Initialize vLLM model."""
        self._model_name = model_path
        
        # Set up vLLM environment variables
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        os.environ['DISABLE_XFORMERS'] = '1'
        os.environ['FLASH_ATTENTION_SKIP_CUDA_BUILD'] = '1'
        os.environ['ENFORCE_EAGER'] = '1'
        
        # Import vLLM components
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        from core.constants import POST_EDIT_JSON_SCHEMA
        
        # Default vLLM configuration
        default_config = {
            'max_model_len': 6000,
            'gpu_memory_utilization': 0.95,
            'disable_log_stats': True,
            'tensor_parallel_size': 1,
            'enable_chunked_prefill': True,
            'enforce_eager': True,
        }
        
        # Merge with user-provided kwargs
        config = {**default_config, **kwargs}
        
        print(f"Initializing vLLM model: {model_path}")
        try:
            self.model = LLM(model_path, **config)
            self.sampling_params = SamplingParams(
                max_tokens=512,
                temperature=0.0,
                guided_decoding=GuidedDecodingParams(json=POST_EDIT_JSON_SCHEMA),
            )
            print("✅ vLLM model initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing vLLM model: {e}")
            self._print_troubleshooting_info(e)
            raise e
    
    def _print_troubleshooting_info(self, error):
        """Print helpful troubleshooting information."""
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Check CUDA_VISIBLE_DEVICES environment variable")
        print("2. Verify GPU availability with: nvidia-smi")
        print("3. Try setting CUDA_VISIBLE_DEVICES=0 before running")
        print("4. Check if the model name/path is correct")
        print("5. Ensure sufficient GPU memory is available")
        
        # Try to provide more specific error information
        if "Device string must not be empty" in str(error):
            print("\n🎯 Specific fix for 'Device string must not be empty' error:")
            print("   - This usually indicates a CUDA device detection issue")
            print("   - Try: export CUDA_VISIBLE_DEVICES=0")
            print("   - Or run: CUDA_VISIBLE_DEVICES=0 python your_script.py")
            print("   - Check if CUDA is properly installed and accessible")

    def generate(self, messages: List[dict]) -> str:
        """Generate single response using vLLM."""
        outputs = self.model.chat([messages], sampling_params=self.sampling_params)
        raw_response = outputs[0].outputs[0].text
        return parse_json_response(raw_response)
    
    def generate_batch(self, messages_list: List[List[dict]], sequential=False) -> List[str]:
        """Generate responses for a batch of messages using vLLM's efficient batching."""
        if sequential:
            # Force sequential processing without batching
            print(f"Generating {len(messages_list)} responses using vLLM SEQUENTIAL processing...")
            results = []
            for i, messages in enumerate(messages_list):
                try:
                    result = self.generate(messages)
                    results.append(result)
                    print(f"  Processed vLLM request {i+1}/{len(messages_list)}")
                except Exception as e:
                    print(f"  Error in vLLM request {i+1}: {e}")
                    results.append("")
            return results
        
        print(f"Generating {len(messages_list)} responses using vLLM batch processing...")
        outputs = self.model.chat(messages_list, sampling_params=self.sampling_params)
        
        # Parse all JSON responses
        results = []
        for output in outputs:
            raw_response = output.outputs[0].text
            parsed_response = parse_json_response(raw_response)
            results.append(parsed_response)
        
        return results
    
    @property
    def supports_batch(self) -> bool:
        """vLLM supports efficient batch processing."""
        return True
