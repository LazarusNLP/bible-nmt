"""
Prompt templates for translation tasks.
Variables in prompts use {variable_name} format for substitution.
"""

PROMPTS = {
    "default": {
        "system": "You are an expert post-editor specializing in improving machine translation output. Your task is to review and correct machine-translated text to produce high-quality, fluent, and accurate translations in the target language only. Focus on fixing grammatical errors, improving naturalness, and ensuring the meaning is preserved. You must respond ONLY with a JSON object containing the 'post_edited_text' field with the corrected translation in the target language. Do not include the source text, do not provide explanations, and do not echo the original machine translation unless it needs no correction.",
        "user_template": "Source text ({src_lang_name}): {src_text}\n\nMachine translation ({tgt_lang_name}): {pred_text}\n\nProvide the corrected and improved translation in {tgt_lang_name} as JSON: {{\"post_edited_text\": \"your corrected translation in {tgt_lang_name} here\"}}",        
    },
    "post_editing": {
        "system": "You are an expert post-editor specializing in improving machine translation output. Your task is to review and correct machine-translated text to produce high-quality, fluent, and accurate translations in the target language only. Focus on fixing grammatical errors, improving naturalness, and ensuring the meaning is preserved. You must respond ONLY with a JSON object containing the 'post_edited_text' field with the corrected translation in the target language. Do not include the source text, do not provide explanations, and do not echo the original machine translation unless it needs no correction.",
        "user_template": "Source text ({src_lang_name}): {src_text}\n\nMachine translation ({tgt_lang_name}): {pred_text}\n\nProvide the corrected and improved translation in {tgt_lang_name} as JSON: {{\"post_edited_text\": \"your corrected translation in {tgt_lang_name} here\"}}",
    },
}

def get_prompt(prompt_key: str):
    """
    Get prompt template by key.
    
    Args:
        prompt_key: Key to look up in PROMPTS dictionary
        
    Returns:
        Dictionary containing system, user_template, and assistant_template
        
    Raises:
        KeyError: If prompt_key is not found in PROMPTS
    """
    if prompt_key not in PROMPTS:
        available_keys = ", ".join(PROMPTS.keys())
        raise KeyError(f"Prompt key '{prompt_key}' not found. Available keys: {available_keys}")
    
    return PROMPTS[prompt_key]

def list_available_prompts():
    """Return list of available prompt keys."""
    return list(PROMPTS.keys())
