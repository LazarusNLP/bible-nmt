"""
Main processing engine for post-editing pipeline.
"""

import json
import os
from typing import List, Optional
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from core import Row, DataHandler, MetricsCalculator, GlossaryEntry
from models import BaseLLM, TokenCounter
from retrieval import FewShotSelector, GlossarySelector


class ProcessingEngine:
    """Main processing engine that orchestrates the post-editing pipeline."""
    
    def __init__(self, llm: BaseLLM, few_shot_selector: Optional[FewShotSelector] = None,
                 glossary_selector: Optional[GlossarySelector] = None,
                 token_counter: Optional[TokenCounter] = None):
        """
        Initialize processing engine.
        
        Args:
            llm: Language model interface
            few_shot_selector: Few-shot example selector (optional)
            glossary_selector: Glossary selector for terminological assistance (optional)
            token_counter: Token counter for statistics (optional)
        """
        self.llm = llm
        self.few_shot_selector = few_shot_selector
        self.glossary_selector = glossary_selector
        self.token_counter = token_counter or TokenCounter()
        
    def process_rows(self, rows: List[Row], output_filepath: str, prompt_key: str = "default",
                    few_shot_corpus: Optional[List[tuple]] = None, num_few_shot: int = 5,
                    max_glossary_entries: Optional[int] = None, batch_size: Optional[int] = None, 
                    num_workers: int = 8, debug: bool = False, prompt_only: bool = False,
                    sequential: bool = False) -> dict:
        """
        Process rows through the post-editing pipeline.
        
        Args:
            rows: List of rows to process
            output_filepath: Path to save results
            prompt_key: Prompt template key
            few_shot_corpus: Corpus for few-shot examples
            num_few_shot: Number of few-shot examples
            max_glossary_entries: Maximum number of glossary entries per row (None = all available)
            batch_size: Batch size for processing
            num_workers: Number of worker threads
            debug: Enable debug output
            prompt_only: If True, print first sample prompt and exit without processing
            sequential: If True, process requests sequentially without concurrency
            
        Returns:
            Dictionary with processing statistics and metrics
        """
        # Find unprocessed rows
        unprocessed_rows = DataHandler.find_unprocessed_rows(rows, output_filepath)
        
        if not unprocessed_rows:
            print("All rows have already been processed!")
            return self._calculate_final_metrics(output_filepath)
        
        # Get few-shot examples for unprocessed rows
        few_shot_examples_list = self._prepare_few_shot_examples(
            unprocessed_rows, few_shot_corpus, num_few_shot
        )
        
        # Get glossary entries for unprocessed rows
        glossary_entries_list = self._prepare_glossary_entries(
            unprocessed_rows, max_glossary_entries
        )
        
        # Generate sample message for logging
        sample_message = unprocessed_rows[0].get_messages(
            prompt_key, few_shot_examples_list[0], glossary_entries_list[0]
        )
        
        # Handle prompt-only mode - print and exit
        if prompt_only:
            print("\n" + "="*80)
            print("🔍 PROMPT-ONLY MODE: FINAL LLM PROMPT FOR FIRST SAMPLE")
            print("="*80)
            print(f"📄 Input Row Details:")
            print(f"   Source Text: {unprocessed_rows[0].src_text}")
            print(f"   Target Text (Ground Truth): {unprocessed_rows[0].tgt_text}")
            print(f"   Pred Text (MT Output): {unprocessed_rows[0].pred_tgt_text}")
            print(f"   Language Pair: {unprocessed_rows[0].src_lang_name} → {unprocessed_rows[0].tgt_lang_name}")
            print()
            
            # Show few-shot examples count
            if few_shot_examples_list[0]:
                print(f"📚 Few-shot Examples: {len(few_shot_examples_list[0])} examples")
            else:
                print(f"📚 Few-shot Examples: None")
                
            # Show glossary entries count
            if glossary_entries_list[0]:
                print(f"📖 Glossary Entries: {len(glossary_entries_list[0])} entries")
            else:
                print(f"📖 Glossary Entries: None")
            
            print()
            print("🤖 COMPLETE PROMPT THAT WILL BE SENT TO LLM:")
            print("-" * 80)
            
            for i, msg in enumerate(sample_message):
                print(f"\n[MESSAGE {i+1}: {msg['role'].upper()}]")
                print(">" * 40)
                # Truncate few-shot examples for display (show first 20 and last 20)
                display_content = self._truncate_few_shot_examples_for_display(msg['content'])
                print(display_content)
                print("<" * 40)
            
            # Token count if available
            if self.token_counter:
                token_count = self.token_counter.count_tokens_in_messages(sample_message)
                print(f"\n🔢 ESTIMATED TOKEN COUNT: {token_count} tokens")
            
            print("\n" + "="*80)
            print("✅ PROMPT-ONLY MODE COMPLETE - EXITING WITHOUT POST-EDITING")
            print("="*80)
            
            return {"prompt_only_mode": True, "message_displayed": True}
        
        if debug:
            # Show detailed glossary matching for first sample
            if self.glossary_selector and glossary_entries_list[0]:
                print("\n" + "="*60)
                print("DEBUG: GLOSSARY MATCHING DETAILS FOR FIRST SAMPLE")
                print("="*60)
                print(f"Input text: {unprocessed_rows[0].src_text}")
                _, debug_info = self.glossary_selector.get_relevant_entries_debug(unprocessed_rows[0].src_text)
                print(f"Words extracted: {debug_info['words']}")
                if debug_info.get('total_ngrams'):
                    print(f"N-grams extracted: {debug_info['total_ngrams']} total (max length: {debug_info.get('max_ngram_length', 'N/A')})")
                print(f"Exact matches: {debug_info['exact_matches']}")
                print(f"Normalized matches: {debug_info['normalized_matches']}")
                print(f"Lemmatization matches: {debug_info['lemma_matches']}")
                if debug_info['lemma_details']:
                    print(f"Lemmatizations: {', '.join(debug_info['lemma_details'])}")
                print(f"Total glossary entries found: {debug_info['total_entries']}")
                if debug_info.get('match_details'):
                    print("Match details (with n-gram info):")
                    for detail in debug_info['match_details'][:10]:  # Show first 10 (may include multi-word matches)
                        print(f"  {detail}")
                if debug_info['sample_entries']:
                    print("Sample entries:")
                    for src, tgt, pos in debug_info['sample_entries']:
                        pos_str = f" ({pos})" if pos else ""
                        print(f"  - {src}{pos_str} -> {tgt}")
                print("="*60)
            
            self._print_debug_messages(sample_message)
            
            # Add token counting for the first query in debug mode
            if self.token_counter:
                first_query_tokens = self.token_counter.count_tokens_in_messages(sample_message)
                print(f"\n🔢 TOKEN COUNT FOR FIRST QUERY: {first_query_tokens} tokens")
                print("="*60)
        else:
            print("Sample message generated (use --debug to see full content)")
        
        # Process rows
        successful_count = 0
        skipped_count = 0
        
        try:
            if sequential:
                # Force sequential processing regardless of other settings
                print(f"Sequential mode: Processing {len(unprocessed_rows)} rows one by one...")
                successful_count, skipped_count = self._process_individually(
                    unprocessed_rows, few_shot_examples_list, glossary_entries_list,
                    output_filepath, prompt_key, debug
                )
            elif batch_size and len(unprocessed_rows) > batch_size:
                successful_count, skipped_count = self._process_in_batches(
                    unprocessed_rows, few_shot_examples_list, glossary_entries_list, 
                    output_filepath, prompt_key, batch_size, num_workers, debug, sequential
                )
            else:
                # Process all rows at once (no outer batching)
                # This is optimal for Gemini async processing
                successful_count, skipped_count = self._process_all_at_once(
                    unprocessed_rows, few_shot_examples_list, glossary_entries_list,
                    output_filepath, prompt_key, num_workers, debug, sequential
                )
                
        except Exception as e:
            print(f"Error during processing: {e}")
            print("Attempting to process remaining rows individually...")
            successful_count, skipped_count = self._process_individually(
                unprocessed_rows, few_shot_examples_list, glossary_entries_list,
                output_filepath, prompt_key, debug
            )
        
        print(f"Processing complete: {successful_count} successful, {skipped_count} skipped")
        
        # Calculate and return final metrics
        return self._calculate_final_metrics(output_filepath)
    
    def _prepare_few_shot_examples(self, rows: List[Row], corpus: Optional[List[tuple]], 
                                  num_few_shot: int) -> List[Optional[List[tuple]]]:
        """Prepare few-shot examples for all rows."""
        if self.few_shot_selector and corpus:
            return self.few_shot_selector.get_examples_for_rows(rows, corpus, num_few_shot)
        else:
            return [None for _ in rows]
    
    def _prepare_glossary_entries(self, rows: List[Row], max_entries: Optional[int]) -> List[Optional[List[GlossaryEntry]]]:
        """Prepare glossary entries for all rows."""
        if self.glossary_selector:
            return self.glossary_selector.get_entries_for_rows(rows, max_entries)
        else:
            return [None for _ in rows]
    
    def _truncate_few_shot_examples_for_display(self, content: str) -> str:
        """Truncate few-shot examples in content to show only first 20 and last 20 for display."""
        # Look for the few-shot examples section
        if "To help with the translation, here are some example" in content:
            lines = content.split('\n')
            
            # Find the start of few-shot examples
            start_idx = -1
            for i, line in enumerate(lines):
                if "To help with the translation, here are some example" in line:
                    start_idx = i
                    break
            
            if start_idx != -1:
                # Find the end of few-shot examples (before glossary or end)
                end_idx = len(lines)
                for i in range(start_idx + 1, len(lines)):
                    if "To help with the translation, here is a" in lines[i] or not lines[i].strip():
                        if "To help with the translation, here is a" in lines[i]:
                            end_idx = i
                            break
                
                # Extract few-shot lines (skip the header)
                few_shot_lines = []
                for i in range(start_idx + 1, end_idx):
                    if lines[i].strip():  # Skip empty lines
                        few_shot_lines.append(lines[i])
                
                # If we have more than 40 lines, truncate to first 20 and last 20
                if len(few_shot_lines) > 40:
                    truncated_lines = (
                        few_shot_lines[:20] + 
                        [f"\n... [TRUNCATED: showing first 20 and last 20 of {len(few_shot_lines)} total few-shot examples] ...\n"] +
                        few_shot_lines[-20:]
                    )
                    
                    # Reconstruct the content
                    new_lines = (
                        lines[:start_idx + 1] +  # Everything before few-shot examples + header
                        truncated_lines +         # Truncated few-shot examples
                        lines[end_idx:]          # Everything after few-shot examples
                    )
                    return '\n'.join(new_lines)
        
        # If no truncation needed or pattern not found, return original
        return content

    def _print_debug_messages(self, messages: List[dict]):
        """Print debug information for messages."""
        print(f"\n" + "="*80)
        print("DEBUG: COMPLETE FIRST SAMPLE PROMPT")
        print("="*80)
        for i, msg in enumerate(messages):
            print(f"\n[{msg['role'].upper()} MESSAGE]:")
            print("-" * 40)
            print(f"{msg['content']}")
            print("-" * 40)
        print("="*80)
        print("END DEBUG PROMPT")
        print("="*80 + "\n")
    
    def _process_in_batches(self, rows: List[Row], few_shot_list: List, glossary_list: List,
                           output_filepath: str, prompt_key: str, batch_size: int, 
                           num_workers: int, debug: bool, sequential: bool = False) -> tuple:
        """Process rows in batches."""
        print(f"Processing {len(rows)} rows in batches of {batch_size}...")
        
        successful_count = 0
        skipped_count = 0
        total_token_counts = []
        
        for batch_start in range(0, len(rows), batch_size):
            batch_end = min(batch_start + batch_size, len(rows))
            batch_rows = rows[batch_start:batch_end]
            batch_few_shot = few_shot_list[batch_start:batch_end]
            batch_glossary = glossary_list[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(rows) + batch_size - 1)//batch_size} "
                  f"(rows {batch_start+1}-{batch_end})...")
            
            # Generate messages for this batch
            batch_messages = [row.get_messages(prompt_key, few_shot_examples, glossary_entries) 
                            for row, few_shot_examples, glossary_entries 
                            in zip(batch_rows, batch_few_shot, batch_glossary)]
            
            if debug and batch_start == 0:
                print(f"\n--- DEBUG: First batch with {len(batch_messages)} messages ---")
                for i, msg_list in enumerate(batch_messages[:2]):
                    print(f"\nMessage {i+1}:")
                    for j, msg in enumerate(msg_list):
                        print(f"  Role: {msg['role']}")
                        print(f"  Content: {msg['content'][:200]}...")
                print("--- End debug messages ---\n")
            
            # Count tokens
            if self.token_counter:
                batch_token_counts = [self.token_counter.count_tokens_in_messages(msg_list) 
                                    for msg_list in batch_messages]
                total_token_counts.extend(batch_token_counts)
                batch_avg = sum(batch_token_counts) / len(batch_token_counts)
                print(f"  Batch token average: {batch_avg:.1f}")
            
            # Generate translations
            if sequential:
                # Sequential processing - process one by one
                print(f"  Sequential processing: Processing {len(batch_messages)} messages one by one...")
                batch_translations = []
                for i, msg_list in enumerate(batch_messages):
                    try:
                        translation = self.llm.generate(msg_list)
                        batch_translations.append(translation)
                        print(f"    Processed message {i+1}/{len(batch_messages)}")
                    except Exception as e:
                        print(f"    Error in message {i+1}: {e}")
                        batch_translations.append("")
            elif self.llm.supports_batch:
                print(f"  Using {self.llm.model_name} native batch processing with incremental saving...")
                # Check if the model supports incremental saving (Gemini async does)
                if hasattr(self.llm, 'generate_batch') and 'gemini' in self.llm.model_name.lower():
                    batch_translations = self.llm.generate_batch(
                        batch_messages, 
                        rows_data=batch_rows, 
                        output_filepath=output_filepath,
                        sequential=False  # Sequential mode handled above
                    )
                else:
                    # For vLLM and other batch-supporting models
                    batch_translations = self.llm.generate_batch(batch_messages, sequential=False)
            else:
                print(f"  Using thread-based parallel processing with {num_workers} workers...")
                def process_single_message(msg_list):
                    try:
                        return self.llm.generate(msg_list)
                    except Exception as e:
                        print(f"Error in API call: {e}")
                        return ""
                
                batch_translations = thread_map(process_single_message, batch_messages, 
                                                max_workers=num_workers, 
                                                desc=f"Batch {batch_start//batch_size + 1}")
            
            # Process and save results
            # For Gemini with incremental saving, results are already saved during batch processing
            if hasattr(self.llm, 'generate_batch') and 'gemini' in self.llm.model_name.lower() and self.llm.supports_batch:
                # Count successful/skipped without duplicate saving
                batch_success = sum(1 for t in batch_translations if t and not t.startswith("["))
                batch_skipped = len(batch_translations) - batch_success
                print(f"  Batch results already saved incrementally: {batch_success} successful, {batch_skipped} skipped")
            else:
                # For other models, save results normally
                batch_success, batch_skipped = self._save_batch_results(
                    batch_rows, batch_translations, output_filepath
                )
            successful_count += batch_success
            skipped_count += batch_skipped
        
        # Print statistics
        if self.token_counter and total_token_counts:
            self._print_token_statistics(total_token_counts)
        
        return successful_count, skipped_count
    
    def _process_all_at_once(self, rows: List[Row], few_shot_list: List, glossary_list: List,
                            output_filepath: str, prompt_key: str, num_workers: int, debug: bool, sequential: bool = False) -> tuple:
        """Process all rows at once."""
        print("Generating messages for all unprocessed rows...")
        messages = [row.get_messages(prompt_key, few_shot_examples, glossary_entries) 
                   for row, few_shot_examples, glossary_entries 
                   in zip(rows, few_shot_list, glossary_list)]
        
        if debug:
            print(f"\n--- DEBUG: Generated {len(messages)} messages for processing ---")
            for i, msg_list in enumerate(messages[:3]):
                print(f"\nMessage set {i+1}:")
                for j, msg in enumerate(msg_list):
                    print(f"  Role: {msg['role']}")
                    print(f"  Content: {msg['content']}")
            if len(messages) > 3:
                print(f"  ... and {len(messages) - 3} more message sets")
            print("--- End debug messages ---\n")
        
        # Count tokens
        if self.token_counter:
            print("Counting tokens for all messages...")
            token_counts = [self.token_counter.count_tokens_in_messages(msg_list) 
                          for msg_list in messages]
            self._print_token_statistics(token_counts)
            if debug:
                print("First 10 message token counts:", token_counts[:10])
        
        # Generate translations
        print("Generating translations...")
        if sequential:
            # Sequential processing - process one by one
            print(f"Sequential processing: Processing {len(messages)} messages one by one...")
            translations = []
            for i, msg_list in enumerate(messages):
                try:
                    translation = self.llm.generate(msg_list)
                    translations.append(translation)
                    print(f"  Processed message {i+1}/{len(messages)}")
                except Exception as e:
                    print(f"  Error in message {i+1}: {e}")
                    translations.append("")
        elif self.llm.supports_batch:
            print(f"Using {self.llm.model_name} native batch processing with incremental saving...")
            # Check if the model supports incremental saving (Gemini async does)
            if hasattr(self.llm, 'generate_batch') and 'gemini' in self.llm.model_name.lower():
                translations = self.llm.generate_batch(messages, rows_data=rows, output_filepath=output_filepath, sequential=False)
            else:
                # For vLLM and other batch-supporting models
                translations = self.llm.generate_batch(messages, sequential=False)
        else:
            print(f"Using thread-based parallel processing with {num_workers} workers...")
            def process_single_message(msg_list):
                try:
                    return self.llm.generate(msg_list)
                except Exception as e:
                    print(f"Error in API call: {e}")
                    return ""
            
            translations = thread_map(process_single_message, messages, 
                                    max_workers=num_workers, 
                                    desc="Generating translations")
        
        # Print sample results
        print("\n" + "="*50)
        print("POST EDITING RESULT")
        print("Generated translations:")
        for i, translation in enumerate(translations[:5]):
            print(f"  Translation {i+1}: {translation}")
        if len(translations) > 5:
            print(f"  ... and {len(translations) - 5} more translations")
        print("="*50 + "\n")
        
        # Save results
        return self._save_batch_results(rows, translations, output_filepath)
    
    def _process_individually(self, rows: List[Row], few_shot_list: List, glossary_list: List,
                             output_filepath: str, prompt_key: str, debug: bool) -> tuple:
        """Process rows individually as fallback."""
        successful_count = 0
        skipped_count = 0
        
        for i, (row, few_shot_examples, glossary_entries) in enumerate(zip(rows, few_shot_list, glossary_list)):
            try:
                messages = row.get_messages(prompt_key, few_shot_examples, glossary_entries)
                
                if self.token_counter:
                    token_count = self.token_counter.count_tokens_in_messages(messages)
                    print(f"Row {i+1} token count: {token_count}")
                
                if debug:
                    print(f"\n--- DEBUG: Individual processing row {i+1} ---")
                    for j, msg in enumerate(messages):
                        print(f"Message {j+1} ({msg['role']}):")
                        print(f"{msg['content']}")
                    print("--- End debug message ---\n")
                
                translation = self.llm.generate(messages)
                
                if translation and translation.strip() and not translation.startswith("["):
                    row.post_edited_tgt_txt = translation
                    DataHandler.append_row_to_csv(row, output_filepath)
                    successful_count += 1
                else:
                    print(f"Skipping individual row: Invalid translation '{translation}'")
                    skipped_count += 1
                    
            except Exception as row_error:
                print(f"Error processing row {i}: {row_error}")
                skipped_count += 1
        
        return successful_count, skipped_count
    
    def _save_batch_results(self, rows: List[Row], translations: List[str], 
                           output_filepath: str) -> tuple:
        """Save batch results to file."""
        successful_count = 0
        skipped_count = 0
        
        for row, translation in tqdm(zip(rows, translations), desc="Saving results"):
            if not translation or translation.strip() == "" or translation.startswith("["):
                print(f"Skipping row: Invalid content '{translation}' for source: {row.src_text[:50]}...")
                skipped_count += 1
                continue
            
            row.post_edited_tgt_txt = translation
            DataHandler.append_row_to_csv(row, output_filepath)
            successful_count += 1
        
        return successful_count, skipped_count
    
    def _print_token_statistics(self, token_counts: List[int]):
        """Print token counting statistics."""
        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
            max_tokens = max(token_counts)
            min_tokens = min(token_counts)
            print(f"Token statistics - Average: {avg_tokens:.1f}, Min: {min_tokens}, Max: {max_tokens}")
    
    def _calculate_final_metrics(self, output_filepath: str) -> dict:
        """Calculate final metrics from completed results."""
        if os.path.exists(output_filepath):
            completed_rows = DataHandler.load_from_savepath(output_filepath)
            # Filter out rows with empty predictions for metrics calculation
            valid_rows = [row for row in completed_rows 
                         if row.post_edited_tgt_txt and row.post_edited_tgt_txt.strip()]
            print(f"Calculating metrics for {len(valid_rows)} valid completed rows")
            
            if valid_rows:
                metrics = MetricsCalculator.calculate_metrics(valid_rows)
                
                # Save metrics to JSON file
                metrics_filepath = output_filepath.replace('.csv', '_metrics.json')
                with open(metrics_filepath, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                print(f"Metrics saved to: {metrics_filepath}")
                
                return metrics
        
        print("No completed rows found. Cannot calculate metrics.")
        return {}
