"""
Main processing engine for post-editing pipeline.
"""

import json
import os
import time
from typing import List, Optional
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from core import Row, DataHandler, MetricsCalculator, GlossaryEntry
from models import BaseLLM, TokenCounter
from retrieval import FewShotSelector, GlossarySelector
from .timing import TimingTracker


class ProcessingEngine:
    """Main processing engine that orchestrates the post-editing pipeline."""
    
    def __init__(self, llm: BaseLLM, few_shot_selector: Optional[FewShotSelector] = None,
                 glossary_selector: Optional[GlossarySelector] = None,
                 token_counter: Optional[TokenCounter] = None,
                 disable_metrics: bool = False, output_dir: str = "./results"):
        """
        Initialize processing engine.
        
        Args:
            llm: Language model interface
            few_shot_selector: Few-shot example selector (optional)
            glossary_selector: Glossary selector for terminological assistance (optional)
            token_counter: Token counter for statistics (optional)
            disable_metrics: If True, skip individual metrics calculation for faster processing
            output_dir: Output directory for timing logs
        """
        self.llm = llm
        self.few_shot_selector = few_shot_selector
        self.glossary_selector = glossary_selector
        self.token_counter = token_counter or TokenCounter()
        self.disable_metrics = disable_metrics
        self.timing_tracker = TimingTracker(output_dir)
        
    def process_rows(self, rows: List[Row], output_filepath: str, prompt_key: str = "default",
                    few_shot_corpus: Optional[List[tuple]] = None, num_few_shot: int = 5,
                    max_glossary_entries: Optional[int] = None, batch_size: Optional[int] = None, 
                    num_workers: int = 8, debug: bool = False, prompt_only: bool = False,
                    sequential: bool = False, translation_mode: bool = False) -> dict:
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
            translation_mode: If True, use direct translation mode instead of post-editing
            
        Returns:
            Dictionary with processing statistics and metrics
        """
        # Start total timing
        self.timing_tracker.start_total_timing()
        
        # Find unprocessed rows
        with self.timing_tracker.timer("data_loading"):
            unprocessed_rows, has_fallback_rows = DataHandler.find_unprocessed_rows(rows, output_filepath)
            # Temporarily disable fallback replacement to avoid memory issues with large datasets
            # Fallback rows will be reprocessed on next run
            has_fallback_rows = False
        
        if not unprocessed_rows:
            print("All rows have already been processed!")
            return self._calculate_final_metrics(output_filepath)
        
        # Get few-shot examples for unprocessed rows
        with self.timing_tracker.timer("rag_retrieval"):
            few_shot_examples_list = self._prepare_few_shot_examples(
                unprocessed_rows, few_shot_corpus, num_few_shot
            )
        
        # Get glossary entries for unprocessed rows
        with self.timing_tracker.timer("glossary_selection"):
            glossary_entries_list = self._prepare_glossary_entries(
                unprocessed_rows, max_glossary_entries
            )
        
        # Generate sample message for logging
        sample_message = unprocessed_rows[0].get_messages(
            prompt_key, few_shot_examples_list[0], glossary_entries_list[0], translation_mode
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
                    output_filepath, prompt_key, debug, translation_mode, has_fallback_rows
                )
            elif batch_size and len(unprocessed_rows) > batch_size:
                successful_count, skipped_count = self._process_in_batches(
                    unprocessed_rows, few_shot_examples_list, glossary_entries_list, 
                    output_filepath, prompt_key, batch_size, num_workers, debug, sequential, translation_mode, has_fallback_rows
                )
            else:
                # Process all rows at once (no outer batching)
                # This is optimal for Gemini async processing
                successful_count, skipped_count = self._process_all_at_once(
                    unprocessed_rows, few_shot_examples_list, glossary_entries_list,
                    output_filepath, prompt_key, num_workers, debug, sequential, translation_mode, has_fallback_rows
                )
                
        except Exception as e:
            print(f"Error during processing: {e}")
            print("Attempting to process remaining rows individually...")
            successful_count, skipped_count = self._process_individually(
                unprocessed_rows, few_shot_examples_list, glossary_entries_list,
                output_filepath, prompt_key, debug, translation_mode, has_fallback_rows
            )
        
        print(f"Processing complete: {successful_count} successful, {skipped_count} fallback (using original pred_text)")
        
        # Calculate and return final metrics
        with self.timing_tracker.timer("final_metrics_calculation"):
            final_metrics = self._calculate_final_metrics(output_filepath)
        
        # End total timing and save timing analysis
        self.timing_tracker.end_total_timing()
        self.timing_tracker.print_summary()
        self.timing_tracker.save_to_file()
        
        return final_metrics
    
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
                           num_workers: int, debug: bool, sequential: bool = False, translation_mode: bool = False, has_fallback_rows: bool = False) -> tuple:
        """Process rows in batches."""
        print(f"Processing {len(rows)} rows in batches of {batch_size}...")
        
        successful_count = 0
        skipped_count = 0
        total_token_counts = []
        all_processed_rows = []  # Collect all rows if we need to replace
        
        for batch_start in range(0, len(rows), batch_size):
            batch_end = min(batch_start + batch_size, len(rows))
            batch_rows = rows[batch_start:batch_end]
            batch_few_shot = few_shot_list[batch_start:batch_end]
            batch_glossary = glossary_list[batch_start:batch_end]
            
            batch_num = batch_start//batch_size + 1
            total_batches = (len(rows) + batch_size - 1)//batch_size
            print(f"Processing batch {batch_num}/{total_batches} "
                  f"(rows {batch_start+1}-{batch_end})...")
            
            # Track batch timing
            batch_timing = {"batch_id": batch_num, "rows_processed": len(batch_rows)}
            batch_start_time = time.time()
            
            # Generate messages for this batch
            with self.timing_tracker.timer("message_generation"):
                message_gen_start = time.time()
                batch_messages = [row.get_messages(prompt_key, few_shot_examples, glossary_entries, translation_mode) 
                                for row, few_shot_examples, glossary_entries 
                                in zip(batch_rows, batch_few_shot, batch_glossary)]
                batch_timing["message_generation"] = time.time() - message_gen_start
            
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
            with self.timing_tracker.timer("llm_inference"):
                inference_start = time.time()
                
                if sequential:
                    # Sequential processing - process one by one
                    print(f"  Sequential processing: Processing {len(batch_messages)} messages one by one...")
                    batch_translations = []
                    for i, msg_list in enumerate(batch_messages):
                        try:
                            translation = self.llm.generate(msg_list, translation_mode)
                            batch_translations.append(translation)
                            print(f"    Processed message {i+1}/{len(batch_messages)}")
                        except Exception as e:
                            print(f"    Error in message {i+1}: {e}")
                            batch_translations.append("")
                elif self.llm.supports_batch:
                    # Check if the model supports incremental saving (Gemini async does)
                    if hasattr(self.llm, 'generate_batch') and 'gemini' in self.llm.model_name.lower():
                        if has_fallback_rows:
                            # Don't use incremental saving when replacing fallback rows
                            print(f"  Using {self.llm.model_name} native batch processing (no incremental saving for fallback replacement)...")
                            batch_translations = self.llm.generate_batch(
                                batch_messages, 
                                rows_data=batch_rows, 
                                output_filepath=None,  # Disable incremental saving
                                sequential=False,
                                translation_mode=translation_mode
                            )
                        else:
                            # Use incremental saving for normal processing
                            print(f"  Using {self.llm.model_name} native batch processing with incremental saving...")
                            batch_translations = self.llm.generate_batch(
                                batch_messages, 
                                rows_data=batch_rows, 
                                output_filepath=output_filepath,
                                sequential=False,  # Sequential mode handled above
                                translation_mode=translation_mode
                            )
                    else:
                        # For vLLM and other batch-supporting models
                        batch_translations = self.llm.generate_batch(batch_messages, sequential=False, translation_mode=translation_mode)
                else:
                    print(f"  Using thread-based parallel processing with {num_workers} workers...")
                    def process_single_message(msg_list):
                        try:
                            return self.llm.generate(msg_list, translation_mode)
                        except Exception as e:
                            print(f"Error in API call: {e}")
                            return ""
                    
                    batch_translations = thread_map(process_single_message, batch_messages, 
                                                    max_workers=num_workers, 
                                                    desc=f"Batch {batch_num}")
                
                batch_timing["llm_inference"] = time.time() - inference_start
            
            # Process and save results
            # For Gemini with incremental saving, results are already saved during batch processing (unless has_fallback_rows)
            if hasattr(self.llm, 'generate_batch') and 'gemini' in self.llm.model_name.lower() and self.llm.supports_batch:
                # Count successful/fallback
                batch_success = sum(1 for t in batch_translations if t and not t.startswith("["))
                batch_fallback = len(batch_translations) - batch_success
                
                if has_fallback_rows:
                    # When replacing fallback rows, we disabled incremental saving
                    # Process the rows but don't save yet - collect for replacement at the end
                    print(f"  Batch {batch_num}: {batch_success} successful, {batch_fallback} fallback (will replace at end)")
                    
                    # Update the batch_rows with translations and metrics
                    for i, (row, translation) in enumerate(zip(batch_rows, batch_translations)):
                        if translation and translation.strip() and not translation.startswith("["):
                            row.post_edited_tgt_txt = translation.strip()
                            # Calculate metrics if not disabled
                            if not self.disable_metrics:
                                try:
                                    from core.metrics import MetricsCalculator
                                    individual_metrics = MetricsCalculator.calculate_individual_metrics(
                                        original_text=row.pred_tgt_text,
                                        post_edited_text=translation.strip(),
                                        reference_text=row.tgt_text
                                    )
                                    row.spbleu_improvement = individual_metrics["improvements"]["spbleu"]
                                    row.chrf3_improvement = individual_metrics["improvements"]["chrf3"]
                                    row.chrfpp_improvement = individual_metrics["improvements"]["chrfpp"]
                                except Exception as e:
                                    row.spbleu_improvement = None
                                    row.chrf3_improvement = None
                                    row.chrfpp_improvement = None
                        else:
                            # Fallback case
                            row.post_edited_tgt_txt = row.pred_tgt_text
                            if not self.disable_metrics:
                                row.spbleu_improvement = 0.0
                                row.chrf3_improvement = 0.0
                                row.chrfpp_improvement = 0.0
                    
                    # Collect these rows for replacement at the end
                    all_processed_rows.extend(batch_rows)
                else:
                    # Normal incremental saving was used
                    print(f"  Batch results already saved incrementally: {batch_success} successful, {batch_fallback} fallback")
                
                # Calculate metrics for the batch if metrics are enabled (only for incremental saving case)
                if not has_fallback_rows and not self.disable_metrics and batch_success > 0:
                    with self.timing_tracker.timer("individual_metrics_calculation"):
                        metrics_start = time.time()
                        try:
                            print(f"  Calculating metrics for {batch_success} Gemini batch results...")
                            # Load the saved rows to calculate metrics
                            if os.path.exists(output_filepath):
                                # Read just the rows we need from the CSV
                                import pandas as pd
                                df = pd.read_csv(output_filepath)
                                # Get the last batch_success rows (most recently saved)
                                recent_rows = df.tail(batch_success)
                                
                                # Calculate metrics for these rows
                                for idx, (_, row_data) in enumerate(recent_rows.iterrows()):
                                    if row_data['post_edited_tgt_txt'] and pd.notna(row_data['post_edited_tgt_txt']):
                                        try:
                                            from core.metrics import MetricsCalculator
                                            pred_text = row_data['pred_tgt_text']
                                            post_edited_text = row_data['post_edited_tgt_txt']
                                            reference_text = row_data['tgt_text']
                                            
                                            metrics = MetricsCalculator.calculate_individual_metrics(
                                                pred_text, post_edited_text, reference_text
                                            )
                                            
                                            # Update the CSV with metrics
                                            df.loc[df.index[-batch_success + idx], 'spbleu_improvement'] = metrics["improvements"]["spbleu"]
                                            df.loc[df.index[-batch_success + idx], 'chrf3_improvement'] = metrics["improvements"]["chrf3"]
                                            df.loc[df.index[-batch_success + idx], 'chrfpp_improvement'] = metrics["improvements"]["chrfpp"]
                                            
                                        except Exception as e:
                                            print(f"    Warning: Metrics calculation failed for row {idx}: {e}")
                                            df.loc[df.index[-batch_success + idx], 'spbleu_improvement'] = None
                                            df.loc[df.index[-batch_success + idx], 'chrf3_improvement'] = None
                                            df.loc[df.index[-batch_success + idx], 'chrfpp_improvement'] = None
                                
                                # Save updated CSV with metrics
                                df.to_csv(output_filepath, index=False)
                                print(f"    ✅ Updated CSV with metrics for {batch_success} rows")
                                
                        except Exception as e:
                            print(f"  Warning: Batch metrics calculation failed: {e}")
                        batch_timing["individual_metrics_calculation"] = time.time() - metrics_start
            else:
                # For other models, save results normally
                with self.timing_tracker.timer("data_saving"):
                    saving_start = time.time()
                    batch_success, batch_fallback = self._save_batch_results(
                        batch_rows, batch_translations, output_filepath, translation_mode, has_fallback_rows
                    )
                    batch_timing["data_saving"] = time.time() - saving_start
                
                # If we have fallback rows, collect these processed rows for replacement
                if has_fallback_rows:
                    all_processed_rows.extend(batch_rows)
            
            successful_count += batch_success
            skipped_count += batch_fallback
            
            # Record total batch time and log it
            batch_timing["total_batch_time"] = time.time() - batch_start_time
            self.timing_tracker.record_batch_timing(batch_timing)
            print(f"  Batch {batch_num} completed in {batch_timing['total_batch_time']:.2f}s")
        
        # Print statistics
        if self.token_counter and total_token_counts:
            self._print_token_statistics(total_token_counts)
        
        # If we have fallback rows and collected processed rows, replace them in the CSV
        # Write in chunks to avoid memory issues
        if has_fallback_rows and all_processed_rows:
            print(f"Replacing {len(all_processed_rows)} fallback rows in CSV...")
            # Process in chunks of 50 to avoid memory issues
            chunk_size = 50
            for i in range(0, len(all_processed_rows), chunk_size):
                chunk = all_processed_rows[i:i+chunk_size]
                if i == 0:
                    # First chunk does the full replacement
                    DataHandler.replace_rows_in_csv(chunk, output_filepath)
                else:
                    # Subsequent chunks just update existing rows
                    DataHandler.replace_rows_in_csv(chunk, output_filepath)
                print(f"  Processed {min(i+chunk_size, len(all_processed_rows))}/{len(all_processed_rows)} rows...")
            print(f"✅ Replacement complete!")
        
        return successful_count, skipped_count
    
    def _process_all_at_once(self, rows: List[Row], few_shot_list: List, glossary_list: List,
                            output_filepath: str, prompt_key: str, num_workers: int, debug: bool, sequential: bool = False, translation_mode: bool = False, has_fallback_rows: bool = False) -> tuple:
        """Process all rows at once."""
        print("Generating messages for all unprocessed rows...")
        
        # Track timing for this single large batch
        batch_timing = {"batch_id": 1, "rows_processed": len(rows)}
        batch_start_time = time.time()
        
        with self.timing_tracker.timer("message_generation"):
            message_gen_start = time.time()
            messages = [row.get_messages(prompt_key, few_shot_examples, glossary_entries, translation_mode) 
                       for row, few_shot_examples, glossary_entries 
                       in zip(rows, few_shot_list, glossary_list)]
            batch_timing["message_generation"] = time.time() - message_gen_start
        
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
        with self.timing_tracker.timer("llm_inference"):
            inference_start = time.time()
            
            if sequential:
                # Sequential processing - process one by one
                print(f"Sequential processing: Processing {len(messages)} messages one by one...")
                translations = []
                for i, msg_list in enumerate(messages):
                    try:
                        translation = self.llm.generate(msg_list, translation_mode)
                        translations.append(translation)
                        print(f"  Processed message {i+1}/{len(messages)}")
                    except Exception as e:
                        print(f"  Error in message {i+1}: {e}")
                        translations.append("")
            elif self.llm.supports_batch:
                # Check if the model supports incremental saving (Gemini async does)
                if hasattr(self.llm, 'generate_batch') and 'gemini' in self.llm.model_name.lower():
                    if has_fallback_rows:
                        # Don't use incremental saving when replacing fallback rows
                        print(f"Using {self.llm.model_name} native batch processing (no incremental saving for fallback replacement)...")
                        translations = self.llm.generate_batch(messages, rows_data=rows, output_filepath=None, sequential=False, translation_mode=translation_mode)
                    else:
                        # Use incremental saving for normal processing
                        print(f"Using {self.llm.model_name} native batch processing with incremental saving...")
                        translations = self.llm.generate_batch(messages, rows_data=rows, output_filepath=output_filepath, sequential=False, translation_mode=translation_mode)
                else:
                    # For vLLM and other batch-supporting models
                    translations = self.llm.generate_batch(messages, sequential=False, translation_mode=translation_mode)
            else:
                print(f"Using thread-based parallel processing with {num_workers} workers...")
                def process_single_message(msg_list):
                    try:
                        return self.llm.generate(msg_list, translation_mode)
                    except Exception as e:
                        print(f"Error in API call: {e}")
                        return ""
                
                translations = thread_map(process_single_message, messages, 
                                        max_workers=num_workers, 
                                        desc="Generating translations")
            
            batch_timing["llm_inference"] = time.time() - inference_start
        
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
        with self.timing_tracker.timer("data_saving"):
            saving_start = time.time()
            result = self._save_batch_results(rows, translations, output_filepath, translation_mode, has_fallback_rows)
            batch_timing["data_saving"] = time.time() - saving_start
        
        # Record total batch time
        batch_timing["total_batch_time"] = time.time() - batch_start_time
        self.timing_tracker.record_batch_timing(batch_timing)
        print(f"All-at-once processing completed in {batch_timing['total_batch_time']:.2f}s")
        
        return result
    
    def _process_individually(self, rows: List[Row], few_shot_list: List, glossary_list: List,
                             output_filepath: str, prompt_key: str, debug: bool, translation_mode: bool = False, 
                             has_fallback_rows: bool = False) -> tuple:
        """
        Process rows individually as fallback with CSV buffering.
        Uses original pred_text as fallback for failed/invalid translations.
        """
        successful_count = 0
        fallback_count = 0
        
        # Track timing for individual processing (treat as one large batch)
        batch_timing = {"batch_id": 1, "rows_processed": len(rows)}
        batch_start_time = time.time()
        
        # Collect all processed rows instead of using buffer if we need to replace
        all_processed_rows = []
        
        # Use CSV buffer for efficient writing only if not replacing
        csv_buffer = None if has_fallback_rows else DataHandler.create_csv_buffer(output_filepath, buffer_size=25)
        if csv_buffer:
            csv_buffer.__enter__()
            for i, (row, few_shot_examples, glossary_entries) in enumerate(zip(rows, few_shot_list, glossary_list)):
                try:
                    messages = row.get_messages(prompt_key, few_shot_examples, glossary_entries, translation_mode)
                    
                    if self.token_counter:
                        token_count = self.token_counter.count_tokens_in_messages(messages)
                        print(f"Row {i+1} token count: {token_count}")
                    
                    if debug:
                        print(f"\n--- DEBUG: Individual processing row {i+1} ---")
                        for j, msg in enumerate(messages):
                            print(f"Message {j+1} ({msg['role']}):")
                            print(f"{msg['content']}")
                        print("--- End debug message ---\n")
                    
                    translation = self.llm.generate(messages, translation_mode)
                    
                    if translation and translation.strip() and not translation.startswith("["):
                        # Valid translation - use it
                        if translation_mode:
                            row.translated_tgt_txt = translation.strip()
                            row.post_edited_tgt_txt = translation.strip()  # Also populate for compatibility
                        else:
                            row.post_edited_tgt_txt = translation.strip()
                        successful_count += 1
                        
                        # Calculate individual metrics improvements (skip if disabled)
                        if not self.disable_metrics:
                            try:
                                from core.metrics import MetricsCalculator
                                individual_metrics = MetricsCalculator.calculate_individual_metrics(
                                    original_text=row.pred_tgt_text,
                                    post_edited_text=translation.strip(),
                                    reference_text=row.tgt_text
                                )
                                
                                # Store the improvements in the row
                                row.spbleu_improvement = individual_metrics["improvements"]["spbleu"]
                                row.chrf3_improvement = individual_metrics["improvements"]["chrf3"]
                                row.chrfpp_improvement = individual_metrics["improvements"]["chrfpp"]
                                
                            except Exception as e:
                                print(f"Warning: Could not calculate individual metrics for row {i+1}: {e}")
                                # Set metrics to None if calculation fails
                                row.spbleu_improvement = None
                                row.chrf3_improvement = None
                                row.chrfpp_improvement = None
                        else:
                            # Skip metrics calculation for faster processing
                            row.spbleu_improvement = None
                            row.chrf3_improvement = None
                            row.chrfpp_improvement = None
                    else:
                        # Failed/invalid translation - use original pred_text as fallback
                        if translation_mode:
                            row.translated_tgt_txt = row.pred_tgt_text
                            row.post_edited_tgt_txt = row.pred_tgt_text  # Also populate for compatibility
                        else:
                            row.post_edited_tgt_txt = row.pred_tgt_text
                        fallback_count += 1
                        print(f"⚠️ Row {i+1}: Using original pred_text as fallback (API returned: '{translation}')")
                        
                        # For fallback cases, metrics improvements should be 0 (no improvement)
                        if not self.disable_metrics:
                            row.spbleu_improvement = 0.0
                            row.chrf3_improvement = 0.0
                            row.chrfpp_improvement = 0.0
                        else:
                            row.spbleu_improvement = None
                            row.chrf3_improvement = None
                            row.chrfpp_improvement = None
                    
                    # Always add the row (either successful or fallback)
                    if csv_buffer:
                        csv_buffer.add_row(row)
                    else:
                        all_processed_rows.append(row)
                        
                except Exception as row_error:
                    print(f"Error processing row {i+1}: {row_error}")
                    # Even on error, use original pred_text as fallback
                    if translation_mode:
                        row.translated_tgt_txt = row.pred_tgt_text
                        row.post_edited_tgt_txt = row.pred_tgt_text  # Also populate for compatibility
                    else:
                        row.post_edited_tgt_txt = row.pred_tgt_text
                    fallback_count += 1
                    
                    # Set metrics to zero for error cases
                    if not self.disable_metrics:
                        row.spbleu_improvement = 0.0
                        row.chrf3_improvement = 0.0
                        row.chrfpp_improvement = 0.0
                    else:
                        row.spbleu_improvement = None
                        row.chrf3_improvement = None
                        row.chrfpp_improvement = None
                    
                    # Still add the row with fallback
                    if csv_buffer:
                        csv_buffer.add_row(row)
                    else:
                        all_processed_rows.append(row)
        
        # Close buffer if used
        if csv_buffer:
            csv_buffer.__exit__(None, None, None)
        
        # If we collected rows for replacement, save them now
        if has_fallback_rows and all_processed_rows:
            print(f"Replacing fallback rows in CSV...")
            DataHandler.replace_rows_in_csv(all_processed_rows, output_filepath)
        
        # Record total individual processing time
        batch_timing["total_batch_time"] = time.time() - batch_start_time
        batch_timing["llm_inference"] = batch_timing["total_batch_time"]  # Most time is inference for individual
        self.timing_tracker.record_batch_timing(batch_timing)
        print(f"Individual processing completed in {batch_timing['total_batch_time']:.2f}s")
        
        return successful_count, fallback_count
    
    def _save_batch_results(self, rows: List[Row], translations: List[str], 
                           output_filepath: str, translation_mode: bool = False, has_fallback_rows: bool = False) -> tuple:
        """
        Save batch results to file efficiently using batch processing.
        Uses original pred_text as fallback for failed/invalid translations to ensure 
        we always have the same number of output rows as input rows.
        """
        if len(rows) != len(translations):
            print(f"Warning: Mismatch between rows ({len(rows)}) and translations ({len(translations)})")
            # Pad with empty strings if needed
            while len(translations) < len(rows):
                translations.append("")
        
        all_processed_rows = []
        successful_count = 0
        fallback_count = 0
        
        # Process all rows - never skip any
        valid_data = []
        for i, (row, translation) in enumerate(zip(rows, translations)):
            if translation and translation.strip() and not translation.startswith("["):
                # Valid translation - use it
                if translation_mode:
                    row.translated_tgt_txt = translation.strip()
                    row.post_edited_tgt_txt = translation.strip()  # Also populate for compatibility
                else:
                    row.post_edited_tgt_txt = translation.strip()
                successful_count += 1
                valid_data.append((row.pred_tgt_text, translation.strip(), row.tgt_text))
            else:
                # Failed/invalid translation - use original pred_text as fallback
                if translation_mode:
                    row.translated_tgt_txt = row.pred_tgt_text
                    row.post_edited_tgt_txt = row.pred_tgt_text  # Also populate for compatibility
                else:
                    row.post_edited_tgt_txt = row.pred_tgt_text  # Use original prediction as fallback
                fallback_count += 1
                print(f"  ⚠️ Row {i+1}: Using original pred_text as fallback (API returned: '{translation}')")
                # For fallback, we'll set metrics to zero (no improvement)
                valid_data.append((row.pred_tgt_text, row.pred_tgt_text, row.tgt_text))
            
            all_processed_rows.append(row)
        
        # Batch calculate metrics for all rows (skip if disabled)
        if valid_data and not self.disable_metrics:
            with self.timing_tracker.timer("individual_metrics_calculation"):
                try:
                    from core.metrics import MetricsCalculator
                    print(f"Calculating metrics for {len(valid_data)} rows in batch ({successful_count} successful, {fallback_count} fallback)...")
                    batch_metrics = MetricsCalculator.calculate_batch_individual_metrics(valid_data)
                    
                    # Apply metrics to all rows (including fallback ones)
                    for row, metrics in zip(all_processed_rows, batch_metrics):
                        row.spbleu_improvement = metrics["improvements"]["spbleu"]
                        row.chrf3_improvement = metrics["improvements"]["chrf3"]
                        row.chrfpp_improvement = metrics["improvements"]["chrfpp"]
                        
                except Exception as e:
                    print(f"Warning: Batch metrics calculation failed, falling back to individual: {e}")
                    # Fallback to individual calculation for all rows
                    for row in all_processed_rows:
                        try:
                            individual_metrics = MetricsCalculator.calculate_individual_metrics(
                                original_text=row.pred_tgt_text,
                                post_edited_text=row.post_edited_tgt_txt,
                                reference_text=row.tgt_text
                            )
                            row.spbleu_improvement = individual_metrics["improvements"]["spbleu"]
                            row.chrf3_improvement = individual_metrics["improvements"]["chrf3"]
                            row.chrfpp_improvement = individual_metrics["improvements"]["chrfpp"]
                        except Exception as row_e:
                            print(f"Warning: Individual metrics failed for row: {row_e}")
                            row.spbleu_improvement = None
                            row.chrf3_improvement = None
                            row.chrfpp_improvement = None
        elif self.disable_metrics:
            # Set metrics to None when disabled for faster processing
            print(f"Metrics calculation disabled - setting {len(all_processed_rows)} rows to None metrics")
            for row in all_processed_rows:
                # Ensure metrics are None, not empty strings
                row.spbleu_improvement = None
                row.chrf3_improvement = None
                row.chrfpp_improvement = None
                # Also handle existing empty string values
                if hasattr(row, 'spbleu_improvement') and row.spbleu_improvement == '':
                    row.spbleu_improvement = None
                if hasattr(row, 'chrf3_improvement') and row.chrf3_improvement == '':
                    row.chrf3_improvement = None
                if hasattr(row, 'chrfpp_improvement') and row.chrfpp_improvement == '':
                    row.chrfpp_improvement = None
        
        # Batch write ALL processed rows to CSV (successful + fallback)
        if all_processed_rows:
            with self.timing_tracker.timer("data_saving"):
                if has_fallback_rows:
                    # Replace existing rows instead of appending
                    print(f"Replacing {len(all_processed_rows)} rows in CSV ({successful_count} successful, {fallback_count} fallback)...")
                    DataHandler.replace_rows_in_csv(all_processed_rows, output_filepath)
                else:
                    # Normal append behavior
                    print(f"Writing {len(all_processed_rows)} rows to CSV in batch ({successful_count} successful, {fallback_count} fallback)...")
                    DataHandler.batch_append_to_csv(all_processed_rows, output_filepath)
        
        return successful_count, fallback_count
    
    def _print_token_statistics(self, token_counts: List[int]):
        """Print token counting statistics."""
        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
            max_tokens = max(token_counts)
            min_tokens = min(token_counts)
            print(f"Token statistics - Average: {avg_tokens:.1f}, Min: {min_tokens}, Max: {max_tokens}")
    
    def _calculate_final_metrics(self, output_filepath: str) -> dict:
        """Calculate final metrics from completed results."""
        if self.disable_metrics:
            print("Individual metrics calculation was disabled - calculating FINAL summary metrics only")
        else:
            print("Calculating final metrics from completed results with individual metrics")
            
        if os.path.exists(output_filepath):
            try:
                completed_rows = DataHandler.load_from_savepath(output_filepath)
                # Filter out rows with empty predictions for metrics calculation
                valid_rows = [row for row in completed_rows 
                             if row.post_edited_tgt_txt and row.post_edited_tgt_txt.strip()]
                print(f"Calculating final summary metrics for {len(valid_rows)} valid completed rows")
                
                if valid_rows:
                    # Calculate final summary metrics regardless of disable_metrics flag
                    metrics = MetricsCalculator.calculate_metrics(valid_rows)
                    
                    # Add processing info if individual metrics were disabled
                    if self.disable_metrics:
                        metrics["processing_info"] = {
                            "individual_metrics_disabled": True,
                            "note": "Individual row metrics were disabled for faster processing. Summary metrics calculated from final results."
                        }
                    
                    # Save metrics to JSON file
                    metrics_filepath = output_filepath.replace('.csv', '_metrics.json')
                    with open(metrics_filepath, 'w', encoding='utf-8') as f:
                        json.dump(metrics, f, indent=2, ensure_ascii=False)
                    print(f"Final summary metrics saved to: {metrics_filepath}")
                    
                    return metrics
                else:
                    print("No valid rows found for metrics calculation")
                    return {}
                    
            except Exception as e:
                print(f"Warning: Could not load results for final metrics calculation: {e}")
                # If we can't parse the CSV (e.g., due to empty strings), create basic summary
                if self.disable_metrics:
                    try:
                        import pandas as pd
                        df = pd.read_csv(output_filepath)
                        valid_rows = len(df[df['post_edited_tgt_txt'].notna() & (df['post_edited_tgt_txt'] != '')])
                        total_rows = len(df)
                        
                        summary = {
                            "processing_summary": {
                                "total_rows": total_rows,
                                "successfully_processed": valid_rows,
                                "skipped_rows": total_rows - valid_rows,
                                "individual_metrics_disabled": True
                            },
                            "note": "Could not calculate summary metrics due to CSV parsing issues. Individual metrics were disabled for faster processing."
                        }
                        
                        # Save summary to JSON file
                        summary_filepath = output_filepath.replace('.csv', '_metrics.json')
                        with open(summary_filepath, 'w', encoding='utf-8') as f:
                            json.dump(summary, f, indent=2, ensure_ascii=False)
                        print(f"Basic processing summary saved to: {summary_filepath}")
                        
                        return summary
                    except Exception as fallback_error:
                        print(f"Warning: Could not create even basic summary: {fallback_error}")
                return {}
        
        print("No completed rows found. Cannot calculate metrics.")
        return {}

