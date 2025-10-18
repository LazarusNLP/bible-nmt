"""
Data input/output operations for post-editing pipeline.
"""

import csv
import os
from pathlib import Path
from typing import List, Union, Optional
from core.data_models import Row, GlossaryEntry


class CSVBuffer:
    """Efficient CSV buffer for batch writing rows to avoid frequent file I/O."""
    
    def __init__(self, filepath: str, buffer_size: int = 50):
        """
        Initialize CSV buffer.
        
        Args:
            filepath: Path to CSV file
            buffer_size: Number of rows to buffer before flushing to disk
        """
        self.filepath = filepath
        self.buffer = []
        self.buffer_size = buffer_size
        self.file_exists = os.path.exists(filepath)
        
        # Ensure parent directory exists
        os.makedirs(Path(filepath).parent, exist_ok=True)
    
    def add_row(self, row: Row):
        """Add a row to the buffer, flushing if buffer is full."""
        self.buffer.append(row)
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def add_rows(self, rows: List[Row]):
        """Add multiple rows to the buffer."""
        for row in rows:
            self.add_row(row)
    
    def flush(self):
        """Write all buffered rows to file with sanitization."""
        if not self.buffer:
            return
        
        # Sanitize all buffered rows before writing
        for row in self.buffer:
            row.src_text = DataHandler.sanitize_text_for_csv(row.src_text)
            row.tgt_text = DataHandler.sanitize_text_for_csv(row.tgt_text) 
            row.pred_tgt_text = DataHandler.sanitize_text_for_csv(row.pred_tgt_text)
            row.post_edited_tgt_txt = DataHandler.sanitize_text_for_csv(row.post_edited_tgt_txt)
        
        with open(self.filepath, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=Row.model_fields.keys(), 
                                  quoting=csv.QUOTE_ALL, quotechar='"')  # Use QUOTE_ALL for safety
            
            # Write header only if file doesn't exist
            if not self.file_exists:
                writer.writeheader()
                self.file_exists = True
            
            # Write all buffered rows at once
            for row in self.buffer:
                writer.writerow(row.model_dump())
        
        print(f"Flushed {len(self.buffer)} rows to {self.filepath}")
        self.buffer.clear()
    
    def close(self):
        """Flush any remaining rows and close buffer."""
        self.flush()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures flush on cleanup."""
        self.close()


class DataHandler:
    """Handles data loading and saving operations."""
    
    @staticmethod
    def load_from_csv(csv_path: str, src_lang: str, tgt_lang: str, 
                      src_lang_name: str, tgt_lang_name: str, 
                      max_samples: int = None) -> List[Row]:
        """Load data from CSV file containing source_text, target_text, and pred_text columns."""
        print(f"Loading data from {csv_path}")
        if max_samples:
            print(f"Limiting to first {max_samples} samples")
        
        rows = []
        # Detect delimiter - TSV files use tabs, CSV files use commas
        delimiter = '\t' if csv_path.lower().endswith('.tsv') else ','
        
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            for i, row_data in enumerate(reader):
                # Check if we've reached the maximum number of samples
                if max_samples and len(rows) >= max_samples:
                    break
                    
                # Check for required columns - be flexible with pred column name
                if 'source_text' not in row_data or 'target_text' not in row_data:
                    raise ValueError(f"CSV file must contain 'source_text' and 'target_text' columns")
                
                # Handle different pred column names - pred_text is the standard
                pred_col = None
                if 'pred_text' in row_data:
                    pred_col = 'pred_text'
                elif 'pred_tgt_text' in row_data:
                    pred_col = 'pred_tgt_text'
                elif 'pred_target_text' in row_data:
                    pred_col = 'pred_target_text'
                else:
                    raise ValueError(f"CSV file must contain one of: 'pred_text', 'pred_tgt_text', or 'pred_target_text' columns")
                
                src_text = row_data['source_text'].strip()
                tgt_text = row_data['target_text'].strip()
                pred_tgt_text = row_data[pred_col].strip()
                
                # Skip empty lines
                if src_text and tgt_text and pred_tgt_text:
                    rows.append(Row(
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

    @staticmethod
    def load_from_savepath(filepath: str) -> List[Row]:
        """Load rows from existing CSV output file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [Row.model_validate(row) for row in reader]
    
    @staticmethod
    def save_to_csv(rows: List[Row], filepath: str):
        """Save rows to CSV file."""
        print(f"Saving {len(rows)} rows to {filepath}")
        os.makedirs(Path(filepath).parent, exist_ok=True)
        with open(file=filepath, mode="w", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=Row.model_fields.keys(), 
                                  quoting=csv.QUOTE_MINIMAL, quotechar='"')
            writer.writeheader()
            for row in rows:
                writer.writerow(row.model_dump())
    
    @staticmethod
    def sanitize_text_for_csv(text: str) -> str:
        """Sanitize text to prevent CSV corruption."""
        if not text:
            return text
            
        # Remove control characters that could break CSV parsing
        import re
        # Remove control characters except tabs, newlines, and carriage returns
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Replace problematic newlines with spaces (preserve intentional line breaks in JSON context)
        if text.count('\n') > 2:  # More than 2 newlines suggests unwanted content
            text = text.replace('\n', ' ').replace('\r', ' ')
            # Clean up multiple spaces
            text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    @staticmethod
    def append_row_to_csv(row: Row, filepath: str):
        """Append a single row to the CSV file with sanitization."""
        # Ensure parent directory exists
        os.makedirs(Path(filepath).parent, exist_ok=True)
        file_exists = os.path.exists(filepath)
        
        # Sanitize all text fields to prevent CSV corruption
        row.src_text = DataHandler.sanitize_text_for_csv(row.src_text)
        row.tgt_text = DataHandler.sanitize_text_for_csv(row.tgt_text) 
        row.pred_tgt_text = DataHandler.sanitize_text_for_csv(row.pred_tgt_text)
        row.post_edited_tgt_txt = DataHandler.sanitize_text_for_csv(row.post_edited_tgt_txt)
        
        with open(filepath, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=Row.model_fields.keys(), 
                                  quoting=csv.QUOTE_ALL, quotechar='"')  # Use QUOTE_ALL for safety
            
            # Write header only if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row.model_dump())
    
    @staticmethod
    def batch_append_to_csv(rows: List[Row], filepath: str):
        """Efficiently append multiple rows to CSV file in one operation with sanitization."""
        if not rows:
            return
        
        # Ensure parent directory exists
        os.makedirs(Path(filepath).parent, exist_ok=True)
        file_exists = os.path.exists(filepath)
        
        # Sanitize all rows before writing
        for row in rows:
            row.src_text = DataHandler.sanitize_text_for_csv(row.src_text)
            row.tgt_text = DataHandler.sanitize_text_for_csv(row.tgt_text) 
            row.pred_tgt_text = DataHandler.sanitize_text_for_csv(row.pred_tgt_text)
            row.post_edited_tgt_txt = DataHandler.sanitize_text_for_csv(row.post_edited_tgt_txt)
        
        with open(filepath, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=Row.model_fields.keys(), 
                                  quoting=csv.QUOTE_ALL, quotechar='"')  # Use QUOTE_ALL for safety
            
            # Write header only if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            # Write all rows at once
            for row in rows:
                writer.writerow(row.model_dump())
        
        print(f"Batch wrote {len(rows)} rows to {filepath}")
    
    @staticmethod
    def replace_rows_in_csv(new_rows: List[Row], filepath: str):
        """
        Replace rows in CSV file by merging with existing rows.
        Uses src_text as the unique identifier to match and replace rows.
        This is useful for replacing fallback rows with successfully processed ones.
        """
        if not new_rows:
            return
        
        # Load existing rows if file exists
        existing_rows = []
        if os.path.exists(filepath):
            existing_rows = DataHandler.load_from_savepath(filepath)
            print(f"Loaded {len(existing_rows)} existing rows from {filepath}")
        
        # Create a mapping of src_text to new rows for fast lookup
        new_rows_map = {row.src_text: row for row in new_rows}
        
        # Build the final list: replace existing rows with new ones where available
        final_rows = []
        replaced_count = 0
        
        for existing_row in existing_rows:
            if existing_row.src_text in new_rows_map:
                # Replace with new row
                final_rows.append(new_rows_map[existing_row.src_text])
                replaced_count += 1
                # Remove from map so we can track which rows are truly new
                del new_rows_map[existing_row.src_text]
            else:
                # Keep existing row
                final_rows.append(existing_row)
        
        # Add any remaining new rows that weren't replacements
        if new_rows_map:
            final_rows.extend(new_rows_map.values())
            print(f"Added {len(new_rows_map)} new rows")
        
        # Now write all rows back to the file
        print(f"Replacing {replaced_count} rows in {filepath}")
        DataHandler.save_to_csv(final_rows, filepath)
    
    @staticmethod
    def create_csv_buffer(filepath: str, buffer_size: int = 50) -> CSVBuffer:
        """Create a CSV buffer for efficient batch writing."""
        return CSVBuffer(filepath, buffer_size)

    @staticmethod
    def find_unprocessed_rows(all_rows: List[Row], output_filepath: str) -> tuple:
        """
        Find rows that haven't been processed yet by comparing with existing CSV file.
        Treats rows where post_edited_tgt_txt == pred_tgt_text as unprocessed (likely fallback cases).
        
        Returns:
            Tuple of (unprocessed_rows, has_fallback_rows)
            - unprocessed_rows: List of rows that need processing
            - has_fallback_rows: True if any fallback rows were found that need replacement
        """
        if not os.path.exists(output_filepath):
            print("No existing output file found. Processing all rows.")
            return all_rows, False
        
        try:
            existing_rows = DataHandler.load_from_savepath(output_filepath)
            print(f"Found existing output file with {len(existing_rows)} completed rows")
            
            # Create a set of successfully completed row identifiers (using src_text as unique identifier)
            # A row is considered "successfully completed" if:
            # 1. It has post_edited_tgt_txt (not None/empty)
            # 2. post_edited_tgt_txt is different from pred_tgt_text (not a fallback case)
            successfully_completed_src_texts = set()
            fallback_count = 0
            
            for row in existing_rows:
                if (row.post_edited_tgt_txt and 
                    row.post_edited_tgt_txt.strip() and 
                    row.post_edited_tgt_txt.strip() != row.pred_tgt_text.strip()):
                    # This row was successfully post-edited (not a fallback)
                    successfully_completed_src_texts.add(row.src_text)
                elif (row.post_edited_tgt_txt and 
                      row.post_edited_tgt_txt.strip() == row.pred_tgt_text.strip()):
                    # This row used fallback (post_edited == original pred) - needs reprocessing
                    fallback_count += 1
            
            # Find unprocessed rows (including fallback cases that need retry)
            unprocessed_rows = [row for row in all_rows if row.src_text not in successfully_completed_src_texts]
            
            successful_count = len(successfully_completed_src_texts)
            total_existing = len(existing_rows)
            has_fallback = fallback_count > 0
            
            print(f"Existing CSV analysis:")
            print(f"  - {successful_count} successfully post-edited rows")
            print(f"  - {fallback_count} fallback rows (will be reprocessed)")
            print(f"  - {total_existing - successful_count - fallback_count} other rows")
            print(f"Found {len(unprocessed_rows)} rows to process (including {fallback_count} fallback retries)")
            
            return unprocessed_rows, has_fallback
        except Exception as e:
            print(f"Error reading existing file: {e}. Processing all rows.")
            return all_rows, False

    @staticmethod
    def load_few_shot_corpus(corpus_paths: Union[str, List[str]]) -> List[tuple]:
        """Load few-shot corpus from text file(s) or CSV file(s) with parallel source/target text.
        
        Args:
            corpus_paths: Single path (str) or list of paths to corpus files
            
        Returns:
            List of tuples containing (source_text, target_text) pairs
        """
        # Convert single path to list for uniform processing
        if isinstance(corpus_paths, str):
            corpus_paths = [corpus_paths]
        
        print(f"Loading few-shot corpus from {len(corpus_paths)} file(s): {corpus_paths}")
        
        all_corpus = []
        
        for corpus_path in corpus_paths:
            print(f"Processing file: {corpus_path}")
            
            # Check if file is CSV or TXT based on extension
            if corpus_path.lower().endswith('.csv'):
                corpus = []
                with open(corpus_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    required_columns = ['source_text', 'target_text']
                    missing_columns = [col for col in required_columns if col not in reader.fieldnames]
                    if missing_columns:
                        raise ValueError(f"CSV file {corpus_path} must contain {required_columns} columns. Missing: {missing_columns}. Found columns: {reader.fieldnames}")
                    
                    for row in reader:
                        src_text = row['source_text'].strip()
                        tgt_text = row['target_text'].strip()
                        if src_text and tgt_text:  # Skip empty lines
                            corpus.append((src_text, tgt_text))
                print(f"Loaded {len(corpus)} parallel sentence pairs from {corpus_path}")
            else:
                # Default behavior for text files - treat each line as source text, duplicate as target
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    corpus = [(line, line) for line in lines]  # Use same text for both source and target
                print(f"Loaded {len(corpus)} sentences from {corpus_path} (duplicated as source-target pairs)")
            
            all_corpus.extend(corpus)
        
        print(f"Total few-shot corpus size: {len(all_corpus)} parallel sentence pairs")
        return all_corpus

    @staticmethod
    def load_glossary(glossary_path: str) -> List[GlossaryEntry]:
        """Load glossary from CSV file with source_word, target_word, and optional pos columns."""
        print(f"Loading glossary from {glossary_path}")
        
        glossary = []
        with open(glossary_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            required_columns = ['source_word', 'target_word']
            missing_columns = [col for col in required_columns if col not in reader.fieldnames]
            if missing_columns:
                raise ValueError(f"Glossary CSV file must contain {required_columns} columns. Missing: {missing_columns}. Found columns: {reader.fieldnames}")
            
            for row in reader:
                source_word = row['source_word'].strip()
                target_word = row['target_word'].strip()
                pos_tag = row.get('pos', '').strip() if 'pos' in row else None
                
                if source_word and target_word:  # Skip empty entries
                    glossary.append(GlossaryEntry(
                        source_word=source_word,
                        target_word=target_word,
                        pos_tag=pos_tag if pos_tag else None
                    ))
        
        print(f"Loaded {len(glossary)} glossary entries")
        return glossary
