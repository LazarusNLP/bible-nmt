"""
Data input/output operations for post-editing pipeline.
"""

import csv
import os
from pathlib import Path
from typing import List
from core.data_models import Row, GlossaryEntry


class DataHandler:
    """Handles data loading and saving operations."""
    
    @staticmethod
    def load_from_csv(csv_path: str, src_lang: str, tgt_lang: str, 
                      src_lang_name: str, tgt_lang_name: str, 
                      max_samples: int = None) -> List[Row]:
        """Load data from CSV file containing source_text, target_text, and pred_target_text columns."""
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
                if 'source_text' not in row_data or 'target_text' not in row_data or 'pred_text' not in row_data:
                    raise ValueError(f"CSV file must contain 'source_text', 'target_text', and 'pred_text' columns")
                
                src_text = row_data['source_text'].strip()
                tgt_text = row_data['target_text'].strip()
                pred_tgt_text = row_data['pred_text'].strip()
                
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
    def append_row_to_csv(row: Row, filepath: str):
        """Append a single row to the CSV file."""
        # Ensure parent directory exists
        os.makedirs(Path(filepath).parent, exist_ok=True)
        file_exists = os.path.exists(filepath)
        
        with open(filepath, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=Row.model_fields.keys(), 
                                  quoting=csv.QUOTE_MINIMAL, quotechar='"')
            
            # Write header only if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row.model_dump())

    @staticmethod
    def find_unprocessed_rows(all_rows: List[Row], output_filepath: str) -> List[Row]:
        """Find rows that haven't been processed yet by comparing with existing CSV file."""
        if not os.path.exists(output_filepath):
            print("No existing output file found. Processing all rows.")
            return all_rows
        
        try:
            existing_rows = DataHandler.load_from_savepath(output_filepath)
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

    @staticmethod
    def load_few_shot_corpus(corpus_path: str) -> List[tuple]:
        """Load few-shot corpus from text file or CSV file with parallel source/target text."""
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
