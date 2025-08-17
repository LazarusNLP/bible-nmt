#!/usr/bin/env python3
"""
Script to create aligned Indonesian Bible translation datasets from source and target text files.
Handles verse ranges where multiple verses are combined into single entries.
"""

import argparse
import os
from typing import List, Tuple, Dict
from datasets import Dataset, DatasetDict


def load_corpus_files(source_name: str, target_name: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Load source, target, and verse reference files.
    
    Args:
        source_name: Name of source file (e.g., 'ind-indyt.txt')
        target_name: Name of target file (e.g., 'aaz-aaz.txt')
    
    Returns:
        Tuple of (source_lines, target_lines, vref_lines)
    """
    corpus_path = "/data/gpfs/projects/punim0478/setiawand/ebible/corpus"
    metadata_path = "/data/gpfs/projects/punim0478/setiawand/ebible/metadata"
    
    source_path = os.path.join(corpus_path, source_name)
    target_path = os.path.join(corpus_path, target_name)
    vref_path = os.path.join(metadata_path, "vref.txt")
    
    with open(source_path, 'r', encoding='utf-8') as f:
        source_lines = f.readlines()
    
    with open(target_path, 'r', encoding='utf-8') as f:
        target_lines = f.readlines()
    
    with open(vref_path, 'r', encoding='utf-8') as f:
        vref_lines = f.readlines()
    
    # Strip newlines but preserve content
    source_lines = [line.rstrip('\n') for line in source_lines]
    target_lines = [line.rstrip('\n') for line in target_lines]
    vref_lines = [line.rstrip('\n') for line in vref_lines]
    
    return source_lines, target_lines, vref_lines


def create_verse_reference(vref_lines: List[str], start_idx: int, end_idx: int) -> str:
    """
    Create a verse reference string for a single verse or range.
    
    Args:
        vref_lines: List of verse references
        start_idx: Starting index (0-based)
        end_idx: Ending index (0-based)
    
    Returns:
        Formatted verse reference (e.g., "GEN 1:14" or "GEN 1:14-17")
    """
    start_ref = vref_lines[start_idx].strip()
    
    if start_idx == end_idx:
        return start_ref
    
    # Extract book, chapter, and verse components
    end_ref = vref_lines[end_idx].strip()
    
    # Parse start reference (e.g., "GEN 1:14")
    start_parts = start_ref.split()
    if len(start_parts) >= 2:
        book = start_parts[0]
        start_chapter_verse = start_parts[1]
        
        # Parse end reference
        end_parts = end_ref.split()
        if len(end_parts) >= 2:
            end_chapter_verse = end_parts[1]
            
            # Check if same chapter
            start_chapter = start_chapter_verse.split(':')[0]
            end_chapter = end_chapter_verse.split(':')[0]
            
            if start_chapter == end_chapter:
                # Same chapter, format as "BOOK CH:V1-V2"
                start_verse = start_chapter_verse.split(':')[1] if ':' in start_chapter_verse else start_chapter_verse
                end_verse = end_chapter_verse.split(':')[1] if ':' in end_chapter_verse else end_chapter_verse
                return f"{book} {start_chapter}:{start_verse}-{end_verse}"
            else:
                # Different chapters, format as "BOOK CH1:V1-CH2:V2"
                return f"{book} {start_chapter_verse}-{end_chapter_verse}"
    
    # Fallback to simple format
    return f"{start_ref}-{end_ref.split()[-1]}"


def identify_verse_ranges(target_lines: List[str]) -> List[Tuple[int, int]]:
    """
    Identify verse ranges in target text by finding <range> tokens.
    When <range> appears on line N, line N-1 contains combined text for verses N-1 and N.
    
    Args:
        target_lines: List of target text lines
    
    Returns:
        List of tuples (start_idx, end_idx) for each range
    """
    ranges = []
    i = 0
    
    while i < len(target_lines):
        # Check if next line(s) contain <range> token
        if i + 1 < len(target_lines) and target_lines[i + 1] == '<range>':
            # Found start of a range at index i
            start_idx = i
            end_idx = i + 1
            
            # Find consecutive <range> tokens
            while end_idx + 1 < len(target_lines) and target_lines[end_idx + 1] == '<range>':
                end_idx += 1
            
            ranges.append((start_idx, end_idx))
            print(f"  Found range: lines {start_idx}-{end_idx} (verses {start_idx+1} to {end_idx+1})")
            i = end_idx + 1
        else:
            i += 1
    
    print(f"  Total ranges identified: {len(ranges)}")
    return ranges


def align_texts(source_lines: List[str], target_lines: List[str], vref_lines: List[str],
                source_lang: str, target_lang: str) -> List[Dict[str, str]]:
    """
    Align source and target texts, handling verse ranges.
    When <range> appears on line N, line N-1 contains the combined text for verses N-1 and N.
    Skip any verses where either source or target is blank/empty.
    
    Args:
        source_lines: List of source text lines
        target_lines: List of target text lines
        vref_lines: List of verse references
        source_lang: Source language code
        target_lang: Target language code
    
    Returns:
        List of aligned dataset entries
    """
    aligned_data = []
    ranges = identify_verse_ranges(target_lines)
    range_dict = {r[0]: r for r in ranges}  # Map start index to range
    processed_indices = set()
    skipped_count = 0
    
    for idx in range(len(target_lines)):
        if idx in processed_indices:
            continue
        
        if target_lines[idx] == '<range>':
            # Skip <range> tokens themselves
            processed_indices.add(idx)
            continue
        
        if idx in range_dict:
            # This is the start of a range
            start_idx, end_idx = range_dict[idx]
            
            # Check if any line in the range is empty (skip entire range if so)
            skip_range = False
            for i in range(start_idx, end_idx + 1):
                if i < len(source_lines) and i < len(target_lines):
                    # Skip if source is empty/blank (and not <range>)
                    if source_lines[i] != '<range>' and not source_lines[i].strip():
                        skip_range = True
                        break
                    # Skip if target is empty/blank (and not <range>)
                    if target_lines[i] != '<range>' and not target_lines[i].strip():
                        skip_range = True
                        break
            
            if skip_range:
                skipped_count += 1
                # Mark all indices in range as processed
                for i in range(start_idx, end_idx + 1):
                    processed_indices.add(i)
                continue
            
            # Concatenate source lines for the range (excluding <range> tokens and empty lines)
            source_texts = []
            for i in range(start_idx, end_idx + 1):
                if i < len(source_lines) and source_lines[i] != '<range>' and source_lines[i].strip():
                    source_texts.append(source_lines[i])
            
            source_text = ' '.join(source_texts)
            
            # Target text is at the start index (contains combined text)
            target_text = target_lines[start_idx]
            
            # Final check: only add if both source and target are non-empty
            if source_text.strip() and target_text.strip():
                # Create verse reference for the range
                verse_ref = create_verse_reference(vref_lines, start_idx, end_idx)
                
                aligned_data.append({
                    'source_text': source_text,
                    'target_text': target_text,
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'verse': verse_ref
                })
            else:
                skipped_count += 1
            
            # Mark all indices in range as processed
            for i in range(start_idx, end_idx + 1):
                processed_indices.add(i)
        else:
            # Regular single verse
            if idx < len(source_lines) and idx < len(target_lines) and idx < len(vref_lines):
                source_text = source_lines[idx].strip()
                target_text = target_lines[idx].strip()
                verse_ref = vref_lines[idx].strip() if vref_lines[idx] else f"VERSE_{idx+1}"
                
                # Only add if BOTH source and target are non-empty
                if source_text and target_text:
                    aligned_data.append({
                        'source_text': source_text,
                        'target_text': target_text,
                        'source_lang': source_lang,
                        'target_lang': target_lang,
                        'verse': verse_ref
                    })
                elif not source_text or not target_text:
                    # Count as skipped if either is empty
                    skipped_count += 1
                    
                processed_indices.add(idx)
    
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} verses/ranges due to blank lines")
    
    return aligned_data


def save_as_hf_dataset(aligned_data: List[Dict[str, str]], source_lang: str, target_lang: str, output_dir: str = "."):
    """
    Save aligned data as a HuggingFace dataset.
    
    Args:
        aligned_data: List of aligned dataset entries
        source_lang: Source language code
        target_lang: Target language code
        output_dir: Directory to save the dataset
    """
    # Create dataset from aligned data
    dataset = Dataset.from_list(aligned_data)
    
    # Create dataset dictionary with train split
    dataset_dict = DatasetDict({
        'train': dataset
    })
    
    # Save with the specified naming format
    dataset_name = f"{source_lang}_{target_lang}"
    save_path = os.path.join(output_dir, dataset_name)
    
    dataset_dict.save_to_disk(save_path)
    print(f"Dataset saved to: {save_path}")
    print(f"Total aligned pairs: {len(aligned_data)}")
    
    # Print sample entries
    if aligned_data:
        print("\nSample entries:")
        for i, entry in enumerate(aligned_data[:3]):
            print(f"\nEntry {i+1}:")
            print(f"  Verse: {entry['verse']}")
            print(f"  Source ({entry['source_lang']}): {entry['source_text'][:100]}...")
            print(f"  Target ({entry['target_lang']}): {entry['target_text'][:100]}...")


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description='Create aligned Indonesian Bible translation datasets with verse range handling'
    )
    parser.add_argument('--source_name', required=True,
                        help='Source file name (e.g., ind-indyt.txt)')
    parser.add_argument('--target_name', required=True,
                        help='Target file name (e.g., aaz-aaz.txt)')
    parser.add_argument('--source_lang', required=True,
                        help='Source language code (e.g., ind)')
    parser.add_argument('--target_lang', required=True,
                        help='Target language code (e.g., aaz)')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for the dataset (default: ./data)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Creating aligned dataset: {args.source_lang} -> {args.target_lang}")
    print("=" * 60)
    print(f"Source file: {args.source_name}")
    print(f"Target file: {args.target_name}")
    print(f"Output dir:  {args.output_dir}")
    print("-" * 60)
    
    # Load files
    print("\n1. Loading corpus files...")
    source_lines, target_lines, vref_lines = load_corpus_files(
        args.source_name, args.target_name
    )
    
    print(f"  Loaded {len(source_lines)} source lines")
    print(f"  Loaded {len(target_lines)} target lines")
    print(f"  Loaded {len(vref_lines)} verse references")
    
    # Verify file lengths
    if len(source_lines) != len(target_lines):
        print(f"  WARNING: Source and target have different lengths!")
    if len(vref_lines) != len(source_lines) and len(vref_lines) != len(target_lines):
        print(f"  WARNING: Vref length doesn't match source/target!")
    
    # Align texts
    print("\n2. Identifying verse ranges and aligning texts...")
    aligned_data = align_texts(
        source_lines, target_lines, vref_lines,
        args.source_lang, args.target_lang
    )
    print(f"  Created {len(aligned_data)} aligned text pairs")
    
    # Save as HuggingFace dataset
    print("\n3. Saving as HuggingFace dataset...")
    save_as_hf_dataset(aligned_data, args.source_lang, args.target_lang, args.output_dir)
    
    print("-" * 60)
    print("Dataset creation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()