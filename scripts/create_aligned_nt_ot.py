#!/usr/bin/env python3
"""
Create aligned Bible corpus from two corpus files (source and target) with improved range handling.

Creates CSV files with three columns: 'verse', 'source_text', 'target_text'.
Separates Old Testament (OT) and New Testament (NT) data based on verse book codes.

IMPROVED VERSION: Handles mismatched range markers between source and target corpora.

Notes on <range> markers (based on project loaders):
- Lines with the literal token '<range>' indicate a continuation of the previous verse,
  i.e., the previous line contains the merged text for a span of verses.
- For such spans we keep one row with the combined text and a verse reference span
  like 'GEN 1:1-3'.
- This version detects ranges in BOTH source and target and handles mismatches gracefully.
"""

import argparse
import os
import csv
from typing import List, Tuple, Dict, Set
from pathlib import Path
import sys

# Add the current directory to Python path to import normalize_text
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from normalize_text import normalize_text


# New Testament 3-letter book codes (copied from src/run_translation_v3.py)
NT_BOOKS = [
    "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO",
    "GAL", "EPH", "PHP", "COL", "1TH", "2TH", "1TI", "2TI",
    "TIT", "PHM", "HEB", "JAB", "JAS", "1PE", "2PE", "1JN",
    "2JN", "3JN", "JUD", "REV",
]


def load_lines(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def identify_verse_ranges(text_lines: List[str]) -> List[Tuple[int, int]]:
    """
    Identify verse ranges by finding consecutive '<range>' lines following a content line.
    Returns list of (start_idx, end_idx) indices where 'start_idx' holds the combined text
    and start_idx+1..end_idx are '<range>' markers.
    """
    ranges: List[Tuple[int, int]] = []
    i = 0
    n = len(text_lines)
    while i < n:
        # If next line is a <range>, this is the start of a span
        if i + 1 < n and text_lines[i + 1] == "<range>":
            start_idx = i
            end_idx = i + 1
            while end_idx + 1 < n and text_lines[end_idx + 1] == "<range>":
                end_idx += 1
            ranges.append((start_idx, end_idx))
            i = end_idx + 1
        else:
            i += 1
    return ranges


def create_verse_reference(vref_lines: List[str], start_idx: int, end_idx: int) -> str:
    """
    Create a verse reference string for a single verse or a span.
    Example single: 'GEN 1:14'  |  span (same chapter): 'GEN 1:14-17'
    span (cross chapter): 'GEN 1:31-2:3'
    """
    start_ref = vref_lines[start_idx].strip()
    if start_idx == end_idx:
        return start_ref

    end_ref = vref_lines[end_idx].strip()

    start_parts = start_ref.split()
    end_parts = end_ref.split()
    if len(start_parts) >= 2 and len(end_parts) >= 2:
        book = start_parts[0]
        start_ch_verse = start_parts[1]
        end_ch_verse = end_parts[1]
        start_ch = start_ch_verse.split(":")[0]
        end_ch = end_ch_verse.split(":")[0]
        if start_ch == end_ch:
            start_v = start_ch_verse.split(":")[1] if ":" in start_ch_verse else start_ch_verse
            end_v = end_ch_verse.split(":")[1] if ":" in end_ch_verse else end_ch_verse
            return f"{book} {start_ch}:{start_v}-{end_v}"
        return f"{book} {start_ch_verse}-{end_ch_verse}"
    # Fallback
    return f"{start_ref}-{end_ref.split()[-1]}"


def detect_range_conflicts(source_ranges: List[Tuple[int, int]], 
                          target_ranges: List[Tuple[int, int]]) -> List[Dict]:
    """
    Detect conflicts between source and target ranges.
    Returns list of conflict descriptions for reporting.
    """
    conflicts = []
    
    source_range_indices = set()
    for start, end in source_ranges:
        for i in range(start, end + 1):
            source_range_indices.add(i)
    
    target_range_indices = set()
    for start, end in target_ranges:
        for i in range(start, end + 1):
            target_range_indices.add(i)
    
    # Find indices where source has range but target doesn't
    source_only = source_range_indices - target_range_indices
    if source_only:
        conflicts.append({
            "type": "source_range_only",
            "indices": sorted(source_only),
            "description": "Source has range markers but target doesn't"
        })
    
    # Find indices where target has range but source doesn't
    target_only = target_range_indices - source_range_indices
    if target_only:
        conflicts.append({
            "type": "target_range_only", 
            "indices": sorted(target_only),
            "description": "Target has range markers but source doesn't"
        })
    
    return conflicts


def merge_range_maps(source_ranges: List[Tuple[int, int]], 
                    target_ranges: List[Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
    """
    Create a unified range map that handles both source and target ranges.
    When there's a conflict, use the longer range (more conservative approach).
    """
    # Convert to dictionaries for easier manipulation
    source_map = {start: (start, end) for start, end in source_ranges}
    target_map = {start: (start, end) for start, end in target_ranges}
    
    # Start with all source ranges
    unified_map = source_map.copy()
    
    # Add target ranges, resolving conflicts
    for start, (t_start, t_end) in target_map.items():
        if start in unified_map:
            # Conflict: both have ranges starting at same index
            s_start, s_end = unified_map[start]
            # Use the longer range (more conservative)
            if (t_end - t_start) > (s_end - s_start):
                unified_map[start] = (t_start, t_end)
        else:
            # Check if this target range overlaps with any source range
            overlaps = False
            for s_start in list(unified_map.keys()):
                s_start_idx, s_end_idx = unified_map[s_start]
                if (t_start <= s_end_idx and t_end >= s_start_idx):
                    # Overlapping ranges - merge them
                    new_start = min(s_start_idx, t_start)
                    new_end = max(s_end_idx, t_end)
                    # Remove old range and add merged range
                    del unified_map[s_start]
                    unified_map[new_start] = (new_start, new_end)
                    overlaps = True
                    break
            
            if not overlaps:
                # No overlap, add target range
                unified_map[start] = (t_start, t_end)
    
    return unified_map


def build_aligned_data_improved(
    source_text_lines: List[str], 
    target_text_lines: List[str], 
    vref_lines: List[str],
    preserve_case: bool = False,
    normalize_punctuation: bool = True,
    normalize_quotes: bool = True
) -> Tuple[List[Dict[str, str]], List[Dict]]:
    """
    Build aligned data with improved range handling.
    Returns (data, conflicts) where conflicts describes any range mismatches found.
    """
    if len(source_text_lines) != len(target_text_lines) or len(source_text_lines) != len(vref_lines):
        print(f"Warning: Mismatched line counts - source: {len(source_text_lines)}, "
              f"target: {len(target_text_lines)}, vref: {len(vref_lines)}")
        # Take the minimum length to avoid index errors
        min_len = min(len(source_text_lines), len(target_text_lines), len(vref_lines))
        source_text_lines = source_text_lines[:min_len]
        target_text_lines = target_text_lines[:min_len]
        vref_lines = vref_lines[:min_len]
    
    # Identify ranges in both source and target
    source_ranges = identify_verse_ranges(source_text_lines)
    target_ranges = identify_verse_ranges(target_text_lines)
    
    # Detect and report conflicts
    conflicts = detect_range_conflicts(source_ranges, target_ranges)
    
    # Create unified range map
    range_map = merge_range_maps(source_ranges, target_ranges)

    rows: List[Dict[str, str]] = []
    processed: Set[int] = set()

    for idx, (source_line, target_line) in enumerate(zip(source_text_lines, target_text_lines)):
        if idx in processed:
            continue
        if (not source_line or source_line == "<range>") and (not target_line or target_line == "<range>"):
            processed.add(idx)
            continue

        if idx in range_map:
            start_idx, end_idx = range_map[idx]
            verse_ref = create_verse_reference(vref_lines, start_idx, end_idx)
            
            # For ranges, combine all non-<range> text within the span
            source_texts = []
            target_texts = []
            
            for i in range(start_idx, end_idx + 1):
                if i < len(source_text_lines) and source_text_lines[i] != "<range>":
                    source_text = source_text_lines[i].strip()
                    if source_text:
                        source_texts.append(source_text)
                
                if i < len(target_text_lines) and target_text_lines[i] != "<range>":
                    target_text = target_text_lines[i].strip()
                    if target_text:
                        target_texts.append(target_text)
            
            # Join multiple texts with space (for ranges that span multiple verses)
            combined_source = " ".join(source_texts)
            combined_target = " ".join(target_texts)
            
            if combined_source and combined_target:
                # Apply text normalization
                normalized_source = normalize_text(combined_source, preserve_case, normalize_punctuation, normalize_quotes)
                normalized_target = normalize_text(combined_target, preserve_case, normalize_punctuation, normalize_quotes)
                
                rows.append({
                    "verse": verse_ref, 
                    "source_text": normalized_source,
                    "target_text": normalized_target
                })
            
            # Mark all indices in this range as processed
            for j in range(start_idx, end_idx + 1):
                processed.add(j)
            continue

        # Regular single-verse line
        verse_ref = vref_lines[idx].strip() if idx < len(vref_lines) else f"VERSE_{idx+1}"
        source_text = source_line.strip()
        target_text = target_line.strip()
        if source_text and target_text:
            # Apply text normalization
            normalized_source = normalize_text(source_text, preserve_case, normalize_punctuation, normalize_quotes)
            normalized_target = normalize_text(target_text, preserve_case, normalize_punctuation, normalize_quotes)
            
            rows.append({
                "verse": verse_ref, 
                "source_text": normalized_source,
                "target_text": normalized_target
            })
        processed.add(idx)

    return rows, conflicts


def is_nt_verse(verse_ref: str) -> bool:
    if not verse_ref:
        return False
    book = verse_ref.split()[0]
    return book in NT_BOOKS


def split_ot_nt(data: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    nt_data = [row for row in data if is_nt_verse(row["verse"])]
    ot_data = [row for row in data if not is_nt_verse(row["verse"])]
    return ot_data, nt_data


def write_csv(data: List[Dict[str, str]], filepath: str):
    """Write data to CSV file"""
    if not data:
        print(f"Warning: No data to write to {filepath}")
        return
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['verse', 'source_text', 'target_text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(description="Create aligned Bible corpus with improved range handling")
    parser.add_argument(
        "--source_corpus_path",
        type=str,
        required=True,
        help="Absolute path to source corpus file",
    )
    parser.add_argument(
        "--target_corpus_path", 
        type=str,
        required=True,
        help="Absolute path to target corpus file",
    )
    parser.add_argument(
        "--vref_path",
        type=str,
        default="/data/gpfs/projects/punim0478/setiawand/ebible/metadata/vref.txt",
        help="Absolute path to vref.txt with verse references",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="ind",
        help="Source language code",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="dhao",
        help="Target language code",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/aligned_corpus",
        help="Directory to save CSV outputs",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="aligned",
        help="Prefix for output filenames (default: aligned)",
    )
    parser.add_argument(
        "--preserve-case",
        action="store_true",
        help="Preserve original capitalization in normalized text (default: convert to lowercase)"
    )
    parser.add_argument(
        "--no-punctuation",
        action="store_true",
        help="Skip punctuation normalization"
    )
    parser.add_argument(
        "--no-quotes",
        action="store_true",
        help="Skip quote removal"
    )

    args = parser.parse_args()

    if not os.path.exists(args.source_corpus_path):
        raise FileNotFoundError(f"Source corpus not found: {args.source_corpus_path}")
    if not os.path.exists(args.target_corpus_path):
        raise FileNotFoundError(f"Target corpus not found: {args.target_corpus_path}")
    if not os.path.exists(args.vref_path):
        raise FileNotFoundError(f"Verse refs not found: {args.vref_path}")

    print(f"Loading source corpus: {args.source_corpus_path}")
    source_lines = load_lines(args.source_corpus_path)
    
    print(f"Loading target corpus: {args.target_corpus_path}")
    target_lines = load_lines(args.target_corpus_path)
    
    print(f"Loading verse references: {args.vref_path}")
    vref_lines = load_lines(args.vref_path)

    print("Building aligned data with improved range handling...")
    data, conflicts = build_aligned_data_improved(
        source_lines, 
        target_lines, 
        vref_lines,
        preserve_case=args.preserve_case,
        normalize_punctuation=not args.no_punctuation,
        normalize_quotes=not args.no_quotes
    )
    
    # Report any conflicts found
    if conflicts:
        print(f"\n⚠️  Found {len(conflicts)} range conflicts:")
        for conflict in conflicts:
            print(f"  - {conflict['description']} at indices: {conflict['indices']}")
        print("  These conflicts have been resolved by merging ranges.")
    else:
        print("✅ No range conflicts detected.")
    
    print("\nSplitting into OT and NT...")
    ot_data, nt_data = split_ot_nt(data)

    # Basic analysis summary
    print(f"\nBuilt aligned data with verse/source_text/target_text")
    print(f"Total rows: {len(data)} | OT: {len(ot_data)} | NT: {len(nt_data)}")

    # Optional: counts by book
    def book_from_verse(v: str) -> str:
        return v.split()[0] if v and isinstance(v, str) and " " in v else v

    book_counts = {}
    for row in data:
        book = book_from_verse(row["verse"])
        book_counts[book] = book_counts.get(book, 0) + 1
    
    print("\nCounts by book (all):")
    for book in sorted(book_counts.keys()):
        print(f"  {book}: {book_counts[book]}")

    # Save CSV files
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_csv = os.path.join(args.output_dir, f"{args.output_prefix}-{args.src_lang}-{args.tgt_lang}-all.csv")
    ot_csv = os.path.join(args.output_dir, f"{args.output_prefix}-{args.src_lang}-{args.tgt_lang}-ot.csv")
    nt_csv = os.path.join(args.output_dir, f"{args.output_prefix}-{args.src_lang}-{args.tgt_lang}-nt.csv")
    
    write_csv(data, all_csv)
    write_csv(ot_data, ot_csv)
    write_csv(nt_data, nt_csv)
    
    print(f"\nSaved CSV files:")
    print(f"  All: {all_csv}")
    print(f"  OT:  {ot_csv}")
    print(f"  NT:  {nt_csv}")

    # Save conflict report if any conflicts were found
    if conflicts:
        conflict_report = os.path.join(args.output_dir, f"{args.output_prefix}-conflicts.txt")
        with open(conflict_report, 'w', encoding='utf-8') as f:
            f.write("Range Conflict Report\n")
            f.write("====================\n\n")
            for conflict in conflicts:
                f.write(f"Type: {conflict['type']}\n")
                f.write(f"Description: {conflict['description']}\n")
                f.write(f"Affected indices: {conflict['indices']}\n\n")
        print(f"  Conflict report: {conflict_report}")


if __name__ == "__main__":
    main()
