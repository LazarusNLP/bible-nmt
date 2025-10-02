#!/usr/bin/env python3
"""
Missing Verse Checker

This script checks for missing verses between English Bible versions and the Dhao (nfa) Bible.
It identifies cases where a verse exists in one version but not the other, excluding cases
where both versions don't have content.

Based on the alignment script create_aligned_nt_ot.py logic.
"""

import argparse
import os
import csv
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import glob


def load_lines(file_path: str) -> List[str]:
    """Load lines from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def identify_verse_ranges(text_lines: List[str]) -> List[Tuple[int, int]]:
    """
    Identify verse ranges by finding consecutive '<range>' lines following a content line.
    Returns list of (start_idx, end_idx) indices where 'start_idx' holds the combined text
    and start_idx+1..end_idx are '<range>' markers.
    """
    ranges = []
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


def get_verse_content_status(text_lines: List[str], vref_lines: List[str]) -> Dict[str, str]:
    """
    Get the content status for each verse reference.
    Returns dict mapping verse_ref -> status where status is:
    - 'content': verse has actual content
    - 'empty': verse is empty/blank
    - 'range': verse is part of a range (marked with <range>)
    """
    ranges = identify_verse_ranges(text_lines)
    range_map = {start: (start, end) for start, end in ranges}
    
    verse_status = {}
    processed = set()
    
    for idx, (text_line, vref_line) in enumerate(zip(text_lines, vref_lines)):
        if idx in processed:
            continue
            
        verse_ref = vref_line.strip()
        
        # Handle ranges
        if idx in range_map:
            start_idx, end_idx = range_map[idx]
            # The first line in range contains the combined content
            content = text_lines[start_idx].strip()
            if content:
                verse_status[verse_ref] = 'content'
            else:
                verse_status[verse_ref] = 'empty'
            
            # Mark range lines as 'range'
            for i in range(start_idx + 1, end_idx + 1):
                if i < len(vref_lines):
                    range_verse_ref = vref_lines[i].strip()
                    verse_status[range_verse_ref] = 'range'
            
            # Mark all indices as processed
            for i in range(start_idx, end_idx + 1):
                processed.add(i)
        else:
            # Regular single verse
            content = text_line.strip()
            if content == "<range>":
                verse_status[verse_ref] = 'range'
            elif content:
                verse_status[verse_ref] = 'content'
            else:
                verse_status[verse_ref] = 'empty'
            processed.add(idx)
    
    return verse_status


def compare_versions(eng_status: Dict[str, str], dhao_status: Dict[str, str], 
                    eng_version: str) -> Dict[str, List[str]]:
    """
    Compare English and Dhao verse statuses and find mismatches.
    Returns dict with different types of mismatches.
    """
    mismatches = {
        'eng_has_content_dhao_missing': [],      # English has content, Dhao is empty
        'dhao_has_content_eng_missing': [],      # Dhao has content, English is empty
        'eng_has_content_dhao_range': [],        # English has content, Dhao is range marker
        'dhao_has_content_eng_range': [],        # Dhao has content, English is range marker
    }
    
    # Get all verse references
    all_verses = set(eng_status.keys()) | set(dhao_status.keys())
    
    for verse_ref in sorted(all_verses):
        eng_stat = eng_status.get(verse_ref, 'missing')
        dhao_stat = dhao_status.get(verse_ref, 'missing')
        
        # Skip if both are empty or both are ranges - these are expected
        if (eng_stat == 'empty' and dhao_stat == 'empty') or \
           (eng_stat == 'range' and dhao_stat == 'range'):
            continue
        
        # Find problematic cases
        if eng_stat == 'content' and dhao_stat == 'empty':
            mismatches['eng_has_content_dhao_missing'].append(verse_ref)
        elif dhao_stat == 'content' and eng_stat == 'empty':
            mismatches['dhao_has_content_eng_missing'].append(verse_ref)
        elif eng_stat == 'content' and dhao_stat == 'range':
            mismatches['eng_has_content_dhao_range'].append(verse_ref)
        elif dhao_stat == 'content' and eng_stat == 'range':
            mismatches['dhao_has_content_eng_range'].append(verse_ref)
    
    return mismatches


def analyze_bible_versions(english_corpus_dir: str, dhao_corpus_path: str, vref_path: str):
    """
    Analyze all English Bible versions against Dhao version for missing verses.
    """
    print("=== Missing Verse Analysis ===")
    print(f"English corpus directory: {english_corpus_dir}")
    print(f"Dhao corpus path: {dhao_corpus_path}")
    print(f"Verse reference path: {vref_path}")
    print()
    
    # Load reference verse IDs and Dhao content
    print("Loading reference data...")
    vref_lines = load_lines(vref_path)
    dhao_lines = load_lines(dhao_corpus_path)
    
    if len(vref_lines) != len(dhao_lines):
        print(f"Warning: Line count mismatch - vref: {len(vref_lines)}, dhao: {len(dhao_lines)}")
        min_len = min(len(vref_lines), len(dhao_lines))
        vref_lines = vref_lines[:min_len]
        dhao_lines = dhao_lines[:min_len]
    
    print(f"Loaded {len(vref_lines)} verse references")
    
    # Get Dhao verse status
    print("Analyzing Dhao verse content...")
    dhao_status = get_verse_content_status(dhao_lines, vref_lines)
    
    dhao_content_count = sum(1 for status in dhao_status.values() if status == 'content')
    dhao_empty_count = sum(1 for status in dhao_status.values() if status == 'empty')
    dhao_range_count = sum(1 for status in dhao_status.values() if status == 'range')
    
    print(f"Dhao status: {dhao_content_count} content, {dhao_empty_count} empty, {dhao_range_count} range")
    print()
    
    # Get all English corpus files
    eng_pattern = os.path.join(english_corpus_dir, "eng-*.txt")
    eng_files = sorted(glob.glob(eng_pattern))
    
    print(f"Found {len(eng_files)} English Bible versions")
    print()
    
    # Store results for summary
    all_results = {}
    total_mismatches = defaultdict(int)
    
    # Analyze each English version
    for eng_file in eng_files:
        eng_version = os.path.basename(eng_file)[4:-4]  # Remove 'eng-' prefix and '.txt' suffix
        print(f"Analyzing {eng_version}...")
        
        # Load English version
        eng_lines = load_lines(eng_file)
        
        if len(eng_lines) != len(vref_lines):
            print(f"  Warning: Line count mismatch - {eng_version}: {len(eng_lines)}, vref: {len(vref_lines)}")
            min_len = min(len(eng_lines), len(vref_lines))
            eng_lines = eng_lines[:min_len]
            vref_subset = vref_lines[:min_len]
        else:
            vref_subset = vref_lines
        
        # Get English verse status
        eng_status = get_verse_content_status(eng_lines, vref_subset)
        
        # Compare with Dhao
        mismatches = compare_versions(eng_status, dhao_status, eng_version)
        
        # Count mismatches
        total_issues = sum(len(issues) for issues in mismatches.values())
        
        if total_issues > 0:
            print(f"  Found {total_issues} mismatches:")
            for mismatch_type, verses in mismatches.items():
                if verses:
                    print(f"    {mismatch_type}: {len(verses)} verses")
                    total_mismatches[mismatch_type] += len(verses)
        else:
            print(f"  No mismatches found!")
        
        all_results[eng_version] = mismatches
        print()
    
    # Generate detailed report
    print("=== DETAILED MISMATCH REPORT ===")
    print()
    
    # Create output directory
    output_dir = Path("missing_verses_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed CSV report
    csv_file = output_dir / "missing_verses_detailed.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['english_version', 'mismatch_type', 'verse_reference', 'description'])
        
        for eng_version, mismatches in all_results.items():
            for mismatch_type, verses in mismatches.items():
                description = {
                    'eng_has_content_dhao_missing': 'English has content, Dhao is empty',
                    'dhao_has_content_eng_missing': 'Dhao has content, English is empty',
                    'eng_has_content_dhao_range': 'English has content, Dhao is range marker',
                    'dhao_has_content_eng_range': 'Dhao has content, English is range marker'
                }[mismatch_type]
                
                for verse in verses:
                    writer.writerow([eng_version, mismatch_type, verse, description])
    
    print(f"Detailed report saved to: {csv_file}")
    
    # Summary report
    summary_file = output_dir / "missing_verses_summary.csv"
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['english_version', 'eng_content_dhao_missing', 'dhao_content_eng_missing', 
                        'eng_content_dhao_range', 'dhao_content_eng_range', 'total_mismatches'])
        
        for eng_version, mismatches in all_results.items():
            row = [eng_version]
            total = 0
            for mismatch_type in ['eng_has_content_dhao_missing', 'dhao_has_content_eng_missing',
                                'eng_has_content_dhao_range', 'dhao_has_content_eng_range']:
                count = len(mismatches[mismatch_type])
                row.append(count)
                total += count
            row.append(total)
            writer.writerow(row)
    
    print(f"Summary report saved to: {summary_file}")
    print()
    
    # Print summary statistics
    print("=== SUMMARY STATISTICS ===")
    print(f"Total English versions analyzed: {len(eng_files)}")
    print(f"Total verses checked: {len(vref_lines)}")
    print()
    print("Overall mismatch counts across all versions:")
    for mismatch_type, count in total_mismatches.items():
        description = {
            'eng_has_content_dhao_missing': 'English has content, Dhao missing',
            'dhao_has_content_eng_missing': 'Dhao has content, English missing',
            'eng_has_content_dhao_range': 'English content, Dhao range',
            'dhao_has_content_eng_range': 'Dhao content, English range'
        }[mismatch_type]
        print(f"  {description}: {count}")
    
    print()
    
    # Find versions with most/least mismatches
    version_totals = [(eng_version, sum(len(issues) for issues in mismatches.values())) 
                     for eng_version, mismatches in all_results.items()]
    version_totals.sort(key=lambda x: x[1])
    
    print("Versions with fewest mismatches:")
    for version, count in version_totals[:5]:
        print(f"  {version}: {count} mismatches")
    print()
    
    print("Versions with most mismatches:")
    for version, count in version_totals[-5:]:
        print(f"  {version}: {count} mismatches")
    print()
    
    # Show some example verses with issues
    print("=== EXAMPLE PROBLEMATIC VERSES ===")
    for mismatch_type, description in [
        ('eng_has_content_dhao_missing', 'English has content but Dhao is missing'),
        ('dhao_has_content_eng_missing', 'Dhao has content but English is missing')
    ]:
        examples = []
        for eng_version, mismatches in all_results.items():
            examples.extend([(eng_version, verse) for verse in mismatches[mismatch_type][:3]])
        
        if examples:
            print(f"\n{description}:")
            for eng_version, verse in examples[:10]:  # Show max 10 examples
                print(f"  {verse} (in {eng_version})")
    
    print("\nAnalysis complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Check for missing verses between English Bible versions and Dhao Bible"
    )
    parser.add_argument(
        "--english_corpus_dir",
        type=str,
        default="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/eng/corpus",
        help="Directory containing English Bible corpus files"
    )
    parser.add_argument(
        "--dhao_corpus_path",
        type=str,
        default="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/nfa-nfa.txt",
        help="Path to Dhao Bible corpus file"
    )
    parser.add_argument(
        "--vref_path",
        type=str,
        default="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/vref.txt",
        help="Path to verse reference file"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.english_corpus_dir):
        print(f"Error: English corpus directory not found: {args.english_corpus_dir}")
        return 1
    
    if not os.path.exists(args.dhao_corpus_path):
        print(f"Error: Dhao corpus file not found: {args.dhao_corpus_path}")
        return 1
    
    if not os.path.exists(args.vref_path):
        print(f"Error: Verse reference file not found: {args.vref_path}")
        return 1
    
    analyze_bible_versions(args.english_corpus_dir, args.dhao_corpus_path, args.vref_path)
    return 0


if __name__ == "__main__":
    exit(main())
