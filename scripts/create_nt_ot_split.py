#!/usr/bin/env python3
"""
Analyze Indonesian Bible corpus (single language) from ind-indayt.txt.

Creates a pandas DataFrame with two columns: 'verse' and 'text'.
Separates Old Testament (OT) and New Testament (NT) data based on verse book codes.

Notes on <range> markers (based on project loaders):
- Lines with the literal token '<range>' indicate a continuation of the previous verse,
  i.e., the previous line contains the merged text for a span of verses.
- For such spans we keep one row with the combined text and a verse reference span
  like 'GEN 1:1-3'.
"""

import argparse
import os
from typing import List, Tuple, Dict
from pathlib import Path
import pandas as pd


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


def build_ind_dataframe(ind_text_lines: List[str] | str, vref_lines: List[str] | str) -> pd.DataFrame:
    """
    Build DataFrame with columns: verse, text.
    Handles <range> markers by keeping one row for the combined verse span.
    Skips blank lines and literal '<range>' lines.
    """
    # Convenience: allow passing file paths directly
    if isinstance(ind_text_lines, str):
        candidate = os.path.expanduser(os.path.expandvars(ind_text_lines)).strip()
        if os.path.exists(candidate):
            ind_text_lines = load_lines(candidate)
        else:
            # If a string but not a path, treat as single-line content
            ind_text_lines = [ind_text_lines]
    if isinstance(vref_lines, str):
        candidate = os.path.expanduser(os.path.expandvars(vref_lines)).strip()
        if os.path.exists(candidate):
            vref_lines = load_lines(candidate)
        else:
            vref_lines = [vref_lines]
    ranges = identify_verse_ranges(ind_text_lines)
    range_map: Dict[int, Tuple[int, int]] = {start: (start, end) for start, end in ranges}

    rows: List[Dict[str, str]] = []
    processed: set[int] = set()

    for idx, line in enumerate(ind_text_lines):
        if idx in processed:
            continue
        if not line or line == "<range>":
            processed.add(idx)
            continue

        if idx in range_map:
            start_idx, end_idx = range_map[idx]
            verse_ref = create_verse_reference(vref_lines, start_idx, end_idx)
            text = ind_text_lines[start_idx].strip()
            if text:
                rows.append({"verse": verse_ref, "text": text})
            for j in range(start_idx, end_idx + 1):
                processed.add(j)
            continue

        # Regular single-verse line
        verse_ref = vref_lines[idx].strip() if idx < len(vref_lines) else f"VERSE_{idx+1}"
        text = line.strip()
        if text:
            rows.append({"verse": verse_ref, "text": text})
        processed.add(idx)

    df = pd.DataFrame(rows, columns=["verse", "text"]).reset_index(drop=True)
    return df


def is_nt_verse(verse_ref: str) -> bool:
    if not verse_ref:
        return False
    book = verse_ref.split()[0]
    return book in NT_BOOKS


def split_ot_nt(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nt_df = df[df["verse"].apply(is_nt_verse)].reset_index(drop=True)
    ot_df = df[~df["verse"].apply(is_nt_verse)].reset_index(drop=True)
    return ot_df, nt_df


def main():
    parser = argparse.ArgumentParser(description="Build DataFrame from ind-indayt.txt and split OT/NT")
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="/data/gpfs/projects/punim0478/setiawand/ebible/corpus/ind-indayt.txt",
        help="Absolute path to ind-indayt.txt",
    )
    parser.add_argument(
        "--vref_path",
        type=str,
        default="/data/gpfs/projects/punim0478/setiawand/ebible/metadata/vref.txt",
        help="Absolute path to vref.txt with verse references",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/ind_corpus",
        help="Directory to save optional CSV outputs",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="If set, saves CSVs: all.csv, ot.csv, nt.csv",
    )

    args = parser.parse_args()

    if not os.path.exists(args.corpus_path):
        raise FileNotFoundError(f"Corpus not found: {args.corpus_path}")
    if not os.path.exists(args.vref_path):
        raise FileNotFoundError(f"Verse refs not found: {args.vref_path}")

    ind_lines = load_lines(args.corpus_path)
    vref_lines = load_lines(args.vref_path)

    df = build_ind_dataframe(ind_lines, vref_lines)
    ot_df, nt_df = split_ot_nt(df)

    # Basic analysis summary
    print("Built DataFrame with verse/text")
    print(f"Total rows: {len(df)} | OT: {len(ot_df)} | NT: {len(nt_df)}")

    # Optional: counts by book
    def book_from_verse(v: str) -> str:
        return v.split()[0] if v and isinstance(v, str) and " " in v else v

    df_books = df.copy()
    df_books["book"] = df_books["verse"].apply(book_from_verse)
    counts_by_book = df_books["book"].value_counts().sort_index()
    print("\nCounts by book (all):")
    for book, cnt in counts_by_book.items():
        print(f"  {book}: {cnt}")

    if args.save_csv:
        os.makedirs(args.output_dir, exist_ok=True)
        filename = Path(args.corpus_path).stem
        ot_csv = os.path.join(args.output_dir, f"{filename}-ot.csv")
        nt_csv = os.path.join(args.output_dir, f"{filename}-nt.csv")
        ot_df.to_csv(ot_csv, index=False)
        nt_df.to_csv(nt_csv, index=False)
        print(f"\nSaved CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()


