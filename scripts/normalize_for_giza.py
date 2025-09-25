#!/usr/bin/env python3
"""
Enhanced text normalization specifically for GIZA++ word alignment.

This script provides more aggressive normalization optimized for statistical
word alignment models like GIZA++.
"""

import argparse
import re
import os
from pathlib import Path


def normalize_for_giza(
    text, remove_all_punct=True, normalize_numbers=True, min_length=3
):
    """
    Normalize text specifically for GIZA++ word alignment.

    Args:
        text (str): Input text to normalize
        remove_all_punct (bool): If True, removes all punctuation
        normalize_numbers (bool): If True, replaces numbers with <NUM> token
        min_length (int): Minimum number of tokens to keep sentence

    Returns:
        str: Normalized text, or None if sentence is too short
    """
    # Convert to lowercase for better alignment
    text = text.lower()

    # Remove square brackets and parentheses (keep content)
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace("(", "")
    text = text.replace(")", "")

    # Handle numbers with verbalized versions: "99 (ceo nguru ceo)" -> "ceo nguru ceo"
    text = re.sub(
        r"\d+\s*\([^)]+\)", lambda m: m.group(0).split("(")[1].rstrip(")"), text
    )

    # Number normalization
    if normalize_numbers:
        text = re.sub(r"\b\d+\b", "<NUM>", text)

    # Remove quotation marks but preserve single quotes within words (contractions)
    text = re.sub(r'["""' "`´" '""]', "", text)  # Remove double quotes and fancy quotes

    # Handle spaced contractions like "God 's" -> "god's"
    text = re.sub(r"(\w)\s+'(\w)", r"\1'\2", text)  # word 's -> word's
    text = re.sub(r"(\w)'\s+(\w)", r"\1'\2", text)  # word' s -> word's

    # Remove single quotes only when they're not part of contractions
    # Keep apostrophes in contractions like don't, won't, God's, etc.
    text = re.sub(
        r"(?<!\w)'(?!\w)", " ", text
    )  # Remove quotes not between word characters

    # Punctuation handling
    if remove_all_punct:
        # Remove punctuation but preserve single quotes within words (contractions)
        text = re.sub(r"[^\w\s']", " ", text)  # Keep apostrophes
    else:
        # Keep only sentence boundary punctuation and apostrophes within words
        text = re.sub(r"[^\w\s\.\!\?']", " ", text)  # Keep apostrophes
        # Normalize dash variants
        text = text.replace("—", "-")
        text = text.replace("–", "-")
        # Remove multiple punctuation
        text = re.sub(r"([.!?])\1+", r"\1", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Filter out very short sentences
    tokens = text.split()

    return text


def process_parallel_files(
    source_file,
    target_file,
    output_source,
    output_target,
    remove_all_punct=True,
    normalize_numbers=True,
    min_length=3,
):
    """
    Process parallel files for GIZA++ training.

    Args:
        source_file (str): Path to source language file
        target_file (str): Path to target language file
        output_source (str): Output path for normalized source
        output_target (str): Output path for normalized target
        remove_all_punct (bool): Remove all punctuation
        normalize_numbers (bool): Normalize numbers
        min_length (int): Minimum sentence length in tokens
    """
    print(f"Processing parallel files:")
    print(f"  Source: {source_file} -> {output_source}")
    print(f"  Target: {target_file} -> {output_target}")

    # Read both files
    with open(source_file, "r", encoding="utf-8") as f:
        source_lines = f.readlines()

    with open(target_file, "r", encoding="utf-8") as f:
        target_lines = f.readlines()

    if len(source_lines) != len(target_lines):
        print(
            f"Warning: Line count mismatch - Source: {len(source_lines)}, Target: {len(target_lines)}"
        )
        min_lines = min(len(source_lines), len(target_lines))
        source_lines = source_lines[:min_lines]
        target_lines = target_lines[:min_lines]

    # Process and filter parallel lines
    normalized_source = []
    normalized_target = []
    skipped_count = 0

    for i, (src_line, tgt_line) in enumerate(zip(source_lines, target_lines)):
        src_norm = normalize_for_giza(
            src_line.strip(), remove_all_punct, normalize_numbers, min_length
        )
        tgt_norm = normalize_for_giza(
            tgt_line.strip(), remove_all_punct, normalize_numbers, min_length
        )

        # Keep pair only if both sides are valid
        if src_norm is not None and tgt_norm is not None:
            normalized_source.append(src_norm)
            normalized_target.append(tgt_norm)
        else:
            skipped_count += 1

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} lines...")

    # Write normalized files
    with open(output_source, "w", encoding="utf-8") as f:
        for line in normalized_source:
            f.write(line + "\n")

    with open(output_target, "w", encoding="utf-8") as f:
        for line in normalized_target:
            f.write(line + "\n")

    print(f"Normalization completed!")
    print(f"  Original pairs: {len(source_lines)}")
    print(f"  Kept pairs: {len(normalized_source)}")
    print(f"  Skipped pairs: {skipped_count}")
    print(f"  Retention rate: {len(normalized_source)/len(source_lines)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize parallel text files for GIZA++ word alignment training"
    )
    parser.add_argument("source_file", help="Source language file")
    parser.add_argument("target_file", help="Target language file")
    parser.add_argument(
        "-s", "--output-source", required=True, help="Output source file"
    )
    parser.add_argument(
        "-t", "--output-target", required=True, help="Output target file"
    )
    parser.add_argument(
        "--keep-punct", action="store_true", help="Keep sentence boundary punctuation"
    )
    parser.add_argument(
        "--keep-numbers",
        action="store_true",
        help="Keep original numbers (don't normalize)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=3,
        help="Minimum sentence length in tokens (default: 3)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Preview first N pairs without making changes",
    )

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.source_file):
        print(f"Error: Source file not found: {args.source_file}")
        return 1

    if not os.path.exists(args.target_file):
        print(f"Error: Target file not found: {args.target_file}")
        return 1

    # Preview mode
    if args.preview > 0:
        print(f"Preview mode: showing first {args.preview} pairs")
        print("=" * 80)

        with open(args.source_file, "r", encoding="utf-8") as sf, open(
            args.target_file, "r", encoding="utf-8"
        ) as tf:

            for i, (src_line, tgt_line) in enumerate(zip(sf, tf)):
                if i >= args.preview:
                    break

                src_orig = src_line.strip()
                tgt_orig = tgt_line.strip()
                src_norm = normalize_for_giza(
                    src_orig,
                    not args.keep_punct,
                    not args.keep_numbers,
                    args.min_length,
                )
                tgt_norm = normalize_for_giza(
                    tgt_orig,
                    not args.keep_punct,
                    not args.keep_numbers,
                    args.min_length,
                )

                print(f"Pair {i+1}:")
                print(f"  SRC Original:   {src_orig}")
                print(f"  SRC Normalized: {src_norm}")
                print(f"  TGT Original:   {tgt_orig}")
                print(f"  TGT Normalized: {tgt_norm}")
                print()
        return 0

    # Process files
    try:
        process_parallel_files(
            args.source_file,
            args.target_file,
            args.output_source,
            args.output_target,
            not args.keep_punct,
            not args.keep_numbers,
            args.min_length,
        )
    except Exception as e:
        print(f"Error processing files: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
