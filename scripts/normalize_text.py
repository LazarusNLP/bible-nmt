#!/usr/bin/env python3
"""
Text normalization script for creating parallel Bible corpora.

This script normalizes Bible text files for better alignment across languages by:
1. Removing square bracket characters [ and ] while preserving content
2. Converting numbers with verbalized forms: "99 (ceo nguru ceo)" -> "ceo nguru ceo"
3. Removing remaining parentheses ( and ) while preserving content  
4. Normalizing punctuation (semicolons and colons to commas, dash normalization)
5. Removing all quotation marks (straight, curly, angle quotes)
6. Normalizing whitespace and punctuation spacing
7. Optional case normalization (lowercase for better alignment)

Designed specifically for creating parallel translation pairs where consistent
formatting across different language versions is critical.
"""

import argparse
import re
import os
from pathlib import Path


def normalize_text(text, preserve_case=False, normalize_punctuation=True, normalize_quotes=True):
    """
    Normalize text for parallel corpus creation by standardizing punctuation, quotes, and whitespace.
    
    Args:
        text (str): Input text to normalize
        preserve_case (bool): If False, converts to lowercase for better alignment
        normalize_punctuation (bool): If True, normalizes punctuation marks
        normalize_quotes (bool): If True, removes all quotation marks
        
    Returns:
        str: Normalized text
    """
    # Remove square bracket characters [ and ] but keep their content
    text = text.replace('[', '')
    text = text.replace(']', '')
    
    # Handle numbers with verbalized versions: "99 (ceo nguru ceo)" -> "ceo nguru ceo"
    # This pattern matches digits followed by parentheses containing the verbalized form
    text = re.sub(r'\d+\s*\(([^)]+)\)', r'\1', text)
    
    # Remove remaining parentheses but keep their content (often used for added words)
    text = text.replace('(', '')
    text = text.replace(')', '')
    
    if normalize_punctuation:
        # Normalize punctuation for better alignment
        # text = text.replace(';', ',')    # Convert semicolons to commas
        # text = text.replace(':', ',')    # Convert colons to commas
        text = text.replace('—', '-')    # Em dash to hyphen
        text = text.replace('–', '-')    # En dash to hyphen
        
        # Remove multiple punctuation marks (e.g., "!!" -> "!")
        text = re.sub(r'([.!?])\1+', r'\1', text)
        text = re.sub(r'([,;:])\1+', r'\1', text)
    
    if normalize_quotes:
        # Remove all types of quotes (double and single, straight and curly)
        text = text.replace('“', '')     # Unicode left double quotation mark
        text = text.replace('”', '')     # Unicode right double quotation mark
        text = text.replace("ꞌ", "'")     # Straight single quote
        text = text.replace("’", "'")     # Straight single quote
    
    # Normalize whitespace around punctuation
    text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)  # Ensure single space after punctuation
    
    # Normalize whitespace - remove extra spaces and clean up
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()  # Remove leading/trailing whitespace
    
    # Optional case normalization for better alignment
    if not preserve_case:
        text = text.lower()
    
    return text


def process_file(input_path, output_path=None, preserve_case=False, normalize_punctuation=True, normalize_quotes=True):
    """
    Process a text file to normalize its content.
    
    Args:
        input_path (str): Path to input file
        output_path (str): Path to output file (if None, overwrites input)
        preserve_case (bool): If False, converts to lowercase for better alignment
        normalize_punctuation (bool): If True, normalizes punctuation marks
        normalize_quotes (bool): If True, removes all quotation marks
    """
    print(f"Processing file: {input_path}")
    
    # Read the input file
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line
    normalized_lines = []
    for line_num, line in enumerate(lines, 1):
        original_line = line.rstrip('\n')
        normalized_line = normalize_text(original_line, preserve_case, normalize_punctuation, normalize_quotes)
        normalized_lines.append(normalized_line)
        
        # Show progress for large files
        if line_num % 1000 == 0:
            print(f"Processed {line_num} lines...")
    
    # Determine output path
    if output_path is None:
        output_path = input_path
    
    # Write the normalized content
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in normalized_lines:
            f.write(line + '\n')
    
    print(f"Normalized text saved to: {output_path}")
    print(f"Processed {len(normalized_lines)} lines")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize Bible text by removing square bracket characters and quotes"
    )
    parser.add_argument(
        "input_file",
        help="Path to input text file to normalize"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (if not specified, overwrites input file)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a backup of the original file before processing"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Preview first N lines without making changes (for testing)"
    )
    parser.add_argument(
        "--preserve-case",
        action="store_true",
        help="Preserve original capitalization (default: convert to lowercase)"
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
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    
    # Preview mode - show what changes would be made
    if args.preview > 0:
        print(f"Preview mode: showing first {args.preview} lines")
        print("=" * 60)
        
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= args.preview:
                    break
                original = line.rstrip('\n')
                normalized = normalize_text(original, args.preserve_case, not args.no_punctuation, not args.no_quotes)
                print(f"Line {i+1}:")
                print(f"  Original:   {original}")
                print(f"  Normalized: {normalized}")
                print()
        return 0
    
    # Create backup if requested
    if args.backup:
        backup_path = args.input_file + '.backup'
        print(f"Creating backup: {backup_path}")
        import shutil
        shutil.copy2(args.input_file, backup_path)
    
    # Process the file
    try:
        process_file(args.input_file, args.output, args.preserve_case, not args.no_punctuation, not args.no_quotes)
        print("Normalization completed successfully!")
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
