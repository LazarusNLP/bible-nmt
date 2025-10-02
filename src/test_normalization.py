#!/usr/bin/env python3
"""
Test script for text normalization on Dhao text files.
Usage: python test_normalization.py [path_to_text_file]
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from text_normalization import normalize_text, TextNormalizer

def test_dhao_specific_examples():
    """Test normalization with Dhao-specific examples."""
    print("Testing Dhao-specific examples:")
    print("=" * 50)
    
    # Test cases for Dhao language
    test_cases = [
        "word1-word2",  # Dhao compound words with dash
        "nala-nala",    # Example Dhao word
        "Hello , world !",  # Punctuation test
        "This is a test  sentence   with   multiple spaces.",  # Whitespace test
        "Mixed—punctuation; test: here?",  # Mixed punctuation
        "  Leading and trailing spaces  ",  # Boundary whitespace
        "Dhao-specific word-connections should-remain intact.",  # Multiple dashes
    ]
    
    normalizer = TextNormalizer()
    
    for i, text in enumerate(test_cases, 1):
        normalized = normalizer.normalize(text)
        print(f"Test {i}:")
        print(f"  Original:   '{text}'")
        print(f"  Normalized: '{normalized}'")
        if text != normalized:
            print(f"  → Changed: {text != normalized}")
        print()

def test_text_file(file_path):
    """Test normalization on a text file."""
    print(f"Testing normalization on file: {file_path}")
    print("=" * 50)
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    # Read first 10 lines for testing
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()[:10] if line.strip()]
    
    print(f"Processing first {len(lines)} non-empty lines from the file:")
    print()
    
    normalizer = TextNormalizer()
    
    for i, line in enumerate(lines, 1):
        normalized = normalize_text(line)
        print(f"Line {i}:")
        print(f"  Original:   '{line}'")
        print(f"  Normalized: '{normalized}'")
        if line != normalized:
            print(f"  → Changed: {line != normalized}")
        print()

def test_full_file_processing(input_file, output_file=None):
    """Process entire file and save normalized version."""
    print(f"Processing entire file: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return
    
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_normalized{ext}"
    
    normalizer = TextNormalizer()
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            line_count = 0
            changed_count = 0
            
            for line in infile:
                original_line = line.rstrip('\n\r')
                if original_line.strip():  # Only process non-empty lines
                    normalized_line = normalizer.normalize(original_line)
                    if normalized_line != original_line:
                        changed_count += 1
                    outfile.write(normalized_line + '\n')
                    line_count += 1
                else:
                    outfile.write(line)  # Keep empty lines as-is
    
    print(f"✅ Processed {line_count} lines")
    print(f"✅ Changed {changed_count} lines ({changed_count/line_count*100:.1f}%)")
    print(f"✅ Saved normalized text to: {output_file}")

def main():
    """Main function."""
    print("🔤 Text Normalization Test for Dhao Language")
    print("=" * 60)
    print()
    
    # Always run Dhao-specific tests
    test_dhao_specific_examples()
    
    # Test with file if provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print("\n" + "=" * 60)
        test_text_file(file_path)
        
        # Ask if user wants to process the entire file
        print("\n" + "=" * 60)
        response = input("Do you want to process the entire file? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            test_full_file_processing(file_path)
    else:
        print("To test on a specific file, run:")
        print("python test_normalization.py /path/to/your/file.txt")
        print()
        print("For example:")
        print("python test_normalization.py /data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/nfa-nfa.txt")

if __name__ == "__main__":
    main()
