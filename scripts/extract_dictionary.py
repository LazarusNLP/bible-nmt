#!/usr/bin/env python3
r"""
Dictionary Extractor from LaTeX Format

This script extracts Indonesian-Dhao dictionary entries from a LaTeX table format
and exports them to CSV. The script handles the specific format used in the
Kamus Dhao-Indonesia bersih.md file.

Usage:
    python extract_dictionary.py <input_file> <output_file>

Example:
    python extract_dictionary.py dictionary/Kamus\ Dhao-Indonesia\ bersih.md dictionary_output.csv
"""

import re
import csv
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict


class DictionaryExtractor:
    def __init__(self):
        # Pattern to match \hline entries
        self.hline_pattern = re.compile(r"\\hline\s+(.*?)\\\\?$", re.MULTILINE)

        # Pattern to extract words from entries like "kb. word" or "kkt. word" etc.
        self.dhao_word_pattern = re.compile(
            r"(?:kb|kkt|kki|ks|phub|cnj|neg|idiom|aux|v|q|deic|tam|qnt|prep|kdep|cmplzr|vpref|time|interj|n)\.\s*([^.;,()&\\]+?)(?:[.;,()]|$)",
            re.IGNORECASE,
        )

        # Pattern to clean up formatting
        self.cleanup_pattern = re.compile(
            r"\\[a-zA-Z]+\{[^}]*\}|\\[a-zA-Z]+|[{}\\]|\([^)]*\)|lihat:[^;,]*"
        )

        # Pattern to split entries within a hline (separated by &)
        self.entry_split_pattern = re.compile(r"\s*&\s*")

    def clean_text(self, text: str) -> str:
        """Clean LaTeX formatting and unwanted characters from text."""
        if not text:
            return ""

        # Remove LaTeX commands and formatting
        text = self.cleanup_pattern.sub("", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove special characters but keep accented characters
        text = re.sub(r"[\\{}]", "", text)

        return text

    def extract_dhao_word(self, dhao_entry: str) -> str:
        """Extract the actual Dhao word from entries like 'kb. word (2).'"""
        if not dhao_entry:
            return ""

        # Clean the entry first
        entry = dhao_entry.strip()

        # Try to match the pattern for grammatical info + word
        matches = self.dhao_word_pattern.findall(entry)
        if matches:
            # Take the first match and clean it
            word = matches[0].strip()
            # Remove numbers in parentheses and extra punctuation
            word = re.sub(r"\([^)]*\)", "", word)
            word = re.sub(r"[.;,\\\s]+$", "", word)
            word = re.sub(r"^[.;,\\\s]+", "", word)
            word = word.strip()

            # Only return if it's a meaningful word
            if word and len(word) > 1 and not word.isdigit():
                return word

        # If no grammatical marker found, check if it's just a word without markers
        # Sometimes entries are just the word itself
        cleaned = self.clean_text(entry)
        if cleaned:
            # Remove common prefixes and suffixes
            cleaned = re.sub(r"^\w+\.\s*", "", cleaned)  # Remove grammatical markers
            cleaned = re.sub(r"\([^)]*\)", "", cleaned)  # Remove parentheses
            cleaned = re.sub(r"[.;,]+$", "", cleaned)  # Remove trailing punctuation
            cleaned = cleaned.strip()

            # Take the first meaningful word
            if cleaned:
                words = cleaned.split()
                for word in words:
                    # Skip common patterns
                    if (
                        len(word) > 1
                        and not word.isdigit()
                        and word not in ["lihat", "see", "the", "and"]
                        and not re.match(r"^[.;,\\]+$", word)
                    ):
                        return word

        return ""

    def parse_hline_entry(self, hline_content: str) -> List[Tuple[str, str]]:
        r"""Parse a single \hline entry and extract all Indonesian-Dhao pairs."""
        pairs = []

        # Split by & to get individual columns
        parts = self.entry_split_pattern.split(hline_content)

        # Process parts in pairs (columns 1&2, 3&4 typically contain word pairs)
        # Each table row has 4 columns: Indonesian1 & Dhao1 & Indonesian2 & Dhao2
        for i in range(0, len(parts) - 1, 2):
            if i + 1 >= len(parts):
                break

            indonesian_part = parts[i].strip()
            dhao_part = parts[i + 1].strip()

            # Skip empty parts or LaTeX formatting entries
            if (
                not indonesian_part
                or not dhao_part
                or "multicolumn" in indonesian_part.lower()
                or "multirow" in indonesian_part.lower()
                or indonesian_part.strip() == ""
                or dhao_part.strip() == ""
                or indonesian_part.startswith("\\")
                or dhao_part.startswith("\\")
            ):
                continue

            # Clean the Indonesian word
            indonesian_word = self.clean_text(indonesian_part)

            # Remove any remaining formatting
            indonesian_word = re.sub(r"^\W+", "", indonesian_word)
            indonesian_word = re.sub(r"\W+$", "", indonesian_word)

            # Extract Dhao word
            dhao_word = self.extract_dhao_word(dhao_part)

            # Only add if both words are meaningful and not formatting artifacts
            if (
                indonesian_word
                and dhao_word
                and len(indonesian_word) > 1
                and len(dhao_word) > 1
                and not indonesian_word.startswith("kb.")
                and not indonesian_word.startswith("kkt.")
                and not indonesian_word.startswith("kki.")
            ):
                pairs.append((indonesian_word, dhao_word))

        return pairs

    def extract_dictionary(self, content: str) -> List[Tuple[str, str]]:
        """Extract all dictionary entries from the content."""
        all_pairs = []

        # Find all \hline entries
        hline_matches = self.hline_pattern.findall(content)

        for hline_content in hline_matches:
            # Parse each hline entry
            pairs = self.parse_hline_entry(hline_content)
            all_pairs.extend(pairs)

        # Remove duplicates while preserving order
        seen = set()
        unique_pairs = []
        for indonesian, dhao in all_pairs:
            # Create a normalized version for duplicate checking
            normalized = (indonesian.lower().strip(), dhao.lower().strip())
            if normalized not in seen and indonesian and dhao:
                seen.add(normalized)
                unique_pairs.append((indonesian, dhao))

        return unique_pairs

    def save_to_csv(self, pairs: List[Tuple[str, str]], output_file: str):
        """Save dictionary pairs to CSV file."""
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(["Indonesian", "Dhao"])

            # Write dictionary entries
            for indonesian, dhao in pairs:
                writer.writerow([indonesian, dhao])

    def process_file(self, input_file: str, output_file: str):
        """Process the input file and create CSV output."""
        try:
            # Read input file
            with open(input_file, "r", encoding="utf-8") as f:
                content = f.read()

            print(f"Reading from: {input_file}")

            # Extract dictionary entries
            pairs = self.extract_dictionary(content)

            print(f"Extracted {len(pairs)} dictionary entries")

            # Save to CSV
            self.save_to_csv(pairs, output_file)

            print(f"Dictionary saved to: {output_file}")

            # Print some sample entries
            print("\nSample entries:")
            for i, (indonesian, dhao) in enumerate(pairs[:10]):
                print(f"{i+1:2d}. {indonesian:<20} -> {dhao}")

            if len(pairs) > 10:
                print(f"... and {len(pairs) - 10} more entries")

        except FileNotFoundError:
            print(f"Error: Input file '{input_file}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error processing file: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Extract dictionary from LaTeX format to CSV"
    )
    parser.add_argument("input_file", help="Input markdown file with LaTeX dictionary")
    parser.add_argument("output_file", help="Output CSV file")

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)

    # Create extractor and process file
    extractor = DictionaryExtractor()
    extractor.process_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
