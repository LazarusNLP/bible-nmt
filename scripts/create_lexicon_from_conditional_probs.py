import math
import csv
import argparse
import string
from collections import defaultdict


def is_punctuation(word):
    """Check if a word is primarily punctuation"""
    return all(c in string.punctuation for c in word) if word else False


def normalize_punctuation(word):
    """Normalize common punctuation variations"""
    # Normalize quotation marks
    word = word.replace("“", '"')
    word = word.replace("’", "'")
    return word


def preprocess_text_for_alignment(text):
    """
    Preprocess text for better word alignment quality.
    This function can be used before running fast_align.

    Args:
        text: Input text string

    Returns:
        Preprocessed text string with normalized punctuation and tokenization
    """
    try:
        from nltk.tokenize import word_tokenize

        NLTK_AVAILABLE = True
    except ImportError:
        NLTK_AVAILABLE = False

    # Normalize punctuation
    text = normalize_punctuation(text)

    if NLTK_AVAILABLE:
        # Use NLTK tokenization (separates punctuation)
        tokens = word_tokenize(text)
        return " ".join(tokens)
    else:
        # Simple fallback tokenization
        import re

        # Add spaces around punctuation
        text = re.sub(r"([.!?,:;])", r" \1 ", text)
        # Clean up multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()


def convert_conditional_probs_to_lexicon(
    conditional_probs_file,
    output_csv,
    min_prob=0.01,
    exclude_null=True,
    exclude_punctuation=False,
    normalize_punct=True,
):
    """
    Convert fast_align conditional probability table to bilingual lexicon CSV

    Args:
        conditional_probs_file: Input file from fast_align -p option
        output_csv: Output CSV file
        min_prob: Minimum probability threshold (default: 0.01 = 1%)
        exclude_null: Whether to exclude <eps> (null) alignments
        exclude_punctuation: Whether to exclude punctuation-only alignments
        normalize_punct: Whether to normalize punctuation variations
    """
    print(f"Reading conditional probabilities from: {conditional_probs_file}")

    lexicon_entries = []
    total_entries = 0
    filtered_entries = 0
    punctuation_entries = 0

    with open(conditional_probs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 3:
                continue

            source_word, target_word, log_prob = parts
            total_entries += 1

            # Skip null alignments if requested
            if exclude_null and (source_word == "<eps>" or target_word == "<eps>"):
                filtered_entries += 1
                continue

            # Skip punctuation alignments if requested
            if exclude_punctuation and (
                is_punctuation(source_word) or is_punctuation(target_word)
            ):
                punctuation_entries += 1
                filtered_entries += 1
                continue

            # Normalize punctuation if requested
            if normalize_punct:
                source_word = normalize_punctuation(source_word)
                target_word = normalize_punctuation(target_word)

            try:
                probability = math.exp(float(log_prob))

                # Probabilities must be between 0 and 1. If not, the input file is wrong.
                if not (0.0 <= probability <= 1.0):
                    filtered_entries += 1
                    continue

                # Filter by minimum probability
                if probability >= min_prob:
                    lexicon_entries.append(
                        {
                            "source_word": source_word,
                            "target_word": target_word,
                            "confidence_score": probability,
                        }
                    )
                else:
                    filtered_entries += 1

            except ValueError:
                filtered_entries += 1
                continue

    # Keep only the highest confidence target word for each source word
    source_word_best = {}
    for entry in lexicon_entries:
        source_word = entry["source_word"]
        confidence = entry["confidence_score"]

        if (
            source_word not in source_word_best
            or confidence > source_word_best[source_word]["confidence_score"]
        ):
            source_word_best[source_word] = entry

    # Convert back to list and sort by confidence score (lowest first)
    lexicon_entries = list(source_word_best.values())
    lexicon_entries.sort(key=lambda x: x["confidence_score"], reverse=False)

    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["source_word", "target_word", "confidence_score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for entry in lexicon_entries:
            writer.writerow(entry)

    print(f"\n=== LEXICON EXTRACTION RESULTS ===")
    print(f"Total entries processed: {total_entries:,}")
    print(f"Entries filtered out: {filtered_entries:,}")
    if exclude_punctuation:
        print(f"  - Punctuation entries: {punctuation_entries:,}")
    print(f"Final lexicon size: {len(lexicon_entries):,}")
    print(f"Output written to: {output_csv}")

    # Count punctuation vs content words in final lexicon
    punct_in_lexicon = sum(
        1
        for entry in lexicon_entries
        if is_punctuation(entry["source_word"]) or is_punctuation(entry["target_word"])
    )
    content_in_lexicon = len(lexicon_entries) - punct_in_lexicon

    print(f"Lexicon composition:")
    print(f"  - Content word translations: {content_in_lexicon:,}")
    print(f"  - Punctuation alignments: {punct_in_lexicon:,}")

    # Show top 10 translations
    print(f"\n=== TOP 10 LOWEST CONFIDENCE TRANSLATIONS ===")
    for i, entry in enumerate(lexicon_entries[:10], 1):
        print(
            f"{i:2d}. {entry['source_word']} → {entry['target_word']} "
            f"(confidence: {entry['confidence_score']:.4f})"
        )

    return lexicon_entries


def analyze_lexicon_stats(lexicon_entries):
    """Print detailed statistics about the lexicon"""
    if not lexicon_entries:
        return

    # Confidence score distribution
    scores = [entry["confidence_score"] for entry in lexicon_entries]

    print(f"\n=== CONFIDENCE SCORE STATISTICS ===")
    print(f"Highest confidence: {max(scores):.4f}")
    print(f"Lowest confidence: {min(scores):.4f}")
    print(f"Average confidence: {sum(scores)/len(scores):.4f}")

    # Count unique source and target words
    source_words = set(entry["source_word"] for entry in lexicon_entries)
    target_words = set(entry["target_word"] for entry in lexicon_entries)

    print(f"Unique source words: {len(source_words):,}")
    print(f"Unique target words: {len(target_words):,}")

    # Confidence ranges - updated for more realistic ranges
    extreme_high = len([s for s in scores if s >= 100])
    high_conf = len([s for s in scores if 10 <= s < 100])
    med_conf = len([s for s in scores if 1 <= s < 10])
    low_conf = len([s for s in scores if s < 1])

    print(f"\nConfidence Distribution:")
    print(
        f"  Extreme (≥100):   {extreme_high:,} ({extreme_high/len(scores)*100:.1f}%) - Potentially problematic"
    )
    print(f"  High (10-100):    {high_conf:,} ({high_conf/len(scores)*100:.1f}%)")
    print(f"  Medium (1-10):    {med_conf:,} ({med_conf/len(scores)*100:.1f}%)")
    print(f"  Low (<1):         {low_conf:,} ({low_conf/len(scores)*100:.1f}%)")

    # Analyze target word frequency (potential over-alignment)
    target_word_counts = {}
    for entry in lexicon_entries:
        target = entry["target_word"]
        target_word_counts[target] = target_word_counts.get(target, 0) + 1

    # Find target words that appear very frequently (potential over-alignment)
    over_aligned = [
        (word, count) for word, count in target_word_counts.items() if count >= 5
    ]
    over_aligned.sort(key=lambda x: x[1], reverse=True)

    if over_aligned:
        print(f"\n=== POTENTIALLY OVER-ALIGNED TARGET WORDS ===")
        print(
            "(Target words aligned to many different source words - often problematic)"
        )
        for word, count in over_aligned[:10]:
            print(f"  '{word}' aligned to {count} different source words")

    # Show examples of different confidence levels
    sorted_entries = sorted(lexicon_entries, key=lambda x: x["confidence_score"])

    print(f"\n=== EXAMPLES BY CONFIDENCE LEVEL ===")
    print("Lowest confidence examples (often more accurate):")
    for entry in sorted_entries[:5]:
        print(
            f"  {entry['source_word']} → {entry['target_word']} (conf: {entry['confidence_score']:.4f})"
        )

    print("\nHighest confidence examples (check for over-alignment):")
    for entry in sorted_entries[-5:]:
        print(
            f"  {entry['source_word']} → {entry['target_word']} (conf: {entry['confidence_score']:.4f})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert fast_align conditional probabilities to bilingual lexicon CSV"
    )
    parser.add_argument(
        "conditional_probs_file",
        help="Input conditional probabilities file from fast_align -p",
    )
    parser.add_argument("output_csv", help="Output CSV file")
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.00,
        help="Minimum probability threshold (default: 0.01)",
    )
    parser.add_argument(
        "--include-null", action="store_true", help="Include <eps> (null) alignments"
    )
    parser.add_argument(
        "--exclude-punctuation",
        action="store_true",
        help="Exclude punctuation-only alignments from lexicon",
    )
    parser.add_argument(
        "--no-normalize-punct",
        action="store_true",
        help="Don't normalize punctuation variations (quotes, dashes, etc.)",
    )

    args = parser.parse_args()

    lexicon_entries = convert_conditional_probs_to_lexicon(
        args.conditional_probs_file,
        args.output_csv,
        args.min_prob,
        exclude_null=not args.include_null,
        exclude_punctuation=args.exclude_punctuation,
        normalize_punct=not args.no_normalize_punct,
    )

    analyze_lexicon_stats(lexicon_entries)


if __name__ == "__main__":
    main()
