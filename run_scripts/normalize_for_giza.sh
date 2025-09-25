#!/bin/bash

# Normalize parallel text files for GIZA++ word alignment training
# This script applies optimized normalization that preserves contractions
#
# Key features:
# - Preserves contractions (don't, won't, God's, etc.)
# - Removes most punctuation except apostrophes in contractions
# - Converts to lowercase
# - Normalizes numbers to <NUM> tokens
# - Filters very short sentences

# Input files
SOURCE_FILE="/Users/davidsamuel/Documents/Unimelb Masters/Thesis/bible-nmt/ebible-corpus/eng-webp/eng-webp.txt"
TARGET_FILE="/Users/davidsamuel/Documents/Unimelb Masters/Thesis/bible-nmt/ebible-corpus/dhao-eng/nfa-nfa.txt"

# Output files
OUTPUT_SOURCE="ebible-corpus/dhao-eng/train.en-webp"
OUTPUT_TARGET="ebible-corpus/dhao-eng/train.nfa"

echo "Normalizing parallel text for GIZA++ training..."
echo "Features: Preserves contractions, removes punctuation, normalizes case"

python scripts/normalize_for_giza.py \
    "$SOURCE_FILE" \
    "$TARGET_FILE" \
    --output-source "$OUTPUT_SOURCE" \
    --output-target "$OUTPUT_TARGET" \
    --min-length 3

echo "GIZA++ normalization completed!"
echo "Files ready for alignment training:"
echo "  Source: $OUTPUT_SOURCE"
echo "  Target: $OUTPUT_TARGET"
