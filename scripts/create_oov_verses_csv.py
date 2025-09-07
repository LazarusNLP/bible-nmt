#!/usr/bin/env python3

import csv
import re
import argparse
import os
from collections import Counter, defaultdict

try:
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except (ImportError, ValueError) as e:
    NLTK_AVAILABLE = False
    print(f"Warning: NLTK not available ({str(e)[:100]}...), using simple tokenization")

def tokenize_text(text):
    """Tokenize text using NLTK if available, otherwise use simple regex"""
        # Use the same tokenization as the notebook
        # First replace curly double quotes
    text = text.replace('“', '"')
    # Then replace curly single quotes (right single quotation mark)
    text = text.replace('’', "'")  # Unicode for right single quotation mark
    normalized_tokens = word_tokenize(text)
    clean_tokens = [word.lower() for word in normalized_tokens if word.isalpha()]
    return clean_tokens


def get_oov_words(ind_ot_path, ind_nt_path):
    """Get top 100 OOV words from Genesis that don't appear in NT"""
    print("Loading Indonesian OT data...")
    ot_words = []
    with open(ind_ot_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'GEN' in row['verse']:
                words = tokenize_text(row['text'])
                ot_words.extend(words)
    
    print("Loading Indonesian NT data...")
    nt_words = []
    with open(ind_nt_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            words = tokenize_text(row['text'])
            nt_words.extend(words)
    
    # Only keep words in ot_words that are not in nt_words
    nt_words_set = set(nt_words)
    oov_words = [w for w in ot_words if w not in nt_words_set]
    
    word_counts = Counter(oov_words)
    most_common = word_counts.most_common(100)  # Top 100 words
    
    print(f"Found {len(most_common)} OOV words")
    return [word for word, count in most_common]

def load_verses_data(ind_ot_path, ind_nt_path, dhao_ot_path, dhao_nt_path):
    """Load all verse data from Indonesian and Dhao files"""
    print("Loading Indonesian verses...")
    indonesian_verses = {}
    
    # Load Indonesian OT
    with open(ind_ot_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            indonesian_verses[row['verse']] = row['text']
    
    # Load Indonesian NT
    with open(ind_nt_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            indonesian_verses[row['verse']] = row['text']
    
    print("Loading Dhao verses...")
    dhao_verses = {}
    
    # Load Dhao OT
    with open(dhao_ot_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dhao_verses[row['verse']] = row['text']
    
    # Load Dhao NT
    with open(dhao_nt_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dhao_verses[row['verse']] = row['text']
    
    return indonesian_verses, dhao_verses

def expand_verse_range(verse_ref):
    """Expand verse ranges like 'GEN 1:1-3' to ['GEN 1:1', 'GEN 1:2', 'GEN 1:3']"""
    if '-' not in verse_ref:
        return [verse_ref]
    
    # Parse verse range
    match = re.match(r'([A-Z0-9]+)\s+(\d+):(\d+)-(\d+)', verse_ref)
    if not match:
        return [verse_ref]  # Return original if we can't parse
    
    book, chapter, start_verse, end_verse = match.groups()
    verses = []
    for v in range(int(start_verse), int(end_verse) + 1):
        verses.append(f"{book} {chapter}:{v}")
    
    return verses

def find_verse_with_word(word, verses_dict):
    """Find a verse that contains the given word"""
    for verse_ref, text in verses_dict.items():
        if word.lower() in tokenize_text(text):
            return verse_ref, text
    return None, None

def align_verses(dhao_verse_ref, indonesian_verses):
    """Get aligned Indonesian verses for a Dhao verse reference"""
    expanded_verses = expand_verse_range(dhao_verse_ref)
    aligned_texts = []
    
    for verse_ref in expanded_verses:
        if verse_ref in indonesian_verses:
            aligned_texts.append(indonesian_verses[verse_ref])
        else:
            print(f"Warning: Verse {verse_ref} not found in Indonesian data")
    
    return " ".join(aligned_texts)

def create_oov_csv(ind_ot_path, ind_nt_path, dhao_ot_path, dhao_nt_path, output_path):
    """Create CSV with OOV words and aligned verses"""
    print("Getting OOV words...")
    oov_words = get_oov_words(ind_ot_path, ind_nt_path)
    
    print("Loading verse data...")
    indonesian_verses, dhao_verses = load_verses_data(ind_ot_path, ind_nt_path, dhao_ot_path, dhao_nt_path)
    
    print("Creating aligned verses CSV...")
    output_rows = []
    
    for word in oov_words:
        print(f"Processing word: {word}")
        
        # Find Indonesian verse with this word
        ind_verse_ref, ind_text = find_verse_with_word(word, indonesian_verses)
        if not ind_verse_ref:
            print(f"Warning: Could not find Indonesian verse for word '{word}'")
            continue
        
        # Find corresponding Dhao verse
        dhao_text = None
        dhao_verse_ref = None
        
        # First try exact match
        if ind_verse_ref in dhao_verses:
            dhao_text = dhao_verses[ind_verse_ref]
            dhao_verse_ref = ind_verse_ref
        else:
            # Try to find in Dhao verses that might be ranges
            for dhao_ref, dhao_content in dhao_verses.items():
                expanded = expand_verse_range(dhao_ref)
                if ind_verse_ref in expanded:
                    dhao_text = dhao_content
                    dhao_verse_ref = dhao_ref
                    break
        
        if not dhao_text:
            print(f"Warning: Could not find Dhao verse for {ind_verse_ref}")
            continue
        
        # If Dhao verse is a range, get aligned Indonesian text
        if '-' in dhao_verse_ref:
            aligned_ind_text = align_verses(dhao_verse_ref, indonesian_verses)
        else:
            aligned_ind_text = ind_text
        
        output_rows.append({
            'oov_word': word,
            'verse_reference': dhao_verse_ref,
            'indonesian_text': aligned_ind_text,
            'dhao_text': dhao_text
        })
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['oov_word', 'verse_reference', 'indonesian_text', 'dhao_text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    
    print(f"Created {output_path} with {len(output_rows)} entries")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Create CSV with OOV words and aligned verses from Indonesian and Dhao')
    parser.add_argument('--ind-ot', required=True, help='Path to Indonesian Old Testament CSV file')
    parser.add_argument('--ind-nt', required=True, help='Path to Indonesian New Testament CSV file')
    parser.add_argument('--dhao-ot', required=True, help='Path to Dhao Old Testament CSV file')
    parser.add_argument('--dhao-nt', required=True, help='Path to Dhao New Testament CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Validate input files exist
    for file_path in [args.ind_ot, args.ind_nt, args.dhao_ot, args.dhao_nt]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    create_oov_csv(args.ind_ot, args.ind_nt, args.dhao_ot, args.dhao_nt, args.output)

if __name__ == "__main__":
    main()
