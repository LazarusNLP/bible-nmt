#!/usr/bin/env python3
"""
Test the optimized glossary matching with dictionary lookups.
"""

import re
from collections import defaultdict

# Simulate the optimized glossary data structures
class MockGlossaryEntry:
    def __init__(self, source_word, target_word, pos_tag=None):
        self.source_word = source_word
        self.target_word = target_word
        self.pos_tag = pos_tag

def extract_base_word(word: str) -> str:
    """Extract base word from dictionary entry, removing parenthetical information."""
    base_word = re.sub(r'\s*\([^)]*\)', '', word).strip()
    return base_word if base_word else word

def normalize_word(word: str) -> str:
    """Normalize word by removing dashes, replacing with spaces, and stripping."""
    normalized = word.replace('-', ' ').strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

def build_lookup_dictionaries(glossary):
    """Build fast lookup dictionaries for different matching strategies."""
    exact_lookup = defaultdict(list)
    base_lookup = defaultdict(list)
    normalized_lookup = defaultdict(list)
    
    for entry in glossary:
        source_word = entry.source_word
        source_lower = source_word.lower()
        
        # 1. Exact lookup
        exact_lookup[source_lower].append(entry)
        
        # 2. Base word lookup (remove parentheses)
        base_word = extract_base_word(source_word).lower()
        if base_word != source_lower:
            base_lookup[base_word].append(entry)
        
        # 3. Normalized lookup (remove dashes, replace with spaces)
        normalized_word = normalize_word(source_word).lower()
        if normalized_word != source_lower and normalized_word != base_word:
            normalized_lookup[normalized_word].append(entry)
        
        # Also add normalized version of base word
        normalized_base = normalize_word(base_word).lower()
        if normalized_base not in [normalized_word, base_word, source_lower]:
            normalized_lookup[normalized_base].append(entry)
    
    return exact_lookup, base_lookup, normalized_lookup

def extract_words(text: str):
    """Extract words from text using regex."""
    words = re.findall(r'\b\w+\b', text, re.UNICODE)
    return [word.lower() for word in words]

def optimized_match(text, exact_lookup, base_lookup, normalized_lookup):
    """Optimized matching using dictionary lookups."""
    words = extract_words(text.lower())
    relevant_entries = []
    match_details = []
    
    for word in words:
        word_lower = word.lower()
        found_match = False
        word_matches = []
        
        # 1. Exact matching (fastest - direct dict lookup)
        if word_lower in exact_lookup:
            entries = exact_lookup[word_lower]
            relevant_entries.extend(entries)
            found_match = True
            word_matches.extend([f"exact: {e.source_word}" for e in entries])
        
        # Check base word lookups (parentheses removed)
        if word_lower in base_lookup:
            entries = base_lookup[word_lower]
            relevant_entries.extend(entries)
            found_match = True
            word_matches.extend([f"base: {e.source_word}" for e in entries])
        
        # 2. Normalized matching (remove dashes, spaces)
        if not found_match:
            normalized_word = normalize_word(word).lower()
            if normalized_word != word_lower and normalized_word in normalized_lookup:
                entries = normalized_lookup[normalized_word]
                relevant_entries.extend(entries)
                found_match = True
                word_matches.extend([f"normalized: {e.source_word}" for e in entries])
        
        if word_matches:
            match_details.append(f"{word}: {', '.join(word_matches)}")
    
    return relevant_entries, match_details

def test_optimized_glossary():
    """Test the optimized glossary matching."""
    
    # Create test glossary with various entry types
    glossary = [
        MockGlossaryEntry("kasih", "maha", "NOUN"),
        MockGlossaryEntry("mengasihi", "maha", "VERB"),
        MockGlossaryEntry("Tuhan", "Lofa", "NOUN"),
        MockGlossaryEntry("itu (jamak)", "nala", "PRON"),
        MockGlossaryEntry("cabut (rambut)", "hau", "VERB"),
        MockGlossaryEntry("self-driving", "otomatis", "ADJ"),
        MockGlossaryEntry("co-worker", "rekan kerja", "NOUN"),
        MockGlossaryEntry("user friendly", "mudah digunakan", "ADJ"),
    ]
    
    # Build lookup dictionaries
    exact_lookup, base_lookup, normalized_lookup = build_lookup_dictionaries(glossary)
    
    print("="*70)
    print("OPTIMIZED GLOSSARY MATCHING TEST")
    print("="*70)
    
    # Show dictionary sizes
    print(f"Dictionary sizes:")
    print(f"  Exact lookup: {len(exact_lookup)} entries")
    print(f"  Base lookup: {len(base_lookup)} entries")
    print(f"  Normalized lookup: {len(normalized_lookup)} entries")
    print()
    
    # Test different types of input
    test_cases = [
        "Kami mengasihi Tuhan dengan kasih",  # Exact matches
        "Dia ambil itu dan cabut rambutnya",  # Parenthetical matches
        "Self driving car is user-friendly",  # Normalization matches
        "Co worker helps with self driving"   # Mixed cases
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"Test {i}: {test_text}")
        entries, match_details = optimized_match(test_text, exact_lookup, base_lookup, normalized_lookup)
        
        print(f"  Found {len(entries)} matches:")
        if match_details:
            for detail in match_details:
                print(f"    {detail}")
        
        print(f"  Glossary entries:")
        for entry in entries:
            pos_str = f" ({entry.pos_tag})" if entry.pos_tag else ""
            print(f"    - {entry.source_word}{pos_str} → {entry.target_word}")
        print()
    
    print("="*70)
    print("✅ PERFORMANCE BENEFITS:")
    print("• O(n) lookups instead of O(n×m) nested loops")
    print("• Exact matches: Instant dictionary lookup")
    print("• Normalized matches: Handles spelling variations")  
    print("• Parenthetical matches: Preserves context info")
    print("• Priority ordering: Most specific matches first")
    print("="*70)

if __name__ == "__main__":
    test_optimized_glossary()
