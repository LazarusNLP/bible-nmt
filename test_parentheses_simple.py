#!/usr/bin/env python3
"""
Simple test for parenthetical matching without full dependency stack.
"""

import re

def extract_base_word(word: str) -> str:
    """Extract base word from dictionary entry, removing parenthetical information."""
    base_word = re.sub(r'\s*\([^)]*\)', '', word).strip()
    return base_word if base_word else word

def test_parenthetical_extraction():
    """Test the parenthetical extraction functionality."""
    
    test_cases = [
        ("itu (jamak)", "itu"),
        ("cabut (rambut)", "cabut"),
        ("makan (daging)", "makan"),
        ("normal_word", "normal_word"),
        ("word (info1) (info2)", "word"),
        ("", ""),
        ("   spaced (info)   ", "spaced"),
    ]
    
    print("="*60)
    print("PARENTHETICAL EXTRACTION TEST")
    print("="*60)
    
    for original, expected in test_cases:
        result = extract_base_word(original)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{original}' → '{result}' (expected: '{expected}')")
    
    print("\n" + "="*60)
    print("MATCHING SIMULATION")
    print("="*60)
    
    # Simulate dictionary entries
    dict_entries = [
        "itu (jamak)",
        "cabut (rambut)", 
        "makan (daging)",
        "normal_word"
    ]
    
    # Simulate input words
    input_words = ["itu", "cabut", "makan", "normal_word", "other"]
    
    print("Dictionary entries with parentheses:")
    for entry in dict_entries:
        print(f"  - {entry}")
    
    print(f"\nInput words: {input_words}")
    print("\nMatching results:")
    
    for word in input_words:
        matches = []
        for entry in dict_entries:
            base_word = extract_base_word(entry)
            if word.lower() == base_word.lower() or word.lower() == entry.lower():
                matches.append(entry)
        
        if matches:
            print(f"  '{word}' matches: {matches}")
        else:
            print(f"  '{word}' matches: (none)")
    
    print("="*60)
    print("✅ Test complete!")
    
    print("\nKey insights:")
    print("• 'itu' matches 'itu (jamak)' - contextual info preserved")
    print("• 'cabut' matches 'cabut (rambut)' - semantic context retained")  
    print("• Base word extraction allows flexible matching")
    print("• Full entries with parentheses passed to LLM for context")


if __name__ == "__main__":
    test_parenthetical_extraction()

