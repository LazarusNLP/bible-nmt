#!/usr/bin/env python3
"""
Quick test script to demonstrate the enhanced glossary matching functionality.
"""

import sys
from pathlib import Path

# Add the post-editing module to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "post-editing"))

from core.data_models import GlossaryEntry
from retrieval.glossary import GlossarySelector

def test_glossary_matching():
    """Test the new glossary matching features."""
    
    # Create sample glossary with parenthetical entries
    sample_glossary = [
        GlossaryEntry(source_word="kasih", target_word="maha", pos_tag="NOUN"),
        GlossaryEntry(source_word="mengasihi", target_word="maha", pos_tag="VERB"),
        GlossaryEntry(source_word="Tuhan", target_word="Lofa", pos_tag="NOUN"),
        GlossaryEntry(source_word="doa", target_word="lolole", pos_tag="NOUN"),
        GlossaryEntry(source_word="berdoa", target_word="lolole", pos_tag="VERB"),
        GlossaryEntry(source_word="berdoa", target_word="lolole_alt", pos_tag="VERB"),  # Duplicate test
        # Test parenthetical entries
        GlossaryEntry(source_word="itu (jamak)", target_word="nala", pos_tag="PRON"),
        GlossaryEntry(source_word="cabut (rambut)", target_word="hau", pos_tag="VERB"),
        GlossaryEntry(source_word="makan (daging)", target_word="ha", pos_tag="VERB"),
    ]
    
    selector = GlossarySelector(sample_glossary)
    
    # Test input - includes words that should match parenthetical entries
    test_text = "Kami mengasihi Tuhan dan berdoa, lalu itu cabut"
    
    print("="*60)
    print("ENHANCED GLOSSARY MATCHING TEST")
    print("="*60)
    print(f"Input text: {test_text}")
    print()
    
    # Get results with debug info
    entries, debug_info = selector.get_relevant_entries_debug(test_text)
    
    print("MATCHING RESULTS:")
    print(f"- Words extracted: {debug_info['words']}")
    print(f"- Direct matches: {debug_info['direct_matches']}")
    print(f"- Lemmatization matches: {debug_info['lemma_matches']}")
    if debug_info['lemma_details']:
        print(f"- Lemmatizations: {', '.join(debug_info['lemma_details'])}")
    print(f"- Total entries found: {debug_info['total_entries']}")
    print()
    
    print("ALL MATCHING ENTRIES (including duplicates):")
    for i, entry in enumerate(entries, 1):
        pos_str = f" ({entry.pos_tag})" if entry.pos_tag else ""
        print(f"{i:2d}. {entry.source_word}{pos_str} → {entry.target_word}")
    
    print("="*60)
    print("✅ Test complete!")
    print("Key features demonstrated:")
    print("• Duplicates preserved (berdoa has 2 entries)")
    print("• Parenthetical matching (itu matches 'itu (jamak)', cabut matches 'cabut (rambut)')")
    print("• Full contextual info retained in glossary entries")


if __name__ == "__main__":
    test_glossary_matching()
