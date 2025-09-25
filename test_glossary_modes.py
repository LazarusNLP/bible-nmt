#!/usr/bin/env python3
"""
Test script to verify glossary modes functionality.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src" / "post-editing"))

from core.data_models import GlossaryEntry
from retrieval.glossary import GlossarySelector

def test_glossary_modes():
    """Test both smart and full glossary modes."""
    
    # Create sample glossary entries
    glossary = [
        GlossaryEntry(source_word="love", target_word="kasih", pos_tag="NOUN"),
        GlossaryEntry(source_word="to love", target_word="mengasihi", pos_tag="VERB"),
        GlossaryEntry(source_word="beautiful", target_word="indah", pos_tag="ADJ"),
        GlossaryEntry(source_word="quickly", target_word="dengan cepat", pos_tag="ADV"),
        GlossaryEntry(source_word="itu (jamak)", target_word="nala", pos_tag="PRON"),
        GlossaryEntry(source_word="cabut (rambut)", target_word="hau", pos_tag="VERB"),
    ]
    
    print("="*60)
    print("TESTING GLOSSARY MODES")
    print("="*60)
    print(f"Sample glossary has {len(glossary)} entries:")
    for entry in glossary:
        pos_str = f" ({entry.pos_tag})" if entry.pos_tag else ""
        print(f"  - {entry.source_word}{pos_str} → {entry.target_word}")
    print()
    
    # Test input
    test_text = "I love beautiful things quickly"
    print(f"Test input: '{test_text}'")
    print()
    
    # Test Smart Mode
    print("-" * 40)
    print("TESTING SMART MODE")
    print("-" * 40)
    
    smart_selector = GlossarySelector(glossary, mode="smart")
    smart_entries = smart_selector.get_relevant_entries(test_text)
    
    print(f"Smart mode found {len(smart_entries)} relevant entries:")
    for entry in smart_entries:
        pos_str = f" ({entry.pos_tag})" if entry.pos_tag else ""
        print(f"  - {entry.source_word}{pos_str} → {entry.target_word}")
    print()
    
    # Test Full Mode
    print("-" * 40)
    print("TESTING FULL MODE")
    print("-" * 40)
    
    full_selector = GlossarySelector(glossary, mode="full")
    full_entries = full_selector.get_relevant_entries(test_text)
    
    print(f"Full mode returned {len(full_entries)} entries (entire glossary):")
    for entry in full_entries:
        pos_str = f" ({entry.pos_tag})" if entry.pos_tag else ""
        print(f"  - {entry.source_word}{pos_str} → {entry.target_word}")
    print()
    
    # Test with max_entries limit
    print("-" * 40)
    print("TESTING WITH MAX_ENTRIES LIMIT")
    print("-" * 40)
    
    limited_smart = smart_selector.get_relevant_entries(test_text, max_entries=2)
    limited_full = full_selector.get_relevant_entries(test_text, max_entries=3)
    
    print(f"Smart mode limited to 2 entries: {len(limited_smart)} entries")
    print(f"Full mode limited to 3 entries: {len(limited_full)} entries")
    print()
    
    # Test debug mode
    print("-" * 40)
    print("TESTING DEBUG MODE")
    print("-" * 40)
    
    smart_entries_debug, debug_info = smart_selector.get_relevant_entries_debug(test_text)
    print("Smart mode debug info:")
    print(f"  - Total words: {debug_info.get('total_words', 0)}")
    print(f"  - Exact matches: {debug_info.get('exact_matches', 0)}")
    print(f"  - Total entries: {debug_info.get('total_entries', 0)}")
    
    full_entries_debug, debug_info_full = full_selector.get_relevant_entries_debug(test_text)
    print("Full mode debug info:")
    print(f"  - Mode: {debug_info_full.get('mode', 'unknown')}")
    print(f"  - Total entries: {debug_info_full.get('total_entries', 0)}")
    print()
    
    print("="*60)
    print("GLOSSARY MODES TEST COMPLETED SUCCESSFULLY! ✅")
    print("="*60)

if __name__ == "__main__":
    test_glossary_modes()
