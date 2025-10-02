"""
Glossary-based selection for post-editing.
"""

import re
from typing import List, Set
from core.data_models import GlossaryEntry

# Try to import spaCy for lemmatization
try:
    import spacy
    SPACY_AVAILABLE = True
    
    # Initialize Indonesian lemmatizer
    try:
        nlp = spacy.blank("id")
        nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
        nlp.initialize()
        print("✅ SpaCy Indonesian lemmatizer initialized for glossary matching")
    except Exception as e:
        print(f"⚠️ SpaCy lemmatizer initialization failed: {e}")
        nlp = None
        SPACY_AVAILABLE = False
except ImportError:
    print("⚠️ SpaCy not available. Install with: pip install spacy")
    SPACY_AVAILABLE = False
    nlp = None


class GlossarySelector:
    """
    Selects relevant glossary entries for given text.
    
    Supports three modes:
    - smart: Intelligent matching based on input text (optimized lookups)
    - full: Returns entire glossary for every request
    - word_fuzzy: Fuzzy matching per word, similar to word_parallel mode for few-shot examples
    """
    
    def __init__(self, glossary: List[GlossaryEntry], mode: str = "smart", top_n_per_word_glossary: int = 3):
        """
        Initialize glossary selector and build lookup dictionaries for fast matching.
        
        Args:
            glossary: List of glossary entries
            mode: Either "smart" (intelligent matching), "full" (return all entries), or "word_fuzzy" (fuzzy matching per word)
            top_n_per_word_glossary: For word_fuzzy mode, number of top matches to retrieve per word
        """
        self.glossary = glossary
        self.mode = mode
        self.top_n_per_word_glossary = top_n_per_word_glossary
        
        # Initialize max_ngram_length for all modes (needed for debugging output)
        self.max_ngram_length = 1
        
        # Import fuzzy matching functions from vectorizers
        from .vectorizers import _extract_words, _efficient_string_similarity
        self._extract_words = _extract_words
        self._efficient_string_similarity = _efficient_string_similarity
        
        if mode == "smart":
            # Build lookup dictionaries for efficient matching
            self._build_lookup_dictionaries()
            print(f"✅ Glossary selector initialized in smart mode with {len(glossary)} entries")
        elif mode == "full":
            # Calculate max_ngram_length for full mode too
            for entry in glossary:
                word_count = len(entry.source_word.lower().strip().split())
                self.max_ngram_length = max(self.max_ngram_length, word_count)
            print(f"✅ Glossary selector initialized in full mode with {len(glossary)} entries")
        elif mode == "word_fuzzy":
            # Calculate max_ngram_length for word_fuzzy mode too
            for entry in glossary:
                word_count = len(entry.source_word.lower().strip().split())
                self.max_ngram_length = max(self.max_ngram_length, word_count)
            print(f"✅ Glossary selector initialized in word_fuzzy mode with {len(glossary)} entries (top-{top_n_per_word_glossary} per word)")
        else:
            raise ValueError(f"Unsupported glossary mode: {mode}. Use 'smart', 'full', or 'word_fuzzy'.")
    
    def _build_lookup_dictionaries(self):
        """Build fast lookup dictionaries for different matching strategies including n-grams."""
        # Dictionary for exact matches: phrase -> list of entries
        self.exact_lookup = {}
        
        # Dictionary for base word matches (parentheses removed): phrase -> list of entries  
        self.base_lookup = {}
        
        # Dictionary for normalized matches: normalized_phrase -> list of entries
        self.normalized_lookup = {}
        
        # Track max n-gram length for efficient matching
        self.max_ngram_length = 1
        
        print(f"Building glossary lookup dictionaries from {len(self.glossary)} entries...")
        
        for entry in self.glossary:
            self._add_entry_to_lookups(entry)
        
        print(f"✅ Built lookup dictionaries: {len(self.exact_lookup)} exact, {len(self.base_lookup)} base, {len(self.normalized_lookup)} normalized")
        print(f"📏 Maximum n-gram length in glossary: {self.max_ngram_length} words")
    
    def _add_entry_to_lookups(self, entry: GlossaryEntry):
        """Add a single entry to all lookup dictionaries."""
        source_word = entry.source_word
        source_lower = source_word.lower().strip()
        
        # Count words to track max n-gram length
        word_count = len(source_lower.split())
        self.max_ngram_length = max(self.max_ngram_length, word_count)
        
        # 1. Exact lookup
        if source_lower not in self.exact_lookup:
            self.exact_lookup[source_lower] = []
        self.exact_lookup[source_lower].append(entry)
        
        # 2. Base word lookup (remove parentheses)
        base_phrase = self._extract_base_word(source_word).lower().strip()
        if base_phrase != source_lower:  # Only add if different from exact
            if base_phrase not in self.base_lookup:
                self.base_lookup[base_phrase] = []
            self.base_lookup[base_phrase].append(entry)
        
        # 3. Normalized lookup (remove dashes, replace with spaces)
        normalized_phrase = self._normalize_word(source_word).lower().strip()
        if normalized_phrase != source_lower and normalized_phrase != base_phrase:
            if normalized_phrase not in self.normalized_lookup:
                self.normalized_lookup[normalized_phrase] = []
            self.normalized_lookup[normalized_phrase].append(entry)
        
        # Also add normalized version of base phrase
        normalized_base = self._normalize_word(base_phrase).lower().strip()
        if (normalized_base != normalized_phrase and normalized_base != base_phrase and 
            normalized_base != source_lower):
            if normalized_base not in self.normalized_lookup:
                self.normalized_lookup[normalized_base] = []
            self.normalized_lookup[normalized_base].append(entry)
    
    def _lemmatize_word(self, word: str) -> str:
        """
        Lemmatize a word using spaCy if available.
        
        Args:
            word: Word to lemmatize
            
        Returns:
            Lemmatized word or original word if lemmatization fails
        """
        if not SPACY_AVAILABLE or nlp is None:
            return word.lower()
        
        try:
            doc = nlp(word)
            if doc and len(doc) > 0:
                lemma = doc[0].lemma_.lower()
                # Return lemma only if it's different and valid
                if lemma and lemma != word.lower() and lemma != "-PRON-":
                    return lemma
        except Exception:
            pass
        
        return word.lower()
    
    def _normalize_word(self, word: str) -> str:
        """
        Normalize word by removing dashes, replacing with spaces, and stripping.
        
        Examples:
        - 'self-driving' → 'self driving'
        - 'co-worker' → 'co worker'
        - ' spaced-word ' → 'spaced word'
        
        Args:
            word: Word to normalize
            
        Returns:
            Normalized word
        """
        # Replace dashes with spaces and strip
        normalized = word.replace('-', ' ').strip()
        # Collapse multiple spaces into single spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def get_relevant_entries(self, text: str, max_entries: int = None) -> List[GlossaryEntry]:
        """
        Get relevant glossary entries for the given text.
        
        Args:
            text: Input text to find glossary entries for
            max_entries: Maximum number of entries to return (None for all)
            
        Returns:
            List of relevant glossary entries
        """
        if not self.glossary:
            return []
        
        if self.mode == "full":
            # Return entire glossary (optionally limited)
            if max_entries is not None:
                return self.glossary[:max_entries]
            else:
                return self.glossary
        
        elif self.mode == "smart":
            # Use intelligent matching with optimized lookups
            return self._get_relevant_entries_smart(text, max_entries)
        
        elif self.mode == "word_fuzzy":
            # Use word-based fuzzy matching similar to word_parallel mode
            return self._get_relevant_entries_word_fuzzy(text, max_entries)
        
        else:
            raise ValueError(f"Unknown glossary mode: {self.mode}")
    
    def _get_relevant_entries_word_fuzzy(self, text: str, max_entries: int = None, similarity_threshold: float = 0.5, min_word_length: int = 2) -> List[GlossaryEntry]:
        """
        Get relevant glossary entries using word-based fuzzy matching.
        
        For each word in the input text, finds the top n glossary entries that best match
        that word using fuzzy string similarity, then aggregates scores and returns top entries.
        Similar to WordBasedParallelRetriever but for glossary entries.
        
        Args:
            text: Input text to find relevant glossary entries for
            max_entries: Maximum number of entries to return (None = all matched entries)
            similarity_threshold: Minimum similarity score to consider a match (0.0-1.0)
            min_word_length: Minimum word length to consider for matching
            
        Returns:
            List of relevant glossary entries
        """
        if not self.glossary:
            return []
        
        # Extract words from the input text
        query_words = self._extract_words(text, min_word_length)
        
        if not query_words:
            print("Warning: No valid words found in input text after filtering")
            # Fallback to first max_entries if no words found
            return self.glossary[:max_entries] if max_entries else self.glossary[:10]
        
        print(f"Word-fuzzy glossary matching with {len(query_words)} words: {query_words}")
        
        # For each word, find top n matching glossary entries
        word_entry_matches = {}  # word -> [(entry_idx, score), ...]
        entry_aggregate_scores = {}  # entry_idx -> aggregate_score
        
        for word in query_words:
            word_matches = []
            
            # Score all glossary entries for this word
            for entry_idx, entry in enumerate(self.glossary):
                # Score against the source word in the glossary entry
                word_score = self._score_entry_for_word(word, entry.source_word, similarity_threshold)
                if word_score > 0:
                    word_matches.append((entry_idx, word_score))
            
            # Sort by score and take top n for this word
            word_matches.sort(key=lambda x: x[1], reverse=True)
            top_matches_for_word = word_matches[:self.top_n_per_word_glossary]
            word_entry_matches[word] = top_matches_for_word
            
            # Aggregate scores for final ranking
            for entry_idx, score in top_matches_for_word:
                if entry_idx not in entry_aggregate_scores:
                    entry_aggregate_scores[entry_idx] = 0.0
                entry_aggregate_scores[entry_idx] += score
        
        # Debug information
        total_word_matches = sum(len(matches) for matches in word_entry_matches.values())
        unique_entries = len(entry_aggregate_scores)
        print(f"Found {total_word_matches} total word-entry matches across {unique_entries} unique entries")
        
        # Sort entries by aggregate score and return top entries
        sorted_entries = sorted(
            entry_aggregate_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Get indices - either top-max_entries or all available
        if max_entries is not None:
            top_indices = [idx for idx, score in sorted_entries[:max_entries]]
        else:
            top_indices = [idx for idx, score in sorted_entries]
        
        # Print some debugging info
        if sorted_entries:
            show_count = min(5, len(sorted_entries))
            print(f"Top entry aggregate scores: {[(idx, f'{score:.3f}') for idx, score in sorted_entries[:show_count]]}")
        
        # Show word-level breakdown for debugging
        if len(query_words) <= 10:  # Only show details for reasonable number of words
            print("Word-level matches breakdown:")
            for word, matches in word_entry_matches.items():
                if matches:
                    match_info = []
                    display_count = min(self.top_n_per_word_glossary, len(matches))
                    for entry_idx, score in matches[:display_count]:
                        entry = self.glossary[entry_idx]
                        match_info.append(f"{entry.source_word} ({score:.3f})")
                    print(f"  '{word}': {len(matches)} matches, top {display_count}: {match_info}")
        
        return [self.glossary[i] for i in top_indices]
    
    def _score_entry_for_word(self, word: str, entry_source_word: str, similarity_threshold: float = 0.5) -> float:
        """Score a glossary entry based on how well it matches a given word using efficient string similarity."""
        # Extract base word from entry (remove parentheses if present)
        base_entry_word = self._extract_base_word(entry_source_word)
        
        # Find the best match among the words in the base entry (in case it's a phrase)
        entry_words = self._extract_words(base_entry_word)
        
        best_similarity = 0.0
        for entry_word in entry_words:
            similarity = self._efficient_string_similarity(word, entry_word)
            if similarity > best_similarity:
                best_similarity = similarity
        
        # Only return score if it meets the threshold
        return best_similarity if best_similarity >= similarity_threshold else 0.0

    def _get_relevant_entries_smart(self, text: str, max_entries: int = None) -> List[GlossaryEntry]:
        """
        Get relevant glossary entries for given text using optimized n-gram dictionary lookups.
        Tries matching in order: exact → normalized → lemmatized, prioritizing longer matches.
        Does NOT remove duplicates - returns all matching entries.
        
        Args:
            text: Input text to find relevant glossary entries for
            max_entries: Maximum number of entries to return (None = all relevant entries)
            
        Returns:
            List of relevant glossary entries (may contain duplicates)
        """
        # Extract n-grams from text (includes unigrams, bigrams, trigrams, etc.)
        ngrams = self._extract_ngrams(text.lower())
        
        # Find matching entries using fast dictionary lookups
        relevant_entries = []
        matched_spans = set()  # Track which text spans have been matched to avoid overlaps
        
        # Sort n-grams by length (descending) to prioritize longer matches
        ngrams_by_length = {}
        for ngram in ngrams:
            length = len(ngram.split())
            if length not in ngrams_by_length:
                ngrams_by_length[length] = []
            ngrams_by_length[length].append(ngram)
        
        # Process n-grams from longest to shortest
        for length in sorted(ngrams_by_length.keys(), reverse=True):
            for ngram in ngrams_by_length[length]:
                # Skip if this n-gram overlaps with an already matched span
                if self._overlaps_with_matched_spans(ngram, matched_spans, text):
                    continue
                
                found_match = False
                
                # 1. Exact matching (fastest - direct dict lookup)
                if ngram in self.exact_lookup:
                    relevant_entries.extend(self.exact_lookup[ngram])
                    matched_spans.add(ngram)
                    found_match = True
                
                # Check base phrase lookups (parentheses removed)
                if ngram in self.base_lookup:
                    relevant_entries.extend(self.base_lookup[ngram])
                    matched_spans.add(ngram)
                    found_match = True
                
                # 2. Normalized matching (remove dashes, spaces)
                if not found_match:
                    normalized_ngram = self._normalize_word(ngram).lower().strip()
                    if normalized_ngram != ngram and normalized_ngram in self.normalized_lookup:
                        relevant_entries.extend(self.normalized_lookup[normalized_ngram])
                        matched_spans.add(ngram)
                        found_match = True
                
                # 3. Lemmatized matching (slowest - only if no other matches and single word)
                if not found_match and length == 1 and SPACY_AVAILABLE and nlp is not None:
                    lemma = self._lemmatize_word(ngram)
                    if lemma != ngram:
                        # Try lemma in exact lookup
                        if lemma in self.exact_lookup:
                            relevant_entries.extend(self.exact_lookup[lemma])
                            matched_spans.add(ngram)
                            found_match = True
                        # Try lemma in base lookup
                        elif lemma in self.base_lookup:
                            relevant_entries.extend(self.base_lookup[lemma])
                            matched_spans.add(ngram)
                            found_match = True
                        # Try lemma in normalized lookup
                        elif lemma in self.normalized_lookup:
                            relevant_entries.extend(self.normalized_lookup[lemma])
                            matched_spans.add(ngram)
                            found_match = True
        
        # Limit to max_entries if specified (but keep all duplicates within limit)
        if max_entries is not None:
            return relevant_entries[:max_entries]
        else:
            return relevant_entries
    
    def _overlaps_with_matched_spans(self, ngram: str, matched_spans: set, text: str) -> bool:
        """
        Check if an n-gram overlaps with already matched spans to avoid double-counting.
        
        Args:
            ngram: The n-gram to check
            matched_spans: Set of already matched n-grams
            text: Original text for position checking
            
        Returns:
            True if overlaps, False otherwise
        """
        # Simple approach: check if any word in the ngram is already part of a longer matched span
        ngram_words = set(ngram.split())
        
        for matched_span in matched_spans:
            matched_words = set(matched_span.split())
            if ngram_words & matched_words:  # If there's any word overlap
                # Only consider it an overlap if the matched span is longer
                if len(matched_span.split()) > len(ngram.split()):
                    return True
        
        return False
    
    def get_relevant_entries_debug(self, text: str, max_entries: int = None) -> tuple:
        """
        Get relevant glossary entries with debug information.
        
        Returns:
            Tuple of (entries, debug_info)
        """
        if not self.glossary:
            return [], {"total_words": 0, "exact_matches": 0, "normalized_matches": 0, "lemma_matches": 0, "total_entries": 0}
        
        if self.mode == "full":
            # Return entire glossary with simplified debug info
            entries = self.glossary[:max_entries] if max_entries else self.glossary
            # Extract words from text for consistency with smart mode debug output
            words = self._extract_words(text.lower())
            debug_info = {
                "mode": "full",
                "total_words": len(words),
                "words": words,
                "exact_matches": 0,
                "normalized_matches": 0,
                "lemma_matches": 0,
                "lemma_details": [],
                "match_details": [f"Full mode: returning all {len(entries)} entries"],
                "total_entries": len(entries),
                "sample_entries": [(e.source_word, e.target_word, e.pos_tag) for e in entries[:5]]
            }
            return entries, debug_info
        
        elif self.mode == "smart":
            # Use the detailed smart matching debug
            return self._get_relevant_entries_debug_smart(text, max_entries)
        
        elif self.mode == "word_fuzzy":
            # Use word-fuzzy matching with debug info
            return self._get_relevant_entries_debug_word_fuzzy(text, max_entries)
        
        else:
            raise ValueError(f"Unknown glossary mode: {self.mode}")
    
    def _get_relevant_entries_debug_word_fuzzy(self, text: str, max_entries: int = None) -> tuple:
        """
        Get relevant glossary entries with debug information using word-fuzzy matching.
        
        Returns:
            Tuple of (entries, debug_info)
        """
        # Get the entries using the regular method
        entries = self._get_relevant_entries_word_fuzzy(text, max_entries)
        
        # Extract words from text for debug output
        words = self._extract_words(text.lower())
        
        debug_info = {
            "mode": "word_fuzzy",
            "total_words": len(words),
            "words": words,
            "top_n_per_word": self.top_n_per_word_glossary,
            "exact_matches": 0,  # Not applicable for word_fuzzy mode
            "normalized_matches": 0,  # Not applicable for word_fuzzy mode
            "lemma_matches": 0,  # Not applicable for word_fuzzy mode
            "lemma_details": [],
            "match_details": [f"Word-fuzzy mode: returning {len(entries)} entries based on fuzzy word matching (top-{self.top_n_per_word_glossary} per word)"],
            "total_entries": len(entries),
            "sample_entries": [(e.source_word, e.target_word, e.pos_tag) for e in entries[:5]]
        }
        
        return entries, debug_info

    def _get_relevant_entries_debug_smart(self, text: str, max_entries: int = None) -> tuple:
        """
        Get relevant glossary entries with debug information using optimized n-gram lookups.
        
        Returns:
            Tuple of (entries, debug_info)
        """
        
        # Extract n-grams from text
        ngrams = self._extract_ngrams(text.lower())
        words = self._extract_words(text.lower())  # For backward compatibility in debug output
        
        # Track matching statistics
        relevant_entries = []
        matched_spans = set()
        exact_matches = 0
        normalized_matches = 0
        lemma_matches = 0
        lemma_details = []
        match_details = []
        ngram_details = []
        
        # Sort n-grams by length (descending) to prioritize longer matches
        ngrams_by_length = {}
        for ngram in ngrams:
            length = len(ngram.split())
            if length not in ngrams_by_length:
                ngrams_by_length[length] = []
            ngrams_by_length[length].append(ngram)
        
        # Process n-grams from longest to shortest
        for length in sorted(ngrams_by_length.keys(), reverse=True):
            for ngram in ngrams_by_length[length]:
                # Skip if this n-gram overlaps with an already matched span
                if self._overlaps_with_matched_spans(ngram, matched_spans, text):
                    continue
                
                found_match = False
                ngram_matches = []
                
                # 1. Exact matching
                if ngram in self.exact_lookup:
                    entries = self.exact_lookup[ngram]
                    relevant_entries.extend(entries)
                    exact_matches += len(entries)
                    matched_spans.add(ngram)
                    found_match = True
                    ngram_matches.extend([f"exact: {e.source_word}" for e in entries])
                
                # Check base phrase lookups (parentheses removed)
                if ngram in self.base_lookup:
                    entries = self.base_lookup[ngram]
                    relevant_entries.extend(entries)
                    exact_matches += len(entries)  # Count as exact since it's still direct lookup
                    matched_spans.add(ngram)
                    found_match = True
                    ngram_matches.extend([f"base: {e.source_word}" for e in entries])
                
                # 2. Normalized matching
                if not found_match:
                    normalized_ngram = self._normalize_word(ngram).lower().strip()
                    if normalized_ngram != ngram and normalized_ngram in self.normalized_lookup:
                        entries = self.normalized_lookup[normalized_ngram]
                        relevant_entries.extend(entries)
                        normalized_matches += len(entries)
                        matched_spans.add(ngram)
                        found_match = True
                        ngram_matches.extend([f"normalized: {e.source_word}" for e in entries])
                
                # 3. Lemmatized matching (only for single words)
                if not found_match and length == 1 and SPACY_AVAILABLE and nlp is not None:
                    lemma = self._lemmatize_word(ngram)
                    if lemma != ngram:
                        lemma_details.append(f"{ngram} → {lemma}")
                        
                        # Try lemma in all lookups
                        lemma_found = False
                        if lemma in self.exact_lookup:
                            entries = self.exact_lookup[lemma]
                            relevant_entries.extend(entries)
                            lemma_matches += len(entries)
                            matched_spans.add(ngram)
                            lemma_found = True
                            ngram_matches.extend([f"lemma-exact: {e.source_word}" for e in entries])
                        
                        if lemma in self.base_lookup:
                            entries = self.base_lookup[lemma]
                            relevant_entries.extend(entries)
                            lemma_matches += len(entries)
                            matched_spans.add(ngram)
                            lemma_found = True
                            ngram_matches.extend([f"lemma-base: {e.source_word}" for e in entries])
                        
                        if lemma in self.normalized_lookup:
                            entries = self.normalized_lookup[lemma]
                            relevant_entries.extend(entries)
                            lemma_matches += len(entries)
                            matched_spans.add(ngram)
                            lemma_found = True
                            ngram_matches.extend([f"lemma-norm: {e.source_word}" for e in entries])
                        
                        found_match = lemma_found
                
                if ngram_matches:
                    match_type = f"{length}-gram" if length > 1 else "word"
                    match_details.append(f"{ngram} ({match_type}): {', '.join(ngram_matches)}")
        
        # Limit results if specified
        if max_entries is not None:
            relevant_entries = relevant_entries[:max_entries]
        
        debug_info = {
            "total_words": len(words),
            "words": words,
            "total_ngrams": len(ngrams),
            "max_ngram_length": self.max_ngram_length,
            "exact_matches": exact_matches,
            "normalized_matches": normalized_matches,
            "lemma_matches": lemma_matches,
            "lemma_details": lemma_details,
            "match_details": match_details,
            "total_entries": len(relevant_entries),
            "sample_entries": [(e.source_word, e.target_word, e.pos_tag) for e in relevant_entries[:5]]
        }
        
        return relevant_entries, debug_info
    
    def get_entries_for_rows(self, rows: List, max_entries_per_row: int = None) -> List[List[GlossaryEntry]]:
        """
        Get relevant glossary entries for multiple rows.
        
        Args:
            rows: List of row objects with src_text attribute
            max_entries_per_row: Maximum number of entries per row (None = all relevant entries)
            
        Returns:
            List of glossary entry lists, one per row
        """
        print(f"Selecting glossary entries for {len(rows)} rows...")
        
        entries_list = []
        total_entries = 0
        
        for row in rows:
            entries = self.get_relevant_entries(row.src_text, max_entries_per_row)
            entries_list.append(entries)
            total_entries += len(entries)
        
        avg_entries = total_entries / len(rows) if rows else 0
        if max_entries_per_row is None:
            print(f"Selected {total_entries} total glossary entries (avg: {avg_entries:.1f} per row, all available entries)")
        else:
            print(f"Selected {total_entries} total glossary entries (avg: {avg_entries:.1f} per row, max: {max_entries_per_row})")
        
        if SPACY_AVAILABLE:
            print(f"📚 Using enhanced n-gram matching: exact + normalized + lemmatization (max {self.max_ngram_length}-grams, includes all duplicates)")
        else:
            print(f"📚 Using basic n-gram matching: exact + normalized only (max {self.max_ngram_length}-grams, includes all duplicates)")
        
        return entries_list
    
    def _extract_base_word(self, word: str) -> str:
        """
        Extract base word from dictionary entry, removing parenthetical information.
        
        Examples:
        - 'itu (jamak)' → 'itu'
        - 'cabut (rambut)' → 'cabut'
        - 'normal_word' → 'normal_word'
        
        Args:
            word: Dictionary word (potentially with parentheses)
            
        Returns:
            Base word without parenthetical information
        """
        # Remove parentheses and everything inside them, then strip whitespace
        base_word = re.sub(r'\s*\([^)]*\)', '', word).strip()
        return base_word if base_word else word  # Fallback to original if something goes wrong
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract individual words from text using regex."""
        # Extract words (letters and numbers, handling Unicode)
        words = re.findall(r'\b\w+\b', text, re.UNICODE)
        return [word.lower() for word in words]
    
    def _extract_ngrams(self, text: str, max_n: int = None) -> List[str]:
        """
        Extract n-grams of various lengths from text for matching against multi-word glossary entries.
        
        Args:
            text: Input text to extract n-grams from
            max_n: Maximum n-gram length (defaults to self.max_ngram_length)
            
        Returns:
            List of n-grams of lengths 1 to max_n
        """
        if max_n is None:
            max_n = getattr(self, 'max_ngram_length', 3)  # Default to trigrams if not set
        
        # Extract words first
        words = self._extract_words(text)
        
        if not words:
            return []
        
        ngrams = []
        
        # Generate n-grams of different lengths
        for n in range(1, min(len(words) + 1, max_n + 1)):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i + n])
                ngrams.append(ngram)
        
        return ngrams
