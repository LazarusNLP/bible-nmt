"""
Text normalization script based on silnlp preprocessing framework.
Provides comprehensive text cleaning and normalization for Bible translation tasks.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import regex

# Optional dependency for Moses punctuation normalization
try:
    from sacremoses import MosesPunctNormalizer
    MOSES_AVAILABLE = True
except ImportError:
    print("Warning: sacremoses not available. Install with: pip install sacremoses")
    MOSES_AVAILABLE = False

logger = logging.getLogger(__name__)


class PunctuationCategory(Enum):
    """Categories for punctuation normalization."""
    LEFT_CLINGING = "LEFT_CLINGING"
    RIGHT_CLINGING = "RIGHT_CLINGING"
    LEFT_RIGHT_CLINGING = "LEFT_RIGHT_CLINGING"
    UNCLINGING = "UNCLINGING"


@dataclass(frozen=True)
class PunctuationNormalizationRule:
    """Rule for normalizing specific punctuation characters."""
    character: str  # length 1
    category: PunctuationCategory


@dataclass(frozen=True)
class StringSlice:
    """Represents a section of a string."""
    start_index: int
    end_index: int
    slice: str
    outer: str

    def length(self) -> int:
        return self.end_index - self.start_index


@dataclass(frozen=True)
class SentenceTransformation:
    """A representation of a delta applied to a sentence at a particular position."""
    slice: StringSlice
    replacement: str
    description: str


@dataclass(frozen=True)
class SentenceNormalizationSummary:
    """A summary of the normalization process for a particular sentence."""
    original_sentence: str
    normalized_sentence: str
    transformations: List[SentenceTransformation]


class TextNormalizer:
    """
    Comprehensive text normalizer based on silnlp framework.
    Handles punctuation, whitespace, and Unicode normalization.
    """
    
    def __init__(self, punctuation_rules: Optional[List[PunctuationNormalizationRule]] = None,
                 use_moses: bool = True, use_nfc: bool = True):
        """
        Initialize text normalizer.
        
        Args:
            punctuation_rules: Custom punctuation normalization rules
            use_moses: Whether to use Moses punctuation normalizer
            use_nfc: Whether to apply NFC Unicode normalization
        """
        self.use_moses = use_moses and MOSES_AVAILABLE
        self.use_nfc = use_nfc
        
        # Initialize Moses punctuation normalizer if available
        if self.use_moses:
            self._mpn = MosesPunctNormalizer()
            logger.info("Initialized Moses punctuation normalizer")
        
        # Set up punctuation rules
        if punctuation_rules is None:
            punctuation_rules = self._get_default_punctuation_rules()
        
        self.punctuation_rules = punctuation_rules
        self._build_punctuation_processors()
        
        logger.info(f"Initialized TextNormalizer with {len(self.punctuation_rules)} punctuation rules")
    
    def _get_default_punctuation_rules(self) -> List[PunctuationNormalizationRule]:
        """Get default punctuation normalization rules based on silnlp."""
        return [
            PunctuationNormalizationRule(".", PunctuationCategory.RIGHT_CLINGING),
            PunctuationNormalizationRule(",", PunctuationCategory.RIGHT_CLINGING),
            PunctuationNormalizationRule("!", PunctuationCategory.RIGHT_CLINGING),
            PunctuationNormalizationRule("?", PunctuationCategory.RIGHT_CLINGING),
            PunctuationNormalizationRule(":", PunctuationCategory.RIGHT_CLINGING),
            PunctuationNormalizationRule(";", PunctuationCategory.RIGHT_CLINGING),
            PunctuationNormalizationRule("(", PunctuationCategory.LEFT_CLINGING),
            PunctuationNormalizationRule(")", PunctuationCategory.RIGHT_CLINGING),
            PunctuationNormalizationRule("[", PunctuationCategory.LEFT_CLINGING),
            PunctuationNormalizationRule("]", PunctuationCategory.RIGHT_CLINGING),
            PunctuationNormalizationRule("{", PunctuationCategory.LEFT_CLINGING),
            PunctuationNormalizationRule("}", PunctuationCategory.RIGHT_CLINGING),
            PunctuationNormalizationRule("<", PunctuationCategory.LEFT_CLINGING),
            PunctuationNormalizationRule(">", PunctuationCategory.RIGHT_CLINGING),
            PunctuationNormalizationRule("'", PunctuationCategory.LEFT_RIGHT_CLINGING),
            PunctuationNormalizationRule('"', PunctuationCategory.LEFT_RIGHT_CLINGING),
            PunctuationNormalizationRule(""", PunctuationCategory.LEFT_CLINGING),
            PunctuationNormalizationRule(""", PunctuationCategory.RIGHT_CLINGING),
            PunctuationNormalizationRule("'", PunctuationCategory.LEFT_CLINGING),
            PunctuationNormalizationRule("'", PunctuationCategory.RIGHT_CLINGING),
            # Note: Dashes are NOT treated as punctuation for Dhao language
            # as they connect words (e.g., "word1-word2"). They will be left unchanged.
        ]
    
    def _build_punctuation_processors(self):
        """Build punctuation processing components."""
        self.punctuation_char_2_rule: Dict[str, PunctuationNormalizationRule] = {
            rule.character: rule for rule in self.punctuation_rules
        }
        self.supported_punctuation: Set[str] = set(self.punctuation_char_2_rule.keys())
        
        # Regex for consecutive spaces
        self.consecutive_spaces = regex.compile(r'\s+')
        
        # Build punctuation regexes
        escaped_punctuation_chars = "".join(regex.escape(rule.character) for rule in self.punctuation_rules)
        
        # Single punctuation with optional whitespace
        self.single_punctuation_with_optional_whitespace_regex = regex.compile(
            f"(?<![{escaped_punctuation_chars}\s])\s*[{escaped_punctuation_chars}]\s*(?![{escaped_punctuation_chars}\s])"
        )
        
        # Multiple punctuation
        self.multiple_punctuation_regex = regex.compile(
            f"[{escaped_punctuation_chars}][{escaped_punctuation_chars}\s]*[{escaped_punctuation_chars}]"
        )
    
    def normalize(self, text: str) -> str:
        """
        Apply comprehensive text normalization.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text or not text.strip():
            return text
        
        normalized = text
        
        # Step 1: Moses punctuation normalization (if enabled)
        if self.use_moses:
            try:
                normalized = self._mpn.normalize(normalized)
            except Exception as e:
                logger.warning(f"Moses normalization failed: {e}")
        
        # Step 2: NFC Unicode normalization (if enabled)
        if self.use_nfc:
            normalized = unicodedata.normalize('NFC', normalized)
        
        # Step 3: Custom punctuation and whitespace normalization
        normalized = self._normalize_punctuation_and_whitespace(normalized)
        
        # Step 4: Clean up terms (based on silnlp paratext.py)
        normalized = self._clean_term(normalized)
        
        return normalized
    
    def _normalize_punctuation_and_whitespace(self, text: str) -> str:
        """Apply punctuation and whitespace normalization."""
        summary = self._get_normalization_transformations(text)
        return summary.normalized_sentence
    
    def _get_normalization_transformations(self, sentence: str) -> SentenceNormalizationSummary:
        """Generate normalization transformations for a sentence."""
        all_transformations = self._find_transformations_sorted(sentence)
        
        if all_transformations:
            # Apply transformations to rebuild string
            parts = []
            last_part_end_index = 0
            for transformation in all_transformations:
                # Part prior to this normalization segment
                parts.append(sentence[last_part_end_index:transformation.slice.start_index])
                last_part_end_index = transformation.slice.end_index
                parts.append(transformation.replacement)
            # Deal with the ending original part
            parts.append(sentence[last_part_end_index:])
            
            normalized = "".join(parts)
        else:
            normalized = sentence
        
        return SentenceNormalizationSummary(
            original_sentence=sentence,
            normalized_sentence=normalized,
            transformations=all_transformations
        )
    
    def _find_transformations_sorted(self, sentence: str) -> List[SentenceTransformation]:
        """Find all normalization transformations for a sentence."""
        # Handle boundary whitespace
        boundary_transformations, sentence_trimmed, trim_offset = self._compute_boundary_transformations(sentence)
        
        # Handle single punctuation
        single_punctuation_transformations = []
        for slice_match in self._find_slices(self.single_punctuation_with_optional_whitespace_regex, sentence_trimmed):
            punctuation_char = regex.sub(self.consecutive_spaces, "", slice_match.slice)
            if punctuation_char in self.punctuation_char_2_rule:
                rule = self.punctuation_char_2_rule[punctuation_char]
                normalized = self._normalize_single_punctuation_slice(rule, slice_match)
                if normalized is not None and normalized != slice_match.slice:
                    single_punctuation_transformations.append(
                        SentenceTransformation(
                            slice=self._shift_slice(slice_match, trim_offset, sentence),
                            replacement=normalized,
                            description=f"Punctuation '{punctuation_char}' normalized by rule {rule.category}"
                        )
                    )
        
        # Handle consecutive spaces
        consecutive_spaces_transformations = []
        for slice_match in self._find_slices(self.consecutive_spaces, sentence_trimmed):
            if slice_match.slice != " ":  # Don't transform single spaces
                shifted_slice = self._shift_slice(slice_match, trim_offset, sentence)
                # Check if this overlaps with punctuation transformations
                if not any(self._slice_contains(t.slice, shifted_slice) for t in single_punctuation_transformations):
                    consecutive_spaces_transformations.append(
                        SentenceTransformation(
                            slice=shifted_slice,
                            replacement=" ",
                            description="Whitespace normalized to single space"
                        )
                    )
        
        # Combine and sort all transformations
        all_transformations = boundary_transformations + consecutive_spaces_transformations + single_punctuation_transformations
        return sorted(all_transformations, key=lambda t: t.slice.start_index)
    
    def _compute_boundary_transformations(self, sentence: str) -> Tuple[List[SentenceTransformation], str, int]:
        """Compute boundary whitespace transformations."""
        transformations = []
        
        left_trimmed = sentence.lstrip()
        if sentence != left_trimmed:
            slice_length = len(sentence) - len(left_trimmed)
            transformations.append(
                SentenceTransformation(
                    slice=self._build_slice(0, slice_length, sentence),
                    replacement="",
                    description="Removing left boundary whitespace"
                )
            )
        
        right_trimmed = sentence.rstrip()
        if sentence != right_trimmed:
            slice_length = len(sentence) - len(right_trimmed)
            transformations.append(
                SentenceTransformation(
                    slice=self._build_slice(len(right_trimmed), len(sentence), sentence),
                    replacement="",
                    description="Removing right boundary whitespace"
                )
            )
        
        return transformations, sentence.strip(), len(sentence) - len(left_trimmed)
    
    def _normalize_single_punctuation_slice(self, rule: PunctuationNormalizationRule, slice_obj: StringSlice) -> Optional[str]:
        """Normalize a single punctuation character slice."""
        punctuation_char = rule.character
        
        if rule.category == PunctuationCategory.LEFT_CLINGING:
            if slice_obj.start_index != 0:
                return " " + punctuation_char
            else:
                return punctuation_char
        elif rule.category == PunctuationCategory.RIGHT_CLINGING:
            if slice_obj.end_index != len(slice_obj.outer):
                return punctuation_char + " "
            else:
                return punctuation_char
        elif rule.category == PunctuationCategory.LEFT_RIGHT_CLINGING:
            if slice_obj.start_index == 0 or slice_obj.end_index == len(slice_obj.outer):
                return punctuation_char
            elif slice_obj.slice == punctuation_char:
                return None  # No change needed
            elif slice_obj.slice[0] != punctuation_char and slice_obj.slice[-1] != punctuation_char:
                return " " + punctuation_char + " "
            elif slice_obj.slice[0] == punctuation_char:
                return punctuation_char + " "
            else:
                return " " + punctuation_char
        elif rule.category == PunctuationCategory.UNCLINGING:
            if slice_obj.start_index == 0:
                return punctuation_char + " "
            elif slice_obj.end_index == len(slice_obj.outer):
                return " " + punctuation_char
            else:
                return " " + punctuation_char + " "
        
        return None
    
    def _clean_term(self, text: str) -> str:
        """
        Clean terms based on silnlp paratext.py clean_term function.
        
        - Strip whitespace
        - Remove parentheses content  
        - Normalize internal whitespace
        """
        # Strip outer whitespace
        cleaned = text.strip()
        
        # Remove parenthetical content (like silnlp)
        cleaned = self._strip_parens(cleaned)
        
        # Normalize internal whitespace
        cleaned = " ".join(cleaned.split())
        
        return cleaned
    
    def _strip_parens(self, text: str) -> str:
        """Remove parenthetical content, similar to silnlp."""
        # Simple approach: remove content in parentheses
        # This matches the pattern from silnlp but is simplified
        return re.sub(r'\s*\([^)]*\)', '', text).strip()
    
    def _find_slices(self, pattern: regex.Pattern, text: str) -> List[StringSlice]:
        """Find all regex matches as StringSlice objects."""
        return [
            StringSlice(
                start_index=match.span()[0],
                end_index=match.span()[1], 
                slice=match.group(),
                outer=text
            )
            for match in regex.finditer(pattern, text)
        ]
    
    def _build_slice(self, start_index: int, end_index: int, outer: str) -> StringSlice:
        """Build a StringSlice from indices."""
        return StringSlice(
            start_index=start_index,
            end_index=end_index,
            slice=outer[start_index:end_index],
            outer=outer
        )
    
    def _shift_slice(self, slice_obj: StringSlice, offset: int, new_outer: str) -> StringSlice:
        """Shift a slice by an offset to a new outer string."""
        return StringSlice(
            start_index=slice_obj.start_index + offset,
            end_index=slice_obj.end_index + offset,
            slice=slice_obj.slice,
            outer=new_outer
        )
    
    def _slice_contains(self, outer: StringSlice, inner: StringSlice) -> bool:
        """Check if outer slice contains inner slice."""
        return (outer.start_index <= inner.start_index) and (outer.end_index >= inner.end_index)


# Global normalizer instance
_global_normalizer = None


def get_text_normalizer(use_moses: bool = True, use_nfc: bool = True) -> TextNormalizer:
    """
    Get a global text normalizer instance (singleton pattern).
    
    Args:
        use_moses: Whether to use Moses punctuation normalizer
        use_nfc: Whether to apply NFC Unicode normalization
        
    Returns:
        TextNormalizer instance
    """
    global _global_normalizer
    if _global_normalizer is None:
        _global_normalizer = TextNormalizer(use_moses=use_moses, use_nfc=use_nfc)
    return _global_normalizer


def normalize_text(text: str, use_moses: bool = True, use_nfc: bool = True) -> str:
    """
    Convenient function to normalize text using the global normalizer.
    
    Args:
        text: Input text to normalize
        use_moses: Whether to use Moses punctuation normalizer  
        use_nfc: Whether to apply NFC Unicode normalization
        
    Returns:
        Normalized text
    """
    if not text or not text.strip():
        return text
        
    normalizer = get_text_normalizer(use_moses=use_moses, use_nfc=use_nfc)
    return normalizer.normalize(text)


def normalize_parallel_texts(source_texts: List[str], target_texts: List[str], 
                           use_moses: bool = True, use_nfc: bool = True) -> Tuple[List[str], List[str]]:
    """
    Normalize parallel source and target texts.
    
    Args:
        source_texts: List of source texts
        target_texts: List of target texts  
        use_moses: Whether to use Moses punctuation normalizer
        use_nfc: Whether to apply NFC Unicode normalization
        
    Returns:
        Tuple of (normalized_sources, normalized_targets)
    """
    normalizer = get_text_normalizer(use_moses=use_moses, use_nfc=use_nfc)
    
    normalized_sources = [normalizer.normalize(text) for text in source_texts]
    normalized_targets = [normalizer.normalize(text) for text in target_texts]
    
    return normalized_sources, normalized_targets


if __name__ == "__main__":
    # Test the normalizer
    test_texts = [
        "Hello , world !",
        "This is a test  sentence   with   multiple spaces.",
        "Punctuation( test )with parentheses.",
        "Quote test 'single' and \"double\" quotes.",
        "  Leading and trailing spaces  ",
        "Mixed—punctuation; test: here?",
    ]
    
    normalizer = TextNormalizer()
    
    print("Text Normalization Test:")
    print("=" * 50)
    
    for text in test_texts:
        normalized = normalizer.normalize(text)
        print(f"Original:   '{text}'")
        print(f"Normalized: '{normalized}'")
        print()
    
    print("Moses normalization test (if available):")
    print(f"Moses available: {MOSES_AVAILABLE}")
    if MOSES_AVAILABLE:
        test_moses = "Hello , world ! How are you ?"
        norm_moses = normalize_text(test_moses, use_moses=True)
        norm_no_moses = normalize_text(test_moses, use_moses=False)
        print(f"Original:    '{test_moses}'")
        print(f"With Moses:  '{norm_moses}'")
        print(f"Without:     '{norm_no_moses}'")
