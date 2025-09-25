"""
Different vectorizer implementations for similarity-based retrieval.

For optimal performance with WordBasedLCSRetriever, install rapidfuzz:
    pip install rapidfuzz
    
This provides much faster string similarity computations than the fallback difflib.
"""

import collections
import numpy as np
from typing import List, Tuple, Dict, Counter, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import re

# Try to import efficient string similarity libraries
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz
        RAPIDFUZZ_AVAILABLE = True
    except ImportError:
        RAPIDFUZZ_AVAILABLE = False
        import difflib

from .base import BaseRetriever
from .caching import get_sbert_model, get_corpus_embeddings


class BM25Retriever(BaseRetriever):
    """BM25-based retrieval for few-shot examples."""
    
    def get_similar_examples(self, query: str, corpus: List[Tuple], k: int) -> List[Tuple]:
        """Get top-k similar examples using BM25."""
        print(f"Getting {k} few-shot examples using BM25 vectorizer")
        
        # Extract source texts for similarity computation
        source_texts = [pair[0] for pair in corpus]
        
        tokenized_corpus = [doc.split() for doc in source_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        top_k_indices = scores.argsort()[-k:][::-1]
        return [corpus[i] for i in top_k_indices]


class TFIDFRetriever(BaseRetriever):
    """TF-IDF-based retrieval for few-shot examples."""
    
    def get_similar_examples(self, query: str, corpus: List[Tuple], k: int) -> List[Tuple]:
        """Get top-k similar examples using TF-IDF."""
        print(f"Getting {k} few-shot examples using TF-IDF vectorizer")
        
        # Extract source texts for similarity computation
        source_texts = [pair[0] for pair in corpus]
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(source_texts)
        query_vector = vectorizer.transform([query])
        cosine_similarities = (tfidf_matrix * query_vector.T).toarray().flatten()
        top_k_indices = cosine_similarities.argsort()[-k:][::-1]
        return [corpus[i] for i in top_k_indices]


class SBERTRetriever(BaseRetriever):
    """SBERT-based retrieval with caching for few-shot examples."""
    
    def __init__(self, corpus_id: str = None):
        """Initialize SBERT retriever with optional corpus ID for caching."""
        self.corpus_id = corpus_id
    
    def get_similar_examples(self, query: str, corpus: List[Tuple], k: int) -> List[Tuple]:
        """Get top-k similar examples using cached SBERT embeddings."""
        print(f"Getting {k} few-shot examples using SBERT vectorizer (with caching)")
        
        # Get cached corpus embeddings (this will compute and cache if not already cached)
        corpus_data = get_corpus_embeddings(corpus, self.corpus_id)
        embeddings = corpus_data['embeddings']
        cached_corpus = corpus_data['corpus']
        
        # Compute query embedding only
        sbert_model = get_sbert_model()
        query_embedding = sbert_model.encode([query], show_progress_bar=False)  # (1, dim)
        
        # Compute similarities using cached corpus embeddings
        cosine_similarities = embeddings @ query_embedding.T  # (n_samples, 1)
        top_k_indices = cosine_similarities[:, 0].argsort()[-k:][::-1]
        return [cached_corpus[i] for i in top_k_indices]


def _extract_char_ngrams(text: str, n_order: int) -> Counter[str]:
    """Extracts character n-grams of a specific order from a string."""
    return collections.Counter(text[i:i+n_order] for i in range(len(text) - n_order + 1))


def _efficient_string_similarity(word1: str, word2: str) -> float:
    """
    Calculate similarity score between two words using efficient libraries.
    
    Uses rapidfuzz/fuzzywuzzy if available, otherwise falls back to difflib.
    """
    if not word1 or not word2:
        return 0.0
    
    word1_lower = word1.lower()
    word2_lower = word2.lower()
    
    if RAPIDFUZZ_AVAILABLE:
        # Use rapidfuzz/fuzzywuzzy partial_ratio which is similar to LCS matching
        # It finds the best matching substring and normalizes the score
        similarity = fuzz.partial_ratio(word1_lower, word2_lower) / 100.0
        return similarity
    else:
        # Fallback to difflib.SequenceMatcher (built-in Python)
        matcher = difflib.SequenceMatcher(None, word1_lower, word2_lower)
        # Use ratio() which gives a similarity score between 0 and 1
        return matcher.ratio()


def _longest_common_substring_length_difflib(s1: str, s2: str) -> int:
    """Fallback LCS implementation using difflib when no fast libraries are available."""
    if not s1 or not s2:
        return 0
    
    matcher = difflib.SequenceMatcher(None, s1.lower(), s2.lower())
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    return match.size


def _extract_words(text: str, min_length: int = 2) -> List[str]:
    """Extract words from text, filtering by minimum length."""
    # Use regex to extract words (letters and numbers)
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return [word for word in words if len(word) >= min_length]


def _score_sentence_for_word(word: str, sentence: str, similarity_threshold: float = 0.5) -> float:
    """Score a sentence based on how well it matches a given word using efficient string similarity."""
    sentence_words = _extract_words(sentence)
    
    # Find the best match for the word in the sentence
    best_similarity = 0.0
    
    for sentence_word in sentence_words:
        similarity = _efficient_string_similarity(word, sentence_word)
        if similarity > best_similarity:
            best_similarity = similarity
    
    # Only return score if it meets the threshold
    return best_similarity if best_similarity >= similarity_threshold else 0.0


class CHRFRAGRetriever(BaseRetriever):
    """CHRF-counterweighted RAG exemplar selection for few-shot examples."""
    
    def __init__(self, alpha: float = 2.0, chrf_order: int = 4):
        """
        Initialize CHRF-RAG retriever.
        
        Args:
            alpha: The penalty factor for redundancy. The paper uses alpha=2.
            chrf_order: The maximum order of n-grams for CHRF calculation.
        """
        self.alpha = alpha
        self.chrf_order = chrf_order
        self._preprocessed_corpus = None
        self._corpus_cache = None
    
    def _preprocess_corpus(self, corpus: List[Tuple]):
        """
        Preprocess the corpus by extracting n-grams for all source texts.
        
        Args:
            corpus: List of (source, target) text pairs
        """
        if self._corpus_cache == corpus:
            return  # Already preprocessed this corpus
        
        print("Preprocessing corpus for CHRF-RAG (extracting n-grams)...")
        self._preprocessed_corpus = []
        
        for source, target in tqdm(corpus):
            ngrams = {
                n: _extract_char_ngrams(source, n)
                for n in range(1, self.chrf_order + 1)
            }
            self._preprocessed_corpus.append({
                'source': source,
                'target': target,
                'ngrams': ngrams
            })
        
        self._corpus_cache = corpus
    
    def _calculate_modified_chrf(
        self,
        candidate_ngrams: Dict[int, Counter[str]],
        eval_ngrams: Dict[int, Counter[str]],
        seen_ngrams_counts: Dict[int, Counter[str]]
    ) -> float:
        """Calculates the modified CHRF score.
        
        Args:
            candidate_ngrams: N-grams from the candidate exemplar.
            eval_ngrams: N-grams from the query text.
            seen_ngrams_counts: Counts of n-grams seen so far.
        """
        total_f_score = 0.0
        for n in range(1, self.chrf_order + 1):
            cand_n_grams = candidate_ngrams.get(n, collections.Counter())
            eval_n_grams = eval_ngrams.get(n, collections.Counter())
            seen_n_grams = seen_ngrams_counts.get(n, collections.Counter())

            weighted_tp = 0.0
            common_keys = cand_n_grams.keys() & eval_n_grams.keys()
            for ngram in common_keys:
                matches = min(cand_n_grams[ngram], eval_n_grams[ngram])
                ci = seen_n_grams.get(ngram, 0)
                weight = (1 + ci) ** -self.alpha
                weighted_tp += matches * weight

            total_cand_ngrams = sum(cand_n_grams.values())
            total_eval_ngrams = sum(eval_n_grams.values())
            precision = weighted_tp / total_cand_ngrams if total_cand_ngrams > 0 else 0.0
            recall = weighted_tp / total_eval_ngrams if total_eval_ngrams > 0 else 0.0
            f_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
            total_f_score += f_score

        return total_f_score / self.chrf_order

    def get_similar_examples(self, query: str, corpus: List[Tuple], k: int) -> List[Tuple]:
        """Get top-k similar examples using CHRF-RAG algorithm."""
        print(f"Getting {k} few-shot examples using CHRF-RAG vectorizer")
        
        if k < 1:
            return []
        
        # Preprocess corpus if needed
        self._preprocess_corpus(corpus)
        
        # Extract n-grams from query
        eval_ngrams = {
            n: _extract_char_ngrams(query, n)
            for n in range(1, self.chrf_order + 1)
        }
        
        selected_examples = []
        selected_ids = set()
        seen_ngrams_for_current_eval: Dict[int, Counter[str]] = {}
        
        for _ in range(k):
            best_score = -1.0
            best_candidate = None
            best_idx = None
            
            for idx, candidate in enumerate(self._preprocessed_corpus):
                if idx in selected_ids:
                    continue
                
                score = self._calculate_modified_chrf(
                    candidate['ngrams'],
                    eval_ngrams,
                    seen_ngrams_for_current_eval
                )
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    best_idx = idx
            
            if best_candidate and best_idx is not None:
                # Add to selected examples as (source, target) tuple
                selected_examples.append((best_candidate['source'], best_candidate['target']))
                selected_ids.add(best_idx)
                
                # Update seen n-grams
                for n, ngrams_to_add in best_candidate['ngrams'].items():
                    if n not in seen_ngrams_for_current_eval:
                        seen_ngrams_for_current_eval[n] = collections.Counter()
                    seen_ngrams_for_current_eval[n].update(ngrams_to_add)
            else:
                # Stop if no more candidates can be found
                break
        
        return selected_examples


class WordBasedLCSRetriever(BaseRetriever):
    """Word-based retrieval using longest-common substring distance for few-shot examples."""
    
    def __init__(self, min_word_length: int = 2, similarity_threshold: float = 0.5):
        """
        Initialize word-based LCS retriever.
        
        Args:
            min_word_length: Minimum word length to consider for matching
            similarity_threshold: Minimum LCS similarity score to consider a match (0.0-1.0)
        """
        self.min_word_length = min_word_length
        self.similarity_threshold = similarity_threshold
        
        # Print information about which similarity library is being used
        if RAPIDFUZZ_AVAILABLE:
            try:
                from rapidfuzz import __version__
                print(f"Using rapidfuzz v{__version__} for efficient string similarity")
            except ImportError:
                try:
                    from fuzzywuzzy import __version__
                    print(f"Using fuzzywuzzy v{__version__} for string similarity")
                except:
                    print("Using rapidfuzz/fuzzywuzzy for efficient string similarity")
        else:
            print("Using difflib (built-in) for string similarity. Install 'rapidfuzz' for better performance.")
    
    def get_similar_examples(self, query: str, corpus: List[Tuple], k: int) -> List[Tuple]:
        """Get top-k similar examples using word-based LCS matching."""
        print(f"Getting {k} few-shot examples using Word-based LCS vectorizer")
        
        if k < 1:
            return []
        
        # Extract source texts for similarity computation
        source_texts = [pair[0] for pair in corpus]
        
        # Extract words from the query
        query_words = _extract_words(query, self.min_word_length)
        
        if not query_words:
            print("Warning: No valid words found in query after filtering")
            # Fallback to first k examples if no words found
            return corpus[:k]
        
        print(f"Processing query with {len(query_words)} words: {query_words}")
        
        # Score each sentence in the corpus
        sentence_scores = {}
        
        for sent_idx, source_text in enumerate(source_texts):
            total_score = 0.0
            matched_words = 0
            
            # For each word in the query, find best match in this sentence
            for query_word in query_words:
                word_score = _score_sentence_for_word(
                    query_word, source_text, self.similarity_threshold
                )
                if word_score > 0:
                    total_score += word_score
                    matched_words += 1
            
            # Calculate final score: average similarity * coverage factor
            if matched_words > 0:
                avg_similarity = total_score / matched_words
                coverage = matched_words / len(query_words)
                final_score = avg_similarity * coverage
                sentence_scores[sent_idx] = final_score
            else:
                sentence_scores[sent_idx] = 0.0
        
        # Sort sentences by score and return top-k
        sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top-k indices
        top_k_indices = [idx for idx, score in sorted_sentences[:k]]
        
        # Print some debugging info
        if sorted_sentences:
            print(f"Top sentence scores: {[(idx, f'{score:.3f}') for idx, score in sorted_sentences[:min(5, len(sorted_sentences))]]}")
        
        return [corpus[i] for i in top_k_indices]
