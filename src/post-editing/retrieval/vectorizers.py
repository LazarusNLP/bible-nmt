"""
Different vectorizer implementations for similarity-based retrieval.

For optimal performance with word-based retrievers, install rapidfuzz:
    pip install rapidfuzz

This provides much faster string similarity computations than the fallback difflib.
"""

import collections
import numpy as np
import random
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

    def get_similar_examples(
        self, query: str, corpus: List[Tuple], k: int
    ) -> List[Tuple]:
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

    def get_similar_examples(
        self, query: str, corpus: List[Tuple], k: int
    ) -> List[Tuple]:
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

    def __init__(
        self, corpus_id: str = None, model_name: str = "LazarusNLP/all-indo-e5-small-v4"
    ):
        """
        Initialize SBERT retriever with optional corpus ID for caching.

        Args:
            corpus_id: Optional corpus ID for caching embeddings
            model_name: SentenceTransformer model name to use
        """
        self.corpus_id = corpus_id
        self.model_name = model_name

    def get_similar_examples(
        self, query: str, corpus: List[Tuple], k: int
    ) -> List[Tuple]:
        """Get top-k similar examples using cached SBERT embeddings."""
        print(
            f"Getting {k} few-shot examples using SBERT vectorizer (model: {self.model_name})"
        )

        # Get cached corpus embeddings (this will compute and cache if not already cached)
        corpus_data = get_corpus_embeddings(corpus, self.corpus_id, self.model_name)
        embeddings = corpus_data["embeddings"]
        cached_corpus = corpus_data["corpus"]

        # Compute query embedding only
        sbert_model = get_sbert_model(self.model_name)
        query_embedding = sbert_model.encode(
            [query], show_progress_bar=False
        )  # (1, dim)

        # Compute similarities using cached corpus embeddings
        cosine_similarities = embeddings @ query_embedding.T  # (n_samples, 1)
        top_k_indices = cosine_similarities[:, 0].argsort()[-k:][::-1]
        return [cached_corpus[i] for i in top_k_indices]


def _extract_char_ngrams(text: str, n_order: int) -> Counter[str]:
    """Extracts character n-grams of a specific order from a string."""
    return collections.Counter(
        text[i : i + n_order] for i in range(len(text) - n_order + 1)
    )


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
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [word for word in words if len(word) >= min_length]


def _score_sentence_for_word(
    word: str, sentence: str, similarity_threshold: float = 0.5
) -> float:
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
            self._preprocessed_corpus.append(
                {"source": source, "target": target, "ngrams": ngrams}
            )

        self._corpus_cache = corpus

    def _calculate_modified_chrf(
        self,
        candidate_ngrams: Dict[int, Counter[str]],
        eval_ngrams: Dict[int, Counter[str]],
        seen_ngrams_counts: Dict[int, Counter[str]],
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
            precision = (
                weighted_tp / total_cand_ngrams if total_cand_ngrams > 0 else 0.0
            )
            recall = weighted_tp / total_eval_ngrams if total_eval_ngrams > 0 else 0.0
            f_score = (
                2 * (precision * recall) / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            total_f_score += f_score

        return total_f_score / self.chrf_order

    def get_similar_examples(
        self, query: str, corpus: List[Tuple], k: int
    ) -> List[Tuple]:
        """Get top-k similar examples using CHRF-RAG algorithm."""
        print(f"Getting {k} few-shot examples using CHRF-RAG vectorizer")

        if k < 1:
            return []

        # Preprocess corpus if needed
        self._preprocess_corpus(corpus)

        # Extract n-grams from query
        eval_ngrams = {
            n: _extract_char_ngrams(query, n) for n in range(1, self.chrf_order + 1)
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
                    candidate["ngrams"], eval_ngrams, seen_ngrams_for_current_eval
                )

                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    best_idx = idx

            if best_candidate and best_idx is not None:
                # Add to selected examples as (source, target) tuple
                selected_examples.append(
                    (best_candidate["source"], best_candidate["target"])
                )
                selected_ids.add(best_idx)

                # Update seen n-grams
                for n, ngrams_to_add in best_candidate["ngrams"].items():
                    if n not in seen_ngrams_for_current_eval:
                        seen_ngrams_for_current_eval[n] = collections.Counter()
                    seen_ngrams_for_current_eval[n].update(ngrams_to_add)
            else:
                # Stop if no more candidates can be found
                break

        return selected_examples


class WordBasedParallelRetriever(BaseRetriever):
    """
    Word-based parallel sentence retrieval for few-shot examples with optimization.

    For each word in the input source sentence, this retriever finds sentences
    in the parallel corpus that contain the word or a close match according to
    longest-common substring distance. It takes the top n matches per word
    where n is a configurable hyperparameter.

    OPTIMIZATION: Uses precomputed corpus word index for batch processing efficiency.
    """

    def __init__(
        self,
        min_word_length: int = 2,
        similarity_threshold: float = 0.5,
        top_n_per_word: int = 3,
    ):
        """
        Initialize word-based parallel retriever.

        Args:
            min_word_length: Minimum word length to consider for matching
            similarity_threshold: Minimum LCS similarity score to consider a match (0.0-1.0)
            top_n_per_word: Number of top matches to retrieve per word (hyperparameter)
        """
        self.min_word_length = min_word_length
        self.similarity_threshold = similarity_threshold
        self.top_n_per_word = top_n_per_word

        # Optimization: Cache for corpus preprocessing
        self._corpus_word_index = None  # word -> [(sent_idx, similarity_score), ...]
        self._corpus_word_cache = None  # sent_idx -> [word1, word2, word3, ...]
        self._corpus_texts = None  # sent_idx -> source_text
        self._indexed_corpus_id = None  # Track which corpus was indexed

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
            print(
                "Using difflib (built-in) for string similarity. Install 'rapidfuzz' for better performance."
            )

    def _get_corpus_id(self, corpus: List[Tuple]) -> str:
        """Generate a unique ID for the corpus to track if it's been indexed."""
        import hashlib

        # Use first and last few sentences to create a corpus fingerprint
        sample_texts = []
        if len(corpus) > 0:
            sample_texts.append(corpus[0][0])
        if len(corpus) > 10:
            sample_texts.append(corpus[len(corpus) // 2][0])
        if len(corpus) > 1:
            sample_texts.append(corpus[-1][0])

        fingerprint = "|".join(sample_texts) + f"|size:{len(corpus)}"
        return hashlib.md5(fingerprint.encode()).hexdigest()[:12]

    def _preprocess_corpus(self, corpus: List[Tuple]) -> None:
        """
        Preprocess corpus to build word index for efficient batch processing.

        This builds:
        1. Word-to-sentence mapping for fast lookups
        2. Cached word extractions to avoid recomputing
        3. Text cache for reference
        """
        corpus_id = self._get_corpus_id(corpus)

        # Skip if already processed
        if self._indexed_corpus_id == corpus_id:
            return

        print(f"🔄 Preprocessing corpus for optimized word_parallel retrieval...")
        print(f"   Building word index for {len(corpus)} sentences...")

        # Initialize data structures
        self._corpus_word_index = {}  # word -> list of sentence indices containing it
        self._corpus_word_cache = {}  # sent_idx -> extracted words
        self._corpus_texts = {}  # sent_idx -> original text

        # Process each sentence in corpus
        for sent_idx, (source_text, target_text) in enumerate(
            tqdm(corpus, desc="Indexing corpus")
        ):
            # Extract and cache words for this sentence
            words = _extract_words(source_text, self.min_word_length)
            self._corpus_word_cache[sent_idx] = words
            self._corpus_texts[sent_idx] = source_text

            # Build word-to-sentence index
            for word in set(words):  # Use set to avoid duplicates
                if word not in self._corpus_word_index:
                    self._corpus_word_index[word] = []
                self._corpus_word_index[word].append(sent_idx)

        self._indexed_corpus_id = corpus_id

        # Print statistics
        total_unique_words = len(self._corpus_word_index)
        avg_sentences_per_word = (
            sum(len(sentences) for sentences in self._corpus_word_index.values())
            / total_unique_words
        )
        print(f"✅ Corpus preprocessing complete:")
        print(f"   📚 {len(corpus)} sentences indexed")
        print(f"   📝 {total_unique_words} unique words found")
        print(f"   📊 Average {avg_sentences_per_word:.1f} sentences per word")

    def get_similar_examples_batch(
        self, queries: List[str], corpus: List[Tuple], k: int
    ) -> List[List[Tuple]]:
        """
        Efficiently process multiple queries using precomputed corpus index.

        Args:
            queries: List of query strings
            corpus: List of (source, target) text pairs
            k: Number of examples per query (-1 for all matches)

        Returns:
            List of example lists, one per query
        """
        # Preprocess corpus if needed
        self._preprocess_corpus(corpus)

        use_all_results = k == -1
        if use_all_results:
            print(
                f"🚀 BATCH processing {len(queries)} queries using optimized Word-based Parallel vectorizer"
            )
            print(
                f"   Getting ALL word-matched examples (top-{self.top_n_per_word} per word)"
            )
        else:
            print(
                f"🚀 BATCH processing {len(queries)} queries using optimized Word-based Parallel vectorizer"
            )
            print(
                f"   Getting {k} few-shot examples (top-{self.top_n_per_word} per word)"
            )

        results = []

        # Process each query using the precomputed index
        for query_idx, query in enumerate(tqdm(queries, desc="Processing queries")):
            query_words = _extract_words(query, self.min_word_length)

            if not query_words:
                print(f"Warning: No valid words found in query {query_idx+1}")
                results.append(corpus[:k] if k > 0 else [])
                continue

            # Use precomputed index for efficient processing
            sentence_aggregate_scores = {}
            total_word_matches = 0

            for word in query_words:
                # Get candidate sentences for this word from precomputed index
                candidate_sentence_indices = self._corpus_word_index.get(word, [])
                word_matches = []

                # Only check sentences that contain similar words (HUGE optimization!)
                for sent_idx in candidate_sentence_indices:
                    cached_sentence_words = self._corpus_word_cache[sent_idx]

                    # Find best similarity with precomputed words
                    best_similarity = 0.0
                    for sentence_word in cached_sentence_words:
                        similarity = _efficient_string_similarity(word, sentence_word)
                        if similarity > best_similarity:
                            best_similarity = similarity

                    if best_similarity >= self.similarity_threshold:
                        word_matches.append((sent_idx, best_similarity))

                # Sort and take top matches for this word
                word_matches.sort(key=lambda x: x[1], reverse=True)
                top_matches_for_word = word_matches[: self.top_n_per_word]
                total_word_matches += len(top_matches_for_word)

                # Aggregate scores
                for sent_idx, score in top_matches_for_word:
                    if sent_idx not in sentence_aggregate_scores:
                        sentence_aggregate_scores[sent_idx] = 0.0
                    sentence_aggregate_scores[sent_idx] += score

            # Sort sentences by aggregate score and select results
            sorted_sentences = sorted(
                sentence_aggregate_scores.items(), key=lambda x: x[1], reverse=True
            )

            if use_all_results:
                top_indices = [idx for idx, score in sorted_sentences]
            else:
                top_indices = [idx for idx, score in sorted_sentences[:k]]

            query_results = [corpus[i] for i in top_indices]
            results.append(query_results)

        print(
            f"✅ Batch processing complete: processed {len(queries)} queries efficiently"
        )
        return results

    def get_similar_examples(
        self, query: str, corpus: List[Tuple], k: int
    ) -> List[Tuple]:
        """
        Get few-shot examples using word-based parallel sentence retrieval.

        For each word in the query, retrieves the top n sentences containing
        that word or close matches, then returns the top k unique sentences overall.

        Args:
            query: Query text to find similar examples for
            corpus: List of (source, target) text pairs
            k: Final number of examples to retrieve (use -1 for all retrieved examples)

        Returns:
            List of top-k most relevant (source, target) pairs (or all if k=-1)
        """
        use_all_results = k == -1
        if use_all_results:
            print(
                f"Getting ALL word-matched examples using Word-based Parallel vectorizer "
                f"(top-{self.top_n_per_word} per word)"
            )
        else:
            print(
                f"Getting {k} few-shot examples using Word-based Parallel vectorizer "
                f"(top-{self.top_n_per_word} per word)"
            )

        if k < 1 and not use_all_results:
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

        # For each word, find top n matching sentences
        word_sentence_matches = {}  # word -> [(sentence_idx, score), ...]
        sentence_aggregate_scores = {}  # sentence_idx -> aggregate_score

        for word in query_words:
            word_matches = []

            # Score all sentences for this word
            for sent_idx, source_text in enumerate(source_texts):
                word_score = _score_sentence_for_word(
                    word, source_text, self.similarity_threshold
                )
                if word_score > 0:
                    word_matches.append((sent_idx, word_score))

            # Sort by score and take top n for this word
            word_matches.sort(key=lambda x: x[1], reverse=True)
            top_matches_for_word = word_matches[: self.top_n_per_word]
            word_sentence_matches[word] = top_matches_for_word

            # Aggregate scores for final ranking
            for sent_idx, score in top_matches_for_word:
                if sent_idx not in sentence_aggregate_scores:
                    sentence_aggregate_scores[sent_idx] = 0.0
                sentence_aggregate_scores[sent_idx] += score

        # Debug information
        total_word_matches = sum(
            len(matches) for matches in word_sentence_matches.values()
        )
        unique_sentences = len(sentence_aggregate_scores)
        print(
            f"Found {total_word_matches} total word-sentence matches across {unique_sentences} unique sentences"
        )

        # Sort sentences by aggregate score and return top k
        sorted_sentences = sorted(
            sentence_aggregate_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Get indices - either top-k or all available
        if use_all_results:
            top_indices = [idx for idx, score in sorted_sentences]
            actual_k = len(top_indices)
            print(f"Using all {actual_k} word-matched sentences")
        else:
            top_indices = [idx for idx, score in sorted_sentences[:k]]
            actual_k = min(k, len(sorted_sentences))

        # Print some debugging info
        if sorted_sentences:
            show_count = min(5, len(sorted_sentences))
            print(
                f"Top sentence aggregate scores: {[(idx, f'{score:.3f}') for idx, score in sorted_sentences[:show_count]]}"
            )

        # Show word-level breakdown for debugging
        if len(query_words) <= 10:  # Only show details for reasonable number of words
            print("Word-level matches breakdown:")
            for word, matches in word_sentence_matches.items():
                if matches:
                    match_info = [(idx, f"{score:.3f}") for idx, score in matches[:3]]
                    print(f"  '{word}': {len(matches)} matches, top 3: {match_info}")

        return [corpus[i] for i in top_indices]


class RandomRetriever(BaseRetriever):
    """Random baseline retriever for few-shot examples."""

    def __init__(self, seed: int = 42):
        """
        Initialize random retriever with optional seed for reproducibility.

        Args:
            seed: Random seed for reproducible results
        """
        self.seed = seed
        self.selected_examples = None

    def get_similar_examples(
        self, query: str, corpus: List[Tuple], k: int
    ) -> List[Tuple]:
        """
        Get k random examples from corpus.

        On first call, selects k random examples and stores them for consistency.
        All subsequent calls return the same k examples regardless of the query.

        Args:
            query: Query text (ignored for random baseline)
            corpus: List of (source, target) text pairs
            k: Number of examples to retrieve

        Returns:
            List of k randomly selected (source, target) pairs (same across all queries)
        """
        # If this is the first call, select k random examples and store them
        if self.selected_examples is None:
            print(
                f"Selecting {k} random few-shot examples as baseline (seed={self.seed})"
            )

            # Set random seed for reproducibility
            random.seed(self.seed)

            # Select k random examples from corpus
            if k >= len(corpus):
                self.selected_examples = corpus.copy()
                print(
                    f"Warning: requested {k} examples but corpus only has {len(corpus)}. Using all corpus examples."
                )
            else:
                self.selected_examples = random.sample(corpus, k)

            print(
                f"Selected {len(self.selected_examples)} random examples for all queries"
            )
        else:
            print(
                f"Using previously selected {len(self.selected_examples)} random examples"
            )

        return self.selected_examples
