import collections
from dataclasses import dataclass, field
from typing import List, Dict, Counter
from tqdm import tqdm

from utils.smol_data import SmolFewShotExample
from utils.madlad_data import MADLADExample

# helper to extract character n-grams of a specific order from a string
def _extract_char_ngrams(text: str, n_order: int) -> Counter[str]:
    """Extracts character n-grams of a specific order from a string."""
    return collections.Counter(text[i:i+n_order] for i in range(len(text) - n_order + 1))


class CHRFCounterweightedRAGSelector:
    """
    Implements the CHRF-counterweighted RAG exemplar selection algorithm.

    Similar to SMoL paper.
    """

    def __init__(
        self,
        exemplar_pool: List[SmolFewShotExample] | List[MADLADExample],
        n_shots: int,
        alpha: float = 2.0,
        chrf_order: int = 4,
    ):
        """
        Initializes the selector and preprocesses the exemplar pool.

        Args:
            exemplar_pool (List[SmolFewShotExample]): The fixed pool of candidate exemplars.
            n_shots (int): The number of exemplars to select (N).
            alpha (float): The penalty factor for redundancy. The paper uses alpha=2.
            chrf_order (int): The maximum order of n-grams for CHRF calculation.
        """
        if n_shots < 1:
            raise ValueError("n_shots must be at least 1.")
        
        self.n_shots = n_shots
        self.alpha = alpha
        self.chrf_order = chrf_order
        self.exemplar_pool = exemplar_pool
        
        # Pre-process the pool immediately upon initialization
        self._preprocess_pool()

    def _preprocess_pool(self):
        """
        Populates the 'ngrams' field for each SmolFewShotExample object in the
        stored exemplar pool. This modifies the objects in-place.
        """
        print("Preprocessing exemplar pool (populating n-grams)...")
        for exemplar in tqdm(self.exemplar_pool):
            if not getattr(exemplar, 'ngrams', None):
                exemplar.ngrams = {
                    n: _extract_char_ngrams(exemplar.src, n)
                    for n in range(1, self.chrf_order + 1)
                }

    def _calculate_modified_chrf(
        self,
        candidate_ngrams: Dict[int, Counter[str]],
        eval_ngrams: Dict[int, Counter[str]],
        seen_ngrams_counts: Dict[int, Counter[str]]
    ) -> float:
        """Calculates the modified CHRF score.
        
        Args:
            candidate_ngrams (Dict[int, Counter[str]]): N-grams from the candidate exemplar.
            eval_ngrams (Dict[int, Counter[str]]): N-grams from the src_text.
            seen_ngrams_counts (Dict[int, Counter[str]]): Counts of n-grams seen so far.
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

    def get_examples(self, eval_source: str) -> List[SmolFewShotExample]:
        """
        Selects N exemplars for a single evaluation source sentence.

        Args:
            eval_source (str): The source sentence to find exemplars for.

        Returns:
            List[SmolFewShotExample]: A list of N selected exemplars.
        """
        eval_ngrams = {
            n: _extract_char_ngrams(eval_source, n)
            for n in range(1, self.chrf_order + 1)
        }

        selected_exemplars = []
        selected_ids = set()
        seen_ngrams_for_current_eval: Dict[int, Counter[str]] = {}

        for _ in range(self.n_shots):
            best_score = -1.0
            best_candidate = None

            for candidate in self.exemplar_pool:
                if id(candidate) in selected_ids:
                    continue
                
                score = self._calculate_modified_chrf(
                    candidate.ngrams,
                    eval_ngrams,
                    seen_ngrams_for_current_eval
                )

                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate:
                selected_exemplars.append(best_candidate)
                selected_ids.add(id(best_candidate))
                
                for n, ngrams_to_add in best_candidate.ngrams.items():
                    if n not in seen_ngrams_for_current_eval:
                        seen_ngrams_for_current_eval[n] = collections.Counter()
                    seen_ngrams_for_current_eval[n].update(ngrams_to_add)
            else:
                # Stop if no more candidates can be found
                break

        return selected_exemplars