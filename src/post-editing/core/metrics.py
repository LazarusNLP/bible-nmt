"""
Metrics calculation for post-editing evaluation.
"""

from typing import List
try:
    import evaluate
except ImportError:
    evaluate = None
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from core.data_models import Row


class MetricsCalculator:
    """Handles calculation of various translation metrics."""
    
    @staticmethod
    def _compute_metrics_for_predictions(predictions: List[str], references: List[str]) -> dict:
        """Helper method to compute metrics for given predictions and references."""
        # Postprocess text (strip whitespace)
        cleaned_preds = [pred.strip() for pred in predictions]
        cleaned_labels = [[label.strip()] for label in references]
        
        # Load evaluation metrics
        chrf = evaluate.load("chrf")
        sacrebleu = evaluate.load("sacrebleu")
        spbleu = evaluate.load("sacrebleu")
        
        # Calculate BLEU score using NLTK (matching run_evaluation.py)
        tokenized_preds = [word_tokenize(pred.lower().strip()) for pred in cleaned_preds]
        tokenized_refs = [[word_tokenize(ref[0].lower().strip())] for ref in cleaned_labels]
        bleu_score = corpus_bleu(tokenized_refs, tokenized_preds)
        
        # Calculate SacreBLEU scores
        sacrebleu_result = sacrebleu.compute(predictions=cleaned_preds, references=cleaned_labels)
        spbleu_result = spbleu.compute(predictions=cleaned_preds, references=cleaned_labels, tokenize="flores200")
        
        # Calculate chrF variants
        chrf_result = chrf.compute(predictions=cleaned_preds, references=cleaned_labels)
        chrf3_result = chrf.compute(predictions=cleaned_preds, references=cleaned_labels, beta=3)
        chrf_plus_result = chrf.compute(predictions=cleaned_preds, references=cleaned_labels, word_order=2)
        
        # Compile results
        eval_result = {
            "bleu": bleu_score * 100,  # Convert to percentage like run_evaluation.py
            "sacrebleu": sacrebleu_result["score"],
            "spbleu": spbleu_result["score"],
            "chrf": chrf_result["score"],
            "chrf3": chrf3_result["score"],
            "chrf++": chrf_plus_result["score"],
        }
        eval_result = {k: round(v, 4) for k, v in eval_result.items()}
        return eval_result

    @staticmethod
    def calculate_batch_individual_metrics(rows_data: List[tuple]) -> List[dict]:
        """
        Calculate individual metrics for multiple rows efficiently by batching metric calculations.
        
        Args:
            rows_data: List of tuples (original_text, post_edited_text, reference_text)
            
        Returns:
            List of dictionaries with individual metric scores for each row
        """
        if not rows_data:
            return []
        
        # Separate data for batch processing
        original_texts = []
        post_edited_texts = []
        reference_texts = []
        
        for original, post_edited, reference in rows_data:
            original_texts.append(original.strip())
            post_edited_texts.append(post_edited.strip())
            reference_texts.append(reference.strip())
        
        # Load evaluation metrics once
        chrf = evaluate.load("chrf")
        spbleu = evaluate.load("sacrebleu")
        
        # Batch calculate metrics for original texts
        original_spbleu_results = spbleu.compute(
            predictions=original_texts,
            references=[[ref] for ref in reference_texts],
            tokenize="flores200"
        )
        
        original_chrf3_results = chrf.compute(
            predictions=original_texts,
            references=[[ref] for ref in reference_texts],
            beta=3
        )
        
        original_chrfpp_results = chrf.compute(
            predictions=original_texts,
            references=[[ref] for ref in reference_texts],
            word_order=2
        )
        
        # Batch calculate metrics for post-edited texts
        post_edited_spbleu_results = spbleu.compute(
            predictions=post_edited_texts,
            references=[[ref] for ref in reference_texts],
            tokenize="flores200"
        )
        
        post_edited_chrf3_results = chrf.compute(
            predictions=post_edited_texts,
            references=[[ref] for ref in reference_texts],
            beta=3
        )
        
        post_edited_chrfpp_results = chrf.compute(
            predictions=post_edited_texts,
            references=[[ref] for ref in reference_texts],
            word_order=2
        )
        
        # Combine results for each row
        results = []
        for i in range(len(rows_data)):
            # Extract individual scores from batch results
            original_spbleu = original_spbleu_results["precisions"][i] if "precisions" in original_spbleu_results else original_spbleu_results["score"]
            original_chrf3 = original_chrf3_results["score"] if isinstance(original_chrf3_results["score"], (int, float)) else original_chrf3_results["score"][i] if hasattr(original_chrf3_results["score"], '__getitem__') else original_chrf3_results["score"]
            original_chrfpp = original_chrfpp_results["score"] if isinstance(original_chrfpp_results["score"], (int, float)) else original_chrfpp_results["score"][i] if hasattr(original_chrfpp_results["score"], '__getitem__') else original_chrfpp_results["score"]
            
            post_edited_spbleu = post_edited_spbleu_results["precisions"][i] if "precisions" in post_edited_spbleu_results else post_edited_spbleu_results["score"]
            post_edited_chrf3 = post_edited_chrf3_results["score"] if isinstance(post_edited_chrf3_results["score"], (int, float)) else post_edited_chrf3_results["score"][i] if hasattr(post_edited_chrf3_results["score"], '__getitem__') else post_edited_chrf3_results["score"]
            post_edited_chrfpp = post_edited_chrfpp_results["score"] if isinstance(post_edited_chrfpp_results["score"], (int, float)) else post_edited_chrfpp_results["score"][i] if hasattr(post_edited_chrfpp_results["score"], '__getitem__') else post_edited_chrfpp_results["score"]
            
            # Calculate improvements
            spbleu_improvement = post_edited_spbleu - original_spbleu
            chrf3_improvement = post_edited_chrf3 - original_chrf3
            chrfpp_improvement = post_edited_chrfpp - original_chrfpp
            
            results.append({
                "original_metrics": {
                    "spbleu": round(original_spbleu, 4),
                    "chrf3": round(original_chrf3, 4),
                    "chrfpp": round(original_chrfpp, 4)
                },
                "post_edited_metrics": {
                    "spbleu": round(post_edited_spbleu, 4),
                    "chrf3": round(post_edited_chrf3, 4),
                    "chrfpp": round(post_edited_chrfpp, 4)
                },
                "improvements": {
                    "spbleu": round(spbleu_improvement, 4),
                    "chrf3": round(chrf3_improvement, 4),
                    "chrfpp": round(chrfpp_improvement, 4)
                }
            })
        
        return results

    @staticmethod
    def calculate_individual_metrics(original_text: str, post_edited_text: str, reference_text: str) -> dict:
        """
        Calculate individual sentence metrics for the three specified metrics.
        
        Args:
            original_text: Original MT output
            post_edited_text: Post-edited output  
            reference_text: Ground truth reference
            
        Returns:
            Dictionary with individual metric scores for original and post-edited texts
        """
        # Clean texts
        original_clean = original_text.strip()
        post_edited_clean = post_edited_text.strip()
        reference_clean = reference_text.strip()
        
        # Load evaluation metrics
        chrf = evaluate.load("chrf")
        spbleu = evaluate.load("sacrebleu")
        
        # Calculate metrics for original text
        original_spbleu = spbleu.compute(
            predictions=[original_clean], 
            references=[[reference_clean]], 
            tokenize="flores200"
        )["score"]
        
        original_chrf3 = chrf.compute(
            predictions=[original_clean], 
            references=[[reference_clean]], 
            beta=3
        )["score"]
        
        original_chrfpp = chrf.compute(
            predictions=[original_clean], 
            references=[[reference_clean]], 
            word_order=2
        )["score"]
        
        # Calculate metrics for post-edited text
        post_edited_spbleu = spbleu.compute(
            predictions=[post_edited_clean], 
            references=[[reference_clean]], 
            tokenize="flores200"
        )["score"]
        
        post_edited_chrf3 = chrf.compute(
            predictions=[post_edited_clean], 
            references=[[reference_clean]], 
            beta=3
        )["score"]
        
        post_edited_chrfpp = chrf.compute(
            predictions=[post_edited_clean], 
            references=[[reference_clean]], 
            word_order=2
        )["score"]
        
        return {
            "original": {
                "spbleu": original_spbleu,
                "chrf3": original_chrf3,
                "chrfpp": original_chrfpp
            },
            "post_edited": {
                "spbleu": post_edited_spbleu,
                "chrf3": post_edited_chrf3,
                "chrfpp": post_edited_chrfpp
            },
            "improvements": {
                "spbleu": round(post_edited_spbleu - original_spbleu, 4),
                "chrf3": round(post_edited_chrf3 - original_chrf3, 4),
                "chrfpp": round(post_edited_chrfpp - original_chrfpp, 4)
            }
        }

    @classmethod
    def calculate_metrics(cls, rows: List[Row]) -> dict:
        """Calculate all evaluation metrics including original MT and post-edited results with improvements."""
        # make sure all the rows have post_edited_tgt_txt
        assert all(r.post_edited_tgt_txt is not None for r in rows), "Some rows do not have post_edited_tgt_txt"
        
        # Extract texts
        original_predictions = [r.pred_tgt_text for r in rows]  # Original MT output
        post_edited_predictions = [r.post_edited_tgt_txt for r in rows]  # Post-edited output
        references = [r.tgt_text for r in rows]  # Ground truth
        
        # Calculate metrics for original MT output
        print("Calculating metrics for original MT output...")
        original_metrics = cls._compute_metrics_for_predictions(original_predictions, references)
        
        # Calculate metrics for post-edited output
        print("Calculating metrics for post-edited output...")
        post_edited_metrics = cls._compute_metrics_for_predictions(post_edited_predictions, references)
        
        # Calculate improvements (delta)
        improvements = {}
        for metric in original_metrics.keys():
            improvement = post_edited_metrics[metric] - original_metrics[metric]
            improvements[metric] = round(improvement, 4)
        
        # Compile final results
        final_results = {
            "original_mt_metrics": original_metrics,
            "post_edited_metrics": post_edited_metrics,
            "improvements": improvements
        }
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS SUMMARY")
        print("="*50)
        print("\nOriginal MT Metrics:")
        for metric, score in original_metrics.items():
            print(f"  {metric.upper()}: {score:.2f}")
        
        print("\nPost-Edited Metrics:")
        for metric, score in post_edited_metrics.items():
            print(f"  {metric.upper()}: {score:.2f}")
        
        print("\nImprovements (Post-Edited - Original):")
        for metric, improvement in improvements.items():
            sign = "+" if improvement >= 0 else ""
            print(f"  {metric.upper()}: {sign}{improvement:.2f}")
        print("="*50)
        
        return final_results
