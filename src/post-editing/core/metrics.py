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
