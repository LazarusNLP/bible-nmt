#!/usr/bin/env python3
"""
Script to filter post-editing results to first 500 rows and recalculate metrics.

This script:
1. Loads output CSV and input CSV files
2. Filters to first 500 rows of input
3. Removes duplicates and keeps best chrfpp_improvement scores
4. Recalculates metrics using the same method as post-editing pipeline
5. Saves filtered CSV and updated metrics JSON

Usage:
    python filter_and_calculate_metrics.py \
        --input_csv /path/to/input.csv \
        --output_csv /path/to/output.csv \
        --filtered_output /path/to/filtered_output.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add post-editing module to path
script_dir = Path(__file__).parent
post_editing_dir = script_dir.parent / "src" / "post-editing"
sys.path.insert(0, str(post_editing_dir))

try:
    from core.data_models import Row
    from core.metrics import MetricsCalculator
    print("✅ Successfully imported post-editing modules")
except ImportError as e:
    print(f"❌ Failed to import post-editing modules: {e}")
    print(f"Make sure the post-editing directory exists at: {post_editing_dir}")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter post-editing results and recalculate metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input_csv", required=True,
                       help="Path to input CSV file (original aligned data)")
    parser.add_argument("--output_csv", required=True, 
                       help="Path to output CSV file (post-editing results)")
    parser.add_argument("--filtered_output", required=True,
                       help="Path to save filtered output CSV")
    parser.add_argument("--max_rows", type=int, default=500,
                       help="Maximum number of rows to keep from input (default: 500)")
    
    return parser.parse_args()


def load_and_filter_data(input_csv_path, output_csv_path, max_rows=500):
    """
    Load and filter data following the same logic as the notebook.
    
    Args:
        input_csv_path: Path to input CSV
        output_csv_path: Path to output CSV  
        max_rows: Maximum rows to keep from input
        
    Returns:
        Filtered pandas DataFrame
    """
    print(f"Loading data from:")
    print(f"  Input: {input_csv_path}")
    print(f"  Output: {output_csv_path}")
    
    # Load both dataframes
    output = pd.read_csv(output_csv_path)
    input_df = pd.read_csv(input_csv_path)
    
    print(f"Original sizes - Input: {len(input_df)}, Output: {len(output)}")
    
    # Take only first max_rows of input dataframe
    input_df = input_df.head(max_rows)
    print(f"Limited input to first {max_rows} rows")
    
    # Check for duplicate rows in the output dataframe
    print(f"Total rows before removing duplicates: {len(output)}")
    
    # Remove duplicate rows, keeping the one with highest chrfpp_improvement
    # Sort by chrfpp_improvement in descending order first
    output = output.sort_values('chrfpp_improvement', ascending=False)
    output_no_duplicates = output.drop_duplicates(keep='first')
    print(f"Total rows after removing duplicates: {len(output_no_duplicates)}")
    print(f"Removed {len(output) - len(output_no_duplicates)} duplicate rows (kept highest chrfpp_improvement)")
    
    # Replace the original dataframe with the deduplicated one
    output = output_no_duplicates
    
    # Keep only rows in output that exist in input_df based on source_text
    output = output[output['src_text'].isin(input_df['source_text'])]
    print(f"Total rows after removing rows not in input: {len(output)}")
    
    # For duplicate source texts, keep only the one with the highest chrfpp_improvement
    print(f"Rows before handling src_text duplicates: {len(output)}")
    duplicate_src_count = output.duplicated(subset=['src_text']).sum()
    print(f"Number of duplicate src_text rows: {duplicate_src_count}")
    
    # Sort by chrfpp_improvement in descending order, then drop duplicates keeping the first (highest score)
    output = output.sort_values('chrfpp_improvement', ascending=False).drop_duplicates(subset=['src_text'], keep='first')
    print(f"Rows after keeping highest chrfpp_improvement for each src_text: {len(output)}")
    
    # Create a mapping from source_text to the original order in input_df
    input_order = {text: i for i, text in enumerate(input_df['source_text'])}
    
    # Add a sorting column based on the original order
    output['sort_order'] = output['src_text'].map(input_order)
    
    # Sort the output dataframe to match the input order
    output = output.sort_values('sort_order').drop('sort_order', axis=1)
    
    print(f"Final filtered dataframe has {len(output)} rows")
    
    return output


def convert_df_to_rows(df):
    """
    Convert pandas DataFrame to list of Row objects for metric calculation.
    
    Args:
        df: Pandas DataFrame with post-editing results
        
    Returns:
        List of Row objects
    """
    rows = []
    
    for _, row_data in df.iterrows():
        # Create Row object - map DataFrame columns to Row fields
        row = Row(
            src_text=row_data['src_text'],
            tgt_text=row_data['tgt_text'], 
            pred_tgt_text=row_data['pred_tgt_text'],
            post_edited_tgt_txt=row_data['post_edited_tgt_txt'],
            src_lang=row_data['src_lang'],
            tgt_lang=row_data['tgt_lang'],
            src_lang_name=row_data['src_lang_name'],
            tgt_lang_name=row_data['tgt_lang_name'],
            spbleu_improvement=row_data.get('spbleu_improvement'),
            chrf3_improvement=row_data.get('chrf3_improvement'),
            chrfpp_improvement=row_data.get('chrfpp_improvement')
        )
        rows.append(row)
    
    return rows


def calculate_metrics(rows):
    """
    Calculate metrics using the same method as post-editing pipeline.
    
    Args:
        rows: List of Row objects
        
    Returns:
        Dictionary with metrics results
    """
    print(f"\nCalculating metrics for {len(rows)} rows...")
    
    # Use the same MetricsCalculator as post-editing pipeline
    metrics = MetricsCalculator.calculate_metrics(rows)
    
    return metrics


def save_results(filtered_df, metrics, filtered_output_path):
    """
    Save filtered DataFrame and metrics JSON.
    
    Args:
        filtered_df: Filtered pandas DataFrame
        metrics: Metrics dictionary
        filtered_output_path: Path to save filtered CSV
    """
    # Save filtered CSV
    print(f"\nSaving filtered results to: {filtered_output_path}")
    os.makedirs(Path(filtered_output_path).parent, exist_ok=True)
    filtered_df.to_csv(filtered_output_path, index=False)
    
    # Save metrics JSON
    base_path = Path(filtered_output_path)
    metrics_path = base_path.parent / (base_path.stem + '_metrics.json')
    print(f"Saving metrics to: {metrics_path}")
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved successfully!")
    
    return str(metrics_path)


def print_metrics_summary(metrics):
    """Print metrics summary like the post-editing pipeline does."""
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    
    if "original_mt_metrics" in metrics:
        print("\nOriginal MT Metrics:")
        for metric, score in metrics["original_mt_metrics"].items():
            print(f"  {metric.upper()}: {score:.4f}")
    
    if "post_edited_metrics" in metrics:
        print("\nPost-Edited Metrics:")
        for metric, score in metrics["post_edited_metrics"].items():
            print(f"  {metric.upper()}: {score:.4f}")
    
    if "improvements" in metrics:
        print("\nImprovements (Post-Edited - Original):")
        for metric, improvement in metrics["improvements"].items():
            sign = "+" if improvement >= 0 else ""
            print(f"  {metric.upper()}: {sign}{improvement:.4f}")
    
    print("="*60)


def main():
    """Main function."""
    args = parse_args()
    
    print("="*60)
    print("POST-EDITING RESULTS FILTER AND METRICS CALCULATOR")
    print("="*60)
    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Filtered Output: {args.filtered_output}")
    print(f"Max Rows: {args.max_rows}")
    print("="*60)
    
    try:
        # Validate input files
        if not os.path.exists(args.input_csv):
            raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
        if not os.path.exists(args.output_csv):
            raise FileNotFoundError(f"Output CSV not found: {args.output_csv}")
        
        # Load and filter data
        print("\n1. Loading and filtering data...")
        filtered_df = load_and_filter_data(args.input_csv, args.output_csv, args.max_rows)
        
        # Convert to Row objects for metric calculation
        print("\n2. Converting data for metric calculation...")
        rows = convert_df_to_rows(filtered_df)
        print(f"Converted {len(rows)} rows for metric calculation")
        
        # Calculate metrics
        print("\n3. Calculating metrics...")
        metrics = calculate_metrics(rows)
        
        # Save results
        print("\n4. Saving results...")
        metrics_path = save_results(filtered_df, metrics, args.filtered_output)
        
        # Print summary
        print_metrics_summary(metrics)
        
        print(f"\n🎉 Processing complete!")
        print(f"📁 Filtered CSV: {args.filtered_output}")
        print(f"📊 Metrics JSON: {metrics_path}")
        print(f"📈 Final dataset: {len(filtered_df)} rows")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
