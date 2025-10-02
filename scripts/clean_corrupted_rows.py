#!/usr/bin/env python3
"""
Simple script to remove corrupted rows from CSV and let the post-editing script reprocess them.
"""

import pandas as pd
import sys

def clean_corrupted_rows(csv_path):
    """Remove rows where post_edited_tgt_txt contains reasoning text or JSON artifacts."""
    
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Original row count: {len(df)}")
    
    # Identify corrupted rows
    corrupted_mask = df['post_edited_tgt_txt'].str.contains(
        r'The user wants me to post-edit|post_edited_text|I need to correct|{"post_edited_text"',
        na=False,
        regex=True
    )
    
    corrupted_count = corrupted_mask.sum()
    print(f"Found {corrupted_count} corrupted rows")
    
    if corrupted_count > 0:
        # Show examples of what will be removed
        print("\nExamples of corrupted entries being removed:")
        for idx in df[corrupted_mask].index[:3]:
            post_edited = df.loc[idx, 'post_edited_tgt_txt']
            print(f"Row {idx}: {post_edited[:100]}...")
        
        # Remove corrupted rows
        df_clean = df[~corrupted_mask].copy()
        
        # Save cleaned CSV
        backup_path = csv_path.replace('.csv', '_backup.csv')
        print(f"\nCreating backup: {backup_path}")
        df.to_csv(backup_path, index=False)
        
        print(f"Saving cleaned CSV: {csv_path}")
        df_clean.to_csv(csv_path, index=False, quoting=1)  # QUOTE_ALL for safety
        
        print(f"✅ Removed {corrupted_count} corrupted rows")
        print(f"✅ Remaining rows: {len(df_clean)}")
        print(f"✅ Missing rows will be reprocessed on next run")
        
    else:
        print("✅ No corrupted rows found")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_corrupted_rows.py <csv_file_path>")
        print("Example: python clean_corrupted_rows.py /path/to/your/output.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    clean_corrupted_rows(csv_path)
