from argparse import ArgumentParser
import json
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Import text normalization
from text_normalization import normalize_text


def parse_args():
    parser = ArgumentParser(description="Run inference on text file or CSV file using a trained model")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input file (.txt or .csv)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--target_path", type=str, required=False, help="Path to save the inference results (optional for CSV input - will update input CSV directly)")
    parser.add_argument("--src_lang_nllb", type=str, default="ind_Latn", help="Source language code for NLLB")
    parser.add_argument("--tgt_lang_nllb", type=str, required=True, help="Target language code for NLLB")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length for generation")
    parser.add_argument("--num_beams", type=int, default=8, help="Number of beams for beam search")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--output_format", type=str, choices=["txt", "json"], default="txt", help="Output format: txt or json (ignored for CSV input)")
    parser.add_argument("--input_mode", type=str, choices=["txt", "csv"], default="auto", help="Input mode: txt, csv, or auto (auto-detect from file extension)")
    
    return parser.parse_args()


def load_model_and_tokenizer(model_path, src_lang_nllb, tgt_lang_nllb, device):
    """Load model and tokenizer from the given path"""
    print(f"Loading model from: {model_path}")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    # Move model to GPU before creating the pipeline
    model = model.to(device)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading tokenizer from model path: {e}")
        print("Trying to load tokenizer from parent directories...")
        
        # Go 2 parent directories up (similar to run_evaluation.py)
        model_dir = os.path.abspath(os.path.join(model_path, "..", ".."))
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    tokenizer.src_lang = src_lang_nllb
    tokenizer.tgt_lang = tgt_lang_nllb

    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        src_lang=src_lang_nllb,
        tgt_lang=tgt_lang_nllb,
    )
    
    return translator


def read_input_file(input_path):
    """Read lines from input text file"""
    print(f"Reading input file: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [normalize_text(line.strip()) for line in f.readlines() if line.strip()]
    print(f"Loaded {len(lines)} lines from input file (with text normalization)")
    return lines


def read_csv_file(input_path):
    """Read CSV file and extract source_text column"""
    print(f"Reading CSV file: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    # Check if required columns exist
    required_cols = ['verse', 'source_text', 'target_text']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    
    print(f"Loaded CSV with {len(df)} rows")
    return df


def save_csv_results(df, predictions, input_path, target_path=None):
    """Save CSV with predictions as new column"""
    
    # Add predictions as new column
    df['pred_text'] = predictions
    
    # Determine output path - use target_path if provided, otherwise update input CSV
    output_path = target_path if target_path else input_path
    print(f"Saving results to CSV: {output_path}")
    
    # Create target directory if it doesn't exist (only needed for separate target_path)
    if target_path:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Results saved successfully to {output_path}")


def detect_input_mode(input_path, input_mode):
    """Detect input mode based on file extension or explicit mode"""
    if input_mode == "auto":
        if input_path.lower().endswith('.csv'):
            return "csv"
        elif input_path.lower().endswith('.txt'):
            return "txt"
        else:
            raise ValueError(f"Cannot auto-detect input mode for file: {input_path}. Please specify --input_mode")
    return input_mode


def run_inference(translator, input_lines, batch_size, max_length, num_beams):
    """Run inference on input lines using the translator pipeline"""
    print(f"Running inference on {len(input_lines)} lines...")
    
    predictions = []
    
    # Process in batches
    for i in range(0, len(input_lines), batch_size):
        batch = input_lines[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(input_lines) + batch_size - 1)//batch_size}")
        
        try:
            batch_predictions = translator(
                batch,
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
            )
            
            # Extract translation text
            batch_results = [pred["translation_text"] for pred in batch_predictions]
            predictions.extend(batch_results)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Add empty predictions for failed batch
            predictions.extend([""] * len(batch))
    
    return predictions


def save_results(predictions, input_lines, target_path, output_format):
    """Save inference results to target path"""
    print(f"Saving results to: {target_path}")
    
    # Create target directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    if output_format == "txt":
        # Save predictions only, one per line
        with open(target_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(pred + '\n')
    
    elif output_format == "json":
        # Save as JSON with source and target pairs
        results = []
        for i, (source, prediction) in enumerate(zip(input_lines, predictions)):
            results.append({
                "line_id": i + 1,
                "source": source,
                "prediction": prediction
            })
        
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved successfully to {target_path}")


def main():
    args = parse_args()
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Detect input mode
    mode = detect_input_mode(args.input_path, args.input_mode)
    print(f"Input mode: {mode}")
    
    # Load model and tokenizer
    translator = load_model_and_tokenizer(
        args.model_path, 
        args.src_lang_nllb, 
        args.tgt_lang_nllb, 
        device
    )
    
    if mode == "csv":
        # Handle CSV input
        df = read_csv_file(args.input_path)
        # Apply text normalization to source text from CSV
        input_lines = [normalize_text(text) for text in df['source_text'].tolist()]
        
        # Run inference
        predictions = run_inference(
            translator, 
            input_lines, 
            args.per_device_eval_batch_size, 
            args.max_length, 
            args.num_beams
        )
        
        # Save CSV results
        save_csv_results(df, predictions, args.input_path, args.target_path)
        
    else:
        # Handle TXT input (original behavior)
        if not args.target_path:
            raise ValueError("--target_path is required for TXT input mode")
            
        input_lines = read_input_file(args.input_path)
        
        # Run inference
        predictions = run_inference(
            translator, 
            input_lines, 
            args.per_device_eval_batch_size, 
            args.max_length, 
            args.num_beams
        )
        
        # Save results
        save_results(predictions, input_lines, args.target_path, args.output_format)
    
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
