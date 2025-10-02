#!/usr/bin/env python3
"""
Bible Perplexity Scorer

This script finds the best English Bible version paired with a target language
based on perplexity scores from GIZA++ symmetric alignment.

Pipeline:
1. Create verse alignment using create_aligned_nt_ot.py
2. Normalize and tokenize using normalize_for_giza.py
3. Run GIZA++ symmetric alignment
4. Extract perplexity scores
5. Generate CSV report with results

Usage:
    python bible_perplexity_scorer.py --target_corpus_path <path> --output_csv <path>
"""

import argparse
import os
import sys
import subprocess
import tempfile
import shutil
import csv
import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BiblePerplexityScorer:
    def __init__(self, 
                 english_corpus_dir: str,
                 target_corpus_path: str,
                 vref_path: str,
                 output_dir: str,
                 giza_py_path: str = "/data/projects/punim0478/setiawand/giza-py",
                 temp_dir: str = None,
                 keep_temp: bool = False):
        """
        Initialize the Bible Perplexity Scorer.
        
        Args:
            english_corpus_dir: Directory containing English Bible versions
            target_corpus_path: Path to target language Bible file
            vref_path: Path to verse reference file
            output_dir: Directory to save temporary and output files
            giza_py_path: Path to giza-py directory
            temp_dir: Temporary directory (if None, creates one)
            keep_temp: Whether to keep temporary files for debugging
        """
        self.english_corpus_dir = Path(english_corpus_dir)
        self.target_corpus_path = Path(target_corpus_path)
        self.vref_path = Path(vref_path)
        self.output_dir = Path(output_dir)
        self.giza_py_path = Path(giza_py_path)
        self.keep_temp = keep_temp
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="bible_perplexity_"))
        
        logger.info(f"English corpus directory: {self.english_corpus_dir}")
        logger.info(f"Target corpus: {self.target_corpus_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Temporary directory: {self.temp_dir}")
    
    def get_english_corpus_files(self) -> List[Path]:
        """Get list of English corpus files."""
        pattern = str(self.english_corpus_dir / "eng-*.txt")
        files = [Path(f) for f in glob.glob(pattern)]
        files.sort()
        logger.info(f"Found {len(files)} English corpus files")
        return files
    
    def extract_version_name(self, filepath: Path) -> str:
        """Extract version name from English corpus filename."""
        # Remove 'eng-' prefix and '.txt' suffix
        filename = filepath.name
        if filename.startswith('eng-'):
            version = filename[4:]  # Remove 'eng-' prefix
        else:
            version = filename
        
        if version.endswith('.txt'):
            version = version[:-4]  # Remove '.txt' suffix
            
        return version
    
    def create_alignment(self, english_file: Path, version_name: str) -> Optional[Path]:
        """Create verse alignment between English and target language."""
        logger.info(f"Creating alignment for {version_name}...")
        
        # Output file for aligned data
        aligned_file = self.temp_dir / f"aligned-{version_name}-dhao-all.txt"
        
        # Run create_aligned_nt_ot.py
        cmd = [
            "python3", "scripts/create_aligned_nt_ot.py",
            "--source_corpus_path", str(english_file),
            "--target_corpus_path", str(self.target_corpus_path),
            "--vref_path", str(self.vref_path),
            "--src_lang", "eng",
            "--tgt_lang", "dhao",
            "--output_dir", str(self.temp_dir),
            "--output_prefix", f"aligned-{version_name}-dhao"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode != 0:
                logger.error(f"Alignment creation failed for {version_name}: {result.stderr}")
                return None
            
            # Check if the aligned file was created (it creates a CSV, we need to convert)
            # The create_aligned_nt_ot.py creates files with pattern: {prefix}-{src_lang}-{tgt_lang}-all.csv
            csv_file = self.temp_dir / f"aligned-{version_name}-dhao-eng-dhao-all.csv"
            if not csv_file.exists():
                logger.error(f"Expected CSV file not found: {csv_file}")
                # Let's also check what files were actually created
                logger.debug(f"Files in temp directory: {list(self.temp_dir.glob('*'))}")
                return None
            
            # Convert CSV to parallel text format for GIZA++
            self.convert_csv_to_parallel_text(csv_file, aligned_file)
            
            return aligned_file
            
        except Exception as e:
            logger.error(f"Error creating alignment for {version_name}: {e}")
            return None
    
    def convert_csv_to_parallel_text(self, csv_file: Path, output_file: Path):
        """Convert CSV alignment to parallel text format."""
        logger.debug(f"Converting {csv_file} to parallel text format")
        
        with open(csv_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            reader = csv.DictReader(f_in)
            for row in reader:
                source_text = row.get('source_text', '').strip()
                target_text = row.get('target_text', '').strip()
                
                if source_text and target_text:
                    f_out.write(f"{source_text} ||| {target_text}\n")
    
    def normalize_texts(self, aligned_file: Path, version_name: str) -> Tuple[Optional[Path], Optional[Path]]:
        """Normalize and tokenize source and target texts for GIZA++."""
        logger.info(f"Normalizing texts for {version_name}...")
        
        # Temporary files for source and target
        temp_source = self.temp_dir / f"temp_source_{version_name}.txt"
        temp_target = self.temp_dir / f"temp_target_{version_name}.txt"
        
        # Output files for normalized texts
        norm_source = self.temp_dir / f"train.en-{version_name}"
        norm_target = self.temp_dir / f"train.nfa"
        
        try:
            # Split the parallel text into separate source and target files
            with open(aligned_file, 'r', encoding='utf-8') as f:
                with open(temp_source, 'w', encoding='utf-8') as f_src, \
                     open(temp_target, 'w', encoding='utf-8') as f_tgt:
                    
                    for line in f:
                        line = line.strip()
                        if ' ||| ' in line:
                            source, target = line.split(' ||| ', 1)
                            f_src.write(source + '\n')
                            f_tgt.write(target + '\n')
            
            # Run normalize_for_giza.py
            cmd = [
                "python3", "scripts/normalize_for_giza.py",
                str(temp_source),
                str(temp_target),
                "--output-source", str(norm_source),
                "--output-target", str(norm_target),
                "--min-length", "1"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode != 0:
                logger.error(f"Normalization failed for {version_name}: {result.stderr}")
                return None, None
            
            if not norm_source.exists() or not norm_target.exists():
                logger.error(f"Normalized files not created for {version_name}")
                return None, None
            
            return norm_source, norm_target
            
        except Exception as e:
            logger.error(f"Error normalizing texts for {version_name}: {e}")
            return None, None
        finally:
            # Clean up temporary files
            for temp_file in [temp_source, temp_target]:
                if temp_file.exists():
                    temp_file.unlink()
    
    def run_giza_alignment(self, source_file: Path, target_file: Path, version_name: str) -> Optional[float]:
        """Run GIZA++ symmetric alignment and extract perplexity."""
        logger.info(f"Running GIZA++ alignment for {version_name}...")
        
        try:
            # Change to giza-py directory for running the command
            giza_output_dir = self.temp_dir / f"giza_output_{version_name}"
            giza_output_dir.mkdir(exist_ok=True)
            
            cmd = [
                "python3", "giza.py",
                "--source", str(source_file),
                "--target", str(target_file),
                "--alignments", str(giza_output_dir / "alignment"),
                "--sym-heuristic", "intersection"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.giza_py_path)
            
            if result.returncode != 0:
                logger.error(f"GIZA++ alignment failed for {version_name}: {result.stderr}")
                return None
            
            # Extract perplexity from stdout
            perplexity = self.extract_perplexity_from_output(result.stdout)
            
            if perplexity is None:
                logger.warning(f"Could not extract perplexity from GIZA++ output for {version_name}")
                # Try to extract from stderr as well
                perplexity = self.extract_perplexity_from_output(result.stderr)
            
            return perplexity
            
        except Exception as e:
            logger.error(f"Error running GIZA++ alignment for {version_name}: {e}")
            return None
    
    def extract_perplexity_from_output(self, output: str) -> Optional[float]:
        """Extract perplexity score from GIZA++ output."""
        if not output:
            return None
        
        # Look for perplexity patterns in the output
        patterns = [
            r'PERPLEXITY\s+([0-9]+\.?[0-9]*)',
            r'perplexity:\s*([0-9]+\.?[0-9]*)',
            r'Perplexity\s*=\s*([0-9]+\.?[0-9]*)',
            r'TRAIN\s+PERPLEXITY\s+([0-9]+\.?[0-9]*)',
            r'Model\d+:\s+.*PERPLEXITY\s+([0-9]+\.?[0-9]*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    # Return the last match (usually the final perplexity)
                    return float(matches[-1])
                except ValueError:
                    continue
        
        return None
    
    def process_single_version(self, english_file: Path) -> Optional[Tuple[str, str, float]]:
        """Process a single English Bible version and return results."""
        version_name = self.extract_version_name(english_file)
        target_lang = "dhao"
        
        try:
            # Step 1: Create alignment
            aligned_file = self.create_alignment(english_file, version_name)
            if not aligned_file:
                return None
            
            # Step 2: Normalize texts
            norm_source, norm_target = self.normalize_texts(aligned_file, version_name)
            if not norm_source or not norm_target:
                return None
            
            # Step 3: Run GIZA++ alignment
            perplexity = self.run_giza_alignment(norm_source, norm_target, version_name)
            if perplexity is None:
                return None
            
            logger.info(f"✓ {version_name}: perplexity = {perplexity}")
            return version_name, target_lang, perplexity
            
        except Exception as e:
            logger.error(f"Error processing {version_name}: {e}")
            return None
    
    def run_scoring(self, max_versions: int = None) -> List[Tuple[str, str, float]]:
        """Run perplexity scoring for all English versions."""
        english_files = self.get_english_corpus_files()
        
        if max_versions:
            english_files = english_files[:max_versions]
            logger.info(f"Processing first {max_versions} versions for testing")
        
        results = []
        
        for i, english_file in enumerate(english_files, 1):
            logger.info(f"Processing {i}/{len(english_files)}: {english_file.name}")
            
            result = self.process_single_version(english_file)
            if result:
                results.append(result)
            else:
                logger.warning(f"Failed to process {english_file.name}")
        
        return results
    
    def save_results(self, results: List[Tuple[str, str, float]], output_file: Path):
        """Save results to CSV file."""
        logger.info(f"Saving results to {output_file}")
        
        # Sort by perplexity (lower is better)
        results.sort(key=lambda x: x[2])
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['english_version', 'target_language', 'perplexity_score'])
            writer.writerows(results)
        
        logger.info(f"Saved {len(results)} results to {output_file}")
        
        # Log top 5 results
        if results:
            logger.info("Top 5 best alignments (lowest perplexity):")
            for i, (version, target, perplexity) in enumerate(results[:5], 1):
                logger.info(f"  {i}. {version} -> {target}: {perplexity:.4f}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if not self.keep_temp and self.temp_dir.exists():
            logger.info(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
        elif self.keep_temp:
            logger.info(f"Keeping temporary files in: {self.temp_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Find best English Bible version for target language based on GIZA++ perplexity scores"
    )
    parser.add_argument(
        "--english_corpus_dir",
        type=str,
        default="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/eng/corpus",
        help="Directory containing English Bible corpus files"
    )
    parser.add_argument(
        "--target_corpus_path",
        type=str,
        default="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/nfa-nfa.txt",
        help="Path to target language corpus file"
    )
    parser.add_argument(
        "--vref_path",
        type=str,
        default="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/vref.txt",
        help="Path to verse reference file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/perplexity_scoring",
        help="Output directory for results and temporary files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="bible_perplexity_scores.csv",
        help="Output CSV filename"
    )
    parser.add_argument(
        "--giza_py_path",
        type=str,
        default="/data/projects/punim0478/setiawand/giza-py",
        help="Path to giza-py directory"
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        help="Temporary directory (default: auto-generated)"
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary files for debugging"
    )
    parser.add_argument(
        "--max_versions",
        type=int,
        help="Maximum number of versions to process (for testing)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input files
    if not Path(args.english_corpus_dir).exists():
        logger.error(f"English corpus directory not found: {args.english_corpus_dir}")
        sys.exit(1)
    
    if not Path(args.target_corpus_path).exists():
        logger.error(f"Target corpus file not found: {args.target_corpus_path}")
        sys.exit(1)
    
    if not Path(args.vref_path).exists():
        logger.error(f"Verse reference file not found: {args.vref_path}")
        sys.exit(1)
    
    if not Path(args.giza_py_path).exists():
        logger.error(f"GIZA-py directory not found: {args.giza_py_path}")
        sys.exit(1)
    
    # Create scorer and run
    scorer = BiblePerplexityScorer(
        english_corpus_dir=args.english_corpus_dir,
        target_corpus_path=args.target_corpus_path,
        vref_path=args.vref_path,
        output_dir=args.output_dir,
        giza_py_path=args.giza_py_path,
        temp_dir=args.temp_dir,
        keep_temp=args.keep_temp
    )
    
    try:
        logger.info("=== Bible Perplexity Scoring Pipeline ===")
        
        # Run scoring
        results = scorer.run_scoring(max_versions=args.max_versions)
        
        if not results:
            logger.error("No successful alignments found!")
            sys.exit(1)
        
        # Save results
        output_file = Path(args.output_dir) / args.output_file
        scorer.save_results(results, output_file)
        
        logger.info("=== Pipeline completed successfully! ===")
        logger.info(f"Results saved to: {output_file}")
        
    finally:
        scorer.cleanup()


if __name__ == "__main__":
    main()
