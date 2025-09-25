#!/usr/bin/env python3
"""
GIZA++ Alignment Runner

Helper script to run GIZA++ alignment and extract log-likelihood scores.
This script handles the details of preparing data for GIZA++, running the alignment,
and extracting meaningful scores from the output.
"""

import argparse
import os
import sys
import subprocess
import logging
from pathlib import Path
import tempfile
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GIZAAlignmentRunner:
    def __init__(self, work_dir: str = None):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        self.work_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"GIZA work directory: {self.work_dir}")
    
    def prepare_corpus_files(self, source_file: Path, target_file: Path) -> tuple[Path, Path, Path]:
        """Prepare corpus files for GIZA++ using plain2snt."""
        try:
            # Copy files to work directory with standard names
            src_corpus = self.work_dir / "source.txt"
            tgt_corpus = self.work_dir / "target.txt"
            
            shutil.copy2(source_file, src_corpus)
            shutil.copy2(target_file, tgt_corpus)
            
            # Run plain2snt to create .vcb and .snt files
            cmd = ["plain2snt", str(src_corpus), str(tgt_corpus)]
            
            result = subprocess.run(cmd, cwd=self.work_dir, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"plain2snt failed: {result.stderr}")
                return None, None, None
            
            # Expected output files
            src_vcb = self.work_dir / "source.txt.vcb"
            tgt_vcb = self.work_dir / "target.txt.vcb"
            snt_file = self.work_dir / "source.txt_target.txt.snt"
            
            # Verify files were created
            if not all(f.exists() for f in [src_vcb, tgt_vcb, snt_file]):
                logger.error("plain2snt did not create expected output files")
                return None, None, None
                
            logger.debug("Successfully prepared corpus files with plain2snt")
            return src_vcb, tgt_vcb, snt_file
            
        except Exception as e:
            logger.error(f"Error preparing corpus files: {e}")
            return None, None, None
    
    def run_giza(self, src_vcb: Path, tgt_vcb: Path, snt_file: Path, 
                 output_prefix: str = "alignment") -> bool:
        """Run GIZA++ alignment."""
        try:
            cmd = [
                "GIZA++",
                "-S", str(src_vcb),
                "-T", str(tgt_vcb),
                "-C", str(snt_file),
                "-o", output_prefix,
                "-outputpath", str(self.work_dir),
                # Additional GIZA++ parameters for better alignment
                "-m1", "5",  # IBM Model 1 iterations
                "-m2", "0",  # IBM Model 2 iterations (skip for speed)
                "-mh", "5",  # HMM iterations
                "-m3", "3",  # IBM Model 3 iterations
                "-m4", "3",  # IBM Model 4 iterations
                "-nodumps",  # Don't create dump files to save space
            ]
            
            result = subprocess.run(cmd, cwd=self.work_dir, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"GIZA++ failed: {result.stderr}")
                return False
            
            logger.debug(f"GIZA++ completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error running GIZA++: {e}")
            return False
    
    def extract_log_likelihood(self, output_prefix: str) -> float:
        """Extract log-likelihood from GIZA++ output."""
        try:
            # Look for perplexity file
            perp_file = self.work_dir / f"{output_prefix}.perp"
            
            if not perp_file.exists():
                logger.error(f"Perplexity file not found: {perp_file}")
                return 0.0
            
            # Read the perplexity file and extract final training perplexity
            with open(perp_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            if not lines:
                logger.error("No data lines found in perplexity file")
                return 0.0
            
            # Get the last line which should contain final iteration results
            last_line = lines[-1]
            parts = last_line.split()
            
            if len(parts) < 5:
                logger.error(f"Invalid perplexity file format: {last_line}")
                return 0.0
            
            # Format: training_size test_size iteration model train_perplexity ...
            train_perplexity = float(parts[4])
            
            # Convert perplexity to log-likelihood (negative log probability)
            # Lower perplexity = higher likelihood = better alignment
            log_likelihood = -train_perplexity
            
            logger.debug(f"Extracted log-likelihood: {log_likelihood} (perplexity: {train_perplexity})")
            return log_likelihood
            
        except Exception as e:
            logger.error(f"Error extracting log-likelihood: {e}")
            return 0.0
    
    def run_alignment(self, source_file: Path, target_file: Path, 
                     output_prefix: str = "alignment") -> float:
        """Run complete GIZA++ alignment and return log-likelihood."""
        try:
            # Step 1: Prepare corpus files
            src_vcb, tgt_vcb, snt_file = self.prepare_corpus_files(source_file, target_file)
            
            if not all([src_vcb, tgt_vcb, snt_file]):
                return 0.0
            
            # Step 2: Run GIZA++
            if not self.run_giza(src_vcb, tgt_vcb, snt_file, output_prefix):
                return 0.0
            
            # Step 3: Extract log-likelihood
            log_likelihood = self.extract_log_likelihood(output_prefix)
            
            return log_likelihood
            
        except Exception as e:
            logger.error(f"Error in run_alignment: {e}")
            return 0.0
    
    def cleanup(self):
        """Clean up work directory."""
        if self.work_dir.exists():
            logger.debug(f"Cleaning up work directory: {self.work_dir}")
            shutil.rmtree(self.work_dir)


def main():
    parser = argparse.ArgumentParser(description="Run GIZA++ alignment and extract log-likelihood")
    parser.add_argument("source_file", help="Source corpus file")
    parser.add_argument("target_file", help="Target corpus file")
    parser.add_argument("--output_prefix", default="alignment", help="Output prefix for GIZA++ files")
    parser.add_argument("--work_dir", help="Work directory (default: temp directory)")
    parser.add_argument("--keep_files", action="store_true", help="Keep work files for debugging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    source_file = Path(args.source_file)
    target_file = Path(args.target_file)
    
    if not source_file.exists():
        logger.error(f"Source file not found: {source_file}")
        sys.exit(1)
    
    if not target_file.exists():
        logger.error(f"Target file not found: {target_file}")
        sys.exit(1)
    
    runner = GIZAAlignmentRunner(args.work_dir)
    
    try:
        log_likelihood = runner.run_alignment(source_file, target_file, args.output_prefix)
        
        print(f"Log-likelihood: {log_likelihood}")
        
        if log_likelihood == 0.0:
            logger.error("Alignment failed or returned zero log-likelihood")
            sys.exit(1)
            
    finally:
        if not args.keep_files:
            runner.cleanup()


if __name__ == "__main__":
    main()
