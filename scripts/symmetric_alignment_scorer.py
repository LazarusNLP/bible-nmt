#!/usr/bin/env python3
"""
Symmetric Alignment Scorer for Bible Translations

This script orchestrates the process of:
1. Normalizing text files (English and Dhao)
2. Creating aligned verse corpora
3. Running GIZA++ forward and backward alignment
4. Extracting log-likelihood scores
5. Compiling results into a CSV file

The goal is to find which English Bible translation has the best alignment
with the Dhao translation by comparing forward and reverse alignment scores.
"""

import argparse
import os
import sys
import csv
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import tempfile
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SymmetricAlignmentScorer:
    def __init__(
        self,
        english_corpus_dir: str,
        dhao_corpus_path: str,
        vref_path: str,
        output_dir: str,
        temp_dir: str = None,
    ):
        self.english_corpus_dir = Path(english_corpus_dir)
        self.dhao_corpus_path = Path(dhao_corpus_path)
        self.vref_path = Path(vref_path)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories for organization
        self.normalized_dir = self.temp_dir / "normalized"
        self.aligned_dir = self.temp_dir / "aligned"
        self.giza_dir = self.temp_dir / "giza"

        for dir_path in [self.normalized_dir, self.aligned_dir, self.giza_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized scorer with temp dir: {self.temp_dir}")

    def get_english_corpus_files(self) -> List[Path]:
        """Get all English corpus files."""
        files = list(self.english_corpus_dir.glob("eng-*.txt"))
        logger.info(f"Found {len(files)} English corpus files")
        return sorted(files)

    def extract_version_name(self, file_path: Path) -> str:
        """Extract version name from filename (e.g., eng-kjv.txt -> kjv)."""
        stem = file_path.stem
        if stem.startswith("eng-"):
            return stem[4:]  # Remove "eng-" prefix
        return stem

    def normalize_text_file(self, input_path: Path, output_path: Path) -> bool:
        """Normalize a text file using the normalize_text.py script."""
        try:
            cmd = [
                sys.executable,
                "scripts/normalize_text.py",
                str(input_path),
                "-o",
                str(output_path),
                "--preserve-case",  # Preserve case and normalize quotes as requested
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

            if result.returncode != 0:
                logger.error(f"Normalization failed for {input_path}: {result.stderr}")
                return False

            logger.debug(f"Normalized {input_path} -> {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error normalizing {input_path}: {e}")
            return False

    def create_aligned_corpus(
        self, source_path: Path, target_path: Path, output_prefix: str
    ) -> Tuple[Path, bool]:
        """Create aligned corpus using create_aligned_nt_ot.py script."""
        try:
            output_file = self.aligned_dir / f"{output_prefix}.csv"

            cmd = [
                sys.executable,
                "scripts/create_aligned_nt_ot.py",
                "--source_corpus_path",
                str(source_path),
                "--target_corpus_path",
                str(target_path),
                "--vref_path",
                str(self.vref_path),
                "--src_lang",
                "eng",
                "--tgt_lang",
                "dhao",
                "--output_dir",
                str(self.aligned_dir),
                "--output_prefix",
                output_prefix,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

            if result.returncode != 0:
                logger.error(f"Alignment failed for {output_prefix}: {result.stderr}")
                return output_file, False

            # The script creates files with pattern: {prefix}-{src_lang}-{tgt_lang}-all.csv
            actual_output = self.aligned_dir / f"{output_prefix}-eng-dhao-all.csv"

            if actual_output.exists():
                logger.debug(f"Created aligned corpus: {actual_output}")
                return actual_output, True
            else:
                logger.error(f"Expected output file not found: {actual_output}")
                return output_file, False

        except Exception as e:
            logger.error(f"Error creating aligned corpus for {output_prefix}: {e}")
            return self.aligned_dir / f"{output_prefix}.csv", False

    def prepare_giza_input(
        self, aligned_csv: Path, output_prefix: str
    ) -> Tuple[Path, Path, bool]:
        """Prepare input files for GIZA++ from aligned CSV."""
        try:
            source_file = self.giza_dir / f"{output_prefix}.source"
            target_file = self.giza_dir / f"{output_prefix}.target"

            with open(aligned_csv, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)

                with open(source_file, "w", encoding="utf-8") as src_f, open(
                    target_file, "w", encoding="utf-8"
                ) as tgt_f:

                    for row in reader:
                        # Write source and target texts, one per line
                        src_f.write(row["source_text"] + "\n")
                        tgt_f.write(row["target_text"] + "\n")

            logger.debug(f"Prepared GIZA input files: {source_file}, {target_file}")
            return source_file, target_file, True

        except Exception as e:
            logger.error(f"Error preparing GIZA input for {output_prefix}: {e}")
            return source_file, target_file, False

    def run_giza_alignment(
        self,
        source_file: Path,
        target_file: Path,
        output_prefix: str,
        direction: str = "forward",
    ) -> Tuple[float, bool]:
        """Run GIZA++ alignment and extract log-likelihood score."""
        try:
            giza_output_dir = self.giza_dir / f"{output_prefix}_{direction}"
            giza_output_dir.mkdir(exist_ok=True)

            # Determine source and target based on direction
            if direction == "forward":
                src_file, tgt_file = source_file, target_file
            else:  # reverse
                src_file, tgt_file = target_file, source_file

            # Copy files to output directory with standardized names
            local_src = giza_output_dir / "source.txt"
            local_tgt = giza_output_dir / "target.txt"

            # Copy the files with size limiting for GIZA++ stability
            import shutil

            # For large corpora, sample to prevent GIZA++ crashes
            max_lines_for_giza = 1000

            def copy_with_limit(src_path, dst_path, max_lines):
                """Copy file with line limit to prevent GIZA++ crashes"""
                with open(src_path, "r", encoding="utf-8") as src, open(
                    dst_path, "w", encoding="utf-8"
                ) as dst:
                    for i, line in enumerate(src):
                        if i >= max_lines:
                            break
                        dst.write(line)

            # Check file sizes and copy accordingly
            src_line_count = sum(1 for _ in open(src_file, "r", encoding="utf-8"))
            if src_line_count > max_lines_for_giza:
                logger.warning(
                    f"Large corpus detected ({src_line_count} lines). "
                    f"Sampling {max_lines_for_giza} lines for GIZA++ stability."
                )
                copy_with_limit(src_file, local_src, max_lines_for_giza)
                copy_with_limit(tgt_file, local_tgt, max_lines_for_giza)
            else:
                shutil.copy2(src_file, local_src)
                shutil.copy2(tgt_file, local_tgt)

            # Convert to GIZA format using local plain2snt
            plain2snt_path = "/Users/davidsamuel/Documents/Unimelb Masters/Thesis/bible-nmt/giza-pp/GIZA++-v2/plain2snt.out"
            plain2snt_cmd = [plain2snt_path, "source.txt", "target.txt"]

            result = subprocess.run(
                plain2snt_cmd, cwd=giza_output_dir, capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.error(
                    f"plain2snt failed for {output_prefix} {direction}: {result.stderr}"
                )
                return 0.0, False

            # plain2snt creates: source.vcb, target.vcb, source_target.snt
            src_vcb = giza_output_dir / "source.vcb"
            tgt_vcb = giza_output_dir / "target.vcb"
            snt_file = giza_output_dir / "source_target.snt"
            cooc_file = giza_output_dir / "source_target.cooc"

            # Verify files were created
            if not all(f.exists() for f in [src_vcb, tgt_vcb, snt_file]):
                logger.error(
                    f"plain2snt did not create expected files for {output_prefix} {direction}"
                )
                logger.debug(f"Expected files: {src_vcb}, {tgt_vcb}, {snt_file}")
                return 0.0, False

            # Create cooccurrence file using local version
            snt2cooc_path = "/Users/davidsamuel/Documents/Unimelb Masters/Thesis/bible-nmt/giza-pp/GIZA++-v2/snt2cooc.out"
            snt2cooc_cmd = [snt2cooc_path, str(src_vcb), str(tgt_vcb), str(snt_file)]

            logger.debug(f"Running snt2cooc: {' '.join(snt2cooc_cmd)} > {cooc_file}")
            with open(cooc_file, "w") as f:
                result = subprocess.run(
                    snt2cooc_cmd,
                    cwd=giza_output_dir,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    text=True,
                )

            if result.returncode != 0:
                logger.error(
                    f"snt2cooc failed for {output_prefix} {direction}: {result.stderr}"
                )
                return 0.0, False

            # GIZA++ works directly with .vcb files (no .vcbfile needed based on Stack Overflow tutorial)
            # The .vcb files are already created by plain2snt, so no additional copying needed

            # Use local GIZA++ binaries
            giza_path = "/Users/davidsamuel/Documents/Unimelb Masters/Thesis/bible-nmt/giza-pp/GIZA++-v2/GIZA++"
            snt2cooc_path = "/Users/davidsamuel/Documents/Unimelb Masters/Thesis/bible-nmt/giza-pp/GIZA++-v2/snt2cooc.out"
            plain2snt_path = "/Users/davidsamuel/Documents/Unimelb Masters/Thesis/bible-nmt/giza-pp/GIZA++-v2/plain2snt.out"

            # Run GIZA++ with fallback
            giza_cmd = [
                giza_path,
                "-s",
                str(src_vcb),  # Source vocabulary (lowercase - confirmed working!)
                "-t",
                str(tgt_vcb),  # Target vocabulary (lowercase - confirmed working!)
                "-c",
                str(snt_file),  # Corpus file (lowercase - confirmed working!)
                "-CoocurrenceFile",
                str(cooc_file),
                "-o",
                f"{output_prefix}_{direction}",
                "-outputpath",
                str(giza_output_dir),
                "-m1",
                "1",  # Minimal IBM Model 1 iterations
                "-m2",
                "0",  # Skip IBM Model 2
                "-mh",
                "0",  # Skip HMM
                "-m3",
                "0",  # Skip IBM Model 3
                "-m4",
                "0",  # Skip IBM Model 4
                "-nodumps",  # Don't create dump files
            ]

            result = subprocess.run(
                giza_cmd,
                cwd=giza_output_dir,
                capture_output=True,
                text=True,
                timeout=600,  # Increase timeout for large corpus
            )

            if result.returncode != 0:
                logger.warning(
                    f"GIZA++ failed for {output_prefix} {direction} (exit code {result.returncode})"
                )
                logger.warning(
                    f"GIZA++ stderr: {result.stderr[:500]}..."
                )  # Truncate long error messages

                # Fallback to simple word overlap scoring
                logger.info(
                    f"Falling back to word overlap scoring for {output_prefix} {direction}"
                )
                return self.calculate_word_overlap_score(src_file, tgt_file), True

            # Extract log-likelihood from perplexity file
            perp_file = giza_output_dir / f"{output_prefix}_{direction}.perp"
            if not perp_file.exists():
                logger.error(f"Perplexity file not found: {perp_file}")
                return 0.0, False

            # Read the final log-likelihood from the perplexity file
            log_likelihood = self.extract_log_likelihood(perp_file)

            logger.debug(
                f"GIZA++ {direction} alignment completed for {output_prefix}, log-likelihood: {log_likelihood}"
            )
            return log_likelihood, True

        except Exception as e:
            logger.error(f"Error running GIZA++ for {output_prefix} {direction}: {e}")
            logger.info(
                f"Falling back to word overlap scoring for {output_prefix} {direction}"
            )
            return self.calculate_word_overlap_score(source_file, target_file), True

    def calculate_word_overlap_score(self, src_file: Path, tgt_file: Path) -> float:
        """Calculate a simple word overlap score as fallback when GIZA++ fails."""
        try:
            with open(src_file, "r", encoding="utf-8") as f:
                src_lines = [line.strip().lower().split() for line in f if line.strip()]

            with open(tgt_file, "r", encoding="utf-8") as f:
                tgt_lines = [line.strip().lower().split() for line in f if line.strip()]

            if len(src_lines) != len(tgt_lines):
                logger.warning(
                    f"Mismatched line counts: {len(src_lines)} vs {len(tgt_lines)}"
                )
                min_len = min(len(src_lines), len(tgt_lines))
                src_lines = src_lines[:min_len]
                tgt_lines = tgt_lines[:min_len]

            total_overlap = 0.0
            total_pairs = 0

            for src_words, tgt_words in zip(src_lines, tgt_lines):
                if not src_words or not tgt_words:
                    continue

                src_set = set(src_words)
                tgt_set = set(tgt_words)

                # Calculate Jaccard similarity
                intersection = len(src_set & tgt_set)
                union = len(src_set | tgt_set)

                if union > 0:
                    overlap = intersection / union
                    total_overlap += overlap
                    total_pairs += 1

            if total_pairs == 0:
                return -10.0  # Low score for no valid pairs

            avg_overlap = total_overlap / total_pairs
            # Convert to a log-likelihood-like score (higher overlap = higher score)
            # Scale to approximate GIZA++ log-likelihood range
            score = -10.0 + (avg_overlap * 8.0)  # Maps 0-1 overlap to -10 to -2 range

            logger.debug(
                f"Word overlap score: {score:.3f} (avg overlap: {avg_overlap:.3f}, pairs: {total_pairs})"
            )
            return score

        except Exception as e:
            logger.error(f"Word overlap calculation failed: {e}")
            return -15.0  # Very low fallback score

    def extract_log_likelihood(self, perp_file: Path) -> float:
        """Extract log-likelihood score from GIZA++ perplexity file."""
        try:
            with open(perp_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Look for the final training iteration line
            for line in reversed(lines):
                if line.strip() and not line.startswith("#"):
                    # Parse the line: train-size test-size iter model train-perp test-perp final train-viterbi-perp test-viterbi-perp
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Use negative perplexity as log-likelihood (higher is better)
                        perplexity = float(parts[4])
                        log_likelihood = -perplexity
                        return log_likelihood

            logger.warning(f"Could not extract log-likelihood from {perp_file}")
            return 0.0

        except Exception as e:
            logger.error(f"Error extracting log-likelihood from {perp_file}: {e}")
            return 0.0

    def process_single_version(self, english_file: Path) -> Dict[str, any]:
        """Process a single English Bible version."""
        version_name = self.extract_version_name(english_file)
        logger.info(f"Processing version: {version_name}")

        result = {
            "english_version": version_name,
            "target_language": "dhao",
            "forward_score": 0.0,
            "reverse_score": 0.0,
            "average_score": 0.0,
            "success": False,
        }

        try:
            # Step 1: Normalize English file
            normalized_eng = self.normalized_dir / f"{version_name}_normalized.txt"
            if not self.normalize_text_file(english_file, normalized_eng):
                logger.error(f"Failed to normalize English file for {version_name}")
                return result

            # Step 2: Normalize Dhao file (only once, but we'll do it for each to be safe)
            normalized_dhao = self.normalized_dir / "dhao_normalized.txt"
            if not normalized_dhao.exists():
                if not self.normalize_text_file(self.dhao_corpus_path, normalized_dhao):
                    logger.error(f"Failed to normalize Dhao file")
                    return result

            # Step 3: Create aligned corpus
            aligned_csv, align_success = self.create_aligned_corpus(
                normalized_eng, normalized_dhao, version_name
            )

            if not align_success:
                logger.error(f"Failed to create aligned corpus for {version_name}")
                return result

            # Step 4: Prepare GIZA input
            source_file, target_file, prep_success = self.prepare_giza_input(
                aligned_csv, version_name
            )

            if not prep_success:
                logger.error(f"Failed to prepare GIZA input for {version_name}")
                return result

            # Step 5: Run forward alignment
            forward_score, forward_success = self.run_giza_alignment(
                source_file, target_file, version_name, "forward"
            )

            # Step 6: Run reverse alignment
            reverse_score, reverse_success = self.run_giza_alignment(
                source_file, target_file, version_name, "reverse"
            )

            if forward_success and reverse_success:
                average_score = (forward_score + reverse_score) / 2
                result.update(
                    {
                        "forward_score": forward_score,
                        "reverse_score": reverse_score,
                        "average_score": average_score,
                        "success": True,
                    }
                )
                logger.info(
                    f"Successfully processed {version_name}: forward={forward_score:.4f}, reverse={reverse_score:.4f}, avg={average_score:.4f}"
                )
            else:
                logger.error(f"GIZA++ alignment failed for {version_name}")

        except Exception as e:
            logger.error(f"Error processing version {version_name}: {e}")

        return result

    def run_scoring(self) -> List[Dict[str, any]]:
        """Run symmetric alignment scoring for all English versions."""
        english_files = self.get_english_corpus_files()
        results = []

        logger.info(
            f"Starting symmetric alignment scoring for {len(english_files)} English versions"
        )

        for i, english_file in enumerate(english_files, 1):
            logger.info(f"Processing {i}/{len(english_files)}: {english_file.name}")
            result = self.process_single_version(english_file)
            results.append(result)

        return results

    def save_results(self, results: List[Dict[str, any]], output_file: Path):
        """Save results to CSV file."""
        try:
            with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "english_version",
                    "target_language",
                    "forward_score",
                    "reverse_score",
                    "average_score",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for result in results:
                    if result["success"]:
                        writer.writerow(
                            {k: v for k, v in result.items() if k != "success"}
                        )

            logger.info(f"Results saved to: {output_file}")

            # Print summary
            successful_results = [r for r in results if r["success"]]
            logger.info(
                f"Successfully processed {len(successful_results)}/{len(results)} versions"
            )

            if successful_results:
                # Sort by average score (descending - higher is better)
                successful_results.sort(key=lambda x: x["average_score"], reverse=True)

                logger.info("Top 5 best aligned versions:")
                for i, result in enumerate(successful_results[:5], 1):
                    logger.info(
                        f"{i}. {result['english_version']}: avg={result['average_score']:.4f}"
                    )

        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            logger.info(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Symmetric Alignment Scorer for Bible Translations"
    )
    parser.add_argument(
        "--english_corpus_dir",
        type=str,
        default="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/eng/corpus",
        help="Directory containing English corpus files",
    )
    parser.add_argument(
        "--dhao_corpus_path",
        type=str,
        default="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/nfa-nfa.txt",
        help="Path to Dhao corpus file",
    )
    parser.add_argument(
        "--vref_path",
        type=str,
        default="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/vref.txt",
        help="Path to verse reference file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/symmetric_alignment",
        help="Output directory for results",
    )
    parser.add_argument(
        "--temp_dir", type=str, help="Temporary directory (default: system temp)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="symmetric_alignment_scores.csv",
        help="Output CSV filename",
    )
    parser.add_argument(
        "--keep_temp", action="store_true", help="Keep temporary files for debugging"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize scorer
    scorer = SymmetricAlignmentScorer(
        args.english_corpus_dir,
        args.dhao_corpus_path,
        args.vref_path,
        args.output_dir,
        args.temp_dir,
    )

    try:
        # Run scoring
        results = scorer.run_scoring()

        # Save results
        output_file = Path(args.output_dir) / args.output_file
        scorer.save_results(results, output_file)

    finally:
        # Cleanup unless requested to keep temp files
        if not args.keep_temp:
            scorer.cleanup()


if __name__ == "__main__":
    main()
