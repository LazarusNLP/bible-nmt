#!/usr/bin/env python3
"""
Direct GIZA++ alignment runner.

This script runs GIZA++ alignment directly using the compiled binaries,
bypassing the problematic Python wrapper dependencies.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_giza_alignment(source_file, target_file, output_dir, giza_bin_dir):
    """
    Run GIZA++ alignment directly using the compiled binaries.

    Args:
        source_file (str): Path to source language file
        target_file (str): Path to target language file
        output_dir (str): Output directory for alignment files
        giza_bin_dir (str): Directory containing GIZA++ binaries
    """

    # Convert to Path objects
    source_path = Path(source_file)
    target_path = Path(target_file)
    output_path = Path(output_dir)
    bin_path = Path(giza_bin_dir)

    # Validate inputs
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_file}")
    if not bin_path.exists():
        raise FileNotFoundError(f"GIZA++ bin directory not found: {giza_bin_dir}")

    # Create output directory
    output_path.mkdir(exist_ok=True)

    print(f"Source: {source_file}")
    print(f"Target: {target_file}")
    print(f"Output: {output_dir}")
    print(f"GIZA++ binaries: {giza_bin_dir}")

    # Step 1: Convert plain text to snt format
    print("\n=== Step 1: Converting to snt format ===")
    plain2snt_cmd = [str(bin_path / "plain2snt"), str(source_path), str(target_path)]

    # Run plain2snt in output directory
    result = subprocess.run(
        plain2snt_cmd, cwd=output_path, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"plain2snt failed: {result.stderr}")
        return False
    print("✓ Converted to snt format")

    # Step 2: Generate cooccurrence file
    print("\n=== Step 2: Generating cooccurrence file ===")
    snt2cooc_cmd = [
        str(bin_path / "snt2cooc"),
        str(output_path / "cooc.cooc"),
        str(output_path / f"{source_path.stem}.vcb"),
        str(output_path / f"{target_path.stem}.vcb"),
        str(output_path / f"{source_path.stem}_{target_path.stem}.snt"),
    ]

    result = subprocess.run(
        snt2cooc_cmd, cwd=output_path, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"snt2cooc failed: {result.stderr}")
        return False
    print("✓ Generated cooccurrence file")

    # Step 3: Generate word classes using mkcls
    print("\n=== Step 3: Generating word classes ===")
    # Generate classes for source
    mkcls_src_cmd = [
        str(bin_path / "mkcls"),
        "-p" + str(source_path),
        "-V" + str(output_path / f"{source_path.stem}.vcb.classes"),
    ]

    result = subprocess.run(
        mkcls_src_cmd, cwd=output_path, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"mkcls (source) failed: {result.stderr}")
        # Continue anyway, classes are optional
        print("⚠ Warning: Source word classes generation failed, continuing...")
    else:
        print("✓ Generated source word classes")

    # Generate classes for target
    mkcls_tgt_cmd = [
        str(bin_path / "mkcls"),
        "-p" + str(target_path),
        "-V" + str(output_path / f"{target_path.stem}.vcb.classes"),
    ]

    result = subprocess.run(
        mkcls_tgt_cmd, cwd=output_path, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"mkcls (target) failed: {result.stderr}")
        print("⚠ Warning: Target word classes generation failed, continuing...")
    else:
        print("✓ Generated target word classes")

    # Step 4: Run GIZA++ alignment
    print("\n=== Step 4: Running GIZA++ alignment ===")
    giza_cmd = [
        str(bin_path / "GIZA++"),
        "-S" + str(output_path / f"{source_path.stem}.vcb"),
        "-T" + str(output_path / f"{target_path.stem}.vcb"),
        "-C" + str(output_path / f"{source_path.stem}_{target_path.stem}.snt"),
        "-o" + "alignment",
        "-outputpath" + str(output_path) + "/",
        "-m1" + "5",  # 5 iterations of Model 1
        "-m2" + "0",  # Skip Model 2
        "-mh" + "5",  # 5 iterations of HMM
        "-m3" + "3",  # 3 iterations of Model 3
        "-m4" + "3",  # 3 iterations of Model 4
    ]

    result = subprocess.run(giza_cmd, cwd=output_path, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"GIZA++ failed: {result.stderr}")
        return False

    print("✓ GIZA++ alignment completed!")

    # List output files
    print(f"\n=== Alignment files created in {output_path} ===")
    for file_path in sorted(output_path.glob("alignment.*")):
        print(f"  - {file_path.name}")

    return True


def main():
    """Main function to run GIZA++ alignment."""
    import argparse

    parser = argparse.ArgumentParser(description="Run GIZA++ alignment directly")
    parser.add_argument("source", help="Source language file")
    parser.add_argument("target", help="Target language file")
    parser.add_argument(
        "--output",
        "-o",
        default="giza_output",
        help="Output directory (default: giza_output)",
    )
    parser.add_argument(
        "--bin-dir",
        default="giza-pp/.bin",
        help="GIZA++ binaries directory (default: giza-pp/.bin)",
    )

    args = parser.parse_args()

    try:
        success = run_giza_alignment(
            args.source, args.target, args.output, args.bin_dir
        )
        if success:
            print("\n🎉 GIZA++ alignment completed successfully!")
            print(f"Check the output directory: {args.output}")
        else:
            print("\n❌ GIZA++ alignment failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
