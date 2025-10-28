import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import difflib


# Copy the exact word extraction logic from vectorizers.py
def extract_words(text: str, min_length: int = 2):
    """Extract words from text, filtering by minimum length (from vectorizers.py)"""
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [word for word in words if len(word) >= min_length]


def efficient_string_similarity(word1: str, word2: str) -> float:
    """Calculate similarity using difflib (simplified version from vectorizers.py)"""
    if not word1 or not word2:
        return 0.0
    matcher = difflib.SequenceMatcher(None, word1.lower(), word2.lower())
    return matcher.ratio()


# Global variables for multiprocessing
corpus_words_cache_global = None
top_n_per_word_global = None
similarity_threshold_global = None


def init_worker(corpus_words_cache, top_n_per_word, similarity_threshold):
    """Initialize global variables for each worker process"""
    global corpus_words_cache_global, top_n_per_word_global, similarity_threshold_global
    corpus_words_cache_global = corpus_words_cache
    top_n_per_word_global = top_n_per_word
    similarity_threshold_global = similarity_threshold


def process_verse(src_text):
    """Process a single verse and return unique sentence count"""
    query_words = extract_words(src_text, min_length=2)
    sentence_aggregate_scores = {}

    for word in query_words:
        word_matches = []

        # Score all corpus sentences for this word
        for sent_idx, corpus_sent_words in enumerate(corpus_words_cache_global):
            # Find best similarity with words in this sentence
            best_similarity = 0.0
            for sent_word in corpus_sent_words:
                similarity = efficient_string_similarity(word, sent_word)
                if similarity > best_similarity:
                    best_similarity = similarity

            if best_similarity >= similarity_threshold_global:
                word_matches.append((sent_idx, best_similarity))

        # Sort and take top-n for this word
        word_matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = word_matches[:top_n_per_word_global]

        # Aggregate scores
        for sent_idx, score in top_matches:
            if sent_idx not in sentence_aggregate_scores:
                sentence_aggregate_scores[sent_idx] = 0.0
            sentence_aggregate_scores[sent_idx] += score

    return len(sentence_aggregate_scores)


def analyze_retrieval_budget():
    """
    Compare actual retrieval budget between word-level and sentence-level methods.

    Simulates the exact preprocessing used in the retrieval code to measure:
    - Word-level (top-10): How many unique sentences actually retrieved per verse?
    - Sentence-level (k=60): Fixed 60 sentences per verse

    Uses first 500 verses from OT test set.
    """

    print("=" * 80)
    print("RETRIEVAL BUDGET ANALYSIS: Word-Level vs Sentence-Level")
    print("=" * 80)

    # Load test data from OT
    base_path = Path("/Users/davidsamuel/Documents/Unimelb Masters/Thesis/bible-nmt")
    test_file = (
        base_path
        / "results"
        / "gemini-2.5-flash-engwebp-dhao"
        / "vectorizer_ablation"
        / "word_parallel_top10"
        / "gemini-2.5-flash_aligned-eng-engwebp-ot_eng_nfa.csv"
    )

    if not test_file.exists():
        print(f"Error: Test file not found at {test_file}")
        return

    # Load the test data
    test_df = pd.read_csv(test_file)
    print(f"\nLoaded test data: {len(test_df)} total verses")

    # Take first 500 verses
    test_df_500 = test_df.head(500)
    print(f"Analyzing first {len(test_df_500)} verses")

    # Analyze word-level retrieval budget
    print(f"\n" + "=" * 60)
    print("WORD-LEVEL RETRIEVAL (top-10 per word)")
    print("=" * 60)

    top_n_per_word = 10
    total_words = 0
    word_counts_per_verse = []

    for idx, row in test_df_500.iterrows():
        src_text = row["src_text"]
        words = extract_words(src_text, min_length=2)
        total_words += len(words)
        word_counts_per_verse.append(len(words))

    avg_words_per_verse = total_words / len(test_df_500)

    print(f"Word extraction statistics:")
    print(f"  Total words extracted: {total_words}")
    print(f"  Average words per verse: {avg_words_per_verse:.2f}")
    print(f"  Min words per verse: {min(word_counts_per_verse)}")
    print(f"  Max words per verse: {max(word_counts_per_verse)}")
    print(f"  Median words per verse: {np.median(word_counts_per_verse):.2f}")

    # Load NT corpus (few-shot corpus used during actual retrieval)
    print(f"\n📚 Loading NT corpus (few-shot examples)...")
    nt_corpus_file = (
        base_path
        / "ebible-corpus"
        / "dhao-eng"
        / "engwebp"
        / "aligned-eng-engwebp-nt.csv"
    )

    if not nt_corpus_file.exists():
        print(f"Error: NT corpus not found at {nt_corpus_file}")
        return

    nt_corpus_df = pd.read_csv(nt_corpus_file)
    nt_corpus = [
        (row["source_text"], row["target_text"]) for _, row in nt_corpus_df.iterrows()
    ]
    print(f"Loaded NT corpus: {len(nt_corpus)} parallel sentences")

    # Precompute corpus words for efficient similarity matching
    print(f"Preprocessing corpus...")
    corpus_source_texts = [pair[0] for pair in nt_corpus]
    corpus_words_cache = [
        extract_words(text, min_length=2) for text in corpus_source_texts
    ]

    # Actually run word-based retrieval on each verse using parallel processing
    print(
        f"\n🔄 Running actual word-based retrieval (top-{top_n_per_word} per word)..."
    )
    print(f"   Processing {len(test_df_500)} verses with parallel workers...")

    similarity_threshold = 0.5

    # Prepare source texts for parallel processing
    n_workers = min(cpu_count() - 1, 8)  # Use up to 8 workers, leave 1 core free
    print(f"   Using {n_workers} parallel workers")

    src_texts = [row["src_text"] for _, row in test_df_500.iterrows()]

    # Process in parallel with progress bar
    with Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(corpus_words_cache, top_n_per_word, similarity_threshold),
    ) as pool:
        actual_unique_examples = list(
            tqdm(
                pool.imap(process_verse, src_texts),
                total=len(src_texts),
                desc="Retrieving",
            )
        )

    avg_actual_unique = np.mean(actual_unique_examples)

    # Also calculate the theoretical maximum
    max_examples_per_verse = [w * top_n_per_word for w in word_counts_per_verse]
    avg_max_examples = np.mean(max_examples_per_verse)

    print(f"\n✅ Actual retrieval complete!")
    print(f"\nRetrieval budget (top-{top_n_per_word} per word):")
    print(f"  Maximum possible (no overlap): {avg_max_examples:.2f} examples/verse")
    print(f"  Actual unique retrieved: {avg_actual_unique:.2f} examples/verse")
    print(f"  Actual overlap rate: {(1 - avg_actual_unique/avg_max_examples)*100:.1f}%")

    print(f"\nDetailed statistics:")
    print(f"  Min unique per verse: {min(actual_unique_examples)}")
    print(f"  Max unique per verse: {max(actual_unique_examples)}")
    print(f"  Median unique per verse: {np.median(actual_unique_examples):.2f}")
    print(f"  Total for 500 verses: {sum(actual_unique_examples):,}")

    # Analyze sentence-level retrieval budget
    print(f"\n" + "=" * 60)
    print("SENTENCE-LEVEL RETRIEVAL (k=60 fixed)")
    print("=" * 60)

    k_sentence = 60
    total_sentence_examples = len(test_df_500) * k_sentence

    print(f"Retrieval budget (k={k_sentence}):")
    print(f"  Examples per verse: {k_sentence} (fixed)")
    print(f"  Total for 500 verses: {total_sentence_examples:,}")

    # Comparison
    print(f"\n" + "=" * 60)
    print("BUDGET COMPARISON")
    print("=" * 60)

    print(f"\nWord-level (top-{top_n_per_word}):")
    print(
        f"  Theoretical maximum: {avg_max_examples:.2f} examples/verse ({sum(max_examples_per_verse):,} total)"
    )
    print(
        f"  Actual unique retrieved: {avg_actual_unique:.2f} examples/verse ({sum(actual_unique_examples):,} total)"
    )

    print(f"\nSentence-level (k={k_sentence}):")
    print(f"  Fixed: {k_sentence} examples/verse ({total_sentence_examples:,} total)")

    print(f"\nBudget efficiency:")
    word_to_sentence_ratio = avg_actual_unique / k_sentence
    if word_to_sentence_ratio < 1:
        print(
            f"  ✅ Word-level uses {word_to_sentence_ratio:.1%} of sentence-level budget"
        )
        print(
            f"  ✅ Word-level is {1/word_to_sentence_ratio:.2f}x more budget-efficient"
        )
    else:
        print(
            f"  ⚠️  Word-level uses {word_to_sentence_ratio:.1%} of sentence-level budget"
        )
        print(f"  ⚠️  Word-level uses {word_to_sentence_ratio:.2f}x more examples")

    # Load metrics data for performance comparison
    vectorizer_path = (
        base_path / "results" / "gemini-2.5-flash-engwebp-dhao" / "vectorizer_ablation"
    )
    all_data = []

    # Load vectorizer_ablation subdirectories
    for subdir in vectorizer_path.iterdir():
        if not subdir.is_dir():
            continue

        method_name = subdir.name

        # Parse method type and parameter
        if method_name.startswith("bge_k"):
            method_type, parameter = "BGE", int(method_name.split("_k")[1])
        elif method_name.startswith("bm25_k"):
            method_type, parameter = "BM25", int(method_name.split("_k")[1])
        elif method_name.startswith("chrf_rag_k"):
            method_type, parameter = "ChrF RAG", int(method_name.split("_k")[1])
        elif method_name.startswith("word_parallel_top"):
            method_type, parameter = "Fuzzy Word Matching", int(
                method_name.split("_top")[1]
            )
        else:
            continue

        # Load metrics file
        metrics_file = (
            subdir / "gemini-2.5-flash_aligned-eng-engwebp-ot_eng_nfa_metrics.json"
        )
        if metrics_file.exists():
            import json

            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            original = metrics["original_mt_metrics"]
            post_edited = metrics["post_edited_metrics"]
            improvements = metrics["improvements"]

            for metric in original.keys():
                all_data.append(
                    {
                        "method_name": method_name,
                        "method_type": method_type,
                        "parameter": parameter,
                        "metric": metric,
                        "improvement": improvements[metric],
                    }
                )

    df = pd.DataFrame(all_data)

    # Performance comparison (from best configs)
    df_chrf = df[df["metric"] == "chrf++"].copy()

    word_level_perf = df_chrf[df_chrf["method_name"] == "word_parallel_top10"][
        "improvement"
    ].values
    sentence_level_60 = df_chrf[
        (df_chrf["method_type"].isin(["BGE", "ChrF RAG", "BM25"]))
        & (df_chrf["parameter"] == 60)
    ]

    if len(word_level_perf) > 0 and len(sentence_level_60) > 0:
        word_improvement = word_level_perf[0]
        # Get best sentence-level method at k=60
        best_sentence_60 = sentence_level_60.loc[
            sentence_level_60["improvement"].idxmax()
        ]

        print(f"\n" + "=" * 60)
        print("PERFORMANCE COMPARISON (chrF++)")
        print("=" * 60)

        print(f"\nWord-level (top-{top_n_per_word}):")
        print(f"  chrF++ improvement: {word_improvement:.3f}")
        print(f"  Budget: {avg_actual_unique:.1f} examples/verse (actual)")
        print(
            f"  Efficiency: {word_improvement/avg_actual_unique:.4f} improvement/example"
        )

        print(f"\nSentence-level ({best_sentence_60['method_type']} k={k_sentence}):")
        print(f"  chrF++ improvement: {best_sentence_60['improvement']:.3f}")
        print(f"  Budget: {k_sentence} examples/verse")
        print(
            f"  Efficiency: {best_sentence_60['improvement']/k_sentence:.4f} improvement/example"
        )

        print(f"\n🎯 VERDICT:")
        word_efficiency = word_improvement / avg_actual_unique
        sentence_efficiency = best_sentence_60["improvement"] / k_sentence

        if word_efficiency > sentence_efficiency:
            ratio = word_efficiency / sentence_efficiency
            print(f"  ✅ Word-level is {ratio:.2f}x more budget-efficient!")
            print(
                f"     ({word_efficiency:.4f} vs {sentence_efficiency:.4f} improvement per example)"
            )
        else:
            ratio = sentence_efficiency / word_efficiency
            print(f"  ⚠️  Sentence-level is {ratio:.2f}x more budget-efficient")
            print(
                f"     ({sentence_efficiency:.4f} vs {word_efficiency:.4f} improvement per example)"
            )

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Budget comparison
    methods = ["Word-level\n(top-10)", f"Sentence-level\n(k={k_sentence})"]
    budgets = [avg_actual_unique, k_sentence]
    colors_budget = ["#d62728", "#2ca02c"]

    bars1 = ax1.bar(
        methods,
        budgets,
        color=colors_budget,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.set_ylabel("Examples per Verse", fontsize=14, fontweight="bold")
    ax1.set_title("Retrieval Budget Comparison", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, budget in zip(bars1, budgets):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{budget:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    # Plot 2: Efficiency comparison (improvement per example)
    if len(word_level_perf) > 0 and len(sentence_level_60) > 0:
        efficiencies = [word_efficiency, sentence_efficiency]
        bars2 = ax2.bar(
            methods,
            efficiencies,
            color=colors_budget,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax2.set_ylabel("chrF++ Improvement per Example", fontsize=14, fontweight="bold")
        ax2.set_title("Budget Efficiency Comparison", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, eff in zip(bars2, efficiencies):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{eff:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
            )

    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("budget_analysis.png", dpi=300, bbox_inches="tight")
    print("\n📊 Chart saved as 'budget_analysis.png'")
    plt.show()


if __name__ == "__main__":
    analyze_retrieval_budget()
