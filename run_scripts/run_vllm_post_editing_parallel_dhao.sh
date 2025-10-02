#!/bin/bash

# vLLM Post-Editing using the new modular system
# Direct replacement for run_vllm_post_editing.sh

# Set working directory to project root
cd /data/projects/punim0478/setiawand/bible-nmt

# Change to post-editing directory for imports to work correctly
cd src/post-editing

# Generate timestamp for unique output directories
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT="/data/projects/punim0478/setiawand/bible-nmt/results"
ADDITIONAL_FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/lexical-resource/eng-dhao/dhao-english-parallel-full.csv"
SRC="eng"
TGT="nfa"
SRC_LANG_NAME="English"
TGT_LANG_NAME="Dhao"

# =============================================================================
# Experiment 1-3: engBBE dataset (3 variations)
# =============================================================================
echo "Starting engBBE experiments..."
CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engBBE/aligned-eng-engBBE-ot.csv"
FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engBBE/aligned-eng-engBBE-nt.csv"

# engBBE - parallel_full
python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ${CSV_PATH} \
  --few_shot_corpus_path ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
  --src ${SRC} \
  --tgt ${TGT} \
  --src_lang_name ${SRC_LANG_NAME} \
  --tgt_lang_name ${TGT_LANG_NAME} \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-engBBE-dhao/parallel_full" \
  --prompt "dhao_post_editing" \
  --few_shot_mode "parallel" \
  --vectorizer "word_parallel" \
  --top_n_per_word 5 \
  --max_samples 500 \
  --debug \
  --disable-metrics

# engBBE - parallel_nt
python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ${CSV_PATH} \
  --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} \
  --src ${SRC} \
  --tgt ${TGT} \
  --src_lang_name ${SRC_LANG_NAME} \
  --tgt_lang_name ${TGT_LANG_NAME} \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-engBBE-dhao/parallel_nt" \
  --prompt "dhao_post_editing" \
  --few_shot_mode "parallel" \
  --vectorizer "word_parallel" \
  --top_n_per_word 5 \
  --max_samples 500 \
  --debug \
  --disable-metrics

# engBBE - parallel_full+nt
python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ${CSV_PATH} \
  --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
  --src ${SRC} \
  --tgt ${TGT} \
  --src_lang_name ${SRC_LANG_NAME} \
  --tgt_lang_name ${TGT_LANG_NAME} \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-engBBE-dhao/parallel_full+nt" \
  --prompt "dhao_post_editing" \
  --few_shot_mode "parallel" \
  --vectorizer "word_parallel" \
  --top_n_per_word 5 \
  --max_samples 500 \
  --debug \
  --disable-metrics

# =============================================================================
# Experiment 4-6: engwebp dataset (3 variations)
# =============================================================================
echo "Starting engwebp experiments..."
CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engwebp/aligned-eng-engwebp-ot.csv"
FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engwebp/aligned-eng-engwebp-nt.csv"

# engwebp - parallel_full
python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ${CSV_PATH} \
  --few_shot_corpus_path ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
  --src ${SRC} \
  --tgt ${TGT} \
  --src_lang_name ${SRC_LANG_NAME} \
  --tgt_lang_name ${TGT_LANG_NAME} \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-engwebp-dhao/parallel_full" \
  --prompt "dhao_post_editing" \
  --few_shot_mode "parallel" \
  --vectorizer "word_parallel" \
  --top_n_per_word 5 \
  --max_samples 500 \
  --debug \
  --disable-metrics

# engwebp - parallel_nt
python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ${CSV_PATH} \
  --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} \
  --src ${SRC} \
  --tgt ${TGT} \
  --src_lang_name ${SRC_LANG_NAME} \
  --tgt_lang_name ${TGT_LANG_NAME} \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-engwebp-dhao/parallel_nt" \
  --prompt "dhao_post_editing" \
  --few_shot_mode "parallel" \
  --vectorizer "word_parallel" \
  --top_n_per_word 5 \
  --max_samples 500 \
  --debug \
  --disable-metrics

# engwebp - parallel_full+nt
python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ${CSV_PATH} \
  --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
  --src ${SRC} \
  --tgt ${TGT} \
  --src_lang_name ${SRC_LANG_NAME} \
  --tgt_lang_name ${TGT_LANG_NAME} \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-engwebp-dhao/parallel_full+nt" \
  --prompt "dhao_post_editing" \
  --few_shot_mode "parallel" \
  --vectorizer "word_parallel" \
  --top_n_per_word 5 \
  --max_samples 500 \
  --debug \
  --disable-metrics

# =============================================================================
# Experiment 7-9: engwyc2017 dataset (3 variations)
# =============================================================================
echo "Starting engwyc2017 experiments..."
CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engwyc2017/aligned-eng-engwyc2017-ot.csv"
FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engwyc2017/aligned-eng-engwyc2017-nt.csv"

# engwyc2017 - parallel_full
python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ${CSV_PATH} \
  --few_shot_corpus_path ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
  --src ${SRC} \
  --tgt ${TGT} \
  --src_lang_name ${SRC_LANG_NAME} \
  --tgt_lang_name ${TGT_LANG_NAME} \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-engwyc2017-dhao/parallel_full" \
  --prompt "dhao_post_editing" \
  --few_shot_mode "parallel" \
  --vectorizer "word_parallel" \
  --top_n_per_word 5 \
  --max_samples 500 \
  --debug \
  --disable-metrics

# engwyc2017 - parallel_nt
python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ${CSV_PATH} \
  --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} \
  --src ${SRC} \
  --tgt ${TGT} \
  --src_lang_name ${SRC_LANG_NAME} \
  --tgt_lang_name ${TGT_LANG_NAME} \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-engwyc2017-dhao/parallel_nt" \
  --prompt "dhao_post_editing" \
  --few_shot_mode "parallel" \
  --vectorizer "word_parallel" \
  --top_n_per_word 5 \
  --max_samples 500 \
  --debug \
  --disable-metrics

# engwyc2017 - parallel_full+nt
python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ${CSV_PATH} \
  --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
  --src ${SRC} \
  --tgt ${TGT} \
  --src_lang_name ${SRC_LANG_NAME} \
  --tgt_lang_name ${TGT_LANG_NAME} \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-engwyc2017-dhao/parallel_full+nt" \
  --prompt "dhao_post_editing" \
  --few_shot_mode "parallel" \
  --vectorizer "word_parallel" \
  --top_n_per_word 5 \
  --max_samples 500 \
  --debug \
  --disable-metrics

# =============================================================================
# Experiment 10-12: engwyc2018 dataset (3 variations)
# =============================================================================
echo "Starting engwyc2018 experiments..."
CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engwyc2018/aligned-eng-engwyc2018-ot.csv"
FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engwyc2018/aligned-eng-engwyc2018-nt.csv"

# engwyc2018 - parallel_full
python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ${CSV_PATH} \
  --few_shot_corpus_path ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
  --src ${SRC} \
  --tgt ${TGT} \
  --src_lang_name ${SRC_LANG_NAME} \
  --tgt_lang_name ${TGT_LANG_NAME} \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-engwyc2018-dhao/parallel_full" \
  --prompt "dhao_post_editing" \
  --few_shot_mode "parallel" \
  --vectorizer "word_parallel" \
  --top_n_per_word 5 \
  --max_samples 500 \
  --debug \
  --disable-metrics

# engwyc2018 - parallel_nt
python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ${CSV_PATH} \
  --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} \
  --src ${SRC} \
  --tgt ${TGT} \
  --src_lang_name ${SRC_LANG_NAME} \
  --tgt_lang_name ${TGT_LANG_NAME} \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-engwyc2018-dhao/parallel_nt" \
  --prompt "dhao_post_editing" \
  --few_shot_mode "parallel" \
  --vectorizer "word_parallel" \
  --top_n_per_word 5 \
  --max_samples 500 \
  --debug \
  --disable-metrics

# engwyc2018 - parallel_full+nt
python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ${CSV_PATH} \
  --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH} ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} \
  --src ${SRC} \
  --tgt ${TGT} \
  --src_lang_name ${SRC_LANG_NAME} \
  --tgt_lang_name ${TGT_LANG_NAME} \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-engwyc2018-dhao/parallel_full+nt" \
  --prompt "dhao_post_editing" \
  --few_shot_mode "parallel" \
  --vectorizer "word_parallel" \
  --top_n_per_word 5 \
  --max_samples 500 \
  --debug \
  --disable-metrics

echo "All 12 experiments completed! Results saved to: ${BASE_OUTPUT}"
echo "Experiment summary:"
echo "- engBBE: 3 variations (parallel_full, parallel_nt, parallel_full+nt)"  
echo "- engwebp: 3 variations (parallel_full, parallel_nt, parallel_full+nt)"
echo "- engwyc2017: 3 variations (parallel_full, parallel_nt, parallel_full+nt)"
echo "- engwyc2018: 3 variations (parallel_full, parallel_nt, parallel_full+nt)"

# Return to project root
cd ../..
