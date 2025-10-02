#!/bin/bash

# vLLM Direct Translation using the new modular system
# Direct replacement for run_vllm_post_editing.sh but for translation mode

# Set working directory to project root
cd /data/projects/punim0478/setiawand/bible-nmt

# Change to post-editing directory for imports to work correctly
cd src/post-editing

# Generate timestamp for unique output directories
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT="/data/projects/punim0478/setiawand/bible-nmt/results"

vectorizers=(tfidf bm25 all_mpnet chrf_rag word_lcs word_parallel)
vectorizers=(bm25)
BASE_OUTPUT="/data/projects/punim0478/setiawand/bible-nmt/results"
CSV_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engBBE/aligned-eng-engBBE-ot.csv"
FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/ebible-corpus/dhao-eng/engBBE/aligned-eng-engBBE-nt.csv"
ADDITIONAL_FEW_SHOT_CORPUS_PATH="/data/projects/punim0478/setiawand/bible-nmt/lexical-resource/eng-dhao/dhao-english-parallel-full.csv"
SRC="eng"
TGT="nfa"
SRC_LANG_NAME="English"
TGT_LANG_NAME="Dhao"

python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ${CSV_PATH} \
  --few_shot_corpus_path ${ADDITIONAL_FEW_SHOT_CORPUS_PATH}\
  --src ${SRC} \
  --tgt ${TGT} \
  --src_lang_name ${SRC_LANG_NAME} \
  --tgt_lang_name ${TGT_LANG_NAME} \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-translation-5-shot/parallel_full" \
  --prompt "direct_translation" \
  --few_shot_mode "parallel" \
  --vectorizer "word_parallel" \
  --top_n_per_word 5 \
  --max_samples 500 \
  --translation-mode \
  --debug \
  --disable-metrics

python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ${CSV_PATH} \
  --few_shot_corpus_path ${ADDITIONAL_FEW_SHOT_CORPUS_PATH} ${FEW_SHOT_CORPUS_PATH}\
  --src ${SRC} \
  --tgt ${TGT} \
  --src_lang_name ${SRC_LANG_NAME} \
  --tgt_lang_name ${TGT_LANG_NAME} \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-translation-5-shot/parallel_full+nt" \
  --prompt "direct_translation" \
  --few_shot_mode "parallel" \
  --vectorizer "word_parallel" \
  --top_n_per_word 5 \
  --max_samples 500 \
  --translation-mode \
  --disable-metrics

python run_post_editing.py \
  --model_type "vllm" \
  --model_name "MISHANM/google-gemma-3-12b-it-fp8" \
  --csv_path ${CSV_PATH} \
  --few_shot_corpus_path ${FEW_SHOT_CORPUS_PATH}\
  --src ${SRC} \
  --tgt ${TGT} \
  --src_lang_name ${SRC_LANG_NAME} \
  --tgt_lang_name ${TGT_LANG_NAME} \
  --output_dir "${BASE_OUTPUT}/gemma-3-12b-it-translation-5-shot/parallel_nt" \
  --prompt "direct_translation" \
  --few_shot_mode "parallel" \
  --vectorizer "word_parallel" \
  --top_n_per_word 5 \
  --max_samples 500 \
  --translation-mode \
  --disable-metrics

echo "All translation experiments completed! Results saved to: ${BASE_OUTPUT}"

# Return to project root
cd ../..
