# 📖 Bible Neural Machine Translation with NLLB and LLMs

## 🔧 Setup

```sh
pip install -r requirements.txt
```

You also need access to a GPU that supports FlashAttention and bfloat16 (e.g. H100/L40S) if you're fine-tuning NLLB models.

## 🌏 Fine-tuning with NLLB

Fine-tune a multilingual translation model (e.g. [NLLB-200](huggingface.co/facebook/nllb-200-distilled-1.3b/)) for Bible verse translation between languages, using domain-specific corpora like:

- [bible-nlp/biblenlp-corpus](https://huggingface.co/datasets/bible-nlp/biblenlp-corpus)
- [LazarusNLP/alkitab-sabda-mt](https://huggingface.co/datasets/LazarusNLP/alkitab-sabda-mt)

It supports language-pair fine-tuning and evaluation using BLEU and chrF metrics, leveraging `transformers` and `datasets` from 🤗 Hugging Face, and is optimized for fast training using Flash Attention and bfloat16.

### 🚀 Usage

To fine-tune the NLLB-200 distilled 1.3B model from Indonesian to a regional language (e.g. Bambam) using eBible corpus, run:

```sh
python src/run_translation.py \
    --model_name facebook/nllb-200-distilled-1.3B \
    --dataset_name bible-nlp/biblenlp-corpus \
    --src_lang ind \
    --tgt_lang ptu \
    --src_lang_nllb ind_Latn \
    --tgt_lang_nllb ptu_Latn \
    --max_length 256 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 16 \
    --learning_rate 2e-4 \
    --max_steps 5000
```

To fine-tune the model from Indonesian to a regional language (e.g. Batak Karo) using Alkitab Sabda, run:

```sh
python src/run_translation.py \
    --model_name facebook/nllb-200-distilled-1.3B \
    --dataset_name LazarusNLP/alkitab-sabda-mt \
    --src_lang ind \
    --tgt_lang btx \
    --src_lang_nllb ind_Latn \
    --tgt_lang_nllb btx_Latn \
    --max_length 256 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 16 \
    --learning_rate 2e-4 \
    --max_steps 5000
```

### 📊 Results

| Source | Target | Dataset            | Eval Set          |  BLEU   |  chrF   |
| ------ | ------ | ------------------ | ----------------- | :-----: | :-----: |
| `ind`  | `ptu`  | `biblenlp-corpus`  | `validation`      | 18.4774 | 47.1922 |
| `ind`  | `ptu`  | `biblenlp-corpus`  | `test`            | 19.4204 | 49.0809 |
| `ind`  | `btx`  | `alkitab-sabda-mt` | `validation` (NT) | 26.2684 | 52.8884 |
| `ind`  | `btx`  | `alkitab-sabda-mt` | `test` (OT)       | 8.4475  | 28.9892 |

## 🦙 Few-shot Translation with LLMs

We can also leverage LLMs (e.g. GPT, Gemini) for few-shot translation of Bible verses, using the same datasets. This approach is more flexible and can be used for languages not covered by NLLB. We implemented few-shot prompting using TF-IDF, BM25, and Sentence Transformers to retrieve similar paired examples from the corpus, and then use them to prompt the LLM for translation.

### 🚀 Usage

To run few-shot translation using an LLM (e.g. Gemini 2.0 Flash) from Indonesian to Bambam with BM25 as the retriever, you can run the following command:

```sh
python src/run_llm_few_shot.py \
    --model gemini-2.0-flash \
    --src_lang ind \
    --tgt_lang ptu \
    --src_lang_name Indonesian \
    --tgt_lang_name Bambam \
    --vectorizer bm25 \
    --num_few_shot 5
```

### 📊 Results

| Source | Target | Dataset           | Eval Set |  BLEU   |  chrF   |
| ------ | ------ | ----------------- | -------- | :-----: | :-----: |
| `ind`  | `ptu`  | `biblenlp-corpus` | `test`   | 14.5615 | 41.1438 |



### Data Statistics
```
Language: aaz
  Source <=256: 9068 (100.00%)
  Source >256: 0 (0.00%)
  Target <=256: 9043 (99.72%)
  Target >256: 25 (0.28%)
  Total examples: 9068
----------------------------------------
Language: ptu
  Source <=256: 9345 (100.00%)
  Source >256: 0 (0.00%)
  Target <=256: 9345 (100.00%)
  Target >256: 0 (0.00%)
  Total examples: 9345
----------------------------------------
Language: nfa
  Source <=256: 9070 (100.00%)
  Source >256: 0 (0.00%)
  Target <=256: 9046 (99.74%)
  Target >256: 24 (0.26%)
  Total examples: 9070
----------------------------------------
Language: heg
  Source <=256: 9073 (100.00%)
  Source >256: 0 (0.00%)
  Target <=256: 9066 (99.92%)
  Target >256: 7 (0.08%)
  Total examples: 9073
----------------------------------------
Language: lex
  Source <=256: 9673 (99.99%)
  Source >256: 1 (0.01%)
  Target <=256: 9670 (99.96%)
  Target >256: 4 (0.04%)
  Total examples: 9674
----------------------------------------
Language: row
  Source <=256: 9070 (100.00%)
  Source >256: 0 (0.00%)
  Target <=256: 9062 (99.91%)
  Target >256: 8 (0.09%)
  Total examples: 9070
----------------------------------------
Language: llg
  Source <=256: 9071 (100.00%)
  Source >256: 0 (0.00%)
  Target <=256: 9065 (99.93%)
  Target >256: 6 (0.07%)
  Total examples: 9071
----------------------------------------
Language: rgu
  Source <=256: 8223 (100.00%)
  Source >256: 0 (0.00%)
  Target <=256: 8211 (99.85%)
  Target >256: 12 (0.15%)
  Total examples: 8223
----------------------------------------
Language: txq
  Source <=256: 9069 (100.00%)
  Source >256: 0 (0.00%)
  Target <=256: 9055 (99.85%)
  Target >256: 14 (0.15%)
  Total examples: 9069
----------------------------------------
Language: tet
  Source <=256: 9069 (100.00%)
  Source >256: 0 (0.00%)
  Target <=256: 9065 (99.96%)
  Target >256: 4 (0.04%)
  Total examples: 9069
----------------------------------------
Language: wrs
  Source <=256: 8981 (100.00%)
  Source >256: 0 (0.00%)
  Target <=256: 8979 (99.98%)
  Target >256: 2 (0.02%)
  Total examples: 8981
----------------------------------------
```