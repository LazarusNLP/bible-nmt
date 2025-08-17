"""eBible Indonesian Local Language Corpus dataset."""

import json
import os
from typing import Dict, List, Any
from datasets import Dataset, DatasetDict, DownloadConfig, SplitGenerator, Version, Features, Value


_CITATION = """\n\n@misc{ebible_indonesian_2024,
  title={eBible Indonesian Local Language Corpus},
  author={Dataset created from eBible corpus},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/Davidsamuel101/ebible_local_ind_corpus}
}\n\n"""

_DESCRIPTION = """\n\nThis dataset contains parallel Bible translations between Indonesian and various local languages from Indonesia, particularly from Eastern Indonesia regions.\n\n"""

_HOMEPAGE = "https://huggingface.co/datasets/Davidsamuel101/ebible_local_ind_corpus"

_LICENSE = "Please refer to the original eBible corpus license terms."

_URLS = {
    "ind_aaz": "data/ind_aaz-00000-of-00001.parquet",
    "ind_heg": "data/ind_heg-00000-of-00001.parquet",
    "ind_lex": "data/ind_lex-00000-of-00001.parquet",
    "ind_llg": "data/ind_llg-00000-of-00001.parquet",
    "ind_nfa": "data/ind_nfa-00000-of-00001.parquet",
    "ind_ptu": "data/ind_ptu-00000-of-00001.parquet",
    "ind_rgu": "data/ind_rgu-00000-of-00001.parquet",
    "ind_row": "data/ind_row-00000-of-00001.parquet",
    "ind_tet": "data/ind_tet-00000-of-00001.parquet",
    "ind_txq": "data/ind_txq-00000-of-00001.parquet",
    "ind_wrs": "data/ind_wrs-00000-of-00001.parquet",
}

_FEATURES = Features({
    "source_text": Value("string"),
    "target_text": Value("string"),
    "source_lang": Value("string"),
    "target_lang": Value("string"),
    "verse": Value("string"),
})

_VERSION = Version("0.0.0")


class EbibleLocalIndCorpusConfig(datasets.BuilderConfig):
    """BuilderConfig for eBible Local Indonesian Corpus."""
    
    def __init__(self, **kwargs):
        super().__init__(version=_VERSION, **kwargs)


class EbibleLocalIndCorpus(datasets.GeneratorBasedBuilder):
    """eBible Indonesian Local Language Corpus dataset."""

    VERSION = _VERSION
    BUILDER_CONFIGS = [
        EbibleLocalIndCorpusConfig(
            name="ind_aaz",
            description="Indonesian to Amarasi (Uab Meto) Bible translation",
        ),
        EbibleLocalIndCorpusConfig(
            name="ind_heg",
            description="Indonesian to Helong Bible translation",
        ),
        EbibleLocalIndCorpusConfig(
            name="ind_lex",
            description="Indonesian to Luang Bible translation",
        ),
        EbibleLocalIndCorpusConfig(
            name="ind_llg",
            description="Indonesian to Lole Bible translation",
        ),
        EbibleLocalIndCorpusConfig(
            name="ind_nfa",
            description="Indonesian to Dhao Bible translation",
        ),
        EbibleLocalIndCorpusConfig(
            name="ind_ptu",
            description="Indonesian to Bambam Bible translation",
        ),
        EbibleLocalIndCorpusConfig(
            name="ind_rgu",
            description="Indonesian to Rikou Bible translation",
        ),
        EbibleLocalIndCorpusConfig(
            name="ind_row",
            description="Indonesian to Dela-Oenale Bible translation",
        ),
        EbibleLocalIndCorpusConfig(
            name="ind_tet",
            description="Indonesian to Tetun Bible translation",
        ),
        EbibleLocalIndCorpusConfig(
            name="ind_txq",
            description="Indonesian to Tii Bible translation",
        ),
        EbibleLocalIndCorpusConfig(
            name="ind_wrs",
            description="Indonesian to Waris Bible translation",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls = _URLS[self.config.name]
        data_file = dl_manager.download_and_extract(urls)
        
        return [
            SplitGenerator(
                name="train",
                gen_kwargs={"filepath": data_file},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        import pandas as pd
        
        df = pd.read_parquet(filepath)
        
        for idx, row in df.iterrows():
            yield idx, {
                "source_text": row["source_text"],
                "target_text": row["target_text"],
                "source_lang": row["source_lang"],
                "target_lang": row["target_lang"],
                "verse": row["verse"],
            }
