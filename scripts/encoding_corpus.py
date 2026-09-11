from project_blue.tokenizer import encode_corpus
from pathlib import Path

encode_corpus(
    parquet_path=Path("../data/processed/processed_dataset.parquet"),
    tokenizer_path=Path("../data/tokenized/tokenizer_24k.json"),
    output_path=Path("../data/tokenized/encoded_corpus.npy"),
)