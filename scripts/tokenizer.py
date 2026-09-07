from pathlib import Path
from project_blue.tokenizer import train_tokenizer

train_tokenizer(
    Path("../data/archive/processed_corpus__17-00__08_31_2026.txt"),
    24000,
    Path("../data/tokenized/tokenizer_24k.json")
)