import pandas as pd
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def export_text_for_tokenizer(parquet_path: Path, output_path: Path) -> None:
    """"""
    df = pd.read_parquet(parquet_path)

    with output_path.open("w", encoding="utf-8") as f:
        for text in df.text.values:
            text = " ".join(text.splitlines())
            f.write(text + "\n")


def train_tokenizer(corpus_path: Path, vocab_size: int, output_path: Path) -> None:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)
    tokenizer.save(str(output_path))

