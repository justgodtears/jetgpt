import pandas as pd
import json
import re
import hashlib
import tqdm
from lingua import Language, LanguageDetectorBuilder
from pathlib import Path

detector = LanguageDetectorBuilder.from_all_languages().build()

def verify_language(text: str, langs_tag: list, detector_object) -> bool:
    """"""
    if "en" in langs_tag:
        # Returns a list with languages probabilities
        result = detector_object.compute_language_confidence_values(text)

        if result[0].language == Language.ENGLISH:
            return True
        else:
            return False
    else:
        return False


def normalize_text(text: str) -> str:
    """"""
    # Changing all Urls to [URL]
    text = re.sub(r"(https?://\S+|www\.\S+)", "[URL]", text)

    # Changing all mentions in post to "[MENTION]"
    text = re.sub(r"@\S+", "[MENTION]", text)

    # Removing all hashtags symbol before words
    text = re.sub(r"#", "", text)

    return text

def compute_hash(text: str) -> str:
    """"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def passes_length_filter(text: str, min_words: int = 3) -> bool:
    """"""
    text = text.split()
    if len(text) < min_words:
        return False
    else:
        return True

def process_posts(raw_data_dir: Path, output_path: Path, detector_object, min_words: int = 3) -> bool:
    ### Counting all lines in all files to properly set tqdm bar
    ### to see live progress of processing
    total_lines = 0
    for file in raw_data_dir.iterdir():
        with open(file, encoding="utf-8") as f:
            total_lines += sum(1 for _ in f)

    ### Main part of function
    seen_hashes = set()
    processed_posts = []

    with tqdm.tqdm(total=total_lines, desc="Processing posts") as pbar:
        for file in raw_data_dir.iterdir():
            with open(file, encoding="utf-8") as f:
                for row in f:
                    pbar.update(1)
                    try:
                        data = json.loads(row)
                        text = data["record"]["text"] if "text" in data["record"] else ""
                        langs_tag = data["record"]["langs"] if "langs" in data["record"] else None

                        if not isinstance(text, str):
                            continue

                        if not text.strip():
                            continue

                        if langs_tag is not None:
                            verified = verify_language(text, langs_tag, detector_object)

                            if verified:
                                normalized_text = normalize_text(text)
                                computed_hash = compute_hash(normalized_text)

                                if computed_hash in seen_hashes:
                                    continue
                                else:
                                    seen_hashes.add(computed_hash)

                                is_passed = passes_length_filter(normalized_text, min_words)

                                if is_passed:
                                    processed_posts.append({"text": normalized_text, "langs": langs_tag})
                        else:
                            continue
                    except Exception:
                        continue
    df = pd.DataFrame(processed_posts)
    df.to_parquet(output_path)

    return True


def merge_processed_parquets(parquet_paths: list[Path], output_path: Path) -> None:
    """"""
    parquets = [pd.read_parquet(file) for file in parquet_paths]

    df = pd.concat(parquets, ignore_index=True)
    print(f"Total rows before final dedup: {df.count()}")
    filter_df = df[~df["text"].str.contains(r"www\.\S+")]
    filter_df["hash"] = filter_df["text"].apply(compute_hash)
    deduplicated_df = filter_df.drop_duplicates(subset="hash", keep="first")
    deduplicated_df.drop(columns=["hash"], inplace=True)
    print(f"Total rows after final dedup: {deduplicated_df.count()}")

    deduplicated_df.to_parquet(output_path)