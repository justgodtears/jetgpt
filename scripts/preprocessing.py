from project_blue.preprocessing import process_posts
from project_blue.preprocessing import detector
from pathlib import Path

process_posts(
    raw_data_dir=Path("../data/raw"),
    output_path=Path("../data/processed/processed_data__17-00__08_31_2026.parquet"),
    detector_object=detector,
    min_words=3
)