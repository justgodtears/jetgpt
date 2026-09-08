from project_blue.preprocessing import merge_processed_parquets
from pathlib import Path

merge_processed_parquets(
    parquet_paths=[
        Path("../data/processed/processed_data__09-00__09_06_2026.parquet"),
        Path("../data/processed/processed_data__13-30__09_04_2026.parquet"),
        Path("../data/processed/processed_data__17-00__08_31_2026.parquet"),
        Path("../data/processed/processed_data__17-45__09_02_2026.parquet"),
    ],
    output_path=Path("../data/processed/processed_dataset.parquet")
)