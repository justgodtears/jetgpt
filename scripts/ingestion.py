from project_blue.ingestion import main
from pathlib import Path
import uuid
import asyncio

# Directiories
seq_file: Path = Path("../data/state/last_seq.txt")
raw_data_dir: Path = Path(f"../data/raw/data-{uuid.uuid4()}.json")


asyncio.run(main(raw_data_dir, seq_file))