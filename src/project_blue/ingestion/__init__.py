import uuid
import websockets
import datetime
import asyncio
import json
from pathlib import Path
from typing import TextIO


def save_data_to_jsonl(opened_file: TextIO, data: dict) -> None:
    """
    Saves data to opened file in JSON-L format.
    Args:
        opened_file: object of file where data is to be saved
        data: dictionary from firehose

    Returns:
        nothing
    """
    line = json.dumps(data)
    opened_file.write(line + "\n")


def save_checkpoint(path: Path, seq: int) -> None:
    """
    Saves last checkpoint to text file.
    Args:
        path: to file as string
        seq: sequence number as string

    Returns:
        nothing
    """
    with open(path, "w") as f:
        f.write(str(seq))


def load_checkpoint(path: Path) -> int | None:
    """

    Args:
        path:

    Returns:

    """
    if not Path(path).exists():
        print("Checkpoint file does not exist")
        return None
    with open(path, "r") as f:
        content = f.read().strip()
        print("Checkpoint file loaded correctly")
        return int(content) if content else None


def build_uri(cursor=None) -> str:
    """"""
    base = (
        "wss://jetstream.us-east.bsky.network/xrpc/network.bsky.jetstream.subscribeEvents"
        "?collections=app.bsky.feed.post&kinds=commit"
    )
    if cursor is not None:
        base += f"&cursor={cursor}"
    return base


async def listen(opened_file: TextIO, checkpoint: Path) -> None:
    """"""
    last_seq = load_checkpoint(checkpoint)
    count = 0
    start_time = datetime.datetime.now()

    count_on_last_heartbeat = 0
    last_heartbeat_time = datetime.datetime.now()

    try:
        while True:
            uri = build_uri(last_seq)
            try:
                async with websockets.connect(uri, subprotocols=['xrpc.v1.json']) as ws:
                    print("Connected to Jetstream, fetching data...")
                    async for frame in ws:
                        event = json.loads(frame)["payload"]

                        if event["operation"] != "delete":
                            save_data_to_jsonl(opened_file, event)
                            last_seq = event["seq"]

                            count += 1
                            if count % 44000 == 0:
                                save_checkpoint(checkpoint, last_seq)

                            if count % 5000 == 0:
                                time_elapsed = datetime.datetime.now() - last_heartbeat_time
                                new_records = count - count_on_last_heartbeat
                                throughput = new_records / time_elapsed.total_seconds()
                                print(f"New records: {new_records} throughput: {throughput}")
                                count_on_last_heartbeat = count
                                last_heartbeat_time = datetime.datetime.now()

            except websockets.exceptions.ConnectionClosed:
                print("Connection closed, reconnecting")
                await asyncio.sleep(15)
    finally:

        if last_seq is not None:
            save_checkpoint(checkpoint, last_seq)

        print("Data stream closed")
        print(f"New rows: {count}")
        total_time_elapsed = (datetime.datetime.now() - start_time).total_seconds()
        print(f"Time elapsed: {total_time_elapsed}")
        print(f"Average throughput: {count / total_time_elapsed}")


async def main(raw_data_path: Path, checkpoint: Path) -> None:
    print("Starting ingestion...")
    with open(str(raw_data_path), "a") as opened_file:
        await listen(opened_file, checkpoint)
