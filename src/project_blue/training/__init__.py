import torch
import numpy as np
from pathlib import Path

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: Path, seq_len: int):
        self.tokens_table = np.load(data_path)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens_table) // (self.seq_len + 1)

    def __getitem__(self, index):
        start = index * (self.seq_len + 1)
        chunk = self.tokens_table[start:start + self.seq_len + 1]
        input_data = chunk[:-1]
        target_data = chunk[1:]
        return torch.LongTensor(input_data), torch.LongTensor(target_data)