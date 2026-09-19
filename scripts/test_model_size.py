from pathlib import Path
from src.project_blue.model import *

model = JetGPT(
    vocab_size=24000,
    seq_len=128,
    embed_dim=384,
    num_heads=8,
    hidden_dim=1536,
    num_layers=8,
) #32700864


print(count_parameters(model))