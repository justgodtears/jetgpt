from pathlib import Path
from src.project_blue.model import *

# model = JetGPT(
#     vocab_size=24000,
#     seq_len=128,
#     embed_dim=256,
#     num_heads=8,
#     hidden_dim=1024,
#     num_layers=4,
# ) #15503808 Parameters

model = JetGPT(
    vocab_size=24000,
    seq_len=128,
    embed_dim=256,
    num_heads=8,
    hidden_dim=1024,
    num_layers=8,
) #18662848 Parameters


print(count_parameters(model))