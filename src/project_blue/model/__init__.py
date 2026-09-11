import torch
from torch import nn

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, embed_dim: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)

    def forward(self, token_ids):
        token_embeddings = self.token_embedding(token_ids)
        position_tensor = torch.arange(token_ids.shape[1])
        position_embeddings = self.position_embedding(position_tensor)
        return token_embeddings + position_embeddings