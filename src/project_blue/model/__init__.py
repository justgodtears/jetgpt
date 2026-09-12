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

# It was only used to help me learn the Single-Head architecture
# all new GPTs use Multi-Head, so this class is not used
class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        scaled = attention_scores / (self.embed_dim ** 0.5)
        softmax = torch.softmax(scaled, dim=-1)
        weighted_sum = torch.matmul(softmax, V)
        return weighted_sum


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
