from src.project_blue.training import TokenDataset
from pathlib import Path
from torch.utils.data import DataLoader
from src.project_blue.model import JetGPT
import torch


data_path = Path("../data/tokenized/encoded_corpus.npy")
dataset = TokenDataset(data_path, 128)

dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
model = JetGPT(
    vocab_size=24000,
    seq_len=128,
    embed_dim=384,
    num_heads=8,
    hidden_dim=1536,
    num_layers=8,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for input_batch, target_batch in dataloader:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    output = model(input_batch)

    loss_fn = torch.nn.CrossEntropyLoss()
    output_flat = output.view(-1, output.shape[-1])
    target_flat = target_batch.view(-1)
    loss = loss_fn(output_flat, target_flat)
    print(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
