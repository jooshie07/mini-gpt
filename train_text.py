import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# -----------------------
# LOAD BPE VOCAB
# -----------------------
with open("bpe_vocab.json", "r", encoding="utf-8") as f:
    vocab_data = json.load(f)

stoi = vocab_data["stoi"]
itos = {int(k): v for k, v in vocab_data["itos"].items()}

vocab_size = len(stoi)

# -----------------------
# LOAD DATA
# -----------------------
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = list(text)
data = torch.tensor([stoi[c] for c in tokens], dtype=torch.long)

# -----------------------
# HYPERPARAMETERS
# -----------------------
block_size = 16
embed_dim = 64
batch_size = 32
lr = 3e-4

# -----------------------
# BATCH
# -----------------------
def get_batch():
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# -----------------------
# MODEL
# -----------------------
token_embed = nn.Embedding(vocab_size, embed_dim)
position_embed = nn.Embedding(block_size, embed_dim)

encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=4,
    batch_first=True
)

transformer = nn.TransformerEncoder(
    encoder_layer,
    num_layers=3
)

lm_head = nn.Linear(embed_dim, vocab_size)

optimizer = torch.optim.Adam(
    list(token_embed.parameters()) +
    list(position_embed.parameters()) +
    list(transformer.parameters()) +
    list(lm_head.parameters()),
    lr=lr
)

# -----------------------
# TRAINING
# -----------------------
for step in range(3000):

    x, y = get_batch()
    positions = torch.arange(block_size)

    x_embed = token_embed(x) + position_embed(positions)

    seq_len = x.shape[1]
    mask = torch.triu(
        torch.ones(seq_len, seq_len) * float('-inf'),
        diagonal=1
    )

    x_transformed = transformer(x_embed, mask=mask)
    logits = lm_head(x_transformed)

    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print("Step:", step, "Loss:", loss.item())

# -----------------------
# SAVE MODEL
# -----------------------
torch.save({
    "token_embed": token_embed.state_dict(),
    "position_embed": position_embed.state_dict(),
    "transformer": transformer.state_dict(),
    "lm_head": lm_head.state_dict()
}, "mini_gpt_model.pth")

print("Model saved!")