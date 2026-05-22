import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# -----------------------
# LOAD VOCAB
# -----------------------
with open("bpe_vocab.json", "r", encoding="utf-8") as f:
    vocab_data = json.load(f)

stoi = vocab_data["stoi"]
itos = {int(k): v for k, v in vocab_data["itos"].items()}

vocab_size = len(stoi)

# -----------------------
# MODEL STRUCTURE
# -----------------------
block_size = 16
embed_dim = 64

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

# -----------------------
# LOAD MODEL
# -----------------------
checkpoint = torch.load("mini_gpt_model.pth")

token_embed.load_state_dict(checkpoint["token_embed"])
position_embed.load_state_dict(checkpoint["position_embed"])
transformer.load_state_dict(checkpoint["transformer"])
lm_head.load_state_dict(checkpoint["lm_head"])

print("Model loaded!")

# -----------------------
# GENERATION
# -----------------------
with torch.no_grad():

    start = "hello"

    context = torch.tensor([[stoi.get(c, 0) for c in start]], dtype=torch.long)

    print("Start:", start)

    for _ in range(20):

        positions = torch.arange(context.shape[1])

        x_embed = token_embed(context) + position_embed(positions)

        seq_len = context.shape[1]
        mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'),
            diagonal=1
        )

        x_transformed = transformer(x_embed, mask=mask)
        logits = lm_head(x_transformed)

        temperature = 0.7

        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)

        next_token = torch.multinomial(probs, 1)

        context = torch.cat([context, next_token], dim=1)
        context = context[:, -block_size:]

    generated = "".join([itos[i.item()] for i in context[0]])

    print("Generated:", generated)