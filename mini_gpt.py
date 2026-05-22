import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# LOAD DATA
# -----------------------
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

chars = sorted(list(set(text)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
vocab_size = len(chars)

print("Vocab size:", vocab_size)

# -----------------------
# HYPERPARAMETERS
# -----------------------
block_size = 64
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

    seq_len = x.shape[1]

    positions = torch.arange(seq_len)

    x_embed = token_embed(x) + position_embed(positions)

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
# CHAT GENERATION
# -----------------------
conversation = []

def generate(prompt, max_new_tokens=100):

    context_text = "\n".join(conversation + [prompt])

    context = torch.tensor(
        [[stoi.get(c, 0) for c in context_text]],
        dtype=torch.long
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            seq_len = context.shape[1]
            positions = torch.arange(seq_len)

            x_embed = token_embed(context) + position_embed(positions)

            mask = torch.triu(
                torch.ones(seq_len, seq_len) * float('-inf'),
                diagonal=1
            )

            x_transformed = transformer(x_embed, mask=mask)
            logits = lm_head(x_transformed)

            probs = F.softmax(logits[:, -1, :] / 0.7, dim=-1)

            next_token = torch.multinomial(probs, 1)

            context = torch.cat([context, next_token], dim=1)
            context = context[:, -block_size:]

    return "".join([itos[i.item()] for i in context[0]])

# -----------------------
# CHAT LOOP
# -----------------------
print("\n--- CHAT MODE ---")

while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = f"user:{user_input}\nassistant:"
    output = generate(prompt)

    print("\nAI:", output)