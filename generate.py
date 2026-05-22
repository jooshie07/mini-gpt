import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# IMPORT YOUR MODEL ARCHITECTURE
# -----------------------
from mini_gpt import token_embed, position_embed, transformer, lm_head, block_size, stoi, itos

# -----------------------
# LOAD MODEL WEIGHTS
# -----------------------
checkpoint = torch.load("mini_gpt_model.pth")

token_embed.load_state_dict(checkpoint["token_embed"])
position_embed.load_state_dict(checkpoint["position_embed"])
transformer.load_state_dict(checkpoint["transformer"])
lm_head.load_state_dict(checkpoint["lm_head"])

print("Model loaded!")

# -----------------------
# GENERATION FUNCTION
# -----------------------
def generate(start_text, max_new_tokens=50):

    context = torch.tensor(
        [[stoi.get(c, 0) for c in start_text]],
        dtype=torch.long
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            seq_len = context.shape[1]
            positions = torch.arange(seq_len)

            x_embed = token_embed(context) + position_embed(positions)

            mask = torch.triu(
                torch.ones(seq_len, seq_len) * float("-inf"),
                diagonal=1
            )

            x = transformer(x_embed, mask=mask)
            logits = lm_head(x)

            probs = F.softmax(logits[:, -1, :] / 0.7, dim=-1)

            next_token = torch.multinomial(probs, 1)

            context = torch.cat([context, next_token], dim=1)
            context = context[:, -block_size:]

    return "".join([itos[i.item()] for i in context[0]])

# -----------------------
# TEST
# -----------------------
start = "hell"
print("\nStart:", start)

output = generate(start, 80)

print("\nGenerated:", output)