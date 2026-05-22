import torch
import torch.nn as nn

# toy input: 3 tokens, embedding size 4
x = torch.randn(3, 4)

# attention layer
attention = nn.MultiheadAttention(
    embed_dim=4,
    num_heads=1,
    batch_first=True
)

# feed-forward network
ffn = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 4)
)

# layer normalization
norm1 = nn.LayerNorm(4)
norm2 = nn.LayerNorm(4)

# batch dimension
x = x.unsqueeze(0)

# self-attention
attn_output, _ = attention(x, x, x)

# residual + norm
x = norm1(x + attn_output)

# feed forward
ffn_output = ffn(x)

# residual + norm
x = norm2(x + ffn_output)

print("Transformer Block Output:")
print(x)