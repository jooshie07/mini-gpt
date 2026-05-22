from collections import defaultdict
import json

# -----------------------
# LOAD TEXT
# -----------------------
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = list(text)

# -----------------------
# BPE FUNCTIONS
# -----------------------
def get_stats(tokens):
    pairs = defaultdict(int)
    for i in range(len(tokens) - 1):
        pairs[(tokens[i], tokens[i + 1])] += 1
    return pairs

def merge(tokens, pair):
    new_tokens = []
    i = 0

    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            new_tokens.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1

    return new_tokens

# -----------------------
# TRAIN BPE
# -----------------------
num_merges = 100

for i in range(num_merges):
    stats = get_stats(tokens)
    if not stats:
        break

    best_pair = max(stats, key=stats.get)
    tokens = merge(tokens, best_pair)

# -----------------------
# BUILD VOCAB
# -----------------------
vocab = sorted(list(set(tokens)))

stoi = {tok: i for i, tok in enumerate(vocab)}
itos = {i: tok for i, tok in enumerate(vocab)}

# save vocab
with open("bpe_vocab.json", "w", encoding="utf-8") as f:
    json.dump({"stoi": stoi, "itos": itos}, f)

print("BPE done!")
print("Vocab size:", len(vocab))