# 🚀 Mini GPT (Transformer from Scratch)

A minimal GPT-style language model built from scratch using PyTorch.  
This project demonstrates how transformer-based language models work internally.

---

# 🧠 What This Project Does

This project trains a character-level Transformer model that learns patterns in text and generates new sequences based on input prompts.

It includes:
- Training pipeline
- Transformer architecture
- Token embedding + positional encoding
- Text generation using sampling

---

# ⚙️ Model Architecture

- Embedding Layer (Token + Position)
- Transformer Encoder (multi-layer)
- Linear output head
- Causal masking for autoregressive generation

---

# 📂 Project Structure

---

# 📊 Training Details

- Dataset: Custom text file (`data.txt`)
- Tokenization: Character-level (or BPE if enabled)
- Loss Function: Cross Entropy
- Optimizer: Adam
- Training Steps: ~1000–3000 iterations

---

# 🚀 How to Run

## 1. Install dependencies

---

## 2. Train the model

---

## 3. Generate text

---

# 💡 Example Output
Start: hell
Generated: hello world this is a simple transformer model


# 🧠 Key Learnings

While building this Mini GPT, I gained a deeper understanding of how modern language models work internally.

## 🧩 1. How Transformers actually work
- Learned how self-attention helps models focus on relevant tokens
- Understood why transformers replaced RNNs for language tasks
- Saw how stacked layers gradually improve representations

## 🔤 2. Tokenization is the foundation
- Understood how text is converted into numbers
- Learned difference between character-level and subword (BPE) tokenization
- Realized how vocabulary design affects model quality

## ⚙️ 3. Training language models
- Learned how next-token prediction works
- Understood loss functions like Cross Entropy in sequence learning
- Saw how gradient descent updates model behavior over time

## 🎯 4. Autoregressive generation
- Learned how models generate text one token at a time
- Understood why context window (block size) matters
- Observed repetition issues when model is small or undertrained

## 🧠 5. Importance of data quality
- Realized that model performance depends heavily on dataset size and diversity
- Small datasets lead to repetition and weak generalization

## 🔥 6. Sampling strategies matter
- Learned difference between greedy vs probabilistic sampling
- Used techniques like temperature to control randomness

## 🧱 7. End-to-end ML pipeline understanding
- Built full workflow: data → training → saving → loading → inference
- Understood how real LLM systems are structured in production

## 🚀 8. Real insight
- Even a small transformer can learn patterns surprisingly well
- Most “intelligence” comes from data + architecture, not just code

# ⚠️ Challenges Faced

## 1. Model repetition issues
The model initially produced repetitive outputs due to:
- small dataset size
- lack of advanced sampling techniques

## 2. Tokenization limitations
Character-level encoding limited semantic understanding of words.

## 3. Training instability
Loss fluctuations were observed due to:
- small batch size
- limited data diversity

## 4. Debugging transformer shapes
Ensuring correct tensor shapes between embeddings and transformer layers was critical.


# ⚡ Limitations

## 1. Small dataset
The model is trained on a very small custom dataset, which limits its ability to generalize or produce diverse outputs.

## 2. Character-level tokenization
The model operates on characters instead of subwords (like BPE), which:
- reduces semantic understanding
- makes learning slower
- increases sequence length

## 3. Limited model size
The transformer used is lightweight:
- fewer layers
- smaller embedding dimensions  
This limits its ability to learn complex patterns.

## 4. No long-term memory
The model only uses a fixed context window (block size), meaning:
- it cannot remember long conversations
- older tokens are discarded during generation

## 5. Basic sampling strategy
Text generation uses simple sampling methods (e.g., temperature + multinomial):
- no top-k or nucleus sampling
- can lead to repetition or randomness

## 6. No large-scale training optimization
The model is not trained with:
- distributed training
- learning rate scheduling
- large-scale datasets

## 7. Not production-ready
This model is for educational purposes only and is not suitable for real-world deployment or deployment-scale inference.