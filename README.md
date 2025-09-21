# 🤖 GPT-from-scratch

Character-level GPT built from scratch (in PyTorch), inspired by Andrej Karpathy’s nanoGPT lecture series. Includes a minimal bigram baseline and a multi-head Transformer with positional embeddings, causal self-attention, and checkpointing.

## 🌟 Highlights

- Trains on Tiny Shakespeare to generate Shakespeare-like text!
- Simple training loop with periodic eval and checkpoint saving
- Text generation with context windowing (`block_size`)

## 📦 Repository Structure

```
GPT-from-scratch/
├─ data/                   # dataset folder (tiny Shakespeare)
├─ scripts/
│  ├─ bigram.py            # bigram baseline
│  └─ v2.py                # Transformer language model
├─ notebooks/              # exploration notebooks
├─ environment.yaml        # conda environment (CPU by default)
├─ checkpoints/            # saved checkpoints (created at runtime)
└─ README.md
```

## ⬇️ Setup

Using conda (recommended):

```bash
conda env create -f environment.yaml
conda activate gpt
```

Or with pip (Python 3.11):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch==2.2.2 wget
```

Dataset:

- `scripts/v2.py` will auto-download Tiny Shakespeare to `data/input.txt` if missing.
- `scripts/bigram.py` expects `data/input.txt` to exist (place your own text file if desired).

## 🚀 Usage

Run the bigram baseline:

```bash
python scripts/bigram.py
```

Run the Transformer (multi-head attention) model:

```bash
python scripts/v2.py
```

What it does:

- Trains for `max_iters` steps, evaluating every `eval_interval`.
- Saves checkpoints to `checkpoints/model_<timestamp>.pt`.
- Generates sample text at the end (500 tokens by default).

Notes:

- Hyperparameters are defined at the top of each script (no CLI flags). Key ones: `block_size`, `embedding_size`, `num_heads`, `num_layers`, `dropout`, `learning_rate`, `batch_size`.
- Generation respects the context window: sequences are cropped to the last `block_size` tokens to keep positional embeddings and attention shapes valid.

## 🧠 Model Overview (v2)

- Token embeddings: `nn.Embedding(vocab_size, embedding_size)`
- Positional embeddings: `nn.Embedding(block_size, embedding_size)`
- Self-attention: multi-head (`num_heads`), each head uses Q/K/V projections and a causal mask
- MLP: 2-layer feed-forward with ReLU and Dropout
- Residual connections + LayerNorm per block
- Output head: `nn.Linear(embedding_size, vocab_size)`

Checkpoints are loaded automatically at start (latest in `checkpoints/`) if present.

## 📄 License

This repo re-implements concepts from public lectures for educational purposes. Add your preferred license if you plan to distribute.

## 💬 Contributing

Issues and PRs are welcome. Ideas: CLI arg support, more datasets, better sampling, configurable checkpoints, or tests.
