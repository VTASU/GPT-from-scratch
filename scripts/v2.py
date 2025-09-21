import torch
import torch.nn as nn
from torch.nn import functional as F
import os


# Hyperparameters
save_dir = "checkpoints"
torch.manual_seed(1337)
batch_size = 64  # how many INDEPENDENT sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
embedding_size = 384
train_val_split = 0.9
eval_iters = 200
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
num_heads = 6
num_layers = 6
dropout = 0.2


def load_latest_checkpoint(model, optimizer=None, device="cpu"):
    pt_files = [f for f in os.listdir(save_dir) if f.endswith(".pt")]
    if not pt_files:
        # raise FileNotFoundError(f"No checkpoints found in {save_dir}")
        return model

    def get_step(fname):
        parts = fname.split("_")
        step_str = parts[-1].split(".")[0]
        return int(step_str)

    latest = max(pt_files, key=get_step)
    path = os.path.join(save_dir, latest)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt if isinstance(ckpt, dict) else ckpt["model"])

    if optimizer and isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    print(f"Loaded checkpoint: {latest}")
    return model


# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Raw Data loading
with open("../notebooks/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)


# Simple Word Embeddings with encoding and decoding
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])


# Data Preprocessing
data = torch.tensor(encode(text), dtype=torch.long)
train_size = int(len(data) * train_val_split)
train_data = data[:train_size]
val_data = data[train_size:]


# Helper functions
def get_batch(split: str):
    data: torch.Tensor = train_data if split == "train" else val_data
    ix: torch.Tensor = torch.randint(len(data) - block_size, (batch_size,))
    x: torch.Tensor = torch.stack([data[i : i + block_size] for i in ix])
    y: torch.Tensor = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def evaluate_loss():
    out = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


# Model
class Head(nn.Module):
    def __init__(self, head_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Key = nn.Linear(embedding_size, head_size, bias=False)
        self.Query = nn.Linear(embedding_size, head_size, bias=False)
        self.Value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones((block_size, block_size), device=device))
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k: torch.Tensor = self.Key(x)  # (B,T,head_size)
        q: torch.Tensor = self.Query(x)  # (B,T,head_size)
        v: torch.Tensor = self.Value(x)  # (B,T,head_size)
        wei: torch.Tensor = q @ k.transpose(-2, -1) * (C**-0.5)  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)
        wei = self.do(wei)
        return wei @ v  # (B,T,head_size)


class Multihead(nn.Module):
    def __init__(self, num_heads, head_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.do(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, embedding_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embedding_size, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        head_size = embedding_size // num_heads
        self.multihead = Multihead(num_heads, head_size)
        self.ffwd = FeedFoward(embedding_size)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x + self.multihead(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(block_size, embedding_size)
        self.blocks = nn.Sequential(
            *[Block(embedding_size, num_heads) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,embedding_size)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T,embedding_size)
        x = tok_emb + pos_emb  # (broadcast) (B,T,embedding_size)
        x = self.blocks(x)
        x = self.ln(x)
        # x = self.ffwd(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            d = idx[:, -block_size:]
            logits, loss = self(d)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Model Initialization
m = BigramLanguageModel()
m = m.to(device)


# Optimizer Initialization
optm = torch.optim.AdamW(m.parameters(), lr=learning_rate)
m = load_latest_checkpoint(m, optimizer=optm, device=device)


# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = evaluate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        torch.save(m.state_dict(), f"{save_dir}/model_step_{iter}.pt")

    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optm.zero_grad(set_to_none=True)
    loss.backward()
    optm.step()


# Output generation
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
