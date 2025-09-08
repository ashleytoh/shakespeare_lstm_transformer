import os, json, time, math, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")  # headless plots
import matplotlib.pyplot as plt

# ---------------- Reproducibility ----------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- Data / Vocab ----------------
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_vocab_char(text: str):
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text: str, stoi: dict) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)

def decode(ids: torch.Tensor, itos: dict) -> str:
    return "".join(itos[int(i)] for i in ids)

def split_indices(n_tokens: int, train=0.8, val=0.1):
    n_train = int(n_tokens * train)
    n_val = int(n_tokens * val)
    n_test = n_tokens - n_train - n_val
    train_idx = (0, n_train)
    val_idx   = (n_train, n_train + n_val)
    test_idx  = (n_train + n_val, n_tokens)
    return train_idx, val_idx, test_idx

class CharDataset(Dataset):
    """Returns (x, y) where y is x shifted by 1 token."""
    def __init__(self, ids: torch.Tensor, seq_len: int):
        self.ids = ids
        self.seq_len = seq_len
    def __len__(self):
        return len(self.ids) - self.seq_len - 1
    def __getitem__(self, i):
        x = self.ids[i : i + self.seq_len]
        y = self.ids[i + 1 : i + 1 + self.seq_len]
        return x, y

def make_loader(ids, seq_len, batch_size, shuffle=True, device="cpu"):
    ds = CharDataset(ids, seq_len)
    # simple collate -> move to device inside training loop for clarity
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=True, pin_memory=False)

# ---------------- Logging / Plots ----------------
def plot_curves(train_losses, val_losses, out_png: str, title: str):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("cross-entropy"); plt.title(title); plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def save_ckpt(path, model, stoi, itos, args_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "args": args_dict,
    }, path)

def load_ckpt(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)
