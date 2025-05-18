import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3 x (B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Cálculo de atención scaled dot-product con máscara causal
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)

        out = att @ v  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        out = self.out_proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = SelfAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPTSimple(nn.Module):
    def __init__(self, vocab_size, max_seq_len=128, embed_dim=128, num_heads=4, num_layers=4, ff_dim=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.max_seq_len, "Secuencia muy larga"

        tok_emb = self.token_emb(idx)              # (B, T, C)
        pos = torch.arange(T, device=idx.device)  # (T)
        pos_emb = self.pos_emb(pos)                # (T, C)
        x = tok_emb + pos_emb                      # (B, T, C)

        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)                       # (B, T, vocab_size)
        return logits
