import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from config import DensyConfig


def precompute_rope(head_dim: int, seq_len: int, theta: float, device: torch.device):
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = torch.outer(pos, freqs)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, T, D = x.shape
    x1, x2 = x[..., ::2], x[..., 1::2]
    c = cos[:T].unsqueeze(0).unsqueeze(0).to(x.dtype)
    s = sin[:T].unsqueeze(0).unsqueeze(0).to(x.dtype)
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class MHA(nn.Module):
    def __init__(self, cfg: DensyConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask=None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn_drop = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=attn_drop, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, cfg: DensyConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.mlp_dim, bias=False)
        self.fc2 = nn.Linear(cfg.mlp_dim, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: DensyConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MHA(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask=None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, mask)
        x = x + self.mlp(self.ln2(x))
        return x


class DensyModel(nn.Module):
    def __init__(self, cfg: DensyConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.weight_tie:
            self.lm_head.weight = self.embed.weight
        self.rope_cos = None
        self.rope_sin = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_rope(self, T: int, device: torch.device):
        if self.rope_cos is None or self.rope_cos.shape[0] < T:
            cos, sin = precompute_rope(self.cfg.head_dim, max(T, self.cfg.seq_len), self.cfg.rope_theta, device)
            self.rope_cos = cos
            self.rope_sin = sin
        return self.rope_cos[:T], self.rope_sin[:T]

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        cos, sin = self._get_rope(T, input_ids.device)
        x = self.drop(self.embed(input_ids))
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.cfg.vocab_size), shift_labels.view(-1), ignore_index=-100)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128, temperature: float = 0.8,
                 top_k: int = 50, top_p: float = 0.9, repetition_penalty: float = 1.1,
                 eos_token_id: int = 2) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.cfg.seq_len:]
            logits = self.forward(ctx)["logits"][:, -1, :].float()
            if repetition_penalty != 1.0:
                for b in range(input_ids.shape[0]):
                    for tok in input_ids[b].unique():
                        logits[b, tok] /= repetition_penalty
            if temperature != 1.0:
                logits /= temperature
            if top_k > 0:
                topk_vals = logits.topk(top_k, dim=-1).values[:, -1:]
                logits[logits < topk_vals] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_logits[(cum_probs - sorted_logits.softmax(dim=-1)) > top_p] = float("-inf")
                logits.scatter_(-1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            if (next_tok == eos_token_id).all():
                break
        return input_ids


def build_densy_330m(cfg: Optional[DensyConfig] = None) -> DensyModel:
    if cfg is None:
        cfg = DensyConfig()
    m = DensyModel(cfg)
    total_p = sum(p.numel() for p in m.parameters())
    print(f"DensyModel: {total_p/1e9:.3f}B total params | d={cfg.d_model}, L={cfg.n_layers}, H={cfg.n_heads} | weight_tied={cfg.weight_tie}")
    return m


build_densy = build_densy_330m
build_densy_500m = build_densy_330m
