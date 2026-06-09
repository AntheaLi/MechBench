"""Tiny Transformer with configurable attention branch for retrieval tasks.

Architecture (~500K params at d_model=64, n_layers=2, n_heads=2):
  - Token embedding + learned positional embedding
  - N Transformer layers with causal self-attention
  - Optional StructuredRoutingBranch on each layer
  - Linear head predicting the next token

The branch is the "claimed mechanism" in the benchmark. Its structure and
type are controlled by the variant configuration, enabling different causal
worlds (true mechanism, parameter laundering, norm laundering, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ModelConfig:
    vocab_size: int = 100
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 256
    max_seq_len: int = 64
    dropout: float = 0.0

    # Branch configuration
    branch_type: str = "none"
    # "none":      baseline, no branch
    # "geometric": structured routing relation (the claimed mechanism)
    # "generic":   generic learned MLP branch (same param count)
    # "random":    fixed random projection branch (same param count)
    # "destroyed": geometric branch with randomly rotated projections

    branch_alpha: float = 0.3        # residual weight of branch output
    branch_d_inner: int = 32         # inner dimension of the branch
    branch_norm_scale: float = 1.0   # multiplicative scale on branch output norm
    branch_init_scale: float = 1.0   # initialization scale

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class StructuredRoutingBranch(nn.Module):
    """The 'geometric attention branch' — the claimed mechanism.

    Computes a structured routing relation between query and key
    representations, producing an additive correction to attention output.

    Architecture:
        q_proj(x) -> Q_branch  (d_model -> d_inner)
        k_proj(x) -> K_branch  (d_model -> d_inner)
        routing_score = Q_branch @ K_branch^T / sqrt(d_inner)
        v_proj(x) -> V_branch  (d_model -> d_inner)
        out = softmax(routing_score) @ V_branch
        output = out_proj(out)  (d_inner -> d_model)

    This is essentially a small parallel attention head with its own
    learned projections. The "geometric structure" is the learned Q/K
    alignment that specializes for key-value retrieval.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.d_model
        d_inner = config.branch_d_inner

        self.q_proj = nn.Linear(d, d_inner, bias=False)
        self.k_proj = nn.Linear(d, d_inner, bias=False)
        self.v_proj = nn.Linear(d, d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d, bias=False)
        self.scale = 1.0 / math.sqrt(d_inner)
        self.alpha = config.branch_alpha
        self.norm_scale = config.branch_norm_scale

        self._init_weights(config)

    def _init_weights(self, config: ModelConfig):
        s = config.branch_init_scale
        for p in [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight]:
            nn.init.normal_(p, std=0.02 * s)
        nn.init.normal_(self.out_proj.weight, std=0.02 * s / math.sqrt(2))

    def forward(self, x: Tensor, causal_mask: Tensor | None = None) -> Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = self.out_proj(out)
        return out * self.alpha * self.norm_scale


class GenericLearnedBranch(nn.Module):
    """A generic MLP branch with the same parameter count as the geometric branch.

    No key-value routing structure — just feedforward capacity.
    Used for parameter_laundering worlds where generic capacity drives the gain.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.d_model
        d_inner = config.branch_d_inner

        # Match param count: 4 * d * d_inner (same as 4 projection matrices)
        self.net = nn.Sequential(
            nn.Linear(d, d_inner * 2, bias=False),
            nn.GELU(),
            nn.Linear(d_inner * 2, d, bias=False),
        )
        self.alpha = config.branch_alpha
        self.norm_scale = config.branch_norm_scale

        s = config.branch_init_scale
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02 * s)

    def forward(self, x: Tensor, causal_mask: Tensor | None = None) -> Tensor:
        return self.net(x) * self.alpha * self.norm_scale


class RandomProjectionBranch(nn.Module):
    """Fixed random projection branch — same architecture as geometric but frozen.

    Used to test whether learned structure matters vs just having extra parameters.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.d_model
        d_inner = config.branch_d_inner

        # Register as buffers — they don't get gradients
        self.register_buffer("q_weight", torch.randn(d_inner, d) * 0.02)
        self.register_buffer("k_weight", torch.randn(d_inner, d) * 0.02)
        self.register_buffer("v_weight", torch.randn(d_inner, d) * 0.02)
        self.register_buffer("out_weight", torch.randn(d, d_inner) * 0.02)
        self.scale = 1.0 / math.sqrt(d_inner)
        self.alpha = config.branch_alpha
        self.norm_scale = config.branch_norm_scale

    def forward(self, x: Tensor, causal_mask: Tensor | None = None) -> Tensor:
        Q = F.linear(x, self.q_weight)
        K = F.linear(x, self.k_weight)
        V = F.linear(x, self.v_weight)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = F.linear(out, self.out_weight)
        return out * self.alpha * self.norm_scale


class DestroyedGeometryBranch(StructuredRoutingBranch):
    """Geometric branch with randomly rotated Q/K projections.

    Preserves parameter count, trainability, and output norm, but
    destroys the structured routing relation. Used to test whether
    the specific geometry matters.
    """

    def _init_weights(self, config: ModelConfig):
        super()._init_weights(config)
        # Apply random orthogonal rotation to Q and K projections
        # This preserves norms but destroys alignment
        for proj in [self.q_proj, self.k_proj]:
            with torch.no_grad():
                d_inner, _ = proj.weight.shape
                Q, _ = torch.linalg.qr(torch.randn(d_inner, d_inner))
                proj.weight.copy_(Q @ proj.weight)


def _make_branch(config: ModelConfig) -> nn.Module | None:
    """Factory for creating the appropriate branch type."""
    if config.branch_type == "none":
        return None
    if config.branch_type == "geometric":
        return StructuredRoutingBranch(config)
    if config.branch_type == "generic":
        return GenericLearnedBranch(config)
    if config.branch_type == "random":
        return RandomProjectionBranch(config)
    if config.branch_type == "destroyed":
        return DestroyedGeometryBranch(config)
    raise ValueError(f"unknown branch_type: {config.branch_type}")


class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig, branch: nn.Module | None = None):
        super().__init__()
        d = config.d_model

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            d, config.n_heads, dropout=config.dropout, batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, d),
        )
        self.ln2 = nn.LayerNorm(d)

        # Optional branch
        self.branch = branch

    def forward(self, x: Tensor, causal_mask: Tensor | None = None, attn_mask: Tensor | None = None) -> Tensor:
        # Self-attention
        residual = x
        x = self.ln1(x)
        x_attn, _ = self.attn(x, x, x, attn_mask=attn_mask, is_causal=False)
        x = residual + x_attn

        # Branch (additive correction to attention output)
        if self.branch is not None:
            branch_out = self.branch(self.ln1(residual), causal_mask)
            x = x + branch_out

        # Feedforward
        residual = x
        x = self.ln2(x)
        x = residual + self.ff(x)

        return x


class Transformer(nn.Module):
    """Tiny Transformer for synthetic retrieval tasks."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        branch = _make_branch(config)
        self.layers = nn.ModuleList([
            TransformerLayer(
                config,
                branch=branch if i == config.n_layers - 1 else None,
            )
            for i in range(config.n_layers)
        ])

        self.ln_final = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and not hasattr(module, "_branch_init"):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass. Returns logits (batch, seq_len, vocab_size)."""
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))

        # Causal mask
        causal = torch.tril(torch.ones(T, T, device=input_ids.device)).bool()
        # For nn.MultiheadAttention: float mask where True positions are masked
        attn_mask = torch.zeros(T, T, device=input_ids.device)
        attn_mask.masked_fill_(~causal, float("-inf"))

        for layer in self.layers:
            x = layer(x, causal_mask=causal, attn_mask=attn_mask)

        x = self.ln_final(x)
        return self.head(x)

    def predict_last(self, input_ids: Tensor) -> Tensor:
        """Return logits at the last position only. (batch, vocab_size)"""
        logits = self.forward(input_ids)
        return logits[:, -1, :]

    def branch_output_norm(self, input_ids: Tensor) -> float:
        """Compute mean L2 norm of branch output (for norm instrumentation)."""
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))
        causal = torch.tril(torch.ones(T, T, device=input_ids.device)).bool()
        attn_mask = torch.zeros(T, T, device=input_ids.device)
        attn_mask.masked_fill_(~causal, float("-inf"))

        for layer in self.layers:
            if layer.branch is not None:
                h = layer.ln1(x)
                branch_out = layer.branch(h, causal)
                return branch_out.norm(dim=-1).mean().item()
            residual = x
            x = layer.ln1(x)
            x_attn, _ = layer.attn(x, x, x, attn_mask=attn_mask, is_causal=False)
            x = residual + x_attn
            residual = x
            x = layer.ln2(x)
            x = residual + layer.ff(x)
        return 0.0

    def base_attention_norm(self, input_ids: Tensor) -> float:
        """Compute mean L2 norm of base attention output (for norm ratio)."""
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))
        attn_mask = torch.zeros(T, T, device=input_ids.device)
        causal = torch.tril(torch.ones(T, T, device=input_ids.device)).bool()
        attn_mask.masked_fill_(~causal, float("-inf"))

        for layer in self.layers:
            h = layer.ln1(x)
            x_attn, _ = layer.attn(h, h, h, attn_mask=attn_mask, is_causal=False)
            return x_attn.norm(dim=-1).mean().item()
        return 0.0
