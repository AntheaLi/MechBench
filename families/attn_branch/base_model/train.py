"""Training loop for the tiny Transformer retrieval model.

Deterministic training with seeded data and model initialization.
Returns metrics dict suitable for the family experiment runner.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

try:
    from families.attn_branch.base_model.data import (
        SPLIT_CONFIGS,
        RetrievalDataset,
        SplitConfig,
        generate_split,
        get_split_config,
    )
    from families.attn_branch.base_model.model import ModelConfig, Transformer
except ModuleNotFoundError:
    from base_model.data import (
        SPLIT_CONFIGS,
        RetrievalDataset,
        SplitConfig,
        generate_split,
        get_split_config,
    )
    from base_model.model import ModelConfig, Transformer


@dataclass
class TrainConfig:
    # Training
    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 0.01
    warmup_steps: int = 50
    grad_clip: float = 1.0

    # Seeding
    train_seed: int = 0
    data_seed: int = 0

    # Data
    train_examples: int = 8192
    seq_len: int = 32

    # Eval
    eval_splits: list[str] = field(default_factory=lambda: ["validation_id"])
    num_pairs: int = 1  # key-value pairs per sequence (>1 for harder retrieval)

    # Logging
    log_every: int = 5  # epochs
    save_dir: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainResult:
    """Holds training outcomes."""
    final_metrics: dict[str, float]
    training_curve: list[dict[str, float]]
    model_stats: dict[str, Any]
    elapsed_seconds: float
    model: Transformer | None = None  # optionally keep reference

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_metrics": self.final_metrics,
            "training_curve": self.training_curve,
            "model_stats": self.model_stats,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def train_model(
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> TrainResult:
    """Train a Transformer and return metrics.

    Fully deterministic given the same seeds and config.
    """
    torch.manual_seed(train_config.train_seed)

    device = torch.device("cpu")
    model = Transformer(model_config).to(device)
    param_count = count_parameters(model)

    # Data
    train_split_cfg = SplitConfig(
        seq_len=train_config.seq_len,
        num_examples=train_config.train_examples,
        num_pairs=train_config.num_pairs,
    )
    train_data = RetrievalDataset.from_split(
        "train",
        data_seed=train_config.data_seed,
        batch_size=train_config.batch_size,
        config=train_split_cfg,
    )

    # Optimizer with linear warmup + cosine decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    total_steps = train_config.epochs * len(train_data)
    scheduler = _make_scheduler(optimizer, train_config.warmup_steps, total_steps)

    loss_fn = nn.CrossEntropyLoss()

    training_curve: list[dict[str, float]] = []
    global_step = 0
    t0 = time.monotonic()

    for epoch in range(train_config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for input_ids, targets in train_data:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            logits = model.predict_last(input_ids)
            loss = loss_fn(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            if train_config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == targets).sum().item()
            epoch_total += input_ids.size(0)
            global_step += 1

        train_acc = epoch_correct / max(epoch_total, 1)
        train_loss = epoch_loss / max(epoch_total, 1)

        if (epoch + 1) % train_config.log_every == 0 or epoch == train_config.epochs - 1:
            eval_metrics = evaluate_model(model, model_config, train_config, device)
            entry = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                **{f"{k}": round(v, 4) for k, v in eval_metrics.items()},
            }
            training_curve.append(entry)

    # Final eval on all requested splits
    elapsed = time.monotonic() - t0
    final_metrics = evaluate_model(model, model_config, train_config, device)
    final_metrics["train_accuracy"] = round(train_acc, 4)
    final_metrics["train_loss"] = round(train_loss, 4)

    # Model stats
    model_stats = {
        "parameter_count": param_count,
        "trainable_parameter_count": param_count,
        "total_parameter_count": count_parameters(model, trainable_only=False),
        "branch_type": model_config.branch_type,
        "branch_alpha": model_config.branch_alpha,
        "branch_norm_scale": model_config.branch_norm_scale,
    }

    # Branch/attention norm ratio if branch exists
    if model_config.branch_type != "none":
        with torch.no_grad():
            sample_input = train_data.input_ids[:8].to(device)
            branch_norm = model.branch_output_norm(sample_input)
            attn_norm = model.base_attention_norm(sample_input)
            model_stats["branch_output_norm"] = round(branch_norm, 4)
            model_stats["base_attention_norm"] = round(attn_norm, 4)
            model_stats["branch_base_norm_ratio"] = (
                round(branch_norm / attn_norm, 4) if attn_norm > 1e-8 else 0.0
            )

    if train_config.save_dir:
        save_path = Path(train_config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path / "model.pt")
        with open(save_path / "metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)
        with open(save_path / "training_curve.json", "w") as f:
            json.dump(training_curve, f, indent=2)

    return TrainResult(
        final_metrics=final_metrics,
        training_curve=training_curve,
        model_stats=model_stats,
        elapsed_seconds=elapsed,
        model=model,
    )


@torch.no_grad()
def evaluate_model(
    model: Transformer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate on all requested splits, returning {split.accuracy: float, split.delta: float}."""
    model.eval()
    metrics: dict[str, float] = {}

    for split_name in train_config.eval_splits:
        try:
            cfg = get_split_config(split_name, num_pairs=train_config.num_pairs)
        except KeyError:
            continue
        inputs, targets = generate_split(split_name, data_seed=train_config.data_seed, config=cfg)
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Process in chunks to avoid memory issues
        correct = 0
        total = 0
        for start in range(0, len(inputs), train_config.batch_size):
            end = min(start + train_config.batch_size, len(inputs))
            logits = model.predict_last(inputs[start:end])
            preds = logits.argmax(dim=-1)
            correct += (preds == targets[start:end]).sum().item()
            total += end - start

        accuracy = correct / max(total, 1)
        metrics[f"{split_name}.accuracy"] = accuracy

    # Compute delta relative to a "baseline" accuracy if available
    if "validation_id.accuracy" in metrics:
        for split_name in train_config.eval_splits:
            key = f"{split_name}.accuracy"
            if key in metrics:
                metrics[f"{split_name}.delta"] = metrics[key]  # absolute, baseline subtracted later

    return metrics


def _make_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay."""
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
