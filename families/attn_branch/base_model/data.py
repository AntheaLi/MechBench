"""Synthetic retrieval dataset for the attn_branch family.

Generates passkey-retrieval sequences where the model must recall a target
token that was placed at a specific position amid distractors.

Sequence layout::

    [DISTRACTOR ...] KEY_TOKEN VALUE_TOKEN [DISTRACTOR ...] QUERY_TOKEN → VALUE_TOKEN

The query token is always the last input token; the model must predict the
value token that was paired with the matching key earlier in the sequence.

Supports controlled OOD shifts:
  - validation_id:              same distribution as training
  - validation_long:            longer sequences
  - validation_position_shift:  key placed in different position range
  - validation_distractor_shift: different distractor distribution
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterator

import torch
from torch import Tensor


# Reserve tokens 0-9 for structure, 10+ for content.
PAD_TOKEN = 0
QUERY_TOKEN_BASE = 1   # query tokens: 1..NUM_KEYS
KEY_TOKEN_BASE = 10     # key tokens: 10..10+NUM_KEYS-1
VALUE_TOKEN_BASE = 30   # value tokens: 30..30+NUM_VALUES-1
DISTRACTOR_BASE = 60    # distractor tokens: 60..60+NUM_DISTRACTORS-1

NUM_KEYS = 8
NUM_VALUES = 16
NUM_DISTRACTORS = 40
VOCAB_SIZE = DISTRACTOR_BASE + NUM_DISTRACTORS  # 100


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for a single data split."""
    seq_len: int = 32
    key_position_range: tuple[int, int] = (2, 20)
    distractor_range: tuple[int, int] = (0, NUM_DISTRACTORS)
    num_examples: int = 2048
    num_pairs: int = 1  # key-value pairs per sequence


# Default splits — 1 pair per sequence (original difficulty).
# World configs can set training.num_pairs to override for harder variants.
SPLIT_CONFIGS: dict[str, SplitConfig] = {
    "train": SplitConfig(
        seq_len=32,
        key_position_range=(2, 22),
        num_examples=8192,
        num_pairs=1,
    ),
    "validation_id": SplitConfig(
        seq_len=32,
        key_position_range=(2, 22),
        num_examples=1024,
        num_pairs=1,
    ),
    "validation_long": SplitConfig(
        seq_len=48,
        key_position_range=(2, 36),
        num_examples=512,
        num_pairs=1,
    ),
    "validation_position_shift": SplitConfig(
        seq_len=32,
        key_position_range=(16, 28),
        num_examples=512,
        num_pairs=1,
    ),
    "validation_distractor_shift": SplitConfig(
        seq_len=32,
        key_position_range=(2, 22),
        distractor_range=(NUM_DISTRACTORS // 2, NUM_DISTRACTORS),
        num_examples=512,
        num_pairs=1,
    ),
}


def get_split_config(split_name: str, *, num_pairs: int | None = None) -> SplitConfig:
    """Return a SplitConfig, optionally overriding num_pairs.

    World configs can set ``training.num_pairs`` to increase difficulty
    (e.g. 3 pairs forces multi-key-value retrieval).  The long split
    always uses at least ``num_pairs + 1`` for an OOD generalization test.
    """
    base = SPLIT_CONFIGS.get(split_name)
    if base is None:
        raise KeyError(f"unknown split: {split_name}")
    if num_pairs is None or num_pairs == base.num_pairs:
        return base
    pairs = num_pairs
    if split_name == "validation_long":
        pairs = num_pairs + 1  # OOD: one extra pair
    return SplitConfig(
        seq_len=base.seq_len,
        key_position_range=base.key_position_range,
        distractor_range=base.distractor_range,
        num_examples=base.num_examples,
        num_pairs=pairs,
    )


def generate_split(
    split_name: str,
    data_seed: int = 0,
    config: SplitConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate (input_ids, target_ids) for a split.

    Returns:
        input_ids:  (num_examples, seq_len) long tensor
        target_ids: (num_examples,) long tensor — the value token to predict
    """
    cfg = config or SPLIT_CONFIGS[split_name]
    split_digest = hashlib.sha256(split_name.encode("utf-8")).digest()
    split_offset = int.from_bytes(split_digest[:4], "big") % 10000
    rng = torch.Generator().manual_seed(data_seed * 1000 + split_offset)

    all_inputs = []
    all_targets = []

    for _ in range(cfg.num_examples):
        seq = torch.full((cfg.seq_len,), PAD_TOKEN, dtype=torch.long)

        lo, hi = cfg.key_position_range
        hi = min(hi, cfg.seq_len - 3)
        lo = min(lo, hi)

        if cfg.num_pairs <= 1:
            # Original single-pair path — identical random-state sequence
            # to the v0.1 code, preserving backward compatibility.
            key_idx = torch.randint(0, NUM_KEYS, (1,), generator=rng).item()
            value_idx = torch.randint(0, NUM_VALUES, (1,), generator=rng).item()
            key_token = KEY_TOKEN_BASE + key_idx
            value_token = VALUE_TOKEN_BASE + value_idx
            query_token = QUERY_TOKEN_BASE + key_idx

            key_pos = torch.randint(lo, hi + 1, (1,), generator=rng).item()
            seq[key_pos] = key_token
            seq[key_pos + 1] = value_token
            target_value = value_token
        else:
            # Multi-pair: place num_pairs key-value pairs with unique keys.
            n_pairs = min(cfg.num_pairs, NUM_KEYS)
            key_indices = torch.randperm(NUM_KEYS, generator=rng)[:n_pairs].tolist()

            placed_positions: set[int] = set()
            pair_values: dict[int, int] = {}
            for key_idx in key_indices:
                value_idx = torch.randint(0, NUM_VALUES, (1,), generator=rng).item()
                value_token = VALUE_TOKEN_BASE + value_idx
                pair_values[key_idx] = value_token

                for _attempt in range(50):
                    pos = torch.randint(lo, hi + 1, (1,), generator=rng).item()
                    if pos not in placed_positions and (pos + 1) not in placed_positions:
                        break
                seq[pos] = KEY_TOKEN_BASE + key_idx
                seq[pos + 1] = value_token
                placed_positions.add(pos)
                placed_positions.add(pos + 1)

            query_key_idx = key_indices[0]
            target_value = pair_values[query_key_idx]
            query_token = QUERY_TOKEN_BASE + query_key_idx

        # Fill remaining positions with distractors
        d_lo, d_hi = cfg.distractor_range
        for pos in range(cfg.seq_len - 1):
            if seq[pos] == PAD_TOKEN:
                d = torch.randint(d_lo, d_hi, (1,), generator=rng).item()
                seq[pos] = DISTRACTOR_BASE + d

        seq[-1] = query_token
        all_inputs.append(seq)
        all_targets.append(target_value)

    return torch.stack(all_inputs), torch.tensor(all_targets, dtype=torch.long)


@dataclass
class RetrievalDataset:
    """Wraps a generated split for batched iteration."""
    input_ids: Tensor
    target_ids: Tensor
    batch_size: int = 64

    def __len__(self) -> int:
        return math.ceil(len(self.input_ids) / self.batch_size)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        n = len(self.input_ids)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            yield self.input_ids[start:end], self.target_ids[start:end]

    @classmethod
    def from_split(
        cls,
        split_name: str,
        data_seed: int = 0,
        batch_size: int = 64,
        config: SplitConfig | None = None,
    ) -> "RetrievalDataset":
        inputs, targets = generate_split(split_name, data_seed, config)
        return cls(inputs, targets, batch_size)
