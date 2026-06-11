"""Deterministic budget accounting."""

from __future__ import annotations

from dataclasses import dataclass

from mechbench.interface.schemas import BudgetConfig


class BudgetExceeded(RuntimeError):
    """Raised when an experiment would exceed the episode budget."""


@dataclass
class Budget:
    config: BudgetConfig
    spent: float = 0.0

    @property
    def remaining(self) -> float:
        return max(0.0, self.config.max_cost - self.spent)

    def can_spend(self, cost: float) -> bool:
        return self.spent + cost <= self.config.max_cost + 1e-9

    def consume(self, cost: float) -> None:
        if cost < 0:
            raise ValueError("cost must be non-negative")
        if not self.can_spend(cost):
            raise BudgetExceeded(
                f"experiment cost {cost:.3f} exceeds remaining budget {self.remaining:.3f}"
            )
        self.spent += cost

    def to_dict(self) -> dict[str, float]:
        return {
            "max_cost": self.config.max_cost,
            "spent": self.spent,
            "remaining": self.remaining,
        }

