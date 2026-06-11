"""Family and world discovery."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from mechbench.core.episode import Episode
from mechbench.core.family import FamilyInterface, TableFamily, World
from mechbench.interface.schemas import BudgetConfig, Headline, InterventionSpec
from mechbench.utils.config import load_config


class Registry:
    """Auto-discovers families and worlds from a `families/` directory."""

    def __init__(self, families_dir: str | Path = "families"):
        self.families_dir = Path(families_dir)
        self.families: dict[str, FamilyInterface] = {}
        self.worlds: dict[str, World] = {}
        self._discover()

    def _discover(self) -> None:
        if not self.families_dir.exists():
            return
        for family_dir in sorted(self.families_dir.iterdir()):
            if family_dir.name.startswith("_") or not family_dir.is_dir():
                continue
            family_config_path = family_dir / "family.yaml"
            if not family_config_path.exists():
                continue
            family = self._load_family(family_dir, load_config(family_config_path))
            self.families[family.name] = family
            worlds_dir = family_dir / "worlds"
            if not worlds_dir.exists():
                continue
            for world_dir in sorted(worlds_dir.iterdir()):
                if world_dir.name.startswith("_") or not world_dir.is_dir():
                    continue
                world_config_path = world_dir / "world.yaml"
                if not world_config_path.exists():
                    continue
                world = self._load_world(world_dir, family, load_config(world_config_path))
                self.worlds[world.world_id] = world

    def _load_module(self, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(f"mechbench_family_{path.parent.name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not import {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_family(self, family_dir: Path, config: dict[str, Any]) -> FamilyInterface:
        family_py = family_dir / "family.py"
        if family_py.exists():
            module = self._load_module(family_py)
            if hasattr(module, "build_family"):
                family = module.build_family(family_dir, config)
            elif hasattr(module, "Family"):
                family = module.Family(family_dir, config)
            else:
                raise ValueError(f"{family_py} must define build_family() or Family")
            if not isinstance(family, FamilyInterface):
                raise TypeError(f"{family_py} did not create a FamilyInterface")
            return family
        return TableFamily(family_dir, config)

    def _load_world(self, world_dir: Path, family: FamilyInterface, config: dict[str, Any]) -> World:
        headline = Headline.from_dict(config.get("headline", {}))
        budget = BudgetConfig.from_dict(config.get("budget", {}))
        interventions = [InterventionSpec.from_dict(item) for item in config.get("interventions", [])]
        causal_label = str(config.get("causal_label", ""))
        if not causal_label:
            raise ValueError(f"{world_dir / 'world.yaml'} missing causal_label")
        return World(
            name=str(config.get("name", world_dir.name)),
            family=family.name,
            root=world_dir,
            config=config,
            headline=headline,
            budget=budget,
            interventions=interventions,
            causal_label=causal_label,
            certificate=dict(config.get("certificate", {})),
        )

    def get_episode(self, world_id: str) -> Episode:
        world = self.worlds[world_id]
        family = self.families[world.family]
        return Episode(family=family, world=world)

    def list_worlds(self, family: str | None = None) -> list[str]:
        if family:
            return sorted(world_id for world_id in self.worlds if world_id.startswith(f"{family}/"))
        return sorted(self.worlds)

