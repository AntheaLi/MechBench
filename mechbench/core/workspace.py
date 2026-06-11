"""Per-episode workspace setup and safe inspection."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mechbench.core.family import FamilyInterface, World


class WorkspaceAccessError(PermissionError):
    """Raised when an inspect call tries to leave the workspace."""


@dataclass
class Workspace:
    family: FamilyInterface
    world: World
    root: Path | None = None
    access_log: list[dict[str, Any]] = field(default_factory=list)

    def create(self) -> "Workspace":
        if self.root is None:
            self.root = Path(tempfile.mkdtemp(prefix="mechbench_"))
        self.root.mkdir(parents=True, exist_ok=True)

        for relative in self.family.public_paths():
            source = self.family.root / relative
            if not source.exists():
                continue
            destination = self.root / relative
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

        results_dir = self.root / "results"
        results_dir.mkdir(exist_ok=True)
        (results_dir / "initial_result.json").write_text(
            json.dumps(self.world.headline.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.root / "README.md").write_text(self._workspace_readme(), encoding="utf-8")
        (self.root / "evidence").mkdir(exist_ok=True)
        self.family.setup_workspace(self.world, self.root)
        return self

    def _workspace_readme(self) -> str:
        return (
            f"# MechanismBench Episode: {self.world.public_id}\n\n"
            "You are investigating a claimed improvement. Inspect the public files, "
            "run controlled experiments through the API, and submit a structured "
            "causal report plus held-out intervention predictions.\n\n"
            "## Headline Result\n\n"
            f"{self.world.headline.description}\n"
        )

    def _resolve(self, path: str) -> Path:
        if self.root is None:
            raise RuntimeError("workspace has not been created")
        candidate = (self.root / path).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            raise WorkspaceAccessError(f"path escapes workspace: {path}")
        return candidate

    def inspect(self, path: str) -> str:
        target = self._resolve(path)
        if target.is_dir():
            listing = sorted(str(item.relative_to(target)) for item in target.iterdir())
            content = "\n".join(listing)
        else:
            content = target.read_text(encoding="utf-8")
        self.access_log.append({"path": path, "resolved": str(target)})
        return content

    def list_files(self) -> list[str]:
        if self.root is None:
            raise RuntimeError("workspace has not been created")
        return sorted(
            str(path.relative_to(self.root))
            for path in self.root.rglob("*")
            if path.is_file()
        )

