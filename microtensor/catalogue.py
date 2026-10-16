from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from microtensor.constants import AUTO


def normal(name: str) -> str:
    return name.strip().lower().replace(" ", "-")


@dataclass(frozen=True, slots=True)
class Served:
    task: str
    model: str
    title: str
    aliases: tuple[str, ...]
    live: bool

    @property
    def names(self) -> tuple[str, ...]:
        raw = [
            self.task,
            self.task.replace("_", "-"),
            self.model,
            self.model.removeprefix("mt/"),
            *self.aliases,
        ]
        return tuple(dict.fromkeys(normal(n) for n in raw if n))

    @property
    def label(self) -> str:
        return self.aliases[0] if self.aliases else self.task


@dataclass(frozen=True, slots=True)
class Catalogue:
    served: tuple[Served, ...]
    frontier: tuple[str, ...]

    @classmethod
    def from_models(cls, models: Sequence[Mapping[str, Any]]) -> Catalogue:
        entries = [dict(m) for m in models]
        picked: dict[str, str] = {}
        live: dict[str, bool] = {}
        for entry in entries:
            if entry.get("id") != AUTO:
                continue
            for item in entry.get("tasks") or []:
                task = str(item.get("task", ""))
                if task:
                    picked[task] = str(item.get("model", ""))
                    live[task] = bool(item.get("live", False))
        for entry in entries:
            task = str(entry.get("task", ""))
            if task and entry.get("id") != AUTO and task not in picked:
                picked[task] = str(entry.get("id", ""))
                live[task] = bool((entry.get("capacity") or {}).get("live", False))
        by_id = {str(e.get("id", "")): e for e in entries}
        served = tuple(
            Served(
                task=task,
                model=model,
                title=str(by_id.get(model, {}).get("title", "")),
                aliases=tuple(str(a) for a in by_id.get(model, {}).get("aliases") or []),
                live=live.get(task, False),
            )
            for task, model in sorted(picked.items())
        )
        frontier = tuple(
            str(e["id"]) for e in entries if e.get("kind") == "frontier" and e.get("id")
        )
        return cls(served=served, frontier=frontier)

    def resolve(self, name: str) -> Served | None:
        wanted = normal(name)
        for entry in self.served:
            if wanted in entry.names:
                return entry
        return None

    def describe(self) -> str:
        parts: list[str] = []
        for entry in self.served:
            extra = f" ({', '.join(entry.aliases)})" if entry.aliases else ""
            parts.append(f"{entry.task}{extra}")
        return ", ".join(parts) or "none"
