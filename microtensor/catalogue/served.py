from __future__ import annotations

from dataclasses import dataclass


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
