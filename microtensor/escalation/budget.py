from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


def today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


@dataclass(slots=True)
class Budget:
    usd_per_day: float | None = None
    spent: float = 0.0
    day: dt.date = field(default_factory=today)

    def allows(self, cost: float) -> bool:
        self._roll()
        return self.usd_per_day is None or self.spent + cost <= self.usd_per_day

    def charge(self, cost: float) -> None:
        self._roll()
        self.spent += cost

    def _roll(self) -> None:
        now = today()
        if now != self.day:
            self.day = now
            self.spent = 0.0
