from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Result:
    answer: str
    task: str
    specialist: str
    confidence: float
    below_floor: bool
    escalated: bool
    escalated_to: str | None
    cost: float
    request_id: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_response(cls, found: dict[str, Any]) -> Result:
        choices = found.get("choices") or [{}]
        message = choices[0].get("message") or {}
        routing = found.get("routing") or {}
        return cls(
            answer=str(message.get("content", "")),
            task=str(routing.get("task", "")),
            specialist=str(routing.get("specialist", found.get("model", ""))),
            confidence=float(routing.get("answer_confidence") or 0.0),
            below_floor=not bool(routing.get("floor_met", False)),
            escalated=bool(routing.get("escalated", False)),
            escalated_to=None,
            cost=float((found.get("usage") or {}).get("cost") or 0.0),
            request_id=str(found.get("id", "")),
            raw=found,
        )
