from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import httpx

DEFAULT_BASE_URL: Final[str] = "https://api.microtensor.cloud/v1"
AUTO: Final[str] = "mt/auto"
DEFAULT_FLOOR: Final[float] = 0.9
TIMEOUT_SECONDS: Final[float] = 180.0
NO_KEY: Final[str] = "set MICROTENSOR_API_KEY or pass api_key"
LOCAL_PREVIEW: Final[str] = (
    "the local backend ships with the Enterprise preview; install microtensor-engine[local] "
    "with a licence token, or use backend='hosted'"
)


class EngineError(RuntimeError):
    pass


class TaskUnavailable(EngineError):
    pass


def _normal(name: str) -> str:
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
        return tuple(dict.fromkeys(_normal(n) for n in raw if n))

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
        wanted = _normal(name)
        for entry in self.served:
            if wanted in entry.names:
                return entry
        return None

    def describe(self) -> str:
        parts = []
        for entry in self.served:
            extra = f" ({', '.join(entry.aliases)})" if entry.aliases else ""
            parts.append(f"{entry.task}{extra}")
        return ", ".join(parts) or "none"


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


@dataclass(slots=True)
class Budget:
    usd_per_day: float | None = None
    spent: float = 0.0
    day: dt.date = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).date())

    def allows(self, cost: float) -> bool:
        self._roll()
        return self.usd_per_day is None or self.spent + cost <= self.usd_per_day

    def charge(self, cost: float) -> None:
        self._roll()
        self.spent += cost

    def _roll(self) -> None:
        today = dt.datetime.now(dt.timezone.utc).date()
        if today != self.day:
            self.day = today
            self.spent = 0.0


def _messages(question: str, context: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if context.strip():
        found.append({"role": "system", "content": context})
    found.append({"role": "user", "content": question})
    return found


class Hosted:
    def __init__(
        self, api_key: str, base_url: str, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        if not api_key:
            raise EngineError(NO_KEY)
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"authorization": f"Bearer {api_key}", "accept": "application/json"},
            timeout=TIMEOUT_SECONDS,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def _body(self, response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            raise EngineError("the API answered with something that is not JSON") from exc
        if response.status_code >= 400:
            error = body.get("error") if isinstance(body, dict) else {}
            error = error if isinstance(error, dict) else {}
            raise EngineError(
                f"{error.get('code', 'error')}: {error.get('message', response.text[:200])}"
            )
        if not isinstance(body, dict):
            raise EngineError("unexpected answer from the API")
        return body

    def models(self) -> list[dict[str, Any]]:
        body = self._body(self._http.get("/models"))
        return [dict(m) for m in body.get("data", [])]

    def catalogue(self) -> Catalogue:
        return Catalogue.from_models(self.models())

    def chat(
        self, *, model: str, messages: Sequence[Mapping[str, str]], quality_floor: float
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [dict(m) for m in messages],
            "quality_floor": quality_floor,
            "stream": False,
        }
        return self._body(self._http.post("/chat/completions", json=payload))


def _hosted(api_key: str | None, base_url: str, transport: httpx.BaseTransport | None) -> Hosted:
    return Hosted(
        api_key or os.environ.get("MICROTENSOR_API_KEY", ""), base_url, transport=transport
    )


def catalogue(
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    transport: httpx.BaseTransport | None = None,
) -> Catalogue:
    hosted = _hosted(api_key, base_url, transport)
    try:
        return hosted.catalogue()
    finally:
        hosted.close()


def served_tasks(
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    transport: httpx.BaseTransport | None = None,
) -> list[Served]:
    return list(catalogue(api_key=api_key, base_url=base_url, transport=transport).served)


def frontier_models(
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    transport: httpx.BaseTransport | None = None,
) -> list[str]:
    return list(catalogue(api_key=api_key, base_url=base_url, transport=transport).frontier)


class Engine:
    def __init__(
        self,
        *,
        hosted: Hosted,
        catalogue: Catalogue,
        tasks: dict[str, str],
        quality_floor: float,
        escalation: bool,
        escalation_model: str | Mapping[str, Any],
        budget: Budget,
    ) -> None:
        self._hosted = hosted
        self._catalogue = catalogue
        self._tasks = tasks
        self.quality_floor = quality_floor
        self.escalation = escalation
        self._escalation_model = escalation_model
        self.budget = budget

    @staticmethod
    def frontier_models(
        *, api_key: str | None = None, base_url: str = DEFAULT_BASE_URL
    ) -> list[str]:
        return frontier_models(api_key=api_key, base_url=base_url)

    @staticmethod
    def served_tasks(
        *, api_key: str | None = None, base_url: str = DEFAULT_BASE_URL
    ) -> list[Served]:
        return served_tasks(api_key=api_key, base_url=base_url)

    @classmethod
    def load(
        cls,
        tasks: Sequence[str],
        *,
        device: str = "cuda",
        quality_floor: float = DEFAULT_FLOOR,
        escalation: bool = True,
        escalation_model: str | Mapping[str, Any] = "auto",
        escalation_budget: Mapping[str, float] | None = None,
        backend: str = "hosted",
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> Engine:
        if backend != "hosted":
            raise EngineError(LOCAL_PREVIEW)
        hosted = _hosted(api_key, base_url, transport)
        found = hosted.catalogue()
        wanted: dict[str, str] = {}
        for name in tasks:
            entry = found.resolve(name)
            if entry is None:
                hosted.close()
                raise TaskUnavailable(
                    f"{name!r} is not served yet; served tasks: {found.describe()}"
                )
            wanted[entry.task] = entry.model
        budget = Budget(
            usd_per_day=float((escalation_budget or {}).get("usd_per_day", 0) or 0) or None
        )
        return cls(
            hosted=hosted,
            catalogue=found,
            tasks=wanted,
            quality_floor=quality_floor,
            escalation=escalation,
            escalation_model=escalation_model,
            budget=budget,
        )

    @property
    def tasks(self) -> dict[str, str]:
        return dict(self._tasks)

    @property
    def catalogue(self) -> Catalogue:
        return self._catalogue

    def close(self) -> None:
        self._hosted.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _escalation_for(self, task: str) -> str | None:
        chosen: Any = self._escalation_model
        if isinstance(chosen, Mapping):
            entry = self._catalogue.resolve(task)
            names = entry.names if entry else (_normal(task),)
            chosen = next((chosen[k] for k in chosen if _normal(str(k)) in names), "auto")
        if chosen is False or chosen is None:
            return None
        frontier = self._catalogue.frontier
        if chosen == "auto":
            return frontier[0] if frontier else None
        name = str(chosen)
        if name not in frontier:
            raise EngineError(f"{name!r} is not a frontier model the network serves")
        return name

    def run(self, question: str, *, context: str = "", task: str | None = None) -> Result:
        model = AUTO
        if task is not None:
            entry = self._catalogue.resolve(task)
            model = self._tasks.get(entry.task, "") if entry else ""
            if not model:
                raise TaskUnavailable(f"{task!r} was not loaded; loaded: {self._loaded()}")
        messages = _messages(question, context)
        found = self._hosted.chat(model=model, messages=messages, quality_floor=self.quality_floor)
        result = self._result(found)
        if not result.below_floor or not self.escalation:
            return result
        target = self._escalation_for(result.task)
        if target is None or not self.budget.allows(0.0):
            return result
        again = self._hosted.chat(model=target, messages=messages, quality_floor=self.quality_floor)
        escalated = self._result(again)
        self.budget.charge(escalated.cost)
        return Result(
            answer=escalated.answer,
            task=result.task,
            specialist=escalated.specialist,
            confidence=escalated.confidence,
            below_floor=escalated.below_floor,
            escalated=True,
            escalated_to=target,
            cost=escalated.cost,
            request_id=escalated.request_id,
            raw=again,
        )

    def _loaded(self) -> str:
        return ", ".join(sorted(self._tasks)) or "none"

    def _result(self, found: dict[str, Any]) -> Result:
        choices = found.get("choices") or [{}]
        message = choices[0].get("message") or {}
        routing = found.get("routing") or {}
        return Result(
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
