from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import httpx

from microtensor.backends import Backend, Hosted
from microtensor.backends import local as local_backend
from microtensor.backends.hosted import messages
from microtensor.catalogue import Catalogue, Served
from microtensor.core.constants import AUTO, DEFAULT_BASE_URL, DEFAULT_FLOOR
from microtensor.core.errors import TaskUnavailable
from microtensor.core.result import Result
from microtensor.escalation import Budget, choose


def catalogue(
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    transport: httpx.BaseTransport | None = None,
) -> Catalogue:
    hosted = Hosted.connect(api_key, base_url, transport)
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
        backend: Backend,
        catalogue: Catalogue,
        tasks: dict[str, str],
        quality_floor: float,
        escalation: bool,
        escalation_model: str | Mapping[str, Any],
        budget: Budget,
    ) -> None:
        self._backend = backend
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
            local_backend.connect(device=device)
        hosted = Hosted.connect(api_key, base_url, transport)
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
            backend=hosted,
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
        self._backend.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def run(self, question: str, *, context: str = "", task: str | None = None) -> Result:
        model = AUTO if task is None else self._model_for(task)
        sent = messages(question, context)
        result = Result.from_response(
            self._backend.chat(model=model, messages=sent, quality_floor=self.quality_floor)
        )
        if not result.below_floor or not self.escalation:
            return result
        target = choose(self._escalation_model, task=result.task, catalogue=self._catalogue)
        if target is None or not self.budget.allows(0.0):
            return result
        escalated = Result.from_response(
            self._backend.chat(model=target, messages=sent, quality_floor=self.quality_floor)
        )
        self.budget.charge(escalated.cost)
        return replace(escalated, task=result.task, escalated=True, escalated_to=target)

    def _model_for(self, task: str) -> str:
        entry = self._catalogue.resolve(task)
        model = self._tasks.get(entry.task, "") if entry else ""
        if not model:
            loaded = ", ".join(sorted(self._tasks)) or "none"
            raise TaskUnavailable(f"{task!r} was not loaded; loaded: {loaded}")
        return model
