from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from microtensor.catalogue import Catalogue, normal
from microtensor.core.errors import EngineError

NOT_FRONTIER = "{name!r} is not a frontier model the network serves"


def choose(target: str | Mapping[str, Any], *, task: str, catalogue: Catalogue) -> str | None:
    chosen: Any = target
    if isinstance(chosen, Mapping):
        entry = catalogue.resolve(task)
        names = entry.names if entry else (normal(task),)
        chosen = next((chosen[k] for k in chosen if normal(str(k)) in names), "auto")
    if chosen is False or chosen is None:
        return None
    if chosen == "auto":
        return catalogue.frontier[0] if catalogue.frontier else None
    name = str(chosen)
    if name not in catalogue.frontier:
        raise EngineError(NOT_FRONTIER.format(name=name))
    return name
