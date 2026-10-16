from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from microtensor.catalogue import Catalogue


class Backend(Protocol):
    def models(self) -> list[dict[str, Any]]: ...

    def catalogue(self) -> Catalogue: ...

    def chat(
        self, *, model: str, messages: Sequence[Mapping[str, str]], quality_floor: float
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...
