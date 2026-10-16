from __future__ import annotations

from typing import Final, NoReturn

from microtensor.core.errors import EngineError

LOCAL_PREVIEW: Final[str] = (
    "the local backend ships with the Enterprise preview; install microtensor-engine[local] "
    "with a licence token, or use backend='hosted'"
)


def connect(*_: object, **__: object) -> NoReturn:
    raise EngineError(LOCAL_PREVIEW)


__all__ = ["LOCAL_PREVIEW", "connect"]
