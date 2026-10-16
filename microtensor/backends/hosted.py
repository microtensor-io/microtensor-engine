from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Final

import httpx

from microtensor.catalogue import Catalogue
from microtensor.constants import TIMEOUT_SECONDS
from microtensor.errors import EngineError

NO_KEY: Final[str] = "set MICROTENSOR_API_KEY or pass api_key"


def messages(question: str, context: str) -> list[dict[str, str]]:
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

    @classmethod
    def connect(
        cls,
        api_key: str | None,
        base_url: str,
        transport: httpx.BaseTransport | None = None,
    ) -> Hosted:
        return cls(
            api_key or os.environ.get("MICROTENSOR_API_KEY", ""), base_url, transport=transport
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
