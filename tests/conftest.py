from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

SQL = {
    "id": "mt/sql_gen_s1",
    "task": "sql_generation",
    "title": "SQL generation, season 1",
    "aliases": ["text-to-sql", "sql"],
    "quality": 0.73,
    "capacity": {"operators": 2, "live": True},
}
CLASSIFY = {
    "id": "mt/classify_s1",
    "task": "classification",
    "title": "Classification, season 1",
    "capacity": {"operators": 0, "live": False},
}
AUTO = {
    "id": "mt/auto",
    "tasks": [{"task": "sql_generation", "model": "mt/sql_gen_s1", "live": True}],
}
FRONTIER = {"id": "deepseek-v4-pro", "kind": "frontier", "object": "model"}

MODELS: list[dict[str, Any]] = [AUTO, SQL, CLASSIFY]


def make_gateway(models: list[dict[str, Any]]) -> Callable[[httpx.Request], httpx.Response]:
    def gateway(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": models})
        body = json.loads(request.content)
        frontier = body["model"] == FRONTIER["id"]
        confident = frontier or "2025" in body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "id": "cmpl_1",
                "model": body["model"],
                "choices": [{"message": {"content": "SELECT count(*) FROM orders"}}],
                "routing": {
                    "task": "sql_generation",
                    "specialist": body["model"],
                    "answer_confidence": 0.96 if confident else 0.4,
                    "floor_met": confident,
                    "quality_floor": body["quality_floor"],
                },
                "usage": {"total_tokens": 49, "cost": 0.002 if frontier else 0.0},
            },
        )

    return gateway


@pytest.fixture
def models() -> list[dict[str, Any]]:
    return MODELS


@pytest.fixture
def transport() -> httpx.MockTransport:
    return httpx.MockTransport(make_gateway(MODELS))


@pytest.fixture
def frontier_transport() -> httpx.MockTransport:
    return httpx.MockTransport(make_gateway([*MODELS, FRONTIER]))
