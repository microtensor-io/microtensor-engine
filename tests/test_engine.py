from __future__ import annotations

import json

import httpx
import pytest

from microtensor import Engine, EngineError, TaskUnavailable, frontier_models

MODELS = {
    "data": [
        {
            "id": "mt/auto",
            "tasks": [{"task": "sql_generation", "model": "mt/sql_gen_s1", "live": True}],
        },
        {"id": "mt/sql_gen_s1", "task": "sql_generation", "quality": 0.73},
    ]
}


def gateway(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/models"):
        return httpx.Response(200, json=MODELS)
    body = json.loads(request.content)
    confident = "2025" in body["messages"][-1]["content"]
    return httpx.Response(
        200,
        json={
            "id": "cmpl_1",
            "model": "mt/sql_gen_s1",
            "choices": [{"message": {"content": "SELECT count(*) FROM orders"}}],
            "routing": {
                "task": "sql_generation",
                "specialist": "mt/sql_gen_s1",
                "answer_confidence": 0.96 if confident else 0.4,
                "floor_met": confident,
                "quality_floor": body["quality_floor"],
            },
            "usage": {"total_tokens": 49},
        },
    )


def test_hosted_engine_maps_tasks_and_reports_the_floor() -> None:
    transport = httpx.MockTransport(gateway)
    engine = Engine.load(
        ["text-to-sql"], api_key="mt_sk_x", base_url="http://g/v1", transport=transport
    )
    assert engine.tasks == {"sql_generation": "mt/sql_gen_s1"}

    sure = engine.run("How many orders in 2025?", context="CREATE TABLE orders(x)")
    assert sure.answer.startswith("SELECT") and sure.task == "sql_generation"
    assert sure.confidence == 0.96 and not sure.below_floor and not sure.escalated

    unsure = engine.run("How many orders?", context="CREATE TABLE orders(x)")
    assert unsure.below_floor and not unsure.escalated and unsure.escalated_to is None
    engine.close()

    with pytest.raises(TaskUnavailable):
        Engine.load(
            ["classification"], api_key="mt_sk_x", base_url="http://g/v1", transport=transport
        )
    assert frontier_models(api_key="mt_sk_x", base_url="http://g/v1", transport=transport) == []


def test_the_local_backend_is_named_as_a_preview_and_a_key_is_required() -> None:
    with pytest.raises(EngineError):
        Engine.load(["text-to-sql"], backend="local", api_key="mt_sk_x")
    with pytest.raises(EngineError):
        Engine.load(["text-to-sql"], api_key="", base_url="http://g/v1")
