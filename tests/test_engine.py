from __future__ import annotations

import json

import httpx
import pytest

from microtensor import Engine, EngineError, TaskUnavailable, frontier_models, served_tasks

MODELS = {
    "data": [
        {
            "id": "mt/auto",
            "tasks": [{"task": "sql_generation", "model": "mt/sql_gen_s1", "live": True}],
        },
        {
            "id": "mt/sql_gen_s1",
            "task": "sql_generation",
            "title": "SQL generation, season 1",
            "aliases": ["text-to-sql", "sql"],
            "quality": 0.73,
            "capacity": {"operators": 2, "live": True},
        },
        {
            "id": "mt/classify_s1",
            "task": "classification",
            "title": "Classification, season 1",
            "capacity": {"operators": 0, "live": False},
        },
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
            "model": body["model"],
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


def test_task_names_come_from_the_live_catalogue() -> None:
    transport = httpx.MockTransport(gateway)
    served = served_tasks(api_key="mt_sk_x", base_url="http://g/v1", transport=transport)
    assert [s.task for s in served] == ["classification", "sql_generation"]
    assert served[1].aliases == ("text-to-sql", "sql") and served[1].live
    assert not served[0].live and served[0].label == "classification"

    for name in ("text-to-sql", "SQL", "sql_generation", "sql-generation", "sql_gen_s1"):
        engine = Engine.load([name], api_key="mt_sk_x", base_url="http://g/v1", transport=transport)
        assert engine.tasks == {"sql_generation": "mt/sql_gen_s1"}
        engine.close()

    with pytest.raises(TaskUnavailable, match="text-to-sql"):
        Engine.load(
            ["summarisation"], api_key="mt_sk_x", base_url="http://g/v1", transport=transport
        )
    assert frontier_models(api_key="mt_sk_x", base_url="http://g/v1", transport=transport) == []


def test_hosted_engine_reports_the_floor_and_pins_a_loaded_task() -> None:
    transport = httpx.MockTransport(gateway)
    engine = Engine.load(
        ["text-to-sql"], api_key="mt_sk_x", base_url="http://g/v1", transport=transport
    )
    sure = engine.run("How many orders in 2025?", context="CREATE TABLE orders(x)")
    assert sure.answer.startswith("SELECT") and sure.task == "sql_generation"
    assert sure.confidence == 0.96 and not sure.below_floor and not sure.escalated

    unsure = engine.run("How many orders?", context="CREATE TABLE orders(x)")
    assert unsure.below_floor and not unsure.escalated and unsure.escalated_to is None

    pinned = engine.run("How many orders in 2025?", context="x", task="text-to-sql")
    assert pinned.raw["model"] == "mt/sql_gen_s1"
    with pytest.raises(TaskUnavailable):
        engine.run("Is this spam?", task="classification")
    engine.close()


def test_the_local_backend_is_named_as_a_preview_and_a_key_is_required() -> None:
    with pytest.raises(EngineError):
        Engine.load(["text-to-sql"], backend="local", api_key="mt_sk_x")
    with pytest.raises(EngineError):
        Engine.load(["text-to-sql"], api_key="", base_url="http://g/v1")
