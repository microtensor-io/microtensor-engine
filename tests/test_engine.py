from __future__ import annotations

import httpx
import pytest

from microtensor import Engine, EngineError, TaskUnavailable

KEY = {"api_key": "mt_sk_x", "base_url": "http://g/v1"}
SCHEMA = "CREATE TABLE orders(x)"


def test_hosted_engine_reports_the_floor_and_pins_a_loaded_task(
    transport: httpx.MockTransport,
) -> None:
    with Engine.load(["text-to-sql"], **KEY, transport=transport) as engine:
        sure = engine.run("How many orders in 2025?", context=SCHEMA)
        assert sure.answer.startswith("SELECT") and sure.task == "sql_generation"
        assert sure.confidence == 0.96 and not sure.below_floor and not sure.escalated

        unsure = engine.run("How many orders?", context=SCHEMA)
        assert unsure.below_floor and not unsure.escalated and unsure.escalated_to is None

        pinned = engine.run("How many orders in 2025?", context=SCHEMA, task="text-to-sql")
        assert pinned.raw["model"] == "mt/sql_gen_s1"
        with pytest.raises(TaskUnavailable):
            engine.run("Is this spam?", task="classification")


def test_escalation_uses_a_served_frontier_model_within_budget(
    frontier_transport: httpx.MockTransport,
) -> None:
    with Engine.load(
        ["text-to-sql"],
        escalation_model={"text-to-sql": "deepseek-v4-pro"},
        escalation_budget={"usd_per_day": 1},
        **KEY,
        transport=frontier_transport,
    ) as engine:
        result = engine.run("How many orders?", context=SCHEMA)
        assert result.escalated and result.escalated_to == "deepseek-v4-pro"
        assert result.task == "sql_generation" and not result.below_floor
        assert result.cost == 0.002 and engine.budget.spent == 0.002

    with Engine.load(["text-to-sql"], escalation=False, **KEY, transport=frontier_transport) as off:
        assert off.run("How many orders?", context=SCHEMA).below_floor

    with pytest.raises(EngineError, match="frontier"):
        Engine.load(
            ["text-to-sql"], escalation_model="gpt-5", **KEY, transport=frontier_transport
        ).run("How many orders?", context=SCHEMA)


def test_the_local_backend_is_named_as_a_preview_and_a_key_is_required() -> None:
    with pytest.raises(EngineError):
        Engine.load(["text-to-sql"], backend="local", api_key="mt_sk_x")
    with pytest.raises(EngineError):
        Engine.load(["text-to-sql"], api_key="", base_url="http://g/v1")
