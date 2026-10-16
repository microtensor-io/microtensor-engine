from __future__ import annotations

from typing import Any

import httpx
import pytest

from microtensor import Catalogue, Engine, TaskUnavailable, frontier_models, served_tasks

KEY = {"api_key": "mt_sk_x", "base_url": "http://g/v1"}


def test_the_catalogue_reads_tasks_aliases_and_liveness(models: list[dict[str, Any]]) -> None:
    found = Catalogue.from_models(models)
    assert [s.task for s in found.served] == ["classification", "sql_generation"]
    sql = found.served[1]
    assert sql.aliases == ("text-to-sql", "sql") and sql.live and sql.label == "text-to-sql"
    assert not found.served[0].live and found.served[0].label == "classification"
    assert found.frontier == ()
    for name in ("text-to-sql", "SQL", "sql_generation", "sql-generation", "sql_gen_s1"):
        assert found.resolve(name) is sql
    assert found.resolve("mt/sql_gen_s1") is sql and found.resolve("summarisation") is None
    assert found.describe() == "classification, sql_generation (text-to-sql, sql)"


def test_served_tasks_and_frontier_models_read_the_live_list(
    transport: httpx.MockTransport, frontier_transport: httpx.MockTransport
) -> None:
    served = served_tasks(**KEY, transport=transport)
    assert [s.task for s in served] == ["classification", "sql_generation"]
    assert frontier_models(**KEY, transport=transport) == []
    assert frontier_models(**KEY, transport=frontier_transport) == ["deepseek-v4-pro"]


def test_load_resolves_names_against_the_catalogue(transport: httpx.MockTransport) -> None:
    for name in ("text-to-sql", "SQL", "sql_generation"):
        engine = Engine.load([name], **KEY, transport=transport)
        assert engine.tasks == {"sql_generation": "mt/sql_gen_s1"}
        engine.close()
    with pytest.raises(TaskUnavailable, match="text-to-sql"):
        Engine.load(["summarisation"], **KEY, transport=transport)
