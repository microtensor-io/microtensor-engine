from __future__ import annotations

import httpx
import pytest

from microtensor import EngineError, Hosted
from microtensor.backends import local
from microtensor.backends.hosted import messages


def test_the_hosted_backend_needs_a_key_and_reads_the_environment(
    monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport
) -> None:
    monkeypatch.delenv("MICROTENSOR_API_KEY", raising=False)
    with pytest.raises(EngineError):
        Hosted.connect(None, "http://g/v1", transport)
    monkeypatch.setenv("MICROTENSOR_API_KEY", "mt_sk_env")
    hosted = Hosted.connect(None, "http://g/v1", transport)
    assert [s.task for s in hosted.catalogue().served] == ["classification", "sql_generation"]
    hosted.close()


def test_messages_put_the_context_in_the_system_message() -> None:
    assert messages("q", "ctx") == [
        {"role": "system", "content": "ctx"},
        {"role": "user", "content": "q"},
    ]
    assert messages("q", "  ") == [{"role": "user", "content": "q"}]


def test_the_local_backend_names_the_preview() -> None:
    with pytest.raises(EngineError, match="Enterprise preview"):
        local.connect(device="cuda")
