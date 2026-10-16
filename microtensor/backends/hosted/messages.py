from __future__ import annotations


def messages(question: str, context: str = "") -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if context.strip():
        found.append({"role": "system", "content": context})
    found.append({"role": "user", "content": question})
    return found
