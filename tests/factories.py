"""Builders for test data, so each test only spells out the fields it cares about."""

from __future__ import annotations

from typing import Any

from ragnosis.wikipedia import Article


def make_article(**overrides: Any) -> Article:
    fields: dict[str, Any] = {
        "title": "Blackbeard",
        "page_id": 1,
        "revision_id": 100,
        "url": "https://en.wikipedia.org/wiki/Blackbeard",
        "language": "en",
        "text": "Edward Teach, better known as Blackbeard, was an English pirate.",
        "fetched_at": "2026-10-21T12:00:00+00:00",
    }
    fields.update(overrides)
    return Article(**fields)


def api_page(**overrides: Any) -> dict[str, Any]:
    """One page as the API returns it with formatversion=2."""
    page: dict[str, Any] = {
        "pageid": 1,
        "ns": 0,
        "title": "Blackbeard",
        "extract": "Edward Teach, better known as Blackbeard, was an English pirate.",
        "lastrevid": 100,
        "fullurl": "https://en.wikipedia.org/wiki/Blackbeard",
    }
    page.update(overrides)
    return page


def api_response(*pages: dict[str, Any]) -> dict[str, Any]:
    return {"batchcomplete": True, "query": {"pages": list(pages)}}
