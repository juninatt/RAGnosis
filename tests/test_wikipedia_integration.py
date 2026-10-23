"""Checks against the real Wikipedia API. Opt-in: ``uv run pytest -m integration``.

They catch what mocks cannot: changes in the API's response format.
"""

from __future__ import annotations

import pytest

from ragnosis.wikipedia import ArticleNotFoundError, DisambiguationPageError, WikipediaClient

pytestmark = pytest.mark.integration


def test_fetches_a_real_article_through_a_redirect() -> None:
    with WikipediaClient("en") as client:
        article = client.fetch_article("Edward Teach")

    assert article.title == "Blackbeard"
    assert article.page_id > 0
    assert article.revision_id > 0
    assert article.url == "https://en.wikipedia.org/wiki/Blackbeard"
    assert "pirate" in article.text.lower()


def test_real_missing_article() -> None:
    with WikipediaClient("en") as client, pytest.raises(ArticleNotFoundError):
        client.fetch_article("Ragnosis test page that does not exist 7f3a")


def test_real_disambiguation_page() -> None:
    with WikipediaClient("en") as client, pytest.raises(DisambiguationPageError):
        client.fetch_article("Mercury")
