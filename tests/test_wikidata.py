"""Tests for fetching other names from Wikidata, using a mock transport."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from ragnosis.wikidata import (
    BATCH_SIZE,
    MAX_NAMES,
    WikidataClient,
    WikidataError,
    clean_names,
)

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(
    handler: Handler, *, name_languages: tuple[str, ...] = ("en", "sv")
) -> WikidataClient:
    return WikidataClient(
        "en",
        name_languages,
        transport=httpx.MockTransport(handler),
        max_attempts=2,
        sleep=lambda seconds: None,
    )


def entity(title: str, labels: dict[str, str], aliases: dict[str, list[str]]) -> dict[str, Any]:
    """One Wikidata item as wbgetentities returns it with formatversion=2."""
    return {
        "id": "Q1",
        "labels": {lang: {"language": lang, "value": value} for lang, value in labels.items()},
        "aliases": {
            lang: [{"language": lang, "value": value} for value in values]
            for lang, values in aliases.items()
        },
        "sitelinks": {"enwiki": {"site": "enwiki", "title": title}},
    }


def returning(*entities: dict[str, Any]) -> Handler:
    payload = {"entities": {f"Q{n}": e for n, e in enumerate(entities, start=1)}}
    return lambda request: httpx.Response(200, json=payload)


BLACKBEARD = entity(
    "Blackbeard",
    {"en": "Blackbeard", "sv": "Svartskägg"},
    {"en": ["Edward Teach", "Edward Thatch"], "sv": ["Edward Teach"]},
)


# --- clean_names ---


def test_clean_names_drops_the_title_duplicates_and_short_names() -> None:
    names = clean_names("Blackbeard", ["Blackbeard", "Svartskägg", "svartskägg", "Din", "Ed"])
    assert names == ["Svartskägg"]


def test_clean_names_normalises_spaces_and_ignores_case_of_the_title() -> None:
    assert clean_names("John Rackham", ["john rackham", " Calico  Jack "]) == ["Calico Jack"]


def test_clean_names_keeps_at_most_the_maximum_in_order() -> None:
    names = clean_names("X", [f"Name {n}" for n in range(MAX_NAMES + 5)])
    assert names == [f"Name {n}" for n in range(MAX_NAMES)]


# --- fetch_names ---


def test_fetches_labels_then_aliases_in_language_order() -> None:
    with make_client(returning(BLACKBEARD)) as client:
        names = client.fetch_names(["Blackbeard"])
    assert names == {"Blackbeard": ["Svartskägg", "Edward Teach", "Edward Thatch"]}


def test_name_languages_decide_the_order() -> None:
    with make_client(returning(BLACKBEARD), name_languages=("sv", "en")) as client:
        names = client.fetch_names(["Blackbeard"])
    assert names["Blackbeard"][0] == "Svartskägg"


def test_sends_the_expected_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"entities": {}})

    with make_client(handler) as client:
        client.fetch_names(["Blackbeard", "John Rackham"])

    (request,) = requests
    assert request.url.host == "www.wikidata.org"
    params = request.url.params
    assert params["action"] == "wbgetentities"
    assert params["sites"] == "enwiki"
    assert params["titles"] == "Blackbeard|John Rackham"
    assert params["languages"] == "en|sv"


def test_asks_for_at_most_fifty_titles_at_a_time() -> None:
    batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        batches.append(request.url.params["titles"].split("|"))
        return httpx.Response(200, json={"entities": {}})

    titles = [f"Title {n}" for n in range(BATCH_SIZE + 3)]
    with make_client(handler) as client:
        names = client.fetch_names(titles)

    assert [len(batch) for batch in batches] == [BATCH_SIZE, 3]
    assert names == {title: [] for title in titles}


def test_titles_without_an_item_get_no_names() -> None:
    missing = {"site": "enwiki", "title": "No such page", "missing": ""}
    with make_client(returning(BLACKBEARD, missing)) as client:
        names = client.fetch_names(["Blackbeard", "No such page"])
    assert names["No such page"] == []


def test_items_for_titles_not_asked_for_are_ignored() -> None:
    other = entity("Anne Bonny", {"en": "Anne Bonny"}, {})
    with make_client(returning(other)) as client:
        assert client.fetch_names(["Blackbeard"]) == {"Blackbeard": []}


def test_asking_for_nothing_sends_nothing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    with make_client(handler) as client:
        assert client.fetch_names([]) == {}


# --- Failures ---


def test_failed_requests_raise_wikidata_errors() -> None:
    client = make_client(lambda request: httpx.Response(503))
    with client, pytest.raises(WikidataError, match="Giving up after 2 attempts"):
        client.fetch_names(["Blackbeard"])


def test_api_error_object_raises() -> None:
    payload = {"error": {"code": "param-missing"}}
    client = make_client(lambda request: httpx.Response(200, json=payload))
    with client, pytest.raises(WikidataError, match="API error"):
        client.fetch_names(["Blackbeard"])


@pytest.mark.parametrize(
    "payload",
    [{}, {"entities": []}, {"entities": {"Q1": {"labels": {}}}}],
    ids=["no-entities", "entities-not-object", "no-sitelinks"],
)
def test_unexpected_structure_raises(payload: dict[str, Any]) -> None:
    client = make_client(lambda request: httpx.Response(200, json=payload))
    with client, pytest.raises(WikidataError, match="Unexpected Wikidata response"):
        client.fetch_names(["Blackbeard"])


# --- Construction ---


@pytest.mark.parametrize(
    ("wiki_language", "name_languages"),
    [("EN", ("en",)), ("en", ("sv", "en.evil.com"))],
)
def test_rejects_invalid_language_codes(
    wiki_language: str, name_languages: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="language code"):
        WikidataClient(wiki_language, name_languages)


def test_rejects_no_name_languages() -> None:
    with pytest.raises(ValueError, match="At least one"):
        WikidataClient("en", ())


def test_site_follows_the_wiki_language() -> None:
    assert WikidataClient("zh-yue", ("en",)).site == "zh_yuewiki"
