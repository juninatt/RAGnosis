"""Tests for turning tables, infoboxes and captions in article HTML into text."""

from __future__ import annotations

import pytest

from ragnosis.parts import Part, extract_parts

INFOBOX = """
<table class="infobox vcard">
  <tr><th colspan="2" class="infobox-above">Blackbeard</th></tr>
  <tr><th class="infobox-label">Born</th>
      <td class="infobox-data">Edward Teach<br>c. 1680<br>Possibly
        <a href="/wiki/Bristol">Bristol</a>, England</td></tr>
  <tr><th class="infobox-label">Died</th>
      <td class="infobox-data">22 November 1718<sup class="reference">[3]</sup>
        (aged 35–40)</td></tr>
  <tr><th colspan="2" class="infobox-header">Belligerents</th></tr>
  <tr><td><ul><li>Athens</li><li>Sparta</li></ul></td><td>Persia</td></tr>
</table>
"""

TABLE = """
<div class="mw-heading mw-heading2"><h2 id="Forces">Opposing forces</h2>
  <span class="mw-editsection">[edit]</span></div>
<div class="mw-heading mw-heading3"><h3 id="Greek">Greek fleet</h3></div>
<table class="wikitable">
  <caption>Ships per city</caption>
  <tr><th>City</th><th>Number of ships</th></tr>
  <tr><td>Athens<sup class="reference">[61]</sup></td><td>180</td></tr>
  <tr><td>Corinth</td><td>40</td></tr>
</table>
"""


def test_infobox_becomes_one_line_per_fact() -> None:
    assert extract_parts(INFOBOX) == [
        Part(
            "Facts",
            "Born: Edward Teach; c. 1680; Possibly Bristol, England\n"
            "Died: 22 November 1718 (aged 35–40)\n"
            "Belligerents: Athens; Sparta | Persia",
        )
    ]


def test_table_becomes_one_line_per_row_under_its_heading() -> None:
    assert extract_parts(TABLE) == [
        Part(
            "Opposing forces > Greek fleet > Table",
            "Ships per city\n"
            "City: Athens; Number of ships: 180.\n"
            "City: Corinth; Number of ships: 40.",
        )
    ]


def test_merged_cells_are_repeated_where_they_span() -> None:
    html = """
    <table class="wikitable">
      <tr><th>Year</th><th>Side</th><th>Ships</th></tr>
      <tr><td rowspan="2">480 BC</td><td>Greeks</td><td>371</td></tr>
      <tr><td>Persians</td><td>600</td></tr>
      <tr><td colspan="2">Total</td><td>971</td></tr>
      <tr></tr>
    </table>
    """
    (part,) = extract_parts(html)
    assert part.text.splitlines() == [
        "Year: 480 BC; Side: Greeks; Ships: 371.",
        "Year: 480 BC; Side: Persians; Ships: 600.",
        "Year: Total; Side: Total; Ships: 971.",
    ]


def test_cells_spanning_three_rows_reach_the_third() -> None:
    html = """
    <table class="wikitable">
      <tr><td rowspan="3">Athens</td><td>480 BC</td></tr>
      <tr><td>479 BC</td></tr>
      <tr><td>478 BC</td></tr>
    </table>
    """
    assert extract_parts(html)[0].text.splitlines() == [
        "Athens; 480 BC.",
        "Athens; 479 BC.",
        "Athens; 478 BC.",
    ]


def test_table_without_a_heading_row_lists_its_cells() -> None:
    html = '<table class="wikitable"><tr><td>Athens</td><td></td><td>180</td></tr></table>'
    assert extract_parts(html) == [Part("Table", "Athens; 180.")]


def test_rows_of_another_length_than_the_heading_list_their_cells() -> None:
    html = """
    <table class="wikitable">
      <tr><th>City</th><th>Ships</th></tr>
      <tr><td>Plain numbers are triremes</td></tr>
    </table>
    """
    assert extract_parts(html)[0].text == "Plain numbers are triremes."


def test_empty_and_unreadable_spans_count_as_one() -> None:
    html = """
    <table class="wikitable">
      <tr><td colspan="x">A</td><td rowspan="">B</td></tr>
      <tr><td></td><td></td></tr>
    </table>
    """
    assert extract_parts(html) == [Part("Table", "A; B.")]


def test_captions_of_a_section_are_joined_and_short_ones_left_out() -> None:
    html = """
    <h2>Death</h2>
    <figure><figcaption>Teach's severed head hangs from Maynard's bowsprit</figcaption></figure>
    <figure><figcaption>Blackbeard, 1736</figcaption></figure>
    <figure><figcaption>A map of the area around Ocracoke Inlet</figcaption></figure>
    <h2>Legacy</h2>
    <figure><figcaption>The flag commonly attributed to Blackbeard</figcaption></figure>
    """
    assert extract_parts(html) == [
        Part(
            "Death > Images",
            "Teach's severed head hangs from Maynard's bowsprit\n"
            "A map of the area around Ocracoke Inlet",
        ),
        Part("Legacy > Images", "The flag commonly attributed to Blackbeard"),
    ]


def test_captions_before_the_first_heading_belong_to_the_introduction() -> None:
    html = "<figure><figcaption>A portrait of Edward Teach from 1736</figcaption></figure>"
    assert extract_parts(html) == [Part("Images", "A portrait of Edward Teach from 1736")]


def test_parts_keep_the_order_of_the_article() -> None:
    html = TABLE.replace(
        "</table>",
        "</table><figure><figcaption>Fleet of triremes based on a replica</figcaption></figure>",
    )
    html = "<figure><figcaption>Map of the Greek world at the time</figcaption></figure>" + html
    assert [part.path for part in extract_parts(html)] == [
        "Images",
        "Opposing forces > Greek fleet > Table",
        "Opposing forces > Greek fleet > Images",
    ]


def test_heading_levels_pop_back_to_the_parent() -> None:
    html = "<h2>A</h2><h3>B</h3><h2>C</h2>" + TABLE.split("</div>", 2)[2]
    assert extract_parts(html)[0].path == "C > Table"


def test_battle_infoboxes_read_their_nested_table() -> None:
    html = """
    <table class="infobox">
      <tr><td colspan="2"><table>
        <tr><th>Date</th><td>480 BC</td></tr>
        <tr><th>Result</th><td>Greek victory</td></tr>
      </table></td></tr>
    </table>
    """
    assert extract_parts(html) == [Part("Facts", "Date: 480 BC\nResult: Greek victory")]


def test_labels_are_kept_on_one_line_and_empty_values_dropped() -> None:
    html = """
    <table class="infobox">
      <tr><th>Territorial<br>changes</th><td>Persia withdrew</td></tr>
      <tr><th>Strength</th><td>371 ships<br>Modern estimates:<br>400 ships</td></tr>
      <tr><td></td></tr>
    </table>
    """
    assert extract_parts(html)[0].text == (
        "Territorial changes: Persia withdrew\nStrength: 371 ships; Modern estimates: 400 ships"
    )


def test_figures_and_tables_inside_a_table_belong_to_it() -> None:
    html = """
    <table class="infobox">
      <tr><td><figure><figcaption>The Capuan bust reputedly of Hannibal</figcaption></figure>
      </td></tr>
      <tr><td><table class="wikitable"><tr><td>Inner</td></tr></table></td></tr>
    </table>
    """
    assert [part.path for part in extract_parts(html)] == ["Facts"]


@pytest.mark.parametrize(
    "html",
    [
        "",
        "<p>Only text.</p>",
        '<table class="navbox"><tr><td>Pirates of the Caribbean</td></tr></table>',
        '<table class="wikitable"><tr><td></td></tr></table>',
        '<table class="infobox"></table>',
        '<span class="noprint">hidden</span><table class="wikitable"></table>',
    ],
    ids=["empty", "text", "navbox", "empty-table", "empty-infobox", "hidden"],
)
def test_nothing_wanted_gives_no_parts(html: str) -> None:
    assert extract_parts(html) == []


def test_coordinates_keep_one_form() -> None:
    html = """
    <table class="infobox"><tr><th>Location</th><td>Straits of Salamis
      <span class="geo-default">37°57′N</span><span class="geo-multi-punct"> / </span>
      <span class="geo-nondefault">37.95°N</span>&#xfeff;</td></tr></table>
    """
    assert extract_parts(html)[0].text == "Location: Straits of Salamis 37°57′N"
