"""Tests for headerless IMDb CSV loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.utils.imdb_schema import load_relation_rows
from src.utils.types import AttributeName

if TYPE_CHECKING:
    from pathlib import Path

COLUMNS = [
    AttributeName(name) for name in ("id", "movie_id", "info_type_id", "info", "note")
]

TITLE_COLUMNS = [
    AttributeName(name)
    for name in ("id", "title", "imdb_index", "kind_id", "production_year")
]


def test_backslash_escaped_quotes_do_not_split_a_field(tmp_path: Path) -> None:
    """IMDb escapes an inner quote as ``\\"``, not by doubling it.

    Line 1064433 of the real ``movie_info.csv``. With the CSV default the
    reader closes the quoted field at ``\\"`` and the two commas inside the
    value split it into extra fields, producing 7 columns for a 5-column row.
    """
    csv_path = tmp_path / "movie_info.csv"
    csv_path.write_text(
        '8330555,86921,18,"\\"Hatikva\\" neighborhood, Tel Aviv, Israel",\n',
        encoding="utf-8",
    )

    rows = load_relation_rows(csv_path, COLUMNS)

    assert len(rows) == 1
    assert rows[0]["info"] == '"Hatikva" neighborhood, Tel Aviv, Israel'
    assert rows[0]["note"] == ""


def test_unquoted_backslash_does_not_swallow_the_delimiter(tmp_path: Path) -> None:
    """Line 2514451 of the real ``title.csv``: an unquoted title ending in ``\\``.

    PostgreSQL CSV applies the escape only inside quotes, so this backslash is
    literal and the following comma still separates fields. Reading with
    ``escapechar`` instead merges two fields and loses a column.
    """
    csv_path = tmp_path / "title.csv"
    csv_path.write_text("2522636,\\Frag'ile\\,,1,2010\n", encoding="utf-8")

    rows = load_relation_rows(csv_path, TITLE_COLUMNS)

    assert rows[0]["title"] == "\\Frag'ile\\"
    assert rows[0]["imdb_index"] == ""
    assert rows[0]["production_year"] == "2010"


def test_plain_quoted_field_with_commas_is_one_value(tmp_path: Path) -> None:
    """Ordinary quoting still works; the escape fix must not break it."""
    csv_path = tmp_path / "movie_info.csv"
    csv_path.write_text('1,2,3,"Berlin, Germany",\n', encoding="utf-8")

    rows = load_relation_rows(csv_path, COLUMNS)

    assert rows[0]["info"] == "Berlin, Germany"


def test_wrong_schema_width_raises(tmp_path: Path) -> None:
    """A field-count mismatch must fail loudly, not mis-align columns."""
    csv_path = tmp_path / "movie_info.csv"
    csv_path.write_text("1,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected 5 fields"):
        load_relation_rows(csv_path, COLUMNS)
