"""Unit tests for the tells detector."""

from __future__ import annotations

from pipelines.tells_detector import TellsDetector, detect_tells, summarize_tells


def test_detects_em_dash() -> None:
    tells = detect_tells("This is fine \u2014 and so is that.")
    assert any(t["tell_type"] == "em_dash" for t in tells)


def test_detects_delve_cluster() -> None:
    text = "Let us delve into the intricate tapestry of leveraging this realm."
    tells = detect_tells(text)
    matched = {t["matched_text"].lower() for t in tells}
    assert "delve" in matched
    assert "intricate" in matched
    assert "tapestry" in matched
    assert "realm" in matched


def test_detects_opening_compliment() -> None:
    tells = detect_tells("Great question! Here is the answer.")
    assert any(t["tell_type"] == "opening_compliment" for t in tells)


def test_detects_closing_offer() -> None:
    tells = detect_tells("Done. Let me know if you have any other questions.")
    assert any(t["tell_type"] == "closing_offer" for t in tells)


def test_detects_not_x_but_y() -> None:
    tells = detect_tells("It's not just a tool, it's a revolution.")
    assert any(t["tell_type"] == "not_x_but_y" for t in tells)


def test_severity_filter() -> None:
    text = "It is various and somewhat arguably interesting."
    only_high = detect_tells(text, min_severity=4)
    for t in only_high:
        assert t["severity"] >= 4


def test_summary_groups_by_category() -> None:
    text = "Great question! Let me delve into this fascinating tapestry \u2014 it's a paradigm shift."
    detector = TellsDetector()
    tells = detector.detect(text)
    summary = summarize_tells(tells)
    assert summary["total"] == len(tells)
    assert "lexical" in summary["by_category"]
    assert "punctuation" in summary["by_category"]


def test_empty_text() -> None:
    assert detect_tells("") == []


def test_line_numbers_correct() -> None:
    text = "First line.\nSecond line has delve in it.\nThird line."
    tells = detect_tells(text)
    delve_tells = [t for t in tells if t["matched_text"] == "delve"]
    assert delve_tells
    assert delve_tells[0]["line"] == 2
