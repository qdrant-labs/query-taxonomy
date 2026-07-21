"""CorpusReport smoke checks — text-section presence and rollup mechanics
(SPEC d27). Byte-level assertions live in the notebook demo; here we only
guard against structural regressions."""

from query_taxonomy.features import FeatureExtractor
from query_taxonomy.reporting import CorpusReport
from query_taxonomy.taxonomy import Domain, FeatureGroup


def _report_for(*texts: str) -> tuple[CorpusReport, str]:
    corpus = FeatureExtractor().extract(list(texts))
    report = CorpusReport(corpus)
    return report, report.text()


def test_text_sections_by_group_with_domain_rollup():
    report, text = _report_for(
        "upgrade to v1.9.2 after CVE-2024-3094",
        "see DOI 10.1145/3539618 for details",
        "no identifiers in this one",
    )

    header, *_ = text.splitlines()
    assert header == "3 queries profiled."
    assert "STRUCTURED IDENTIFIERS" in text
    assert "SENTENCE MARKERS" in text
    assert "tech" in text and "media" in text
    assert "cve" not in text  # detail lives on the rollup, not bank names
    assert "CVE-2024-3094 (1)" in text
    assert str(report) == text


def test_untagged_corpus_still_reports_stats():
    report, text = _report_for("walking shoes", "camping stoves")
    assert text.startswith("2 queries profiled.")
    assert "STRUCTURED IDENTIFIERS" not in text
    assert "STATISTICAL METRICS" in text
    assert "length_words" in text


def test_groups_filter_limits_extraction():
    corpus = FeatureExtractor().extract(
        ["no identifiers in this one"],
        groups=[FeatureGroup.STRUCTURED_IDENTIFIERS],
    )
    text = CorpusReport(corpus).text()
    assert text == "1 queries profiled."


def test_domain_rollup_respects_slice_budget():
    report, _ = _report_for(
        "upgrade to v1.9.2 after CVE-2024-3094",
        "see DOI 10.1145/3539618 for details",
    )
    slices = report.domain_rollup()
    assert len(slices) <= CorpusReport.SLICE_BUDGET
    assert all(s.spans == s.certified + s.assumed for s in slices)


def test_domain_query_share_aggregates_general():
    """`general` stays whole in the companion bar even though the donut
    explodes it — the two answer different questions (d27 reconciliation)."""
    report, _ = _report_for(
        "the release is v1.9.2",
        "chapter 3 covers CVE-2024-3094",
    )
    share = report.domain_query_share()
    # NUMBER + version_string + cve => at least one general-domain hit is
    # aggregated under the single "general" label here.
    assert Domain.GENERAL.value in share
    assert 0.0 < share[Domain.GENERAL.value] <= 1.0
