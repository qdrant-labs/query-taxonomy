"""Model-engine bank checks — skipped when the downloaded spaCy model is
missing. importorskip stays inside fixtures: at module level it would skip
the whole file and hide other tests."""

import pytest


@pytest.fixture(scope="module")
def pos_bank():
    pytest.importorskip("spacy")
    from query_taxonomy.metrics.config import DEFAULT_SPACY_MODEL
    from query_taxonomy.metrics.pos import NaturalLanguageSignalBank

    import spacy

    if not spacy.util.is_package(DEFAULT_SPACY_MODEL):
        pytest.skip(f"{DEFAULT_SPACY_MODEL} not downloaded")
    return NaturalLanguageSignalBank()


class TestSpacyBanks:
    def test_sentence_beats_telegram_on_natural_language_share(self, pos_bank):
        telegram = {
            stat.name: stat.value
            for stat in pos_bank.compute("cheap laptops berlin")
        }
        sentence = {
            stat.name: stat.value
            for stat in pos_bank.compute(
                "where can i buy a cheap laptop in berlin"
            )
        }
        assert telegram["natural_language_share"] < 0.2
        assert sentence["natural_language_share"] > telegram["natural_language_share"]

    def test_empty_text(self, pos_bank):
        stats = {stat.name: stat.value for stat in pos_bank.compute("")}
        assert stats == {"natural_language_share": 0.0}

    def test_morphology_separates_inflected_from_lemmas(self, pos_bank):
        from query_taxonomy.metrics.pos import MorphologyBank

        bank = MorphologyBank()
        # running -> run, shoes -> shoe, feet -> foot
        inflected = {
            s.name: s.value
            for s in bank.compute("running shoes for flat feet")
        }
        lemmas = {s.name: s.value for s in bank.compute("run shoe flat foot")}
        assert 0.0 < inflected["word_variation_share"] <= 1.0
        assert lemmas["word_variation_share"] == 0.0

    def test_syntactic_depth_separates_deep_from_flat(self, pos_bank):
        from query_taxonomy.metrics.pos import SyntacticDepthBank

        bank = SyntacticDepthBank()
        flat = {s.name: s.value for s in bank.compute("attention mechanism paper")}
        deep = {
            s.name: s.value
            for s in bank.compute(
                "the paper that introduced the attention mechanism used in transformers"
            )
        }
        assert deep["nesting_depth"] > flat["nesting_depth"]
        assert deep["statement_count"] >= 2
