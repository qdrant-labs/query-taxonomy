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

    def test_coordination_measures_widest_chain(self, pos_bank):
        from query_taxonomy.metrics.pos import CoordinationBank

        bank = CoordinationBank()
        cities = {
            s.name: s.value
            for s in bank.compute("cheap flights to boston, paris and tokyo")
        }
        assert cities == {"widest_list_size": 3.0}

        actions = {
            s.name: s.value
            for s in bank.compute("install docker and configure the network")
        }
        assert actions == {"widest_list_size": 2.0}

    def test_coordination_zero_when_nothing_is_coordinated(self, pos_bank):
        from query_taxonomy.metrics.pos import CoordinationBank

        stats = {
            s.name: s.value
            for s in CoordinationBank().compute("quantum computing paper")
        }
        assert stats == {"widest_list_size": 0.0}


@pytest.fixture(scope="module")
def rarity_bank():
    pytest.importorskip("wordfreq")
    from query_taxonomy.metrics.frequency import RarityBank

    return RarityBank()


class TestRarityBank:
    def test_rare_query_is_rarer_than_common_query(self, rarity_bank):
        common = {
            s.name: s.value for s in rarity_bank.compute("what is the best day")
        }
        rare = {
            s.name: s.value
            for s in rarity_bank.compute("myocardial infarction pathophysiology")
        }
        assert common["mean_zipf"] > 4.0  # common words are frequent
        assert rare["mean_zipf"] < common["mean_zipf"]
        assert rare["rare_share"] > common["rare_share"]

    def test_empty_text_emits_nothing(self, rarity_bank):
        assert rarity_bank.compute("") == []

    def test_all_oov_emits_nothing(self, rarity_bank):
        # no in-vocabulary token -> rarity undefined, not a maximally-rare 0.0
        assert rarity_bank.compute("zzqxk wvbxzq") == []


@pytest.fixture(scope="module")
def fragmentation_bank():
    pytest.importorskip("tokenizers")
    from query_taxonomy.metrics.fragmentation import FragmentationBank, _tokenizer

    try:
        _tokenizer()  # from_pretrained fetches the vocab on first use
    except Exception as exc:  # offline / hub unreachable
        pytest.skip(f"tokenizer unavailable: {exc}")
    return FragmentationBank()


class TestFragmentationBank:
    def test_technical_word_shatters_more_than_common_words(
        self, fragmentation_bank
    ):
        clean = {
            s.name: s.value
            for s in fragmentation_bank.compute("what is the best day")
        }
        technical = {
            s.name: s.value
            for s in fragmentation_bank.compute("phosphorylation dysregulation")
        }
        assert clean["mean_pieces_per_word"] == pytest.approx(1.0)
        assert technical["max_pieces_per_word"] > 1.0
        assert (
            technical["mean_pieces_per_word"] > clean["mean_pieces_per_word"]
        )

    def test_punctuation_and_hyphens_do_not_inflate(self, fragmentation_bank):
        # regression: text.split() made "learning?" / "state-of-the-art" read
        # as one badly-shattered token; word_ids grouping keeps each clean.
        question = {
            s.name: s.value
            for s in fragmentation_bank.compute("what is machine learning?")
        }
        hyphenated = {
            s.name: s.value
            for s in fragmentation_bank.compute("a state-of-the-art tool")
        }
        assert question["max_pieces_per_word"] == pytest.approx(1.0)
        assert hyphenated["max_pieces_per_word"] == pytest.approx(1.0)

    def test_empty_text_emits_nothing(self, fragmentation_bank):
        assert fragmentation_bank.compute("") == []


@pytest.fixture(scope="module")
def unknown_rate_bank():
    pytest.importorskip("wordfreq")
    from query_taxonomy.corruption import UnknownTokenRateBank

    return UnknownTokenRateBank()


class TestUnknownTokenRateBank:
    def test_share_of_absent_words(self, unknown_rate_bank):
        stats = {
            s.name: s.value
            for s in unknown_rate_bank.compute("the zxqwvk machine")
        }
        assert stats["unknown_token_rate"] == pytest.approx(1 / 3)

    def test_word_shape_guard_excludes_identifiers(self, unknown_rate_bank):
        # digits/identifiers are not word-shaped -> not counted as unknown
        stats = {
            s.name: s.value
            for s in unknown_rate_bank.compute("error 0x80070005 code")
        }
        assert stats["unknown_token_rate"] == 0.0

    def test_no_word_shaped_tokens_emits_nothing(self, unknown_rate_bank):
        assert unknown_rate_bank.compute("123 456") == []


@pytest.fixture(scope="module")
def typo_bank():
    pytest.importorskip("wordfreq")
    from query_taxonomy.corruption import TypoBank

    return TypoBank()


class TestTypoBank:
    def test_flags_keyword_adjacent_typo(self, typo_bank):
        spans = typo_bank.compute("how to confgure the server")
        assert [s.text for s in spans] == ["confgure"]  # -> configure

    def test_ignores_rare_real_words_and_names(self, typo_bank):
        # a name absent from the table but not one edit from a frequent word
        assert typo_bank.compute("deploy Qdrant now") == []

    def test_ignores_short_tokens(self, typo_bank):
        # a single OOV letter is one edit from 'a'/'i' but too short to judge
        assert typo_bank.compute("q z x") == []

    def test_case_guard_ignores_code_identifiers(self, typo_bank):
        # census false positives: internal caps -> camelCase/PascalCase code,
        # not a slip. One edit from a frequent word but never a typo.
        for token in ("asList", "resolveA", "dataA", "TypeId"):
            assert typo_bank.compute(token) == [], token

    def test_still_flags_real_typos_from_census(self, typo_bank):
        # the real misspellings the census surfaced, drowned by the FPs above
        for token in ("implemeted", "sequnece", "smalest", "algebric", "operato"):
            assert [s.text for s in typo_bank.compute(token)] == [token], token

    def test_accepted_floor_lowercase_jargon_still_fires(self, typo_bank):
        # documented ceiling: all-lowercase domain jargon one edit from a common
        # word (pthread->thread) is indistinguishable from a slip within-group,
        # so it stays flagged. Read the lane rate, not the per-span claim.
        assert typo_bank.compute("pthread colcon stdio")
