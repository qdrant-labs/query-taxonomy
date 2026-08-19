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


@pytest.fixture(scope="module")
def segmentation_bank():
    pytest.importorskip("wordfreq")
    from query_taxonomy.semantical import SegmentationBank

    return SegmentationBank()


class TestSegmentationBank:
    def _langs(self, bank, text):
        return [s.language for s in bank.compute(text)]

    def test_monolingual_english_single_span(self, segmentation_bank):
        assert self._langs(segmentation_bank, "best vector database") == ["en"]

    def test_detects_latin_latin_single_word(self, segmentation_bank):
        # the case lingua structurally could not do: a lone Latin foreign word
        langs = {s.language for s in segmentation_bank.compute("the fichier is missing")}
        assert langs == {"en", "fr"}

    def test_carrier_flips_to_dominant_language(self, segmentation_bank):
        # a mostly-German query: carrier is de, not en
        assert self._langs(segmentation_bank, "der server ist kaputt heute") == ["de"]

    def test_detects_nonlatin_code_switch(self, segmentation_bank):
        langs = {
            s.language
            for s in segmentation_bank.compute(
                "привет мир today I deploy a vector database on my server"
            )
        }
        assert "ru" in langs and "en" in langs

    def test_letterless_query_defaults_carrier(self, segmentation_bank):
        # numbers/identifiers are not language -> carrier (English-dominant)
        assert self._langs(segmentation_bank, "12345 0x1f") == ["en"]

    def test_empty_emits_nothing(self, segmentation_bank):
        assert segmentation_bank.compute("") == []

    def test_code_heavy_english_not_a_false_switch(self, segmentation_bank):
        # OOV code tokens must not switch off the carrier (target-floor guard)
        assert self._langs(segmentation_bank, "0x80070005 error code rviz colcon") == ["en"]


@pytest.fixture(scope="module")
def langid_extractor():
    pytest.importorskip("wordfreq")
    from query_taxonomy.core import Engine
    from query_taxonomy.features import FeatureExtractor

    return FeatureExtractor(engines=[Engine.LANGID])


class TestSemanticalDerivedViews:
    def _resolve(self, extractor, text):
        from query_taxonomy.taxonomy import FeatureGroup

        return extractor.resolve(text, groups=[FeatureGroup.SEMANTICAL])

    def test_english_not_code_switched(self, langid_extractor):
        f = self._resolve(langid_extractor, "best vector database")
        assert f.language_set == ["en"]
        assert f.language_count == 1
        assert f.is_code_switched is False

    def test_mixed_is_code_switched(self, langid_extractor):
        f = self._resolve(
            langid_extractor,
            "привет мир today I deploy a vector database on my server",
        )
        assert "ru" in f.language_set and "en" in f.language_set
        assert f.is_code_switched is True

    def test_derived_views_none_when_not_measured(self):
        # default extractor is regex-only -> the LANGID bank never ran, so the
        # derived views are None (not a false is_code_switched=False)
        from query_taxonomy.features import FeatureExtractor

        f = FeatureExtractor().resolve("привет мир this is clearly code-switched")
        assert f.language_set is None
        assert f.language_count is None
        assert f.is_code_switched is None


class TestCodeSwitchEval:
    """Falsifiability gate: the hand-authored, wordfreq-independent eval. These
    thresholds sit just under the calibrated point (csP 0.94 / csR 0.83 / set 0.90)."""

    def test_meets_calibrated_thresholds(self, segmentation_bank):
        import json
        from pathlib import Path

        path = Path(__file__).parents[2] / "eval" / "codeswitch_queries.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        tp = fp = fn = set_hit = 0
        for row in rows:
            pred = sorted({s.language for s in segmentation_bank.compute(row["text"])})
            pred_cs, gold_cs = len(pred) >= 2, row["code_switched"]
            tp += pred_cs and gold_cs
            fp += pred_cs and not gold_cs
            fn += (not pred_cs) and gold_cs
            set_hit += pred == row["languages"]
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        assert precision >= 0.88, precision
        assert recall >= 0.75, recall
        assert set_hit / len(rows) >= 0.82, set_hit / len(rows)
