
DEFAULT_SPACY_MODEL = "en_core_web_sm"

# Universal Dependencies (UD) closed-class part-of-speech tags, MINUS numerals.
# Closed-class categories contain a small, relatively fixed inventory of words.
# NUM is closed-class in UD and deliberately excluded here: this set answers
# "is the query grammatical glue or a keyword telegram?", and a numeral is
# content, not glue. Keeping it counted 502 in "v1.2.3 nginx.conf 502" as a
# function word, and spaCy tags bare identifiers NUM, so the error landed
# hardest on exactly the identifier-dense short queries the signal is meant to
# score near 0.0 (measured: 3 of 7 real symbol piles pushed over a 0.1 band;
# 24.8% of natural queries inflated, 2.0% crossing that band).
CLOSED_CLASS = frozenset(
    {
        "ADP",    # Adposition: prepositions/postpositions expressing grammatical relations (e.g. "in", "to", "with").
        "AUX",    # Auxiliary verb: grammatical verb marking tense, aspect, mood, voice, or polarity (e.g. "is", "have", "will").
        "CCONJ",  # Coordinating conjunction: links elements of equal syntactic status (e.g. "and", "or", "but").
        "DET",    # Determiner: specifies or limits a noun (e.g. "the", "this", "some", "each").
        "PART",   # Particle: function word not fitting other categories, often marking negation or infinitives (e.g. "not", "to").
        "PRON",   # Pronoun: substitutes for a noun phrase or refers to discourse participants (e.g. "he", "they", "who").
        "SCONJ",  # Subordinating conjunction: introduces a subordinate clause (e.g. "because", "if", "although").
    }
)

# Universal Dependencies relations headed by a clause.
CLAUSAL_DEPS = frozenset(
    {
        "ROOT",   # Root of the sentence: the main predicate of the entire utterance.
        "ccomp",  # Clausal complement: finite or non-finite clause functioning as an argument with its own subject.
        "xcomp",  # Open clausal complement: argument clause whose subject is controlled by another argument.
        "advcl",  # Adverbial clause modifier: subordinate clause expressing time, reason, condition, purpose, etc.
        "acl",    # Clausal modifier of a noun: clause modifying a noun (e.g. participial or infinitival modifier).
        "relcl",  # Relative clause modifier: clause modifying a noun through relativization.
        "csubj",  # Clausal subject: clause functioning as the syntactic subject of a predicate.
    }
)
