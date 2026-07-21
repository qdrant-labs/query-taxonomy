
DEFAULT_SPACY_MODEL = "en_core_web_sm"

# Universal Dependencies (UD) closed-class part-of-speech tags.
# Closed-class categories contain a small, relatively fixed inventory of words.
CLOSED_CLASS = frozenset(
    {
        "ADP",    # Adposition: prepositions/postpositions expressing grammatical relations (e.g. "in", "to", "with").
        "AUX",    # Auxiliary verb: grammatical verb marking tense, aspect, mood, voice, or polarity (e.g. "is", "have", "will").
        "CCONJ",  # Coordinating conjunction: links elements of equal syntactic status (e.g. "and", "or", "but").
        "DET",    # Determiner: specifies or limits a noun (e.g. "the", "this", "some", "each").
        "NUM",    # Numeral: cardinal or other numeric expression functioning as a number (e.g. "three", "42").
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
