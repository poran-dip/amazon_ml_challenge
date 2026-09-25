"""Normalization rule tables and the polars expressions that apply them.

Everything here is vectorized polars string work — no Python UDFs — because the
full dataset is ~10M rows per split and a per-row Python call would dominate
runtime.

Design notes
------------
* Country is an OPEN SET. The test set contains France, which never appears in
  training, so nothing below branches on a country value. Postal codes are
  extracted with a generic digit-run rule rather than per-country regexes.
* Rules are deliberately conservative: this stage only collapses variants that
  are *always* the same thing (Rd/Road, Pvt/Private). Anything judgement-based
  (dropping landmarks, guessing cities) is left to the feature stage, where a
  model can weigh it instead of the rules destroying information.
* Every rule table is a plain module-level dict so iteration on step 1 means
  editing data, not code.
"""

from __future__ import annotations

import polars as pl

# --------------------------------------------------------------------------
# Rule tables — edit these to iterate on normalization quality.
# Keys are regex fragments matched on whole tokens; values are replacements.
# --------------------------------------------------------------------------

# Legal form abbreviations -> canonical long form. Applied to names only.
LEGAL_ABBREV: dict[str, str] = {
    "pvt": "private",
    "pte": "private",
    "ltd": "limited",
    "ltda": "limited",
    "inc": "incorporated",
    "corp": "corporation",
    "corpn": "corporation",
    "co": "company",
    "cos": "company",
    "llc": "llc",
    "l l c": "llc",
    "llp": "llp",
    "plc": "plc",
    "intl": "international",
    "intnl": "international",
    "assn": "association",
    "assoc": "associates",
    "bros": "brothers",
    "mfg": "manufacturing",
    "svcs": "services",
    "svc": "services",
    "tech": "technologies",
    "techs": "technologies",
    "ind": "industries",
    "inds": "industries",
    "ent": "enterprises",
    "ents": "enterprises",
    "dept": "department",
    "univ": "university",
    "natl": "national",
    "mkt": "market",
    "mktg": "marketing",
}

# Tokens stripped to form `name_core`. Pure legal-form markers only: words like
# "group" or "holdings" are NOT here because they distinguish real businesses.
LEGAL_TOKENS: tuple[str, ...] = (
    "private", "limited", "incorporated", "corporation", "company", "llc",
    "llp", "plc", "gmbh", "mbh", "ag", "sarl", "sas", "sa", "srl", "spa",
    "bv", "nv", "oy", "ab", "as", "aps", "kft", "sdn", "bhd", "pty",
    "the", "and", "of",
    # Romanized Indic spellings of the same legal words. `प्राइवेट लिमिटेड` is
    # the English "private limited" written phonetically, so after
    # `translit.romanize` it arrives as praivet/limiteda rather than as any of
    # the forms above. Variants cover transliterator output plus schwa handling.
    "praivet", "praiveta", "praivate", "prive", "praibhet",
    "limiteda", "limitad", "limitet", "limted",
    "kampani", "kampeni", "korporesan", "inkarporeted",
)

# Street / locality abbreviations -> long form. Applied to addresses only.
ADDR_ABBREV: dict[str, str] = {
    "rd": "road", "rds": "road",
    "st": "street", "str": "street",
    "ave": "avenue", "av": "avenue", "avn": "avenue",
    "blvd": "boulevard", "bldv": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "ter": "terrace",
    "hwy": "highway",
    "pkwy": "parkway", "pky": "parkway",
    "cir": "circle",
    "apt": "apartment", "apts": "apartment",
    "ste": "suite",
    "fl": "floor", "flr": "floor",
    "bldg": "building", "bldgs": "building",
    "blk": "block",
    "nr": "near",
    "opp": "opposite",
    "behd": "behind",
    "jn": "junction", "jnc": "junction", "jct": "junction",
    "sec": "sector", "sect": "sector",
    "ph": "phase",
    "ext": "extension",
    "mg": "mahatma gandhi",
    "no": "number", "nos": "number",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest",
    "po": "post office",
    "dist": "district",
    "vlg": "village",
    "twp": "township",
    "mt": "mount",
    "ft": "fort",
    "pk": "park",
    "gdn": "garden", "gdns": "garden",
    "hts": "heights",
    "mkt": "market",
    "cmplx": "complex",
    "indl": "industrial",
    "estd": "estate",
}

# Locality noise words that carry no matching signal on their own. Kept in
# `addr_norm` but removed from `addr_core` so token-overlap features are not
# inflated by them.
ADDR_STOPWORDS: tuple[str, ...] = (
    "near", "opposite", "behind", "beside", "next", "to", "the", "at",
    "in", "on", "of", "and", "number",
)

# --------------------------------------------------------------------------
# Expression builders
# --------------------------------------------------------------------------

_PUNCT = r"[^\p{L}\p{N}]+"
# ONLY the Latin combining-diacritics block (U+0300-U+036F), never `\p{M}`.
# In Indic scripts the vowel signs and virama are themselves marks, so a blanket
# `\p{M}` strip deletes the vowels: डेवलपर्स -> डवलपरस, 33 chars down to 22.
# This range also covers the dot-below/macron used by IAST transliteration.
_COMBINING = r"[̀-ͯ]+"

# Precomposed accented characters -> ASCII. Applied as one Aho-Corasick pass
# (`str.replace_many`), which is a single scan regardless of table size. The
# `\p{M}` regex above additionally handles already-decomposed input.
_ACCENT_SRC = (
    "àáâãäåāăą" "èéêëēĕėęě" "ìíîïĩīĭįı" "òóôõöøōŏő" "ùúûüũūŭůűų"
    "çćĉċč" "ñńņňŉ" "ýÿŷ" "šśŝşș" "žźżž" "ďđ" "ğĝġģ" "ĥħ" "ĵ" "ķ"
    "ĺļľłŀ" "ŕŗř" "ţťț" "ŵ"
)
_ACCENT_DST = (
    "aaaaaaaaa" "eeeeeeeee" "iiiiiiiii" "ooooooooo" "uuuuuuuuuu"
    "ccccc" "nnnnn" "yyy" "sssss" "zzzz" "dd" "gggg" "hh" "j" "k"
    "lllll" "rrr" "ttt" "w"
)
assert len(_ACCENT_SRC) == len(_ACCENT_DST), "accent map is misaligned"
ACCENT_MAP: dict[str, str] = dict(zip(_ACCENT_SRC, _ACCENT_DST))
# Multi-character expansions kept separate (source and target differ in length).
ACCENT_MAP.update(
    {"æ": "ae", "œ": "oe", "ß": "ss", "þ": "th", "ð": "d", "ĳ": "ij"}
)
# IAST characters produced by `translit.romanize`, folded to bare ASCII. The
# library emits these precomposed, so the combining-mark regex alone misses them.
ACCENT_MAP.update({
    "ā": "a", "ī": "i", "ū": "u", "ē": "e", "ō": "o",
    "ṛ": "r", "ṝ": "r", "ṟ": "r", "ṙ": "r",
    "ḷ": "l", "ḹ": "l", "ḻ": "l", "ḽ": "l",
    "ṃ": "m", "ṁ": "m", "ṅ": "n", "ñ": "n", "ṇ": "n", "ṉ": "n", "ṋ": "n",
    "ḥ": "h", "ḫ": "h",
    "ṭ": "t", "ṯ": "t", "ṱ": "t",
    "ḍ": "d", "ḏ": "d", "ḑ": "d",
    "ś": "s", "ṣ": "s", "ṡ": "s",
    "ḳ": "k", "ḵ": "k", "ġ": "g", "ḡ": "g", "ḃ": "b", "ṗ": "p", "ṽ": "v",
})
_ACCENT_KEYS = list(ACCENT_MAP)
_ACCENT_VALS = [ACCENT_MAP[k] for k in _ACCENT_KEYS]


# Runs of single letters are glued back into one token: "S.B.I." -> "sbi",
# "M.G. Road" -> "mg road", "S.A.R.L." -> "sarl". This MUST run before the
# abbreviation tables, otherwise a lone "s" in "S B I" is expanded to "south".
# The regex crate has no lookaround, so fixed-width patterns are applied
# longest-first; two passes of the 2-letter rule mop up longer runs.
_GLUE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(^|\s)(\p{L}) (\p{L}) (\p{L}) (\p{L})($|\s)", "$1$2$3$4$5$6"),
    (r"(^|\s)(\p{L}) (\p{L}) (\p{L})($|\s)", "$1$2$3$4$5"),
    (r"(^|\s)(\p{L}) (\p{L})($|\s)", "$1$2$3$4"),
    (r"(^|\s)(\p{L}) (\p{L})($|\s)", "$1$2$3$4"),
)


def _glue_initials(col: pl.Expr) -> pl.Expr:
    for pattern, repl in _GLUE_PATTERNS:
        col = col.str.replace_all(pattern, repl)
    return col


def _base_clean(col: pl.Expr) -> pl.Expr:
    """Lowercase, strip accents/diacritics, fold punctuation to spaces.

    Folding accents turns transliteration variants like `Café`/`Cafe` and
    `Bengalūru`/`Bengaluru` into the same string.
    """
    return (
        col.fill_null("")
        .str.to_lowercase()
        .str.replace_many(_ACCENT_KEYS, _ACCENT_VALS)
        .str.replace_all(_COMBINING, "")
        # & -> and, before punctuation folding eats the ampersand
        .str.replace_all(r"&", " and ")
        # drop apostrophes so o'reilly == oreilly (no space inserted)
        .str.replace_all(r"['‘’ʼ]", "")
        .str.replace_all(_PUNCT, " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .pipe(_glue_initials)
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def _expand(col: pl.Expr, table: dict[str, str]) -> pl.Expr:
    """Apply whole-token replacements from `table`."""
    for short, long in table.items():
        col = col.str.replace_all(rf"\b{short}\b", long)
    return col.str.replace_all(r"\s+", " ").str.strip_chars()


def _drop_tokens(col: pl.Expr, tokens: tuple[str, ...]) -> pl.Expr:
    pattern = r"\b(?:" + "|".join(tokens) + r")\b"
    return (
        col.str.replace_all(pattern, " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def _sorted_tokens(col: pl.Expr, drop_numeric: bool = False) -> pl.Expr:
    """Order-invariant token signature: unique tokens, sorted, space-joined.

    Collapses word-order transpositions ("Sharma Traders" / "Traders Sharma").

    `drop_numeric` removes tokens that are *entirely* digits — those are already
    captured in `addr_numbers`. Mixed tokens like `4b` or `5th` are kept, since
    a flat digit-strip would turn `4b` into `b` and destroy a strong signal.
    """
    keep = pl.element().str.len_chars() > 0
    if drop_numeric:
        keep = keep & ~pl.element().str.contains(r"^\d+$")
    return (
        col.str.split(" ")
        .list.eval(pl.element().filter(keep))
        .list.unique()
        .list.sort()
        .list.join(" ")
    )


def _acronym(col: pl.Expr) -> pl.Expr:
    """First letter of each token, e.g. 'international business machines' -> 'ibm'.

    Lets a later stage link an expanded name to its initialism.
    """
    return (
        col.str.extract_all(r"\b\p{L}")
        .list.join("")
    )


def name_expressions(col: str = "business_name") -> list[pl.Expr]:
    base = _base_clean(pl.col(col))
    norm = _expand(base, LEGAL_ABBREV)
    core = _drop_tokens(norm, LEGAL_TOKENS)
    # If stripping legal tokens empties the name, fall back to the full form.
    core = pl.when(core.str.len_chars() == 0).then(norm).otherwise(core)
    return [
        norm.alias("name_norm"),
        core.alias("name_core"),
        _sorted_tokens(core).alias("name_sorted"),
        core.str.replace_all(r"\s+", "").alias("name_nospace"),
        _acronym(core).alias("name_acronym"),
        core.str.split(" ").list.len().cast(pl.UInt16).alias("name_token_count"),
    ]


def address_expressions(col: str = "business_address") -> list[pl.Expr]:
    base = _base_clean(pl.col(col))
    norm = _expand(base, ADDR_ABBREV)
    core = _drop_tokens(norm, ADDR_STOPWORDS)
    digits = pl.col(col).fill_null("").str.extract_all(r"\d+")
    return [
        norm.alias("addr_norm"),
        core.alias("addr_core"),
        _sorted_tokens(core, drop_numeric=True).alias("addr_sorted"),
        digits.alias("addr_numbers"),
        # Generic postal-code candidates: 4-6 digit runs. Covers US (5),
        # India (6) and France (5) without hard-coding any country.
        digits.list.eval(
            pl.element().filter(
                (pl.element().str.len_chars() >= 4) & (pl.element().str.len_chars() <= 6)
            )
        ).alias("addr_postal"),
        core.str.split(" ").list.len().cast(pl.UInt16).alias("addr_token_count"),
    ]


def country_expression(col: str = "country") -> pl.Expr:
    """Country is treated as an opaque label: cleaned, never enumerated."""
    return _base_clean(pl.col(col)).alias("country_norm")
