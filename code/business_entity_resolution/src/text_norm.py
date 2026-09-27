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

# Web-address debris. `jgsreal.com` survives punctuation folding as the two
# tokens `jgsreal` and `com`, which is why the backlog showed `com jgsreal`
# against `jgs real`. Only unambiguous TLDs: `co` is already a legal
# abbreviation and `in`/`io` collide with ordinary words.
WEB_TOKENS: tuple[str, ...] = ("www", "com", "net", "org", "edu", "gov", "info", "biz")

# France-only overlays.
#
# France appears in the TEST split and nowhere in training, so nothing here can
# be measured against the ground truth — which is exactly why both tables are
# gated on the country rather than applied globally. A gated rule cannot change
# a single US or India record, so the worst case is that it does nothing, and
# the blast radius of a wrong entry is confined to the country it was written
# for. Applied globally, `ste -> sainte` alone would rewrite every US "Suite".
#
# Legal forms, dropped from `name_core` the way LEGAL_TOKENS are. Only pure
# form markers: `groupe`, `entreprise` and `freres` are NOT here because they
# distinguish one business from another.
FRENCH_LEGAL_TOKENS: tuple[str, ...] = (
    "societe", "ste", "soc", "sasu", "snc", "sci", "scp", "scm",
    "selarl", "selas", "selafa", "eurl", "eirl", "scs", "sca", "scop",
    "scic", "gie", "geie", "sem", "seml", "association", "assoc",
    "etablissements", "etablissement", "ets", "compagnie", "cie",
    "cooperative", "mutuelle", "syndicat",
    # The French `and`/`of`/`the`, which LEGAL_TOKENS already drops in
    # English. Without these, `Ets Martin et Cie` and `Martin` differ by a
    # token that carries no information.
    "et", "de", "des", "du", "la", "le", "les", "aux",
)

# Street-type and building abbreviations. `av`, `pl` and `sq` are already in
# ADDR_ABBREV and mean the same thing in both languages, so they are not
# repeated. `r` (rue), `all` (allee) and `rd` are deliberately absent: one
# letter is too short to disambiguate, `all` is an English word, and `rd`
# already means road.
#
# `ste` is an OVERRIDE, not an addition. ADDR_ABBREV maps it to `suite`, which
# is right for the US and wrong for France, where it is Sainte in a street
# name. The override only works because this table runs BEFORE the global one
# on French rows — `_expand` replaces whole tokens, so once the global pass has
# turned `ste` into `suite` there is nothing left for a later pass to match.
# Applying it second, which is what the first version did, silently left every
# French `Ste Anne` as `suite anne`.
FRENCH_ADDR_ABBREV: dict[str, str] = {
    "bd": "boulevard", "bld": "boulevard", "boul": "boulevard",
    "imp": "impasse",
    "che": "chemin", "chem": "chemin",
    "rte": "route",
    "fg": "faubourg", "fbg": "faubourg",
    "crs": "cours",
    "res": "residence", "resid": "residence",
    "bat": "batiment", "batim": "batiment",
    "appt": "appartement", "app": "appartement",
    "etg": "etage",
    "bp": "boite postale",
    "zi": "zone industrielle", "za": "zone activite",
    "rpt": "rond point",
    "lieudit": "lieu dit",
    "sainte": "sainte", "ste": "sainte",
}

# What `country_norm` looks like for France after `_base_clean`. Matched as a
# set rather than a substring, so a locality that merely contains the letters
# is unaffected.
FRANCE_LABELS: frozenset[str] = frozenset({"france", "fr", "fra"})


def _is_france(col: str = "country") -> pl.Expr:
    return _base_clean(pl.col(col)).is_in(list(FRANCE_LABELS))


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
    "cir": "circle", "cirs": "circle",
    # Street types found missing by reading the addresses of pairs that did
    # NOT match. `3028 Hidden Forest Cove` against `3028 HIDDEN FOREST CV`
    # differed on one three-letter abbreviation.
    "cv": "cove", "cvs": "cove",
    "trl": "trail", "xing": "crossing", "plz": "plaza",
    "ctr": "center", "cntr": "center",
    "grv": "grove", "grn": "green",
    "hts": "heights", "mnr": "manor", "vw": "view", "vly": "valley",
    "crk": "creek", "bnd": "bend", "lndg": "landing", "frst": "forest",
    "rdg": "ridge", "spg": "spring", "spgs": "spring", "mtn": "mountain",
    "pt": "point", "prk": "park", "cmn": "common", "clb": "club",
    "est": "estate", "ests": "estate", "trce": "trace", "psge": "passage",
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

# Tokens dropped to form `addr_place`: the locality-and-street core of an
# address, with everything that varies between two records of the same
# building removed.
#
# `addr_sorted` agrees on only 23.48% of true pairs while `addr_tokens` — the
# same field, one token at a time — agrees on 95.19%. The whole of that gap is
# tokens that one side carries and the other does not, and reading the missed
# pairs it is always the same two kinds:
#
#   1692 Hill Field Road, Unit Q, Layton, UT  vs  1692 HILL FIELD RD, LAYTON, UT
#   8080 River Road, South Whitley, IN        vs  8080 River Rd, S Whitley, Indiana
#
# a sub-premise marker, and the state written two ways. Neither identifies the
# building; both break equality. Country is already a hard partition, so the
# state carries almost no information inside it, and the unit is the one part
# of an address two records of the same business routinely disagree about.
SUBPREMISE_TOKENS: tuple[str, ...] = (
    "unit", "suite", "apartment", "floor", "room", "flat", "door", "gate",
    "wing", "box", "postal", "boite", "pmb", "cedex", "building", "bldg",
    "premises", "plot",
    # Administrative wrappers. `2697 Lavery Court, Thousand Oaks` against
    # `2697 LAVERY CT, CITY OF THOUSAND OAKS` differs by exactly one of
    # these, and `of` is already an address stopword.
    #
    # `city` is the one real trade here: it drops the distinction between
    # `Kansas City` and `Kansas`. It is dropped on BOTH sides, so it can only
    # merge two localities that already share every other token, inside one
    # country — and it is a blocking key, not a decision. The model still
    # sees the full `addr_core` through every address feature.
    "city", "town", "village", "district", "dist", "taluk", "tehsil",
    "mandal", "county", "province", "region", "state",
)

# US states, both spellings, plus DC. Written out rather than derived, because
# there is no list to derive it from and a wrong entry here is a silently
# looser key rather than an error.
US_STATE_TOKENS: tuple[str, ...] = (
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "ohio", "oklahoma", "oregon",
    "pennsylvania", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington", "wisconsin", "wyoming",
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id",
    "il", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo",
    "mt", "ne", "nv", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx",
    "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc",
    # Multi-word states arrive as separate tokens after `_base_clean`, and
    # `new`, `north`, `south` and `west` are all real street words — so only
    # the distinctive halves are listed. `hampshire`, `jersey`, `mexico` and
    # `dakota` do not appear in street names; `carolina` and `york` are rare
    # enough in one to be worth the trade.
    "hampshire", "jersey", "mexico", "dakota", "carolina", "york",
)

# Indian states and union territories, plus the abbreviations that appear in
# this data (`TG` for Telangana, `TN` for Tamil Nadu). `nadu` and `pradesh`
# are listed so the two-word names fall apart cleanly.
IN_STATE_TOKENS: tuple[str, ...] = (
    "andhra", "arunachal", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
    "haryana", "himachal", "jharkhand", "karnataka", "kerala", "keralam",
    "madhya", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "orissa", "punjab", "rajasthan", "sikkim", "tamil", "telangana",
    "tripura", "uttarakhand", "uttar", "bengal", "delhi", "puducherry",
    "pondicherry", "chandigarh", "ladakh", "lakshadweep", "andaman",
    "nicobar", "nadu", "pradesh",
    "tg", "tn", "ap", "ka", "kl", "mh", "up", "wb", "gj", "rj", "mp", "hr",
    "pb", "br", "od", "as", "jk", "hp", "uk", "cg", "ts",
)

# A handful of these (`ct`, `fl`, `mt`, `ne`) are also ADDR_ABBREV keys, and
# that is not a conflict: expansion runs BEFORE this drop, so `CT` is already
# `court` and is never considered as Connecticut. The effect is that those
# four states are not removed — applied identically to both sides of every
# pair, so equality is unaffected, and the alternative (expanding them as
# states) would break far more real street names than it fixed.
PLACE_STOPWORDS: tuple[str, ...] = (
    SUBPREMISE_TOKENS + US_STATE_TOKENS + IN_STATE_TOKENS
)


# --------------------------------------------------------------------------
# Expression builders
# --------------------------------------------------------------------------

_PUNCT = r"[^\p{L}\p{N}]+"

# Injected reference markers, stripped from the RAW name before punctuation
# folding. `(ID: 84887)` survives `_base_clean` as two ordinary tokens, `id`
# and `84887`, which then join `name_sorted`, `name_core`, `name_nospace`,
# every trigram and the acronym — one record in a true pair carries them and
# the other does not, so every name key disagrees on a pair whose names are
# otherwise identical. Four of the first twenty `no-key-agrees` misses read
# this way.
#
# Only mechanical markers are listed. `formerly:`, `née` and `Dr` also appear
# in the data and are NOT here: those are real variation in how a business is
# written, and dropping them is a normalization judgement rather than removing
# an artefact.
#
# Applied before punctuation folding because the brackets and the colon are
# what identify the marker; after folding there is nothing left to match on
# but a bare number, and bare numbers are load-bearing in this data.
NOISE_PATTERNS: tuple[str, ...] = (
    # (ID: 84887) / [id 84887] / (Id#84887)
    r"(?i)[\(\[\{]\s*id\s*[:#]?\s*\d+\s*[\)\]\}]",
    # a trailing reference number: "Corp #26282"
    r"#\s*\d{3,}",
)


def _strip_noise(col: pl.Expr) -> pl.Expr:
    for pattern in NOISE_PATTERNS:
        col = col.str.replace_all(pattern, " ")
    return col


# Minimum length for a token to become a blocking value of its own.
#
# Short tokens are the ones whose buckets are enormous — a two-letter token has
# a few hundred possible values across millions of records — and they are also
# the ones that carry least evidence. This is the `name_acronym --min-len`
# argument applied at the source rather than as a cap: drop them before
# bucketing instead of capping the buckets they create. Four keeps `shaw`,
# `yale`, `boag` and `erin`, which are exactly the rare tokens the token keys
# exist to find.
TOKEN_MIN_LEN = 4

# Characters of `name_nospace` that form `name_prefix`.
#
# Twelve is long enough that a shared prefix is evidence rather than a
# coincidence, and short enough to survive a trailing difference. It is the
# only key that reaches a name concatenated into a domain — `assetbuildingfund`
# is a SINGLE token, so no token key can match it against `asset building fund
# care`, while both begin `assetbuildin`.
PREFIX_LEN = 12
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
    """Apply whole-token replacements from `table`.

    Done as one tokenize / map / re-join rather than one `replace_all` pass per
    entry. The two tables hold ~95 entries between them, and a regex pass each
    costs ~25s per million rows — about five minutes of the runtime at full
    scale, paid on every run. `Expr.replace` is a single hash lookup per token
    and leaves unmatched tokens alone, which is exactly whole-token semantics.

    Verified bit-identical to the previous per-entry `\\b{short}\\b` loop over a
    large random corpus, with one dead entry removed: `LEGAL_ABBREV` had a
    multi-word key `"l l c"`, which `_glue_initials` has already collapsed to
    `llc` before this function ever sees it, so no token could match it.
    """
    assert all(" " not in short for short in table), \
        "table keys must be single tokens; _glue_initials runs before this"
    return (
        col.str.split(" ")
        .list.eval(pl.element().replace(table))
        .list.join(" ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def _drop_tokens(col: pl.Expr, tokens: tuple[str, ...]) -> pl.Expr:
    pattern = r"\b(?:" + "|".join(tokens) + r")\b"
    return (
        col.str.replace_all(pattern, " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def _long_tokens(col: pl.Expr, drop_numeric: bool = False) -> pl.Expr:
    """Unique tokens of at least `TOKEN_MIN_LEN` characters, as a list.

    The value of a token key is not any single token but the SET of them: two
    records are candidates if they share one rare token, wherever it sits in
    either name. That is what reaches a pair whose names agree on nothing else
    — a transliterated name that kept one Latin word, an address that kept one
    unusual locality — and it is why this returns a list rather than a string.

    No inverse-document-frequency filter here, deliberately. A token's bucket
    size IS its document frequency, so the bucket cap blocking already applies
    to every key is exactly an IDF cut, computed against the real corpus at
    block time rather than guessed at here. `private` lands in a bucket of
    millions and is dropped; `vaishnavisignature` lands in a bucket of two and
    survives. Adding a second frequency rule here would be the same cut in a
    second place, tuned separately, against a different corpus.
    """
    keep = pl.element().str.len_chars() >= TOKEN_MIN_LEN
    if drop_numeric:
        # Pure digits are covered by `addr_nums`, which applies its own length
        # rule. Mixed tokens like `4b` or `7b` are NOT dropped here — they are
        # dropped by the length rule above, which is a different decision and
        # the reason this does not reuse `_sorted_tokens`.
        keep = keep & ~pl.element().str.contains(r"^\d+$")
    return (
        col.str.split(" ")
        .list.eval(pl.element().filter(keep))
        .list.unique()
    )


def _sorted_tokens(
    col: pl.Expr, drop_numeric: bool = False, min_len: int = 0
) -> pl.Expr:
    """Order-invariant token signature: unique tokens, sorted, space-joined.

    Collapses word-order transpositions ("Sharma Traders" / "Traders Sharma").

    `drop_numeric` removes tokens that are *entirely* digits — those are already
    captured in `addr_numbers`. Mixed tokens like `4b` or `5th` are kept, since
    a flat digit-strip would turn `4b` into `b` and destroy a strong signal.
    """
    keep = pl.element().str.len_chars() > 0
    if min_len > 1:
        keep = keep & (pl.element().str.len_chars() >= min_len)
    if drop_numeric:
        keep = keep & ~pl.element().str.contains(r"^\d+$")
    return (
        col.str.split(" ")
        .list.eval(pl.element().filter(keep))
        .list.unique()
        .list.sort()
        .list.join(" ")
    )


# Character 3-grams are the feature stage's answer to the 23.74% of true pairs
# that no exact key reaches. Those failures are character-level by construction
# — typos (`fnesnius`/`fresenius`), concatenation (`dentcoffee`/`coffee dent`),
# transliteration residue (`kanstraksans`/`constructions`), web debris (`com
# jgsreal`/`jgs real`) — so they share most of their 3-grams while sharing no
# token exactly.
#
# Built here, at normalization time, rather than in the feature stage. The
# alternative is recomputing them for every candidate pair, and an entity
# appears in ~10.7 candidates on average, so per-entity work is an order of
# magnitude cheaper than per-pair work. Normalization is already chunked and
# already cached, which is exactly what this needs.
#
# WHY THIS SHAPE: polars has no windowed string operation and the regex crate
# has no lookaround, so overlapping n-grams cannot be produced by
# `extract_all`. Unrolling into fixed slices is the vectorized way to do it —
# `concat_list` of N slice expressions, then drop the partials. The cost is one
# transient list of N elements per row, which is why this must stay inside the
# chunked normalization pass and not be run over a whole source at once.
#
# The cap TRUNCATES: a `name_core` longer than this contributes no 3-grams past
# the cut. Business names run to a median of 2-3 tokens, so 48 characters
# covers essentially all of them, and truncation degrades the feature for the
# tail rather than corrupting it.
TRIGRAM_MAX_CHARS = 48

# 3-grams are stored HASHED to UInt32, not as strings. Two reasons, both about
# the 12.5M-row scale rather than about elegance:
#
# * memory — a resident List(String) of ~25 short strings per row costs roughly
#   2.5GB across a split once per-string overhead is counted, against ~1.25GB
#   for List(UInt32). The feature stage holds a target-side attribute table and
#   a join output at the same time, and 16GB notices the difference.
# * speed — `list.set_intersection` over integers is materially cheaper than
#   over strings, and it runs on all 23.6M candidate pairs.
#
# Collisions are not a practical concern here: the measured 3-gram vocabulary
# over this corpus is 15,285 distinct grams, so the expected number of 32-bit
# collisions across the whole vocabulary is about 0.03.
#
# The hash is only ever compared against other hashes produced by the SAME
# polars build, which is why `_is_fresh` in `main` now refuses a cache written
# by a different polars version. Mixing versions across the three sources would
# make every cross-source trigram comparison silently return zero overlap.
TRIGRAM_HASH_SEED = 0x9E3779B9


def trigram_expression(col: str = "name_nospace") -> pl.Expr:
    """Overlapping character 3-grams of a column, as a unique list of UInt32 hashes.

    Takes a COLUMN NAME, not an expression, and that is the whole point.

    The first version of this took an expression and was applied directly on
    top of the `name_core` chain, which costs a `replace_many` accent pass, an
    initialism glue, a token-wise abbreviation expansion and a legal-token
    strip. Building 48 slices off that expression made polars recompute the
    entire chain ONCE PER SLICE: measured 246.7s per million rows against
    24.8s when the chain is materialized first. At 12.5M rows per split that
    is 52 minutes of normalization instead of 5.

    Requiring a column name makes the mistake unrepresentable — a caller has to
    have materialized the input before it can name it. `main.prepare_full`
    applies this in a second pass over `name_nospace`, which is exactly the
    string these grams are wanted on.
    """

    padded = pl.concat_str([pl.lit(" "), pl.col(col).fill_null(""), pl.lit(" ")])
    # Padded with a leading and trailing space: it gives the first and last
    # real characters a boundary 3-gram, so `acme` and `acmex` differ at the
    # end as well as the start, and a string shorter than three characters
    # still produces grams instead of an empty list. The padded length is
    # `len + 2`, so a string of length n yields exactly n grams and
    # `range(TRIGRAM_MAX_CHARS)` covers inputs up to that many characters.
    grams = []
    for i in range(TRIGRAM_MAX_CHARS):
        g = padded.str.slice(i, 3)
        grams.append(
            # Positions past the end yield "" or a 1-2 character remainder.
            # Nulling those and dropping them afterwards is what makes this a
            # clean n-gram set rather than a ragged one.
            pl.when(g.str.len_chars() == 3)
            # The modulo is deliberate rather than a plain cast: polars `hash`
            # returns UInt64, and a strict narrowing cast to UInt32 errors on
            # overflow while a non-strict one yields NULL — which would poison
            # the set operations silently. Taking the low 32 bits explicitly is
            # defined behaviour for every input.
            .then((g.hash(seed=TRIGRAM_HASH_SEED) % 4294967296).cast(pl.UInt32))
            .otherwise(None)
        )
    # `concat_list` + `drop_nulls` rather than `list.eval(...filter...)`.
    # `list.eval` runs a per-row sub-query; this is 48 vectorized column
    # operations instead, and measured 9.0s per million rows against 12.7s.
    # Output is bit-identical — verified over a random corpus.
    #
    # `unique` is NOT optional: `features.py` computes union size as
    # `|A| + |B| - |A and B|`, an identity that holds for sets and fails for
    # multisets.
    return pl.concat_list(grams).list.drop_nulls().list.unique()


def _acronym(col: pl.Expr) -> pl.Expr:
    """First letter of each token, e.g. 'international business machines' -> 'ibm'.

    Lets a later stage link an expanded name to its initialism.
    """
    return (
        col.str.extract_all(r"\b\p{L}")
        .list.join("")
    )


def name_expressions(
    col: str = "business_name", strip_noise: bool = True
) -> list[pl.Expr]:
    """Every name-derived column, from the raw name.

    `strip_noise` removes injected reference markers before anything else —
    see `NOISE_PATTERNS`. It is a parameter rather than unconditional because
    it moves every existing key at once, so a regression has to stay
    attributable to it; `main --no-strip-noise` reproduces the previous
    normalization exactly.
    """
    raw = pl.col(col)
    if strip_noise:
        raw = _strip_noise(raw)
    base = _base_clean(raw)
    norm = _expand(base, LEGAL_ABBREV)
    core = _drop_tokens(norm, LEGAL_TOKENS + WEB_TOKENS)
    # France-only second strip. Both branches are computed and then selected,
    # which is what a conditional expression costs; the alternative is a
    # separate pass over a filtered frame, and at 12.5M rows the branch is
    # cheaper than the split and re-concatenation.
    core = (
        pl.when(_is_france())
        .then(_drop_tokens(core, FRENCH_LEGAL_TOKENS))
        .otherwise(core)
    )
    # If stripping legal tokens empties the name, fall back to the full form.
    # Checked AFTER the French strip, so a French company whose entire name is
    # its legal form still gets the fallback rather than an empty key.
    core = pl.when(core.str.len_chars() == 0).then(norm).otherwise(core)
    nospace = core.str.replace_all(r"\s+", "")
    return [
        norm.alias("name_norm"),
        core.alias("name_core"),
        _sorted_tokens(core).alias("name_sorted"),
        nospace.alias("name_nospace"),
        # CHARACTER anagram: despace the whole name, then sort its characters.
        # `enterprises acme` -> `aceeeeimnprrsst`.
        #
        # This is not a sorted-token key with the spaces taken out — that
        # would give `acmeenterprises` and would still require the two sides
        # to tokenize the same way. Sorting the characters removes the token
        # boundaries entirely, which is what reaches the failure class nothing
        # else does: a name concatenated in a different order.
        # `Combs and Silverman Navigation` and `navigationcombssilverman.com`
        # share no token-level key and have a trigram Jaccard of 0.71 — below
        # the LSH bands' reach — and they are exact anagrams of each other.
        #
        # It subsumes a token-sorted-despaced key (equal token multisets imply
        # equal character multisets), so only this one is kept.
        #
        # The cost is collisions: every anagram of a name is now one bucket,
        # so this is the lossiest name key in the set. It is priced the way
        # every other key is — the bucket cap drops the values that collide
        # with everything, and the inverse-frequency term in `block_score`
        # demotes what survives.
        #
        # `str.split("")` splits into single characters; `str.extract_all(".")`
        # is equivalent and equally fast (both measured at 45s per 12.5M rows,
        # and verified to agree on the same inputs).
        nospace.str.split("").list.sort().list.join("")
        .alias("name_charsort"),
        _acronym(core).alias("name_acronym"),
        core.str.split(" ").list.len().cast(pl.UInt16).alias("name_token_count"),
        _long_tokens(core).alias("name_tokens"),
        # Null below the length, rather than the whole short name. A name
        # shorter than the prefix would otherwise produce a key identical to
        # `name_nospace`, which is already a key: every pair it found would be
        # found twice, and the only effect would be doubling this key's
        # fan-out and its weight in `block_score`.
        pl.when(nospace.str.len_chars() >= PREFIX_LEN)
        .then(nospace.str.slice(0, PREFIX_LEN))
        .otherwise(None)
        .alias("name_prefix"),
    ]
    # NOTE: `name_trigrams` is deliberately NOT built here. It is applied in a
    # second pass over the materialized `name_nospace` column — see
    # `trigram_expression` for the 10x reason.


def address_expressions(col: str = "business_address") -> list[pl.Expr]:
    base = _base_clean(pl.col(col))
    # France-only overlay, applied BEFORE the global table so its `ste` entry
    # is seen while the token is still `ste`. See FRENCH_ADDR_ABBREV.
    norm = (
        pl.when(_is_france())
        .then(_expand(_expand(base, FRENCH_ADDR_ABBREV), ADDR_ABBREV))
        .otherwise(_expand(base, ADDR_ABBREV))
    )
    core = _drop_tokens(norm, ADDR_STOPWORDS)
    digits = pl.col(col).fill_null("").str.extract_all(r"\d+")
    return [
        norm.alias("addr_norm"),
        core.alias("addr_core"),
        _sorted_tokens(core, drop_numeric=True).alias("addr_sorted"),
        # The street-and-locality core: `addr_sorted` with sub-premise markers
        # and state names removed, and single characters dropped (a bare `q`
        # is a unit label, never a place).
        #
        # A NEW key rather than a change to `addr_sorted` or `addr_core`,
        # deliberately. Those two feed the scorer's address features, and
        # loosening them would move every address feature the model was fitted
        # on at the same time as adding reach — two changes in one, neither
        # attributable. This only adds a key.
        _sorted_tokens(
            _drop_tokens(core, PLACE_STOPWORDS), drop_numeric=True, min_len=2
        ).alias("addr_place"),
        digits.alias("addr_numbers"),
        _long_tokens(core, drop_numeric=True).alias("addr_tokens"),
        # House and postal numbers as blocking values in their own right.
        # Distinct from `addr_numbers`, which keeps every run of digits
        # including the single-character ones: `2` is in half the addresses in
        # the corpus and blocking on it is a full scan of the country.
        digits.list.eval(
            pl.element().filter(
                (pl.element().str.len_chars() >= 3)
                & (pl.element().str.len_chars() <= 8)
            )
        ).list.unique().alias("addr_nums"),
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
