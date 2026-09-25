"""Romanization of Indic-script records.

Why this exists
---------------
EDA on the training set showed the same business appearing in different
scripts: `international systems` in one source and `ఇంటర్నేషనల్ ... సిస్టమ్స్` in
another. No token- or character-level similarity can bridge that, so those true
pairs are unreachable for *any* blocking strategy until both sides live in the
same alphabet.

The Indic strings are not translations — they are the English words spelled
phonetically. `लिमिटेड` is "limited", not the Hindi word for it. So romanizing
lands within fuzzy-match distance of the Latin-script version:

    डेवलपर्स   -> devalapars   (developers)
    प्राइवेट   -> praivet      (private)
    लिमिटेड    -> limited      (limited)
    సిస్టమ్స్   -> sistams      (systems)

Fair play
---------
This is a deterministic character mapping — the same class of operation as
lowercasing. No external database, API, or data source is consulted, and nothing
here knows anything about any business. `indic-transliteration` is an MIT-licensed
pure-Python library of script tables.

Performance
-----------
The library is per-string Python, which cannot be run over ~10M rows directly.
Two things keep it cheap: a vectorized regex pre-filter means only rows actually
containing Indic characters are touched at all, and results are memoized on the
unique input strings, which collapse heavily in this dataset.
"""

from __future__ import annotations

import re

import polars as pl

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Unicode block -> source scheme. Ordered; first hit wins.
SCRIPT_BLOCKS: tuple[tuple[int, int, str], ...] = (
    (0x0900, 0x097F, sanscript.DEVANAGARI),
    (0x0980, 0x09FF, sanscript.BENGALI),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI),
    (0x0A80, 0x0AFF, sanscript.GUJARATI),
    (0x0B00, 0x0B7F, sanscript.ORIYA),
    (0x0B80, 0x0BFF, sanscript.TAMIL),
    (0x0C00, 0x0C7F, sanscript.TELUGU),
    (0x0C80, 0x0CFF, sanscript.KANNADA),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM),
)

# Vectorized detector — cheap enough to run over every row.
INDIC_RE = r"[ऀ-ൿ]"

# Splits a mixed string into Indic and non-Indic runs so Latin text passes
# through untouched instead of being fed to the transliterator.
_RUN_RE = re.compile(r"[ऀ-ൿ]+")

# Indic abugidas carry an inherent 'a' that Hindi and most modern Indic
# languages do not pronounce word-finally. The transliterator is Sanskrit-
# oriented and keeps it, so `limiteda` needs one more step to reach `limited`.
_TRAILING_SCHWA_RE = re.compile(r"(?<=[^aeiou\s])a\b")

_cache: dict[str, str] = {}


def detect_script(text: str) -> str | None:
    """Scheme name for the first Indic character found, or None."""
    for ch in text:
        code = ord(ch)
        for start, end, scheme in SCRIPT_BLOCKS:
            if start <= code <= end:
                return scheme
    return None


def _romanize_run(run: str) -> str:
    scheme = detect_script(run)
    if scheme is None:
        return run
    try:
        out = transliterate(run, scheme, sanscript.IAST)
    except Exception:
        # A malformed run is not worth failing a 10M-row pass over.
        return run
    return _TRAILING_SCHWA_RE.sub("", out)


def romanize(text: str) -> str:
    """Replace Indic runs in `text` with IAST; leave everything else alone.

    Output still carries IAST diacritics (ṭ, ś, ā); `text_norm.ACCENT_MAP` folds
    those to ASCII in the normal cleaning path.
    """
    if not text:
        return text
    cached = _cache.get(text)
    if cached is not None:
        return cached
    out = _RUN_RE.sub(lambda m: _romanize_run(m.group(0)), text)
    if len(_cache) < 2_000_000:
        _cache[text] = out
    return out


def romanize_series(series: pl.Series) -> pl.Series:
    """Romanize a Utf8 series, touching only rows that contain Indic characters.

    Deduplicates before calling into Python: the same business name recurs many
    times across a 10M-row file, and the transliterator is the slow part.
    """
    if series.len() == 0:
        return series
    mask = series.fill_null("").str.contains(INDIC_RE)
    if not mask.any():
        return series

    targets = series.filter(mask).unique().drop_nulls()
    mapping = pl.DataFrame({
        "_src": targets,
        "_dst": pl.Series([romanize(t) for t in targets], dtype=pl.Utf8),
    })
    joined = (
        pl.DataFrame({"_src": series})
        .join(mapping, on="_src", how="left")
        .select(pl.coalesce(pl.col("_dst"), pl.col("_src")))
    )
    return joined.to_series().rename(series.name)


def indic_flag_expr(*columns: str) -> pl.Expr:
    """True when any of `columns` contained Indic script before romanization.

    Kept as a column because it is a useful model feature later: a cross-script
    pair is a different matching regime from a same-script one.
    """
    expr = pl.lit(False)
    for col in columns:
        expr = expr | pl.col(col).fill_null("").str.contains(INDIC_RE)
    return expr
