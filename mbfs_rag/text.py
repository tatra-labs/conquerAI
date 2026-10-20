"""Text normalisation shared by indexing and the answerability gate.

One tokenizer serves BM25 and the coverage check, so "a query term is covered by the evidence" means
the same thing in both places. (The eval harness deliberately does NOT reuse it: its matchers are
independent, so a normaliser bug cannot hide itself.)
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from py_rust_stemmers import SnowballStemmer

_STEMMER = SnowballStemmer("english")

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had has
    have having he her here hers him his how i if in into is it its itself just let me more most my
    myself no nor not now of off on once only or other our ours out over own same she should so some
    such than that the their theirs them then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your yours per via
    don doesn didn isn aren wasn weren won wouldn shouldn couldn haven hasn hadn ain
    """.split()
)

# Conversational filler that carries no information. Written as a closed list before any eval query.
FILLER_PHRASES = (
    "quick question",
    "quick q",
    "i was wondering",
    "was wondering",
    "could you tell me",
    "can you tell me",
    "can you please",
    "could you please",
    "do you know",
    "i need to know",
    "i want to know",
    "i'd like to know",
    "thanks so much",
    "thank you so much",
    "thanks in advance",
    "thank you",
    "so much",
)
FILLER_WORDS = frozenset("hi hey hello thanks thx ty pls plz please guys folks ok okay um uh hmm lol".split())

# Words that ask *what kind* of answer is wanted rather than *what about*; never a coverage gap.
GENERIC_WORDS = frozenset(
    """
    much many long take takes taking get gets getting got need needs want wants know tell help use using
    work works happen happens happened make makes keep keeps go goes going come new old current same other
    different extra additional first last next soon quick quickly fast exact exactly typical usually
    please way thing things time times really actually right someone something anything plan plans account
    cost costs price prices pricing charge charges charged pay paying
    still even already today tomorrow yesterday now yet again ever only maybe perhaps probably
    """.split()
)

# Brand and plan-family names appear on every page, so they never carry a question's meaning.
NON_SALIENT = frozenset({"merchantbridg", "mbfs", "bridg"})

_NUMBER_RE = r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?"
_WORD_RE = r"[a-z]+(?:'[a-z]+)?(?:-[a-z]+)*"
_TOKEN_RE = re.compile(f"{_NUMBER_RE}|{_WORD_RE}")
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_QUOTES = dict.fromkeys(map(ord, "‘’‛′"), "'")


def fold_typography(text: str) -> str:
    """Curly apostrophes and typographic dashes as typed on phones, Word and Outlook -> ASCII."""
    return text.translate(_DASHES).translate(_QUOTES)


def normalize(text: str) -> str:
    """Drop accents ("Générale" -> "generale"), NFKC, fold typography, lower-case."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return fold_typography(unicodedata.normalize("NFKC", plain)).lower()


_CODE_RE = re.compile(r"\b(?=[a-z0-9-]*\d)(?=[a-z0-9-]*[a-z])[a-z0-9]+(?:-[a-z0-9]+)*\b")


def code_tokens(text: str) -> set[str]:
    """Form and fee codes as whole tokens ("1099-K" -> "1099k", "WIR-03" -> "wir03", "P2PE" -> "p2pe"): the
    tokenizer splits them into a number and a letter, and neither part means anything alone."""
    return {m.replace("-", "") for m in _CODE_RE.findall(normalize(text))}


def stem(word: str) -> str:
    return _STEMMER.stem_word(word)


def _canonical_number(token: str) -> str:
    """ "$45.00" -> "45", "$9,999.99" -> "9999.99", "2.60%" -> "2.6%"."""
    percent = token.endswith("%")
    token = token.replace("$", "").replace(",", "").rstrip("%")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token + ("%" if percent else "")


def terms(text: str) -> list[str]:
    """Stemmed index terms. Numbers are kept in canonical form and hyphenated compounds also emit
    their joined form, so "cutoff", "cut-off" and "cut off" meet in the index."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(normalize(text)):
        if raw[0].isdigit() or raw[0] == "$":
            out.append(_canonical_number(raw))
            continue
        word = raw.split("'")[0]  # possessives and contractions: "cashier's" -> "cashier"
        parts = word.split("-")
        candidates = parts + (["".join(parts)] if len(parts) > 1 else [])
        out.extend(stem(p) for p in candidates if p and p not in STOPWORDS)
    return out


def words(text: str) -> list[str]:
    """Lower-cased surface words in order (possessives stripped), for adjacency checks."""
    return [w.split("'")[0] for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", normalize(text))]


def is_number(term: str) -> bool:
    return term[:1].isdigit()


def strip_filler(text: str) -> str:
    """Remove greetings and courtesy phrases; they swing reranker scores and carry no meaning."""
    out = text
    for phrase in FILLER_PHRASES:
        out = re.sub(rf"\b{re.escape(phrase)}\b", " ", out, flags=re.IGNORECASE)
    tokens = [t for t in re.split(r"(\s+)", out) if t]

    def keep(token: str) -> bool:
        word = re.sub(r"\W", "", token)
        # An all-caps short token is a code ("PLS" = Bridge Plus), not a courtesy word ("pls").
        return token.isspace() or word.lower() not in FILLER_WORDS or (word.isupper() and len(word) <= 4)

    kept = [t for t in tokens if keep(t)]
    cleaned = re.sub(r"\s+", " ", "".join(kept)).strip(" ,-")
    return re.sub(r"^[,.!?\s]+", "", cleaned) or text.strip()


# ---------------------------------------------------------------- numbers and values

_AMOUNT_RE = re.compile(
    r"(?<![\w.])(\$)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(k|thousand|m|million|grand|dollars?)?(?![\w%])",
    re.IGNORECASE,
)
_MULTIPLIER = {"k": 1e3, "thousand": 1e3, "grand": 1e3, "m": 1e6, "million": 1e6}


def parse_amounts(text: str) -> list[float]:
    """Money amounts in a question: "$30,000", "30k", "$9,999.99", "250 thousand", "5,000 dollars".
    Bare integers without a currency marker (years, account digits, day counts) are ignored."""
    amounts = []
    for match in _AMOUNT_RE.finditer(text):
        dollar, whole, fraction, suffix = match.groups()
        has_commas = "," in whole
        if not (dollar or suffix or has_commas):
            continue
        value = float(whole.replace(",", "") + (fraction or ""))
        amounts.append(value * _MULTIPLIER.get((suffix or "").lower(), 1.0))
    return amounts


def parse_money(cell: str) -> float | None:
    """First money amount in a table cell, or None ("No maximum", "Included")."""
    match = re.search(r"\$(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?", cell)
    return float(match.group(1).replace(",", "") + (match.group(2) or "")) if match else None


_VALUE_RE = re.compile(
    r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?!\d)"  # money
    r"|\d+(?:\.\d+)?%"  # percent
    r"|\b\d{1,2}(?::\d{2})?\s?(?:a\.?m\.?|p\.?m\.?)"  # time of day
    r"|\b\d+\s+(?:calendar\s+|business\s+|banking\s+)?(?:days?|months?|years?|hours?|minutes?)\b",  # durations
    re.IGNORECASE,
)


def values(text: str) -> set[str]:
    """Canonical money / percent / time / duration values, for the numeric-grounding check:
    every value stated in an answer sentence must appear in the spans it cites."""
    found = set()
    for raw in _VALUE_RE.findall(normalize(text)):
        if raw.startswith("$"):
            found.add("$" + _canonical_number(raw))
        elif raw.endswith("%"):
            found.add(_canonical_number(raw))
        elif re.search(r"[ap]\.?m\.?$", raw):  # "4:00 p.m." -> "4pm", "2:45 PM" -> "2:45pm"
            found.add(re.sub(r":00(?=[ap])", "", raw.replace(" ", "").replace(".", "")))
        else:  # "10 calendar days" -> "10 calendar day"
            found.add(re.sub(r"\b(day|month|year|hour|minute)s\b", r"\1", re.sub(r"\s+", " ", raw)))
    return found


def value_kind(value: str) -> str:
    """'money' | 'percent' | 'time' | 'duration' for a canonical value from `values()`."""
    if value.startswith("$"):
        return "money"
    if value.endswith("%"):
        return "percent"
    if value.endswith(("am", "pm")):
        return "time"
    return "duration"


_PLAN_VALUE_RE = re.compile(
    r"(\$\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?%)\s+(?:on|for)\s+(?:the\s+)?(?:bridge\s+)?(essentials|plus|premier)\b",
    re.IGNORECASE,
)


def plan_values(text: str) -> dict[str, str]:
    """Values a sentence ties to a plan: "$45 on Essentials, $40 on Plus" -> {"Bridge Plus": "$40", ...}."""
    out = {}
    for value, plan in _PLAN_VALUE_RE.findall(text):
        canonical = values(value)
        if canonical:
            out[f"Bridge {plan.title()}"] = next(iter(canonical))
    return out


# ---------------------------------------------------------------- fuzzy coverage


def _edit_distance_at_most_one(a: str, b: str) -> bool:
    """True when a and b differ by at most one insertion, deletion, substitution or transposition."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diffs = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
        return len(diffs) == 1 or (
            len(diffs) == 2 and diffs[1] == diffs[0] + 1 and a[diffs[0]] == b[diffs[1]] and a[diffs[1]] == b[diffs[0]]
        )
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1 :] == short for i in range(len(long_)))


def covered(term: str, evidence: Iterable[str], known: frozenset[str] | set[str] = frozenset()) -> bool:
    """Is a (stemmed) query term present in the evidence terms?

    A word the corpus knows (`known`) must match exactly. Only an unknown word - most likely a typo such as
    "staton" - may match one edit away, and only inside the cited evidence, never against the whole
    vocabulary. So "cashier" can never be "corrected" into "casher" from "check cashers"."""
    pool = set(evidence)
    if term in pool:
        return True
    if term in known:
        return False
    return len(term) >= 5 and any(len(e) >= 5 and _edit_distance_at_most_one(term, e) for e in pool)


# ---------------------------------------------------------------- sentence splitting

_ABBREVIATIONS = frozenset(
    "e.g i.e etc vs approx inc n.a no u.s a.m p.m mr ms mrs dr st p.o "
    "jan feb mar apr jun jul aug sep sept oct nov dec".split()
)
_BOUNDARY_RE = re.compile(r"[.!?][\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9$])")


def sentence_spans(text: str, offset: int = 0) -> list[tuple[int, int]]:
    """Split `text` into sentences and return (start, end) spans shifted by `offset`.

    Spans are trimmed, so `doc[start:end]` is the exact sentence. A period after an abbreviation
    ("U.S.", "a.m.", "N.A."), a single letter, or an ordered-list marker ("1.") is not a boundary.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        end = match.start() + 1
        if re.fullmatch(r"\s*\d+\.", text[start:end]):
            continue
        last_word = re.search(r"([A-Za-z.]+)\.$", text[start:end])
        if last_word:
            word = last_word.group(1).lower().rstrip(".")
            if word in _ABBREVIATIONS or len(word) == 1:
                continue
        spans.append((start, match.end()))
        start = match.end()
    spans.append((start, len(text)))
    trimmed = []
    for s, e in spans:
        segment = text[s:e]
        lead = len(segment) - len(segment.lstrip())
        trail = len(segment) - len(segment.rstrip())
        if e - trail > s + lead:
            trimmed.append((offset + s + lead, offset + e - trail))
    return trimmed
