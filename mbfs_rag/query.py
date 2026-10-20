"""What we can learn from the question before retrieving anything."""

from __future__ import annotations

import calendar
import re
import unicodedata
from dataclasses import dataclass
from datetime import date

from .corpus import GlossaryEntry
from .text import (
    FILLER_WORDS,
    GENERIC_WORDS,
    NON_SALIENT,
    STOPWORDS,
    fold_typography,
    is_number,
    parse_amounts,
    strip_filler,
    terms,
)

PLANS = ("Bridge Essentials", "Bridge Plus", "Bridge Premier")
# Each sub-question costs a full retrieve-and-rerank pass; a pasted e-mail thread must not stall the service.
MAX_FACETS = 4

# Whose account: the customer's own words ("my payout") or the agent's ("this cust's application").
_WHOSE = (
    r"(?:my|our|their|his|her|(?:the|this|that)\s+(?:customer|cust|merchant|caller|client)'s"
    r"|(?:the|this|that)\s+(?:customer|cust|merchant|caller|client)\s+(?:'s\s+)?)"
)

# Questions that need a specific customer's account data, which this knowledge base cannot see.
_ACCOUNT_PATTERNS = (
    (
        "account_identifier",
        r"\b(?:account|acct)\s*(?:#|number|no\.?|ending(?:\s+in)?)\s*\d{2,}"
        r"|\bending\s+in\s+\d{4}\b|\blast\s+(?:four|4)\b",
    ),
    (
        "my_balance",
        rf"\b(?:what(?:'s| is)|check|see|pull up)\s+{_WHOSE}\s*(?:current\s+|available\s+)?balance\b",
    ),
    (
        "status_of_mine",
        rf"\b(?:where(?:'s| is)|did|has|have|when will|why (?:is|was|did))\s+{_WHOSE}\s*"
        r"(?:payout|deposit|wire|payment|transfer|refund|application|card|check)s?\b",
    ),
    (
        "why_charged",
        r"\bwhy\s+(?:was|were|am|is|did)\s+(?:i|we|they|he|she|(?:the|this)\s+(?:customer|cust|merchant|caller))"
        r"\s+(?:get\s+)?charged\b",
    ),
)

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})


@dataclass(frozen=True)
class Facet:
    text: str  # the sub-question, filler removed
    expanded: str  # text plus glossary meanings — what retrieval and the reranker see
    plans: tuple[str, ...]
    amounts: tuple[float, ...] = ()  # money amounts in this sub-question only


@dataclass(frozen=True)
class QueryAnalysis:
    original: str  # card numbers already masked: the raw question is never kept
    text: str
    expansions: tuple[tuple[str, str], ...]
    facets: tuple[Facet, ...]
    account_intent: str | None
    as_of_override: date | None
    amounts: tuple[float, ...]
    redacted: bool = False
    unanswered: tuple[str, ...] = ()  # sub-questions beyond MAX_FACETS

    @property
    def plans(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(p for f in self.facets for p in f.plans))


# A run of 13+ digits however it was pasted, after paste_normalize(): any whitespace (no-break spaces, PDF and
# table copies), hyphens, dashes, dots or an e-mail quote marker (">") between groups, digits glued to a word
# ("card4111..."), an expiry or CVV attached.
_DIGIT_RUN = re.compile(r"(?<!\d)\d(?:[\s.>-]*\d){12,}(?!\d)")  # one class, never overlapping \d: linear time


def paste_normalize(text: str) -> str:
    """What a pasted question becomes before anything reads it: NFKC (full-width digits and dashes), typographic
    dashes and apostrophes folded to ASCII, invisible format characters (zero-width space, word joiner, soft
    hyphen) removed. Masking runs on this, so no later folding can assemble a card number it did not see."""
    folded = fold_typography(unicodedata.normalize("NFKC", text))
    return "".join(ch for ch in folded if unicodedata.category(ch) != "Cf")


def _luhn(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


MASK = "[card number removed]"


def redact_card_numbers(text: str) -> tuple[str, bool]:
    """Mask every digit run that contains a Luhn-valid 13-19 digit window: agents paste card numbers, and an
    unmasked PAN would pull this tool into PCI scope via logs and traces. The whole run goes, so a trailing
    CVV goes with it; over-masking a long reference number costs nothing, since the KB never needs one.
    It works on paste_normalize(text), and returns that text, masked."""

    def mask(match: re.Match[str]) -> str:
        d = re.sub(r"\D", "", match.group(0))
        windows = (d[i:j] for i in range(len(d) - 12) for j in range(i + 13, min(i + 19, len(d)) + 1))
        return MASK if any(_luhn(w) for w in windows) else match.group(0)

    clean = paste_normalize(text)
    masked = _DIGIT_RUN.sub(mask, clean)
    return masked, masked != clean


def expand_glossary(text: str, glossary: list[GlossaryEntry]) -> tuple[str, list[tuple[str, str]]]:
    """Append the knowledge base's own meaning for shorthand and customer phrasing ("CB", "IP fee",
    "overseas wire"), so retrieval and reranking see the vocabulary the articles use."""
    found: list[tuple[str, str]] = []
    for entry in sorted(glossary, key=lambda e: -len(e.term)):
        term = entry.term
        flags = 0 if term.isupper() and len(term) <= 3 else re.IGNORECASE  # "IP"/"CB": match as typed
        if re.search(rf"(?<![\w+]){re.escape(term)}(?![\w+])", text, flags) and term.lower() not in {
            f.lower() for f, _ in found
        }:
            found.append((term, entry.meaning.split(" — ")[0]))
    if not found:
        return text, []
    return f"{text} ({'; '.join(m for _, m in found)})", found


# "for 2000 transactions" and "for 2000 ACH payments" are quantities; "in 2025 for wires" is still a year.
_COUNTED = (
    r"(?:transactions|payments|items|payouts|wires|transfers|checks|deposits|withdrawals|chargebacks|disputes"
    r"|refunds|statements|cards|terminals|dollars)\b"
)
_FUNCTION = r"(?:for|on|in|of|to|at|by|per|and|or|vs|with|from)\b"
_YEAR = re.compile(rf"\b(?:in|for|during|by)\s+(20\d{{2}})\b(?!\s+(?:(?!{_FUNCTION})[a-z]+\s+)?{_COUNTED})")


def parse_as_of(text: str, snapshot: date) -> date | None:
    """Explicit dates move the point in time the question is about: "in March 2026", "in 2025",
    "next year". Everything else is answered as of the corpus snapshot."""
    lowered = text.lower()
    if re.search(r"\bnext year\b", lowered):
        return date(snapshot.year + 1, 1, 1)
    if re.search(r"\blast year\b", lowered):
        return date(snapshot.year - 1, 12, 31)
    month_year = re.search(r"\b([a-z]{3,9})\.?\s+(20\d{2})\b", lowered)
    if month_year and month_year.group(1) in _MONTHS:
        return date(int(month_year.group(2)), _MONTHS[month_year.group(1)], 15)
    year = _YEAR.search(lowered)
    if year:  # no range check: "in 2019" must not quietly get today's fees
        y = int(year.group(1))
        if y < snapshot.year:
            return date(y, 12, 31)
        if y > snapshot.year:
            return date(y, 1, 1)
    return None


_PRONOUN = re.compile(r"\b(?:it|its|that|this|them|they|those|one)\b", re.IGNORECASE)


def resolve_relative_dates(text: str, snapshot: date) -> str:
    """'next year' -> 'in 2027': the reranker cannot connect a relative date to 'effective January 1, 2027'."""
    text = re.sub(r"\bnext year\b", f"in {snapshot.year + 1}", text, flags=re.IGNORECASE)
    text = re.sub(r"\blast year\b", f"in {snapshot.year - 1}", text, flags=re.IGNORECASE)
    return re.sub(r"\bthis year\b", f"in {snapshot.year}", text, flags=re.IGNORECASE)


def detect_plans(text: str) -> tuple[str, ...]:
    """Plans named in the text. "plus" is also an ordinary word ("2.6% plus $0.10"), so it counts only
    with plan context: "Bridge Plus", "Plus plan", "on Plus", a capitalised "Plus", or a trailing "plus"."""
    found = []
    if re.search(r"\bessentials?\b", text, re.IGNORECASE):
        found.append("Bridge Essentials")
    if (
        re.search(r"\bbridge\s+plus\b|\bplus\s+(?:plan|account|tier|customers?|merchants?)\b", text, re.IGNORECASE)
        or re.search(r"\b(?:on|to|from|with|in|for|the)\s+plus\b|\bplus\W*$", text, re.IGNORECASE)
        or re.search(r"\bPlus\b|\bPLUS\b", text)
    ):
        found.append("Bridge Plus")
    if re.search(r"\bpremier\b", text, re.IGNORECASE):
        found.append("Bridge Premier")
    return tuple(found)


_SPLIT = re.compile(
    r"(\?)\s+|;\s*|,?\s+\b(?:and|also)\s+"
    r"(?=(?:what|how|is|are|do|does|can|could|will|would|should|when|where|which|who)\b)",
    re.IGNORECASE,
)
# How a question starts, when it does not end with "?": "how does it work", "can she get her own card".
_ASKS = re.compile(
    r"^\s*(?:(?:and|also|but|so|then|plus)\s+)?(?:how|what|whats|when|where|why|which|who|whom|whose|can|could|do|does|"
    r"did|is|are|was|were|will|would|should|shall|may|might|must|has|have|any|tell|explain|list|show|need)\b",
    re.IGNORECASE,
)
_GENERIC_STEMS = frozenset(t for w in GENERIC_WORDS | FILLER_WORDS for t in terms(w))


def content_terms(text: str) -> list[str]:
    """Stemmed terms that can carry the meaning of a question."""
    return [
        t
        for t in terms(text)
        if len(t) > 1 and not is_number(t) and t not in _GENERIC_STEMS and t not in NON_SALIENT and t not in STOPWORDS
    ]


def split_facets(text: str) -> list[str]:
    """Split a compound question on clause boundaries only ("…, and what does Plus cost?"), never on
    an 'and' between noun phrases ("Plus and Premier"). Pieces too small to stand alone are merged, and so is a
    statement that follows a question: "chargeback deadline? merchant just got the notice" is one question with
    its context, not two."""
    parts = _SPLIT.split(text)  # pieces interleaved with the captured "?" (None for the other boundaries)
    pieces: list[tuple[str, bool]] = []
    for i in range(0, len(parts), 2):
        raw = parts[i] or ""
        asked = (i + 1 < len(parts) and parts[i + 1] == "?") or raw.rstrip().endswith("?")
        piece = raw.strip(" ,?")
        if piece:
            pieces.append((piece, asked or bool(_ASKS.match(piece))))
    merged: list[str] = []
    for piece, is_question in pieces:
        if merged and (len(content_terms(piece)) < 2 or not is_question):
            merged[-1] = f"{merged[-1]}, {piece}"
        else:
            merged.append(piece)
    if len(merged) > 1 and len(content_terms(merged[0])) < 2:
        merged[1] = f"{merged[0]}, {merged[1]}"
        merged = merged[1:]
    return merged or [text]


def analyze(query: str, glossary: list[GlossaryEntry], snapshot: date, *, strip: bool = True) -> QueryAnalysis:
    masked, redacted = redact_card_numbers(query)
    # paste_normalize() folds curly apostrophes, so "Where’s my payout" meets the same rules as "Where's my payout".
    # The mask would only confuse the reranker, but the account rules must still see that a number was there:
    # "balance on account number 3186..." is an account lookup even when the long number was masked as a card.
    searchable = masked.replace(MASK, " ")
    text = strip_filler(searchable) if strip else searchable.strip()
    rejoined, late = redact_card_numbers(text)
    if late:  # dropping filler ("4111 1111 um 1111 1111") or collapsing whitespace joined a card number's groups
        masked, redacted = rejoined, True
        searchable = masked.replace(MASK, " ")
        text = strip_filler(searchable) if strip else searchable.strip()
    with_numbers = masked.replace(MASK, "0000")
    account_intent = next(
        (name for name, pattern in _ACCOUNT_PATTERNS if re.search(pattern, with_numbers, re.IGNORECASE)), None
    )
    facets = []
    all_expansions: dict[str, str] = {}
    _, whole_expansions = expand_glossary(text, glossary)
    query_plans = detect_plans(text) or detect_plans("; ".join(m for _, m in whole_expansions))
    pieces, seen = [], set()
    for piece in split_facets(text):
        key = " ".join(piece.casefold().split())
        if key not in seen:  # a repeated question costs a full pass and adds nothing
            seen.add(key)
            pieces.append(piece)
    pieces, unanswered = pieces[:MAX_FACETS], pieces[MAX_FACETS:]
    for i, piece in enumerate(pieces):
        expanded, found = expand_glossary(resolve_relative_dates(piece, snapshot), glossary)
        all_expansions.update(found)
        amounts = tuple(parse_amounts(piece))  # amounts belong to the sub-question that states them...
        if i and _PRONOUN.search(piece):
            expanded += f" (context: {pieces[0]})"  # "is there a fee if I don't send it back?" -> it = the terminal
            amounts = amounts or tuple(parse_amounts(pieces[0]))  # ...or that "it" refers back to
        own = detect_plans(piece) or detect_plans("; ".join(m for _, m in found))
        facets.append(Facet(piece, expanded, own or query_plans, amounts))  # unplanned: inherits the query's plans
    return QueryAnalysis(
        original=masked,
        text=text,
        expansions=tuple(all_expansions.items()),
        facets=tuple(facets),
        account_intent=account_intent,
        as_of_override=parse_as_of(text, snapshot),
        amounts=tuple(parse_amounts(text)),
        redacted=redacted,
        unanswered=tuple(unanswered),
    )
