"""Question analysis and text normalisation: the small rules that decide what retrieval even sees."""

from datetime import date

import pytest

from mbfs_rag.corpus import GlossaryEntry
from mbfs_rag.query import (
    analyze,
    detect_plans,
    parse_as_of,
    redact_card_numbers,
    resolve_relative_dates,
    split_facets,
)
from mbfs_rag.text import code_tokens, covered, parse_amounts, sentence_spans, stem, strip_filler, terms, values, words

SNAPSHOT = date(2026, 10, 19)
GLOSSARY = [
    GlossaryEntry("PLS", "Bridge Plus", "x"),
    GlossaryEntry("CB", "Chargeback", "x"),
    GlossaryEntry("Overseas wire", "International wire", "x"),
]


def test_tokenizer_canonicalises_money_percent_and_compounds():
    assert terms("$45.00 and $9,999.99") == ["45", "9999.99"]
    assert "2.6%" in terms("2.60% + $0.10")
    assert "cutoff" in terms("cut-off") and "cutoff" in terms("cutoff")
    assert terms("cashier's check")[0] == "cashier"


def test_sentence_splitter_respects_abbreviations_and_list_markers():
    text = "Lanternstone Bank, N.A., Member FDIC. Wires by 4:00 p.m. ET go today. 1. Lock the card. 2. Report it."
    sentences = [text[s:e] for s, e in sentence_spans(text)]
    assert sentences == [
        "Lanternstone Bank, N.A., Member FDIC.",
        "Wires by 4:00 p.m. ET go today.",
        "1. Lock the card.",
        "2. Report it.",
    ]


def test_typo_tolerance_only_for_unknown_words_and_only_inside_the_evidence(corpus):
    known = corpus.vocabulary
    assert stem("staton") not in known and covered(stem("staton"), {stem("station")}, known)
    # A word the corpus knows matches exactly: "cashier" must stay uncovered by "check cashers" (one edit away),
    # because it is the term that tells the gate the KB has no cashier's-check product.
    assert stem("cashier") in known
    assert not covered(stem("cashier"), {stem("cashers"), stem("check")}, known)
    assert not covered("loan", {"long"})  # short words are never fuzzy-matched


def test_value_canonicalisation_for_numeric_grounding():
    assert values("4:00 PM ET, 4 p.m., $25.00, 2.60%, 10 calendar days") == {"4pm", "$25", "2.6%", "10 calendar day"}


def test_filler_is_stripped_but_codes_are_not():
    assert strip_filler("hey!! quick q pls - whats the chargeback fee?? thx so much") == "whats the chargeback fee??"
    assert "PLS" in strip_filler("CB fee PLS?")


def test_card_numbers_are_masked_only_when_they_pass_luhn():
    masked, hit = redact_card_numbers("card 5555 5555 5555 4444 was stolen")
    assert hit and "5555" not in masked and "[card number removed] was stolen" in masked
    assert redact_card_numbers("account ending 4417")[1] is False


@pytest.mark.parametrize(
    "pasted",
    [
        "card 4111111111111111 123 was stolen",  # a CVV glued on makes 19 digits that fail Luhn as a whole
        "card 4111 1111 1111 1111 12/27 skimmed",
        "ref 12 4111111111111111",
        "4111\xa01111\xa01111\xa01111",  # no-break spaces from a web or e-mail copy
        "4111  1111  1111  1111",
        "4111.1111.1111.1111",
        "4111–1111–1111–1111",
        "card4111111111111111",
        "pan_4111111111111111",
        "amex 3782 822463 10005",
        "card 5555 5555 5555 4444 123 was stolen",
        "card 4111 1111 1111 1111 12 27 was stolen",
        "4111 1111 1111 1111 9 times declined",
    ],
)
def test_card_numbers_are_masked_however_they_are_pasted(pasted):
    masked, hit = redact_card_numbers(pasted)
    assert hit and not any(group in masked for group in ("4111", "3782", "10005")), masked


@pytest.mark.parametrize(
    "text",
    ["call 1-800-555-0142 or 1-800-555-0199", "$2,214 in sales, payout $1,912", "routing 021000021", "up to $250,000"],
)
def test_other_numbers_are_left_alone(text):
    assert redact_card_numbers(text) == (text, False)


def test_a_long_number_that_fails_luhn_is_not_a_card_number():
    assert redact_card_numbers("reference 1234567890123")[1] is False  # 13 digits, Luhn-invalid


def test_typographic_apostrophes_meet_the_same_rules():
    for question in ("Where\u2019s my payout from Friday?", "What\u2019s my current balance?"):
        assert analyze(question, [], SNAPSHOT).account_intent is not None, question


def test_accents_are_folded_not_split():
    assert terms("Soci\u00e9t\u00e9 G\u00e9n\u00e9rale") == terms("Societe Generale")


def test_sub_questions_are_deduplicated_capped_and_keep_their_own_amounts():
    repeated = analyze("What is the wire fee? What is the chargeback fee? " * 40, [], SNAPSHOT)
    assert [f.text for f in repeated.facets] == ["What is the wire fee", "What is the chargeback fee"]
    six = "What is the wire fee? What is the chargeback fee? How do I return a terminal? What is the ACH fee? "
    many = analyze(six + "What is the PCI fee? What is the stop payment fee?", [], SNAPSHOT)
    assert len(many.facets) == 4 and len(many.unanswered) == 2
    follow_up = analyze("I have $30,000 in Bridge Reserve Savings; what APY would it earn?", [], SNAPSHOT)
    assert follow_up.facets[-1].amounts == (30000.0,)  # "it" refers back to the first part
    two = analyze("What APY does Reserve Savings pay, and what is the fee for a $50,000 domestic wire?", [], SNAPSHOT)
    assert [f.amounts for f in two.facets] == [(), (50000.0,)]


def test_plus_is_a_plan_only_in_plan_context():
    assert detect_plans("2.60% plus $0.10 per swipe") == ()
    assert detect_plans("fee on plus?") == ("Bridge Plus",)
    assert detect_plans("upgrade from Essentials to Plus") == ("Bridge Essentials", "Bridge Plus")


def test_amounts_need_a_currency_marker_or_a_scale():
    assert parse_amounts("APY on 30k or $250 thousand or $9,999.99") == [30000.0, 250000.0, 9999.99]
    assert parse_amounts("account ending 4417 in 2026 within 30 days") == []


def test_compound_questions_split_on_clauses_not_on_noun_phrases():
    assert split_facets("Is the fee the same on Plus and Premier?") == ["Is the fee the same on Plus and Premier"]
    assert len(split_facets("What happens to my rented terminal, and is there a fee if I don't send it back?")) == 2


def test_relative_dates_resolve_against_the_snapshot():
    assert resolve_relative_dates("fees going up next year?", SNAPSHOT) == "fees going up in 2027?"
    assert analyze("What was the chargeback fee in March 2026?", [], SNAPSHOT).as_of_override == date(2026, 3, 15)


def test_account_intent_is_narrow():
    assert analyze("What's the current balance on account ending 4417?", [], SNAPSHOT).account_intent
    assert analyze("Where's my payout from Friday?", [], SNAPSHOT).account_intent
    assert analyze("my card got skimmed at a gas station what do i do", [], SNAPSHOT).account_intent is None


def test_glossary_expands_shorthand_and_later_parts_inherit_context():
    a = analyze("CB fee PLS?", GLOSSARY, SNAPSHOT)
    assert "Chargeback" in a.facets[0].expanded and a.facets[0].plans == ("Bridge Plus",)
    b = analyze("What happens to my rented terminal, and is there a fee if I don't send it back?", [], SNAPSHOT)
    assert "rented terminal" in b.facets[1].expanded  # "it" resolved by carrying the first part along


def test_a_masked_account_number_still_routes_to_an_account_lookup():
    analysis = analyze("What's the current balance on account number 31860913909960308?", [], SNAPSHOT)
    assert analysis.account_intent == "account_identifier"  # masked or not, it is still an account number
    assert "3186" not in analysis.original


def test_codes_are_whole_tokens():
    assert code_tokens("When will I get my 1099-K?") == {"1099k"} == code_tokens("Do I get a 1099K?")
    assert code_tokens("What happened in 2027?") == set() == code_tokens("my 401(k) plan")
    assert words("ch\u00e8que deposit") == ["cheque", "deposit"]


def test_a_count_is_not_a_year():
    assert parse_as_of("What is the excess transaction fee for 2000 transactions on Essentials?", SNAPSHOT) is None
    assert parse_as_of("ACH origination on Essentials for 2050 payments?", SNAPSHOT) is None
    assert parse_as_of("What was the chargeback fee in 2025?", SNAPSHOT) == date(2025, 12, 31)
    assert parse_as_of("What was the fee in 2025 for wires?", SNAPSHOT) == date(2025, 12, 31)  # still a year
    assert parse_as_of("What was the chargeback fee in 2019?", SNAPSHOT) == date(2019, 12, 31)  # not today's fee


@pytest.mark.parametrize(
    "pasted",
    [
        "card 4111\u22121111\u22121111\u22121111 was skimmed",  # minus sign, folded to "-" by later normalisation
        "card 4111    1111    1111    1111 was skimmed",  # PDF or table copy
        "card 4111 1111\n> 1111 1111 was used",  # quoted e-mail line wrap
        "card 4111\u200b1111\u200b1111\u200b1111",  # zero-width spaces
        "card 4111\u00ad1111\u00ad1111\u00ad1111",  # soft hyphens
        "card 4111\uff0d1111\uff0d1111\uff0d1111",  # full-width hyphen
        "card \uff14\uff11\uff11\uff11 1111 1111 1111",  # full-width digits
        "card 4111" + "         1111" * 3,  # nine spaces between groups
        "card 4111 1111\n        1111 1111",  # a line wrap with an indent
        "card number is 4111 1111 um 1111 1111, what do I do?",  # filler between groups, removed later
    ],
)
def test_masking_sees_what_later_normalisation_would_assemble(pasted):
    analysis = analyze(pasted, [], SNAPSHOT)
    assert analysis.redacted and "1111" not in analysis.original and "1111" not in analysis.text


def test_a_statement_after_a_question_is_its_context_not_another_question():
    assert split_facets("intl wire fee? cust is on plus") == ["intl wire fee, cust is on plus"]
    assert split_facets("What is the wire fee? What is the chargeback fee?") == [
        "What is the wire fee",
        "What is the chargeback fee",
    ]
    assert len(split_facets("How do I return a terminal, and can I get a refund on the rental?")) == 2


def test_account_questions_in_the_agent_s_voice():
    for question, intent in [
        ("what's this cust's balance right now", "my_balance"),
        ("why did the customer's payout get held?", "status_of_mine"),
        ("why was the merchant charged twice for one sale?", "why_charged"),
    ]:
        assert analyze(question, [], SNAPSHOT).account_intent == intent, question
    assert analyze("Where is the refund policy?", [], SNAPSHOT).account_intent is None
