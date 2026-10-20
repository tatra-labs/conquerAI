# Held-out labeling notes

Recorded by the blind labeler while writing `heldout.toml` (committed as `heldout_v1.toml`). They are not used for scoring; they say where
the gold involved a judgment call, so a reviewer can disagree with the label rather than with the system.

- The corpus file does include a superseded document (fee-schedule-2025-01), although the brief says superseded documents are not in it. It is never cited as a source. *[Author's note: the labeling instructions were wrong here. The superseded schedule is indexed on purpose, so the system can show that it never presents it as current.]* I used it only to set the forbidden stale values: chargeback fee $15 (H01, H06), ATM fee $2.00 (H02), Plus international wire fee $40 (H09) and PCI non-validation fee $14.95 (H12).
- H05: two current documents disagree on how long to keep a mobile-deposited check. The Funds Availability Policy (effective 2025-07-01) says 14 days; the help article (effective 2025-08-01, newer) says 30 days. I followed the stated authority order (policy beats help article) and forbade days:30. Drop that forbidden if the newer help article should win.
- H09: whether 'answer' or 'clarify' should be preferred is borderline, since the cost depends on payment method and plan (about 12 values). The required facts check the Essentials amounts for standard ACH ($0.25), domestic wire ($30) and international wire ($50). $45, the outdated FAQ's Essentials international fee, cannot be forbidden because it is the current Plus fee; requiring $50 catches that mistake instead.
- H01: no policy says outright that processing fees come out of payouts. That fact rests on the 1099-K help article ('before refunds, chargebacks, and processing fees are subtracted') plus fee rows MER-01 and MER-02 cited without a plan column. I removed a planned forbidden for the upcoming 1.75% Instant Payout fee, because a correct answer could mention the January 2027 change.
- H12: the KB never says whether the BridgeTerminal Mini uses point-to-point encryption, so the label expects a hedged 'may qualify for SAQ P2PE'. 'partial' is accepted for answers that flag this gap.
- H11: the KB describes only employee cards as additional cards, and does not explain role assignment or a separate card for a co-owner, so 'partial' is accepted.
- H04: the KB does not say how to cancel a terminal rental or whether a purchased terminal can be refunded, so 'partial' and 'clarify' (rented or purchased?) are accepted alongside 'answer'.
- H03: the change-password fact assumes the customer did not request the code, which is the case the KB's guidance covers. I required the phishing@merchantbridge.example report address rather than the fraud-line number as the third fact.
- H07 and H10 are labeled 'refuse' first but each has one required fact: the 'consult a qualified tax professional' redirect and routing to the Onboarding team. This assumes a refusal should include the redirect the KB provides.
- Matcher assumptions: days:N is assumed to also match 'N calendar days' and 'N-day'; money:2 is assumed not to match $2.50; regexes use an inline (?i) flag. Text and regex matchers cannot detect negation (for example 'yes' in H05, 'required' in H12), so the notes state the correct answer explicitly.

The labeler also validated its own file: every quote is a verbatim substring of its document, every row
and column reference exists and carries the stated amount, reference answers satisfy every fact and trip
no forbidden matcher, and deliberately wrong answers trip the intended forbidden matcher.
