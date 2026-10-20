# MBFS Knowledge Service: results

The final evaluation run, written out in full as plain text. The results sections are
copied from `README.md`, so the two always agree; the worked example and the risk-coverage table are generated
from `eval/results/results.json` (`python scripts/build_results_md.py` rebuilds this file). Answers are as of the
corpus snapshot, 2026-10-19. Frozen eval files (first 16 hex digits of each file's SHA-256):
dev `ec8ca229e011098e`, heldout_v1 `7ce1a409fda69ba5`, heldout_v2 `8b064c86d068cc17`. Every rate is k/n with a 95% Wilson interval: at n = 20 one question moves a rate by 5 points, at
n = 12 by 8.

Contents: [The headline](#the-headline) · [One question, end to end](#one-question-end-to-end-d02) ·
[Results in detail](#results-in-detail) · [What I would do next](#what-i-would-do-next) ·
[Where to look next](#where-to-look-next)

## The headline

| | **held-out v2**<br>blind, after the fixes (n = 20) | held-out v1<br>blind, before the fixes (n = 12) | dev<br>used in development (n = 27) | held-out v1 rerun<br>after the fixes, not blind (n = 12) |
|---|---|---|---|---|
| **passed** | 5/20 (25%, 95% CI 11-47%) | 3/12 (25%, 95% CI 9-53%) | 23/27 (85%, 95% CI 68-94%) | 5/12 (42%, 95% CI 19-68%) |
| acceptable decision | 13/20 (65%, 95% CI 43-82%) | 6/12 (50%, 95% CI 25-75%) | 26/27 (96%, 95% CI 82-99%) | 8/12 (67%, 95% CI 39-86%) |
| unsafe answers (a wrong or stale value, or answered where the label says refuse or clarify) | 1 | 1 | 0 | 0 |
| facts stated in a sentence that cites a supporting span | 12/42 (29%, 95% CI 17-44%) | 5/25 (20%, 95% CI 9-39%) | 26/32 (81%, 95% CI 65-91%) | 7/25 (28%, 95% CI 14-48%) |
| answerable but refused or clarified | 5/17 (29%, 95% CI 13-53%) | 4/10 (40%, 95% CI 17-69%) | 1/24 (4%, 95% CI 1-20%) | 4/10 (40%, 95% CI 17-69%) |
| should refuse or clarify, but answered | 1/3 (33%, 95% CI 6-79%) | 0/2 (0%, 95% CI 0-66%) | 0/3 (0%, 95% CI 0-56%) | 0/2 (0%, 95% CI 0-66%) |
| latency per question, warm, median / max | 740 / 1755 ms | 534 / 997 ms | 815 / 2120 ms | 750 / 1698 ms |

**Bottom line.** On blind questions the system now makes the right call (answer, partial answer, clarify or refuse) about two times in three, up from one in two, and rarely unsafely. But it states every required fact, in a sentence that cites a supporting span, only a quarter of the time. Its answers quote one unit, chosen sentence by sentence by the small ranker, while real questions spread their facts over several. That is the next thing to fix (see [What I would do next](#what-i-would-do-next)).

**Read it left to right.** Held-out v1 was the first blind test, run once: 3 of 12, against 22 of 27 on dev. The gap was the finding. Its failures had general causes (below), which I fixed. Then a fresh blind set, v2, built the same way and run once, measured the result. Held-out v1 rerun after the fixes is shown only to make the change visible; it is no longer a blind number.

Paraphrase invariance: **7/12** dev questions keep their decision and cited documents when reworded in agent shorthand (6/12 before the fixes).

Cold start 2.1 s (models loaded, corpus parsed, embeddings from the on-disk cache). Latency was measured on a shared desktop with other workloads running; ONNX Runtime is capped at 4 threads (`MBFS_ONNX_THREADS`).

## One question, end to end: D02

> What's the fee for an outgoing international wire on the Plus plan?

The real trace of this dev question from the final run, stage by stage: what each component saw and
decided. Logits are raw cross-encoder scores, not probabilities.

**1. Read the question.** After cleaning: "What's the fee for an outgoing international wire on the Plus plan?". Parts: 1. Plans named: Bridge Plus. Amounts: none. Glossary expansions: none. Account data asked: no. Card number masked: no. Date: 2026-10-19 (the corpus snapshot).

**2. Retrieve.** BM25 (body plus a heading field) and dense retrieval (bge-small) each nominate 12 candidates; their union is 17. 37 units could not be retrieved on 2026-10-19: every unit of Business Fee & Rate Schedule (effective January 1, 2025), which is superseded.

| BM25 top 5 | score | dense top 5 | cosine |
|---|---|---|---|
| Wire Transfer Policy > Wire fees | 15.59 | Wire Transfers FAQ > How much does it cost to send an international wire? | 0.834 |
| Wire Transfer Policy > Daily outgoing wire limits, row Bridge Plus | 15.24 | Wire Transfer Policy > Daily outgoing wire limits, row Bridge Plus | 0.785 |
| Wire Transfers FAQ > How much does it cost to send an international wire? | 14.12 | Business Fee & Rate Schedule (effective July 1, 2026) > Wire transfers, row WIR-03 International wire – outgoing | 0.777 |
| Wire Transfer Policy > Daily outgoing wire limits, row Bridge Essentials | 13.62 | Business Fee & Rate Schedule (effective July 1, 2026) > Wire transfers, row WIR-04 International wire – incoming | 0.770 |
| Wire Transfer Policy > Daily outgoing wire limits, row Bridge Premier | 13.62 | Business Fee & Rate Schedule (effective July 1, 2026) > Wire transfers, row WIR-01 Domestic wire – outgoing | 0.764 |

**3. Rerank.** The small cross-encoder (MiniLM-L6) scores every candidate in the union. The top 8:

| candidate | ranker logit | source, state | in the trusted window |
|---|---|---|---|
| Wire Transfers FAQ > How much does it cost to send an international wire? | +8.77 | FAQ, current, flagged possibly outdated | no |
| Wire Transfers FAQ > Is there a daily limit on how much I can wire? | +6.54 | FAQ, current, flagged possibly outdated | no |
| Wire Transfer Policy > Daily outgoing wire limits, row Bridge Plus | +5.77 | policy, current | yes |
| Notice of Fee Changes Effective January 1, 2027 > What's not changing | +5.46 | notice, upcoming | no |
| Business Fee & Rate Schedule (effective July 1, 2026) > Wire transfers, row WIR-03 International wire – outgoing **(chosen)** | +5.06 | fee schedule, current | yes |
| Business Fee & Rate Schedule (effective July 1, 2026) > Wire transfers, row WIR-04 International wire – incoming | +4.61 | fee schedule, current | yes |
| Wire Transfer Policy > Daily outgoing wire limits, row Bridge Premier | +3.28 | policy, current | yes |
| Wire Transfer Policy > Daily outgoing wire limits, row Bridge Essentials | +3.27 | policy, current | yes |

**4. Choose the evidence.** Trust comes before rank. The window holds the current sources not flagged as outdated, within 3 logits of the best of them; an upcoming or outdated source near the top (here the FAQ that ranks first) can only appear as a note. Inside the window each candidate pays 2 logits per question term it does not cover; authority breaks ties.

| candidate in the window | ranker logit | question terms not covered | logit − 2 × not covered |
|---|---|---|---|
| Wire Transfer Policy > Daily outgoing wire limits, row Bridge Plus | +5.77 | 2 | +1.77 |
| Business Fee & Rate Schedule (effective July 1, 2026) > Wire transfers, row WIR-03 International wire – outgoing **(chosen)** | +5.06 | 0 | +5.06 |
| Business Fee & Rate Schedule (effective July 1, 2026) > Wire transfers, row WIR-04 International wire – incoming | +4.61 | 1 | +2.61 |
| Wire Transfer Policy > Daily outgoing wire limits, row Bridge Premier | +3.28 | 3 | -2.72 |
| Wire Transfer Policy > Daily outgoing wire limits, row Bridge Essentials | +3.27 | 3 | -2.73 |
| Wire Transfer Policy > Wire fees | +3.12 | 3 | -2.88 |

The chosen unit is a table row, so structure narrows the rows and picks the cell. Rows of this table in the pool: WIR-03 International wire – outgoing; WIR-04 International wire – incoming. Question words that narrow them (as stems): `intern`, `outgo`. Rows left: 1, WIR-03 International wire – outgoing. The plan named picks the column: Bridge Plus → $45.00.

**5. Verify.** A stronger cross-encoder (bge-reranker-v2-m3) scores only the unit about to be quoted: +3.63, against the answer floor of -3.09 for a table row (and +0.91 for "strong"). The small ranker gave the same unit +5.06; the two scales differ, so each has its own thresholds, fitted on the calibration set.

**6. Gate.** Ordered rules over these signals; the first that fires decides. Re-run here with the gate's own code on the recorded signals:

| # | rule | fires when | this question | fires |
|---|---|---|---|---|
| 1 | `empty_input` | the question is empty | empty: no | no |
| 2 | `account_lookup_with_policy` | it asks about the customer's own account, and policy evidence is strong | account data asked: no | no |
| 3 | `account_lookup` | it asks about the customer's own account | account data asked: no | no |
| 4 | `no_topical_term` | no word names anything the KB covers | names a KB topic: yes | no |
| 5 | `outdated_only` | a fee, time or limit was asked and only an outdated source has one | value only in an outdated source: no | no |
| 6 | `no_relevant_evidence` | the verifier's score is below the floor for the unit type | score +3.63, floor -3.09 | no |
| 7 | `missing_key_term` | a key modifier is uncovered and the evidence is only moderately relevant | uncovered modifiers: none | no |
| 8 | `too_many_matches` | more matching table rows than an agent can scan | rows matched: 1 | no |
| 9 | `uncovered_modifier` | a modifier in the question is not covered by the evidence | uncovered modifiers: none | no |
| 10 | `answer` | none of the above |  | **yes: answer** |

**7. Answer.** Decision `answer`, level `medium`: the score clears the strong threshold (+0.91), but another source gives a conflicting value. What the agent is told to do: "Check the cited source before reading it out."

```text
International wire – outgoing on Bridge Plus: $45.00 per wire. Intermediary or correspondent bank fees may also apply. MerchantBridge does not control fees deducted by intermediary or beneficiary banks. [1][2][3][4][5]
Note: Conflicting source: Wire Transfers FAQ says "The fee for each outgoing international wire is $45 on Essentials, $40 on Plus, and $25 on Premier." This answer follows Business Fee & Rate Schedule (effective July 1, 2026) (fee schedule, effective July 1, 2026). The faq was last reviewed February 10, 2025 and may be outdated. [6]
```

| # | source | cited text | where |
|---|---|---|---|
| 1 | Business Fee & Rate Schedule (effective July 1, 2026) [WIR-03 / Service] | "Service: International wire – outgoing" | `corpus/fees/fee-and-rate-schedule-2026-07.csv` line 16, chars 3021-3059, span `fee-schedule-2026-07:s100` |
| 2 | Business Fee & Rate Schedule (effective July 1, 2026) [WIR-03 / Bridge Plus] | "Bridge Plus: $45.00" | `corpus/fees/fee-and-rate-schedule-2026-07.csv` line 16, chars 3129-3148, span `fee-schedule-2026-07:s103` |
| 3 | Business Fee & Rate Schedule (effective July 1, 2026) [WIR-03 / Unit] | "Unit: per wire" | `corpus/fees/fee-and-rate-schedule-2026-07.csv` line 16, chars 3174-3188, span `fee-schedule-2026-07:s105` |
| 4 | Business Fee & Rate Schedule (effective July 1, 2026) [WIR-03 / Notes] | "Notes: Intermediary or correspondent bank fees may also apply [3]" | `corpus/fees/fee-and-rate-schedule-2026-07.csv` line 16, chars 3190-3255, span `fee-schedule-2026-07:s106` |
| 5 | Business Fee & Rate Schedule (effective July 1, 2026) [WIR-03 / Footnote [3]] | "Footnote [3]: MerchantBridge does not control fees deducted by intermediary or beneficiary banks." | `corpus/fees/fee-and-rate-schedule-2026-07.csv` line 16, chars 3257-3354, span `fee-schedule-2026-07:s107` |
| 6 | Wire Transfers FAQ | "The fee for each outgoing international wire is $45 on Essentials, $40 on Plus, and $25 on Premier." | `corpus/help/wire-transfers-faq.html` line 67, chars 998-1097, span `faq-wire-transfers:s8` |

**Graded against its gold label: pass.**

- Decision: answer; acceptable: answer.
- Required fact `fee` (`money:45`, supported by fee-schedule-2026-07 row WIR-03, column Bridge Plus): stated and cited ([2]).
- Forbidden `money:40` (stale FAQ / superseded schedule value): not stated; it appears only quoted inside the conflict note, and forbidden values are checked only on what the answer asserts.
- Nothing it cites is superseded.

## Results in detail

### How the held-out numbers came about

Held-out v1 (`eval/heldout_v1.toml`, frozen hash `7ce1a409fda69ba5`) was run once, after two rounds of code review: **3/12 (25%, 95% CI 9-53%)**. The run is kept as it came out in [`eval/results/heldout_v1_blind/`](eval/results/heldout_v1_blind/). Its failures had five general causes:

1. **A floor fitted on the wrong questions.** The first calibration questions were templated from the documents' own headings and rows, so they scored far higher than anything an agent types (on-topic median 8.4). Four answerable questions were refused although their right passage ranked first, at 0.22, -1.1, -3.3 and -4.3 against a prose floor of 1.73. The invariance set had shown the same thing.
2. **An outdated source allowed to supply a value.** When nothing current cleared the floor, the stale FAQ answered as a last resort, under a warning, with "$40 on Plus". That was the one unsafe answer.
3. **A context sentence read as a question.** "merchant just got the notice" was split off and refused.
4. **Account rules written for the customer's voice only.** "this cust's application" did not match.
5. **One passage per document.** Two answers missed facts that sit in a second passage of the same article. Nothing general fixes this without a better ranker, so it is listed as a limitation.

What changed (general fixes; no rule names a test question):

- **Realistic calibration.** About 208 on-topic questions in four styles (agent shorthand, customer words, a situation first, typos) were written from the corpus alone by agents that never saw the system or an eval set, each kept only if an independent agent confirmed its quote answers it. They are set against off-topic and not-covered questions in the same styles. Any question overlapping an eval question was removed. The floor is now fitted on the score the gate actually tests (the primary unit, with the gate off), over the questions that reach the floor rule, and ties are broken at the middle of the tied range rather than its most conservative edge.
- **An answerability verifier.** Refitted on realistic questions, the best floor for the small ranker still refused about two thirds of them: it scores real phrasing and not-covered questions in KB vocabulary alike (AUC 0.87). bge-reranker-v2-m3 separates them far better (0.93 on the unit the gate tests), so it now scores the one unit about to be quoted, and the gate tests its score. The ranker still ranks and drives the gap rules.
- **An outdated source never supplies a fee, time or limit** (new gate rule `outdated_only`, refuse with a pointer to the current document).
- **A statement after a question is its context**, not a sub-question.
- **Account rules read the agent's voice**: "this cust's", "the customer's", "the merchant".

Then held-out v2 (`eval/heldout_v2.toml`): 20 new questions, two new writers and two blind labelers, the same protocol, frozen hash `8b064c86d068cc17`, run once after the fixes.

### Held-out v2, question by question

| id | question | decision (acceptable) | rule | result |
|---|---|---|---|---|
| V01 | do I need to be charging sales tax on my online orders that ship out of state? | refuse (refuse) | `no_relevant_evidence` | pass |
| V02 | my payout this morning is like $300 short of what I ran on Saturday, what did you guys take out? | refuse (partial/answer) | `no_relevant_evidence` | **fail**: decision (refuse, wanted partial/answer) |
| V03 | daily limit on the debit card? | answer (answer/clarify) | `answer` | pass |
| V04 | if I send the terminal back do I get my money back? | answer (partial/clarify) | `answer` | **fail**: decision (answer, wanted partial/clarify) |
| V05 | how to add a view-only user for cust's bookkeeper, no acess to move money | answer (answer/partial) | `answer` | **fail**: missing fact |
| V06 | outgoing domestic wire fee on Essentials? | answer (answer) | `answer` | pass |
| V07 | I got an email about a chargeback over the weekend, how long do I have to send you proof? | answer (answer) | `answer` | **fail**: missing fact |
| V08 | cust gave her one-time code to a caller saying he was from MBFS, next steps? | answer (answer) | `answer` | **fail**: missing fact |
| V09 | cust going from Essentials to Premier mid-month, is the fee prorated and does her debit card number change? | refuse (partial) | `no_relevant_evidence + no_relevant_evidence` | **fail**: decision (refuse, wanted partial) |
| V10 | the app keeps rejecting my check, it says something about the endorsement, what am I supposed to write on the back? | answer (answer) | `answer` | pass |
| V11 | Bridge Plus monthly fee? cust just spotted it on their stmt | answer (answer) | `answer` | **fail**: missing fact |
| V12 | How much do you guys charge for one of the card machines? | refuse (answer/clarify) | `no_relevant_evidence` | **fail**: decision (refuse, wanted answer/clarify) |
| V13 | PCI non-compliance fee hit again this month, if they finish the SAQ today can we refund it? | answer (answer) | `answer` | **fail**: missing fact |
| V14 | I sent a wire to my supplier yesterday afternoon and they're saying it never came in. Can you see where it is? | refuse (partial) | `no_relevant_evidence` | **fail**: decision (refuse, wanted partial) |
| V15 | batch cutoff time for next-day payout? | answer (answer) | `answer` | **fail**: missing fact |
| V16 | Someone just called saying they were from MerchantBridge and asked me to read back the code you texted me. Was that actually you? | answer (answer) | `answer` | **fail**: missing fact |
| V17 | cust's mobile depsoit got rejected for endorsement even though they signed the back, what does it need to say? | answer (answer) | `answer` | **fail**: missing fact |
| V18 | I just opened the account. How do I add my business partner so she can log in too, and can she get her own debit card? | refuse (answer/partial) | `no_relevant_evidence + no_relevant_evidence` | **fail**: decision (refuse, wanted answer/partial) |
| V19 | daily limit for new accts? | answer (clarify) | `answer` | **fail**: decision (answer, wanted clarify) |
| V20 | Do I have to charge sales tax on my online orders that ship to other states? | refuse (refuse) | `no_relevant_evidence` | pass |

Split by overlap with v1 (the writers saw the same brief): questions sharing a v1 scenario passed 3/10, new scenarios 2/10. The overlap did not inflate the number.

What went wrong, from each trace:

- **V02**: Refused. A narrative about one payout ($300 short since Saturday): the KB explains weekend batch timing and what a payout deducts, across several documents, but no unit reads as its answer (verifier -6.5).
- **V04**: Answered from the wrong sentence ('If you've lost your return label, call Business Support') for 'do I get my money back?'. The label wanted the bought-or-rented question, or the partial rules.
- **V05**: Half right. The internal caller-verification table says what a view-only user may do, but not how to add one (an owner or admin, in the app); that sits in the online-banking article, which was not quoted.
- **V07**: Right deadline (10 calendar days, weekends included), but not where to respond (the Merchant Portal's Disputes tab), the third fact the label requires.
- **V08**: Weak: quoted the internal SOP's 'MBFS will never ask for these' instead of the steps in the security articles (call the fraud line, reset the password, lock the card).
- **V09**: Refused both parts. A plan change takes effect from the next statement cycle and switching is free (Business Checking Plans), but the best unit scored -3.1.
- **V11**: Right fee ($25.00), but not the waiver threshold ($10,000 average daily balance) the label requires: the fee row only points to Business Checking Plans.
- **V12**: Refused: 'card machines' never meets 'BridgeTerminal'. A vocabulary gap the KB glossary does not cover.
- **V13**: Answered what the PCI non-validation fee is, not whether it can be refunded once the SAQ is done: the answer is in the internal agent guidelines (agents cannot reverse it; Merchant Risk can).
- **V14**: Refused. Tracing one wire needs the account, but the KB covers how to ask for a trace and its $35 fee; the label wanted that partial answer.
- **V15**: Right batch close (8:00 PM ET), but not which plans pay out the next business day (Plus and Premier).
- **V16**: Gave the fraud line and the phishing address, but not that MBFS never asks for a code, or to change the password.
- **V17**: Picked the wrong sentence: 'A check without this endorsement may be rejected' instead of the wording just above it. Sentence choice inside a passage is still the small ranker's.
- **V18**: Refused both parts: the same scenario as held-out v1's H11. The online-banking article (an owner or admin adds users in Settings) and the debit-card article (employee cards) score below the floor.
- **V19**: Answered with the debit card's daily limits, correct for that reading. The label says clarify (which limit: card, ATM, deposits, wires?), so it counts as the one unsafe answer.

### Held-out v1, the blind run, failure by failure

| id | question | decision (acceptable) | what happened, and what changed |
|---|---|---|---|
| H01 | I did $2,214 in sales yesterday but my payout was only $1,912, where did the rest go? | refuse (partial/answer) | Refused. A narrative question (sales versus payout amounts): every passage scores about -6, though the KB explains what a payout deducts across several documents. No fix: no single passage reads as its answer. |
| H03 | I got a text saying it's from MerchantBridge asking for the verification code you just sent me, is that legit? | answer (answer) | Answered from the 'we never ask for your passcode' passage. The label also wants the advice in another passage of the same article (change your password; report the text to phishing@), which ranked outside the window. |
| H04 | cust wants to send her terminal back, how does that work? | refuse (answer/clarify/partial) | Refused, though the right article ranked first (0.22): below the old prose floor of 1.73. Fixed by the realistic calibration. |
| H06 | chargeback response deadline? merchant just got the notice | partial (answer) | 'merchant just got the notice' became a second question and was refused. Fixed: a statement after a question is its context. The first part also gave the dispute steps rather than the 10-day deadline. |
| H08 | reserve savigns monthly withdrawal limit? | refuse (answer) | Refused. With 'savigns' misspelled the right passage scored -3.3, far below the old floor. Addressed by the realistic calibration, which includes typos. |
| H09 | How much do you guys charge to send money to my supplier? | answer (answer/clarify) | Unsafe. Nothing current cleared the floor, so the outdated FAQ answered as a last resort with '$40 on Plus', a stale value, under a warning. Fixed: an outdated source never supplies a fee, time or limit. |
| H10 | why was this cust's application declined? | clarify (refuse/partial) | Asked to clarify instead of routing to an account lookup: 'this cust's application' did not match rules written for 'my application'. Fixed: the account rules also read the agent's voice. |
| H11 | How do I add my business partner to the account, and can she get her own debit card? | refuse (answer/partial) | Refused both parts, though the right articles ranked first for each (-4.3 and -1.1). Addressed by the realistic calibration. |
| H12 | PCI SAQ still required if merchant only uses the Mini? | answer (answer/partial) | Answered that every merchant must complete the SAQ, but not the P2PE nuance the label asks for, which sits in another passage of the PCI policy. |

Rerun after the fixes (not blind): 5/12 (42%, 95% CI 19-68%); now passing: H08, H10; newly failing: none.

### Dev failures

- **D01** (How long do I have to respond to a chargeback?): The answer gives the 7-day retrieval-request window instead of the 10-day chargeback deadline. Both sit in the same section, and the cross-encoder ranks the retrieval-request sentence first (6.85) in every model tried (MiniLM-L6, jina-turbo, bge-reranker-v2-m3). *Predicted before the first run.*
- **D23** (My supplier emailed new bank details. Can I still wire them today?): Refused. 'My supplier emailed new bank details' needs the business-email-compromise warning in the wire policy (call a number you already know), but the unit chosen scores -4.7 on the verifier. The right behaviour is a partial answer that leads with the verification warning.
- **D26** (Are any of my fees going up next year?): The reranker prefers the notice's 'how to avoid the increase' paragraph (3.07) to its change table, even after 'next year' is rewritten to 'in 2027'. The answer is about the right notice, but not the list of increases. *Predicted before the first run.*
- **D27** (Which fees are included or free on Bridge Premier?): Aggregation: 'which fees are included or free on Premier' needs every row whose Premier cell says Included. Top-k retrieval plus row rendering returns the two PCI rows. This needs a table query path. *Predicted before the first run.*

### What each defence buys

Each variant removes one mechanism and reruns all 59 questions (dev and both held-out sets); the full system passes 33/59. McNemar's exact test is on the paired outcomes; at this n most differences are directional, and the flipped questions say what each mechanism is for.

| variant | passed | worse on | better on | exact p | questions that flipped |
|---|---|---|---|---|---|
| no validity windows | 29/59 | 4 | 0 | 0.125 | D02, D06, D09, D10 |
| no trust tiers | 27/59 | 6 | 0 | 0.0312 | D02, D03, D06, D08, D09, D16 |
| no structure (top score wins) | 25/59 | 8 | 0 | 0.0078 | D02, D03, D04, D06, D09, D10, V06, V10 |
| no gap check | 32/59 | 1 | 0 | 1.0 | D15 |
| no KB glossary | 33/59 | 0 | 0 | 1.0 | - |
| no filler stripping | 33/59 | 0 | 0 | 1.0 | - |
| no gate (always answer) | 27/59 | 7 | 1 | 0.0703 | D12, D13, D14, D23, H07, H10, V01, V20 |
| no verifier (gate on the ranker's score) | 30/59 | 3 | 0 | 0.25 | D06, H08, V10 |
| no heading context in retrieval | 29/59 | 4 | 0 | 0.125 | D04, D10, D15, V10 |

### Retrieval

Rank of the first unit that carries gold support, over the 54 questions with gold; share in the top k.

| mode | top 1 | top 3 | top 10 | MRR |
|---|---|---|---|---|
| BM25 (body + heading field) | 50% | 74% | 89% | 0.62 |
| dense (bge-small) | 50% | 76% | 89% | 0.65 |
| RRF fusion, no reranker | 61% | 76% | 93% | 0.71 |
| union of 12 + 12, cross-encoder (used) | 56% | 83% | 94% | 0.69 |
| cross-encoder over every unit | 56% | 83% | 89% | 0.68 |

### Thresholds, and what the score cannot see

Floors in use: prose **-1.47**, table **-3.09** (before: 1.73 and -1.74).

- **prose**: 174 on-topic and 67 off-topic calibration questions reach the floor rule. At the floor, off-topic questions above it: 1; on-topic questions below it: 69.
- **table**: 31 on-topic and 10 off-topic calibration questions reach the floor rule. At the floor, off-topic questions above it: 0; on-topic questions below it: 5.
- With every gold document deleted, the best remaining unit still clears its floor for **16 of 24** answerable dev questions (the gold unit clears it for 21 of 24). The floor separates on-topic from off-topic; it cannot separate the right passage from an on-topic distractor, which is why near-misses are decided by structure.

### Risk and coverage

With the gate off, answering every question in score order ends at 54% wrong among answered. The gate answers 71% of all questions with 38% of those failing (any failure, not only unsafe ones). The report's risk-coverage chart shows the whole curve.

The curve in numbers. With the gate off, answer the questions in descending order of the gate's score and stop at each cut:

| questions answered | gate score at the cut | failing among those answered |
|---|---|---|
| 6 of 59 (10%) | +4.90 | 0% |
| 12 of 59 (20%) | +3.21 | 8% |
| 18 of 59 (30%) | +2.54 | 17% |
| 24 of 59 (41%) | +1.94 | 25% |
| 30 of 59 (51%) | +0.33 | 33% |
| 35 of 59 (59%) | -0.44 | 34% |
| 41 of 59 (70%) | -1.07 | 37% |
| 47 of 59 (80%) | -4.71 | 43% |
| 53 of 59 (90%) | -6.14 | 49% |
| 59 of 59 (100%) | -8.28 | 54% |

For comparison, the gate's rules (not a single cut) answer 42 of 59 (71%), with 38% of those failing. Of those 16 failures, 14 have an acceptable decision but miss a required fact, and 2 are wrong decisions (V04, V19). So at this coverage the failure rate is set by answer completeness, and a stricter floor would only trade those failures for coverage: a single cut stays at or under 10% failing only up to 15 of 59 questions answered (25%). What the gate adds is in which questions it does not answer: see the no-gate ablation above.

## What I would do next

Each of these should be measured on a fresh blind set, built the same way, since v1 and v2 have now been seen.

1. **Answers built from several units, with the verifier choosing sentences.** This is the largest failure class on
   held-out v2: 8 of 15 failures decided correctly but missed a fact. Real questions spread their facts over two or
   three passages (V05, V07, V11, V13, V15 and V16 each missed one that sits in another unit). Sentence choice inside
   a passage is still the small ranker's (V17 quoted the sentence next to the endorsement wording, and D01 the
   7-day rule beside the 10-day one). The fix is to verify candidate sentences across the top units and assemble
   the answer from the verified ones, keeping one citation per sentence.
2. **Live-test the Claude path,** which synthesizes across units under per-sentence citation and value checks: the
   same completeness gap. Run it on dev and a fresh held-out set with risk–coverage per generator, and sweep Haiku
   5.5, Sonnet 5.5 and Opus 5.5 at low effort for correctness, abstention, latency and cost per answered question.
3. **Vocabulary from real queries.** "card machines" never meets "BridgeTerminal" (V12) and "intl" reads as a
   missing modifier. Mine agent queries and searches for shorthand and synonyms, and put them in the KB glossary the
   contact-center team already owns.
4. **Calibrate on logged agent questions and feedback** (thumbs, escalations, edits) instead of written ones, and
   fit a small calibrator on the gate's signals. Then the score can honestly become a probability. Held-out v1
   showed how much the calibration questions' wording matters.
5. **Aggregation questions** ("which fees are free on Premier?") need a table query path (filter rows by a
   predicate), not top-k retrieval.
6. **Content governance from the same signals.** Conflict notes and "possibly outdated" flags are a KB-owner work
   queue; refusals are a content-gap backlog. That is where most of the 38% of contact time agents spend searching
   (the brief's estimate) comes back.

## Where to look next

- `README.md`: the whole write-up: how to run it, how it works and why, the dataset, what was cut, how AI tools
  were used.
- `eval/results/report.md`: the summary tables and every question's decision, rule, score and result, in plain
  text.
- `eval/results/report.html`: the same with charts, plus every question's answer, citations and decision trace.
  `eval/results/walkthrough.html`: a guided tour of decision traces, charts and failures. Both are self-contained
  and work offline; GitHub shows `.html` files as source, so clone the repo or download them to view them.
- `eval/results/results.json`: everything above, as data. The blind held-out v1 run is kept unchanged in
  `eval/results/heldout_v1_blind/`, and `eval/results/reranker_selection.json` holds the model comparison.
- To reproduce: `python -m mbfs_rag download-models` once, then `python -m mbfs_rag eval --ablations`.
  `python scripts/package_submission.py` builds the submission zip in `dist/`, which is not committed.
