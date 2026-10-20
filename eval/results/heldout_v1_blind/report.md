# MBFS RAG - evaluation report

Answers as of 2026-10-19 (corpus snapshot). Frozen inputs: dev e51c84360794ccd7, heldout 7ce1a409fda69ba5.

## dev (n=27)

| measure | result |
|---|---|
| passed | 22/27 (81%, 95% CI 63-92%) |
| decision ok | 26/27 (96%, 95% CI 82-99%) |
| decision exact | 23/27 (85%, 95% CI 68-94%) |
| false answers | 0/3 (0%, 95% CI 0-56%) |
| false refusals | 1/24 (4%, 95% CI 1-20%) |
| fact recall | 25/32 (78%, 95% CI 61-89%) |
| citation support | 25/32 (78%, 95% CI 61-89%) |
| unsafe answers | 0 |
| latency (warm, median / max) | 546 ms / 1274 ms |

## heldout (n=12)

| measure | result |
|---|---|
| passed | 3/12 (25%, 95% CI 9-53%) |
| decision ok | 6/12 (50%, 95% CI 25-75%) |
| decision exact | 6/12 (50%, 95% CI 25-75%) |
| false answers | 0/2 (0%, 95% CI 0-66%) |
| false refusals | 4/10 (40%, 95% CI 17-69%) |
| fact recall | 5/25 (20%, 95% CI 9-39%) |
| citation support | 5/25 (20%, 95% CI 9-39%) |
| unsafe answers | 1 |
| latency (warm, median / max) | 534.5 ms / 997 ms |

## Per question

| id | split | decision | rule | score | result | stage | predicted failure |
|---|---|---|---|---|---|---|---|
| D01 | dev | answer | answer | 6.85 | FAIL | missing fact | Every reranker tried (MiniLM-L6, jina-turbo, bge-reranker-v2-m3) scores the retr |
| D02 | dev | answer | answer | 5.06 | pass | ok |  |
| D03 | dev | answer | answer | 8.26 | pass | ok |  |
| D04 | dev | answer | answer | 5.27 | pass | ok |  |
| D05 | dev | answer | answer | 7.21 | pass | ok |  |
| D06 | dev | partial | no_relevant_evidence + answer | 1.37 | FAIL | missing fact |  |
| D07 | dev | partial | uncovered_modifier + answer | 5.72 | pass | ok |  |
| D08 | dev | answer | answer | 6.7 | pass | ok |  |
| D09 | dev | answer | answer | 6.86 | pass | ok |  |
| D10 | dev | answer | answer | 3.72 | pass | ok | Rows from different sections with different names; facet narrowing may keep the  |
| D11 | dev | answer | answer | 8.51 | pass | ok |  |
| D12 | dev | refuse | account_lookup | -6.0 | pass | ok |  |
| D13 | dev | refuse | missing_key_term | -1.71 | pass | ok |  |
| D14 | dev | refuse | no_relevant_evidence | -4.93 | pass | ok |  |
| D15 | dev | partial | uncovered_modifier | 6.82 | pass | ok |  |
| D16 | dev | answer | answer | 8.15 | pass | ok |  |
| D17 | dev | answer | answer | 9.78 | pass | ok |  |
| D18 | dev | answer | answer | 3.79 | pass | ok |  |
| D19 | dev | answer | answer | 10.77 | pass | ok |  |
| D20 | dev | answer | answer | 3.4 | pass | ok |  |
| D21 | dev | answer | answer | 5.4 | pass | ok |  |
| D22 | dev | answer | answer | 3.95 | pass | ok |  |
| D23 | dev | refuse | no_relevant_evidence | 0.26 | FAIL | decision (refuse, wanted answer/partial) |  |
| D24 | dev | answer | answer | 3.91 | pass | ok |  |
| D25 | dev | answer | answer | 6.05 | pass | ok |  |
| D26 | dev | answer | answer | 3.07 | FAIL | missing fact | The reranker prefers the notice's 'how to avoid the increase' paragraph to its c |
| D27 | dev | answer | answer | 6.8 | FAIL | missing fact | Aggregation across many rows is out of reach for top-k retrieval plus row render |
| H01 | heldout | refuse | no_relevant_evidence | -6.23 | FAIL | decision (refuse, wanted partial/answer) |  |
| H02 | heldout | answer | answer | 7.63 | pass | ok |  |
| H03 | heldout | answer | answer | 3.02 | FAIL | missing fact |  |
| H04 | heldout | refuse | no_relevant_evidence | 0.22 | FAIL | decision (refuse, wanted answer/clarify/partial) |  |
| H05 | heldout | answer | answer | 6.69 | pass | ok |  |
| H06 | heldout | partial | answer + no_relevant_evidence | -1.72 | FAIL | decision (partial, wanted answer) |  |
| H07 | heldout | refuse | no_relevant_evidence | -6.81 | pass | ok |  |
| H08 | heldout | refuse | no_relevant_evidence | -3.28 | FAIL | decision (refuse, wanted answer) |  |
| H09 | heldout | answer | answer | 2.33 | FAIL | wrong or stale value |  |
| H10 | heldout | clarify | no_topical_term | -10.27 | FAIL | decision (clarify, wanted refuse/partial) |  |
| H11 | heldout | refuse | no_relevant_evidence + no_relevant_evidence | -4.33 | FAIL | decision (refuse, wanted answer/partial) |  |
| H12 | heldout | answer | answer | 4.28 | FAIL | missing fact |  |

## Ablations (pass/fail per item; McNemar exact test vs full)

| variant | full-only passes | variant-only passes | p | items that flipped |
|---|---|---|---|---|
| no validity windows | 3 | 0 | 0.25 | D02, D09, D10 |
| no trust tiers | 5 | 0 | 0.0625 | D02, D03, D08, D09, D16 |
| no structure | 5 | 0 | 0.0625 | D02, D03, D04, D09, D10 |
| no gap check | 2 | 0 | 0.5 | D13, D15 |
| no glossary | 0 | 0 | 1.0 |  |
| no filler strip | 0 | 0 | 1.0 |  |
| no gate | 4 | 2 | 0.6875 | D12, D13, D14, D23, H07, H08 |
| no headings | 3 | 0 | 0.25 | D04, D10, D15 |

## Threshold choice

Floors from the synthetic calibration set (10:1 cost of a wrong answer to a refusal):

- prose: floor 1.73 (151 on-topic / 24 off-topic; 0 off-topic above, 0 on-topic below)
- table: floor -1.74 (54 on-topic / 4 off-topic; 0 off-topic above, 0 on-topic below)
- in use: {'answer_prose': 1.73, 'answer_table': -1.74, 'high_margin': 4.0, 'authority_margin': 3.0, 'row_margin': 1.0, 'sentence_margin': 2.5, 'competing_margin': 1.5, 'miss_penalty': 2.0}

## Invariance (paraphrases of dev questions)

6/12 keep the same decision and cite the same documents.

| base | paraphrase | base decision | paraphrase decision | same sources |
|---|---|---|---|---|
| D02 | cust on the Plus plan wants to send an intl wire out - what do we charge for that? | answer | partial | True |
| D03 | Bridge Essentials - what do you guys charge per month for it? | answer | answer | True |
| D04 | if I park $30k in Bridge Reserve Savings, what APY am I getting on that? | answer | answer | True |
| D05 | what fee do you guys hit me with when I get a chargeback? | answer | answer | True |
| D08 | if I leave $3k in my Bridge Essentials acct, does that get me out of the monthly fee? | answer | answer | False |
| D12 | can you pull up the balance right now on the acct ending in 4417? | refuse | refuse | True |
| D13 | need a cashier's check, what's the fee for one? | refuse | refuse | True |
| D15 | biz credit card - what do you charge on international transactions? | partial | refuse | False |
| D16 | intl wires - what time's the deadline to get them in? | answer | refuse | False |
| D19 | do I make any interest keeping money in Bridge business checking? | answer | answer | True |
| D20 | is a 1099-K still coming my way if card sales were just $3,000? | answer | refuse | False |
| D22 | got $400k sitting in Reserve Savings, is that covered by the FDIC? | answer | refuse | False |
