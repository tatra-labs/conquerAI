# MBFS RAG - evaluation report

Answers as of 2026-10-19 (corpus snapshot). Frozen inputs: dev ec8ca229e011098e, heldout_v1 7ce1a409fda69ba5, heldout_v2 8b064c86d068cc17.

## dev (n=27)

| measure | result |
|---|---|
| passed | 23/27 (85%, 95% CI 68-94%) |
| decision ok | 26/27 (96%, 95% CI 82-99%) |
| decision exact | 24/27 (89%, 95% CI 72-96%) |
| false answers | 0/3 (0%, 95% CI 0-56%) |
| false refusals | 1/24 (4%, 95% CI 1-20%) |
| fact recall | 26/32 (81%, 95% CI 65-91%) |
| citation support | 26/32 (81%, 95% CI 65-91%) |
| unsafe answers | 0 |
| latency (warm, median / max) | 815 ms / 2120 ms |

## heldout_v1 (n=12)

| measure | result |
|---|---|
| passed | 5/12 (42%, 95% CI 19-68%) |
| decision ok | 8/12 (67%, 95% CI 39-86%) |
| decision exact | 8/12 (67%, 95% CI 39-86%) |
| false answers | 0/2 (0%, 95% CI 0-66%) |
| false refusals | 4/10 (40%, 95% CI 17-69%) |
| fact recall | 7/25 (28%, 95% CI 14-48%) |
| citation support | 7/25 (28%, 95% CI 14-48%) |
| unsafe answers | 0 |
| latency (warm, median / max) | 749.5 ms / 1698 ms |

## heldout_v2 (n=20)

| measure | result |
|---|---|
| passed | 5/20 (25%, 95% CI 11-47%) |
| decision ok | 13/20 (65%, 95% CI 43-82%) |
| decision exact | 13/20 (65%, 95% CI 43-82%) |
| false answers | 1/3 (33%, 95% CI 6-79%) |
| false refusals | 5/17 (29%, 95% CI 13-53%) |
| fact recall | 12/42 (29%, 95% CI 17-44%) |
| citation support | 12/42 (29%, 95% CI 17-44%) |
| unsafe answers | 1 |
| latency (warm, median / max) | 740.0 ms / 1755 ms |

## Per question

| id | split | decision | rule | score | result | stage | predicted failure |
|---|---|---|---|---|---|---|---|
| D01 | dev | answer | answer | 4.07 | FAIL | missing fact | Every reranker tried (MiniLM-L6, jina-turbo, bge-reranker-v2-m3) scores the retrieval-request… |
| D02 | dev | answer | answer | 3.63 | pass | ok |  |
| D03 | dev | answer | answer | 5.87 | pass | ok |  |
| D04 | dev | answer | answer | -1.07 | pass | ok |  |
| D05 | dev | answer | answer | 5.61 | pass | ok |  |
| D06 | dev | answer | answer + answer | -0.7 | pass | ok |  |
| D07 | dev | partial | uncovered_modifier + answer | 2.89 | pass | ok |  |
| D08 | dev | answer | answer | 2.96 | pass | ok |  |
| D09 | dev | answer | answer | 2.48 | pass | ok |  |
| D10 | dev | answer | answer | 2.34 | pass | ok | Rows from different sections with different names; facet narrowing may keep the wrong subset. |
| D11 | dev | answer | answer | 5.22 | pass | ok |  |
| D12 | dev | refuse | account_lookup | -5.61 | pass | ok |  |
| D13 | dev | refuse | no_relevant_evidence | -6.13 | pass | ok |  |
| D14 | dev | refuse | no_relevant_evidence | -5.52 | pass | ok |  |
| D15 | dev | partial | uncovered_modifier | 2.32 | pass | ok |  |
| D16 | dev | answer | answer | 4.9 | pass | ok |  |
| D17 | dev | answer | answer | 6.73 | pass | ok |  |
| D18 | dev | answer | answer | 4.1 | pass | ok |  |
| D19 | dev | answer | answer | 8.57 | pass | ok |  |
| D20 | dev | answer | answer | -0.14 | pass | ok |  |
| D21 | dev | answer | answer | 1.7 | pass | ok |  |
| D22 | dev | answer | answer | -0.59 | pass | ok |  |
| D23 | dev | refuse | no_relevant_evidence | -4.71 | FAIL | decision (refuse, wanted answer/partial) |  |
| D24 | dev | answer | answer | 0.31 | pass | ok |  |
| D25 | dev | answer | answer | 2.54 | pass | ok |  |
| D26 | dev | answer | answer | 1.89 | FAIL | missing fact | The reranker prefers the notice's 'how to avoid the increase' paragraph to its change table, even… |
| D27 | dev | answer | answer | 2.57 | FAIL | missing fact | Aggregation across many rows is out of reach for top-k retrieval plus row rendering; expect a… |
| H01 | heldout_v1 | refuse | no_relevant_evidence | -6.14 | FAIL | decision (refuse, wanted partial/answer) |  |
| H02 | heldout_v1 | answer | answer | 4.6 | pass | ok |  |
| H03 | heldout_v1 | answer | answer | 2.48 | FAIL | missing fact |  |
| H04 | heldout_v1 | refuse | no_relevant_evidence | -3.87 | FAIL | decision (refuse, wanted answer/clarify/partial) |  |
| H05 | heldout_v1 | answer | answer | 2.96 | pass | ok |  |
| H06 | heldout_v1 | answer | answer | 2.37 | FAIL | missing fact |  |
| H07 | heldout_v1 | refuse | no_relevant_evidence | -7.58 | pass | ok |  |
| H08 | heldout_v1 | answer | answer | 0.71 | pass | ok |  |
| H09 | heldout_v1 | refuse | outdated_only | None | FAIL | decision (refuse, wanted answer/clarify) |  |
| H10 | heldout_v1 | refuse | account_lookup | -8.28 | pass | ok |  |
| H11 | heldout_v1 | refuse | no_relevant_evidence + no_relevant_evidence | -5.53 | FAIL | decision (refuse, wanted answer/partial) |  |
| H12 | heldout_v1 | answer | answer | 0.14 | FAIL | missing fact |  |
| V01 | heldout_v2 | refuse | no_relevant_evidence | -7.02 | pass | ok |  |
| V02 | heldout_v2 | refuse | no_relevant_evidence | -6.48 | FAIL | decision (refuse, wanted partial/answer) |  |
| V03 | heldout_v2 | answer | answer | 3.21 | pass | ok |  |
| V04 | heldout_v2 | answer | answer | -0.44 | FAIL | decision (answer, wanted partial/clarify) |  |
| V05 | heldout_v2 | answer | answer | -2.75 | FAIL | missing fact |  |
| V06 | heldout_v2 | answer | answer | 4.6 | pass | ok |  |
| V07 | heldout_v2 | answer | answer | -0.77 | FAIL | missing fact |  |
| V08 | heldout_v2 | answer | answer | -1.02 | FAIL | missing fact |  |
| V09 | heldout_v2 | refuse | no_relevant_evidence + no_relevant_evidence | -3.13 | FAIL | decision (refuse, wanted partial) |  |
| V10 | heldout_v2 | answer | answer | -0.21 | pass | ok |  |
| V11 | heldout_v2 | answer | answer | 0.33 | FAIL | missing fact |  |
| V12 | heldout_v2 | refuse | no_relevant_evidence | -3.43 | FAIL | decision (refuse, wanted answer/clarify) |  |
| V13 | heldout_v2 | answer | answer | 0.76 | FAIL | missing fact |  |
| V14 | heldout_v2 | refuse | no_relevant_evidence | -4.53 | FAIL | decision (refuse, wanted partial) |  |
| V15 | heldout_v2 | answer | answer | 1.94 | FAIL | missing fact |  |
| V16 | heldout_v2 | answer | answer | 2.67 | FAIL | missing fact |  |
| V17 | heldout_v2 | answer | answer | 1.01 | FAIL | missing fact |  |
| V18 | heldout_v2 | refuse | no_relevant_evidence + no_relevant_evidence | -6.95 | FAIL | decision (refuse, wanted answer/partial) |  |
| V19 | heldout_v2 | answer | answer | -0.86 | FAIL | decision (answer, wanted clarify) |  |
| V20 | heldout_v2 | refuse | no_relevant_evidence | -7.3 | pass | ok |  |

## Ablations (pass/fail per item; McNemar exact test vs full)

| variant | full-only passes | variant-only passes | p | items that flipped |
|---|---|---|---|---|
| no validity windows | 4 | 0 | 0.125 | D02, D06, D09, D10 |
| no trust tiers | 6 | 0 | 0.0312 | D02, D03, D06, D08, D09, D16 |
| no structure | 8 | 0 | 0.0078 | D02, D03, D04, D06, D09, D10, V06, V10 |
| no gap check | 1 | 0 | 1.0 | D15 |
| no glossary | 0 | 0 | 1.0 |  |
| no filler strip | 0 | 0 | 1.0 |  |
| no gate | 7 | 1 | 0.0703 | D12, D13, D14, D23, H07, H10, V01, V20 |
| no verifier | 3 | 0 | 0.25 | D06, H08, V10 |
| no headings | 4 | 0 | 0.125 | D04, D10, D15, V10 |

## Threshold choice

Floors from the realistic calibration set (written from the corpus alone; 10:1 cost of a wrong answer to a refusal; fitted on the score the gate tests, for questions that reach the floor rule):

- prose: floor -1.47 (174 on-topic / 67 off-topic; 1 off-topic above, 69 on-topic below)
- table: floor -3.09 (31 on-topic / 10 off-topic; 0 off-topic above, 5 on-topic below)
- in use: {'relevance_prose': 1.73, 'relevance_table': -1.74, 'answer_prose': -1.47, 'answer_table': -3.09, 'high_margin': 4.0, 'authority_margin': 3.0, 'row_margin': 1.0, 'sentence_margin': 2.5, 'competing_margin': 1.5, 'miss_penalty': 2.0}

## Invariance (paraphrases of dev questions)

7/12 keep the same decision and cite the same documents.

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
| D20 | is a 1099-K still coming my way if card sales were just $3,000? | answer | answer | True |
| D22 | got $400k sitting in Reserve Savings, is that covered by the FDIC? | answer | refuse | False |
