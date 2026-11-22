# Generation error analysis (answer prompt v1, Claude Haiku 5.5)

**Setup.** We audited 65 generated answers: 50 student queries plus 15 stress cases (`stress_cases.csv`).
- Each answer discusses the top-5 scholarships after Stage A and ranking (α = 0.85).
- The audit has three layers:
  1. 15 deterministic rule checks (`audit_answers.py`);
  2. an LLM judge (Claude Sonnet 5.5, JSON schema output);
  3. manual verification of every flag against the exact context the answer was generated from. The manual verdicts are in `verify.py` and `audit_verified.csv`.

## 1. Surface faithfulness: the rule checks

| check | flags |
|---|---|
| invented or out-of-list citations (R01, R02) | 0 |
| numbers, amounts or years not in the context (R04) | 0 |
| quotes not in the context (R05) | 0 |
| countries, universities or scholarships not in the context (R11, R12) | 0 |
| "fully funded" claimed for a non-full award (R06) | 0 |
| truncated answer, wrong language, missing official-page reminder (R13–R15) | 0 |
| UNKNOWN criterion never mentioned (R07) | 1 (C06/S112: minimum grade) |

The first version of the rules produced 88 flags. Manual inspection showed that 87 of them were artefacts of the audit itself:
- quote marks paired across two different quotes (R05, 48 flags);
- residence wording that does not restrict the student, such as a visa or residence permit, the parents' tax residence, or a student who already lives in the required country (R10, 14);
- paragraphs that had already rejected the award for another reason (R07–R09);
- correct derived numbers (R04): the decimal comma "3,8/5" rendered as 3.8, "next cycle (2027)", "10 days from today", and 844 + 100 + 100 = 1,044 €;
- deliberate reordering when the question asked for a subset (R03).

After fixing these, the rules leave 4 flags: 1 real and 3 acceptable.

**Conclusion.** The generator does not invent facts at the surface level. Every citation, number, quote and named entity can be traced to the context.

## 2. Reasoning errors: the judge plus manual verification

The judge raised **70 issues**. Manual checking against the context gave:
- **61 confirmed (87 %)**;
- 2 debatable;
- **7 judge errors**:
  - 4 concern deadlines. Our context marks an annual date whose 2026 occurrence has already passed as UNKNOWN. The judge read this literally and penalised correct statements such as "10-01 has passed, check the next cycle".
  - 2 penalised the correct remark that 1.5 is a German grade.
  - 1 misread an "offer required" condition.

Overall there are **62 real issues**: 61 from the judge and 1 from the rules. By severity: 2 high, 28 medium, 32 low. They affect 40 of 65 answers. 27 answers have at least one high or medium issue: 20 of the 50 student answers and 7 of the 15 stress cases. 20 student answers are completely clean.

| category | issues | answers | high/medium | typical example | root cause |
|---|---|---|---|---|---|
| **Other decisive condition missed or softened** | 13 | 11 | 8 | P26/S178: an 18-year-old applicant fails "completed bachelor's required", but the answer calls it a "possible fit" | long free-text "Other conditions"; the model reports some of them and skips the decisive one |
| **Tie to a specific university or programme not flagged** | 12 | 12 | 5 | P44, C10/S098: the award is only for the MBA at the Alberta School of Business, yet it is called the "strongest fit" | the schema has no host-institution field, so the tie is buried in "Other conditions" |
| **Detail misread or drifted** | 15 | 14 | 3 | P43/S069: "the list does not include France", although the list was cut off | scraped passages are truncated or have headings glued together, and the model fills the gaps |
| **Unchecked / UNKNOWN condition presented as met** | 9 | 7 | 4 | P01/S241: the page has a closed list of 25 countries, but the answer says "Asia is eligible, so Kazakhstan is" | the context shows MATCH for the five core checks, and the model generalises this to overall fit |
| **Field-of-study mismatch stretched** | 5 | 4 | 4 | C04: an aeronautics programme is called the "closest fit" for a chemistry postdoc | ranking ignores the field, so off-field awards reach the top-5 |
| **Wrong "only / strongest" claim** | 3 | 3 | 3 | P02, C15/S199: "the only fully funded option", although other items also list `full` | superlatives need a check across all items, and funding type is multi-valued (`full;partial`) |
| **Deadline misread** | 3 | 3 | 2 | P16/S027: annual 11-12 has passed, but the answer says "check whether it is still open" | the context gives raw dates (`annual:MM-DD`, `YYYY-MM`) instead of a computed status |
| **Grade scales mixed** | 2 | 2 | 1 | P41/S107: "your 3.7 is above the 1.5 cutoff" | the context shows the raw threshold, not the 4.0 value the filter used |

Errors cluster on a few scholarships whose conditions are long and complex. S136 (King Fahd) and S177 (Rosa-Luxemburg-Stiftung) account for 6 issues each, and S178 and S107 for 4 each.

## 3. Stress cases

| result | cases |
|---|---|
| expected behaviour met (11) | C01 no match in Japan, C02 prompt injection, C03 question in Russian (answered in English), C05 US citizen vs international-only awards, C06 amounts, C07 deadlines in 2026, C08 17-year-old (correctly rejected on the minimum age of 19), C09 sanctions, C11 residence and part-time, C13 current student, C14 Korean nationals excluded |
| partly (3) | C10 women-only MBA: the institution tie was missed. C12 history PhD: an unrelated award was called relevant. C15 "only fully funded": the "only one" claim was wrong |
| not met (1) | C04 chemistry postdoc: the answer did not say plainly that nothing fits, and stretched unrelated awards |

## 4. What this means and how to fix it (not applied)

The generator is faithful to its context. Its errors are reasoning errors about conditions **outside the five structured criteria**. This is the same gap that leaves 24–28 % ineligible scholarships in the top-10 at the retrieval stage. Possible fixes, by pipeline layer:

| layer | fix | addresses |
|---|---|---|
| extraction / Stage A | add a `host_institution` field (must be admitted at X → UNKNOWN for applicants) | INST |
| extraction / Stage A | soft field-of-study check or ranking feature | FIELD |
| context builder (code, no prompting) | computed deadline status ("2026 date passed → next 2027-11-12"; "opens 2026-12") | DEADLINE, judge errors |
| context builder | show the converted threshold ("1.5 German ≈ 3.5 / 4.0") | SCALE |
| context builder | label the checks "5 core checks passed, N other conditions unchecked" | OVERCONF |
| prompt | one verdict per "Other condition" (met / not met / unknown); no "only / strongest" without comparing all items; never infer absence from a truncated list; say plainly when no item fits the field | BARRIER, COMPARE, MISREAD, FIELD |

## 5. Limitations of the audit

- **Precision, not recall.** We verified every judge flag, so precision is known (87 %). We did not measure recall: errors the judge missed are not counted, so 62 is a lower bound.
- **Judge independence.** The judge is a single run of a model from the same vendor as the generator. Severity labels are the judge's own.
- **Optimistic re-evaluation.** The fixes above were derived from these same 65 answers. Re-running on the same set would overstate the improvement, so a fair re-evaluation needs new queries.
