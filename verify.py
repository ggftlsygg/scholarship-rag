"""Manual verification of every audit issue (rules after the false-alarm fixes + LLM judge)."""
import json, sys, pandas as pd
V = {  # (job, scholarship, check) -> (verdict, category, note)
("P01","S241","J unknown_as_confirmed"):("real","OVERCONF","region 'Asia' read as 'Kazakhstan eligible'; page has a closed list of 25 countries (Stage A also says MATCH)"),
("P01","S136","J unsupported_claim"):("real","OVERCONF","GPA in the major field is not in the profile"),
("P02","S199","J contradicts_context"):("real","COMPARE","'only fully funded option' although S031 is also full"),
("P02","S113","J unsupported_claim"):("real","MISREAD","Kazakhstan named in one fellowship, not in the general rule"),
("P04","S107","J unsupported_claim"):("judge_error","","1.5 is a German grade; correct background knowledge, conclusion (UNKNOWN) is right"),
("P05","S177","J missed_barrier"):("real","BARRIER","BAfoeG s.8 (German passport / permanent-residence prospects) not explained; called 'strongest match'"),
("P05","S035","J missed_barrier"):("real","INST","only exchange semesters at a partner institution"),
("P07","S215","J wrong_fit_judgement"):("judge_error","","the page requires an offer; an applicant does not hold one - the answer is right"),
("P08","S178","J missed_barrier"):("real","BARRIER","international programme needs a completed bachelor's; student is a bachelor applicant"),
("P09","S067","J missed_barrier"):("real","INST","relAI is tied to a master's at TUM / LMU"),
("P10","S189","J missed_barrier"):("real","INST","A*STAR supervisor + PhD at a Singapore autonomous university"),
("P11","S018","J unsupported_claim"):("debatable","MISREAD","scraped programme list is ambiguous (MSc/LLM labels glued together)"),
("P11","S227","J wrong_fit_judgement"):("real","BARRIER","partnership money is for students in India; student lives in the UK"),
("P11","S173","J missed_barrier"):("real","INST","University of London offers only (answer already says 'not a fit')"),
("P13","S139","J deadline_error"):("real","DEADLINE","window starting 2026-12 described as overlapping today"),
("P16","S027","J deadline_error"):("real","DEADLINE","annual 11-12 already passed, answer says 'check whether still open'"),
("P16","S177","J unsupported_claim"):("real","MISREAD","paraphrase changed who the limited portion is for"),
("P17","S018","J missed_barrier"):("real","BARRIER","admission required; 'not studied abroad before' and 'no other funding' omitted"),
("P17","S153","J missed_barrier"):("real","BARRIER","first degree in a listed subject + English test omitted"),
("P19","S133","J unsupported_claim"):("real","MISREAD","English exam rule is for Masters/PhD, applied to a bachelor applicant"),
("P20","S134","J unsupported_claim"):("real","MISREAD","exemption depends on the degree-awarding country, not nationality"),
("P22","S193","J unknown_as_confirmed"):("real","OVERCONF","degree level UNKNOWN presented as compatible"),
("P22","S136","J unsupported_claim"):("real","MISREAD","B.S.+M.S. GPA rule is for PhD, applied to master"),
("P22","S185","J unsupported_claim"):("real","MISREAD","'want to work in the energy industry' - narrower than the page"),
("P24","S134","J wrong_fit_judgement"):("real","FIELD","science & technology university ranked promising for an IR master"),
("P25","S134","J deadline_error"):("judge_error","","upcoming 2027-01-03 date is given; omitting the passed annual date does not mislead"),
("P26","S178","J wrong_fit_judgement"):("real","BARRIER","clear failure (no completed bachelor's) softened to 'possible fit'"),
("P27","S136","J missed_barrier"):("real","OVERCONF","four-year bachelor's + major-field GPA not flagged"),
("P28","S136","J contradicts_context"):("real","COMPARE","'only scholarship with an open deadline' although S107 is rolling"),
("P28","S136","J unknown_as_confirmed"):("real","OVERCONF","bachelor's GPA unknown, presented as met"),
("P28","S107","J unsupported_claim"):("real","MISREAD","'history' added to a truncated field list"),
("P28","S189","J wrong_fit_judgement"):("real","OVERCONF","'strong match on paper' despite unconfirmed Singapore-degree rule"),
("P30","S177","J missed_barrier"):("real","BARRIER","BAfoeG s.8 meaning not explained for a student living in China"),
("P31","S117","J missed_barrier"):("real","BARRIER","NAACP membership omitted (answer already says 'not a fit')"),
("P32","S185","J unsupported_claim"):("real","MISREAD","'energy sectors' invented from 'sectors of the future'"),
("P33","S177","J deadline_error"):("judge_error","","annual 10-01 has passed for 2026 - the answer is right"),
("P33","S177","J unsupported_claim"):("real","MISREAD","'most applicants restricted' overstates the page"),
("P36","S235","J missed_barrier"):("real","INST","must be enrolled at the University of Montana"),
("P38","S107","J wrong_fit_judgement"):("real","SCALE","US 3.7 vs German 1.5 echoed as MATCH without explaining the scale"),
("P38","S177","J missed_barrier"):("real","BARRIER","BAfoeG s.8 condition omitted"),
("P40","S177","J unknown_as_confirmed"):("judge_error","","'10-01 passed, next cycle 2027' is correct date arithmetic"),
("P40","S107","J unsupported_claim"):("judge_error","","1.5 is a German grade; correct background knowledge"),
("P41","S177","J unknown_as_confirmed"):("real","INST","'accepted at a German university' presented as passed for an applicant"),
("P41","S107","J other"):("real","SCALE","'3.7 is above the 1.5 cutoff' mixes scales"),
("P42","S113","J unsupported_claim"):("real","MISREAD","heading fragment read as 'Pickering fellowship for Germans'; generic page called 'strongest match'"),
("P43","S069","J unsupported_claim"):("real","MISREAD","absence of France inferred from a truncated country list"),
("P43","S177","J deadline_error"):("judge_error","","annual 10-01 has passed for 2026 - the answer is right"),
("P43","S087","J missed_barrier"):("real","INST","must be enrolled at ENS-PSL (answer already says 'likely not a fit')"),
("P44","S098","J missed_barrier"):("real","INST","MBA at the Alberta School of Business; called 'strongest fit'"),
("P47","S136","J unknown_as_confirmed"):("real","OVERCONF","'3.1 GPA meets the requirements', later contradicted"),
("P48","S124","J missed_barrier"):("real","INST","enrolment/acceptance at a German university"),
("P50","S178","J unknown_as_confirmed"):("real","OVERCONF","nationality UNKNOWN treated as settled"),
("C03","S170","J missed_barrier"):("real","FIELD","finance/economics courses for a CS student, not raised"),
("C04","","J ignored_rule"):("real","FIELD","stress case: did not say plainly that nothing fits a chemistry postdoc"),
("C04","S070","J wrong_fit_judgement"):("real","FIELD","aeronautics/space programme called 'closest fit' for chemistry"),
("C04","S047","J missed_barrier"):("debatable","FIELD","work location is not in the profile; biomedical-only hedged with 'probably'"),
("C04","S154","J unsupported_claim"):("real","MISREAD","'research degree program' not in the context"),
("C05","S157","J wrong_fit_judgement"):("real","BARRIER","$22,000 is for international students; student is a US citizen"),
("C06","S124","J missed_barrier"):("real","INST","enrolment/acceptance at a German university"),
("C09","S124","J missed_barrier"):("real","INST","enrolment/acceptance at a German university"),
("C10","S098","J missed_barrier"):("real","INST","MBA at the Alberta School of Business; 'matches every condition'"),
("C10","S051","J unsupported_claim"):("real","MISREAD","nationality criterion called residence"),
("C10","S152","J wrong_fit_judgement"):("real","BARRIER","admitted student clearly has not completed year 1; shown as missing info"),
("C10","S152","J missed_barrier"):("real","BARRIER","preference for Alberta residents omitted"),
("C12","S094","J wrong_fit_judgement"):("real","FIELD","ELES (Jewish candidates, any field) called most relevant for history"),
("C12","S107","J other"):("real","MISREAD","age limit applies at graduation; 'within the age limit' not supported"),
("C13","S178","J unsupported_claim"):("real","MISREAD","domestic-programme rule mixed into the international programme"),
("C14","S091","J missed_barrier"):("real","BARRIER","first-time applicant, no EU dual citizenship, no combining omitted"),
("C14","S091","J deadline_error"):("real","DEADLINE","specific 2027 dates called 'annual'"),
("C15","S199","J contradicts_context"):("real","COMPARE","'only fully funded' although S112/S113 also list full"),
("C01","S223","R07 unknown_not_flagged"):("acceptable","","stress case (off-topic request); award rejected on field, unknowns irrelevant"),
("C06","S112","R07 unknown_not_flagged"):("real","OVERCONF","minimum grade UNKNOWN not mentioned"),
("C01","","R03 order"):("acceptable","","answer reorders on purpose (question asks about Japan / medicine)"),
("C15","","R03 order"):("acceptable","","answer groups full vs not-full awards on purpose"),
}
CAT = {"INST": "Tie to a specific university/programme not flagged",
       "BARRIER": "Other decisive condition missed or softened",
       "OVERCONF": "Unchecked/UNKNOWN condition presented as met",
       "MISREAD": "Detail misread or drifted from a noisy/truncated passage",
       "FIELD": "Field-of-study mismatch stretched",
       "COMPARE": "Wrong 'only / strongest' claim across the list",
       "DEADLINE": "Deadline misread",
       "SCALE": "Grade scales mixed (German 1.5 vs US 3.7)"}
CAUSE = {"INST": "no structured host-institution field; the tie sits inside free-text 'Other conditions'",
         "BARRIER": "long 'Other conditions' lists; the model reports some conditions and skips the decisive one",
         "OVERCONF": "context shows MATCH for the five core checks and the model generalises it to overall fit",
         "MISREAD": "scraped passages are truncated or have glued headings; the model fills gaps",
         "FIELD": "ranking ignores field of study, so off-field awards reach the top-5",
         "COMPARE": "superlatives need a check across all items; funding type is multi-valued (full;partial)",
         "DEADLINE": "context gives raw dates (annual:MM-DD, YYYY-MM) instead of a computed status",
         "SCALE": "context shows the raw threshold, not the converted 4.0 value used by the filter"}
issues = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "audit_issues.csv")   # run audit_answers.py --judge first (judge answers come from judge_cache, no new API calls)
issues["key"] = list(zip(issues.job_id, issues.scholarship_id.fillna(""), issues.check))
missing = [k for k in issues.key if k not in V]
assert not missing, missing
issues["verdict"] = [V[k][0] for k in issues.key]
issues["category"] = [V[k][1] for k in issues.key]
issues["category_name"] = issues.category.map(CAT).fillna("")
issues["note"] = [V[k][2] for k in issues.key]
issues.drop(columns="key").to_csv("audit_verified.csv", index=False)
real = issues[issues.verdict == "real"]
j = issues[issues.source == "judge"]
print("judge:", len(j), j.verdict.value_counts().to_dict())
print("rules:", (issues.source == "rule").sum(), issues[issues.source == "rule"].verdict.value_counts().to_dict())
tab = (real.groupby("category").agg(issues=("job_id", "size"), answers=("job_id", "nunique"),
                                     high_or_medium=("severity", lambda s: int(s.isin(["high", "medium"]).sum())))
       .reindex(list(CAT)).reset_index())
tab["description"] = tab.category.map(CAT); tab["root_cause"] = tab.category.map(CAUSE)
tab.to_csv("error_categories.csv", index=False)
print(tab[["category", "issues", "answers", "high_or_medium"]].to_string(index=False))
jobs = [json.loads(l) for l in open("answers_log.jsonl")]
std = [p["job_id"] for p in jobs if not p["job_id"].startswith("C")]
rm = real[real.severity.isin(["high", "medium"])]
print("answers:", len(jobs), "| with any real issue:", real.job_id.nunique(), "| with real high/medium:", rm.job_id.nunique())
print("student answers (50) with real high/medium:", rm[rm.job_id.isin(std)].job_id.nunique(),
      "| clean (no real issue):", len(set(std) - set(real.job_id)))
print("stress (15) with real high/medium:", rm[~rm.job_id.isin(std)].job_id.nunique())
print("real by severity:", real.severity.value_counts().to_dict())
# scholarships that recur
print(real.groupby("scholarship_id").size().sort_values(ascending=False).head(8).to_dict())
