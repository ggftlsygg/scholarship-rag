"""Audit of generated answers: deterministic rule checks + optional LLM judge.

Input : answers_log.jsonl from generate_answers.py (stores the exact context each answer was written from).
Usage :
  python audit_answers.py                              # rule checks only (free)
  python audit_answers.py --judge                      # + LLM judge (default Claude Sonnet 5.5, ~1-2 USD for 65 answers)
  python audit_answers.py --judge --judge-model claude-haiku-5-5     # cheaper judge (same model as the generator)
Output: audit_issues.csv (one row per issue), audit_summary.csv (issues per check), audit_by_job.csv
Rule checks
  R01 missing_citation        a top-N scholarship is never cited
  R02 foreign_citation        an id is cited that is not in the top-N list
  R03 order                   scholarships discussed out of rank order
  R04 unsupported_number      a number / amount / % / year in the answer does not occur in the context
  R05 unverified_quote        a quoted passage (>= 20 chars) is not in the context
  R06 fully_funded_claim      'fully funded' although funding type is not 'full' and the evidence never says so
  R07 unknown_not_flagged     the scholarship has UNKNOWN criteria but its paragraph never says to check them
  R08 annual_deadline         only annual deadlines, but the paragraph never mentions the next cycle / checking
  R09 gender_age              gender or age condition present but the paragraph does not relate it to the student
  R10 residence               residence condition present but the paragraph does not relate it to the student
  R11 country_not_in_context  a country is named that appears neither in the context nor in the profile
  R12 entity_not_in_context   a university / well-known scholarship is named that is not in the context
  R13 missing_reminder        no closing reminder to check the official pages
  R14 not_english             non-Latin script in the answer
  R15 truncated_or_error      empty answer, API error or stop_reason != end_turn
"""
import argparse, hashlib, json, os, re, sys
import pandas as pd
from criteria_common import countries_in, expand_countries

def norm(t):
    t = (t or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", t).strip().lower()

def loose(t):
    """Lower-case letters and digits only: quote matching that ignores punctuation, quote marks and spacing."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", norm(t))).strip()

FULL_RE = re.compile(r"fully[- ]funded|full funding|full scholarship", re.I)
NEG_RE = re.compile(r"\b(not|no|never|nor|isn't|aren't|wasn't|doesn't|don't|cannot|rather than|instead of|without|"
                    r"unclear whether|unknown whether|out of|left out|exclude[sd]?)\b[^.;]{0,40}$", re.I)

def positive_full_claim(para):
    """True if the paragraph calls the award fully funded without negating it ('not fully funded' is fine)."""
    return any(not NEG_RE.search(para[max(0, m.start() - 60):m.start()]) for m in FULL_RE.finditer(para))

def split_paragraphs(answer, top_ids):
    """Map each scholarship id to the text of the paragraph(s) that discuss it (first id cited in a paragraph)."""
    out = {i: "" for i in top_ids}
    for para in re.split(r"\n\s*\n", answer or ""):
        ids = re.findall(r"\[(S\d{3})\]", para)
        if ids and ids[0] in out: out[ids[0]] += " " + para
    return out

NUM_RE = re.compile(r"(?<![\w.])[$€£]?\s?\d[\d,]*(?:\.\d+)?\s?%?")
ENTITY_RE = re.compile(r"\b(?:University of [A-Z][\w-]+(?: [A-Z][\w-]+)?|[A-Z][\w-]+ University|Harvard|Oxford|Cambridge|"
                       r"Stanford|MIT|Yale|Princeton|Fulbright|Chevening|Erasmus|DAAD|MEXT|Rhodes|Gates)\b")
NOFIT_RE = re.compile(r"not a (good )?fit|poor fit|not (a )?match|do(es)? not meet|not eligible|cannot apply|"
                      r"left (it|them) out|does not qualify|would not qualify", re.I)
DATE_OK_RE = re.compile(r"next (cycle|year|round|intake)|passed|check|confirm|verify|annual|yearly|each year|every year|"
                        r"days? (away|left|from today)|still (ahead|open)|upcoming|after today|not (yet )?passed|"
                        r"january|february|march|april|may|june|july|august|september|october|november|december", re.I)
RES_SKIP_RE = re.compile(r"permit|visa|tax|educated in|cannot|not eligible|excluded?|non-|without", re.I)

def residence_barrier(other_hard, lives_in):
    """Residence clauses that name a place the student does not live in (e.g. 'must reside in Germany')."""
    out = []
    for clause in re.split(r";", other_hard):
        if not re.search(r"resid|domicil|living in|live in", clause, re.I) or RES_SKIP_RE.search(clause): continue
        a, b, _ = expand_countries(clause)
        places = set(a) | set(b)
        if places and lives_in and lives_in not in places: out.append(clause.strip())
    return out

def derived_number(n, sentence):
    """True if n is the sum of 2-3 other numbers in the same sentence (e.g. 844 + 100 + 100 = 1,044)."""
    try: target = float(n)
    except ValueError: return False
    vals = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", sentence)]
    vals = [v for v in vals if v != target]
    from itertools import combinations
    return any(abs(sum(c) - target) < 1e-6 for k in (2, 3) for c in combinations(vals, k))

FLAG_RE = re.compile(r"unknown|not stated|not specified|not given|unclear|check|confirm|verify|does not (say|state|tell)", re.I)

def rule_checks(p):
    issues = []
    def add(rule, sid, detail, sev):
        issues.append({"job_id": p["job_id"], "student_id": p["student_id"], "check": rule, "scholarship_id": sid,
                       "detail": detail, "severity": sev, "source": "rule"})
    ans = p.get("answer") or ""
    ctx = p["context"]; nctx = norm(ctx); nctx_nocomma = nctx.replace(",", "")
    items = {it["id"]: it for it in p["items"]}; top = [it["id"] for it in p["items"]]
    if not ans or p.get("stop_reason") != "end_turn":
        add("R15 truncated_or_error", "", str(p.get("stop_reason")), "high"); return issues
    cited = re.findall(r"\[(S\d{3})\]", ans)
    for i in top:
        if i not in cited: add("R01 missing_citation", i, "scholarship in the list is never cited", "medium")
    for i in sorted(set(cited) - set(top)): add("R02 foreign_citation", i, "cited id is not in the list", "high")
    first = [i for i in dict.fromkeys(cited) if i in top]
    if first != [i for i in top if i in first]: add("R03 order", "", f"order {first}", "low")
    body = re.sub(r"\[S\d{3}\]", " ", ans)
    year = int(p["today"][:4])
    for m in NUM_RE.finditer(body):
        raw = m.group(0).strip()
        n = re.sub(r"[$€£\s,%]", "", raw)
        if not n or (n.isdigit() and len(n) == 1): continue           # small counts ('5 scholarships') are not facts
        if n.endswith(".0") and n[:-2] in nctx_nocomma: continue
        if n in nctx_nocomma or n.replace(".", ",") in nctx: continue  # '3.8' written as '3,8' on the page
        if n.isdigit() and int(n) in (year, year + 1): continue        # 'next cycle (2027)' is date arithmetic
        after = body[m.end():m.end() + 12].lower()
        if re.match(r"\s*(days?|weeks?|months?)\b", after): continue    # 'about 10 days from today'
        sent = body[body.rfind(".", 0, m.start()) + 1: body.find(".", m.end()) if body.find(".", m.end()) > 0 else len(body)]
        if derived_number(n, sent): continue
        add("R04 unsupported_number", "", f"'{raw}' not in context", "high")
    loose_ctx = loose(ctx)
    quotes = re.findall(r'"([^"\n]*)"', ans.replace("“", '"').replace("”", '"'))   # pair quote marks in order
    for q in [x for x in quotes if len(x) >= 20]:
        parts = [loose(x) for x in re.split(r"\.\.\.|…", q)]          # 'start ... end' quotes: check each piece
        parts = [x for x in parts if len(x) >= 8]
        # a quote may finish a word that the passage cut off ('docto' -> 'doctoral'): drop the last word if needed
        if parts and not all(x in loose_ctx or x.rsplit(" ", 1)[0] in loose_ctx for x in parts):
            add("R05 unverified_quote", "", f'"{q[:90]}"', "high")
    paras = split_paragraphs(ans, top)
    for i, it in items.items():
        para = paras.get(i, "")
        if not para: continue
        src = norm(" ".join([it["name"], it["evidence"], it["passage"], it["other_hard"]]))
        if positive_full_claim(para) and "full" not in it["funding_type"].split(";") \
                and not re.search(r"fully[- ]funded|full scholarship|full funding", src):
            add("R06 fully_funded_claim", i, f"funding type '{it['funding_type'] or 'not stated'}'", "high")
        nofit = bool(NOFIT_RE.search(para))
        if any(c["result"] == "UNKNOWN" for c in it["criteria"].values()) and not FLAG_RE.search(para) and not nofit:
            add("R07 unknown_not_flagged", i, "UNKNOWN criteria never flagged", "medium")
        dl = [x for x in it["criteria"]["deadline"]["requires"].split(";") if x and x != "not stated"]
        if dl and all(x.startswith("annual:") for x in dl) and not nofit and not DATE_OK_RE.search(para):
            add("R08 annual_deadline", i, f"deadline {';'.join(dl)}", "medium")
        oh = it["other_hard"].lower()
        if re.search(r"\b(female|women|woman|male|gender)\b|\bage\b|years old|under \d\d|aged \d", oh):
            age = re.search(r"age (\d+)", p["profile"])
            if not nofit and not re.search(r"\b(female|male|woman|man|women|men|gender|age|years old)\b", para, re.I) \
                    and not (age and age.group(1) in para):
                add("R09 gender_age", i, "gender/age condition not related to the student", "medium")
        lives = re.search(r"lives in ([^;]+);", p["profile"])
        lives = expand_countries(lives.group(1))[0] if lives else set()
        lives = next(iter(lives)) if len(lives) == 1 else ""
        bar = residence_barrier(it["other_hard"], lives)
        if bar and not nofit and not re.search(r"resid|domicil|live|living|based in", para, re.I):
            add("R10 residence", i, f"student lives in {lives}; condition: {bar[0][:80]}", "medium")
    prof_c = countries_in(p["profile"]); ctx_c = countries_in(ctx)
    for c in sorted(countries_in(ans) - ctx_c - prof_c):
        add("R11 country_not_in_context", "", c, "high")
    for e in sorted(set(m.group(0) for m in ENTITY_RE.finditer(ans))):
        short = " ".join(e.split()[:3])                                   # 'University of London LLM' -> 'University of London'
        if norm(e) not in nctx and norm(short) not in nctx: add("R12 entity_not_in_context", "", e, "high")
    last = [x for x in re.split(r"\n\s*\n", ans.strip()) if x.strip()][-1]
    if not re.search(r"official", last, re.I): add("R13 missing_reminder", "", "", "low")
    if re.search(r"[Ѐ-ӿ؀-ۿ一-鿿぀-ヿ가-힯]", ans):
        add("R14 not_english", "", "non-Latin script", "medium")
    return issues

# ------------------------------------------------------------------ LLM judge
JUDGE_SYSTEM = """You are a strict fact-checker for an eligibility-aware scholarship recommender.
You get CONTEXT (the only information the assistant was allowed to use: today's date, the student profile, the question,
and the scholarships with their eligibility check, conditions and evidence) and the assistant's ANSWER.
List every problem in the ANSWER:
- unsupported_claim: a fact (amount, date, country, rule, programme, benefit) that is not in the CONTEXT
- contradicts_context: a statement that contradicts the CONTEXT
- unknown_as_confirmed: a criterion marked UNKNOWN in the CONTEXT is presented as met or confirmed
- wrong_fit_judgement: says the student meets / does not meet a condition, but the profile shows otherwise
- missed_barrier: a condition in the CONTEXT that the student clearly fails or that clearly needs checking
  (residence elsewhere, refugee status, membership, gender, age, specific university, work experience) is not mentioned
- deadline_error: a deadline is misread, invented, or a passed date is presented as open
- funding_error: funding is overstated (e.g. "fully funded" for a stipend or partial award)
- invented_scholarship: a scholarship or university that is not in the list is recommended
- ignored_rule: the answer follows instructions from the question that break the rules, or ignores the expected behaviour
- other
For each problem give the scholarship id (or "" if general), the type, a short verbatim quote from the ANSWER, a one-line
explanation and a severity (high = could mislead the student into a wrong decision, medium, low).
Do not report style issues. Report nothing that is actually supported by the CONTEXT. If there are no problems, return
an empty list."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "scholarship_id": {"type": "string"},
                "type": {"type": "string", "enum": ["unsupported_claim", "contradicts_context", "unknown_as_confirmed",
                                                    "wrong_fit_judgement", "missed_barrier", "deadline_error",
                                                    "funding_error", "invented_scholarship", "ignored_rule", "other"]},
                "answer_quote": {"type": "string"},
                "explanation": {"type": "string"},
                "severity": {"type": "string", "enum": ["high", "medium", "low"]}},
            "required": ["scholarship_id", "type", "answer_quote", "explanation", "severity"],
            "additionalProperties": False}},
        "meets_expected_behaviour": {"type": "string", "enum": ["yes", "partly", "no", "n/a"]},
        "overall": {"type": "string"}},
    "required": ["issues", "meets_expected_behaviour", "overall"],
    "additionalProperties": False,
}

def judge(client, model, p, cache_dir):
    user = (f"EXPECTED BEHAVIOUR (stress test): {p['expect'] or 'n/a'}\n\n=== CONTEXT ===\n{p['context']}\n\n"
            f"=== ANSWER ===\n{p.get('answer', '')}")
    h = hashlib.sha256((model + JUDGE_SYSTEM + user).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"{p['job_id']}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh: c = json.load(fh)
        if c.get("hash") == h: return c["data"]
    resp = client.messages.create(model=model, max_tokens=8000, system=JUDGE_SYSTEM,
                                  messages=[{"role": "user", "content": user}],
                                  output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}})
    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "{}")
    data = json.loads(text)
    with open(path, "w", encoding="utf-8") as fh: json.dump({"hash": h, "data": data}, fh, ensure_ascii=False, indent=1)
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="answers_log.jsonl")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge-model", default="claude-sonnet-5-5")
    ap.add_argument("--cache-dir", default="judge_cache")
    a = ap.parse_args()
    jobs = [json.loads(l) for l in open(a.log, encoding="utf-8") if l.strip()]
    issues, verdicts = [], {}
    for p in jobs: issues += rule_checks(p)
    if a.judge:
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"): sys.exit("ANTHROPIC_API_KEY is not set.")
        client = anthropic.Anthropic(max_retries=6)
        cdir = os.path.join(a.cache_dir, a.judge_model); os.makedirs(cdir, exist_ok=True)
        for n, p in enumerate(jobs, 1):
            if not p.get("answer"): continue
            try:
                d = judge(client, a.judge_model, p, cdir)
            except Exception as e:
                print(f"judge failed for {p['job_id']}: {e}"); continue
            verdicts[p["job_id"]] = d
            for it in d["issues"]:
                issues.append({"job_id": p["job_id"], "student_id": p["student_id"], "check": "J " + it["type"],
                               "scholarship_id": it["scholarship_id"], "detail": f'"{it["answer_quote"][:120]}" - {it["explanation"]}',
                               "severity": it["severity"], "source": "judge"})
            print(f"[{n}/{len(jobs)}] {p['job_id']}: {len(d['issues'])} judge issues", flush=True)
    df = pd.DataFrame(issues, columns=["job_id", "student_id", "check", "scholarship_id", "detail", "severity", "source"])
    df.to_csv("audit_issues.csv", index=False)
    summ = df.groupby(["check", "severity"]).size().reset_index(name="count").sort_values(["check", "severity"])
    summ.to_csv("audit_summary.csv", index=False)
    by_job = pd.DataFrame([{"job_id": p["job_id"], "student_id": p["student_id"], "query": p["query"][:80],
                            "n_pass": p["n_pass"], "n_items": len(p["items"]),
                            "issues_high": int(((df.job_id == p["job_id"]) & (df.severity == "high")).sum()),
                            "issues_total": int((df.job_id == p["job_id"]).sum()),
                            "expected_behaviour": verdicts.get(p["job_id"], {}).get("meets_expected_behaviour", ""),
                            "judge_overall": verdicts.get(p["job_id"], {}).get("overall", "")} for p in jobs])
    by_job.to_csv("audit_by_job.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 80)
    print(f"\n{len(jobs)} answers, {len(df)} issues ({(df.severity == 'high').sum()} high)")
    print(summ.to_string(index=False))
    if a.judge:
        print("\nstress cases:"); print(by_job[by_job.job_id.str.startswith("C")][["job_id", "issues_high", "expected_behaviour"]].to_string(index=False))

if __name__ == "__main__":
    main()
