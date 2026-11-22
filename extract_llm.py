"""LLM criteria extractor (Claude Haiku 5.5 by default) with JSON-schema structured output.

Setup (once):   pip install anthropic pandas
API key:        set the environment variable ANTHROPIC_API_KEY (never put the key in the code)
                Windows PowerShell:  $env:ANTHROPIC_API_KEY="sk-ant-..."
                Windows cmd:         set ANTHROPIC_API_KEY=sk-ant-...
                PyCharm: Run > Edit Configurations > Environment variables
Usage:
  python extract_llm.py --dry-run                 # show the prompt for one scholarship, no API call
  python extract_llm.py --limit 5                 # try 5 scholarships first
  python extract_llm.py                           # all scholarships -> extracted_haiku.csv
Responses are cached in llm_cache/<model>/<id>.json, so a re-run only calls the API for missing rows
(use --force to redo everything, e.g. after changing the prompt; a changed prompt is detected automatically).
"""
import argparse, hashlib, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from criteria_common import FIELDS, DEGREES, ENROLL, FUNDING, DEADLINE_PART_RE

PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}  # $/MTok

SYSTEM = """You extract scholarship eligibility criteria for an eligibility-aware scholarship search engine.
Today's date is {date}. You receive ONE scholarship: its name, its URL and the text of its web page.
Fill every field of the output schema.

CORE RULES
1. Use only what the page text states. Do not add outside knowledge about the scholarship, even if you know it.
   Exception: host_country may also be taken from the university name or the URL.
2. If something is not stated, return an empty string or an empty list. Never guess.
3. Preferences are not requirements: "priority to", "preference for", "encouraged to apply" go to notes.
4. If the page lists several awards, describe the award named in <name>. If it is a list page with different
   criteria per award, say so in notes and fill only criteria that apply to the whole page.

FIELDS
- degree_levels: level(s) of study the scholarship funds: bachelor, master, phd, postdoc, other.
  MBA, LLM, MSc, MA = master. High-school senior entering college = bachelor. Professional doctorate = phd.
  "graduate"/"postgraduate" study without any mention of doctoral study = master.
  other = 2-year college, associate degree, certificate, diploma, vocational, language course, exchange or
  research stay that is not degree study. A degree the applicant must ALREADY hold is not a funded level.
- nationality_rule: citizenship requirement, exactly one of:
  ALL                         text explicitly says open to all nationalities / both domestic and international
  INTERNATIONAL               only international (non-domestic) students of the host country
  ONLY:<country>;<country>    only citizens of the listed countries; qualifier in brackets allowed,
                              e.g. ONLY:USA (citizen or PR), ONLY:USA (citizen or resident)
  EXCLUDE:<country>;<country> everyone except citizens of the listed countries, e.g. EXCLUDE:China
  REGION:<group>              a named group whose countries are not listed, e.g. REGION:developing countries
  If the text lists the countries, use ONLY with the list. Use short English country names (USA, UK, South Korea).
  Place of residence alone ("must live in", "ordinarily resident in") is NOT nationality: put it in other_hard.
- min_gpa: minimum grade exactly as written: 3.0, 2:1, first class, 75%, 1.5, good, top third.
  A verbal grade requirement stated as a condition ("excellent academic record required") counts.
  A GPA needed only to KEEP or RENEW the award is not an entry requirement (put it in notes).
- gpa_scale: only if stated: 4.0, UK honours, percentage average, German (1.0 best), ATAR, WAM.
- enrollment_status: status required when applying: applicant (not yet admitted; also awards considered
  automatically with the admission application), admitted (must already hold an offer / be accepted),
  enrolled (currently studying). Add full-time or part-time only if the text restricts the study mode;
  if both modes are allowed, add neither.
- deadline: application deadline(s). A date on or after today -> YYYY-MM-DD. A past date or a date without a
  year -> annual:MM-DD. Month only -> YYYY-MM if in the future, else annual:MM. Open all year / rolling
  -> rolling. Ignore deadlines for other steps (visa, documents, acceptance of the offer).
- field_of_study: ANY if open to all subjects, otherwise the fields as short text.
- funding_type: full (tuition + living), partial (fixed amount or discount below full cost),
  tuition (full tuition waiver only), stipend (monthly living allowance), travel. Only if stated.
- other_hard: other hard conditions separated by "; ": residence, age, gender, language test, work experience,
  financial need, membership, specific university or course, years since last degree.
- evidence_sentence: ONE sentence copied VERBATIM from the page text that states the most important hard
  criterion (prefer nationality or degree level). Do not paraphrase. Empty if there are no criteria.
- host_country: country where the funded study takes place (short English name), empty if unknown.
- notes: doubts: several programmes on one page, outdated text, page does not match the name,
  "no criteria in text".

EXAMPLE (invented)
Text: "The Riverside Global Award supports international students starting a full-time master's degree at
Riverside University (Canada) in September 2027. Applicants must have a minimum GPA of 3.3 on a 4.0 scale.
Students from Canada are not eligible. Apply by 15 February 2027."
Output: degree_levels=["master"], nationality_rule="INTERNATIONAL", min_gpa="3.3", gpa_scale="4.0",
enrollment_status=["applicant","full-time"], deadline=["2027-02-15"], host_country="Canada",
evidence_sentence="Students from Canada are not eligible."
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "degree_levels": {"type": "array", "items": {"type": "string", "enum": DEGREES}},
        "nationality_rule": {"type": "string"},
        "min_gpa": {"type": "string"},
        "gpa_scale": {"type": "string"},
        "enrollment_status": {"type": "array", "items": {"type": "string", "enum": ENROLL}},
        "deadline": {"type": "array", "items": {"type": "string"},
                     "description": "each item: YYYY-MM-DD, YYYY-MM, annual:MM-DD, annual:MM or rolling"},
        "field_of_study": {"type": "string"},
        "funding_type": {"type": "array", "items": {"type": "string", "enum": FUNDING}},
        "other_hard": {"type": "string"},
        "evidence_sentence": {"type": "string"},
        "host_country": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["degree_levels", "nationality_rule", "min_gpa", "gpa_scale", "enrollment_status", "deadline",
                 "field_of_study", "funding_type", "other_hard", "evidence_sentence", "host_country", "notes"],
    "additionalProperties": False,
}

def user_message(r):
    return (f"<name>{r['name']}</name>\n<url>{r.get('url', '')}</url>\n"
            f"<page_text>\n{r['eligibility_text']}\n</page_text>")

def call_model(client, model, system, user, effort, max_tokens):
    output_config = {"format": {"type": "json_schema", "schema": SCHEMA}}
    if effort: output_config["effort"] = effort
    resp = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                  messages=[{"role": "user", "content": user}], output_config=output_config)
    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
    return {"data": json.loads(text) if text else None, "stop_reason": resp.stop_reason,
            "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}}

def postprocess(d, page_text, today):
    """Arrays -> ';' strings, format checks, past dates -> annual, verbatim check of the evidence sentence."""
    out = {f: "" for f in FIELDS}
    fixes = []
    for f in FIELDS:
        v = d.get(f, "")
        if isinstance(v, list): v = ";".join(dict.fromkeys(x.strip() for x in v if x and x.strip()))
        out[f] = (v or "").strip()
    parts = []
    for p in [x for x in out["deadline"].split(";") if x]:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p) and p < today: fixes.append(f"past date {p} -> annual"); p = "annual:" + p[5:]
        if re.fullmatch(r"\d{4}-\d{2}", p) and p < today[:7]: fixes.append(f"past month {p} -> annual"); p = "annual:" + p[5:]
        if DEADLINE_PART_RE.match(p): parts.append(p)
        else: fixes.append(f"dropped bad deadline '{p}'")
    out["deadline"] = ";".join(dict.fromkeys(parts))
    nr = out["nationality_rule"]
    if nr and not re.match(r"^(ALL|INTERNATIONAL|ONLY:.+|EXCLUDE:.+|REGION:.+)$", nr):
        fixes.append(f"bad nationality_rule '{nr}' -> blank"); out["nationality_rule"] = ""
    enr = set(out["enrollment_status"].split(";"))
    if {"full-time", "part-time"} <= enr:
        out["enrollment_status"] = ";".join(x for x in out["enrollment_status"].split(";") if x not in ("full-time", "part-time"))
    ev = re.sub(r"\s+", " ", out["evidence_sentence"])
    out["evidence_verbatim"] = int(bool(ev) and ev in re.sub(r"\s+", " ", page_text))
    out["postprocess_fixes"] = "; ".join(fixes)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="gold_annotations_ok193.csv")
    ap.add_argument("--out", default="extracted_haiku.csv")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default="", help="low / medium / high (empty = model default)")
    ap.add_argument("--date", default="2026-11-21")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--cache-dir", default="llm_cache")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    corpus = pd.read_csv(a.corpus, dtype=str, keep_default_na=False)
    if a.ids: corpus = corpus[corpus["id"].isin(a.ids)]
    if a.limit: corpus = corpus.head(a.limit)
    system = SYSTEM.format(date=a.date)
    if a.dry_run:
        r = corpus.iloc[0]
        print(system); print("=" * 80); print(user_message(r)[:3000])
        chars = sum(len(system) + len(user_message(x)) for _, x in corpus.iterrows())
        print(f"\n{len(corpus)} scholarships, ~{chars / 3.5 / 1e6:.2f}M input tokens (rough estimate)")
        return

    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (see the instructions at the top of this file).")
    client = anthropic.Anthropic(max_retries=6)
    cdir = os.path.join(a.cache_dir, a.model); os.makedirs(cdir, exist_ok=True)

    def work(r):
        user = user_message(r)
        h = hashlib.sha256((a.model + a.effort + system + user).encode()).hexdigest()[:16]
        path = os.path.join(cdir, f"{r['id']}.json")
        if not a.force and os.path.exists(path):
            with open(path, encoding="utf-8") as fh: c = json.load(fh)
            if c.get("hash") == h and c.get("data") is not None: return r["id"], c, True
        res = call_model(client, a.model, system, user, a.effort, a.max_tokens)
        if res["data"] is None or res["stop_reason"] == "max_tokens":
            res = call_model(client, a.model, system, user, a.effort, a.max_tokens * 2)
        res["hash"] = h
        with open(path, "w", encoding="utf-8") as fh: json.dump(res, fh, ensure_ascii=False, indent=1)
        return r["id"], res, False

    results, failed = {}, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, r): r["id"] for _, r in corpus.iterrows()}
        for n, fu in enumerate(as_completed(futs), 1):
            sid = futs[fu]
            try:
                sid, res, cached = fu.result()
                results[sid] = res
                print(f"[{n}/{len(futs)}] {sid} {'cached' if cached else 'ok'}", flush=True)
            except Exception as e:  # keep going; failed rows stay blank and are listed at the end
                failed.append(sid); print(f"[{n}/{len(futs)}] {sid} FAILED: {type(e).__name__}: {e}", flush=True)

    rows, tin, tout = [], 0, 0
    for _, r in corpus.iterrows():
        res = results.get(r["id"])
        if res and res.get("data") is not None:
            row = postprocess(res["data"], r["eligibility_text"], a.date)
            tin += res["usage"]["input_tokens"]; tout += res["usage"]["output_tokens"]
        else:
            row = {f: "" for f in FIELDS}; row.update({"notes": "extraction_failed", "evidence_verbatim": 0,
                                                       "postprocess_fixes": ""})
        rows.append({"id": r["id"], **row})
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    pin, pout = PRICES.get(a.model, (0, 0))
    print(f"\n{len(out)} rows -> {a.out} in {time.time() - t0:.0f}s; failed: {failed or 'none'}")
    print(f"tokens (all rows, incl. cached): input {tin:,}, output {tout:,}; "
          f"cost if all were fresh calls ~${tin / 1e6 * pin + tout / 1e6 * pout:.3f}")
    print(f"evidence copied verbatim: {out['evidence_verbatim'].sum()}/{(out['evidence_sentence'] != '').sum()}")

if __name__ == "__main__":
    main()
