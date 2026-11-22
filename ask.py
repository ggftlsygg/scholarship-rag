"""Ask the system one question - the single entry point for using (and demoing) the scholarship RAG.

profile + question -> Stage A eligibility filter -> Rank = alpha*Sim + (1-alpha)*EligScore -> top-N
-> Claude Haiku 5.5 writes an answer from the retrieved evidence only, citing [S###].

Usage:
  python ask.py --student P01                                   # stored student, stored question
  python ask.py --student P01 --query "fully funded master's in AI in Europe"
  python ask.py --nationality Kazakhstan --degree master --gpa 3.7 --field "data science" \
                --query "fully funded master's in data science abroad"
  python ask.py                                                 # interactive: asks for the profile in the console
  ... --no-llm                                                  # ranked list only: free, no API key needed
  ... --show-context                                            # also print the exact text sent to the model
Output: printed in the console and saved to ask_result.md

Uses: stage_a.py, criteria_common.py, retrieval.py, generate_answers.py (prompt + context builder),
      gold_annotations_ok193.csv (scholarship texts), extracted_haiku.csv (criteria), chosen_alpha.json
"""
import argparse, datetime, difflib, json, os, re, sys
from collections import Counter
import numpy as np
import pandas as pd
import stage_a
from criteria_common import canon_country, COUNTRIES, ALIASES
from retrieval import DenseIndex
from generate_answers import SYSTEM, build_items, build_context, profile_line

CHECKS = [("m_degree", "degree"), ("m_nationality", "nationality"), ("m_gpa", "gpa"),
          ("m_enrollment", "enrollment"), ("m_deadline", "deadline")]
SHORT = {"MATCH": "ok", "UNKNOWN": "?", "NO MATCH": "x"}

# ------------------------------------------------------------------ typo handling
COUNTRY_NAMES = {c.lower(): c for c in COUNTRIES}
COUNTRY_NAMES.update(ALIASES)                                   # alias / demonym -> canonical name

def fix_country(text):
    """Exact name, alias or demonym first ('kazakh'), then the closest spelling ('Kozakhstan').
    Returns (canonical name or None, was_corrected)."""
    c = canon_country(text or "")
    if c: return c, False
    hit = difflib.get_close_matches((text or "").strip().lower(), list(COUNTRY_NAMES), n=1, cutoff=0.8)
    return (COUNTRY_NAMES[hit[0]], True) if hit else (None, False)

def fix_option(text, options):
    """'masters' -> 'master', 'aplicant' -> 'applicant', 'fulltime' -> 'full-time'; None if nothing is close."""
    t = (text or "").strip().lower()
    if t in options: return t
    hit = difflib.get_close_matches(t, options, n=1, cutoff=0.6)
    return hit[0] if hit else None

def corpus_vocab(corpus):
    """Words that occur at least twice in the scholarship texts: the targets for spelling fixes."""
    cnt = Counter(w for t in corpus["name"] + " " + corpus["eligibility_text"] for w in re.findall(r"[a-z]+", t.lower()))
    return {w for w, n in cnt.items() if n >= 2 and len(w) >= 3}

def fix_words(text, vocab, notes):
    """Fix misspelled English words (5+ letters, not in the corpus) to the closest corpus word: 'ecnnomics' -> 'economics'."""
    def one(m):
        w = m.group(0); lw = w.lower()
        if len(lw) < 5 or lw in vocab: return w
        hit = difflib.get_close_matches(lw, vocab, n=1, cutoff=0.85)
        if not hit: return w
        notes.append(f"{w} -> {hit[0]}")
        return hit[0]
    return re.sub(r"[A-Za-z]+", one, text or "")

def ask_profile():
    """Interactive profile entry (Enter keeps the default in brackets)."""
    def q(text, default=""):
        v = input(f"{text}{f' [{default}]' if default else ''}: ").strip()
        return v or default
    def country(text, default):
        while True:
            raw = q(text, default); c, corrected = fix_country(raw)
            if c and (not corrected or q(f"  Did you mean {c}? (y/n)", "y").lower().startswith("y")): return c
            near = difflib.get_close_matches(raw.lower(), list(COUNTRY_NAMES), n=3, cutoff=0.6)
            print(f"  Unknown country '{raw}'." + (f" Close: {', '.join(COUNTRY_NAMES[n] for n in near)}" if near else ""))
    def option(text, options, default):
        while True:
            v = fix_option(q(text, default), options)
            if v: return v
            print(f"  Please type one of: {' / '.join(options)}")
    p = {"nationality": country("Nationality (country)", "Kazakhstan")}
    p["residence"] = country("Country you live in", p["nationality"])
    p["target_degree"] = option("Target degree (bachelor / master / phd / postdoc)", ["bachelor", "master", "phd", "postdoc"], "master")
    p["gpa_4"] = q("GPA on a 4.0 scale (empty if none)", "")
    p["enrollment_status"] = option("Status (applicant / admitted / enrolled)", ["applicant", "admitted", "enrolled"], "applicant")
    p["study_mode"] = option("Study mode (full-time / part-time)", ["full-time", "part-time"], "full-time")
    p["age"] = q("Age", ""); p["gender"] = q("Gender (F / M)", ""); p["field"] = q("Field of study", "")
    query = q("Your question", f"Scholarships for a {p['target_degree']} in {p['field'] or 'any field'}")
    return p, query

def make_profile(a, prof, vocab):
    if a.student:
        if a.student not in prof.index: sys.exit(f"unknown student {a.student}; ids: {', '.join(prof.index[:5])}, ...")
        s = prof.loc[a.student].copy()
        if not a.query: return s, s["query"]
        notes = []; query = fix_words(a.query, vocab, notes)
        if notes: print("Spelling corrected: " + "; ".join(dict.fromkeys(notes)))
        return s, query
    if not a.nationality:
        p, query = ask_profile()
    else:
        p = {"nationality": a.nationality, "residence": a.residence or a.nationality, "target_degree": a.degree,
             "gpa_4": a.gpa, "enrollment_status": a.status, "study_mode": a.mode, "age": a.age, "gender": a.gender,
             "field": a.field}
        query = a.query or f"Scholarships for a {a.degree} in {a.field or 'any field'}"
    notes = []
    for k in ("nationality", "residence"):                       # 'kazakh' -> 'Kazakhstan', 'Kozakhstan' -> 'Kazakhstan'
        c, corrected = fix_country(p[k])
        if not c:
            near = difflib.get_close_matches(p[k].lower(), list(COUNTRY_NAMES), n=3, cutoff=0.6)
            sys.exit(f"Unknown country '{p[k]}'." + (f" Did you mean: {', '.join(COUNTRY_NAMES[n] for n in near)}?" if near else ""))
        if corrected: notes.append(f"{p[k]} -> {c}")
        p[k] = c
    p["field"] = fix_words(p["field"], vocab, notes)
    query = fix_words(query, vocab, notes)
    if notes: print("Spelling corrected: " + "; ".join(dict.fromkeys(notes)))
    if p["target_degree"] not in ("bachelor", "master", "phd", "postdoc"): sys.exit("degree: bachelor/master/phd/postdoc")
    if p["target_degree"] == "postdoc": p["enrollment_status"] = ""
    p["student_id"] = "custom"
    return pd.Series({k: str(v or "") for k, v in p.items()}), query

def main():
    ap = argparse.ArgumentParser(description="Ask the eligibility-aware scholarship RAG one question.")
    ap.add_argument("--student", help="stored student id, e.g. P01")
    ap.add_argument("--query", help="the question (default: the stored one, or a generic one)")
    ap.add_argument("--nationality"); ap.add_argument("--residence")
    ap.add_argument("--degree", default="master", choices=["bachelor", "master", "phd", "postdoc"])
    ap.add_argument("--gpa", default="", help="GPA on a 4.0 scale")
    ap.add_argument("--status", default="applicant", choices=["applicant", "admitted", "enrolled"])
    ap.add_argument("--mode", default="full-time", choices=["full-time", "part-time"])
    ap.add_argument("--age", default=""); ap.add_argument("--gender", default="", choices=["", "F", "M"])
    ap.add_argument("--field", default="")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=None, help="default: chosen on dev (chosen_alpha.json)")
    ap.add_argument("--date", default=datetime.date.today().isoformat(), help="experiments used 2026-11-21")
    ap.add_argument("--embedder", default="bge", choices=["bge", "lsa"])
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--corpus", default="gold_annotations_ok193.csv")
    ap.add_argument("--criteria", default="extracted_haiku.csv")
    ap.add_argument("--profiles", default="student_profiles.csv")
    ap.add_argument("--extra-corpus", default="corpus_extra.csv", help="scholarships added with add_scholarship.py")
    ap.add_argument("--extra-criteria", default="extracted_extra.csv")
    ap.add_argument("--no-extra", action="store_true", help="only the 193 verified scholarships (as in the report)")
    ap.add_argument("--no-llm", action="store_true", help="skip the written answer (free)")
    ap.add_argument("--show-context", action="store_true")
    ap.add_argument("--out", default="ask_result.md")
    a = ap.parse_args()

    stage_a.TODAY = a.date                                        # deadlines are checked against this date
    corpus = pd.read_csv(a.corpus, dtype=str, keep_default_na=False)
    crit = pd.read_csv(a.criteria, dtype=str, keep_default_na=False).set_index("id", drop=False)
    prof = pd.read_csv(a.profiles, dtype=str, keep_default_na=False).set_index("student_id", drop=False)
    n_gold, extra_ids = len(corpus), set()
    if not a.no_extra and os.path.exists(a.extra_corpus) and os.path.exists(a.extra_criteria):
        ec = pd.read_csv(a.extra_corpus, dtype=str, keep_default_na=False)
        ex = pd.read_csv(a.extra_criteria, dtype=str, keep_default_na=False).set_index("id", drop=False)
        corpus = pd.concat([corpus, ec], ignore_index=True).fillna("")
        crit = pd.concat([crit, ex]).fillna("")
        extra_ids = set(ec["id"])
    s, query = make_profile(a, prof, corpus_vocab(corpus))
    alpha = a.alpha
    if alpha is None:
        try: alpha = json.load(open("chosen_alpha.json"))["Ours (haiku)"]
        except (OSError, KeyError): alpha = 0.85

    # 1) eligibility filter (Stage A) on every scholarship
    critc = crit.reindex(corpus["id"]).reset_index(drop=True)
    match = [stage_a.match_pair(critc.iloc[d], s) for d in range(len(corpus))]
    valid = [d for d in range(len(corpus)) if match[d]["passes_filter"]]
    why = {name: sum(1 for m in match if m[col] == 0) for col, name in CHECKS}
    # 2) semantic similarity of the question to every scholarship (best chunk), scaled to 0..1
    dense = DenseIndex(corpus, a.embedder)
    D, B = dense.scores([query])
    sim = (D[0] - D[0].min()) / max(D[0].max() - D[0].min(), 1e-9)
    # 3) ranking among the scholarships that passed
    score = {d: alpha * sim[d] + (1 - alpha) * match[d]["elig_score"] for d in valid}
    top = sorted(valid, key=lambda d: (-score[d], d))[:a.top]
    items = build_items(top, crit, corpus, dense.chunks, B[0], match)
    ctx = build_context(s, query, items, len(valid), len(corpus), a.date)

    out = [f"# {query}", "", f"Profile: {profile_line(s)}", f"Date: {a.date} | alpha = {alpha:g} | top {a.top}", "",
           f"Eligibility filter: {len(valid)} of {len(corpus)} scholarships pass"
           + (f" ({n_gold} verified + {len(extra_ids)} added)." if extra_ids else "."),
           "Excluded because of (one scholarship can fail several): "
           + ", ".join(f"{k} {v}" for k, v in why.items() if v), "",
           "| rank | id | scholarship | score | EligScore | degree | nationality | gpa | enrollment | deadline |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for it in items:
        res = [SHORT[it["criteria"][f]["result"]] for f in
               ("degree_levels", "nationality_rule", "min_gpa", "enrollment_status", "deadline")]
        d = corpus.index[corpus["id"] == it["id"]][0]
        mark = "*" if it["id"] in extra_ids else ""
        out.append(f"| {it['rank']} | {it['id']}{mark} | {it['name'][:55]} | {score[d]:.3f} | {it['elig_score']:.2f} | "
                   + " | ".join(res) + " |")
    out.append("\nok = criterion met, ? = not stated on the page (check it), x = not met (never shown: filtered out)"
               + ("\n* = added with add_scholarship.py: criteria extracted automatically, not verified by hand"
                  if any(it["id"] in extra_ids for it in items) else ""))
    print("\n".join(out))
    if a.show_context: print("\n" + "=" * 80 + "\n" + ctx + "\n" + "=" * 80)

    # 4) grounded answer
    answer = ""
    if a.no_llm:
        print("\n(--no-llm: no written answer)")
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nANTHROPIC_API_KEY is not set: showing the ranked list only. Set the key to get the written answer.")
    else:
        import anthropic
        client = anthropic.Anthropic(max_retries=6)
        try:
            resp = client.messages.create(model=a.model, max_tokens=6000, system=SYSTEM,
                                          messages=[{"role": "user", "content": ctx}])
            answer = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
            u = resp.usage
            print(f"\n{answer}\n\n(tokens in {u.input_tokens:,} / out {u.output_tokens:,}, "
                  f"~${u.input_tokens / 1e6 * 0.10 + u.output_tokens / 1e6 * 0.50:.4f})")
        except Exception as e:
            print(f"\nAPI error: {type(e).__name__}: {e}")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + ("\n\n## Answer\n\n" + answer if answer else "") + "\n")
    print(f"\n-> saved to {a.out}")

if __name__ == "__main__":
    main()
