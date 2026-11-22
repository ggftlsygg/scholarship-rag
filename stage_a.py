"""Stage A: generic hard-eligibility filter.

Works on ANY criteria file with the gold column format (gold annotations or extracted criteria).
Unlike build_ground_truth.py it has no per-scholarship hand-coded rules: everything is decided from the
criteria strings, so it is what the deployed system can actually do.

Match per criterion: 1 = satisfies, 0 = violates, 0.5 = not stated / cannot be decided.
EligScore = mean of the five Match values. A scholarship is EXCLUDED when any Match = 0.

Usage:
  python stage_a.py --criteria extracted_haiku.csv --profiles ../gt/student_profiles.csv --out stage_a_haiku.csv
  (optional) --host-file gold_host_country.csv   fills host_country where the criteria file has none
"""
import argparse, re
import pandas as pd
from criteria_common import (expand_countries, canon_country, is_developing, gpa_to_4, split_multi,
                             DEADLINE_PART_RE)

TODAY = "2026-11-21"
ANNUAL_PASSED = 0.5
EMBARGOED = {"Iran", "Cuba", "North Korea", "Syria"}

def m_degree(levels, target):
    s = set(split_multi(levels))
    if not s: return 0.5
    return 1 if target in s else 0

GROUP_WORD = re.compile(r"\b(?:EU|EEA|europe\w*|asia\w*|africa\w*|arab\w*|latin america\w*|asean|middle east)\b", re.I)
NEG_BEFORE = re.compile(r"(?:non-|outside (?:of )?(?:the )?|except (?:for )?|excluding |other than |not (?:from )?)\s*$",
                        re.I)

def region_negation_flags(body):
    """For each group word in a REGION rule: is it negated? 'EU/EEA' after 'outside' inherits the negation."""
    flags, prev_end, prev_neg = [], 0, False
    for m in GROUP_WORD.finditer(body):
        gap = body[prev_end:m.start()]
        if NEG_BEFORE.search(gap): neg = True
        elif prev_neg and re.fullmatch(r"[\s/,;&-]*(?:and|or)?[\s/,;&-]*", gap, re.I): neg = True
        else: neg = False
        flags.append(neg); prev_end, prev_neg = m.end(), neg
    return flags

def m_nationality(rule, nat, res, host):
    r = (rule or "").strip()
    if not r: return 0.5
    up = r.upper()
    if up.startswith("ALL"): return 1
    if up.startswith("INTERNATIONAL"):
        h = canon_country(host or "")
        if not h: return 0.5
        return 0 if nat == h else 1
    kind, _, body = r.partition(":")
    kind = kind.strip().upper()
    qual = " ".join(re.findall(r"\((.*?)\)", body)).lower()
    explicit, grouped, dev = expand_countries(body, strip_parens=(kind != "REGION"))
    members = explicit | grouped
    if kind == "ONLY":
        if nat in members:
            if re.search(r"resident in|domiciled in|living in", qual) and res not in members: return 0
            return 1
        if res in members and re.search(r"or resident|residents or|or domiciled|domiciled|or living", qual + " " + body.lower()):
            return 0.5 if "permanent" in qual else 1
        if not members and dev: return 0.5 if is_developing(nat) else 0
        if not members: return 0.5
        return 0
    if kind == "EXCLUDE":
        if nat in members: return 0
        if "embargo" in body.lower(): return 0.5 if nat in EMBARGOED else 1
        return 1
    if kind == "REGION":
        # negated groups: 'non-European Union states', 'outside EU/EEA and Switzerland'
        flags = region_negation_flags(body)
        if flags and all(flags): return 0 if nat in (grouped | explicit) else 1   # only negated mentions -> exclusion
        if flags and any(flags): return 1 if nat in (grouped | explicit) else 0.5  # 'non-Europeans ... and Europeans'
        if nat in explicit: return 1
        cond = bool(re.search(r"non-\w+ (are |also )?eligible|non-\w+ .*if", qual))
        if grouped:
            if nat in grouped: return 1
            return 0.5 if cond or (dev and is_developing(nat)) else 0
        if dev: return 0.5 if is_developing(nat) else 0
        return 0.5
    return 0.5

def m_gpa(value, scale, gpa, target):
    if not (value or "").strip() or not (gpa or "").strip(): return 0.5
    conv = gpa_to_4(value, scale, target)
    if conv is None: return 0.5
    t, band = conv; g = float(gpa)
    if not re.search(r"\d", value):        # verbal threshold ('excellent'): too vague to exclude anyone
        return 1 if g >= t + band else 0.5
    if band == 0: return 1 if g >= t - 1e-9 else 0
    if g >= t + band: return 1
    if g < t - band: return 0
    return 0.5

def m_enrollment(val, status, mode):
    parts = set(split_multi(val))
    S = parts & {"applicant", "admitted", "enrolled"}
    M = parts & {"full-time", "part-time"}
    if len(M) == 2: M = set()
    if not S and not M: return 0.5
    out = []
    if S:
        if not status: out.append(0.5)
        elif status in S: out.append(1)
        elif status == "admitted" and "applicant" in S: out.append(1)
        elif status == "applicant" and "admitted" in S: out.append(0.5)
        elif status == "admitted" and "enrolled" in S: out.append(0.5)
        else: out.append(0)
    if M: out.append(1 if mode in M else 0)
    return min(out)

def m_deadline(val, today=None, annual_passed=None):
    today = today or TODAY
    annual_passed = ANNUAL_PASSED if annual_passed is None else annual_passed
    parts = [p for p in split_multi(val) if DEADLINE_PART_RE.match(p)]
    if not parts: return 0.5
    best = 0
    for p in parts:
        if p == "rolling": v = 1
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", p): v = 1 if p >= today else 0
        elif re.fullmatch(r"\d{4}-\d{2}", p): v = 1 if p >= today[:7] else 0
        else:
            md = p[7:]
            v = 1 if md >= (today[5:7] if len(md) == 2 else today[5:]) else annual_passed
        best = max(best, v)
    return best

def match_pair(c, s):
    m = {"m_degree": m_degree(c.get("degree_levels", ""), s["target_degree"]),
         "m_nationality": m_nationality(c.get("nationality_rule", ""), s["nationality"], s["residence"],
                                        c.get("host_country", "")),
         "m_gpa": m_gpa(c.get("min_gpa", ""), c.get("gpa_scale", ""), s["gpa_4"], s["target_degree"]),
         "m_enrollment": m_enrollment(c.get("enrollment_status", ""), s["enrollment_status"], s["study_mode"]),
         "m_deadline": m_deadline(c.get("deadline", ""))}
    vals = list(m.values())
    m["elig_score"] = round(sum(vals) / len(vals), 3)
    m["label"] = "ineligible" if 0 in vals else ("eligible" if all(v == 1 for v in vals) else "uncertain")
    m["passes_filter"] = int(0 not in vals)
    return m

def run(criteria, profiles, host_file=None):
    crit = criteria.copy()
    if "host_country" not in crit.columns: crit["host_country"] = ""
    if host_file is not None:
        hmap = dict(zip(host_file["id"], host_file["host_country"]))
        crit["host_country"] = [h if h else hmap.get(i, "") for i, h in zip(crit["id"], crit["host_country"])]
    rows = []
    for _, s in profiles.iterrows():
        for _, c in crit.iterrows():
            rows.append({"student_id": s["student_id"], "scholarship_id": c["id"], **match_pair(c, s)})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--criteria", required=True)
    ap.add_argument("--profiles", default="../gt/student_profiles.csv")
    ap.add_argument("--host-file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", default=TODAY)
    ap.add_argument("--annual-passed", type=float, default=ANNUAL_PASSED)
    a = ap.parse_args()
    TODAY, ANNUAL_PASSED = a.date, a.annual_passed
    crit = pd.read_csv(a.criteria, dtype=str, keep_default_na=False)
    prof = pd.read_csv(a.profiles, dtype=str, keep_default_na=False)
    host = pd.read_csv(a.host_file, dtype=str, keep_default_na=False) if a.host_file else None
    out = run(crit, prof, host)
    out.to_csv(a.out, index=False)
    print(f"{len(out)} pairs -> {a.out}; label: {out.label.value_counts().to_dict()}")
