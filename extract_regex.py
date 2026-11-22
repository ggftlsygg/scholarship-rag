"""Rule-based (regex) baseline extractor for the five core criteria + host country.

Usage: python extract_regex.py --corpus gold_annotations_ok193.csv --out extracted_regex.csv [--date 2026-11-21]
Only the columns id, name, url, eligibility_text of the corpus file are used.
"""
import argparse, re
import pandas as pd
from criteria_common import (FIELDS, countries_in, canon_country, MONTHS, MON_ABBR, normalize_deadline_part)

def norm(t):
    return re.sub(r"\s+", " ", t or "")

# ------------------------------------------------------------------ degree
DEG_PATTERNS = {
    "bachelor": r"bachelor|undergraduate|freshm[ae]n|first-year college|high[- ]school seniors?|college freshman|\bb\.?sc\b",
    "master": r"\bmasters?\b|master['’]s|\bmsc\b|\bm\.sc\b|\bmba\b|\bllm\b|\bll\.m\b|\bma degree|postgraduate taught|"
              r"taught postgraduate|(?<!under)graduate (students?|programs?|programmes?|study|studies|degree|school)",
    "phd": r"\bph\.?\s?d\b|doctoral|doctorate|\bdphil\b",
    "postdoc": r"post-?doc",
    "other": r"diploma programme|certificate programs?|vocational|career and technical|community college|"
             r"associate['’]?s? degree|intensive english|language course|summer (school|course)|trade school",
}
PRIOR = {"bachelor": r"(hold|holding|completed?|possess|have|obtained|earned|with)\s+(a |an |their )?(first |four-year )?"
                     r"(bachelor|undergraduate)",
         "master": r"(hold|holding|completed?|possess|have|obtained|earned|with)\s+(a |an )?master"}

def ex_degree(name, t):
    s = (name + " . " + t).lower()
    found = {d for d, p in DEG_PATTERNS.items() if re.search(p, s)}
    # a degree mentioned only as a prerequisite is not the funded level
    if "bachelor" in found and (found & {"master", "phd"}) and re.search(PRIOR["bachelor"], s):
        if len(re.findall(DEG_PATTERNS["bachelor"], s)) <= len(re.findall(PRIOR["bachelor"], s)) + 1:
            found.discard("bachelor")
    if "master" in found and "phd" in found and re.search(PRIOR["master"], s):
        if len(re.findall(r"\bmasters?\b|master['’]s", s)) <= len(re.findall(PRIOR["master"], s)) + 1:
            found.discard("master")
    return ";".join(d for d in ["bachelor", "master", "phd", "postdoc", "other"] if d in found)

# ------------------------------------------------------------------ nationality
REQ = r"must|only|eligible|open to|restricted|limited to|requirement|be an? |are required|applicants|candidates"
ALL_RE = re.compile(r"all nationalities|any nationality|regardless of (their )?(nationality|citizenship|country)|"
                    r"open to (students|applicants|candidates) (from|of) (all|any) (countries|country|nationalit)|"
                    r"all countries|(both )?domestic and international|international and domestic|"
                    r"from all over the world|citizens of any country|all international and domestic", re.I)
INT_RE = re.compile(r"international (students?|applicants?|candidates?)|overseas (students?|fee)|non-eu\b|"
                    r"non-u\.?s\.?\b|foreign (students?|nationals?|applicants?)|outside (the )?(uk|eu|us|united states)",
                    re.I)
EXCL_RE = re.compile(r"(not open to|are not eligible|is not eligible|ineligible|excluding|except|excluded|"
                     r"cannot apply|may not apply)([^.]{0,120})", re.I)
CIT_OF_RE = re.compile(r"(?:citizens?|nationals?|citizenship|passport holders?) (?:of|from) ([^.;:]{3,160})", re.I)
CIT_PRE_RE = re.compile(r"([A-Z][A-Za-z.\- ]{1,30}?) (?:citizens?|nationals?|citizenship)\b")
REGION_RE = re.compile(r"(developing countries|low[- ]income (countries|economies)|lower[- ]middle[- ]income|"
                       r"\bDAC\b|least[- ]developed|commonwealth countries|(african|asian|european|latin american|"
                       r"arab|asean|caribbean) (countries|nationals|citizens|students))", re.I)

def window(t, m, before=80, after=40):
    return t[max(0, m.start() - before): m.end() + after]

NON_RE = re.compile(r"\bnon-([A-Z][a-z]+) (?:citizens?|nationals?|students?|applicants?)")
NOT_ELIG_RE = re.compile(r"([^.]{0,80}?)(?:citizens?|nationals?)[^.]{0,60}?(?:are|is) not eligible", re.I)

def ex_nationality(t):
    # 0) 'non-Chinese citizens', 'US citizens ... are not eligible'
    for m in NON_RE.finditer(t):
        c = canon_country(m.group(1))
        if c: return "EXCLUDE:" + c
    for m in NOT_ELIG_RE.finditer(t):
        cs = countries_in(m.group(0))
        if cs: return "EXCLUDE:" + ";".join(sorted(cs))
    # 1) exclusions with countries
    for m in EXCL_RE.finditer(t):
        cs = countries_in(m.group(2))
        if cs and re.search(r"citizen|national|nationality|from", m.group(0), re.I):
            return "EXCLUDE:" + ";".join(sorted(cs))
    # 2) explicit citizenship requirements
    only = set(); pr = False
    for m in CIT_OF_RE.finditer(t):
        if re.search(REQ, window(t, m), re.I) and not re.search(r"not|except|excluding", window(t, m, 40, 0), re.I):
            only |= countries_in(m.group(1))
    for m in CIT_PRE_RE.finditer(t):
        c = canon_country(m.group(1).split()[-1]) or canon_country(m.group(1))
        if c and re.search(REQ, window(t, m), re.I) and not re.search(r"not|non-", window(t, m, 30, 0), re.I):
            only.add(c)
    if only:
        if re.search(r"permanent residen", t, re.I): pr = True
        return "ONLY:" + ";".join(sorted(only)) + (" (citizen or PR)" if pr else "")
    # 3) regions
    m = REGION_RE.search(t)
    if m and re.search(REQ + r"|from", window(t, m), re.I):
        return "REGION:" + m.group(1).lower()
    # 4) open to all / international
    if ALL_RE.search(t): return "ALL"
    if INT_RE.search(t): return "INTERNATIONAL"
    return ""

# ------------------------------------------------------------------ GPA
GPA_RES = [
    re.compile(r"(?:gpa|grade point average|cgpa)\D{0,25}?(\d\.\d{1,2})(?:\s*(?:/|out of|on a)\s*(\d(?:\.\d+)?))?", re.I),
    re.compile(r"(\d\.\d{1,2})\s*(?:/\s*(\d(?:\.\d+)?))?\s*(?:or (?:higher|above|better)\s*)?(?:cumulative |unweighted |"
               r"weighted )?(?:gpa|grade point average|cgpa)", re.I),
]
AVG_RE = re.compile(r"minimum (?:of )?(\d\.\d{1,2})\s*(?:\([A-Z][+-]?\)\s*)?(?:cumulative )?average", re.I)
PCT_RE = re.compile(r"(?:minimum|at least|average of|overall|score of)[^.%]{0,30}?(\d{2}(?:\.\d)?)\s*%", re.I)

def ex_gpa(t):
    for rx in GPA_RES:
        for m in rx.finditer(t):
            x = float(m.group(1)); base = m.group(2)
            if 1.5 <= x <= 4.3 and (not base or float(base) in (4.0, 5.0, 10.0)):
                ctx = window(t, m, 60, 10).lower()
                if re.search(r"maintain|renew|keep|retain", ctx): continue      # renewal condition, not entry
                scale = "4.0" if (base and float(base) == 4.0) or re.search(r"4\.0 scale|/\s*4\b|out of 4", ctx) else (
                    base or "")
                return m.group(1), scale
    m = AVG_RE.search(t)
    if m: return m.group(1), ""
    l = t.lower()
    if re.search(r"upper second|2[:.]1", l): return "2:1", "UK honours"
    if re.search(r"first[- ]class (honours|degree)|1st class", l): return "first class", "UK honours"
    m = PCT_RE.search(t)
    if m and 50 <= float(m.group(1)) <= 100: return m.group(1) + "%", "percentage average"
    return "", ""

# ------------------------------------------------------------------ enrollment
ENR = {
    "enrolled": r"currently enrolled|current(ly)? (students?|studying|registered|pursuing)|must be enrolled|be enrolled|"
                r"continuing students|returning students|registered students|in (their|the) (second|third|final) year|"
                r"enrolled (full|part)[- ]time|already (enrolled|studying)",
    "admitted": r"offer of (admission|a place)|(have|hold|holding|received|receive) (an? )?(unconditional |conditional |"
                r"firm )?offer|admitted (to|students|for)|accepted (to|into|for|by)|confirmed place|been accepted|"
                r"admission letter",
    "applicant": r"prospective (students?|applicants?)|applying (to|for) (admission|a degree|the program)|incoming|"
                 r"entering (students|freshm|first|their first)|new students|high[- ]school seniors?|"
                 r"apply(ing)? for admission|first-time (students|freshm)|future students|admission application|"
                 r"application for admission",
}
MODE_NOUN = r"(students?|study|studies|enrol\w*|programs?|programmes?|basis|degree|course|load|status)"

def ex_enrollment(t):
    l = t.lower()
    out = [k for k in ["applicant", "admitted", "enrolled"] if re.search(ENR[k], l)]
    ft = re.search(r"full[- ]time " + MODE_NOUN, l); pt = re.search(r"part[- ]time " + MODE_NOUN, l)
    if ft and not pt: out.append("full-time")
    if pt and not ft: out.append("part-time")
    return ";".join(out)

# ------------------------------------------------------------------ deadline
MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|" \
      r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DATE_RES = [
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + MON + r"\.?,?\s*(\d{4})?", re.I), "dmy"),
    (re.compile(r"\b" + MON + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?", re.I), "mdy"),
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
    (re.compile(r"\b" + MON + r"\s+(\d{4})\b", re.I), "my"),
]
DL_KEY = re.compile(r"deadline|closing date|closes|close on|apply by|applications? (?:are |is )?due|due (?:date|by|on)|"
                    r"submit(?:ted)? (?:by|before|no later)|no later than|must be received|applications? close", re.I)
ROLLING = re.compile(r"rolling basis|ongoing basis|at any time|anytime|open all year|year-round|throughout the year", re.I)

def month_num(s):
    s = s.lower().rstrip(".")
    return MONTHS.get(s) or MON_ABBR.get(s[:3])

def ex_deadline(t, today):
    parts = []
    for k in DL_KEY.finditer(t):
        seg = t[k.start(): k.end() + 160]
        hits = []
        for rx, kind in DATE_RES:
            for m in rx.finditer(seg):
                try:
                    if kind == "dmy": d, mo, y = int(m.group(1)), month_num(m.group(2)), m.group(3)
                    elif kind == "mdy": mo, d, y = month_num(m.group(1)), int(m.group(2)), m.group(3)
                    elif kind == "iso": y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
                    else: mo, d, y = month_num(m.group(1)), None, m.group(2)
                except (TypeError, ValueError):
                    continue
                if not mo or (d is not None and not 1 <= d <= 31): continue
                yr = int(y) if y else None
                if yr and not 2015 <= yr <= 2030: continue
                hits.append((m.start(), normalize_deadline_part(yr, mo, d, today)))
        if hits:
            hits.sort()
            p = hits[0][1]
            if p not in parts: parts.append(p)
    if not parts and ROLLING.search(t): parts = ["rolling"]
    return ";".join(parts[:4])

# ------------------------------------------------------------------ host country
TLD = {"uk": "UK", "de": "Germany", "ca": "Canada", "au": "Australia", "fr": "France", "edu": "USA", "cn": "China",
       "kr": "South Korea", "jp": "Japan", "nl": "Netherlands", "it": "Italy", "ae": "UAE", "fi": "Finland",
       "be": "Belgium", "se": "Sweden", "ch": "Switzerland", "ie": "Ireland", "nz": "New Zealand", "sg": "Singapore",
       "hu": "Hungary", "tr": "Turkey", "qa": "Qatar", "sa": "Saudi Arabia", "at": "Austria", "es": "Spain",
       "dk": "Denmark", "no": "Norway", "pl": "Poland", "hk": "Hong Kong", "tw": "Taiwan", "my": "Malaysia"}

def ex_host(name, url):
    host = re.sub(r"^https?://", "", url or "").split("/")[0].lower()
    parts = host.split(".")
    if parts and parts[-1] in TLD: return TLD[parts[-1]]
    if len(parts) >= 2 and parts[-2] in ("ac", "edu") and parts[-1] in TLD: return TLD[parts[-1]]
    m = re.search(r"\b(?:in|study in|to study in)\s+(the\s+)?([A-Z][A-Za-z ]{1,25})", name or "")
    if m:
        c = canon_country(m.group(2).strip())
        if c: return c
    return ""

def extract_row(r, today):
    t = norm(r["eligibility_text"])
    gpa, scale = ex_gpa(t)
    out = {f: "" for f in FIELDS}
    out.update({"degree_levels": ex_degree(r["name"], t), "nationality_rule": ex_nationality(t), "min_gpa": gpa,
                "gpa_scale": scale, "enrollment_status": ex_enrollment(t), "deadline": ex_deadline(t, today),
                "host_country": ex_host(r["name"], r.get("url", ""))})
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="gold_annotations_ok193.csv")
    ap.add_argument("--out", default="extracted_regex.csv")
    ap.add_argument("--date", default="2026-11-21")
    a = ap.parse_args()
    corpus = pd.read_csv(a.corpus, dtype=str, keep_default_na=False)
    rows = [{"id": r["id"], **extract_row(r, a.date)} for _, r in corpus.iterrows()]
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"{len(out)} rows -> {a.out}")
    print((out[["degree_levels", "nationality_rule", "min_gpa", "enrollment_status", "deadline", "host_country"]] != "")
          .sum().to_string())
