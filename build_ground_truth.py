"""Ground truth for the Eligibility-Aware scholarship RAG evaluation.

Inputs : gold_annotations_ok193.csv  (gold criteria, 193 scholarships)
         student_profiles.csv        (50 synthetic students)
Outputs: match_matrix.csv   one row per (student, scholarship): Match per criterion, EligScore, labels, relevance
         qrels.csv          student_id, scholarship_id, relevance (only relevance > 0), TREC-style
         gt_summary.csv     per-student counts

Match values: 1 = satisfies, 0 = violates, 0.5 = uncertain (criterion not stated, or cannot be decided from text).
Core criteria (Stage A of the system): degree, nationality, gpa, enrollment, deadline.
Extra criteria (ground truth only, hand-coded from other_hard): residence, gender, age, special population
(membership / affiliation / heritage / religion; profiles are assumed not to belong to such groups).

label_core : ineligible if any core Match = 0; eligible if all core = 1; else uncertain
label_full : same over core + extra criteria  (this is the reference label)
relevance  : 0 if label_full = ineligible or the field is explicitly different;
             2 if the field explicitly matches; 1 if the field is ANY / not stated.

Run:  python build_ground_truth.py [--date 2026-11-21] [--annual-passed 0.5]
"""
import argparse, re
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--gold", default="gold_annotations_ok193.csv")
ap.add_argument("--profiles", default="student_profiles.csv")
ap.add_argument("--date", default="2026-11-21", help="query date used for deadline checks")
ap.add_argument("--annual-passed", type=float, default=0.5,
                help="Match for an annual deadline whose date has already passed this year (0.5 = next cycle expected)")
args = ap.parse_args()
TODAY = args.date

EU = {"Germany", "France", "Netherlands"}                      # EU members among the profile nationalities
EUROPE = EU | {"UK"}
DEVELOPED = {"USA", "UK", "Germany", "France", "Canada", "Japan", "South Korea", "Netherlands"}
ASEAN = {"Indonesia", "Vietnam", "Philippines"}
AFRICA = {"Nigeria", "Kenya", "Ghana", "Egypt"}
# World Bank low / lower-middle income list printed in S018's text (2026), restricted to profile countries
WB_LMIC = {"Bangladesh", "Egypt", "Ghana", "India", "Kenya", "Nigeria", "Pakistan", "Uzbekistan"}
# DAAD development-country list printed in S069's text, restricted to profile countries
DAAD_LIST = {"Kazakhstan", "Uzbekistan", "India", "Pakistan", "Bangladesh", "Nigeria", "Kenya", "Ghana", "Egypt",
             "Indonesia", "Vietnam", "Philippines", "Brazil", "Mexico", "Iran"}
# DUO-Sweden: EU nationals (Swedish side) + Asian countries listed in S073's text
DUO_SWEDEN = EU | {"Kazakhstan", "India", "Pakistan", "Indonesia", "Vietnam", "Philippines", "China", "South Korea", "Japan"}

def dev(n):            # developing country among the profile set
    return n not in DEVELOPED

# ---------------------------------------------------------------- nationality
HOST = {  # host country for INTERNATIONAL rules (from url / name)
    "S003": "UK", "S048": "Australia", "S110": "USA", "S149": "USA", "S170": "China", "S221": "UAE", "S227": "UK",
    "S229": "UK", "S018": "UK", "S053": "Canada", "S195": "Australia", "S239": "Australia", "S030": "France",
    "S058": "USA", "S089": "France", "S104": "Italy", "S124": "Germany", "S186": "South Korea", "S192": "UK",
    "S193": "USA", "S240": "France"}

def nat_special(sid, n, res):
    """Scholarship-specific nationality rules (REGION and qualified ONLY/EXCLUDE). Return None if not special."""
    if sid == "S031": return 0.5 if dev(n) and n != "China" else 0          # Australia Awards: list not in text
    if sid in ("S046", "S047"):                                             # Europeans; non-Europeans conditional
        return 1 if n in EUROPE else 0.5
    if sid == "S073": return 1 if n in DUO_SWEDEN else 0
    if sid == "S078":                                                       # DAC LDC/low/lower-middle (national or domiciled)
        best = max(1 if x in WB_LMIC else (0.5 if x in {"Vietnam", "Philippines", "Indonesia", "Iran"} else 0) for x in (n, res))
        return best
    if sid == "S142":                                                       # South Asia, Africa, Latin America (residents or citizens)
        reg = {"India", "Pakistan", "Bangladesh", "Brazil", "Mexico"} | AFRICA
        return 1 if (n in reg or res in reg) else 0
    if sid == "S241": return 0.5 if dev(n) and n not in {"China", "Turkey"} else 0   # 25 countries, list not in text
    if sid == "S069": return 1 if n in DAAD_LIST else 0
    if sid == "S011":                                                       # ASEAN citizens / students of European ASEM universities
        if n in ASEAN: return 1
        return 0.5 if res in EUROPE else 0
    if sid == "S032": return 1 if n in ASEAN else 0
    if sid == "S060": return 1 if n in EUROPE | {"Turkey"} else 0
    if sid == "S063": return 0 if n == "UK" else 0.5                        # Chevening list not in text
    if sid == "S108":
        if n in {"USA", "France", "Egypt"}: return 1
        return 0.5 if n == "Turkey" else 0
    if sid == "S153": return 1 if (n in AFRICA and res in AFRICA) else 0
    if sid == "S171": return 1 if n in {"India", "Japan", "USA"} | ASEAN else 0
    if sid == "S199": return 0.5 if dev(n) and n not in {"China", "Brazil", "Mexico"} else 0   # AKF list not in text
    if sid == "S210": return 0.5 if n in {"Egypt", "Iran", "Turkey"} else 0  # 'Middle East', countries not listed
    if sid == "S161": return 1 if (n == "Egypt" and res == "Egypt") else 0   # nationals resident in Egypt
    if sid in ("S102", "S191"):                                              # UK home fee status
        if n == "UK": return 1 if res == "UK" else 0.5
        return 0
    if sid == "S117":                                                        # US citizen or resident
        if n == "USA": return 1
        return 0.5 if res == "USA" else 0
    if sid == "S120": return 0.5 if n == "Iran" else 1                       # US-embargoed: list not in text
    if sid == "S222": return 0 if n == "USA" else (0.5 if n == "Iran" else 1)  # Fulbright participating countries
    return None

def countries(rule):
    body = rule.split(":", 1)[1]
    body = re.sub(r"\(.*?\)", "", body)
    return {c.strip() for c in body.split(";") if c.strip()}

def m_nationality(sid, rule, n, res):
    sp = nat_special(sid, n, res)
    if sp is not None: return sp
    if not rule: return 0.5
    if rule == "ALL": return 1
    if rule == "INTERNATIONAL": return 0 if HOST.get(sid) == n else 1
    if rule.startswith("ONLY:"): return 1 if n in countries(rule) else 0
    if rule.startswith("EXCLUDE:"):
        ex = countries(rule)
        if "EU" in ex: ex |= EU
        return 0 if n in ex else 1
    if rule.startswith("REGION:"): raise ValueError(f"unhandled REGION rule {sid}: {rule}")
    raise ValueError(f"bad nationality rule {sid}: {rule}")

# ---------------------------------------------------------------- degree
# ground-truth fallback where the gold field is blank (from the dataset 'type' column or an implied prior degree)
DEGREE_FALLBACK = {"S135": "master;phd", "S154": "master;phd", "S171": "master;phd", "S163": "bachelor"}
def m_degree(levels, target, sid=None):
    if not levels and sid in DEGREE_FALLBACK: levels = DEGREE_FALLBACK[sid]
    s = {x for x in levels.split(";") if x}
    if not s: return 0.5
    return 1 if target in s else 0

# ---------------------------------------------------------------- GPA
# thresholds converted to a 4.0 scale; band = uncertainty zone for converted / verbal thresholds
QUAL = {"good academic standing": (2.0, 0.0), "far above average": (3.5, 0.3), "very good": (3.5, 0.3),
        "excellent": (3.6, 0.3), "excellent academic standing": (3.6, 0.3), "top third": (3.5, 0.3),
        "top half of class": (3.0, 0.3), "above-average degree": (3.3, 0.3), "good": (3.0, 0.2),
        "2:1": (3.3, 0.15), "first class": (3.7, 0.15), "1st class": (3.7, 0.15),
        "first class (or 2:1 + master's distinction)": (3.5, 0.2),
        "98%": (3.9, 0.15), "85%": (3.4, 0.15), "80%": (3.2, 0.15), "75%": (3.0, 0.15),
        "3.5 or 90%": (3.5, 0.0)}

def gpa_threshold(sid, v, scale, target):
    if v in QUAL: return QUAL[v]
    if v == "85 (ATAR) / 65 (WAM)": return (3.4, 0.2) if target == "bachelor" else (3.0, 0.2)
    if v == "1.5" and "German" in scale: return (3.6, 0.2)          # German scale, 1.0 best
    x = float(v)
    if x <= 4.5: return (x, 0.0)                                     # 4.0 scale (stated or assumed)
    raise ValueError(f"unhandled min_gpa {sid}: {v!r}")

def m_gpa(sid, v, scale, gpa, target):
    if not v: return 0.5
    if gpa == "": return 0.5
    t, band = gpa_threshold(sid, v, scale, target)
    g = float(gpa)
    if band == 0: return 1 if g >= t else 0
    if g >= t + band: return 1
    if g < t - band: return 0
    return 0.5

# ---------------------------------------------------------------- enrollment
def m_enrollment(val, status, mode):
    parts = {x for x in val.split(";") if x}
    S = parts & {"applicant", "admitted", "enrolled"}
    M = parts & {"full-time", "part-time"}
    if not S and not M: return 0.5
    res = []
    if S:
        if not status: res.append(0.5)                                   # status not applicable (postdocs)
        elif status in S: res.append(1)
        elif status == "admitted" and "applicant" in S: res.append(1)     # admitted student is still a new student
        elif status == "applicant" and "admitted" in S: res.append(0.5)   # needs the offer first
        elif status == "admitted" and "enrolled" in S: res.append(0.5)    # will be enrolled once the programme starts
        else: res.append(0)                                              # enrolled vs entrance award, or new vs current-only
    if M: res.append(1 if mode in M else 0)
    return min(res)

# ---------------------------------------------------------------- deadline
def m_deadline(val):
    parts = [p for p in val.split(";") if p]
    if not parts: return 0.5
    best = 0
    for p in parts:
        if p == "rolling": v = 1
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", p): v = 1 if p >= TODAY else 0
        elif re.fullmatch(r"\d{4}-\d{2}", p): v = 1 if p >= TODAY[:7] else 0
        elif p.startswith("annual:"):
            md = p[7:]
            if len(md) == 2: v = 1 if md >= TODAY[5:7] else args.annual_passed
            else: v = 1 if md >= TODAY[5:] else args.annual_passed
        else: raise ValueError(p)
        best = max(best, v)
    return best

# ---------------------------------------------------------------- extra criteria (from other_hard)
def m_residence(sid, n, res):
    R = {
        "S017": lambda: 0.5 if res == "Canada" else 0,               # Alberta resident
        "S084": lambda: 0.5 if res == "Canada" else 0,               # Alberta resident
        "S055": lambda: 1 if res in {"USA", "Canada"} else 0,
        "S057": lambda: 1 if res in {"USA", "Canada"} else 0.5,
        "S125": lambda: 0.5 if res == "USA" else 0,                  # CA / NV / IL
        "S174": lambda: 0.5 if res == "USA" else 0,                  # listed US states
        "S141": lambda: 1 if res == "USA" else 0,
        "S184": lambda: 1 if res == "USA" else 0,
        "S014": lambda: 1 if res == "USA" else 0,
        "S149": lambda: 0 if res == "USA" else 1,                    # non-US residency
        "S212": lambda: 0.5 if res == "USA" else 1,                  # out-of-state or international
        "S087": lambda: 0 if res == "France" else 1,                 # not resident in France
        "S018": lambda: 1 if res in WB_LMIC else 0,                  # domiciled in WB low/lower-middle (list in text)
        "S215": lambda: 0.5 if dev(res) else 0,                      # low-income economy on UAL list
        "S208": lambda: 1 if dev(res) else 0,                        # low/middle income, Middle East, Latin America
        "S139": lambda: 1 if res in {"Pakistan", "Bangladesh"} else 0,
        "S143": lambda: 1 if res in EUROPE else 0,                   # UK or EU
        "S192": lambda: 1 if res in EU else 0,                       # EU member state
        "S243": lambda: 1 if res in {"Germany", "Turkey"} else 0,    # listed countries
        "S177": lambda: 1 if (n == "Germany" or res == "Germany") else 0,  # international track: residing in Germany
        "S045": lambda: 1 if res == "USA" else 0,                    # US high school senior
        "S101": lambda: 1 if res == "USA" else 0,                    # US high school senior
        "S036": lambda: 0.5 if res == "USA" else 0,                  # Calhoun County, Michigan
        "S088": lambda: 0,                                           # applicants in Namibia (no profile lives there)
    }
    return R[sid]() if sid in R else None

# Special-population requirements from other_hard (membership, affiliation, heritage, religion, refugee status ...).
# Profiles are assumed NOT to belong to any such group, so these scholarships are 0 for every student;
# 0.5 where the requirement has stated exceptions or cannot be judged from the profile.
SPECIAL_0 = {"S002", "S006", "S007", "S014", "S020", "S027", "S096", "S100", "S114", "S124", "S125", "S127", "S132",
             "S141", "S145", "S146", "S156", "S159", "S166", "S168", "S179", "S183", "S191", "S211", "S220"}
SPECIAL_HALF = {"S094", "S097", "S204"}
def m_special(sid, res):
    if sid in SPECIAL_0: return 0
    if sid in SPECIAL_HALF: return 0.5
    if sid in ("S123", "S242"): return 0.5 if res == "Netherlands" else 0      # Radboud University students / graduates
    return None

FEMALE = {"S057", "S022", "S075", "S084", "S098"}
def m_gender(sid, g):
    if sid in FEMALE: return 1 if g == "F" else 0
    if sid == "S102": return 1 if g == "F" else 0.5                  # or male of a listed under-represented ethnicity
    if sid == "S117": return 1 if g == "M" else 0
    return None

def m_age(sid, age, target):
    lim = {  # maximum age (inclusive) by target degree; '*' = any degree
        "S133": {"bachelor": 25, "master": 30, "phd": 35}, "S237": {"bachelor": 25, "master": 34, "phd": 39},
        "S246": {"master": 34, "phd": 39}, "S241": {"bachelor": 35, "master": 35},
        "S115": {"*": 34}, "S116": {"*": 29}, "S087": {"*": 25}, "S089": {"master": 29, "phd": 35},
        "S104": {"master": 28, "phd": 30}, "S108": {"*": 29}, "S132": {"*": 25}, "S153": {"*": 29},
        "S240": {"*": 29}}
    if sid == "S107": return 1 if age <= 27 else (0.5 if age <= 30 else 0)   # 27 at graduation, extensions possible
    if sid == "S132" and age < 16: return 0
    if sid == "S237" and target == "bachelor" and age < 18: return 0
    if sid in lim:
        d = lim[sid]; m = d.get(target, d.get("*"))
        if m is None: return None
        return 1 if age <= m else 0
    return None

# ---------------------------------------------------------------- field of study
CATS = {"CS": r"computer|data science|artificial intelligence|\bai\b|digital|stem|science and technology",
        "ENG": r"engineering|aeronautic|space|energy|transportation|electrical|mechanical|civil|electronic|landscape architecture|stem|science and technology",
        "BUS": r"business|mba|financ|economics|management|real estate",
        "SCI": r"mathematic|natural science|exact science|physics|chemistry|atmospheric|oceanic|hydrolog|crop|plant|environmental|stem|science and technology|biomedical|biomedical sciences|sciences, engineering|science and engineering",
        "MED": r"medic|health|pharmac|nursing|biomedical|nutrition|psychiatric|addiction",
        "SOC": r"social science|human rights|international development|international law|international political|international relations|political|politic|psycholog|behavioural|refugee|migration|development policy|journalism|economics",
        "HUM": r"humanities|arts|histor|archaeolog|linguistic|bilingual|islamic",
        "LAW": r"\blaw|laws\b|llm",
        "LANG": r"english language"}
FIELD_OVERRIDE = {"S161": {"ENG", "BUS", "SOC"},     # engineering (not computer science), economics, finance, management
                  "S204": {"BUS"},                   # sport management
                  "S057": {"BUS"},                   # commercial real estate
                  "S107": {"HUM"},                   # historical humanities (incl. history of law/science)
                  "S078": {"SCI"},                   # global plant health
                  "S087": {"HUM", "SCI"}}            # arts and humanities or sciences
FIELD_ANY = {"S032"}                                 # 'priority areas' only, not a restriction
def field_cats(sid, f):
    if not f or sid in FIELD_ANY: return None
    if sid in FIELD_OVERRIDE: return FIELD_OVERRIDE[sid]
    if f.startswith("ANY"):
        return {"__ANY_EXCEPT_MED__"} if "except medicine" in f else None
    s = {c for c, p in CATS.items() if re.search(p, f, re.I)}
    if not s: raise ValueError(f"unmapped field {sid}: {f}")
    return s

def m_field(sc, student):
    if sc is None: return None                       # ANY / not stated
    if sc == {"__ANY_EXCEPT_MED__"}: return 0 if student == {"MED"} else None
    return 1 if sc & student else 0

# ---------------------------------------------------------------- run
gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
prof = pd.read_csv(args.profiles, dtype=str, keep_default_na=False)
rows = []
for _, s in prof.iterrows():
    scats = set(s.field_cats.split(";")); age = int(s.age)
    for _, g in gold.iterrows():
        sid = g["id"]
        core = {"m_degree": m_degree(g.degree_levels, s.target_degree, sid),
                "m_nationality": m_nationality(sid, g.nationality_rule, s.nationality, s.residence),
                "m_gpa": m_gpa(sid, g.min_gpa, g.gpa_scale, s.gpa_4, s.target_degree),
                "m_enrollment": m_enrollment(g.enrollment_status, s.enrollment_status, s.study_mode),
                "m_deadline": m_deadline(g.deadline)}
        extra = {"m_residence": m_residence(sid, s.nationality, s.residence),
                 "m_gender": m_gender(sid, s.gender),
                 "m_age": m_age(sid, age, s.target_degree),
                 "m_special": m_special(sid, s.residence)}
        def label(vals):
            vals = [v for v in vals if v is not None]
            if any(v == 0 for v in vals): return "ineligible"
            return "eligible" if all(v == 1 for v in vals) else "uncertain"
        lc = label(core.values()); lf = label(list(core.values()) + list(extra.values()))
        fm = m_field(field_cats(sid, g.field_of_study), scats)
        if lf == "ineligible" or fm == 0: rel = 0
        elif fm == 1: rel = 2
        else: rel = 1
        rows.append({"student_id": s.student_id, "scholarship_id": sid, **core,
                     "elig_score": round(sum(core.values()) / 5, 3), **extra,
                     "label_core": lc, "label_full": lf, "field_match": fm, "relevance": rel})

mm = pd.DataFrame(rows)
mm.to_csv("match_matrix.csv", index=False)
mm[mm.relevance > 0][["student_id", "scholarship_id", "relevance"]].to_csv("qrels.csv", index=False)
summ = mm.groupby("student_id").agg(
    eligible=("label_full", lambda x: (x == "eligible").sum()),
    uncertain=("label_full", lambda x: (x == "uncertain").sum()),
    ineligible=("label_full", lambda x: (x == "ineligible").sum()),
    rel2=("relevance", lambda x: (x == 2).sum()),
    rel1=("relevance", lambda x: (x == 1).sum())).reset_index()
summ.to_csv("gt_summary.csv", index=False)
print(f"{len(mm)} pairs; label_full: {mm.label_full.value_counts().to_dict()}")
print(f"label_core: {mm.label_core.value_counts().to_dict()}")
print(f"relevance: {mm.relevance.value_counts().sort_index().to_dict()}")
print(summ.describe().loc[["min", "mean", "max"]].round(1).to_string())
