"""Shared definitions for extraction, Stage A filtering and evaluation:
field list, country gazetteer and country groups, value normalizers."""
import re

FIELDS = ["degree_levels", "nationality_rule", "min_gpa", "gpa_scale", "enrollment_status", "deadline",
          "field_of_study", "funding_type", "other_hard", "evidence_sentence", "host_country", "notes"]
CORE = ["degree_levels", "nationality_rule", "min_gpa", "enrollment_status", "deadline"]
DEGREES = ["bachelor", "master", "phd", "postdoc", "other"]
ENROLL = ["applicant", "admitted", "enrolled", "full-time", "part-time"]
FUNDING = ["full", "partial", "tuition", "stipend", "travel"]

# ------------------------------------------------------------------ countries
COUNTRIES = """Afghanistan;Albania;Algeria;Andorra;Angola;Antigua and Barbuda;Argentina;Armenia;Australia;Austria;
Azerbaijan;Bahamas;Bahrain;Bangladesh;Barbados;Belarus;Belgium;Belize;Benin;Bhutan;Bolivia;Bosnia and Herzegovina;
Botswana;Brazil;Brunei;Bulgaria;Burkina Faso;Burundi;Cabo Verde;Cambodia;Cameroon;Canada;Central African Republic;Chad;
Chile;China;Colombia;Comoros;Congo;Democratic Republic of the Congo;Costa Rica;Cote d'Ivoire;Croatia;Cuba;Cyprus;
Czech Republic;Denmark;Djibouti;Dominica;Dominican Republic;Ecuador;Egypt;El Salvador;Equatorial Guinea;Eritrea;Estonia;
Eswatini;Ethiopia;Fiji;Finland;France;Gabon;Gambia;Georgia;Germany;Ghana;Greece;Grenada;Guatemala;Guinea;Guinea-Bissau;
Guyana;Haiti;Honduras;Hong Kong;Hungary;Iceland;India;Indonesia;Iran;Iraq;Ireland;Israel;Italy;Jamaica;Japan;Jordan;
Kazakhstan;Kenya;Kiribati;Kosovo;Kuwait;Kyrgyzstan;Laos;Latvia;Lebanon;Lesotho;Liberia;Libya;Liechtenstein;Lithuania;
Luxembourg;Madagascar;Malawi;Malaysia;Maldives;Mali;Malta;Marshall Islands;Mauritania;Mauritius;Mexico;Micronesia;
Moldova;Monaco;Mongolia;Montenegro;Morocco;Mozambique;Myanmar;Namibia;Nauru;Nepal;Netherlands;New Zealand;Nicaragua;
Niger;Nigeria;North Korea;North Macedonia;Norway;Oman;Pakistan;Palau;Palestine;Panama;Papua New Guinea;Paraguay;Peru;
Philippines;Poland;Portugal;Qatar;Romania;Russia;Rwanda;Saint Kitts and Nevis;Saint Lucia;
Saint Vincent and the Grenadines;Samoa;San Marino;Sao Tome and Principe;Saudi Arabia;Senegal;Serbia;Seychelles;
Sierra Leone;Singapore;Slovakia;Slovenia;Solomon Islands;Somalia;South Africa;South Korea;South Sudan;Spain;Sri Lanka;
Sudan;Suriname;Sweden;Switzerland;Syria;Taiwan;Tajikistan;Tanzania;Thailand;Timor-Leste;Togo;Tonga;
Trinidad and Tobago;Tunisia;Turkey;Turkmenistan;Tuvalu;Uganda;Ukraine;UAE;UK;USA;Uruguay;Uzbekistan;Vanuatu;
Venezuela;Vietnam;Yemen;Zambia;Zimbabwe""".replace("\n", "").split(";")
COUNTRIES = [c.strip() for c in COUNTRIES if c.strip()]

ALIASES = {
    "united states": "USA", "united states of america": "USA", "u.s.": "USA", "u.s.a.": "USA", "us": "USA",
    "america": "USA", "american": "USA", "u.s": "USA",
    "united kingdom": "UK", "great britain": "UK", "britain": "UK", "british": "UK", "england": "UK",
    "scotland": "UK", "wales": "UK", "northern ireland": "UK", "u.k.": "UK",
    "korea": "South Korea", "republic of korea": "South Korea", "korean": "South Korea",
    "korea dpr": "North Korea", "democratic people's republic of korea": "North Korea",
    "viet nam": "Vietnam", "vietnamese": "Vietnam", "turkiye": "Turkey", "türkiye": "Turkey",
    "czechia": "Czech Republic", "lao": "Laos", "lao pdr": "Laos", "lao people's democratic republic": "Laos",
    "united arab emirates": "UAE", "emirati": "UAE", "russian federation": "Russia", "côte d'ivoire": "Cote d'Ivoire",
    "ivory coast": "Cote d'Ivoire", "the gambia": "Gambia", "swaziland": "Eswatini", "burma": "Myanmar",
    "cape verde": "Cabo Verde", "syrian arab republic": "Syria", "west bank and gaza": "Palestine",
    "macedonia": "North Macedonia", "holland": "Netherlands", "dutch": "Netherlands", "german": "Germany",
    "french": "France", "chinese": "China", "japanese": "Japan", "indian": "India", "nigerian": "Nigeria",
    "kenyan": "Kenya", "ghanaian": "Ghana", "egyptian": "Egypt", "pakistani": "Pakistan", "iranian": "Iran",
    "canadian": "Canada", "mexican": "Mexico", "brazilian": "Brazil", "indonesian": "Indonesia",
    "filipino": "Philippines", "kazakh": "Kazakhstan", "kazakhstani": "Kazakhstan", "uzbek": "Uzbekistan",
    "bangladeshi": "Bangladesh", "turkish": "Turkey", "australian": "Australia", "hungarian": "Hungary",
    "italian": "Italy", "spanish": "Spain", "swedish": "Sweden", "finnish": "Finland", "irish": "Ireland",
}
_CANON = {c.lower(): c for c in COUNTRIES}

def canon_country(s):
    """Map a country name / alias / demonym to the canonical name, or None."""
    k = s.strip().strip(".").lower()
    k = re.sub(r"^the ", "", k)
    return _CANON.get(k) or ALIASES.get(k)

# pattern to find country mentions in free text (longest names first)
_names = sorted(list(_CANON.keys()) + [a for a in ALIASES if len(a) > 3 and a not in ("america", "american")],
                key=len, reverse=True)
COUNTRY_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(n) for n in _names) + r")(?![A-Za-z])", re.I)
US_RE = re.compile(r"(?<![A-Za-z])(U\.S\.A?\.?|USA|United States)(?![A-Za-z])")
UK_RE = re.compile(r"(?<![A-Za-z])(UK|U\.K\.|United Kingdom)(?![A-Za-z])")

def countries_in(text):
    found = {canon_country(m.group(1)) for m in COUNTRY_RE.finditer(text)}
    if US_RE.search(text): found.add("USA")
    if UK_RE.search(text): found.add("UK")
    return {c for c in found if c}

# ------------------------------------------------------------------ groups
EU = {"Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland",
      "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
      "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"}
EEA = EU | {"Iceland", "Liechtenstein", "Norway"}
EUROPE = EEA | {"UK", "Switzerland", "Albania", "Bosnia and Herzegovina", "Kosovo", "Montenegro", "North Macedonia",
                "Serbia", "Moldova", "Ukraine", "Belarus", "Andorra", "Monaco", "San Marino"}
ASEAN = {"Brunei", "Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar", "Philippines", "Singapore", "Thailand",
         "Vietnam"}
AFRICA = {"Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde", "Cameroon",
          "Central African Republic", "Chad", "Comoros", "Congo", "Democratic Republic of the Congo", "Cote d'Ivoire",
          "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana",
          "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali",
          "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
          "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan",
          "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"}
LATAM = {"Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Costa Rica", "Cuba", "Dominican Republic", "Ecuador",
         "El Salvador", "Guatemala", "Haiti", "Honduras", "Mexico", "Nicaragua", "Panama", "Paraguay", "Peru",
         "Uruguay", "Venezuela"}
SOUTH_ASIA = {"Afghanistan", "Bangladesh", "Bhutan", "India", "Maldives", "Nepal", "Pakistan", "Sri Lanka"}
CENTRAL_ASIA = {"Kazakhstan", "Kyrgyzstan", "Tajikistan", "Turkmenistan", "Uzbekistan"}
MIDDLE_EAST = {"Bahrain", "Egypt", "Iran", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Oman", "Palestine",
               "Qatar", "Saudi Arabia", "Syria", "Turkey", "UAE", "Yemen"}
ARAB = {"Algeria", "Bahrain", "Egypt", "Iraq", "Jordan", "Kuwait", "Lebanon", "Libya", "Mauritania", "Morocco", "Oman",
        "Palestine", "Qatar", "Saudi Arabia", "Somalia", "Sudan", "Syria", "Tunisia", "UAE", "Yemen"}
ASIA = ASEAN | SOUTH_ASIA | CENTRAL_ASIA | {"China", "Japan", "South Korea", "North Korea", "Mongolia", "Taiwan",
                                            "Hong Kong", "Timor-Leste"}
HIGH_INCOME = EEA | {"UK", "Switzerland", "USA", "Canada", "Australia", "New Zealand", "Japan", "South Korea",
                     "Singapore", "Israel", "Hong Kong", "Taiwan", "UAE", "Qatar", "Saudi Arabia", "Kuwait", "Bahrain",
                     "Oman", "Brunei", "Andorra", "Monaco", "San Marino", "Uruguay", "Chile", "Panama"}

GROUPS = [  # (pattern, member set)
    (r"\bEEA\b|european economic area", EEA),
    (r"\bEU\b|european union", EU),
    (r"europe", EUROPE),
    (r"asean|south[- ]?east(ern)? asia", ASEAN),
    (r"sub-saharan|africa", AFRICA),
    (r"latin america|south america|central america|caribbean", LATAM),
    (r"south asia", SOUTH_ASIA),
    (r"central asia", CENTRAL_ASIA),
    (r"middle east|mena\b", MIDDLE_EAST),
    (r"\barab", ARAB),
    (r"\basia(n)?\b", ASIA),
]
DEVELOPING_RE = re.compile(r"developing|low[- ]income|lower[- ]middle|middle[- ]income|\bDAC\b|least[- ]developed|"
                           r"emerging", re.I)

def is_developing(c):
    return c not in HIGH_INCOME

def expand_countries(spec, strip_parens=True):
    """Return (explicit countries, countries from recognised groups, developing_flag) found in a rule string.
    strip_parens=True drops bracketed qualifiers such as '(citizen or PR)' before looking for countries."""
    text = re.sub(r"\(.*?\)", "", spec) if strip_parens else spec
    explicit = set()
    for part in re.split(r"[;,/]| and | or ", text):
        c = canon_country(part)
        if c: explicit.add(c)
    explicit |= countries_in(text)
    grouped, specific_asia = set(), False
    gtext = COUNTRY_RE.sub(" ", text)          # 'South Africa' must not trigger the Africa group
    for pat, members in GROUPS:
        if pat == r"\basia(n)?\b" and specific_asia: continue
        if re.search(pat, gtext, re.I):
            grouped |= members
            if members in (ASEAN, SOUTH_ASIA, CENTRAL_ASIA): specific_asia = True
    return explicit, grouped, bool(DEVELOPING_RE.search(spec))

# ------------------------------------------------------------------ GPA
UK_CLASS = [(r"first|1st", 3.7), (r"2[:.]1|upper second", 3.3), (r"2[:.]2|lower second", 3.0)]
VERBAL = [(r"good academic standing|satisfactory", 2.0, 0.0), (r"far above average|outstanding|excellent|top (5|10)",
           3.6, 0.3), (r"very good|top (third|quarter|25)|above[- ]average|high", 3.4, 0.3),
          (r"top half|good|average", 3.0, 0.3)]
PCT_POINTS = [(60, 2.0), (70, 2.7), (75, 3.0), (80, 3.2), (85, 3.4), (90, 3.6), (95, 3.8), (98, 3.9), (100, 4.0)]

def pct_to_4(p):
    for (x0, y0), (x1, y1) in zip(PCT_POINTS, PCT_POINTS[1:]):
        if x0 <= p <= x1: return y0 + (y1 - y0) * (p - x0) / (x1 - x0)
    return 2.0 if p < 60 else 4.0

def gpa_to_4(value, scale="", target=""):
    """Convert a minimum-grade string to (threshold on 4.0 scale, uncertainty band). None if not convertible."""
    v = (value or "").strip().lower()
    if not v: return None
    s = (scale or "").lower()
    if " or " in v and "(" not in v:                     # alternatives: '3.5 or 90%' -> the easier one
        alts = [gpa_to_4(x, scale, target) for x in v.split(" or ")]
        alts = [a for a in alts if a]
        if alts: return min(alts)
    if "atar" in v and "wam" in v:                       # '85 (ATAR) / 65 (WAM)'
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", v)]
        return (pct_to_4(nums[0]), 0.2) if target == "bachelor" else (pct_to_4(nums[-1]), 0.2)
    for pat, t in UK_CLASS:
        if re.search(pat, v): return (t, 0.2 if "or" in v else 0.15)
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", v)
    if m: return (pct_to_4(float(m.group(1))), 0.15)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*(\d+(?:\.\d+)?)", v)
    if m:
        x, base = float(m.group(1)), float(m.group(2))
        return (x / base * 4.0, 0.0 if base == 4 else 0.15)
    m = re.search(r"\d+(?:\.\d+)?", v)
    if m:
        x = float(m.group(0))
        if "german" in s or "1.0 best" in s: return (max(1.0, 5.0 - x), 0.2)
        if re.match(r"\s*5(\.0)?\b", s) and x <= 5: return (x / 5 * 4, 0.15)
        if x <= 4.3: return (x, 0.0)
        if x <= 100: return (pct_to_4(x), 0.2)
    for pat, t, band in VERBAL:
        if re.search(pat, v): return (t, band)
    return None

# ------------------------------------------------------------------ deadlines
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
MON_ABBR = {m[:3]: i for m, i in MONTHS.items()}

def normalize_deadline_part(year, month, day, today):
    """Return YYYY-MM-DD for a future dated deadline, annual:MM-DD for a past or yearless one."""
    if day:
        md = f"{month:02d}-{day:02d}"
        if year:
            d = f"{year}-{md}"
            return d if d >= today else f"annual:{md}"
        return f"annual:{md}"
    if year:
        ym = f"{year}-{month:02d}"
        return ym if ym >= today[:7] else f"annual:{month:02d}"
    return f"annual:{month:02d}"

DEADLINE_PART_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|annual:\d{2}(-\d{2})?|rolling)$")

def split_multi(v):
    return [x.strip() for x in (v or "").split(";") if x.strip()]
