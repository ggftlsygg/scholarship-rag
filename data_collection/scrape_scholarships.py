"""
Stage 1 — collect scholarship texts for the Eligibility-Aware RAG project.

What it does
  1. Loads the Kaggle CSV, auto-detects the URL / name columns.
  2. Downloads every page in parallel (with cache + retries, so you can re-run safely).
  3. Extracts the main text (trafilatura, BeautifulSoup fallback).
  4. Cuts out the "eligibility" part (sentences around eligibility keywords).
  5. Flags pages that need manual work (errors, too short, list pages, non-English).
  6. Writes:
       out/scholarships_raw.csv     -> id, name, url, status, raw_text, eligibility_text, ...
       out/texts/<id>.txt           -> full text per scholarship (corpus for RAG)
       out/annotation_template.csv  -> empty gold-criteria columns + regex hints
       out/needs_manual.csv         -> rows to fix by hand (copy text from site)

Colab:
  !pip install -q trafilatura beautifulsoup4 pandas requests langdetect
  !python scrape_scholarships.py --csv scholarships.csv --limit 300

Local re-run after fixing things: same command, cached pages are not re-downloaded.
"""

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

try:
    import trafilatura
except ImportError:
    trafilatura = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
try:
    from langdetect import detect as lang_detect
except ImportError:
    lang_detect = None

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

# Sentences containing these are considered eligibility-related.
ELIG_KEYWORDS = [
    r"eligib", r"requirement", r"criteria", r"who can apply", r"applicants? (must|should)",
    r"must (be|have|hold)", r"open (only )?to", r"citizen", r"national(s|ity)?\b",
    r"resident", r"international students?", r"domestic", r"\bgpa\b", r"grade point",
    r"cgpa", r"minimum (grade|score|average)", r"undergraduate", r"bachelor", r"master",
    r"ph\.?d", r"doctoral", r"postgraduate", r"graduate students?", r"enrol", r"full[- ]time",
    r"deadline", r"closing date", r"apply by", r"ielts", r"toefl", r"age (limit|under|below)",
    r"under the age", r"years old", r"financial need", r"developing countr",
]
ELIG_RE = re.compile("|".join(ELIG_KEYWORDS), re.I)

# Quick hints for manual annotation (NOT the final extractor).
HINT_PATTERNS = {
    "gpa": re.compile(r"(?:[^.\n]|\.(?=\d))*(\bgpa\b|cgpa|grade point|minimum (grade|average|score))(?:[^.\n]|\.(?=\d))*", re.I),
    "degree": re.compile(r"[^.\n]*\b(undergraduate|bachelor'?s?|master'?s?|ph\.?d|doctoral|postgraduate)\b[^.\n]*", re.I),
    "nationality": re.compile(r"[^.\n]*\b(citizens?|nationals?|nationality|residents?|international students?|domestic students?|developing countr\w*)\b[^.\n]*", re.I),
    "deadline": re.compile(r"[^.\n]*\b(deadline|closing date|apply by|applications? (close|due))\b[^.\n]*", re.I),
}

GOLD_COLUMNS = [
    "degree_levels",          # e.g. bachelor;master
    "nationality_rule",       # e.g. ALL_INTERNATIONAL / ONLY:Japan / EXCLUDE:USA / REGION:developing
    "min_gpa",                # e.g. 3.0
    "gpa_scale",              # e.g. 4.0 / 100 / 5.0
    "enrollment_status",      # e.g. admitted / enrolled / applicant / full-time
    "deadline",               # YYYY-MM-DD or "annual: MM-DD" or "rolling"
    "field_of_study",         # e.g. ANY / economics;business
    "funding_type",           # full / partial / tuition
    "other_hard",             # e.g. age<=35, IELTS>=6.5
    "evidence_sentence",      # copy the key sentence from the text
    "annotator",              # your initials
    "notes",
]


def find_col(df, candidates):
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        for low, orig in cols.items():
            if cand in low:
                return orig
    return None


def detect_url_col(df):
    col = find_col(df, ["url", "link", "website", "href"])
    if col:
        return col
    # fallback: the column where most values look like URLs
    best, best_share = None, 0.0
    for c in df.columns:
        s = df[c].astype(str)
        share = s.str.match(r"^\s*https?://").mean()
        if share > best_share:
            best, best_share = c, share
    return best if best_share > 0.3 else None


def cache_path(cache_dir, url):
    return cache_dir / (hashlib.md5(url.encode()).hexdigest() + ".json")


def fetch(url, cache_dir, timeout=20, retries=2):
    cp = cache_path(cache_dir, url)
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))
    result = {"url": url, "html": "", "http_status": None, "error": "", "final_url": url}
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
            result["http_status"] = r.status_code
            result["final_url"] = r.url
            ctype = r.headers.get("Content-Type", "")
            if "pdf" in ctype.lower() or r.url.lower().endswith(".pdf"):
                result["error"] = "pdf"  # handle manually or with pdftotext
            elif r.ok:
                r.encoding = r.apparent_encoding or r.encoding
                result["html"] = r.text
                result["error"] = ""
            else:
                result["error"] = f"http_{r.status_code}"
            break
        except Exception as e:  # noqa: BLE001
            result["error"] = type(e).__name__
            time.sleep(1.5 * (attempt + 1))
    cp.write_text(json.dumps(result), encoding="utf-8")
    return result


def extract_text(html, url):
    text = ""
    if trafilatura is not None and html:
        text = trafilatura.extract(html, url=url, include_tables=True,
                                   include_comments=False, favor_recall=True) or ""
    if len(text) < 300 and BeautifulSoup is not None and html:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body or soup
        alt = main.get_text("\n", strip=True)
        if len(alt) > len(text):
            text = alt
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 2]


def eligibility_section(text, window=1):
    """Sentences matching eligibility keywords, plus `window` neighbours for context."""
    sents = split_sentences(text)
    keep = set()
    for i, s in enumerate(sents):
        if ELIG_RE.search(s):
            for j in range(max(0, i - window), min(len(sents), i + window + 1)):
                keep.add(j)
    return "\n".join(sents[i] for i in sorted(keep))


def hints(text):
    out = {}
    for key, pat in HINT_PATTERNS.items():
        found = []
        for m in pat.finditer(text):
            s = m.group(0).strip()
            if s and s not in found:
                found.append(s[:250])
            if len(found) >= 3:
                break
        out[f"hint_{key}"] = " || ".join(found)
    return out


def classify(text, elig_text, fetch_error):
    if fetch_error:
        return fetch_error
    if len(text) < 400:
        return "too_short"          # often JS-rendered page -> copy manually
    if lang_detect is not None:
        try:
            if lang_detect(text[:2000]) != "en":
                return "non_english"
        except Exception:  # noqa: BLE001
            pass
    if not elig_text or len(elig_text) < 150:
        return "no_eligibility_found"
    # landing pages listing many scholarships (lots of "scholarship" headings, little detail)
    if text.lower().count("scholarship") > 40 and len(elig_text) / max(len(text), 1) < 0.1:
        return "list_page"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Kaggle CSV path")
    ap.add_argument("--out", default="out")
    ap.add_argument("--limit", type=int, default=300, help="max scholarships to keep")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "texts").mkdir(parents=True, exist_ok=True)
    cache_dir = out / "cache"
    cache_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.csv, encoding_errors="replace")
    print("Columns:", list(df.columns))
    url_col = detect_url_col(df)
    if url_col is None:
        raise SystemExit("Could not find a URL column — pass a CSV with links.")
    name_col = find_col(df, ["scholarship", "title", "name"]) or url_col
    print(f"Using URL column: {url_col!r}, name column: {name_col!r}")

    df = df[df[url_col].astype(str).str.match(r"^\s*https?://")].copy()
    df[url_col] = df[url_col].astype(str).str.strip()
    df = df.drop_duplicates(subset=[url_col])
    # take more than needed: some will fail; you then pick the best `limit`
    pool = df.sample(n=min(len(df), int(args.limit * 1.5)), random_state=args.seed)
    print(f"Fetching {len(pool)} pages (target {args.limit} usable)...")

    urls = pool[url_col].tolist()
    fetched = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, u, cache_dir): u for u in urls}
        for k, f in enumerate(as_completed(futs), 1):
            fetched[futs[f]] = f.result()
            if k % 25 == 0:
                print(f"  {k}/{len(urls)}")

    rows = []
    for _, r in pool.iterrows():
        u = r[url_col]
        res = fetched[u]
        text = extract_text(res["html"], u) if res["html"] else ""
        elig = eligibility_section(text) if text else ""
        status = classify(text, elig, res["error"])
        row = {c: r[c] for c in pool.columns}  # keep original Kaggle metadata
        row.update({
            "name": r[name_col],
            "url": u,
            "final_url": res["final_url"],
            "status": status,
            "n_chars": len(text),
            "n_chars_elig": len(elig),
            "raw_text": text,
            "eligibility_text": elig,
        })
        rows.append(row)

    res_df = pd.DataFrame(rows)
    # best first: ok, then the rest; keep the top `limit` ok rows as the corpus
    res_df["_ok"] = (res_df["status"] == "ok").astype(int)
    res_df = res_df.sort_values(["_ok", "n_chars_elig"], ascending=False).drop(columns="_ok")
    res_df.insert(0, "id", [f"S{i:03d}" for i in range(1, len(res_df) + 1)])

    res_df.to_csv(out / "scholarships_raw.csv", index=False)
    for _, r in res_df[res_df["status"] == "ok"].iterrows():
        (out / "texts" / f"{r['id']}.txt").write_text(
            f"{r['name']}\n{r['url']}\n\n{r['raw_text']}", encoding="utf-8")

    ok = res_df[res_df["status"] == "ok"].head(args.limit)
    ann = ok[["id", "name", "url", "eligibility_text"]].copy()
    hint_df = pd.DataFrame([hints(t) for t in ok["raw_text"]], index=ok.index)
    ann = pd.concat([ann, hint_df], axis=1)
    for c in GOLD_COLUMNS:
        ann[c] = ""
    ann.to_csv(out / "annotation_template.csv", index=False)

    manual = res_df[res_df["status"] != "ok"][["id", "name", "url", "final_url", "status", "n_chars"]]
    manual.to_csv(out / "needs_manual.csv", index=False)

    print("\nStatus counts:")
    print(res_df["status"].value_counts().to_string())
    print(f"\nUsable (ok): {len(ok)} / target {args.limit}")
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
