"""
Step 0 — the Kaggle dataset has no links, so find the official page for each scholarship.

  1. Cleans the Kaggle CSV: drops duplicate titles (same scholarship listed under
     several countries), course-only offers and obvious marketing/contest listings.
  2. Searches the web (DuckDuckGo, free, no API key) for each title.
  3. Picks the most "official-looking" result: university / government / foundation
     domains first, scholarship aggregators last, social media never.
  4. Writes scholarships_with_urls.csv  ->  feed it to scrape_scholarships.py

Install:  pip install ddgs pandas
Run:      python find_urls.py --csv Universities_Schoolarships_All_Around_the_World.csv
Then:     python scrape_scholarships.py --csv scholarships_with_urls.csv --limit 300

Re-running is safe: finished searches are cached in search_cache.json.
"""

import argparse
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

try:
    from ddgs import DDGS
except ImportError:  # older package name
    from duckduckgo_search import DDGS

NEVER = ["facebook.", "youtube.", "linkedin.", "twitter.", "x.com", "instagram.", "tiktok.",
         "pinterest.", "reddit.", "quora.", "wikipedia.", "scholarshipportal.", "studyportals.",
         "mastersportal.", "bachelorsportal.", "phdportal.", "scribd.", "slideshare.", "tavily.com",
         "finance.yahoo.", "stocktitan.", "huggingface.", "company-information.service.gov.uk",
         "2001-2009.state.gov",]
AGGREGATORS = ["scholars4dev", "opportunitiesforyouth", "scholarshipsads", "afterschoolafrica",
               "opportunitydesk", "worldscholarshipforum", "scholarshipdb", "globalscholarships",
               "internationalscholarships", "iefa.org", "scholarship-positions", "fastweb",
               "scholarships.com", "bold.org", "studyabroad", "youthop", "scholarshiproar",
               "careers360", "buddy4study", "studentscholarships", "greatyop", "scholarshipset",
               "advance-africa", "unigrants", "scholarshipunion", "chevening.org.ng",
               "bigfuture.collegeboard", "wemakescholars", "postgraduatestudentships", "collegevine",
               "collegeessayguy", "opportunitiesforafricans", "opportunitiesforyou", "o4af.", "scholarwing",
               "nomadcredit", "chinesescholarshipcouncil", "mina7portal", "opportunitiescorners",
               "opportunitiescircle", "scholarships360", "scholarshipscorner", "gradgermany",
               "european-funding-guide", "edugist", "theglobalscholarship", "leapscholar", "titansabroad",
               "gradmap", "getintocollege", "jlvcollegecounseling", "accessandequity", "penprofile", "lisan.pk",
               ]
OFFICIAL_HINTS = [".edu", ".ac.", ".gov", ".gc.ca", ".europa.eu", "daad.", "fulbright",
                  "chevening.org", "erasmus", "foundation", "university", "college", "uni-"]

JUNK_TITLE = re.compile(r"\b(?:free course|contest|competition|% off|off your tuition|giveaway|"
                        r"laptop|no-essay|quran learning)\b", re.I)


def domain(url):
    return urlparse(url).netloc.lower().removeprefix("www.")


def clean_query(title):
    t = str(title).split("|")[0].strip()
    t = re.sub(r"\s+", " ", t)
    if t.isupper():
        t = t.title()
    if "scholarship" not in t.lower():
        t += " scholarship"
    return f"{t} eligibility"


def title_tokens(title):
    stop = {"scholarship", "scholarships", "for", "international", "students", "student", "at",
            "the", "of", "in", "and", "award", "awards", "program", "programme", "uk", "usa",
            "canada", "grant", "grants", "fully", "funded", "to", "a", "phd", "master", "masters"}
    return [t for t in re.findall(r"[a-z]{3,}", title.lower()) if t not in stop]


def score_result(res, title):
    url = res.get("href") or res.get("url") or ""
    d = domain(url)
    if not url or any(n in d for n in NEVER) or url.lower().endswith(".pdf"):
        return -100, url
    s = 0
    if any(a in d for a in AGGREGATORS):
        s -= 5
    if any(h in d for h in OFFICIAL_HINTS):
        s += 5
    toks = title_tokens(title)
    blob = (d + " " + (res.get("title") or "")).lower()
    s += sum(1 for t in toks if t in blob)          # e.g. "brighton" in brighton.ac.uk
    if re.search(r"scholarship|funding|fees|award|bursar|fellowship", url, re.I):
        s += 2
    return s, url


import os, json, hashlib
from tavily import TavilyClient

_tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
CACHE_FILE = Path(__file__).parent / "tavily_cache.json"
try:
    with open(CACHE_FILE, encoding="utf-8") as f:
        _cache = json.load(f)
except FileNotFoundError:
    _cache = {}

def _save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False, indent=1)

def search(ddgs, query, n=8):
    """ddgs оставлен для совместимости со старыми вызовами, не используется."""
    key = hashlib.md5(f"{query}|{n}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    for attempt in range(3):
        try:
            resp = _tavily.search(query, max_results=n)
            results = [
                {"title": r.get("title", ""), "href": r["url"], "body": r.get("content", "")}
                for r in resp.get("results", [])
            ]
            if not results:
                print(f"    no results for: {query}")
            _cache[key] = results
            _save_cache()
            return results
        except Exception as e:
            wait = 3 * (2 ** attempt)
            print(f"    search error ({type(e).__name__}: {e}), retry in {wait}s")
            time.sleep(wait)
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="scholarships_with_urls.csv")
    ap.add_argument("--cache", default="search_cache.json")
    ap.add_argument("--delay", type=float, default=0, help="seconds between searches")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, encoding_errors="replace")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    n0 = len(df)
    df["title"] = df["title"].astype(str).str.strip()
    # same scholarship appears under several locations -> keep one, remember all locations
    locs = df.groupby("title")["location"].apply(lambda s: ";".join(sorted(set(s))))
    df = df.drop_duplicates("title").copy()
    df["location"] = df["title"].map(locs)
    df = df[df["degrees"].fillna("") != "Course"]
    df = df[~df["title"].str.contains(JUNK_TITLE)]
    print(f"{n0} rows -> {len(df)} unique, non-junk scholarships")

    cache_file = Path(args.cache)
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}

    with DDGS() as ddgs:
        for k, title in enumerate(df["title"], 1):
            if title in cache:
                continue
            q = f"{title} eligibility requirements"
            cache[title] = search(ddgs, q)
            if k % 20 == 0:
                print(f"  {k}/{len(df)}")
                cache_file.write_text(json.dumps(cache), encoding="utf-8")
            time.sleep(args.delay)
    cache_file.write_text(json.dumps(cache), encoding="utf-8")

    urls, alts, kinds = [], [], []
    for title in df["title"]:
        ranked = sorted((score_result(r, title) for r in cache.get(title, [])), reverse=True)
        ranked = [(s, u) for s, u in ranked if s > -100]
        if not ranked:
            urls.append(""); alts.append(""); kinds.append("not_found")
            continue
        best_s, best_u = ranked[0]
        urls.append(best_u)
        alts.append(" | ".join(u for _, u in ranked[1:4]))
        d = domain(best_u)
        kinds.append("aggregator" if any(a in d for a in AGGREGATORS)
                     else "official_likely" if best_s >= 5 else "check")
    df["url"] = urls
    df["alt_urls"] = alts          # if the first page is bad, try these
    df["url_quality"] = kinds      # official_likely / check / aggregator / not_found
    df.to_csv(args.out, index=False)
    print(df["url_quality"].value_counts().to_string())
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
