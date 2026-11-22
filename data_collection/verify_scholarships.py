"""
Verify the corpus against the live official pages.

For every scholarship it:
  1. opens the `url` (cached, parallel, retries),
  2. extracts the live page text,
  3. compares it with the dataset text (how much of the dataset text is still on the page),
  4. checks that the scholarship name is on the page and finds the latest year mentioned,
  5. assigns a verdict:
       same          - page alive, dataset text still there (>= 60% overlap)        -> OK
       changed       - page alive, text partly updated (20-60%)                      -> re-check criteria
       different     - page alive but content is something else (< 20%, name absent) -> replace / manual
       redirected    - sent to a homepage / generic page                             -> manual
       blocked       - 403 / JS page / too short to compare                          -> open in browser
       dead          - 404, DNS error, timeout                                       -> replace from reserve

Run:  python verify_scholarships.py --csv corpus/corpus_clean.csv
Needs scrape_scholarships.py in the same folder.

Outputs (corpus/):
  verification.csv         one row per scholarship: verdict, overlap, latest_year, final_url, ...
  verification_sample.csv  random 30 "same" rows for a quick manual spot-check
  live_texts/<id>.txt      the current page text (use it if you decide to update a "changed" one)
"""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from scrape_scholarships import extract_text, fetch

STOP = {"the", "of", "for", "and", "in", "at", "to", "a", "scholarship", "scholarships",
        "program", "programme", "international", "students", "student", "award", "awards"}


def shingles(text, n=5):
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def overlap(dataset_text, live_text):
    """Share of the dataset text's 5-word sequences that are still on the live page."""
    a, b = shingles(dataset_text), shingles(live_text)
    return len(a & b) / len(a) if a else 0.0


def name_found(name, live_text):
    toks = [t for t in re.findall(r"[a-z]{3,}", str(name).lower()) if t not in STOP]
    if not toks:
        return True
    low = live_text.lower()
    return sum(t in low for t in toks) / len(toks) >= 0.6


def latest_year(text):
    years = [int(y) for y in re.findall(r"\b(20[12]\d)\b", text)]
    return max(years) if years else None


def is_redirect_to_generic(url, final_url):
    a, b = urlparse(url), urlparse(final_url)
    if a.netloc.removeprefix("www.") != b.netloc.removeprefix("www."):
        return b.path.strip("/") == ""
    return len(b.path.strip("/")) < max(3, len(a.path.strip("/")) // 3)


def verdict(row):
    err = row["fetch_error"]
    if err and err.startswith("http_403"):
        return "blocked"
    if err == "pdf":
        return "blocked"
    if err:
        return "dead"
    if row["live_chars"] < 300:
        return "blocked"          # JS-rendered or cookie wall -> check in a browser
    if row["redirected"] and row["overlap"] < 0.2:
        return "redirected"
    if row["overlap"] >= 0.6:
        return "same"
    if row["overlap"] >= 0.2 or (row["name_on_page"] and row["overlap"] >= 0.08):
        return "changed"
    return "different"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="corpus/corpus_clean.csv")
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out)
    cache = out / "verify_cache"
    live_dir = out / "live_texts"
    cache.mkdir(parents=True, exist_ok=True)
    live_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"Checking {len(df)} pages...")
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, u, cache): u for u in df["url"]}
        for k, f in enumerate(as_completed(futs), 1):
            results[futs[f]] = f.result()
            if k % 25 == 0:
                print(f"  {k}/{len(df)}")

    rows = []
    for _, r in df.iterrows():
        res = results[r["url"]]
        live = extract_text(res["html"], r["url"]) if res["html"] else ""
        if live:
            (live_dir / f"{r['id']}.txt").write_text(live, encoding="utf-8")
        row = {
            "id": r["id"], "name": r["name"], "url": r["url"],
            "final_url": res["final_url"], "http_status": res["http_status"],
            "fetch_error": res["error"] or "",
            "live_chars": len(live),
            "overlap": round(overlap(r["text"], live), 3) if live else 0.0,
            "name_on_page": name_found(r["name"], live) if live else False,
            "redirected": is_redirect_to_generic(r["url"], res["final_url"]),
            "latest_year_dataset": latest_year(str(r["text"])),
            "latest_year_live": latest_year(live) if live else None,
        }
        row["verdict"] = verdict(row)
        rows.append(row)

    v = pd.DataFrame(rows)
    v.to_csv(out / "verification.csv", index=False, encoding="utf-8-sig")
    same = v[v["verdict"] == "same"]
    same.sample(n=min(30, len(same)), random_state=1).to_csv(
        out / "verification_sample.csv", index=False, encoding="utf-8-sig")

    print("\nVerdicts:")
    print(v["verdict"].value_counts().to_string())
    print(f"\nPages alive and matching: {len(same)}/{len(v)} ({len(same)/len(v):.0%})")
    print("Latest year on live pages:",
          v["latest_year_live"].dropna().astype(int).value_counts().sort_index().tail(5).to_dict())


if __name__ == "__main__":
    main()
