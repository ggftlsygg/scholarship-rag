import argparse
import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd

from scrape_scholarships import ELIG_RE, extract_text, fetch
from find_urls import DDGS, AGGREGATORS, domain, score_result, search, clean_query
from verify_scholarships import latest_year, name_found
from prepare_corpus import clean_text

MIN_LIVE_CHARS = 400


def elig_hits(text):
    return len(ELIG_RE.findall(text or ""))


def good_text(text, name, min_chars=MIN_LIVE_CHARS):
    return len(text) >= min_chars and elig_hits(text) >= 2 and name_found(name, text)


def find_new_page(ddgs, name, old_url, cache_dir, search_cache, delay):
    """Search by name; return (url, text, source) of the first page that really is this scholarship."""
    if name not in search_cache:
        search_cache[name] = search(ddgs, clean_query(name))
        time.sleep(delay)
    ranked = sorted((score_result(r, name) for r in search_cache[name]), reverse=True)
    tried = 0
    for score, url in ranked:
        if score <= -100 or url.rstrip("/") == str(old_url).rstrip("/"):
            continue
        tried += 1
        if tried > 4:
            break
        res = fetch(url, cache_dir)
        if res["error"] or not res["html"]:
            continue
        text = clean_text(extract_text(res["html"], url))
        ly = latest_year(text)
        if ly is not None and ly < 2024:
            continue  # страница устарела — пробуем следующую
        if good_text(text, name, min_chars=500):
            src = "new_url_aggregator" if any(a in domain(url) for a in AGGREGATORS) else "new_url_official"
            return res["final_url"] or url, text, src
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/corpus_clean.csv")
    ap.add_argument("--verification", default="corpus/verification.csv")
    ap.add_argument("--live", default="corpus/live_texts")
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--delay", type=float, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    cache_dir = out / "verify_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    search_cache_file = out / "refresh_search_cache.json"
    search_cache = (json.loads(search_cache_file.read_text(encoding="utf-8"))
                    if search_cache_file.exists() else {})

    corpus = pd.read_csv(args.corpus)
    ver = pd.read_csv(args.verification)[["id", "verdict", "name_on_page"]]
    df = corpus.merge(ver, on="id", how="left")
    today = dt.date.today().isoformat()

    kept, dropped = [], []
    to_search = []

    for _, r in df.iterrows():
        live_path = Path(args.live) / f"{r['id']}.txt"
        live = clean_text(live_path.read_text(encoding="utf-8")) if live_path.exists() else ""
        base = {"name": r["name"], "type": r.get("type", ""), "old_url": r["url"]}
        v = r["verdict"]

        if v in ("same", "changed"):
            if good_text(live, r["name"]):
                kept.append({**base, "url": r["url"], "text": live, "text_source": f"live_{v}"})
            else:  # page alive but extraction poor -> keep the (confirmed) dataset text
                kept.append({**base, "url": r["url"], "text": r["text"], "text_source": f"dataset_{v}"})
        elif v == "different" and bool(r["name_on_page"]) and good_text(live, r["name"], 800):
            kept.append({**base, "url": r["url"], "text": live, "text_source": "live_check"})
        elif v == "blocked":
            kept.append({**base, "url": r["url"], "text": r["text"],
                         "text_source": "dataset_check_in_browser"})
        else:
            to_search.append((r, base, v))

    print(f"Kept from verification: {len(kept)}; searching new pages for {len(to_search)}...")
    with DDGS() as ddgs:
        for k, (r, base, v) in enumerate(to_search, 1):
            url, text, src = find_new_page(ddgs, r["name"], r["url"], cache_dir, search_cache, args.delay)
            if url:
                kept.append({**base, "url": url, "text": text, "text_source": src})
            else:
                dropped.append({**base, "verdict": v, "reason": "no current page found"})
            if k % 20 == 0:
                print(f"  {k}/{len(to_search)}")
                search_cache_file.write_text(json.dumps(search_cache), encoding="utf-8")
    search_cache_file.write_text(json.dumps(search_cache), encoding="utf-8")

    fresh = pd.DataFrame(kept)
    fresh["latest_year"] = fresh["text"].map(latest_year)
    fresh["retrieved"] = today
    # the same new page can be found for two names -> keep one
    fresh = fresh.drop_duplicates("url")
    fresh.to_csv(out / "corpus_fresh.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dropped).to_csv(out / "refresh_dropped.csv", index=False, encoding="utf-8-sig")

    print("\nText sources:")
    print(fresh["text_source"].value_counts().to_string())
    current = fresh["text_source"].str.startswith(("live", "new_url")).sum()
    print(f"\nTotal kept: {len(fresh)}  (current live text: {current}), dropped: {len(dropped)}")
    print("Next: python prepare_corpus.py --csv corpus/corpus_fresh.csv --out corpus_v2 --target 250")


if __name__ == "__main__":
    main()
