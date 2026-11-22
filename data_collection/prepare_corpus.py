"""
Stage 1b — turn hf_scholarships.csv (AlberBshara/scholarships_dataset) into
a clean corpus + annotation template.

Run:  python prepare_corpus.py --csv hf_scholarships.csv --target 300
Needs scrape_scholarships.py in the same folder (reuses its eligibility helpers).

Outputs (folder corpus/):
  corpus_clean.csv          all usable scholarships, cleaned (id, name, url, degree_hint, text, ...)
  texts/<id>.txt            one file per scholarship (RAG corpus)
  annotation_template.csv   `target` scholarships, split between annotators A / B,
                            with an overlap set (both annotate) for Cohen's kappa
  reserve.csv               usable but not selected (replacements)
  needs_rescrape.csv        empty / too short texts -> scrape_scholarships.py --csv needs_rescrape.csv
"""

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from scrape_scholarships import GOLD_COLUMNS, ELIG_RE, eligibility_section, hints

MIN_CHARS = 200
LONG_CHARS = 8000

# phrases typical for menus / cookie banners / footers glued into page text
JUNK_LINE = re.compile(
    r"(cookie|privacy policy|accept all|skip to (main )?content|subscribe to our newsletter|"
    r"all rights reserved|follow us on|share this page|back to top)", re.I)


def clean_text(t: str) -> str:
    t = str(t or "")
    t = re.sub(r"\s+", " ", t)
    # "month.The program" -> "month. The program"; "REQUIREMENTSTo be" left alone (rare)
    t = re.sub(r"([a-z0-9\)])([.!?:;])([A-Z])", r"\1\2 \3", t)
    # drop sentences that are obvious site chrome
    sents = re.split(r"(?<=[.!?])\s+", t)
    sents = [s for s in sents if not (JUNK_LINE.search(s) and len(s) < 200)]
    return " ".join(sents).strip()


def degree_hint(type_value) -> str:
    s = str(type_value or "").lower()
    out = []
    if re.search(r"bachelor|undergrad", s):
        out.append("bachelor")
    if re.search(r"master|postgrad|llm|mba", s):
        out.append("master")
    if re.search(r"phd|doctor", s):
        out.append("phd")
    if re.search(r"post ?doc|research|fellow", s) and "phd" not in out:
        out.append("research")
    return ";".join(out)


def tld(url: str) -> str:
    host = urlparse(str(url)).netloc.lower()
    return host.rsplit(".", 1)[-1] if "." in host else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=30, help="items both annotators label (kappa)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS, help="drop texts shorter than this")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "texts").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    n0 = len(df)
    df["text"] = df["text"].fillna("").map(clean_text)
    df["n_chars"] = df["text"].str.len()
    if "type" not in df.columns:
        df["type"] = ""
    df["type"] = df["type"].fillna("").astype(str).str.strip(" :")

    # duplicates: same text, or same name -> keep the longest text
    # (not by url: several scholarships can come from one page, e.g. a list or a PDF of awards)
    df = df.sort_values("n_chars", ascending=False)
    df = df.drop_duplicates("text").drop_duplicates("name")
    n_dedup = len(df)

    short = df[df["n_chars"] < args.min_chars]
    df = df[df["n_chars"] >= args.min_chars].copy()

    df["degree_hint"] = df["type"].map(degree_hint)
    df["tld"] = df["url"].map(tld)
    df["eligibility_text"] = df["text"].map(eligibility_section)
    df["n_elig_hits"] = df["text"].map(lambda t: len(ELIG_RE.findall(t)))
    df["flag"] = ""
    df.loc[df["n_chars"] > LONG_CHARS, "flag"] = "long_check_manually"
    df.loc[df["n_elig_hits"] < 2, "flag"] = "few_eligibility_terms"
    if "text_source" in df.columns:
        chk = df["text_source"].astype(str).str.contains("check|aggregator") & (df["flag"] == "")
        df.loc[chk, "flag"] = "verify_source"

    # stable ids in original-ish order
    df = df.sort_values("name").reset_index(drop=True)
    df.insert(0, "id", [f"S{i:03d}" for i in range(1, len(df) + 1)])

    extra = [c for c in ("text_source", "review_reason", "old_url", "latest_year", "retrieved") if c in df.columns]
    keep_cols = ["id", "name", "url", "type", "degree_hint", "tld", "n_chars",
                 "n_elig_hits", "flag"] + extra + ["text", "eligibility_text"]
    df[keep_cols].to_csv(out / "corpus_clean.csv", index=False, encoding="utf-8-sig")
    for _, r in df.iterrows():
        (out / "texts" / f"{r['id']}.txt").write_text(
            f"{r['name']}\n{r['url']}\n\n{r['text']}", encoding="utf-8")

    # selection: prefer texts that actually talk about eligibility, keep TLD variety
    cand = df[df["flag"] != "few_eligibility_terms"].copy()
    cand["prio"] = cand["n_elig_hits"].clip(upper=25) - (cand["flag"] != "").astype(int) * 5
    cand = cand.sample(frac=1, random_state=args.seed).sort_values("prio", ascending=False)
    cand["tld_rank"] = cand.groupby("tld").cumcount()
    cap = max(10, args.target // 6)  # no single TLD dominates the top picks
    picked = pd.concat([cand[cand["tld_rank"] < cap], cand[cand["tld_rank"] >= cap]]).head(args.target)
    reserve = df[~df["id"].isin(picked["id"])]

    # annotation split: overlap set labelled by both, the rest alternated A / B
    picked = picked.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    picked["annotator"] = ["A+B" if i < args.overlap else ("A" if i % 2 == 0 else "B")
                           for i in range(len(picked))]
    ann_cols = ["id", "name", "url", "type", "degree_hint", "flag"] + \
        [c for c in ("text_source",) if c in picked.columns] + ["annotator", "eligibility_text"]
    ann = picked[ann_cols].copy()
    hint_df = pd.DataFrame([hints(t) for t in picked["text"]])
    ann = pd.concat([ann, hint_df], axis=1)
    for c in GOLD_COLUMNS:
        if c != "annotator":
            ann[c] = ""
    ann = ann.sort_values(["annotator", "id"])
    ann.to_csv(out / "annotation_template.csv", index=False, encoding="utf-8-sig")
    reserve[keep_cols].to_csv(out / "reserve.csv", index=False, encoding="utf-8-sig")
    short[["name", "url", "type", "n_chars"]].to_csv(out / "needs_rescrape.csv", index=False,
                                                     encoding="utf-8-sig")

    print(f"rows: {n0} -> after dedup {n_dedup} -> usable {len(df)} (short/empty: {len(short)})")
    print(f"selected for annotation: {len(picked)}, reserve: {len(reserve)}")
    print("annotator split:", picked["annotator"].value_counts().to_dict())
    print("flags in corpus:", df["flag"].replace('', 'ok').value_counts().to_dict())
    print("top TLDs selected:", picked["tld"].value_counts().head(8).to_dict())
    print("degree hints selected:", picked["degree_hint"].replace('', 'unknown').value_counts().head(6).to_dict())


if __name__ == "__main__":
    main()
