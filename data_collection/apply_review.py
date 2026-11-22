"""
Apply the manual review (Google Sheets export) to the corpus -> corpus/corpus_final.csv

Text priority for every kept scholarship:
  1. manually_extracted_text (pasted by hand)
  2. new page text, if the url in the sheet differs from the corpus url (downloaded now)
  3. PDF text, if reason says use_pdf, or the page text has almost no criteria and the PDF has them
  4. the text already in corpus_fresh.csv (live page / dataset)

Rows:
  decision = keep    -> kept, with new_name if given
  decision = reject  -> dropped (listed in apply_report.csv)
  not in the sheet   -> kept as they are (the confident rows that were not reviewed)
  in the sheet only  -> new scholarships (manual_added, list_page_extract) from manually_extracted_text

Run (project folder, next to scrape_scholarships.py):
  python apply_review.py --review manual_review.csv
  python prepare_corpus.py --csv corpus/corpus_final.csv --out corpus_v2 --target 245 --min-chars 100

Outputs:
  corpus/corpus_final.csv   name, url, type, text, text_source, review_reason, latest_year, retrieved
  corpus/apply_report.csv   what happened to every row (+ warnings to look at)
"""

import argparse
import datetime as dt
import re
from pathlib import Path

import pandas as pd

from scrape_scholarships import ELIG_RE, extract_text, fetch

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

STOP = {"the", "of", "for", "and", "in", "at", "to", "a", "an", "scholarship", "scholarships",
        "program", "programme", "international", "students", "student", "award", "awards",
        "fully", "funded", "uk", "usa", "how", "apply", "step", "process", "university"}


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def clean_text(t):
    t = str(t or "").replace("\r", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def elig_hits(t):
    return len(ELIG_RE.findall(t or ""))


def name_in_text(name, text):
    core = re.split(r"\s[|–]\s", str(name))[0]
    toks = [w for w in re.findall(r"[a-z]{3,}", core.lower()) if w not in STOP]
    if not toks:
        return True
    low = text.lower()
    return sum(w in low for w in toks) / len(toks) >= 0.5


def latest_year(t):
    ys = [int(y) for y in re.findall(r"\b(20[12]\d)\b", t or "")]
    return max(ys) if ys else ""


def read_pdf_text(pdf_file, pdf_url, base, cache_dir):
    """pdf_file is a path like corpus\\pdf_texts\\<hash>.txt (from pdf_texts.py); fall back to downloading."""
    pf = str(pdf_file or "").strip()
    if pf and not pf.lower().startswith("http"):
        p = (base / pf.replace("\\", "/")).resolve()
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    url = pf if pf.lower().startswith("http") else str(pdf_url or "").strip()
    if not url or PdfReader is None:
        return ""
    import io
    import requests
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if not r.ok:
            return ""
        reader = PdfReader(io.BytesIO(r.content))
        return "\n".join((pg.extract_text() or "") for pg in reader.pages)
    except Exception:  # noqa: BLE001
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True, help="manual review exported from Google Sheets (CSV)")
    ap.add_argument("--corpus", default="corpus/corpus_fresh.csv")
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--base", default=".", help="folder the pdf_file paths are relative to")
    args = ap.parse_args()

    out = Path(args.out)
    cache_dir = out / "verify_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.base)
    today = dt.date.today().isoformat()

    corpus = pd.read_csv(args.corpus, dtype=str, keep_default_na=False)
    rev = pd.read_csv(args.review, dtype=str, keep_default_na=False)
    for c in ("new_name", "manually_extracted_text", "pdf_file", "pdf_url", "reason", "decision"):
        if c not in rev.columns:
            rev[c] = ""
    rev["decision"] = rev["decision"].str.strip().str.lower().replace({"rejected": "reject", "kept": "keep"})
    bad = rev[~rev["decision"].isin(["keep", "reject"])]
    if len(bad):
        print(f"WARNING: {len(bad)} sheet rows without keep/reject -> treated as not reviewed:")
        print("  " + "\n  ".join(bad["name"].head(10)))

    corpus["_n"] = corpus["name"].map(norm)
    corpus["_u"] = corpus["url"].map(norm)
    rev["_n"] = rev["name"].map(norm)
    rev["_u"] = rev["url"].map(norm)

    # match sheet rows to corpus rows: name+url, then name alone (url replaced), then url alone (name edited)
    used, match = set(), {}
    for rule in ("both", "name", "url"):
        for i, r in rev.iterrows():
            if i in match:
                continue
            free = ~corpus.index.isin(used)
            if rule == "both":
                cand = corpus[(corpus["_n"] == r["_n"]) & (corpus["_u"] == r["_u"]) & free]
            elif rule == "name":
                cand = corpus[(corpus["_n"] == r["_n"]) & free]
            else:
                cand = corpus[(corpus["_u"] == r["_u"]) & free] if r["_u"] else corpus.iloc[0:0]
            if len(cand) == 1 or (rule != "url" and len(cand)):
                match[i] = cand.index[0]
                used.add(cand.index[0])

    final, report = [], []

    def add(name, url, typ, text, source, reason, note=""):
        text = clean_text(text)
        warn = []
        if len(text) < 200:
            warn.append("short text")
        if elig_hits(text) < 2:
            warn.append("few criteria")
        if not name_in_text(name, text):
            warn.append("name not found in text")
        final.append({"name": name, "url": url, "type": typ, "text": text, "text_source": source,
                      "review_reason": reason, "latest_year": latest_year(text), "retrieved": today})
        report.append({"name": name, "action": "keep", "text_source": source, "chars": len(text),
                       "note": note, "warnings": "; ".join(warn)})

    # 1) reviewed rows
    for i, r in rev.iterrows():
        dec = r["decision"]
        ci = match.get(i)
        c = corpus.loc[ci] if ci is not None else None
        if dec not in ("keep", "reject"):
            continue  # handled with the unreviewed rows below (if it matched the corpus)
        if dec == "reject":
            report.append({"name": r["name"], "action": "reject", "text_source": r.get("text_source", ""),
                           "chars": "", "note": r["reason"], "warnings": ""})
            continue

        name = norm(r["new_name"]) or norm(r["name"])
        url = norm(r["url"]) or (c["url"] if c is not None else "")
        typ = c["type"] if c is not None and "type" in c else ""
        reason = r["reason"]
        manual = clean_text(r["manually_extracted_text"])
        corpus_text = c["text"] if c is not None else ""
        orig_source = c["text_source"] if c is not None else r.get("text_source", "manual")
        url_changed = (c is not None and norm(url) != norm(c["url"])) or (c is None and bool(url))

        if manual:
            add(name, url, typ, manual, "manual_text", reason)
            continue

        page_text, src, note = corpus_text, orig_source, ""
        if url_changed:
            res = fetch(url, cache_dir)
            new_txt = clean_text(extract_text(res["html"], url)) if res.get("html") else ""
            if len(new_txt) >= 300:
                page_text, src, note = new_txt, "new_url_download", "url replaced in review"
            else:
                note = f"new url could not be read ({res.get('error') or 'empty'}) -> kept old text"

        wants_pdf = "pdf" in reason.lower() and "use_site" not in reason.lower()
        weak_page = elig_hits(page_text) < 3
        if (wants_pdf or weak_page) and (r["pdf_file"] or r["pdf_url"]):
            pdf_txt = clean_text(read_pdf_text(r["pdf_file"], r["pdf_url"], base, cache_dir))
            if pdf_txt and (wants_pdf or elig_hits(pdf_txt) > elig_hits(page_text)):
                page_text, src = pdf_txt, "pdf"
                note = (note + "; " if note else "") + ("use_pdf" if wants_pdf else "page had no criteria, used PDF")
            elif wants_pdf:
                note = (note + "; " if note else "") + "PDF text not found -> kept page text"

        if not page_text:
            report.append({"name": name, "action": "drop", "text_source": src, "chars": 0,
                           "note": "keep but no text available", "warnings": "add manually_extracted_text"})
            continue
        add(name, url, typ, page_text, src, reason, note)

    # 2) corpus rows that were not reviewed (or reviewed without a decision)
    #    rows of the "doubtful" groups had to be in the sheet; if one is missing it was probably
    #    renamed AND given a new url there -> drop it instead of keeping an unchecked text
    MUST_REVIEW = ("live_check", "dataset_check_in_browser", "new_url_official", "new_url_aggregator")
    reviewed_ci = {ci for i, ci in match.items() if rev.at[i, "decision"] in ("keep", "reject")}
    for ci, c in corpus.iterrows():
        if ci in reviewed_ci:
            continue
        if c["text_source"] in MUST_REVIEW:
            report.append({"name": c["name"], "action": "drop", "text_source": c["text_source"], "chars": len(c["text"]),
                           "note": "doubtful row not found in the review sheet", "warnings": "check: renamed in sheet?"})
            continue
        add(c["name"], c["url"], c.get("type", ""), c["text"], c["text_source"], "not_reviewed")

    fin = pd.DataFrame(final)
    before = len(fin)
    fin = fin.drop_duplicates(subset=["name"]).drop_duplicates(subset=["text"])
    rep = pd.DataFrame(report)
    fin.to_csv(out / "corpus_final.csv", index=False, encoding="utf-8-sig")
    rep.to_csv(out / "apply_report.csv", index=False, encoding="utf-8-sig")

    print(f"\nFinal corpus: {len(fin)} scholarships (duplicates removed: {before - len(fin)})")
    print("Actions:", rep["action"].value_counts().to_dict())
    print("Text sources:", fin["text_source"].value_counts().to_dict())
    w = rep[rep["warnings"] != ""]
    print(f"Rows with warnings: {len(w)} (see apply_report.csv)")
    print(w["warnings"].str.split("; ").explode().value_counts().to_dict())
    print(f"\nNext: python prepare_corpus.py --csv {out}/corpus_final.csv --out corpus_v2 --target 245 --min-chars 100")


if __name__ == "__main__":
    main()
