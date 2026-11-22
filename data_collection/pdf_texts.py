import hashlib
import io
import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from scrape_scholarships import ELIG_RE, HEADERS, fetch

CACHE = Path("corpus/verify_cache")
OUT = Path("corpus/corpus/pdf_texts")
OUT.mkdir(parents=True, exist_ok=True)
PDF_HINT = re.compile(r"eligib|criteria|requirement|guideline|info|call|application|regulation|"
                      r"rules|merkblatt|ausschreibung|richtlinie|bewerbung", re.I)


def is_pdf(url):
    return ".pdf" in url.lower().split("?")[0]


def pdf_links(html, base, limit=3):
    """PDF links on the page; ones whose link text/URL looks like eligibility info come first."""
    soup = BeautifulSoup(html, "html.parser")
    found = {}
    for a in soup.find_all("a", href=True):
        href = urljoin(base, a["href"])
        if is_pdf(href):
            label = a.get_text(" ", strip=True) + " " + href
            found[href] = max(found.get(href, 0), 1 if PDF_HINT.search(label) else 0)
    return [h for h, _ in sorted(found.items(), key=lambda x: -x[1])][:limit]

import hashlib
import io
import logging
import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from scrape_scholarships import ELIG_RE, HEADERS, fetch

logging.getLogger("pypdf").setLevel(logging.ERROR)

CACHE = Path("corpus/verify_cache")
OUT = Path("corpus/pdf_texts")
OUT.mkdir(parents=True, exist_ok=True)
REVIEW = Path("corpus/manual_review.csv")
PDF_COLS = ["pdf_url", "pdf_chars", "pdf_elig_hits", "pdf_file"]
PDF_HINT = re.compile(r"eligib|criteria|requirement|guideline|info|call|application|regulation|"
                      r"rules|merkblatt|ausschreibung|richtlinie|bewerbung", re.I)


def is_pdf(url):
    return ".pdf" in str(url).lower().split("?")[0]


def pdf_links(html, base, limit=3):
    """PDF links on the page; ones whose link text/URL looks like eligibility info come first."""
    soup = BeautifulSoup(html, "html.parser")
    found = {}
    for a in soup.find_all("a", href=True):
        try:
            href = urljoin(base, a["href"])
        except ValueError:
            continue  # broken link on the site
        if is_pdf(href):
            label = a.get_text(" ", strip=True) + " " + href
            found[href] = max(found.get(href, 0), 1 if PDF_HINT.search(label) else 0)
    return [h for h, _ in sorted(found.items(), key=lambda x: -x[1])][:limit]


def pdf_text(url):
    p = OUT / (hashlib.md5(url.encode()).hexdigest() + ".txt")
    if p.exists():
        return p.read_text(encoding="utf-8"), p
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        reader = PdfReader(io.BytesIO(r.content))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:30])
    except Exception as e:
        print(f"    pdf error {type(e).__name__}: {url}")
        text = ""
    p.write_text(text, encoding="utf-8")
    return text, p


def main():
    df = pd.read_csv(REVIEW)
    df = df.drop(columns=[c for c in PDF_COLS if c in df.columns])  # safe to re-run

    rows = []
    for k, r in enumerate(df.itertuples(), 1):
        candidates = []
        try:
            if is_pdf(r.url):
                candidates = [r.url]
            else:
                res = fetch(r.url, CACHE)
                if res["html"]:
                    candidates = pdf_links(res["html"], res["final_url"] or r.url)
        except Exception as e:
            print(f"    page error {type(e).__name__}: {r.url}")

        best = {"pdf_url": "", "pdf_chars": 0, "pdf_elig_hits": 0, "pdf_file": ""}
        for u in candidates:
            text, path = pdf_text(u)
            hits = len(ELIG_RE.findall(text))
            if hits > best["pdf_elig_hits"]:
                best = {"pdf_url": u, "pdf_chars": len(text), "pdf_elig_hits": hits,
                        "pdf_file": str(path)}
        rows.append(best)
        if k % 20 == 0:
            print(f"  {k}/{len(df)}")

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out.to_csv(REVIEW, index=False, encoding="utf-8-sig")
    print(f"PDF with eligibility found for {(out['pdf_elig_hits'] > 0).sum()} of {len(out)} rows")


if __name__ == "__main__":
    main()