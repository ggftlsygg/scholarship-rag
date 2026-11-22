"""Add new scholarships from a link: download the page, check for duplicates, extract the criteria with Haiku, save.

New scholarships go to corpus_extra.csv + extracted_extra.csv with ids X001, X002, ... The 193 verified
scholarships (gold_annotations_ok193.csv, extracted_haiku.csv) are never changed, so all report numbers stay valid.
ask.py reads both automatically (--no-extra = only the 193).

Usage:
  python add_scholarship.py https://example.org/some-scholarship        # one or more links
  python add_scholarship.py --file links.txt                            # one link per line
  python add_scholarship.py --text page.txt --url https://... --name "X Scholarship"   # site blocks downloads:
                                                                        # save the page text (or .html) and use it
  python add_scholarship.py https://... --dry-run                       # download + duplicate check, no API call
  python add_scholarship.py --list                                      # show added scholarships
  python add_scholarship.py --remove X003                               # delete an added scholarship
Options: --yes (no questions: add similar-looking pages too), --date (today by default)

Duplicate check, against the 193 and the already added ones:
  same link (ignoring http/https, www, trailing slash, tracking parameters)   -> skipped
  almost identical text (TF-IDF cosine >= 0.90)                              -> skipped
  similar text (cosine >= 0.50) or similar name                              -> shown; you decide (y/n)
Distinct scholarships of one university can share template text (up to 0.82 in our corpus), which is why
'similar' is a question and not an automatic skip.
"""
import argparse, datetime, difflib, hashlib, io, json, os, re, sys
from urllib.parse import urlsplit, parse_qsl, urlencode
import pandas as pd
from criteria_common import FIELDS

GOLD_CORPUS, GOLD_CRIT = "gold_annotations_ok193.csv", "extracted_haiku.csv"
EXTRA_CORPUS, EXTRA_CRIT = "corpus_extra.csv", "extracted_extra.csv"
CORPUS_COLS = ["id", "name", "url", "eligibility_text", "added_on", "source"]
CRIT_COLS = ["id"] + FIELDS + ["evidence_verbatim", "postprocess_fixes"]
DUP_COS, SIM_COS, SIM_NAME = 0.90, 0.50, 0.85
MAX_CHARS = 20000

# ------------------------------------------------------------------ files
def load(path, cols):
    if os.path.exists(path): return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=cols)

def next_id(extra):
    nums = [int(i[1:]) for i in extra["id"] if re.fullmatch(r"X\d+", i)]
    return f"X{(max(nums) + 1 if nums else 1):03d}"

# ------------------------------------------------------------------ download + clean
def norm_url(u):
    p = urlsplit((u or "").strip())
    host = p.netloc.lower().removeprefix("www.")
    q = urlencode([(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith(("utm_", "fbclid", "gclid"))])
    return f"{host}{p.path.rstrip('/')}" + (f"?{q}" if q else "")

def clean_title(t):
    t = re.sub(r"\s+", " ", t or "").strip()
    parts = [x for x in re.split(r"\s[|–—-]\s", t) if len(x.strip()) >= 10]
    return (parts[0] if parts else t).strip()

def html_to_text(html):
    """(title, main text). trafilatura keeps the article and drops menus/ads; BeautifulSoup is the fallback."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    og = soup.find("meta", property="og:title")
    title = (og.get("content") if og else "") or (soup.title.string if soup.title and soup.title.string else "")
    text = ""
    try:
        import trafilatura
        text = trafilatura.extract(html, include_tables=True, include_comments=False, favor_recall=True) or ""
    except ImportError:
        pass
    if len(text) < 300:                                            # fallback: all visible text minus page chrome
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]): tag.decompose()
        text = soup.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return clean_title(title), text

def fetch(url):
    import requests
    r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0 (scholarship-rag course project)"})
    r.raise_for_status()
    if "pdf" in r.headers.get("content-type", "").lower() or url.lower().endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join(pg.extract_text() or "" for pg in PdfReader(io.BytesIO(r.content)).pages)
        return clean_title(os.path.basename(urlsplit(url).path)), text
    return html_to_text(r.text)

def read_local(path):
    raw = open(path, encoding="utf-8", errors="ignore").read()
    if path.lower().endswith((".html", ".htm")) or raw.lstrip().startswith("<"): return html_to_text(raw)
    return clean_title(os.path.splitext(os.path.basename(path))[0]), raw.strip()

# ------------------------------------------------------------------ duplicates
def check_duplicates(name, url, text, known):
    """Return ('duplicate' | 'similar' | 'new', [(id, name, why), ...]) against known = gold + extra corpus."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    hits = []
    nu = norm_url(url) if url else ""
    for _, k in known.iterrows():
        if nu and norm_url(k["url"]) == nu: return "duplicate", [(k["id"], k["name"], "same link")]
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english")
    X = vec.fit_transform(list(known["eligibility_text"]) + [text])
    cos = (X[:-1] @ X[-1].T).toarray().ravel()
    order = cos.argsort()[::-1]
    if len(order) and cos[order[0]] >= DUP_COS:
        k = known.iloc[order[0]]; return "duplicate", [(k["id"], k["name"], f"same text (similarity {cos[order[0]]:.2f})")]
    for i in order[:3]:
        if cos[i] >= SIM_COS:
            k = known.iloc[i]; hits.append((k["id"], k["name"], f"similar text ({cos[i]:.2f})"))
    low = re.sub(r"\W+", " ", name.lower()).strip()
    for _, k in known.iterrows():
        r = difflib.SequenceMatcher(None, low, re.sub(r"\W+", " ", k["name"].lower()).strip()).ratio()
        if r >= SIM_NAME and k["id"] not in [h[0] for h in hits]: hits.append((k["id"], k["name"], f"similar name ({r:.2f})"))
    return ("similar" if hits else "new"), hits

# ------------------------------------------------------------------ extraction (same prompt and checks as the 193)
def extract(row, date, model="claude-haiku-5-5", cache_dir="llm_cache"):
    import anthropic
    import extract_llm as ex
    system = ex.SYSTEM.format(date=date); user = ex.user_message(row)
    h = hashlib.sha256((model + system + user).encode()).hexdigest()[:16]
    cdir = os.path.join(cache_dir, model); os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f"{row['id']}.json")
    if os.path.exists(path):
        c = json.load(open(path, encoding="utf-8"))
        if c.get("hash") == h and c.get("data") is not None: return ex.postprocess(c["data"], row["eligibility_text"], date), 0.0
    client = anthropic.Anthropic(max_retries=6)
    res = ex.call_model(client, model, system, user, "", 8000)
    if res["data"] is None or res["stop_reason"] == "max_tokens":
        res = ex.call_model(client, model, system, user, "", 16000)
    if res["data"] is None: raise RuntimeError(f"no structured output (stop_reason {res['stop_reason']})")
    res["hash"] = h
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    u = res["usage"]
    return ex.postprocess(res["data"], row["eligibility_text"], date), u["input_tokens"] / 1e6 * 0.10 + u["output_tokens"] / 1e6 * 0.50

def warnings_for(c):
    w = []
    empty = [f for f in ("degree_levels", "nationality_rule", "min_gpa", "enrollment_status", "deadline") if not c[f]]
    if empty: w.append(f"not stated on the page (will show as '?'): {', '.join(empty)}")
    if c["enrollment_status"]: w.append("check enrollment_status against the page: the extractor often guesses it")
    if c["evidence_sentence"] and not int(c["evidence_verbatim"]): w.append("evidence sentence is not a verbatim quote")
    if c["postprocess_fixes"]: w.append(f"automatic fixes: {c['postprocess_fixes']}")
    return w

# ------------------------------------------------------------------ main
def ask_yes(question, assume_yes):
    if assume_yes: return True
    if not sys.stdin.isatty(): return False
    return input(f"{question} (y/n) [n]: ").strip().lower().startswith("y")

def add_one(name, url, text, a, gold, extra, extra_crit):
    text = text[:MAX_CHARS]
    if len(text) < 300:
        print(f"  page text too short ({len(text)} characters): the site may need JavaScript or block downloads.\n"
              f"  Copy the page text into a .txt file and run: python add_scholarship.py --text page.txt --url {url or '...'}")
        return extra, extra_crit
    known = pd.concat([gold[["id", "name", "url", "eligibility_text"]], extra[["id", "name", "url", "eligibility_text"]]],
                      ignore_index=True)
    status, hits = check_duplicates(name, url, text, known)
    if status == "duplicate":
        sid, sname, why = hits[0]
        print(f"  ALREADY EXISTS: {sid} {sname} ({why}). Not added.")
        return extra, extra_crit
    if status == "similar":
        print("  Looks similar to:")
        for sid, sname, why in hits: print(f"    {sid} {sname} - {why}")
        if not ask_yes("  Is it a different scholarship? Add it", a.yes):
            print("  Not added."); return extra, extra_crit
    if len(text) >= MAX_CHARS: print(f"  long page (cut to {MAX_CHARS} characters): it may list several scholarships")
    sid = next_id(extra)
    row = {"id": sid, "name": name, "url": url or "", "eligibility_text": text,
           "added_on": a.date, "source": "file" if a.text else "link"}
    if a.dry_run:
        print(f"  would be added as {sid}: {name} ({len(text)} characters). --dry-run: no API call, nothing saved.")
        return extra, extra_crit
    if not os.environ.get("ANTHROPIC_API_KEY"): sys.exit("ANTHROPIC_API_KEY is not set.")
    try:
        c, cost = extract(row, a.date)
    except Exception as e:
        print(f"  extraction failed: {type(e).__name__}: {e}. Not added."); return extra, extra_crit
    extra = pd.concat([extra, pd.DataFrame([row])], ignore_index=True)
    extra_crit = pd.concat([extra_crit, pd.DataFrame([{"id": sid, **c}])], ignore_index=True)
    extra[CORPUS_COLS].to_csv(EXTRA_CORPUS, index=False)              # save after every page: nothing is lost on errors
    extra_crit[CRIT_COLS].to_csv(EXTRA_CRIT, index=False)
    print(f"  ADDED as {sid}: {name}  (~${cost:.4f})")
    for f, label in [("degree_levels", "degree"), ("nationality_rule", "nationality"), ("min_gpa", "min grade"),
                     ("enrollment_status", "enrollment"), ("deadline", "deadline"), ("host_country", "host country"),
                     ("funding_type", "funding"), ("other_hard", "other conditions")]:
        print(f"    {label:17}: {c[f][:110] or '-'}")
    for w in warnings_for(c): print(f"    ! {w}")
    return extra, extra_crit

def main():
    ap = argparse.ArgumentParser(description="Add scholarships from links (saved apart from the 193 verified ones).")
    ap.add_argument("urls", nargs="*")
    ap.add_argument("--file", help="text file with one link per line")
    ap.add_argument("--text", help="saved page text (.txt) or page (.html) when the site blocks downloads")
    ap.add_argument("--url", default="", help="link of the --text page"); ap.add_argument("--name", default="")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--yes", action="store_true", help="do not ask; add similar-looking pages too")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--remove", nargs="*", default=None, metavar="ID")
    a = ap.parse_args()

    gold = load(GOLD_CORPUS, CORPUS_COLS)
    extra, extra_crit = load(EXTRA_CORPUS, CORPUS_COLS), load(EXTRA_CRIT, CRIT_COLS)
    if a.list:
        if extra.empty: print("no added scholarships yet"); return
        for _, r in extra.iterrows(): print(f"{r['id']}  {r['added_on']}  {r['name'][:60]}  {r['url']}")
        return
    if a.remove is not None:
        bad = [i for i in a.remove if not i.startswith("X")]
        if bad: sys.exit(f"only added scholarships (X...) can be removed; the 193 verified ones stay: {bad}")
        extra = extra[~extra["id"].isin(a.remove)]; extra_crit = extra_crit[~extra_crit["id"].isin(a.remove)]
        extra[CORPUS_COLS].to_csv(EXTRA_CORPUS, index=False); extra_crit[CRIT_COLS].to_csv(EXTRA_CRIT, index=False)
        print(f"removed {', '.join(a.remove)}"); return

    jobs = []                                                      # (url, name, text or None)
    urls = list(a.urls) + ([l.strip() for l in open(a.file, encoding="utf-8") if l.strip() and not l.startswith("#")]
                           if a.file else [])
    jobs += [(u, "", None) for u in urls]
    if a.text:
        title, text = read_local(a.text); jobs.append((a.url, a.name or title, text))
    if not jobs: ap.print_help(); return
    for url, name, text in jobs:
        print(f"\n{url or a.text}")
        if text is None:
            try:
                title, text = fetch(url)
            except Exception as e:
                print(f"  could not download: {type(e).__name__}: {e}\n"
                      f"  Copy the page text into a .txt file and use --text page.txt --url {url}"); continue
            name = a.name or title or url
        print(f"  name: {name}")
        extra, extra_crit = add_one(name, url, text, a, gold, extra, extra_crit)
    print(f"\n{len(extra)} added scholarships in {EXTRA_CORPUS} (+ 193 verified). ask.py uses them automatically.")

if __name__ == "__main__":
    main()
