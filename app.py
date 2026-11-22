"""Web interface (Streamlit) for the eligibility-aware scholarship RAG.

Run locally:   pip install -r requirements.txt
               streamlit run app.py          -> opens http://localhost:8501 in the browser
API key:       the ANTHROPIC_API_KEY environment variable, Streamlit secrets, or the box in the sidebar.
               Without a key the ranked list still works; only the written answer needs it.

Tabs: Find scholarships | Add a scholarship | About the project
Uses the same modules as ask.py / add_scholarship.py, so results are identical to the command line.
"""
import datetime, json, os, re
import numpy as np
import pandas as pd
import streamlit as st
import stage_a
from criteria_common import COUNTRIES
from retrieval import DenseIndex
from generate_answers import SYSTEM, build_items, build_context, profile_line
from ask import fix_words, corpus_vocab
import add_scholarship as A

st.set_page_config(page_title="Scholarship finder", page_icon="🎓", layout="wide")
CHECKS = [("degree_levels", "m_degree", "Degree"), ("nationality_rule", "m_nationality", "Nationality"),
          ("min_gpa", "m_gpa", "GPA"), ("enrollment_status", "m_enrollment", "Enrollment"), ("deadline", "m_deadline", "Deadline")]
ICON = {"MATCH": "✅", "UNKNOWN": "❔", "NO MATCH": "❌"}

# ------------------------------------------------------------------ data and index (cached between clicks)
def files_key():
    """Changes whenever a scholarship is added or removed, so caches refresh."""
    return "|".join(f"{p}:{os.path.getmtime(p) if os.path.exists(p) else 0}"
                    for p in (A.GOLD_CORPUS, A.GOLD_CRIT, A.EXTRA_CORPUS, A.EXTRA_CRIT))

@st.cache_data
def load_data(use_extra, key):
    corpus = pd.read_csv(A.GOLD_CORPUS, dtype=str, keep_default_na=False)
    crit = pd.read_csv(A.GOLD_CRIT, dtype=str, keep_default_na=False).set_index("id", drop=False)
    extra_ids = []
    if use_extra and os.path.exists(A.EXTRA_CORPUS) and os.path.exists(A.EXTRA_CRIT):
        ec = pd.read_csv(A.EXTRA_CORPUS, dtype=str, keep_default_na=False)
        ex = pd.read_csv(A.EXTRA_CRIT, dtype=str, keep_default_na=False).set_index("id", drop=False)
        corpus = pd.concat([corpus, ec], ignore_index=True).fillna("")
        crit = pd.concat([crit, ex]).fillna("")
        extra_ids = list(ec["id"])
    profiles = pd.read_csv("student_profiles.csv", dtype=str, keep_default_na=False).set_index("student_id", drop=False)
    return corpus, crit, extra_ids, profiles

@st.cache_resource(show_spinner="Preparing the search index (the first time takes about a minute)…")
def get_index(key, embedder, _corpus):
    try:
        return DenseIndex(_corpus, embedder), embedder
    except Exception as e:                                         # no internet for the bge download, etc.
        st.warning(f"Could not load the '{embedder}' model ({type(e).__name__}); using the offline LSA fallback.")
        return DenseIndex(_corpus, "lsa"), "lsa"

@st.cache_data
def vocab(key, _corpus):
    return corpus_vocab(_corpus)

@st.cache_data(show_spinner="Haiku is writing the answer…")
def llm_answer(ctx, model, _api_key):
    import anthropic
    resp = anthropic.Anthropic(api_key=_api_key, max_retries=6).messages.create(
        model=model, max_tokens=6000, system=SYSTEM, messages=[{"role": "user", "content": ctx}])
    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
    u = resp.usage
    return text, u.input_tokens / 1e6 * 0.10 + u.output_tokens / 1e6 * 0.50

def api_key_from_settings():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        try: key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception: key = ""
    return key

# ------------------------------------------------------------------ the pipeline (same steps as ask.py)
def run_query(s, query, corpus, crit, index, alpha, top, date):
    stage_a.TODAY = date
    critc = crit.reindex(corpus["id"]).reset_index(drop=True)
    match = [stage_a.match_pair(critc.iloc[d], s) for d in range(len(corpus))]
    valid = [d for d in range(len(corpus)) if match[d]["passes_filter"]]
    why = {label: sum(1 for m in match if m[col] == 0) for _, col, label in CHECKS}
    D, B = index.scores([query])
    sim = (D[0] - D[0].min()) / max(D[0].max() - D[0].min(), 1e-9)
    score = {d: alpha * sim[d] + (1 - alpha) * match[d]["elig_score"] for d in valid}
    ranked = sorted(valid, key=lambda d: (-score[d], d))[:top]
    items = build_items(ranked, crit, corpus, index.chunks, B[0], match)
    for it, d in zip(items, ranked): it["score"] = score[d]
    ctx = build_context(s, query, items, len(valid), len(corpus), date)
    return {"n_pass": len(valid), "n_total": len(corpus), "why": why, "items": items, "ctx": ctx,
            "profile": profile_line(s), "query": query}

def link_citations(answer, items):
    """[S136] -> clickable link to the scholarship page."""
    urls = {it["id"]: it["url"] for it in items}
    return re.sub(r"\[([SX]\d{3})\]", lambda m: f"[[{m.group(1)}]]({urls[m.group(1)]})" if urls.get(m.group(1))
                  else m.group(0), answer)

# ------------------------------------------------------------------ sidebar: settings
with st.sidebar:
    st.header("Settings")
    use_extra = st.checkbox("Include added scholarships", value=True, help="Off = only the 193 verified ones, as in the report")
    top = st.slider("How many scholarships to show", 3, 10, 5)
    try: alpha_default = json.load(open("chosen_alpha.json"))["Ours (haiku)"]
    except (OSError, KeyError): alpha_default = 0.85
    alpha = st.slider("α: weight of meaning vs. confirmed criteria", 0.0, 1.0, float(alpha_default), 0.05,
                      help="Rank = α·Sim + (1−α)·EligScore. 0.85 was tuned on dev students.")
    date = st.date_input("Date for deadline checks", datetime.date.today()).isoformat()
    embedder = st.selectbox("Embedding model", ["bge", "lsa"], help="bge = the model from the report; lsa = offline fallback")
    api_key = api_key_from_settings()
    if not api_key:
        api_key = st.text_input("Anthropic API key (for the written answer)", type="password",
                                help="Kept only in this browser session. Without it you still get the ranked list.")
    else:
        st.caption("API key found ✓")
    model = "claude-haiku-5-5"

key = files_key()
corpus, crit, extra_ids, profiles = load_data(use_extra, key)
index, used_embedder = get_index(f"{key}|{use_extra}", embedder, corpus)

st.title("🎓 Eligibility-aware scholarship finder")
st.caption(f"{len(corpus)} scholarships ({len(corpus) - len(extra_ids)} verified"
           + (f" + {len(extra_ids)} added" if extra_ids else "") + "). First we remove what you cannot apply for, "
           "then rank the rest by meaning, then write an answer with sources.")
tab_find, tab_add, tab_about = st.tabs(["🔎 Find scholarships", "➕ Add a scholarship", "ℹ️ About the project"])

# ------------------------------------------------------------------ tab 1: find
with tab_find:
    who = st.radio("Profile", ["My own profile", "Example student from the evaluation"], horizontal=True)
    if who.startswith("Example"):
        sid = st.selectbox("Student", list(profiles.index),
                           format_func=lambda i: f"{i}: {profiles.at[i, 'nationality']}, {profiles.at[i, 'target_degree']}, "
                                                 f"{profiles.at[i, 'field']}")
        s = profiles.loc[sid].copy()
        st.caption(profile_line(s))
        query = st.text_area("Question", s["query"], height=80)
        go = st.button("Find scholarships", type="primary")
    else:
        with st.form("profile"):
            c1, c2, c3 = st.columns(3)
            countries = sorted(COUNTRIES)
            nat = c1.selectbox("Nationality", countries, index=countries.index("Kazakhstan"))
            res = c1.selectbox("Country you live in", countries, index=countries.index("Kazakhstan"))
            degree = c2.selectbox("Target degree", ["bachelor", "master", "phd", "postdoc"], index=1)
            status = c2.selectbox("Status", ["applicant", "admitted", "enrolled"],
                                  help="applicant = applying; admitted = has an offer; enrolled = already studying")
            mode = c2.radio("Study mode", ["full-time", "part-time"], horizontal=True)
            has_gpa = c3.checkbox("I have a GPA", value=True)
            gpa = c3.number_input("GPA (4.0 scale)", 0.0, 4.0, 3.5, 0.01)
            age = c3.number_input("Age", 15, 70, 21)
            gender = c3.selectbox("Gender", ["F", "M", "not given"])
            field = st.text_input("Field of study", "data science")
            query = st.text_area("Question", "Fully funded master's scholarships in data science abroad", height=80)
            go = st.form_submit_button("Find scholarships", type="primary")
        notes = []
        field_fixed, query_fixed = fix_words(field, vocab(key, corpus), notes), fix_words(query, vocab(key, corpus), notes)
        s = pd.Series({"student_id": "custom", "nationality": nat, "residence": res, "target_degree": degree,
                       "gpa_4": f"{gpa:g}" if has_gpa else "", "enrollment_status": "" if degree == "postdoc" else status,
                       "study_mode": mode, "age": str(age), "gender": "" if gender == "not given" else gender,
                       "field": field_fixed})
        query = query_fixed
        if go and notes: st.info("Spelling corrected: " + "; ".join(dict.fromkeys(notes)))

    if go:
        st.session_state["result"] = run_query(s, query, corpus, crit, index, alpha, top, date)
        st.session_state["answer"] = None
    r = st.session_state.get("result")
    if r:
        m1, m2, m3 = st.columns(3)
        m1.metric("Pass the eligibility filter", f"{r['n_pass']} of {r['n_total']}")
        m2.metric("Removed", r["n_total"] - r["n_pass"])
        m3.caption("Removed because of (one scholarship can fail several):<br>"
                   + "<br>".join(f"{k}: {v}" for k, v in r["why"].items() if v), unsafe_allow_html=True)
        rows = [{"rank": it["rank"], "id": it["id"] + (" *" if it["id"] in extra_ids else ""), "scholarship": it["name"],
                 "score": round(it["score"], 3), "EligScore": it["elig_score"],
                 **{label: ICON[it["criteria"][f]["result"]] for f, _, label in CHECKS}, "page": it["url"]}
                for it in r["items"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"page": st.column_config.LinkColumn("page", display_text="open")})
        st.caption("✅ criterion met · ❔ not stated on the page, check it · * added automatically, not verified by hand")

        st.subheader("Answer")
        if not r["items"]:
            st.warning("No scholarship passes the eligibility filter for this profile.")
        elif not api_key:
            st.info("Add an Anthropic API key in the sidebar to get the written answer with sources.")
        else:
            try:
                answer, cost = llm_answer(r["ctx"], model, api_key)
                st.markdown(link_citations(answer, r["items"]))
                st.caption(f"Claude Haiku 5.5 · about ${cost:.4f}")
            except Exception as e:
                st.error(f"API error: {type(e).__name__}: {e}")

        st.subheader("Details per scholarship")
        for it in r["items"]:
            with st.expander(f"{it['rank']}. {it['name']} [{it['id']}]"):
                st.markdown(f"[Open the official page]({it['url']})")
                st.table(pd.DataFrame([{"criterion": label, "requires": it["criteria"][f]["requires"],
                                        "result": ICON[it["criteria"][f]["result"]] + " " + it["criteria"][f]["result"]}
                                       for f, _, label in CHECKS]))
                if it["host_country"]: st.markdown(f"**Host country:** {it['host_country']}")
                if it["funding_type"]: st.markdown(f"**Funding:** {it['funding_type']}")
                if it["other_hard"]: st.markdown(f"**Other conditions (not checked by the filter):** {it['other_hard']}")
                if it["evidence"]: st.markdown(f"**Evidence from the page:** “{it['evidence']}”")
                st.markdown(f"**Most relevant passage:** {it['passage']}")
        with st.expander("What the model sees (the exact context sent to Haiku)"):
            st.code(r["ctx"], language=None)

# ------------------------------------------------------------------ tab 2: add
with tab_add:
    st.markdown("Added scholarships are saved apart from the 193 verified ones (ids X001, X002, …), so the report "
                "numbers never change. Criteria are extracted automatically with the same prompt as the 193.")
    how = st.radio("Source", ["Link", "Paste the page text"], horizontal=True)
    if how == "Link":
        url = st.text_input("Link to the scholarship page")
        name_in, text_in = "", ""
    else:
        url = st.text_input("Link (optional, used for the duplicate check)")
        name_in = st.text_input("Scholarship name")
        text_in = st.text_area("Page text", height=200)
    if st.button("Check the page"):
        try:
            if how == "Link":
                if not url: st.stop()
                title, text = A.fetch(url); name = title or url
            else:
                name, text = name_in or "Unnamed scholarship", text_in
            text = text[:A.MAX_CHARS]
            gold = A.load(A.GOLD_CORPUS, A.CORPUS_COLS)
            extra = A.load(A.EXTRA_CORPUS, A.CORPUS_COLS)
            known = pd.concat([gold[["id", "name", "url", "eligibility_text"]], extra[["id", "name", "url", "eligibility_text"]]],
                              ignore_index=True)
            if len(text) < 300:
                st.session_state["pending"] = None
                st.error(f"Page text too short ({len(text)} characters): the site may need JavaScript or block downloads. "
                         "Use 'Paste the page text' instead.")
            else:
                status, hits = A.check_duplicates(name, url, text, known)
                st.session_state["pending"] = {"name": name, "url": url, "text": text, "status": status, "hits": hits}
        except Exception as e:
            st.session_state["pending"] = None
            st.error(f"Could not download the page: {type(e).__name__}: {e}. Use 'Paste the page text' instead.")
    p = st.session_state.get("pending")
    if p:
        st.markdown(f"**{p['name']}** · {len(p['text'])} characters")
        with st.expander("Downloaded text"): st.text(p["text"][:3000])
        if p["status"] == "duplicate":
            sid, sname, why = p["hits"][0]
            st.error(f"Already exists: {sid} {sname} ({why}). Not added.")
        else:
            if p["status"] == "similar":
                st.warning("Looks similar to:\n" + "\n".join(f"- {i} {n} — {w}" for i, n, w in p["hits"])
                           + "\n\nAdd it only if it is a different scholarship.")
            else:
                st.success("No duplicate found.")
            if not api_key:
                st.info("An API key is needed to extract the criteria (sidebar).")
            elif st.button("Extract the criteria and add", type="primary"):
                os.environ["ANTHROPIC_API_KEY"] = api_key
                extra = A.load(A.EXTRA_CORPUS, A.CORPUS_COLS); extra_crit = A.load(A.EXTRA_CRIT, A.CRIT_COLS)
                sid = A.next_id(extra)
                row = {"id": sid, "name": p["name"], "url": p["url"], "eligibility_text": p["text"],
                       "added_on": date, "source": "web app"}
                with st.spinner("Haiku is reading the page…"):
                    try:
                        c, cost = A.extract(row, date)
                    except Exception as e:
                        st.error(f"Extraction failed: {type(e).__name__}: {e}"); st.stop()
                pd.concat([extra, pd.DataFrame([row])], ignore_index=True)[A.CORPUS_COLS].to_csv(A.EXTRA_CORPUS, index=False)
                pd.concat([extra_crit, pd.DataFrame([{"id": sid, **c}])], ignore_index=True)[A.CRIT_COLS].to_csv(A.EXTRA_CRIT, index=False)
                st.success(f"Added as {sid} (about ${cost:.4f}). It is included in searches from now on.")
                st.table(pd.DataFrame([{"field": f, "value": c[f]} for f in
                                       ("degree_levels", "nationality_rule", "min_gpa", "enrollment_status", "deadline",
                                        "host_country", "funding_type", "other_hard")]))
                for w in A.warnings_for(c): st.warning(w)
                st.session_state["pending"] = None
                st.cache_data.clear()
    extra = A.load(A.EXTRA_CORPUS, A.CORPUS_COLS)
    if len(extra):
        st.subheader(f"Added scholarships ({len(extra)})")
        st.dataframe(extra[["id", "name", "url", "added_on"]], hide_index=True, width="stretch",
                     column_config={"url": st.column_config.LinkColumn("url")})
        rm = st.multiselect("Remove", list(extra["id"]))
        if rm and st.button("Remove selected"):
            ec = A.load(A.EXTRA_CRIT, A.CRIT_COLS)
            extra[~extra["id"].isin(rm)][A.CORPUS_COLS].to_csv(A.EXTRA_CORPUS, index=False)
            ec[~ec["id"].isin(rm)][A.CRIT_COLS].to_csv(A.EXTRA_CRIT, index=False)
            st.cache_data.clear(); st.rerun()

# ------------------------------------------------------------------ tab 3: about
with tab_about:
    st.markdown("""
**How it works**
1. **Criteria extraction (once):** Claude Haiku 5.5 reads each scholarship page and fills 12 fields.
2. **Stage A, hard filter:** degree, nationality, GPA, enrollment status and deadline are checked: 1 = met, 0 = violated,
   0.5 = not stated. Any 0 removes the scholarship.
3. **Semantic search:** the question is compared with 150-word chunks of every page (bge-small-en-v1.5).
4. **Ranking:** Rank = α·Sim + (1−α)·EligScore among the scholarships that passed.
5. **Answer:** Haiku writes one paragraph per scholarship, only from the retrieved evidence, with [S###] citations.
""")
    if os.path.exists("results_test.csv"):
        st.markdown("**Results on the test students** (24 students × 193 scholarships)")
        res = pd.read_csv("results_test.csv")
        cols = [c for c in ["method", "P@10", "NDCG@10", "Inelig@10"] if c in res.columns]
        st.dataframe(res[cols], hide_index=True, width="stretch")
    if os.path.exists("alpha_curve.png"):
        st.image("alpha_curve.png", caption="NDCG@10 on the dev students for different α")
    st.markdown("**Limitations:** conditions outside the five criteria (residence, membership, a specific university) are "
                "not filtered; ranking ignores the field of study; the 50 evaluation students are synthetic.")
