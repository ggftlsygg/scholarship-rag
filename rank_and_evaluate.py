"""Ranking + evaluation of all methods on the 50 students x 193 scholarships ground truth.

Methods
  BM25                          lexical baseline
  Semantic RAG (Paper 1)        dense retrieval (bge, chunk MaxP), no eligibility
  TF-IDF + SVM (Paper 2)        linear SVM on query-document term-match features, trained on dev students
  Ours (<criteria>)             Stage A hard filter + Rank = alpha*Sim + (1-alpha)*EligScore
                                criteria = haiku (extracted), regex (extracted), oracle (gold annotations)
  Ablation: alpha=1             Stage A filter + semantic similarity only (no EligScore)
  Ablation: no Stage A          alpha-fusion without the hard filter
Protocol
  Students are split 25/25 into dev and test (stratified by target degree, fixed seed).
  alpha and the SVM are fitted on dev; the main table is on test. A table over all 50 is written too.
Usage
  python rank_and_evaluate.py                       # bge embeddings (downloads the model once)
  python rank_and_evaluate.py --embedder lsa        # offline fallback, no download
Outputs: results_test.csv, results_all.csv, alpha_dev.csv, alpha_curve.png, chosen_alpha.json,
         rankings_test.csv (top-10 lists), per_student_test.csv, significance_test.csv
"""
import argparse, json
import numpy as np
import pandas as pd
import stage_a
from retrieval import BM25Index, DenseIndex, doc_text

def minmax_rows(M):
    lo, hi = M.min(axis=1, keepdims=True), M.max(axis=1, keepdims=True)
    return (M - lo) / np.where(hi - lo == 0, 1, hi - lo)

def split_students(prof, seed):
    rng = np.random.default_rng(seed)
    dev, test = [], []
    for _, grp in prof.groupby("target_degree"):
        idx = list(grp.index); rng.shuffle(idx)
        for j, i in enumerate(idx): (dev if j % 2 == 0 else test).append(i)
    return sorted(dev), sorted(test)

def rank_lists(scores, mask, rows, k):
    out = {}
    for s in rows:
        valid = np.arange(scores.shape[1]) if mask is None else np.where(mask[s])[0]
        order = valid[np.lexsort((valid, -scores[s, valid]))]
        out[s] = order[:k]
    return out

def metrics_for(lists, rel, inelig, rows, k_list=(5, 10)):
    res = {f"{m}@{k}": [] for k in k_list for m in ("P", "R", "NDCG")}
    res.update({f"Inelig@{k}": [] for k in k_list}); res["returned@10"] = []
    for s in rows:
        L = lists[s]; r = rel[s]
        n_rel = (r > 0).sum()
        ideal = np.sort(r)[::-1]
        for k in k_list:
            top = L[:k]; g = r[top]
            res[f"P@{k}"].append((g > 0).sum() / k)
            res[f"R@{k}"].append((g > 0).sum() / n_rel if n_rel else 0.0)
            dcg = ((2.0 ** g - 1) / np.log2(np.arange(2, len(top) + 2))).sum()
            idcg = ((2.0 ** ideal[:k] - 1) / np.log2(np.arange(2, k + 2))).sum()
            res[f"NDCG@{k}"].append(dcg / idcg if idcg else 0.0)
            res[f"Inelig@{k}"].append(inelig[s, top].mean() if len(top) else 0.0)
        res["returned@10"].append(len(L[:10]))
    return res

def false_exclusion(mask, inelig, rows):
    if mask is None: return 0.0
    ok = ~inelig[rows]
    return float((~mask[rows] & ok).sum() / ok.sum())

def svm_scores(corpus, prof, rel, dev_rows):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from scipy import sparse
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2)
    D = vec.fit_transform([doc_text(r) for _, r in corpus.iterrows()])
    Q = vec.transform(prof["query"])
    ns, nd = Q.shape[0], D.shape[0]
    feats = []
    for s in range(ns):
        had = D.multiply(Q[s])                                   # term-match features (query term x doc weight)
        cos = np.asarray(had.sum(axis=1))                        # cosine (both rows are L2-normalised)
        feats.append(sparse.hstack([had, sparse.csr_matrix(cos)]).tocsr())
    X = sparse.vstack(feats).tocsr()
    y = (rel.reshape(-1) > 0).astype(int)
    train = np.concatenate([np.arange(s * nd, (s + 1) * nd) for s in dev_rows])
    clf = LinearSVC(C=0.5, class_weight="balanced", max_iter=20000, random_state=0).fit(X[train], y[train])
    return clf.decision_function(X).reshape(ns, nd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="gold_annotations_ok193.csv")
    ap.add_argument("--profiles", default="student_profiles.csv")
    ap.add_argument("--qrels", default="qrels.csv")
    ap.add_argument("--gt", default="match_matrix.csv")
    ap.add_argument("--criteria", nargs="+",
                    default=["haiku=extracted_haiku.csv", "regex=extracted_regex.csv", "oracle=gold_annotations_ok193.csv"])
    ap.add_argument("--host-file", default="gold_host_country.csv", help="host countries used for the oracle criteria")
    ap.add_argument("--embedder", default="bge", choices=["bge", "lsa"])
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--main", default="haiku", help="criteria source used for the ablations")
    a = ap.parse_args()

    corpus = pd.read_csv(a.corpus, dtype=str, keep_default_na=False)
    prof = pd.read_csv(a.profiles, dtype=str, keep_default_na=False)
    ids = list(corpus["id"]); sid = list(prof["student_id"])
    di = {d: i for i, d in enumerate(ids)}; si = {s: i for i, s in enumerate(sid)}
    rel = np.zeros((len(sid), len(ids)))
    for _, q in pd.read_csv(a.qrels).iterrows(): rel[si[q.student_id], di[q.scholarship_id]] = q.relevance
    gt = pd.read_csv(a.gt)
    inelig = np.zeros_like(rel, dtype=bool)
    for _, g in gt[gt.label_full == "ineligible"].iterrows(): inelig[si[g.student_id], di[g.scholarship_id]] = True

    queries = list(prof["query"])
    print("BM25 ..."); bm25 = BM25Index(corpus).scores(queries)
    print(f"dense ({a.embedder}) ..."); dense, _ = DenseIndex(corpus, a.embedder, a.model).scores(queries)
    sim = minmax_rows(dense)
    dev, test = split_students(prof, a.seed)
    print(f"dev {len(dev)} / test {len(test)} students")
    print("SVM ..."); svm = svm_scores(corpus, prof, rel, dev)

    host = pd.read_csv(a.host_file, dtype=str, keep_default_na=False)
    passes, elig = {}, {}
    for spec in a.criteria:
        name, path = spec.split("=", 1)
        crit = pd.read_csv(path, dtype=str, keep_default_na=False)
        sa = stage_a.run(crit, prof, host if name == "oracle" else None)
        P = np.zeros_like(rel, dtype=bool); E = np.zeros_like(rel)
        for _, r in sa.iterrows():
            P[si[r.student_id], di[r.scholarship_id]] = bool(r.passes_filter)
            E[si[r.student_id], di[r.scholarship_id]] = r.elig_score
        passes[name], elig[name] = P, E

    # ---- tune alpha on dev
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    variants = {f"Ours ({n})": (n, True) for n in passes} | {"Ablation: no Stage A": (a.main, False)}
    alpha_rows, chosen = [], {}
    for vname, (cname, use_filter) in variants.items():
        best = (-1, None)
        for al in grid:
            S = al * sim + (1 - al) * elig[cname]
            lists = rank_lists(S, passes[cname] if use_filter else None, dev, 10)
            nd = np.mean(metrics_for(lists, rel, inelig, dev)["NDCG@10"])
            alpha_rows.append({"variant": vname, "alpha": al, "dev_NDCG@10": round(nd, 4)})
            if nd >= best[0]: best = (nd, al)
        chosen[vname] = float(best[1])
    pd.DataFrame(alpha_rows).to_csv("alpha_dev.csv", index=False)
    json.dump(chosen, open("chosen_alpha.json", "w"), indent=1)
    print("chosen alpha:", chosen)

    # ---- methods
    methods = {"BM25": (bm25, None), "Semantic RAG (Paper 1)": (dense, None), "TF-IDF + SVM (Paper 2)": (svm, None)}
    for vname, (cname, use_filter) in variants.items():
        al = chosen[vname]
        methods[f"{vname} a={al:g}"] = (al * sim + (1 - al) * elig[cname], passes[cname] if use_filter else None)
    methods["Ablation: alpha=1 (filter + Sim)"] = (sim, passes[a.main])

    def table(rows):
        out, per = [], {}
        for m, (S, M) in methods.items():
            lists = rank_lists(S, M, rows, 10)
            met = metrics_for(lists, rel, inelig, rows)
            per[m] = met
            out.append({"method": m, **{k: round(float(np.mean(v)), 4) for k, v in met.items()},
                        "FalseExclusion": round(false_exclusion(M, inelig, rows), 4)})
        return pd.DataFrame(out), per

    res_test, per_test = table(test)
    res_all, _ = table(list(range(len(sid))))
    res_test.to_csv("results_test.csv", index=False); res_all.to_csv("results_all.csv", index=False)

    # per-student NDCG@10 on test + paired Wilcoxon tests of the main system against each other method
    ps = pd.DataFrame({m: per_test[m]["NDCG@10"] for m in methods}, index=[sid[s] for s in test])
    ps.to_csv("per_student_test.csv")
    main_name = next(m for m in methods if m.startswith(f"Ours ({a.main})"))
    try:
        from scipy.stats import wilcoxon
        sig = []
        for m in methods:
            if m == main_name: continue
            d = ps[main_name] - ps[m]
            p = wilcoxon(ps[main_name], ps[m]).pvalue if (d != 0).any() else 1.0
            sig.append({"compared_with": m, "mean_diff_NDCG@10": round(d.mean(), 4), "wilcoxon_p": round(p, 4)})
        pd.DataFrame(sig).to_csv("significance_test.csv", index=False)
    except ImportError:
        pass

    # top-10 lists for inspection
    rk = []
    for m, (S, M) in methods.items():
        for s, L in rank_lists(S, M, test, 10).items():
            for r, d in enumerate(L, 1):
                rk.append({"student_id": sid[s], "method": m, "rank": r, "scholarship_id": ids[d],
                           "name": corpus.at[d, "name"], "score": round(float(S[s, d]), 4),
                           "relevance": int(rel[s, d]), "ineligible": bool(inelig[s, d])})
    pd.DataFrame(rk).to_csv("rankings_test.csv", index=False)

    plot_alpha(pd.DataFrame(alpha_rows), chosen)
    pd.set_option("display.width", 250)
    cols = ["method", "P@5", "P@10", "R@10", "NDCG@5", "NDCG@10", "Inelig@10", "FalseExclusion", "returned@10"]
    print("\nTEST students"); print(res_test[cols].to_string(index=False))
    print("\nALL 50 students"); print(res_all[cols].to_string(index=False))

def plot_alpha(df, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]       # fixed categorical order
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    for c, (v, g) in zip(colors, df.groupby("variant", sort=False)):
        ax.plot(g.alpha, g["dev_NDCG@10"], color=c, lw=2, marker="o", ms=4, label=v)
        al = chosen[v]; y = g.loc[g.alpha == al, "dev_NDCG@10"].iloc[0]
        ax.plot([al], [y], marker="o", ms=8, color=c, markeredgecolor="#fcfcfb", markeredgewidth=2)
        ax.annotate(v, (1.0, g["dev_NDCG@10"].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color="#52514e")
    ax.set_xlabel("alpha (weight of semantic similarity)", color="#52514e")
    ax.set_ylabel("NDCG@10 (dev students)", color="#52514e")
    ax.set_title("Choosing alpha on the dev split (larger dot = chosen)", fontsize=10, color="#0b0b0b", loc="left")
    ax.grid(color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#bdbcb6")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_xlim(0, 1.0)
    fig.tight_layout(); fig.subplots_adjust(right=0.72, bottom=0.27)
    fig.savefig("alpha_curve.png", facecolor=fig.get_facecolor())
    plt.close(fig)

if __name__ == "__main__":
    main()
