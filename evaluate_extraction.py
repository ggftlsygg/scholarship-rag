"""Evaluate criteria extractors against the gold annotations, and their effect on the Stage A filter.

Usage:
  python evaluate_extraction.py --pred regex=extracted_regex.csv haiku=extracted_haiku.csv
Outputs:
  extraction_metrics.csv  per extractor x field: detection P/R, hallucination rate, exact accuracy, label F1
  filter_metrics.csv      per extractor (+ oracle = gold criteria): filter decisions vs ground truth labels
  extraction_errors.csv   every gold/pred disagreement on core fields (for error analysis in the report)
"""
import argparse, re
import pandas as pd
from criteria_common import CORE, split_multi, expand_countries, gpa_to_4, canon_country
import stage_a

def nat_cat(v):
    v = (v or "").strip()
    if not v: return ""
    return v.split(":")[0].strip().upper()

def nat_equal(g, p):
    cg, cp = nat_cat(g), nat_cat(p)
    if cg != cp: return False
    if cg in ("ONLY", "EXCLUDE"):
        eg = expand_countries(g.split(":", 1)[1]); ep = expand_countries(p.split(":", 1)[1])
        return (eg[0] | eg[1]) == (ep[0] | ep[1])
    return True

def label_set(field, v):
    s = set(split_multi(v))
    if field == "enrollment_status" and {"full-time", "part-time"} <= s: s -= {"full-time", "part-time"}
    if field == "funding_type" and len(s) > 1: s.discard("travel")
    return s

def gpa_equal(g, gs, p, ps):
    if not g and not p: return True
    if not g or not p: return False
    if g.strip().lower() == p.strip().lower(): return True
    a, b = gpa_to_4(g, gs), gpa_to_4(p, ps)
    return bool(a and b and abs(a[0] - b[0]) < 0.06)

def field_equal(field, g, p, gs="", ps=""):
    if field == "nationality_rule": return nat_equal(g, p)
    if field == "min_gpa": return gpa_equal(g, gs, p, ps)
    if field in ("degree_levels", "enrollment_status", "deadline", "funding_type"):
        return label_set(field, g) == label_set(field, p)
    return (g or "").strip().lower() == (p or "").strip().lower()

def evaluate_fields(name, gold, pred, hosts):
    m = gold.merge(pred, on="id", suffixes=("_g", "_p"), how="left").fillna("")
    rows, errs = [], []
    for f in CORE:
        g, p = m[f + "_g"], m[f + "_p"]
        gn, pn = g != "", p != ""
        tp = (gn & pn).sum()
        eq = [field_equal(f, a, b, m.at[i, "gpa_scale_g"], m.at[i, "gpa_scale_p"]) for i, (a, b) in enumerate(zip(g, p))]
        r = {"extractor": name, "field": f,
             "gold_stated": int(gn.sum()), "pred_stated": int(pn.sum()),
             "detect_precision": round(tp / pn.sum(), 3) if pn.sum() else None,
             "detect_recall": round(tp / gn.sum(), 3) if gn.sum() else None,
             "hallucination_rate": round((pn & ~gn).sum() / (~gn).sum(), 3) if (~gn).sum() else None,
             "exact_accuracy": round(sum(eq) / len(eq), 3),
             "accuracy_when_stated": round(sum(e for e, s in zip(eq, gn) if s) / gn.sum(), 3) if gn.sum() else None}
        if f in ("degree_levels", "enrollment_status", "deadline"):
            TP = FP = FN = 0
            for a, b in zip(g, p):
                sa, sb = label_set(f, a), label_set(f, b)
                TP += len(sa & sb); FP += len(sb - sa); FN += len(sa - sb)
            pr = TP / (TP + FP) if TP + FP else 0; rc = TP / (TP + FN) if TP + FN else 0
            r.update({"label_precision": round(pr, 3), "label_recall": round(rc, 3),
                      "label_f1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0})
        if f == "nationality_rule":
            r["category_accuracy"] = round(sum(nat_cat(a) == nat_cat(b) for a, b in zip(g, p)) / len(g), 3)
        rows.append(r)
        for i, ok in enumerate(eq):
            if not ok: errs.append({"extractor": name, "id": m.at[i, "id"], "field": f,
                                    "gold": m.at[i, f + "_g"], "pred": m.at[i, f + "_p"]})
    # host country, only where it matters (INTERNATIONAL rules) and is known
    hc = "host_country_p" if "host_country_p" in m.columns else ("host_country" if "host_country" in m.columns else None)
    if hc:
        sub = m[m["id"].isin(hosts)]
        ok = [canon_country(p or "") == hosts[i] for i, p in zip(sub["id"], sub[hc])]
        rows.append({"extractor": name, "field": "host_country (INTERNATIONAL rows)", "gold_stated": len(sub),
                     "pred_stated": int((sub[hc] != "").sum()),
                     "exact_accuracy": round(sum(ok) / len(ok), 3) if ok else None})
    return rows, errs

def evaluate_filter(name, crit, prof, gt, host_file):
    sa = stage_a.run(crit, prof, host_file)
    m = sa.merge(gt[["student_id", "scholarship_id", "label_core", "label_full"]], on=["student_id", "scholarship_id"])
    out = {"criteria": name}
    for ref in ["label_full", "label_core"]:
        bad = m[ref] == "ineligible"
        excluded = m["passes_filter"] == 0
        out[f"{ref}: false_exclusion_rate"] = round((excluded & ~bad).sum() / (~bad).sum(), 4)
        out[f"{ref}: ineligible_caught"] = round((excluded & bad).sum() / bad.sum(), 4)
    out["pairs_passing_filter"] = int((m["passes_filter"] == 1).sum())
    return out, sa

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="gold_annotations_ok193.csv")
    ap.add_argument("--pred", nargs="+", required=True, help="name=file.csv ...")
    ap.add_argument("--profiles", default="student_profiles.csv")
    ap.add_argument("--gt", default="match_matrix.csv")
    ap.add_argument("--host-file", default="gold_host_country.csv")
    a = ap.parse_args()
    gold = pd.read_csv(a.gold, dtype=str, keep_default_na=False)
    prof = pd.read_csv(a.profiles, dtype=str, keep_default_na=False)
    gt = pd.read_csv(a.gt)
    hf = pd.read_csv(a.host_file, dtype=str, keep_default_na=False)
    hosts = dict(zip(hf["id"], hf["host_country"]))

    field_rows, err_rows, filt_rows = [], [], []
    o, _ = evaluate_filter("oracle (gold criteria)", gold, prof, gt, hf); filt_rows.append(o)
    for spec in a.pred:
        name, path = spec.split("=", 1)
        pred = pd.read_csv(path, dtype=str, keep_default_na=False)
        fr, er = evaluate_fields(name, gold, pred, hosts)
        field_rows += fr; err_rows += er
        o, _ = evaluate_filter(name, pred, prof, gt, None); filt_rows.append(o)
    # no Stage A at all: nothing is excluded
    filt_rows.append({"criteria": "no filter", "label_full: false_exclusion_rate": 0.0,
                      "label_full: ineligible_caught": 0.0, "label_core: false_exclusion_rate": 0.0,
                      "label_core: ineligible_caught": 0.0, "pairs_passing_filter": len(gt)})
    fm = pd.DataFrame(field_rows); fl = pd.DataFrame(filt_rows)
    fm.to_csv("extraction_metrics.csv", index=False); fl.to_csv("filter_metrics.csv", index=False)
    pd.DataFrame(err_rows).to_csv("extraction_errors.csv", index=False)
    pd.set_option("display.width", 200)
    print(fm[["extractor", "field", "gold_stated", "pred_stated", "detect_precision", "detect_recall",
              "hallucination_rate", "exact_accuracy", "accuracy_when_stated"]].to_string(index=False))
    print(); print(fl.to_string(index=False))
