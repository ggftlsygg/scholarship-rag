"""Planted-error tests: each rule must fire on a bad answer and stay silent on the original good answers."""
import json, copy
from audit_answers import rule_checks
jobs = {json.loads(l)["job_id"]: json.loads(l) for l in open("answers_log.jsonl")}

def fire(job, answer):
    p = copy.deepcopy(jobs[job]); p["answer"] = answer; p["stop_reason"] = "end_turn"
    return {i["check"].split()[0] for i in rule_checks(p)}

p = jobs["P30"]; ids = [i["id"] for i in p["items"]]
base = "\n\n".join(f"**X** [{i}] matches; the minimum grade is UNKNOWN, check it. You live in China." for i in ids)
base += "\n\nPlease confirm everything on the official pages."
tests = {
    "R01": base.replace(f"[{ids[-1]}]", "", 1),
    "R02": base + " See also [S999].",
    "R03": base.replace(f"[{ids[0]}]", "[TMP]").replace(f"[{ids[1]}]", f"[{ids[0]}]").replace("[TMP]", f"[{ids[1]}]"),
    "R04": base.replace("matches;", "matches and pays $48,123 per year;", 1),
    "R05": base.replace("matches;", 'matches ("applicants must have a perfect record in mathematics");', 1),
    "R11": base.replace("matches;", "matches, and it is also open to Peru;", 1),
    "R12": base.replace("matches;", "matches, like Harvard;", 1),
    "R13": base.replace("\n\nPlease confirm everything on the official pages.", ""),
    "R14": base + " 你好",
}
# R10: S177 asks to reside in Germany; P30 lives in China -> paragraph that never mentions residence
s177 = ids.index("S177") if "S177" in ids else None
if s177 is not None:
    paras = base.split("\n\n")
    paras[s177] = "**Rosa** [S177] matches; the minimum grade is UNKNOWN, check it."
    tests["R10"] = "\n\n".join(paras)
ok = True
for rule, ans in tests.items():
    got = fire("P30", ans); hit = rule in got
    ok &= hit; print(f"{rule}: {'fires' if hit else 'MISSED'}  {sorted(got)}")
# R06 / R07 / R08 / R09 need specific items: find one in the log and plant the error
def find(pred):
    for j, q in jobs.items():
        for k, it in enumerate(q["items"]):
            if pred(it, q): return j, k
def single(job, k, text):
    q = jobs[job]; ids = [i["id"] for i in q["items"]]
    paras = [f"**X** [{i}] is listed; the minimum grade is UNKNOWN, check it." for i in ids]
    paras[k] = text.format(id=ids[k]); return "\n\n".join(paras) + "\n\nCheck the official pages."
j, k = find(lambda it, q: "full" not in it["funding_type"] and "fully" not in (it["evidence"] + it["passage"] + it["name"]).lower())
r = fire(j, single(j, k, "**X** [{id}] is a fully funded award that fits well.")); print("R06:", "fires" if "R06" in r else "MISSED"); ok &= "R06" in r
r = fire(j, single(j, k, "**X** [{id}] is not fully funded, so check the amount.")); print("R06 negated:", "silent" if "R06" not in r else "FALSE ALARM"); ok &= "R06" not in r
j, k = find(lambda it, q: any(c["result"] == "UNKNOWN" for c in it["criteria"].values()))
r = fire(j, single(j, k, "**X** [{id}] fits on degree and nationality.")); print("R07:", "fires" if "R07" in r else "MISSED"); ok &= "R07" in r
j, k = find(lambda it, q: it["criteria"]["deadline"]["requires"].startswith("annual:") and ";" not in it["criteria"]["deadline"]["requires"])
r = fire(j, single(j, k, "**X** [{id}] fits on degree and nationality; the grade is unknown.")); print("R08:", "fires" if "R08" in r else "MISSED"); ok &= "R08" in r
j, k = find(lambda it, q: "male" in it["other_hard"].lower() or "women" in it["other_hard"].lower())
r = fire(j, single(j, k, "**X** [{id}] fits on degree and nationality; the grade is unknown.")); print("R09:", "fires" if "R09" in r else "MISSED"); ok &= "R09" in r
r = fire(j, single(j, k, "**X** [{id}] is not a fit for you.")); print("R09 not-fit:", "silent" if "R09" not in r else "FALSE ALARM"); ok &= "R09" not in r
p = copy.deepcopy(jobs["P01"]); p["stop_reason"] = "max_tokens"; r = {i["check"].split()[0] for i in rule_checks(p)}
print("R15:", "fires" if "R15" in r else "MISSED"); ok &= "R15" in r
print("ALL OK" if ok else "SOME CHECKS FAILED")
