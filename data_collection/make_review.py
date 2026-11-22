import pandas as pd

df = pd.read_csv("corpus/corpus_fresh.csv")

check_all = df[df["text_source"].str.startswith("new_url")
               | df["text_source"].isin(["dataset_check_in_browser", "live_check"])]
changed = df[df["text_source"].isin(["live_changed", "dataset_changed"])]
same = df[df["text_source"].isin(["live_same", "dataset_same"])]

review = pd.concat([
    check_all,
    changed.sample(n=min(15, len(changed)), random_state=1),
    same.sample(n=min(10, len(same)), random_state=1),
])
review = review[["name", "url", "text_source", "latest_year"]].copy()
review["decision"] = ""   # keep / reject
review["reason"] = ""     # wrong_scholarship / list_page / no_criteria / outdated / unreliable / ok_aggregator / text_outdated
review = review.sort_values(["text_source", "name"])
review.to_csv("corpus/manual_review.csv", index=False, encoding="utf-8-sig")
print(review["text_source"].value_counts().to_string())
print(f"Rows to review: {len(review)}")