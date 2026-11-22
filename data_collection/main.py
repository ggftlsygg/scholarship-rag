import pandas as pd

url = "https://huggingface.co/datasets/AlberBshara/scholarships_dataset/resolve/main/scholars.json"
raw = pd.read_json(url)

df = pd.json_normalize(raw["scholars"])      # -> колонки name, url, text
df["text"] = df["text"].fillna("").str.strip()
df["n_chars"] = df["text"].str.len()

print(df.shape)
print(df["n_chars"].describe())
print("empty texts:", (df["n_chars"] < 200).sum())
print("duplicate names:", df["name"].duplicated().sum())

df.to_csv("hf_scholarships.csv", index=False, encoding="utf-8-sig")