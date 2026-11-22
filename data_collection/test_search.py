import time
from ddgs import DDGS

queries = [
    "AMS Minority Scholarship scholarship eligibility 2026",
    "AMS Minority Scholarship eligibility",
    "Aker Cambridge Scholarship eligibility",
]
backends = ["auto", "duckduckgo", "bing", "brave", "google"]

ddgs = DDGS()
for backend in backends:
    for q in queries:
        try:
            res = list(ddgs.text(q, max_results=8, backend=backend))
            print(f"{backend:10} | {len(res)} results | {q}")
        except Exception as e:
            print(f"{backend:10} | ERROR: {e} | {q}")
        time.sleep(2)