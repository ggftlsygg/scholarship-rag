# Eligibility-Aware RAG for Personalized Scholarship Discovery

Generic search recommends scholarships a student cannot apply for. This system first checks five hard criteria (degree level, nationality, GPA, enrollment status, deadline) and removes ineligible scholarships. It then ranks the rest by semantic similarity and writes an answer that cites its sources.

Text Mining course project, Tokyo International University (Khan, Ayaulym).

## How it works

1. **Criteria extraction (offline, once).** Claude Haiku 5.5 reads each of 193 scholarship pages and fills 12 fields (`extract_llm.py` → `extracted_haiku.csv`).
2. **Stage A, hard filter.** Each criterion scores 1 (met), 0 (violated) or 0.5 (not stated). Any 0 removes the scholarship. EligScore is the mean of the five scores (`stage_a.py`, `criteria_common.py`).
3. **Semantic retrieval.** Pages are split into 150-word chunks and embedded with `bge-small-en-v1.5`. A scholarship's score is its best chunk's similarity, and that chunk becomes the evidence (`retrieval.py`).
4. **Ranking.** `Rank = α·Sim + (1−α)·EligScore` over the scholarships that pass the filter. α = 0.85 was tuned on dev students.
5. **Grounded answer.** The top-5, with their check results and verbatim evidence, go to Claude Haiku 5.5, which answers with `[S###]` citations (`generate_answers.py`).

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py                      # web interface at http://localhost:8501
python ask.py --student P01 --no-llm      # command line: ranked list only, no API key needed
python ask.py                             # command line, interactive: enter your own profile
python add_scholarship.py <link>          # add a new scholarship (kept apart from the 193 verified ones)
```

For the written answer, set the `ANTHROPIC_API_KEY` environment variable. One answer costs about $0.002. Typos in the country, degree, field or question are corrected automatically, and the script prints each correction.

## Results (test set: 24 students × 193 scholarships)

| method | P@10 | NDCG@10 | ineligible in top-10 |
|---|---|---|---|
| BM25 | 0.32 | 0.31 | 60% |
| Semantic RAG (Vadlamudi) | 0.35 | 0.32 | 60% |
| TF-IDF + SVM (Al-Dilami) | 0.25 | 0.25 | 67% |
| **Ours (Haiku criteria), α = 0.85** | **0.61** | **0.52** | **28%** |
| Ours with gold criteria (upper bound) | 0.65 | 0.55 | 24% |
| Ablation: no Stage A | 0.50 | 0.42 | 39% |

All differences between Ours and the three baselines are significant (Wilcoxon signed-rank, p < 0.001). An audit of 65 generated answers found no invented citations, numbers or quotes (`generation_error_analysis.md`).

## Repository map

| group | files |
|---|---|
| run the system | `app.py` (web), `ask.py` (command line), `add_scholarship.py` (add scholarships) |
| core modules | `stage_a.py`, `criteria_common.py`, `retrieval.py`, `generate_answers.py` |
| data | `gold_annotations_ok193.csv` (pages + gold annotations), `extracted_haiku.csv`, `student_profiles.csv`, `chosen_alpha.json`, `corpus_extra.csv` + `extracted_extra.csv` (added scholarships, created by `add_scholarship.py`) |
| experiments | `make_profiles.py`, `build_ground_truth.py`, `extract_llm.py`, `extract_regex.py`, `evaluate_extraction.py`, `rank_and_evaluate.py`, `audit_answers.py`, `verify.py`, `test_rules.py`, `stress_cases.csv` |
| results | `results_test.csv`, `significance_test.csv`, `alpha_curve.png`, `extraction_metrics.csv`, `filter_metrics.csv`, `answers.md`, `generation_error_analysis.md` |
| caches | `llm_cache/`, `judge_cache/`: stored API responses, so the experiments rerun without new API calls |
| data collection | `data_collection/`: how the corpus was built — Hugging Face dataset → live-page verification → Tavily search for current pages → PDF texts → manual review (`manual_review.csv`) → 247 scholarships → 193 with status ok after annotation |

## Reproducing the experiments

```bash
python build_ground_truth.py
python extract_regex.py --corpus gold_annotations_ok193.csv --out extracted_regex.csv
python extract_llm.py                     # uses llm_cache/, no new API calls
python evaluate_extraction.py --pred regex=extracted_regex.csv haiku=extracted_haiku.csv
python rank_and_evaluate.py
python generate_answers.py --students all --cases stress_cases.csv
python audit_answers.py --judge           # uses judge_cache/
```

## Limitations

- Conditions outside the five criteria (residence, membership, a specific university) are not filtered. This is why 24–28% of the top-10 remain ineligible even with perfect extraction.
- Haiku invents an enrollment status for 68% of pages that do not state one.
- Ranking ignores the field of study.
- The 50 students are synthetic, the ground truth is our own annotation, and the evaluation uses a single query date (2026-11-21).

## References and data sources

**Baseline papers**

1. Vadlamudi, S., Shashikala, K. S., Sindhu, J., Shaik, B. A., Jerin, J. M., & Bharathy, A. M. V. (2025). Retrieval-augmented generation (RAG) chatbot for scholarship finding. *2025 International Conference on Emerging Technologies in Computing and Communication (ETCC)*, 1–6. IEEE. https://doi.org/10.1109/ETCC65847.2025.11108461
2. Al-Dilami, R., Al-Muaalemi, L., Redwan, O., Al-Zohairi, S., & Mareh, A. (2025). Enhancement of scholarship decisions precision using support vector machine algorithm. *2025 5th International Conference on Emerging Smart Technologies and Applications (eSmarTA)*, 1–8. IEEE. https://doi.org/10.1109/eSmarTA66764.2025.11132240

**Data**

- Starting list of scholarships: Hugging Face dataset [AlberBshara/scholarships_dataset](https://huggingface.co/datasets/AlberBshara/scholarships_dataset).
- Scholarship texts were collected in November 2026 from the public web pages of the universities, foundations and scholarship portals that publish them. Each scholarship keeps its source link in the `url` column of `gold_annotations_ok193.csv`. All page texts remain the property of their respective owners. They are included only for non-commercial, educational research. If you own a page and want its text removed, please open an issue.
- The eligibility annotations, the 50 student profiles and the ground truth were created by the authors.

**Models and tools**

- Embeddings: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), via sentence-transformers.
- Criteria extraction and answers: Claude Haiku 5.5. LLM judge in the audit: Claude Sonnet 5.5 (Anthropic API).
- BM25: [rank_bm25](https://github.com/dorianbrown/rank_bm25). Page search during data collection: Tavily Search API. Page text extraction: trafilatura, BeautifulSoup. Web interface: Streamlit.
