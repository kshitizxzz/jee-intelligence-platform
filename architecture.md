# JEE Intelligence Platform — Architecture

## Overall idea

A Streamlit app that takes a student's JEE Advanced rank, category, and gender pool and gives them four things no existing predictor does well: an explainable, ranked list of realistic IIT branch choices; a round-by-round simulation of JoSAA's actual seat-allocation algorithm with Freeze/Float/Slide advice; a direct, 9-year historical view of what rank bands like theirs actually got; and a plain-English branch matcher for students who don't yet know the jargon. Everything runs as one Python app — no separate backend, no database server, no JS frontend.

## Layers

**1. Data layer**
Architecture: one consolidated CSV (124,861 rows) — every IIT, every round, every branch, every Seat Type x Gender combo, 2016-2024 — plus a small static NIRF ranking table for institute-quality scoring. Loaded into memory at app start; no database needed at this size.
Tech stack: Python, pandas, CSV (~23 MB).
Skills: data cleaning/ETL, DSA (sorting, needed before binary search in layer 5).

**2. Feature engineering layer**
Architecture: raw rank rows turned into model-ready features — rank percentile within year, a branch-demand proxy from cutoff compression, NIRF-based desirability score, one-hot category/gender encodings, year-over-year deltas.
Tech stack: pandas, numpy, scikit-learn preprocessing.
Skills: feature engineering, basic statistics.

**3. ML ranking layer (Choice-List Builder)**
Architecture: given rank + category + gender, an ensemble (KNN for "similar historical admits," Decision Tree/Random Forest/AdaBoost for eligibility) scores every institute-branch pair on eligibility x desirability, blended via a user-adjustable prestige-vs-branch-fit slider, output as a ranked, explainable list. A small MLPClassifier (2 hidden layers, 32→16 neurons) is trained alongside as a benchmark-only comparison — it is scored and shown in the UI but deliberately excluded from production inference. Measured result: the MLP edges out the classical ensemble on held-out accuracy (~83% vs ~79%), but the ensemble's per-model votes stay easier to reason about and debug for a high-stakes decision a 17-year-old is making — an accuracy-vs-interpretability tradeoff that was measured, not assumed.
Tech stack: scikit-learn (KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, MLPClassifier), joblib.
Skills: ML — classification, ensembling, evaluation, deep learning (MLP) benchmarking.

**4. NLP layer (Branch Matcher)**
Architecture: a hand-built topic lexicon (~28 topic keys, e.g. "ai_data," "mechanical," "biotech_bio") expands each formal JEE branch name into plain-language synonyms and concepts, stripping degree-duration boilerplate first. A `TfidfVectorizer` is fit over the expanded texts; a free-text user query is vectorized the same way and ranked against every branch via cosine similarity, with matched topic tags surfaced for explainability. Scikit-learn only — no embedding/transformer models — to stay light enough for Streamlit Cloud.
Tech stack: scikit-learn (TfidfVectorizer, cosine_similarity), pure Python (lexicon, regex cleanup).
Skills: NLP — lexicon-based query expansion, TF-IDF, vector similarity search.

**5. Algorithms layer — the differentiator**
Architecture: three from-scratch components, no library shortcuts: binary search over sorted closing-rank arrays for O(log n) rank-band lookup; a reimplementation of JoSAA's real seat-allocation algorithm (multi-round deferred acceptance / Gale-Shapley variant) to simulate round-by-round movement and drive the Freeze/Float/Slide advisor; a Trie for institute/branch autocomplete.
Tech stack: pure Python.
Skills: DSA — binary search, stable-matching/graph algorithms, tries. This is the showcase layer for a resume.

**6. Forecasting layer**
Architecture: per institute-branch-category-gender, a 9-year closing-rank time series fit with linear regression to project next year's likely closing-rank band with a range.
Tech stack: scikit-learn (LinearRegression), numpy.
Skills: ML — regression, light time-series reasoning.

**7. Application/UI layer**
Architecture: one Streamlit multi-page app — pages for Choice-List Builder, Round Simulator, Rank-Band Explorer, Trend Forecaster, Branch Matcher — with cached data/model loading and interactive charts.
Tech stack: Streamlit (`st.cache_data`, `st.cache_resource`, `pages/` for multipage), Plotly for charts.
Skills: Python only — no HTML/CSS/JS required.

**8. Persistence layer**
Architecture: trained models and precomputed structures (sorted rank arrays, trie, fitted TF-IDF matcher) are built once and serialized/cached, then loaded at startup instead of being retrained per request.
Tech stack: joblib/pickle, `st.cache_resource`, flat files committed to the repo.
Skills: basic packaging discipline.

**9. Deployment layer**
Architecture: push the app, data, and serialized models to a public GitHub repo, connect it to Streamlit Community Cloud. Every `git push` auto-redeploys.
Tech stack: GitHub, Streamlit Community Cloud.
Skills: git basics only.

## On "I don't know CI/CD"

You don't need any. Streamlit Community Cloud's deploy model *is* git-push-to-deploy — there's no pipeline to write, no Docker, no servers. Connect the repo once in the Streamlit Cloud dashboard; every push to `main` rebuilds and redeploys automatically. That's the entire "CI/CD" for this project.

## Directory structure

```
jee-intelligence-platform/
├── Home.py                             # Streamlit — app entry point, landing page
├── requirements.txt                    # pip — streamlit, pandas, scikit-learn, plotly, joblib
├── .streamlit/
│   └── config.toml                     # Streamlit — theme/page config
├── pages/                              # Streamlit — multipage routing (auto-built from filenames)
│   ├── 1_Choice_List_Builder.py        # Streamlit + scikit-learn — ranker UI (layer 3)
│   ├── 2_Round_Simulator.py            # Streamlit + pure Python — Freeze/Float/Slide UI (layer 5)
│   ├── 3_Rank_Band_Explorer.py         # Streamlit + pandas — binary-search lookup UI (layer 5)
│   ├── 4_Trend_Forecaster.py           # Streamlit + Plotly — forecast chart UI (layer 6)
│   └── 5_Branch_Matcher.py             # Streamlit + scikit-learn — NLP matcher UI (layer 4)
├── src/
│   ├── data_pipeline/
│   │   ├── consolidate.py              # pandas — merges raw sources into one schema (layer 1)
│   │   └── features.py                 # pandas, numpy, scikit-learn — feature engineering (layer 2)
│   ├── algorithms/
│   │   ├── rank_lookup.py              # pure Python — binary search, O(log n) (layer 5)
│   │   ├── deferred_acceptance.py      # pure Python — Gale-Shapley JoSAA simulation (layer 5)
│   │   └── trie.py                     # pure Python — autocomplete structure (layer 5)
│   ├── models/
│   │   ├── train_ranker.py             # scikit-learn — KNN/DT/RF/AdaBoost + MLP benchmark (layer 3)
│   │   ├── train_forecaster.py         # scikit-learn — LinearRegression per series (layer 6)
│   │   └── artifacts/                  # joblib — serialized .pkl/.joblib model files (layer 8)
│   ├── nlp/
│   │   └── branch_matcher.py           # scikit-learn — lexicon + TF-IDF branch matcher (layer 4)
│   └── utils/
│       └── helpers.py                  # pure Python — shared constants/formatters/cached loaders
├── data/
│   ├── raw/                            # CSV/XLSX — original GitHub-sourced files
│   ├── processed/
│   │   └── josaa_iit_consolidated.csv  # pandas-cleaned, 124,861 rows (layer 1)
│   └── nirf_rankings.csv               # static CSV — institute desirability input
└── README.md                           # Markdown — project writeup for resume/GitHub
```

Everything under `pages/` and `Home.py` is what Streamlit Cloud runs directly — no build step, no compiled assets.
