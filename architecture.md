# JEE Intelligence Platform — Architecture

## Overall idea

Takes a student's JEE Advanced rank, category, and gender pool and gives them a ranked list of realistic IIT branch choices, a round-by-round simulation of how JoSAA's actual seat allocation works, a plain historical lookup of what rank bands like theirs got in past years, and a branch matcher for people who don't know the jargon yet. It's one Python app — Streamlit does the routing and UI, so there's no separate backend or database to run.

## Layers

**1. Data layer.** One consolidated CSV, 124,861 rows, every IIT/round/branch/Seat-Type x Gender combo from 2016-2024, plus a small static NIRF ranking table for institute-quality scoring. It's about 23 MB, so it just gets loaded into memory at app start rather than needing a real database. Built with Python and pandas; sorting the data here is also what makes the binary search in layer 5 possible.

**2. Feature engineering layer.** Turns the raw rank rows into model-ready features: rank percentile within a year, a branch-demand proxy from how tight the cutoffs are, a NIRF-based desirability score, one-hot category/gender encodings, year-over-year deltas. Pandas, numpy, and a bit of scikit-learn preprocessing.

**3. ML ranking layer (Choice-List Builder).** Given a rank, category, and gender, an ensemble of KNN, Decision Tree, Random Forest, and AdaBoost scores every institute-branch pair on eligibility, then blends that with desirability using a slider the user controls. I also trained a small MLPClassifier (2 hidden layers, 32→16 neurons) alongside it just as a benchmark — it actually beats the ensemble on held-out accuracy (~83% vs ~79%), but I kept the ensemble in production since its per-model votes are much easier to reason about for a decision like this than an MLP's hidden layers would be. Built with scikit-learn (KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, MLPClassifier) and joblib for serialization.

**4. NLP layer (Branch Matcher).** A topic lexicon I built by hand (around 28 topics — "ai_data," "mechanical," "biotech_bio," etc.) expands each formal branch name into plain-language synonyms, after stripping out degree-duration boilerplate. A TfidfVectorizer runs over the expanded text, and a free-text query gets ranked against every branch via cosine similarity, with the matched topics shown so it's clear why something matched. Scikit-learn only, no embedding models, mostly to keep it light enough for free hosting.

**5. Algorithms layer.** This is the part with no library shortcuts: binary search over sorted closing-rank arrays for O(log n) lookup, a from-scratch reimplementation of JoSAA's actual multi-round deferred-acceptance (Gale-Shapley) algorithm to simulate round-by-round seat movement, and a trie for institute/branch autocomplete. All pure Python.

**6. Forecasting layer.** For each institute-branch-category-gender combo, fits a linear regression over its closing-rank history (up to 9 years) to project next year's likely band. Scikit-learn's LinearRegression plus numpy.

**7. Application/UI layer.** A Streamlit multi-page app — Choice-List Builder, Round Simulator, Rank-Band Explorer, Trend Forecaster, Branch Matcher — with cached data/model loading and Plotly charts. No HTML/CSS/JS anywhere.

**8. Persistence layer.** Models and precomputed structures (sorted rank arrays, the trie, the fitted TF-IDF matcher) get built once and cached/serialized rather than rebuilt on every request. Joblib/pickle plus `st.cache_resource`, with the artifact files committed to the repo.

**9. Deployment layer.** Push to GitHub, connect to Streamlit Community Cloud, and every push to `main` redeploys automatically.

## Deployment model

No CI/CD pipeline is needed. Streamlit Community Cloud's deploy model is git-push-to-deploy — no Docker, no servers to manage. The repo is connected once in the Streamlit Cloud dashboard; every push to `main` rebuilds and redeploys automatically.

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
