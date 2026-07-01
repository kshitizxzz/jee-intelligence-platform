# JEE Intelligence Platform

A Streamlit app that turns a JEE Advanced rank into an explainable, ranked list of realistic IIT branch choices — plus the tools to sanity-check that list against 9 years of real JoSAA outcomes.

Built as a resume/portfolio project to demonstrate ML, DL, NLP, and DSA on a real dataset, not toy data.

## What makes this different

Most JEE rank predictors (e.g. CampusLoom) just do a nearest-closing-rank lookup. This project goes further:

1. **Explainable ranking, not a black box.** A KNN + Decision Tree + Random Forest + AdaBoost soft-vote ensemble scores eligibility, blended with an institute-desirability score (NIRF rank + historical demand) on a user-adjustable slider. A small neural net (MLPClassifier) is benchmarked against it — it actually scores ~4 points higher on accuracy, but is kept out of production inference deliberately, because a per-model vote is easier to debug than hidden-layer activations for a decision this consequential. That tradeoff was measured, not assumed.
2. **A real seat-allocation algorithm, reimplemented from scratch.** JoSAA's multi-round seat allocation is a deferred-acceptance / Gale-Shapley stable-matching process. This project reimplements that algorithm in pure Python (no library) to simulate round-by-round seat movement, paired with real historical round-by-round closing-rank data to drive Freeze/Float/Slide advice.
3. **9 years of real outcomes, searchable in O(log n).** 124,861 rows across 2016-2024, every IIT, every round, every Seat-Type x Gender pool, looked up via binary search (`bisect_left`) over sorted closing-rank arrays — not a linear scan.
4. **A plain-English branch matcher.** Describe your interests in normal language ("I like coding and want to work with AI") and get matched to formal branch names like "Computer Science and Engineering," via a hand-built topic lexicon expansion + TF-IDF/cosine-similarity search — no need to already know the jargon.

## Pages

| Page | What it does |
|---|---|
| Home | Landing page, project overview |
| 1. Choice List Builder | Rank + category + gender in → ranked, explainable branch list out. Sidebar shows all model accuracy metrics, including the MLP benchmark comparison. |
| 2. Round Simulator | Real round-by-round closing-rank trend + Freeze/Float/Slide advice, plus a synthetic deferred-acceptance simulation you can run live. |
| 3. Rank Band Explorer | "What did a rank like mine actually get?" — binary-search lookup across all 9 years. |
| 4. Trend Forecaster | Per-branch closing-rank trend, linear-regression-projected one year ahead with a confidence band. |
| 5. Branch Matcher | Free-text interest description → matched formal branch names, cross-referenced with real cutoffs. |

## Architecture

See [`architecture.md`](architecture.md) for the full layer-by-layer breakdown (data, feature engineering, ML ranking, NLP matching, DSA algorithms, forecasting, UI, persistence, deployment).

Everything is one Python app — Streamlit handles routing and UI, so there's no separate backend, no database, and no JS frontend.

## Running locally

```bash
pip install -r requirements.txt

# one-time: build features + train models (writes to src/models/artifacts/)
python3 -m src.data_pipeline.features
python3 -m src.models.train_ranker
python3 -m src.models.train_forecaster

streamlit run Home.py
```

## Deploying

Push this repo to GitHub, connect it to [Streamlit Community Cloud](https://streamlit.io/cloud), point it at `Home.py`. Every `git push` to `main` auto-redeploys — no CI/CD pipeline needed.

## Data

`data/processed/josaa_iit_consolidated.csv` — 124,861 rows, all 23 IITs, 2016-2024 (9 years; JoSAA's official archive has no bulk export for 2025 yet). See [`README_dataset.md`](README_dataset.md) for column definitions, sources, and verification notes.

`data/nirf_rankings.csv` — 2025 NIRF Engineering ranks for all 23 IITs, used as an institute-desirability input.

## Skills demonstrated

ML (classification ensembling, evaluation), DL (MLPClassifier benchmark with a measured interpretability-vs-accuracy tradeoff), NLP (lexicon-based query expansion, TF-IDF, cosine similarity), DSA (binary search, Trie, deferred-acceptance/stable-matching algorithm implemented from scratch), data engineering (multi-source ETL, feature engineering), and basic time-series forecasting (per-segment linear regression).
