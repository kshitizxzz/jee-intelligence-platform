# JEE Intelligence Platform

A Streamlit app that takes a JEE Advanced rank and turns it into a ranked list of realistic IIT branch options, along with a few tools to double check that list against 9 years of actual JoSAA admission data.

I built this as a portfolio project after noticing most "rank predictor" sites just do a nearest-cutoff lookup and call it a day. I wanted to try something with a bit more depth: real ML, the actual seat-allocation algorithm JoSAA runs, and some basic NLP for people who don't already know what half these branch names mean.

## Why it's more than a rank lookup

The core prediction is a soft-voting ensemble (KNN, Decision Tree, Random Forest, AdaBoost) that scores how eligible you are for each institute-branch combo, mixed with an institute desirability score (NIRF rank plus historical demand) that you can weight yourself with a slider. I also trained a small MLP alongside it out of curiosity, and it actually beat the ensemble by about 4 points on accuracy. I kept the ensemble in production anyway, since it's a lot easier to explain "3 out of 4 models said yes" to a 17-year-old than to explain what a hidden layer is doing. I measured that tradeoff rather than just assuming classical ML would win.

JoSAA's counselling runs on a multi-round deferred-acceptance (Gale-Shapley) process, so I reimplemented that algorithm from scratch in Python instead of faking a simplified version. It powers the Round Simulator page, along with real historical round-by-round movement, to back up the Freeze/Float/Slide advice with something more than a guess.

There's also a plain-English branch matcher — type something like "I like coding and want to work with AI" and it maps you to actual branch names like Computer Science and Engineering. That's a hand-built topic lexicon plus TF-IDF and cosine similarity, no embedding models involved, mostly so it stays light enough to run on free hosting.

And underneath all of it sits 124,861 rows of real JoSAA cutoffs from 2016 to 2024, across all 23 IITs, every round, every seat-type/gender pool — looked up with binary search instead of scanning the whole thing.

## Pages

| Page | What it does |
|---|---|
| Home | Landing page, project overview |
| 1. Choice List Builder | Enter rank + category + gender, get a ranked, explainable branch list. Sidebar shows model accuracy for all four models plus the MLP benchmark. |
| 2. Round Simulator | Real round-by-round closing-rank trend, Freeze/Float/Slide advice, and a live synthetic deferred-acceptance run. |
| 3. Rank Band Explorer | Look up what a given rank actually got, across all 9 years. |
| 4. Trend Forecaster | Per-branch closing-rank trend, projected one year ahead with a confidence band. |
| 5. Branch Matcher | Describe your interests in plain English, get matched to formal branch names. |

## Architecture

[`architecture.md`](architecture.md) has the full breakdown layer by layer — data, feature engineering, the ranking model, the NLP matcher, the DSA pieces, forecasting, UI, and deployment.

It's a single Python app. Streamlit handles routing and UI, so there's no separate backend, no database, no JS.

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

Push to GitHub, connect the repo to [Streamlit Community Cloud](https://streamlit.io/cloud), point it at `Home.py`. Every push to `main` redeploys automatically, so there's nothing extra to set up.

## Data

`data/processed/josaa_iit_consolidated.csv` has 124,861 rows covering all 23 IITs from 2016-2024 (JoSAA hasn't published a bulk export for 2025 yet, so that's 9 years, not 10). Column definitions, sources, and how I verified it are in [`README_dataset.md`](README_dataset.md).

`data/nirf_rankings.csv` is the 2025 NIRF Engineering ranking for all 23 IITs, used as the institute-desirability input.

## What this covers

ML (ensembling, evaluation), a bit of DL (the MLP benchmark and the tradeoff around it), NLP (lexicon expansion, TF-IDF, cosine similarity), DSA (binary search, a trie, and the deferred-acceptance algorithm from scratch), and the data engineering to pull three different public sources into one clean schema.
