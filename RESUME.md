# JEE Intelligence Platform — Resume & Interview Content

All numbers below are pulled directly from this project's actual training runs and dataset (verified, not estimated) — safe to repeat in an interview.

## Resume header line

```
JEE Intelligence Platform | Personal Project | Python, scikit-learn, Streamlit, Pandas, ML/DL, NLP, DSA
```

## Bullet bank (pick the set that matches the role)

### Option A — ML/Data Science emphasis

- Engineered an end-to-end ML/data pipeline (Python, Pandas, NumPy, scikit-learn) for predictive classification of IIT seat-allocation eligibility from JEE Advanced rank, using 124,861 feature-engineered records (23 IITs, 2016-2024).
- Designed a soft-voting ensemble classifier (KNN, Decision Tree, Random Forest, AdaBoost boosting) achieving 79.3% accuracy / 0.80 F1-score across 5,240 institute-branch-category-gender combinations.
- Benchmarked a deep-learning MLP neural network (2 hidden layers) against the ensemble, scoring 82.6% accuracy (+3.3 pts), then selected the explainable ensemble for production after evaluating the accuracy-vs-interpretability tradeoff.
- Engineered a time-series forecasting layer using per-segment linear regression to project next-year closing ranks with confidence intervals across 5,349 combinations, with fallback logic for insufficient historical data.

### Option B — DSA / Software Engineering emphasis

- Implemented a Gale-Shapley deferred-acceptance (stable matching) algorithm from scratch in pure Python to simulate India's multi-round national engineering admissions process — the actual algorithm real seat allocation runs on, not a simplified lookup table.
- Built a binary-search-based rank-lookup engine (O(log n)) over 124,861 sorted records, cutting worst-case comparisons from ~124,861 to ~17 (roughly 7,000x fewer) versus a linear scan.
- Designed a Trie data structure for real-time autocomplete across 23 institutes and 183 academic programs.

### Option C — NLP emphasis

- Built a natural-language branch-matching feature: a hand-crafted 28-topic lexicon expands 183 formal academic program names into plain-language concepts, then ranks free-text user queries against them via TF-IDF vectorization and cosine similarity — entirely in scikit-learn, with no embedding-model dependency, keeping the app lightweight to deploy.

### Option D — Concise 3-bullet version (limited resume space)

- Built a 5-page Streamlit app predicting IIT seat-allocation outcomes from JEE Advanced rank, trained on 124,861 historical records across 23 institutes (2016-2024); a 4-model ML ensemble reached 79.3% accuracy, benchmarked against a from-scratch MLP scoring 82.6%.
- Reimplemented JoSAA's real multi-round Gale-Shapley deferred-acceptance algorithm and an O(log n) binary-search rank engine in pure Python — no algorithm libraries used.
- Added an NLP feature matching free-text career interests to formal branch names via TF-IDF/cosine similarity over a custom 28-topic lexicon.

## ATS keyword line

Include this somewhere in the project's tech line or skills section so applicant tracking systems pick up the keywords:

```
Python, Machine Learning, Deep Learning, Natural Language Processing, Data Structures & Algorithms,
scikit-learn, pandas, NumPy, Streamlit, Plotly, TF-IDF, Cosine Similarity, Ensemble Learning,
Neural Networks (MLP), Binary Search, Trie, Stable Matching / Gale-Shapley Algorithm,
Feature Engineering, Time-Series Forecasting, Linear Regression, Data Pipeline / ETL, Git
```

ATS note: keep the actual bullets you paste into your resume as plain text (no tables, no graphics, no icons) — the formatting above is just for this reference doc.

## Numbers cheat sheet — what to say when asked

| If a recruiter asks... | Say this |
|---|---|
| "How big was your dataset?" | 124,861 admission records, scraped/consolidated from 3 public sources |
| "What's the scope?" | 23 IITs, 9 years (2016-2024), 183 branches, 10 seat-type categories x 3 gender pools |
| "How well does your model perform?" | Ensemble: 79.3% accuracy, 0.80 F1, across 5,240 unique combinations |
| "Why an ensemble instead of one model?" | Individual scores: KNN 78.4%, Random Forest 77.3%, Decision Tree 68.5%, AdaBoost 65.9% — voting smoothed out each model's weak spots |
| "Did you try deep learning?" | Yes — an MLP (2 hidden layers, 32→16 units) scored 82.6% accuracy / 0.817 F1, beating the ensemble by 3.3 points |
| "So why not ship the more accurate model?" | This is a high-stakes decision for a 17-year-old; the ensemble's per-model votes are auditable, an MLP's hidden-layer activations aren't. Measured the tradeoff first, then made the call — didn't just assume classical ML would lose |
| "How did you optimize the lookups?" | Binary search (`bisect`) over sorted rank arrays: O(log n), ~17 comparisons worst case vs up to 124,861 for a linear scan |
| "How do you forecast responsibly?" | Linear regression only where ≥4 years of history exists (2,657 combos); flat-band carry-forward fallback elsewhere (2,692 combos) — never silently extrapolates from too little data |
| "What NLP did you use, and why not embeddings?" | Hand-built 28-topic lexicon + TF-IDF + cosine similarity over 183 branch names — scikit-learn only, no transformer/embedding model, keeps deploy footprint small on free hosting |
| "What's the architecture?" | ~1,289 lines of Python across 21 modules, single Streamlit app — no separate backend, database, or JS frontend |
| "What's actually novel vs. existing tools?" | Competing tools (e.g. CampusLoom) do a static nearest-rank lookup. This reimplements JoSAA's actual multi-round Gale-Shapley deferred-acceptance algorithm from scratch to simulate how seats really move round to round |

## One-line pitch (for summary sections / LinkedIn)

"A Python-only ML/NLP system that predicts IIT admission outcomes from JEE Advanced rank and reimplements the actual stable-matching algorithm behind India's national engineering admissions process — built solo, end to end, from raw data through a deployed app."
