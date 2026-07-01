import streamlit as st
from src.utils.helpers import load_raw

st.set_page_config(page_title="JEE Intelligence Platform", page_icon="\U0001F393", layout="wide")

st.title("JEE Intelligence Platform")
st.caption("IIT seat-allocation intelligence built on 9 years of real JoSAA cutoff data (2016-2024)")

df = load_raw()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Historical rows", f"{len(df):,}")
col2.metric("IITs covered", df["Institute"].nunique())
col3.metric("Years of data", f"{df['Year'].nunique()} (2016-2024)")
col4.metric("Branch x category combos", df.groupby(["Institute", "Branch", "Seat_Type", "Gender"]).ngroups)

st.divider()

st.markdown(
    """
This is **not** another rank predictor that interpolates last year's cutoffs. Five things are different:

1. **Choice List Builder** -- an ML ensemble (KNN + Decision Tree + Random Forest + AdaBoost) scores every
   institute-branch pair on eligibility, blended with a prestige-vs-branch-fit slider you control, into one
   ranked, explainable list. A small neural net is benchmarked against it too.
2. **Round Simulator** -- a from-scratch reimplementation of JoSAA's actual deferred-acceptance
   (Gale-Shapley) seat-allocation algorithm, plus real historical round-by-round movement, to ground
   Freeze / Float / Slide advice in how seats actually moved rather than a guess.
3. **Rank Band Explorer** and **Trend Forecaster** -- binary-search lookup over every historical seat, and a
   linear-regression projection of next year's likely cutoff band, per branch and category.
4. **Branch Matcher** -- describe what you're interested in, in plain English, and get matched to formal
   branch names via a lexicon-expanded text-similarity search -- no need to already know the jargon.

Use the sidebar to move between pages. All data is restricted to IITs (All-India quota only) --
no NITs, IIITs, or GFTIs.
"""
)

st.info(
    "Known gap: JoSAA does not publish a bulk export and 2025 cutoffs could not be scraped reliably, "
    "so this covers 2016-2024 (9 years) rather than 10. See README for details.",
    icon="ℹ️",
)
