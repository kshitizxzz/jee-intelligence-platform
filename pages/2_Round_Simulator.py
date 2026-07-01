import plotly.graph_objects as go
import streamlit as st

from src.algorithms.deferred_acceptance import (
    DeferredAcceptanceSimulator, build_synthetic_cohort, historical_round_trend
)
from src.algorithms.trie import Trie
from src.utils.helpers import (
    load_raw, load_features, load_institute_trie, SEAT_TYPES, GENDERS,
    format_rank, institute_short_name, YEARS
)

st.set_page_config(page_title="Round Simulator", page_icon="\U0001F501", layout="wide")
st.title("Round Simulator -- Freeze / Float / Slide")
st.caption("Grounded in real round-by-round movement, plus a from-scratch JoSAA seat-allocation engine.")

raw = load_raw()
institute_trie = load_institute_trie()

st.subheader("1. Real historical round-by-round trend")

col_a, col_b = st.columns([2, 1])
with col_a:
    prefix = st.text_input("Search institute (autocomplete via Trie)", value="IIT Bombay")
matches = institute_trie.autocomplete(prefix.lower(), limit=15) if prefix else sorted(raw["Institute"].unique())
if not matches:
    matches = sorted(raw["Institute"].unique())
institute = st.selectbox("Institute", matches, format_func=institute_short_name)

branch_options = sorted(raw.loc[raw["Institute"] == institute, "Branch"].unique())
branch_trie = Trie().build(branch_options)
branch_prefix = st.text_input("Search branch (autocomplete via Trie)", value="")
branch_matches = branch_trie.autocomplete(branch_prefix.lower(), limit=20) if branch_prefix else branch_options
branch = st.selectbox("Branch", branch_matches or branch_options)

c1, c2, c3 = st.columns(3)
seat_type = c1.selectbox("Seat type", SEAT_TYPES)
gender = c2.selectbox("Gender pool", GENDERS)
year = c3.selectbox("Year", YEARS, index=len(YEARS) - 1)

trend = historical_round_trend(raw, institute, branch, seat_type, gender, year)

if trend is None:
    st.warning("No data for this exact combination/year. Try a different year or seat type.")
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["rounds"], y=trend["closing_ranks"], mode="lines+markers",
                              name="Closing rank", line=dict(color="#D85A30", width=3)))
    fig.update_layout(xaxis_title="Round", yaxis_title="Closing rank", height=350,
                       margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")
    st.metric("Change from round 1 to final round", f"{trend['pct_change']:+.1f}%")
    st.info(trend["advice"], icon="\U0001F4A1")

st.divider()
st.subheader("2. Synthetic deferred-acceptance demo")
st.caption(
    "JoSAA does not publish individual student preference lists, so this section runs the actual "
    "Gale-Shapley / deferred-acceptance algorithm (implemented from scratch, no library) over a "
    "synthetic cohort built from this institute's real combos -- to show *how* seats move round to round, "
    "not to predict your specific outcome."
)

feats = load_features()
inst_feats = feats[feats["Institute"] == institute].drop_duplicates(subset=["Branch", "Seat_Type", "Gender"])
combo_table = [
    {"seat_id": f"{r.Branch} | {r.Seat_Type} | {r.Gender}",
     "opening_rank": r.Opening_Rank, "closing_rank": r.Closing_Rank,
     "desirability": r.Desirability_Score}
    for r in inst_feats.itertuples()
]

n_students = st.slider("Synthetic cohort size", 50, 1000, 300, 50)
if st.button("Run simulation"):
    students, seats = build_synthetic_cohort(combo_table, n_students=n_students)
    sim = DeferredAcceptanceSimulator(seats)
    history, final_allocation = sim.run(students)

    placed = sum(1 for v in final_allocation.values() if v is not None)
    st.metric("Rounds to convergence", len(history))
    st.metric("Students placed", f"{placed} / {len(students)}")

    rows = []
    for h in history:
        for seat_id, ids in h["allocated"].items():
            rows.append({"Round": h["round"], "Seat": seat_id, "Holders": len(ids)})
    import pandas as pd
    hist_df = pd.DataFrame(rows)
    top_seats = hist_df.groupby("Seat")["Holders"].max().nlargest(8).index
    fig2 = go.Figure()
    for seat in top_seats:
        sub = hist_df[hist_df["Seat"] == seat]
        fig2.add_trace(go.Scatter(x=sub["Round"], y=sub["Holders"], mode="lines+markers", name=seat[:40]))
    fig2.update_layout(xaxis_title="Round", yaxis_title="Students holding seat", height=400,
                        legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig2, width="stretch")
