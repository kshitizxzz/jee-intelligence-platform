import numpy as np
import pandas as pd
import streamlit as st

from src.utils.helpers import (
    load_ranker_artifacts, SEAT_TYPES, GENDERS, format_rank, institute_short_name
)

st.set_page_config(page_title="Choice List Builder", page_icon="\U0001F4CB", layout="wide")
st.title("Choice List Builder")
st.caption("Eligibility x desirability, ranked and explained -- not a single cutoff lookup.")

models, encoder, reference, metrics = load_ranker_artifacts()

with st.sidebar:
    st.subheader("Your details")
    rank = st.number_input("JEE Advanced rank (AIR)", min_value=1, max_value=200000, value=2500, step=1)
    seat_type = st.selectbox("Seat type", SEAT_TYPES, index=0)
    gender = st.selectbox("Gender pool", GENDERS, index=0)
    st.divider()
    st.subheader("What matters more to you?")
    slider = st.slider("Institute prestige (NIRF)  <--->  Branch demand", 0.0, 1.0, 0.5, 0.05)
    nirf_weight, demand_weight = slider, 1 - slider
    st.caption(f"Blend: {nirf_weight:.0%} prestige, {demand_weight:.0%} branch demand")
    with st.expander("Model accuracy (held-out test set)"):
        for name, m in metrics.items():
            label = name.replace("_", " ")
            st.write(f"**{label}**: accuracy {m['accuracy']:.1%}, F1 {m['f1']:.1%}")
        st.caption(
            "The neural net benchmark scores higher on raw accuracy, but production uses the classical "
            "ensemble for interpretability -- a high-stakes admissions tool should be debuggable, not just accurate."
        )

production_models = {k: v for k, v in models.items() if k != "neural_net_mlp_benchmark"}

pool = reference[(reference["Seat_Type"] == seat_type) & (reference["Gender"] == gender)].copy()

if pool.empty:
    st.warning("No historical data for this Seat Type x Gender combination.")
    st.stop()

rank_norm = np.clip(rank / pool["Pool_Max_Closing_Rank"], 0, 3).to_numpy().reshape(-1, 1)
numeric = pool[["Demand_Score", "NIRF_Score"]].to_numpy()
cat = encoder.transform(pool[["Seat_Type", "Gender"]])
X = np.hstack([rank_norm, numeric, cat])

proba_sum = np.zeros(len(pool))
for model in production_models.values():
    proba_sum += model.predict_proba(X)[:, 1]
pool["Eligibility_Prob"] = proba_sum / len(production_models)
pool["Final_Desirability"] = nirf_weight * pool["NIRF_Score"] + demand_weight * pool["Demand_Score"]
pool["Institute_Short"] = pool["Institute"].apply(institute_short_name)

safe = pool[pool["Eligibility_Prob"] >= 0.5].sort_values("Final_Desirability", ascending=False)
reach = pool[(pool["Eligibility_Prob"] >= 0.15) & (pool["Eligibility_Prob"] < 0.5)].sort_values(
    "Final_Desirability", ascending=False
)

st.subheader(f"Ranked choices for rank {format_rank(rank)} -- {seat_type} / {gender}")
st.caption(f"{len(safe)} combo(s) the model scores as likely eligible (>=50% probability), out of {len(pool)} historical combos in this pool.")

display_cols = {
    "Institute_Short": "Institute", "Branch": "Branch",
    "Eligibility_Prob": "Eligibility", "Final_Desirability": "Desirability",
    "Year": "As of year", "Opening_Rank": "Last opening rank", "Closing_Rank": "Last closing rank",
}

if safe.empty:
    st.warning("No combo cleared the 50% eligibility bar for this rank. Check the reach list below, "
               "or try a less restrictive Seat Type / Gender pool.")
else:
    top = safe.head(30).copy()
    top["Eligibility_Prob"] = (top["Eligibility_Prob"] * 100).round(1).astype(str) + "%"
    top["Final_Desirability"] = top["Final_Desirability"].round(3)
    top["Opening_Rank"] = top["Opening_Rank"].apply(format_rank)
    top["Closing_Rank"] = top["Closing_Rank"].apply(format_rank)
    st.dataframe(
        top[list(display_cols)].rename(columns=display_cols),
        width="stretch", hide_index=True,
    )

with st.expander(f"Reach options ({len(reach)}) -- eligibility 15-50%, the kind of seats Float/Slide might land you"):
    if reach.empty:
        st.write("None in this pool.")
    else:
        r = reach.head(20).copy()
        r["Eligibility_Prob"] = (r["Eligibility_Prob"] * 100).round(1).astype(str) + "%"
        r["Final_Desirability"] = r["Final_Desirability"].round(3)
        r["Opening_Rank"] = r["Opening_Rank"].apply(format_rank)
        r["Closing_Rank"] = r["Closing_Rank"].apply(format_rank)
        st.dataframe(r[list(display_cols)].rename(columns=display_cols), width="stretch", hide_index=True)

st.caption(
    "Eligibility is predicted by a 4-model ensemble (KNN, Decision Tree, Random Forest, AdaBoost) trained on "
    "synthetic samples drawn around each combo's real historical Opening/Closing Rank interval -- it is not a "
    "raw lookup of last year's cutoff. Desirability blends NIRF Engineering Rank 2025 with how competitive the "
    "branch has historically been in this exact Seat Type x Gender pool."
)
