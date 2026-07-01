import plotly.graph_objects as go
import streamlit as st

from src.models.train_forecaster import final_round_series
from src.utils.helpers import (
    load_forecaster_table, load_institute_trie, SEAT_TYPES, GENDERS,
    format_rank, institute_short_name, load_raw
)
from src.algorithms.trie import Trie

st.set_page_config(page_title="Trend Forecaster", page_icon="\U0001F4C8", layout="wide")
st.title("Trend Forecaster")
st.caption("Per-branch closing-rank trend, projected one year ahead with a confidence band.")


@st.cache_data
def cached_series():
    return final_round_series()


raw = load_raw()
forecast = load_forecaster_table()
institute_trie = load_institute_trie()

prefix = st.text_input("Search institute (autocomplete via Trie)", value="IIT Madras")
matches = institute_trie.autocomplete(prefix.lower(), limit=15) if prefix else sorted(raw["Institute"].unique())
institute = st.selectbox("Institute", matches or sorted(raw["Institute"].unique()), format_func=institute_short_name)

branch_options = sorted(raw.loc[raw["Institute"] == institute, "Branch"].unique())
branch_trie = Trie().build(branch_options)
branch_prefix = st.text_input("Search branch (autocomplete via Trie)", value="")
branch_matches = branch_trie.autocomplete(branch_prefix.lower(), limit=20) if branch_prefix else branch_options
branch = st.selectbox("Branch", branch_matches or branch_options)

c1, c2 = st.columns(2)
seat_type = c1.selectbox("Seat type", SEAT_TYPES)
gender = c2.selectbox("Gender pool", GENDERS)

row = forecast[
    (forecast["Institute"] == institute) & (forecast["Branch"] == branch)
    & (forecast["Seat_Type"] == seat_type) & (forecast["Gender"] == gender)
]

if row.empty:
    st.warning("No forecast available for this exact combination.")
else:
    r = row.iloc[0]
    series = cached_series()
    hist = series[
        (series["Institute"] == institute) & (series["Branch"] == branch)
        & (series["Seat_Type"] == seat_type) & (series["Gender"] == gender)
    ].sort_values("Year")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Year"], y=hist["Closing_Rank"], mode="lines+markers",
                              name="Historical closing rank", line=dict(color="#D85A30", width=3)))
    fig.add_trace(go.Scatter(
        x=[r["next_year"]], y=[r["predicted_closing_rank"]], mode="markers",
        name=f"{r['next_year']} projection", marker=dict(color="#1f77b4", size=12, symbol="diamond"),
        error_y=dict(type="data", symmetric=False,
                      array=[r["upper_band"] - r["predicted_closing_rank"]],
                      arrayminus=[r["predicted_closing_rank"] - r["lower_band"]]),
    ))
    fig.update_layout(xaxis_title="Year", yaxis_title="Closing rank", height=420, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric(f"{int(r['last_year'])} closing rank", format_rank(r["last_closing_rank"]))
    col2.metric(f"{int(r['next_year'])} projected", format_rank(r["predicted_closing_rank"]))
    col3.metric("Projected band", f"{format_rank(r['lower_band'])} - {format_rank(r['upper_band'])}")

    if r["method"] == "carry_forward":
        st.info(
            f"Only {int(r['n_years'])} year(s) of history for this exact combo -- not enough points to fit "
            "a trend line honestly. This is last year's value carried forward with a flat +/-20% band.",
            icon="⚠️",
        )
    else:
        st.info(
            f"Linear regression fit on {int(r['n_years'])} years (R²={r['r_squared']:.2f}). "
            f"Slope: {r['slope_per_year']:+.0f} ranks/year.",
            icon="\U0001F4C8",
        )
