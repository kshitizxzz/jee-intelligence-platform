import pandas as pd
import streamlit as st

from src.utils.helpers import load_rank_lookup, SEAT_TYPES, GENDERS, format_rank, institute_short_name

st.set_page_config(page_title="Rank Band Explorer", page_icon="\U0001F50D", layout="wide")
st.title("Rank Band Explorer")
st.caption("What did a rank like yours actually get, across 9 real years of JoSAA data?")

lookup = load_rank_lookup()

col1, col2, col3 = st.columns(3)
rank = col1.number_input("JEE Advanced rank (AIR)", min_value=1, max_value=200000, value=2500, step=1)
seat_type = col2.selectbox("Seat type", SEAT_TYPES)
gender = col3.selectbox("Gender pool", GENDERS)

final_round_only = st.checkbox("Final round only (the seat you'd actually end up holding)", value=True)

matches = lookup.eligible_across_years(rank, seat_type, gender, final_round_only=final_round_only)

st.subheader(f"{len(matches)} historical seat(s) where rank {format_rank(rank)} fell within [Opening, Closing]")
st.caption("Found via binary search (`bisect_left`) over each year/round's sorted closing-rank array -- O(log n) per lookup, not a linear scan of 124,861 rows.")

if not matches:
    st.warning("No matches. Try a different Seat Type / Gender, or uncheck 'final round only'.")
else:
    df = pd.DataFrame(matches)
    df["Institute"] = df["Institute"].apply(institute_short_name)
    name_filter = st.text_input("Filter by institute or branch name contains...", "")
    if name_filter:
        mask = df["Institute"].str.contains(name_filter, case=False) | df["Branch"].str.contains(name_filter, case=False)
        df = df[mask]
    df["Opening_Rank"] = df["Opening_Rank"].apply(format_rank)
    df["Closing_Rank"] = df["Closing_Rank"].apply(format_rank)
    df = df.sort_values("Year", ascending=False)
    st.dataframe(
        df[["Year", "Round", "Institute", "Branch", "Opening_Rank", "Closing_Rank"]],
        width="stretch", hide_index=True, height=500,
    )
