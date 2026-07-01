import pandas as pd
import streamlit as st

from src.utils.helpers import (
    load_branch_matcher, load_features, SEAT_TYPES, GENDERS,
    format_rank, institute_short_name,
)

st.set_page_config(page_title="Branch Matcher", page_icon="\U0001F9E9", layout="wide")
st.title("Branch Matcher")
st.caption(
    "Don't know what 'Engineering Physics' or 'Mechatronics' actually means? Describe what you're "
    "interested in, in plain English, and get matched to formal JEE branch names."
)
st.caption(
    "Technique: a hand-built topic lexicon expands each formal branch name into plain-language synonyms "
    "and concepts, then TF-IDF + cosine similarity ranks branches against your free-text query. "
    "Scikit-learn only -- no embedding models, so it stays light enough for Streamlit Cloud."
)

matcher = load_branch_matcher()
feats = load_features()

query = st.text_area(
    "Describe your interests",
    placeholder="e.g. I enjoy coding, building apps, and want to work with AI and data",
    height=90,
)
top_k = st.slider("Number of matches", 3, 15, 8)

if query.strip():
    matches = matcher.match(query, top_k=top_k)
    if not matches:
        st.warning("No branch matched that description well enough. Try adding more detail or different words.")
    else:
        st.subheader(f"Top {len(matches)} branch matches")

        with st.expander("Filter matched branches by your rank / seat type / gender (optional)"):
            c1, c2, c3 = st.columns(3)
            use_filter = c1.checkbox("Apply eligibility filter", value=False)
            rank = c2.number_input("JEE Advanced rank (AIR)", min_value=1, max_value=200000, value=2500, step=1)
            seat_type = c3.selectbox("Seat type", SEAT_TYPES)
            gender = st.selectbox("Gender pool", GENDERS)

        for m in matches:
            with st.container(border=True):
                top_row = st.columns([3, 1])
                top_row[0].markdown(f"### {m.branch}")
                top_row[1].metric("Match score", f"{m.score:.2f}")
                if m.topics:
                    st.caption("Matched topics: " + ", ".join(sorted(m.topics)))

                branch_rows = feats[feats["Branch"] == m.branch].drop_duplicates(
                    subset=["Institute", "Seat_Type", "Gender"]
                )
                if use_filter:
                    branch_rows = branch_rows[
                        (branch_rows["Seat_Type"] == seat_type) & (branch_rows["Gender"] == gender)
                        & (branch_rows["Closing_Rank"] >= rank)
                    ]
                if branch_rows.empty:
                    st.caption("No offering institutes match your current filter." if use_filter
                               else "No institute data available for this branch.")
                else:
                    show = branch_rows.copy()
                    show["Institute"] = show["Institute"].apply(institute_short_name)
                    show["Closing_Rank"] = show["Closing_Rank"].apply(format_rank)
                    show = show.sort_values("Closing_Rank", key=lambda s: s.str.replace(",", "").astype(int))
                    st.dataframe(
                        show[["Institute", "Seat_Type", "Gender", "Year", "Closing_Rank"]].head(10),
                        width="stretch", hide_index=True,
                    )
else:
    st.info("Type a description above to see matches.", icon="\U0001F446")
