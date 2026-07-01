"""Shared constants, cached loaders, and formatters used across every page."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import joblib

from src.algorithms.rank_lookup import RankLookup
from src.algorithms.trie import Trie
from src.nlp.branch_matcher import BranchMatcher

SEAT_TYPES = ["OPEN", "OPEN (PwD)", "EWS", "EWS (PwD)", "OBC-NCL", "OBC-NCL (PwD)",
              "SC", "SC (PwD)", "ST", "ST (PwD)"]
GENDERS = ["Gender-Neutral", "Female-only (including Supernumerary)"]
YEARS = list(range(2016, 2025))

DATA_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "src" / "models" / "artifacts"


def format_rank(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


@st.cache_data
def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "josaa_iit_consolidated.csv")


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "features.csv")


@st.cache_resource
def load_rank_lookup() -> RankLookup:
    return RankLookup(load_raw())


@st.cache_resource
def load_institute_trie() -> Trie:
    """Trie keyed on the short display name (e.g. 'IIT Bombay') so that typing the
    short form actually matches -- the raw data stores full names like
    'Indian Institute of Technology Bombay', which users don't type."""
    return Trie().build([institute_short_name(x) for x in load_raw()["Institute"].unique().tolist()])


@st.cache_resource
def load_institute_lookup() -> dict:
    """Map short display name -> full raw institute name used for filtering the data."""
    return {institute_short_name(x): x for x in load_raw()["Institute"].unique().tolist()}


@st.cache_resource
def load_branch_trie() -> Trie:
    return Trie().build(load_raw()["Branch"].unique().tolist())


@st.cache_resource
def load_branch_matcher() -> BranchMatcher:
    return BranchMatcher().fit(load_raw()["Branch"].unique().tolist())


@st.cache_resource
def load_ranker_artifacts():
    models = joblib.load(ARTIFACTS_DIR / "ranker_models.joblib")
    encoder = joblib.load(ARTIFACTS_DIR / "ranker_encoder.joblib")
    reference = pd.read_csv(ARTIFACTS_DIR / "ranker_reference_table.csv")
    metrics = joblib.load(ARTIFACTS_DIR / "ranker_metrics.joblib")
    return models, encoder, reference, metrics


@st.cache_data
def load_forecaster_table() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "forecaster_table.csv")


def institute_short_name(full_name: str) -> str:
    """'Indian Institute of Technology Bombay' -> 'IIT Bombay'."""
    return full_name.replace("Indian Institute of Technology", "IIT").strip()
