"""
Consolidation script that produced data/processed/josaa_iit_consolidated.csv.

Source data is NOT bundled in this repo: it comes from three separate public
GitHub scrape repos (seshaljain/josaa-scrape for 2016-2020, dvishal485/jossa-cutoff-2021
for 2021, ksauraj/jee_counsellor for 2022-2024), each with its own license, file
format, and column naming. Re-running this script requires cloning those repos'
raw files into data/raw/<source_name>/ yourself; the processed CSV in this repo
is the checked-in, already-verified build artifact (see README_dataset.md for
the verification notes: spot-checked cutoffs, zero duplicate keys, zero nulls).
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_CSV = ROOT / "data" / "processed" / "josaa_iit_consolidated.csv"

COLUMN_ALIASES = {
    "institute": "Institute", "college": "Institute",
    "branch": "Branch", "program": "Branch", "academic program name": "Branch",
    "quota": "Quota",
    "seat type": "Seat_Type", "category": "Seat_Type",
    "gender": "Gender",
    "opening rank": "Opening_Rank", "or": "Opening_Rank",
    "closing rank": "Closing_Rank", "cr": "Closing_Rank",
    "round": "Round", "year": "Year",
}

IIT_NAME_PATTERN = re.compile(r"\bIndian Institute of Technology\b", re.IGNORECASE)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: COLUMN_ALIASES[c.strip().lower()] for c in df.columns if c.strip().lower() in COLUMN_ALIASES}
    return df.rename(columns=rename)


def parse_rank(raw_value):
    s = str(raw_value).strip()
    is_prep = s.upper().endswith("P")
    numeric = s[:-1] if is_prep else s
    try:
        return int(float(numeric)), is_prep
    except ValueError:
        return None, False


def load_source(path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    df = normalize_columns(df)
    df = df[df["Institute"].astype(str).str.contains(IIT_NAME_PATTERN)].copy()

    for col, raw_col in [("Opening_Rank", "Opening_Rank"), ("Closing_Rank", "Closing_Rank")]:
        parsed = df[raw_col].apply(parse_rank)
        df[f"{col}_Raw"] = df[raw_col]
        df[col] = parsed.apply(lambda t: t[0])
        if col == "Closing_Rank":
            df["Is_Preparatory"] = parsed.apply(lambda t: t[1])

    df["Quota"] = "AI"
    df["Source"] = source_name
    df = df.dropna(subset=["Opening_Rank", "Closing_Rank"])
    keep = ["Year", "Round", "Institute", "Branch", "Quota", "Seat_Type", "Gender",
            "Opening_Rank", "Closing_Rank", "Is_Preparatory",
            "Opening_Rank_Raw", "Closing_Rank_Raw", "Source"]
    return df[keep]


def build_and_save(sources: dict) -> pd.DataFrame:
    frames = [load_source(Path(p), name) for name, paths in sources.items() for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["Year", "Round", "Institute", "Branch", "Quota", "Seat_Type", "Gender"]
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    return combined


if __name__ == "__main__":
    sources = {
        "josaa-scrape-2016-2020": sorted((RAW_DIR / "josaa-scrape").glob("**/*.csv")),
        "jossa-cutoff-2021": sorted((RAW_DIR / "jossa-cutoff-2021").glob("**/*.csv")),
        "jee_counsellor-2022-2024": sorted((RAW_DIR / "jee_counsellor").glob("**/*.csv")),
    }
    if not any(sources.values()):
        print("data/raw/ is empty -- this script needs the raw source repos cloned in to run. "
              "The processed CSV already checked into this repo is the verified build output.")
    else:
        out = build_and_save(sources)
        print("wrote", OUT_CSV, out.shape)
