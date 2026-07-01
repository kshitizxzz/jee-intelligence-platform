# JoSAA IIT Opening/Closing Rank Dataset (2016-2024)

`josaa_iit_consolidated.csv` — 124,861 rows. All 23 IITs, every round, every branch, every Seat Type x Gender pool, 2016-2024 (9 years; see note on 2025 below).

## Columns

- **Year** — admission year (2016-2024)
- **Round** — counselling round number (rounds per year vary: 5-7)
- **Institute** — full IIT name (23 unique values)
- **Branch** — academic program name, includes degree type/duration
- **Quota** — always "AI" (All-India). IITs have no home-state quota, unlike NITs/IIITs.
- **Seat_Type** — OPEN, EWS, OBC-NCL, SC, ST, and their "(PwD)" variants (10 values)
- **Gender** — "Gender-Neutral", "Female-only (including Supernumerary)", or "Common (pre-2018, no Female-only pool)" for 2016-2017, before IITs introduced the female supernumerary scheme
- **Opening_Rank** / **Closing_Rank** — numeric JEE Advanced AIR for that combo
- **Is_Preparatory** — True if the original rank had a trailing "P" (Preparatory-course rank track, mostly SC/ST), 7,531 rows
- **Opening_Rank_Raw** / **Closing_Rank_Raw** — original string values before parsing (kept for audit)
- **Source** — which underlying repo the row came from

## Sources (all pulled from public GitHub repos, IIT-only filtered)

- 2016-2020: seshaljain/josaa-scrape
- 2021: dvishal485/jossa-cutoff-2021
- 2022-2024: ksauraj/jee_counsellor

## Verified

- Spot-checked IIT Bombay CSE 2024 Round 5 OPEN/Gender-Neutral = (1, 68) and IIT Madras CSE 2023 Round 6 OPEN/Gender-Neutral = (42, 148) — both match known public cutoffs.
- Zero duplicate (Year, Round, Institute, Branch, Quota, Seat_Type, Gender) keys.
- Zero null Opening/Closing rank values.

## Known gap: no 2025 data

JoSAA's official archive (josaa.admissions.nic.in) only exposes data through an interactive ASP.NET dropdown form with no bulk/static export — not scrapable with a simple fetch. A separate snapshot that looked like it might be 2025 (Sbrjt/josaa-cutoffs) was checked against git history and turned out to be 2024 data re-published, not a new year. So this dataset covers **9 years (2016-2024)** rather than 10. Adding 2025 would require either a browser-automation pull of the official archive, or a manual export merged into the same schema.
