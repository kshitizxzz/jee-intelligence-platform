from bisect import bisect_left
from collections import namedtuple
import pandas as pd

Match = namedtuple("Match", ["Year", "Round", "Institute", "Branch", "Opening_Rank", "Closing_Rank"])

class RankLookup:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._groups = {}
        self._build()

    def _build(self):
        key_cols = ["Year", "Round", "Seat_Type", "Gender"]
        for key, g in self.df.groupby(key_cols):
            g_sorted = g.sort_values("Closing_Rank")
            self._groups[key] = {
                "closing": g_sorted["Closing_Rank"].tolist(),
                "opening": g_sorted["Opening_Rank"].tolist(),
                "institute": g_sorted["Institute"].tolist(),
                "branch": g_sorted["Branch"].tolist(),
            }

    def eligible_at(self, rank, seat_type, gender, year, round_):
        key = (year, round_, seat_type, gender)
        bucket = self._groups.get(key)
        if not bucket:
            return []
        closing = bucket["closing"]
        idx = bisect_left(closing, rank)
        matches = []
        for i in range(idx, len(closing)):
            if bucket["opening"][i] <= rank <= bucket["closing"][i]:
                matches.append(Match(year, round_, bucket["institute"][i], bucket["branch"][i],
                                      bucket["opening"][i], bucket["closing"][i]))
        return matches

    def eligible_across_years(self, rank, seat_type, gender, final_round_only=True):
        results = []
        years_rounds = sorted({(y, r) for (y, r, s, g) in self._groups if s == seat_type and g == gender})
        if final_round_only:
            best_round = {}
            for y, r in years_rounds:
                best_round[y] = max(best_round.get(y, 0), r)
            years_rounds = list(best_round.items())
        for y, r in years_rounds:
            results.extend(self.eligible_at(rank, seat_type, gender, y, r))
        return sorted(results, key=lambda m: m.Closing_Rank)
