from dataclasses import dataclass, field
import random

@dataclass
class Student:
    student_id: int
    rank: int
    preferences: list
    current_seat: object = None
    next_pref_idx: int = 0

class DeferredAcceptanceSimulator:
    def __init__(self, seat_capacity: dict):
        self.seat_capacity = seat_capacity
        self.holders = {sid: [] for sid in seat_capacity}

    def run(self, students, max_rounds=6):
        free = list(students)
        history = []
        for round_no in range(1, max_rounds + 1):
            if not free:
                break
            next_free = []
            for s in free:
                while s.next_pref_idx < len(s.preferences):
                    seat_id = s.preferences[s.next_pref_idx]
                    if seat_id in self.seat_capacity:
                        break
                    s.next_pref_idx += 1
                else:
                    continue
                seat_id = s.preferences[s.next_pref_idx]
                self.holders[seat_id].append(s)
                s.current_seat = seat_id

            for seat_id, holders in self.holders.items():
                holders.sort(key=lambda st: st.rank)
                cap = self.seat_capacity[seat_id]
                kept, rejected = holders[:cap], holders[cap:]
                self.holders[seat_id] = kept
                for r in rejected:
                    r.current_seat = None
                    r.next_pref_idx += 1
                    next_free.append(r)

            history.append({
                "round": round_no,
                "allocated": {sid: [st.student_id for st in hs] for sid, hs in self.holders.items()},
            })
            free = [s for s in next_free if s.next_pref_idx < len(s.preferences)]

        final_allocation = {s.student_id: s.current_seat for s in students}
        return history, final_allocation

def build_synthetic_cohort(combo_table, n_students=300, n_prefs=8, seed=42):
    rng = random.Random(seed)
    seats = {
        c["seat_id"]: max(1, round((c["closing_rank"] - c["opening_rank"]) / 50))
        for c in combo_table
    }
    students = []
    for sid in range(n_students):
        rank = rng.randint(1, max(c["closing_rank"] for c in combo_table))
        noisy = sorted(combo_table, key=lambda c: -c["desirability"] + rng.uniform(-0.15, 0.15))
        prefs = [c["seat_id"] for c in noisy[:n_prefs]]
        students.append(Student(student_id=sid, rank=rank, preferences=prefs))
    return students, seats

def historical_round_trend(df, institute, branch, seat_type, gender, year):
    sub = df[(df.Institute == institute) & (df.Branch == branch) &
              (df.Seat_Type == seat_type) & (df.Gender == gender) & (df.Year == year)]
    sub = sub.sort_values("Round")
    if sub.empty:
        return None
    rounds = sub["Round"].tolist()
    closing = sub["Closing_Rank"].tolist()
    loosened = closing[-1] > closing[0]
    pct_change = (closing[-1] - closing[0]) / closing[0] * 100 if closing[0] else 0
    if len(closing) == 1:
        advice = "Only one round recorded that year -- not enough history to advise."
    elif loosened:
        advice = ("Historically this seat's closing rank loosened round over round "
                  "(cutoff relaxed as higher-preference students floated away) -- "
                  "floating/waiting tended to pay off for borderline ranks.")
    else:
        advice = ("Historically this seat's closing rank held steady or tightened "
                  "round over round -- freezing early tended to be the safer choice.")
    return {
        "rounds": rounds, "closing_ranks": closing,
        "pct_change": round(pct_change, 1), "advice": advice,
    }
