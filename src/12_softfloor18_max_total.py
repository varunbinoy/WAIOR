import pandas as pd
from pathlib import Path
from ortools.sat.python import cp_model

OUTPUT_DIR = Path("outputs")

SOFT_FLOOR = 18     # target floor
CAP = 20            # hard cap
TIME_LIMIT = 240
WORKERS = 8

# Weighting:
# We want to push total sessions up, but also avoid dropping below 18.
# Bigger penalty => tries harder to keep >=18.
PENALTY_PER_MISSED_BELOW_18 = 5

def main():
    sections = pd.read_csv(OUTPUT_DIR / "sections.csv")
    enroll = pd.read_csv(OUTPUT_DIR / "section_enrollments.csv")
    courses = pd.read_csv(OUTPUT_DIR / "courses.csv")
    slots = pd.read_csv(OUTPUT_DIR / "slots.csv")

    # Normalize
    sections["section_id"] = sections["section_id"].astype(str).str.strip()
    sections["course_id"] = sections["course_id"].astype(str).str.strip()
    enroll["section_id"] = enroll["section_id"].astype(str).str.strip()
    enroll["student_id"] = enroll["student_id"].astype(str).str.strip()
    slots["slot_id"] = slots["slot_id"].astype(str).str.strip()
    slots["room_capacity"] = slots["room_capacity"].astype(int)

    section_ids = sections["section_id"].unique().tolist()
    slot_ids = slots["slot_id"].unique().tolist()

    sec_to_course = dict(zip(sections["section_id"], sections["course_id"]))

    # Faculty column detect
    faculty_col = None
    for col in ["faculty_raw", "faculty", "faculty_name"]:
        if col in courses.columns:
            faculty_col = col
            break
    if faculty_col is None:
        raise ValueError("No faculty column found in courses.csv")

    courses["course_id"] = courses["course_id"].astype(str).str.strip()
    courses[faculty_col] = courses[faculty_col].astype(str).str.strip()
    course_to_faculty = dict(zip(courses["course_id"], courses[faculty_col]))

    # student -> sections
    student_to_sections = (
        enroll.groupby("student_id")["section_id"]
        .apply(list)
        .to_dict()
    )

    # faculty -> sections
    faculty_to_sections = {}
    for sec in section_ids:
        fac = course_to_faculty[sec_to_course[sec]]
        faculty_to_sections.setdefault(fac, []).append(sec)

    # slot -> capacity
    slot_capacity = dict(zip(slots["slot_id"], slots["room_capacity"]))

    model = cp_model.CpModel()

    x = {(sec, sl): model.NewBoolVar(f"x_{sec}_{sl}") for sec in section_ids for sl in slot_ids}

    # Room capacity
    for sl in slot_ids:
        model.Add(sum(x[(sec, sl)] for sec in section_ids) <= int(slot_capacity[sl]))

    # Faculty conflict
    for fac, secs in faculty_to_sections.items():
        for sl in slot_ids:
            model.Add(sum(x[(sec, sl)] for sec in secs) <= 1)

    # Student conflict
    for student, secs in student_to_sections.items():
        if len(secs) <= 1:
            continue
        for sl in slot_ids:
            model.Add(sum(x[(sec, sl)] for sec in secs) <= 1)

    # Per-section sessions (0..20), no hard floor
    sec_sessions = {}
    shortfall = {}  # max(0, 18 - sessions)
    for sec in section_ids:
        tot = model.NewIntVar(0, CAP, f"sessions_{sec}")
        model.Add(tot == sum(x[(sec, sl)] for sl in slot_ids))
        model.Add(tot <= CAP)
        sec_sessions[sec] = tot

        sf = model.NewIntVar(0, SOFT_FLOOR, f"shortfall_{sec}")
        # sf >= 18 - tot, and sf >= 0
        model.Add(sf >= SOFT_FLOOR - tot)
        model.Add(sf >= 0)
        shortfall[sec] = sf

    total_sessions = model.NewIntVar(0, len(section_ids) * CAP, "total_sessions")
    model.Add(total_sessions == sum(sec_sessions[sec] for sec in section_ids))

    total_shortfall = model.NewIntVar(0, len(section_ids) * SOFT_FLOOR, "total_shortfall")
    model.Add(total_shortfall == sum(shortfall[sec] for sec in section_ids))

    # Objective: maximize total_sessions - penalty * total_shortfall
    # This pushes toward 940 while trying to keep most sections >=18.
    model.Maximize(total_sessions - PENALTY_PER_MISSED_BELOW_18 * total_shortfall)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT
    solver.parameters.num_search_workers = WORKERS

    print("\nSolving (cap=20, softfloor=18, maximize total with penalty)...")
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("❌ No solution.")
        print("Status:", solver.StatusName(status))
        return

    print("\n✅ Solution found.")
    print("Status:", solver.StatusName(status))

    tot_val = solver.Value(total_sessions)
    sf_val = solver.Value(total_shortfall)

    print("Total scheduled sessions:", tot_val, " / 940 max")
    print("Total shortfall below 18:", sf_val)

    # Distribution
    dist = {}
    for sec in section_ids:
        v = solver.Value(sec_sessions[sec])
        dist[v] = dist.get(v, 0) + 1

    print("\nSession count distribution (sessions : #sections):")
    for k in sorted(dist.keys()):
        print(f"{k}: {dist[k]}")

    # Save summary
    out_rows = []
    for sec in section_ids:
        out_rows.append({
            "section_id": sec,
            "course_id": sec_to_course[sec],
            "sessions_scheduled": solver.Value(sec_sessions[sec]),
            "shortfall_below_18": solver.Value(shortfall[sec]),
        })
    out = pd.DataFrame(out_rows).sort_values(["sessions_scheduled", "course_id", "section_id"], ascending=[False, True, True])
    out_path = OUTPUT_DIR / "section_sessions_softfloor18_max.csv"
    out.to_csv(out_path, index=False)
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()

