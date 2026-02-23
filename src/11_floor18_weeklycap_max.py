
import pandas as pd
from pathlib import Path
from ortools.sat.python import cp_model

OUTPUT_DIR = Path("outputs")

FLOOR = 18
CEIL = 20

MAX_PER_WEEK = 2  # 🔥 key improvement

TIME_LIMIT = 240
WORKERS = 8

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
    slots["week"] = slots["week"].astype(int)
    slots["day"] = slots["day"].astype(str).str.strip()
    slots["room_capacity"] = slots["room_capacity"].astype(int)

    section_ids = sections["section_id"].unique().tolist()
    slot_ids = slots["slot_id"].unique().tolist()
    weeks = sorted(slots["week"].unique().tolist())

    # section -> course
    sec_to_course = dict(zip(sections["section_id"], sections["course_id"]))

    # faculty column detect
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

    # slots by week
    slots_by_week = {w: slots.loc[slots["week"] == w, "slot_id"].tolist() for w in weeks}

    # -------- Model --------
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

    # Per-section sessions with floor/ceiling
    sec_sessions = {}
    for sec in section_ids:
        tot = model.NewIntVar(0, len(slot_ids), f"sessions_{sec}")
        model.Add(tot == sum(x[(sec, sl)] for sl in slot_ids))
        model.Add(tot >= FLOOR)
        model.Add(tot <= CEIL)
        sec_sessions[sec] = tot

    # 🔥 Weekly cap: at most 2 sessions per week per section
    for sec in section_ids:
        for w in weeks:
            model.Add(sum(x[(sec, sl)] for sl in slots_by_week[w]) <= MAX_PER_WEEK)

    # Objective: maximize total sessions (after guaranteeing floor)
    total_sessions = model.NewIntVar(0, len(section_ids) * CEIL, "total_sessions")
    model.Add(total_sessions == sum(sec_sessions[sec] for sec in section_ids))
    model.Maximize(total_sessions)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT
    solver.parameters.num_search_workers = WORKERS

    print("\nSolving (floor=18, cap=20, weekly<=2, maximize total)...")
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("\n❌ No solution found.")
        print("Status:", solver.StatusName(status))
        return

    print("\n✅ Solution found.")
    print("Status:", solver.StatusName(status))

    total_val = solver.Value(total_sessions)
    print("Total scheduled sessions:", total_val)
    print("Average per section:", round(total_val / len(section_ids), 2))

    # Distribution
    dist = {18: 0, 19: 0, 20: 0}
    for sec in section_ids:
        v = solver.Value(sec_sessions[sec])
        if v in dist:
            dist[v] += 1
    print("\nSection session distribution:")
    for k in sorted(dist):
        print(f"{k} sessions:", dist[k])

    # Save summary
    out_rows = []
    for sec in section_ids:
        out_rows.append({
            "section_id": sec,
            "course_id": sec_to_course[sec],
            "sessions_scheduled": solver.Value(sec_sessions[sec]),
        })
    summary_df = pd.DataFrame(out_rows).sort_values(["sessions_scheduled", "course_id", "section_id"], ascending=[False, True, True])
    out_path = OUTPUT_DIR / "section_sessions_floor18_weeklycap_max.csv"
    summary_df.to_csv(out_path, index=False)
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()
