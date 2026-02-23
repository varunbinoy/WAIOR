import pandas as pd
from pathlib import Path
from ortools.sat.python import cp_model

OUTPUT_DIR = Path("outputs")
CAP = 20
TIME_LIMIT = 240
WORKERS = 8

def main():
    sections = pd.read_csv(OUTPUT_DIR / "sections.csv")
    enroll = pd.read_csv(OUTPUT_DIR / "section_enrollments.csv")
    courses = pd.read_csv(OUTPUT_DIR / "courses.csv")
    slots = pd.read_csv(OUTPUT_DIR / "slots.csv")

    sections["section_id"] = sections["section_id"].astype(str).str.strip()
    sections["course_id"] = sections["course_id"].astype(str).str.strip()
    enroll["section_id"] = enroll["section_id"].astype(str).str.strip()
    enroll["student_id"] = enroll["student_id"].astype(str).str.strip()
    slots["slot_id"] = slots["slot_id"].astype(str).str.strip()
    slots["room_capacity"] = slots["room_capacity"].astype(int)

    section_ids = sections["section_id"].unique().tolist()
    slot_ids = slots["slot_id"].unique().tolist()
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

    student_to_sections = (
        enroll.groupby("student_id")["section_id"]
        .apply(list)
        .to_dict()
    )

    faculty_to_sections = {}
    for sec in section_ids:
        fac = course_to_faculty[sec_to_course[sec]]
        faculty_to_sections.setdefault(fac, []).append(sec)

    slot_capacity = dict(zip(slots["slot_id"], slots["room_capacity"]))

    model = cp_model.CpModel()
    x = {(sec, sl): model.NewBoolVar(f"x_{sec}_{sl}") for sec in section_ids for sl in slot_ids}

    # constraints
    for sl in slot_ids:
        model.Add(sum(x[(sec, sl)] for sec in section_ids) <= int(slot_capacity[sl]))

    for fac, secs in faculty_to_sections.items():
        for sl in slot_ids:
            model.Add(sum(x[(sec, sl)] for sec in secs) <= 1)

    for student, secs in student_to_sections.items():
        if len(secs) <= 1:
            continue
        for sl in slot_ids:
            model.Add(sum(x[(sec, sl)] for sec in secs) <= 1)

    # cap 20 per section + count vars
    sec_sessions = {}
    for sec in section_ids:
        tot = model.NewIntVar(0, CAP, f"sessions_{sec}")
        model.Add(tot == sum(x[(sec, sl)] for sl in slot_ids))
        model.Add(tot <= CAP)
        sec_sessions[sec] = tot

    total_sessions = model.NewIntVar(0, len(section_ids) * CAP, "total_sessions")
    model.Add(total_sessions == sum(sec_sessions.values()))
    model.Maximize(total_sessions)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT
    solver.parameters.num_search_workers = WORKERS

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("No solution.")
        return

    total_val = solver.Value(total_sessions)
    print("\n✅", solver.StatusName(status))
    print("Max total sessions:", total_val, "/ 940")
    print("Gap:", 940 - total_val)

    # Build summary
    rows = []
    for sec in section_ids:
        rows.append({
            "section_id": sec,
            "course_id": sec_to_course[sec],
            "faculty": course_to_faculty[sec_to_course[sec]],
            "sessions": solver.Value(sec_sessions[sec])
        })
    df = pd.DataFrame(rows)

    print("\nDistribution (sessions : #sections):")
    dist = df["sessions"].value_counts().sort_index()
    print(dist.to_string())

    print("\nWorst 15 sections (lowest sessions):")
    print(df.sort_values("sessions").head(15).to_string(index=False))

    out_path = OUTPUT_DIR / "section_sessions_max900_detail.csv"
    df.sort_values(["sessions","course_id","section_id"]).to_csv(out_path, index=False)
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()
