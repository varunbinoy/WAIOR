import pandas as pd
from pathlib import Path
from ortools.sat.python import cp_model

OUTPUT_DIR = Path("outputs")

FLOOR = 18
CAP = 20

TIME_LIMIT = 240
WORKERS = 8

DTI_COURSE = "DTI"
DTI_PREMID_FACULTY = "Prof. Rohit Kumar"
DTI_POSTMID_FACULTY = "Prof. Rogers"

def main():
    print("Loading data...")

    sections = pd.read_csv(OUTPUT_DIR / "sections.csv")
    enroll = pd.read_csv(OUTPUT_DIR / "section_enrollments.csv")
    courses = pd.read_csv(OUTPUT_DIR / "courses.csv")
    slots = pd.read_csv(OUTPUT_DIR / "slots.csv")

    # Normalize
    sections["section_id"] = sections["section_id"].astype(str).str.strip()
    sections["course_id"] = sections["course_id"].astype(str).str.strip()
    enroll["section_id"] = enroll["section_id"].astype(str).str.strip()
    enroll["student_id"] = enroll["student_id"].astype(str).str.strip()
    courses["course_id"] = courses["course_id"].astype(str).str.strip()
    courses["faculty_raw"] = courses["faculty_raw"].astype(str).str.strip()
    slots["slot_id"] = slots["slot_id"].astype(str).str.strip()
    slots["week"] = slots["week"].astype(int)

    # Room capacity column name handling
    if "room_capacity" in slots.columns:
        slots["room_capacity"] = slots["room_capacity"].astype(int)
        slot_capacity = dict(zip(slots["slot_id"], slots["room_capacity"]))
    elif "room_cap" in slots.columns:
        slots["room_cap"] = slots["room_cap"].astype(int)
        slot_capacity = dict(zip(slots["slot_id"], slots["room_cap"]))
    else:
        raise ValueError("slots.csv must have room_capacity or room_cap column")

    section_ids = sections["section_id"].unique().tolist()
    slot_ids = slots["slot_id"].unique().tolist()

    sec_to_course = dict(zip(sections["section_id"], sections["course_id"]))
    course_to_faculty_raw = dict(zip(courses["course_id"], courses["faculty_raw"]))
    slot_to_week = dict(zip(slots["slot_id"], slots["week"]))

    # student -> sections
    student_to_sections = (
        enroll.groupby("student_id")["section_id"]
        .apply(list)
        .to_dict()
    )

    def get_faculty(course_id: str, week: int) -> str:
        course_id = str(course_id).strip()
        if course_id == DTI_COURSE:
            return DTI_PREMID_FACULTY if week <= 5 else DTI_POSTMID_FACULTY
        return course_to_faculty_raw[course_id]

    print("\nBuilding model...")
    model = cp_model.CpModel()

    x = {(sec, sl): model.NewBoolVar(f"x_{sec}_{sl}") for sec in section_ids for sl in slot_ids}

    # Room capacity
    for sl in slot_ids:
        model.Add(sum(x[(sec, sl)] for sec in section_ids) <= int(slot_capacity[sl]))

    # Student conflict
    for student, secs in student_to_sections.items():
        if len(secs) <= 1:
            continue
        for sl in slot_ids:
            model.Add(sum(x[(sec, sl)] for sec in secs) <= 1)

    # Faculty conflict (DTI split)
    for sl in slot_ids:
        week = int(slot_to_week[sl])
        faculty_groups = {}
        for sec in section_ids:
            cid = sec_to_course[sec]
            fac = get_faculty(cid, week)
            faculty_groups.setdefault(fac, []).append(sec)
        for fac, secs in faculty_groups.items():
            model.Add(sum(x[(sec, sl)] for sec in secs) <= 1)

    # Per-section floor/cap
    sec_sessions = {}
    for sec in section_ids:
        tot = model.NewIntVar(0, len(slot_ids), f"sessions_{sec}")
        model.Add(tot == sum(x[(sec, sl)] for sl in slot_ids))
        model.Add(tot >= FLOOR)
        model.Add(tot <= CAP)
        sec_sessions[sec] = tot

    total_sessions = model.NewIntVar(0, len(section_ids) * CAP, "total_sessions")
    model.Add(total_sessions == sum(sec_sessions.values()))
    model.Maximize(total_sessions)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT
    solver.parameters.num_search_workers = WORKERS

    print(f"\nSolving (floor={FLOOR}, cap={CAP}, maximize total, DTI split ON)...")
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("\n❌ No solution.")
        print("Status:", solver.StatusName(status))
        return

    print("\n✅", solver.StatusName(status))
    tot_val = solver.Value(total_sessions)
    print("Total scheduled sessions:", tot_val)
    print("Average per section:", round(tot_val / len(section_ids), 2))

    # Distribution
    dist = {18: 0, 19: 0, 20: 0}
    for sec in section_ids:
        v = solver.Value(sec_sessions[sec])
        if v in dist:
            dist[v] += 1
    print("\nSection session distribution:")
    for k in sorted(dist):
        print(f"{k} sessions:", dist[k])

    # Save per-section sessions
    rows = []
    for sec in section_ids:
        rows.append({
            "section_id": sec,
            "course_id": sec_to_course[sec],
            "sessions": solver.Value(sec_sessions[sec]),
        })
    out = pd.DataFrame(rows).sort_values(["sessions","course_id","section_id"], ascending=[True, True, True])
    out_path = OUTPUT_DIR / "section_sessions_floor18_cap20_facsplit.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
