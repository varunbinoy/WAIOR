import pandas as pd
from pathlib import Path
from ortools.sat.python import cp_model

OUT = Path("outputs")

FLOOR = 18
CAP = 20

TIME_LIMIT = 300
WORKERS = 8

DTI_COURSE = "DTI"
DTI_PREMID_FACULTY = "Prof. Rohit Kumar"
DTI_POSTMID_FACULTY = "Prof. Rogers"

def main():
    print("Loading data...")

    sections = pd.read_csv(OUT / "sections.csv")
    enroll = pd.read_csv(OUT / "section_enrollments.csv")
    courses = pd.read_csv(OUT / "courses.csv")
    slots = pd.read_csv(OUT / "slots.csv")

    # Normalize
    for df in [sections, enroll, courses, slots]:
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.strip()

    slots["week"] = slots["week"].astype(int)

    # room capacity column name
    if "room_cap" in slots.columns:
        slots["room_cap"] = slots["room_cap"].astype(int)
        slot_capacity = dict(zip(slots["slot_id"], slots["room_cap"]))
    elif "room_capacity" in slots.columns:
        slots["room_capacity"] = slots["room_capacity"].astype(int)
        slot_capacity = dict(zip(slots["slot_id"], slots["room_capacity"]))
    else:
        raise ValueError("slots.csv must contain room_cap or room_capacity")

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

    print("\nBuilding model (floor=18 cap=20 maximize total)...")
    model = cp_model.CpModel()

    x = {(sec, sl): model.NewBoolVar(f"x_{sec}_{sl}")
         for sec in section_ids for sl in slot_ids}

    # Room capacity per slot
    for sl in slot_ids:
        model.Add(sum(x[(sec, sl)] for sec in section_ids) <= int(slot_capacity[sl]))

    # Student conflict per slot
    for student, secs in student_to_sections.items():
        if len(secs) <= 1:
            continue
        for sl in slot_ids:
            model.Add(sum(x[(sec, sl)] for sec in secs) <= 1)

    # Faculty conflict per slot (DTI split week-aware)
    for sl in slot_ids:
        week = int(slot_to_week[sl])
        faculty_groups = {}
        for sec in section_ids:
            cid = sec_to_course[sec]
            fac = get_faculty(cid, week)
            faculty_groups.setdefault(fac, []).append(sec)
        for fac, secs in faculty_groups.items():
            model.Add(sum(x[(sec, sl)] for sec in secs) <= 1)

    # Floor/cap + objective
    sec_sessions = {}
    for sec in section_ids:
        tot = model.NewIntVar(0, CAP, f"sessions_{sec}")
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

    print("\nSolving...")
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("❌ No solution. Status:", solver.StatusName(status))
        return

    print("✅", solver.StatusName(status))
    print("Total scheduled sessions:", solver.Value(total_sessions),
          f"(avg {solver.Value(total_sessions)/len(section_ids):.2f})")

    # Export schedule rows
    rows = []
    for sec in section_ids:
        course_id = sec_to_course[sec]
        for sl in slot_ids:
            if solver.Value(x[(sec, sl)]) == 1:
                week = int(slot_to_week[sl])
                rows.append({
                    "slot_id": sl,
                    "week": week,
                    "section_id": sec,
                    "course_id": course_id,
                    "faculty": get_faculty(course_id, week),
                })

    sched = pd.DataFrame(rows).merge(
        slots[["slot_id", "day", "start", "end"]],
        on="slot_id",
        how="left"
    )

    # Assign room numbers within each slot (Room_1..Room_cap_used)
    sched["room_number"] = ""
    sched = sched.sort_values(["week", "day", "start", "section_id"]).reset_index(drop=True)

    for sl, grp in sched.groupby("slot_id"):
        idxs = grp.index.tolist()
        for i, idx in enumerate(idxs):
            sched.at[idx, "room_number"] = f"Room_{i+1}"

    out_path = OUT / "term_schedule_floor18.csv"
    sched.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # Also save per-section sessions (sanity)
    sess_rows = []
    for sec in section_ids:
        sess_rows.append({
            "section_id": sec,
            "course_id": sec_to_course[sec],
            "sessions": solver.Value(sec_sessions[sec]),
        })
    pd.DataFrame(sess_rows).to_csv(OUT / "term_section_sessions_floor18.csv", index=False)
    print("Saved:", OUT / "term_section_sessions_floor18.csv")

if __name__ == "__main__":
    main()
