
import pandas as pd
from pathlib import Path
from ortools.sat.python import cp_model

OUTPUT_DIR = Path("outputs")

TIME_LIMIT = 60
WORKERS = 8


def main():
    print("\nLoading data...")

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

    print("Sections:", len(section_ids))
    print("Slots:", len(slot_ids))

    sec_to_course = dict(zip(sections["section_id"], sections["course_id"]))

    # Faculty column detect
    faculty_col = None
    for col in ["faculty_raw", "faculty", "faculty_name"]:
        if col in courses.columns:
            faculty_col = col
            break
    if faculty_col is None:
        raise ValueError("No faculty column found")

    courses["course_id"] = courses["course_id"].astype(str).str.strip()
    courses[faculty_col] = courses[faculty_col].astype(str).str.strip()

    course_to_faculty = dict(zip(courses["course_id"], courses[faculty_col]))

    # Student → sections
    student_to_sections = (
        enroll.groupby("student_id")["section_id"]
        .apply(list)
        .to_dict()
    )

    # Faculty → sections
    faculty_to_sections = {}
    for sec in section_ids:
        cid = sec_to_course[sec]
        fac = course_to_faculty[cid]
        faculty_to_sections.setdefault(fac, []).append(sec)

    slot_capacity = dict(zip(slots["slot_id"], slots["room_capacity"]))

    print("\nBuilding model...")
    model = cp_model.CpModel()

    x = {}
    for sec in section_ids:
        for sl in slot_ids:
            x[(sec, sl)] = model.NewBoolVar(f"x_{sec}_{sl}")

    # Room capacity
    for sl in slot_ids:
        cap = int(slot_capacity[sl])
        model.Add(sum(x[(sec, sl)] for sec in section_ids) <= cap)

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

    # Maximize total sessions
    total_sessions = sum(x[(sec, sl)] for sec in section_ids for sl in slot_ids)
    model.Maximize(total_sessions)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT
    solver.parameters.num_search_workers = WORKERS

    print("\nSolving (maximizing total sessions)...")
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("\n❌ Could not solve")
        return

    max_sessions = solver.Value(total_sessions)

    print("\n✅ MAX TOTAL SESSIONS:", max_sessions)
    print("Average per section:", round(max_sessions / len(section_ids), 2))

    print("\nTop 10 section session counts:")
    for sec in section_ids[:10]:
        count = sum(solver.Value(x[(sec, sl)]) for sl in slot_ids)
        print(sec, "→", count)


if __name__ == "__main__":
    main()
