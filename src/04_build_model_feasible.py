
import pandas as pd
from pathlib import Path
from ortools.sat.python import cp_model

OUTPUT_DIR = Path("outputs")

MIN_SESSIONS = 19   # 🔥 changed from 20
TIME_LIMIT = 120
WORKERS = 8


def main():
    print("\nLoading data...")

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

    print("Sections:", len(section_ids))
    print("Slots:", len(slot_ids))

    # ---- Build mappings ----
    sec_to_course = dict(zip(sections["section_id"], sections["course_id"]))

    # Faculty mapping
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

    # Slot → capacity
    slot_capacity = dict(zip(slots["slot_id"], slots["room_capacity"]))

    # ---- Model ----
    print("\nBuilding CP-SAT model...")
    model = cp_model.CpModel()

    x = {}
    for sec in section_ids:
        for sl in slot_ids:
            x[(sec, sl)] = model.NewBoolVar(f"x_{sec}_{sl}")

    # 1️⃣ At least 10 sessions per section
    for sec in section_ids:
        model.Add(sum(x[(sec, sl)] for sl in slot_ids) >= MIN_SESSIONS)

    # 2️⃣ Room capacity
    for sl in slot_ids:
        cap = int(slot_capacity[sl])
        model.Add(sum(x[(sec, sl)] for sec in section_ids) <= cap)

    # 3️⃣ Faculty conflict
    for fac, secs in faculty_to_sections.items():
        for sl in slot_ids:
            model.Add(sum(x[(sec, sl)] for sec in secs) <= 1)

    # 4️⃣ Student conflict
    for student, secs in student_to_sections.items():
        if len(secs) <= 1:
            continue
        for sl in slot_ids:
            model.Add(sum(x[(sec, sl)] for sec in secs) <= 1)

    # No objective — pure feasibility
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT
    solver.parameters.num_search_workers = WORKERS

    print("\nSolving...")
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("\n❌ INFEASIBLE")
        print("Status:", solver.StatusName(status))
        return

    print("\n✅ FEASIBLE")
    print("Status:", solver.StatusName(status))

    # ---- Extract summary ----
    print("\nSessions scheduled per section:")
    for sec in section_ids[:10]:
        count = sum(solver.Value(x[(sec, sl)]) for sl in slot_ids)
        print(sec, "→", count)

    print("\nDone.")


if __name__ == "__main__":
    main()
