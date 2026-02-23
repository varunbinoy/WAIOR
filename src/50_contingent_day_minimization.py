import pandas as pd
from pathlib import Path
from ortools.sat.python import cp_model

OUTPUT = Path("outputs")

MAX_DAYS = 5
SLOTS_PER_DAY = 7
ROOMS = 10

TIME_LIMIT = 180
WORKERS = 8

def main():

    print("Loading data...")

    sections_df = pd.read_csv(OUTPUT / "sections.csv")
    enroll_df = pd.read_csv(OUTPUT / "section_enrollments.csv")
    courses_df = pd.read_csv(OUTPUT / "courses.csv")
    sess_df = pd.read_csv(OUTPUT / "section_sessions_floor18_cap20_facsplit.csv")

    # Clean
    for df in [sections_df, enroll_df, courses_df, sess_df]:
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.strip()

    # Build deficit
    sess_df["deficit"] = 20 - sess_df["sessions"]
    deficit_df = sess_df[sess_df["deficit"] > 0].copy()

    print("Total remaining sessions:", deficit_df["deficit"].sum())

    section_ids = deficit_df["section_id"].tolist()
    deficit_map = dict(zip(deficit_df["section_id"], deficit_df["deficit"]))

    # student mapping
    student_to_sections = (
        enroll_df.groupby("student_id")["section_id"]
        .apply(list)
        .to_dict()
    )

    # section → faculty
    sec_to_course = dict(zip(sections_df["section_id"], sections_df["course_id"]))
    course_to_faculty = dict(zip(courses_df["course_id"], courses_df["faculty_raw"]))

    model = cp_model.CpModel()

    days = range(MAX_DAYS)
    slots = range(SLOTS_PER_DAY)

    # Decision vars
    x = {}
    for s in section_ids:
        for d in days:
            for t in slots:
                x[s,d,t] = model.NewBoolVar(f"x_{s}_{d}_{t}")

    y = {d: model.NewBoolVar(f"y_{d}") for d in days}

    # Each section must get its deficit sessions
    for s in section_ids:
        model.Add(sum(x[s,d,t] for d in days for t in slots) == deficit_map[s])

    # Room capacity
    for d in days:
        for t in slots:
            model.Add(sum(x[s,d,t] for s in section_ids) <= ROOMS)

    # Student conflict
    for student, secs in student_to_sections.items():
        relevant_secs = [s for s in secs if s in section_ids]
        if len(relevant_secs) <= 1:
            continue
        for d in days:
            for t in slots:
                model.Add(sum(x[s,d,t] for s in relevant_secs) <= 1)

    # Faculty conflict
    for d in days:
        for t in slots:
            faculty_groups = {}
            for s in section_ids:
                fac = course_to_faculty[sec_to_course[s]]
                faculty_groups.setdefault(fac, []).append(s)
            for fac, secs in faculty_groups.items():
                model.Add(sum(x[s,d,t] for s in secs) <= 1)

    # Link x to y
    for d in days:
        for s in section_ids:
            for t in slots:
                model.Add(x[s,d,t] <= y[d])

    # Objective: minimize contingent days
    model.Minimize(sum(y[d] for d in days))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT
    solver.parameters.num_search_workers = WORKERS

    print("Solving contingent model...")
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        used_days = [d for d in days if solver.Value(y[d]) == 1]
        print("✅ Solution found")
        print("Contingent days used:", len(used_days))
        print("Days:", used_days)
    else:
        print("❌ No feasible solution")

if __name__ == "__main__":
    main()
