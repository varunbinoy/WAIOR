import pandas as pd
from pathlib import Path
from ortools.sat.python import cp_model

OUT = Path("outputs")

CAP_TOTAL = 20
TIME_LIMIT = 240
WORKERS = 8

def day_num(label: str) -> int:
    # "C10" -> 10
    return int(str(label).strip()[1:])

def main():
    print("Loading data...")

    sections_df = pd.read_csv(OUT / "sections.csv")
    enroll_df = pd.read_csv(OUT / "section_enrollments.csv")
    courses_df = pd.read_csv(OUT / "courses.csv")
    base_df = pd.read_csv(OUT / "section_sessions_floor18_cap20_facsplit.csv")
    cslots_df = pd.read_csv(OUT / "contingent_slots.csv")

    # Clean strings
    for df in [sections_df, enroll_df, courses_df, base_df, cslots_df]:
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.strip()

    # Compute deficits
    base_df["deficit"] = CAP_TOTAL - base_df["sessions"]
    deficit_df = base_df[base_df["deficit"] > 0].copy()

    total_need = int(deficit_df["deficit"].sum())
    print("\nTotal remaining sessions to schedule:", total_need)

    if total_need == 0:
        print("✅ Nothing to schedule.")
        return

    section_ids = deficit_df["section_id"].tolist()
    deficit = dict(zip(deficit_df["section_id"], deficit_df["deficit"]))

    # Maps
    sec_to_course = dict(zip(sections_df["section_id"], sections_df["course_id"]))
    course_to_faculty = dict(zip(courses_df["course_id"], courses_df["faculty_raw"]))

    # Student -> sections then keep only deficit sections
    student_to_sections = (
        enroll_df.groupby("student_id")["section_id"]
        .apply(list)
        .to_dict()
    )
    for sid in list(student_to_sections.keys()):
        student_to_sections[sid] = [s for s in student_to_sections[sid] if s in deficit]
        if not student_to_sections[sid]:
            del student_to_sections[sid]

    # -------------------------
    # DIAGNOSTIC: deficit load per student
    # -------------------------
    student_deficit_load = {}
    for sid, secs in student_to_sections.items():
        student_deficit_load[sid] = sum(deficit[s] for s in secs)

    max_load = max(student_deficit_load.values()) if student_deficit_load else 0
    avg_load = (sum(student_deficit_load.values()) / len(student_deficit_load)) if student_deficit_load else 0.0

    print("\n=== DEFICIT STRUCTURE DIAGNOSTIC ===")
    print("Students affected (have at least 1 pending session):", len(student_deficit_load))
    print("Max pending sessions for any student:", max_load)
    print("Average pending sessions per affected student:", round(avg_load, 2))

    top10 = sorted(student_deficit_load.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 students by pending sessions:")
    for sid, v in top10:
        print(sid, "→", v)

    # Contingent slots
    if "c_slot_id" not in cslots_df.columns or "c_day" not in cslots_df.columns:
        raise ValueError("contingent_slots.csv must contain c_slot_id and c_day columns")

    cslot_ids = cslots_df["c_slot_id"].tolist()
    cslot_cap = dict(zip(cslots_df["c_slot_id"], cslots_df["room_cap"]))
    cslot_to_day = dict(zip(cslots_df["c_slot_id"], cslots_df["c_day"]))

    day_labels = sorted(cslots_df["c_day"].unique().tolist(), key=day_num)
    day_to_slots = {d: cslots_df.loc[cslots_df["c_day"] == d, "c_slot_id"].tolist() for d in day_labels}

    print("\nContingent days available:", len(day_labels))
    if day_labels:
        print("Slots per contingent day:", len(day_to_slots[day_labels[0]]))

    # Faculty -> sections (only deficit sections)
    faculty_to_secs = {}
    for s in section_ids:
        fac = course_to_faculty[sec_to_course[s]]
        faculty_to_secs.setdefault(fac, []).append(s)

    # -------------------------
    # Build CP-SAT model
    # -------------------------
    model = cp_model.CpModel()

    # x[s, cs] : schedule section s in contingent slot cs
    x = {(s, cs): model.NewBoolVar(f"x_{s}_{cs}") for s in section_ids for cs in cslot_ids}

    # y[d] : contingent day used
    y = {d: model.NewBoolVar(f"y_{d}") for d in day_labels}

    # Deficit fulfillment
    for s in section_ids:
        model.Add(sum(x[(s, cs)] for cs in cslot_ids) == int(deficit[s]))

    # Room capacity per contingent slot
    for cs in cslot_ids:
        model.Add(sum(x[(s, cs)] for s in section_ids) <= int(cslot_cap[cs]))

    # Student conflict per contingent slot
    for sid, secs in student_to_sections.items():
        if len(secs) <= 1:
            continue
        for cs in cslot_ids:
            model.Add(sum(x[(s, cs)] for s in secs) <= 1)

    # Faculty conflict per contingent slot
    for cs in cslot_ids:
        for fac, secs in faculty_to_secs.items():
            if len(secs) > 1:
                model.Add(sum(x[(s, cs)] for s in secs) <= 1)

    # Link x to y(day)
    for d in day_labels:
        for cs in day_to_slots[d]:
            for s in section_ids:
                model.Add(x[(s, cs)] <= y[d])

    # Objective: minimize days used
    model.Minimize(sum(y[d] for d in day_labels))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT
    solver.parameters.num_search_workers = WORKERS

    print("\nSolving contingent-days minimization...")
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("❌ No feasible solution even with", len(day_labels), "contingent days.")
        print("Status:", solver.StatusName(status))
        return

    used = [d for d in day_labels if solver.Value(y[d]) == 1]
    print("✅", solver.StatusName(status))
    print("Contingent days used:", len(used))
    print("Days used:", used)

    # Export schedule
    rows = []
    for s in section_ids:
        for cs in cslot_ids:
            if solver.Value(x[(s, cs)]) == 1:
                rows.append({
                    "section_id": s,
                    "course_id": sec_to_course[s],
                    "faculty": course_to_faculty[sec_to_course[s]],
                    "c_slot_id": cs,
                    "c_day": cslot_to_day[cs],
                })

    sched = pd.DataFrame(rows)

    # Merge time columns
    sched = sched.merge(
        cslots_df[["c_slot_id", "start", "end", "room_cap"]],
        on="c_slot_id",
        how="left"
    )

    # Sort days numerically
    sched["c_day_num"] = sched["c_day"].apply(day_num)
    sched = sched.sort_values(["c_day_num", "start", "section_id"]).drop(columns=["c_day_num"]).reset_index(drop=True)

    out_path = OUT / "contingent_schedule.csv"
    sched.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
