import pandas as pd
import math
from pathlib import Path
from ortools.sat.python import cp_model

OUTPUT_DIR = Path("outputs")

MAX_CAP = 70
MIN_CAP = 25

def make_global_bucket(students: list[str]) -> dict[str, int]:
    """
    Deterministic A/B identity:
    sort student_ids and alternate 0/1
    """
    students = sorted([str(s).strip() for s in students])
    bucket = {}
    for i, sid in enumerate(students):
        bucket[sid] = i % 2
    return bucket

def solve_ab_assignment(student_ids: list[str], bucket: dict[str, int], course_id: str):
    """
    For a single course:
    Decide y[s] ∈ {0,1} (0=A, 1=B)
    s.t. MIN_CAP <= sizeA,sizeB <= MAX_CAP
    minimize mismatches to global bucket.
    """
    N = len(student_ids)
    if N <= MAX_CAP:
        # no split needed
        return {sid: 0 for sid in student_ids}, 1

    # For your dataset, N max is around 140 so k should be 2.
    k = math.ceil(N / MAX_CAP)
    if k != 2:
        raise ValueError(f"{course_id}: Expected 2 sections, got k={k} for N={N}. Extend model if needed.")

    model = cp_model.CpModel()

    # y[s] = 1 means Section B, 0 means Section A
    y = {sid: model.NewBoolVar(f"y_{course_id}_{sid}") for sid in student_ids}

    sizeB = model.NewIntVar(0, N, f"sizeB_{course_id}")
    model.Add(sizeB == sum(y[sid] for sid in student_ids))

    sizeA = model.NewIntVar(0, N, f"sizeA_{course_id}")
    model.Add(sizeA == N - sizeB)

    # Size constraints
    model.Add(sizeA >= MIN_CAP)
    model.Add(sizeA <= MAX_CAP)
    model.Add(sizeB >= MIN_CAP)
    model.Add(sizeB <= MAX_CAP)

    # Mismatch penalties: mismatch[s] = 1 if y[s] != bucket[s]
    mismatch = []
    for sid in student_ids:
        m = model.NewBoolVar(f"mismatch_{course_id}_{sid}")
        # m == 1 <=> y != bucket
        if bucket[sid] == 0:
            # mismatch if y=1
            model.Add(m == y[sid])
        else:
            # mismatch if y=0 -> m = 1 - y
            model.Add(m + y[sid] == 1)
        mismatch.append(m)

    # Objective: minimize mismatches (keep student in their global side)
    model.Minimize(sum(mismatch))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"{course_id}: No feasible A/B assignment found. N={N}")

    assign = {sid: int(solver.Value(y[sid])) for sid in student_ids}  # 0=A,1=B
    return assign, 2

def build_sections_ilp():
    enroll_path = OUTPUT_DIR / "enrollments.csv"
    if not enroll_path.exists():
        raise FileNotFoundError("outputs/enrollments.csv missing. Run 01_load_data.py first.")

    enroll = pd.read_csv(enroll_path)
    enroll["course_id"] = enroll["course_id"].astype(str).str.strip()
    enroll["student_id"] = enroll["student_id"].astype(str).str.strip()
    enroll = enroll.drop_duplicates(["course_id", "student_id"])

    all_students = enroll["student_id"].unique().tolist()
    bucket = make_global_bucket(all_students)

    course_counts = enroll.groupby("course_id")["student_id"].nunique().sort_values(ascending=False)

    sections_rows = []
    section_enroll_rows = []

    for course_id, N in course_counts.items():
        studs = enroll.loc[enroll["course_id"] == course_id, "student_id"].unique().tolist()

        # Solve assignment
        assignment, k = solve_ab_assignment(studs, bucket, course_id)

        if k == 1:
            # single section
            sec_id = f"{course_id}_A"
            sections_rows.append({"section_id": sec_id, "course_id": course_id, "section_label": "A", "size": N})
            for sid in studs:
                section_enroll_rows.append({"section_id": sec_id, "student_id": sid})
        else:
            # two sections A/B
            A = [sid for sid in studs if assignment[sid] == 0]
            B = [sid for sid in studs if assignment[sid] == 1]

            # safety
            if not (MIN_CAP <= len(A) <= MAX_CAP and MIN_CAP <= len(B) <= MAX_CAP):
                raise RuntimeError(f"{course_id}: bad sizes after solve A={len(A)} B={len(B)}")

            secA = f"{course_id}_A"
            secB = f"{course_id}_B"

            sections_rows.append({"section_id": secA, "course_id": course_id, "section_label": "A", "size": len(A)})
            sections_rows.append({"section_id": secB, "course_id": course_id, "section_label": "B", "size": len(B)})

            for sid in A:
                section_enroll_rows.append({"section_id": secA, "student_id": sid})
            for sid in B:
                section_enroll_rows.append({"section_id": secB, "student_id": sid})

    sections_df = pd.DataFrame(sections_rows).sort_values(["course_id", "section_label"]).reset_index(drop=True)
    section_enroll_df = pd.DataFrame(section_enroll_rows).drop_duplicates().reset_index(drop=True)

    # Save
    sections_df.to_csv(OUTPUT_DIR / "sections.csv", index=False)
    section_enroll_df.to_csv(OUTPUT_DIR / "section_enrollments.csv", index=False)

    print("\n✅ ILP student-aware sectioning complete")
    print("Courses:", sections_df["course_id"].nunique())
    print("Sections:", len(sections_df))
    print("Section enrollments:", len(section_enroll_df))

    # quick stats
    split_courses = sections_df.groupby("course_id")["section_id"].nunique()
    print("Courses split (>1 section):", int((split_courses > 1).sum()))

    print("\nSection size distribution:")
    print(sections_df["size"].value_counts().sort_index().to_string())

    # sanity
    bad_small = sections_df[sections_df["size"] < MIN_CAP]
    bad_big = sections_df[sections_df["size"] > MAX_CAP]
    print("\nSanity:")
    print("Sections < 25:", len(bad_small))
    print("Sections > 70:", len(bad_big))

if __name__ == "__main__":
    build_sections_ilp()
