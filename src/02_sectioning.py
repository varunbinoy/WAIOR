import pandas as pd
from pathlib import Path
import math
import string

OUTPUT_DIR = Path("outputs")

MAX_CAP = 70
MIN_CAP = 25

def make_section_labels(k: int):
    # A, B, C, ... up to 26
    if k > 26:
        raise ValueError("More than 26 sections not supported in this simple label scheme.")
    return list(string.ascii_uppercase[:k])

def section_course(course_id: str, student_ids: list[str]):
    """
    Returns:
      sections_rows: list[dict]
      section_enroll_rows: list[dict]
    """
    N = len(student_ids)

    # Decide number of sections k using max-cap
    k_min = math.ceil(N / MAX_CAP)
    k_max = math.floor(N / MIN_CAP) if N >= MIN_CAP else 0

    if k_min == 0:
        raise ValueError(f"{course_id}: No students found.")

    if k_min > k_max:
        raise ValueError(
            f"{course_id}: Cannot satisfy both constraints. "
            f"N={N}, need at least k={k_min} sections for max {MAX_CAP}, "
            f"but at most k={k_max} sections for min {MIN_CAP}."
        )

    k = k_min  # choose minimum feasible number of sections (managerially efficient)

    # Balanced sizes
    base = N // k
    rem = N % k
    sizes = [base + (1 if i < rem else 0) for i in range(k)]

    # Validate sizes
    if max(sizes) > MAX_CAP or min(sizes) < MIN_CAP:
        raise ValueError(f"{course_id}: Section sizes invalid: {sizes} for N={N}, k={k}")

    # Deterministic assignment
    student_ids_sorted = sorted([str(s).strip() for s in student_ids])

    labels = make_section_labels(k)
    sections_rows = []
    section_enroll_rows = []

    idx = 0
    for label, size in zip(labels, sizes):
        section_id = f"{course_id}_{label}"
        sections_rows.append({
            "section_id": section_id,
            "course_id": course_id,
            "section_label": label,
            "size": size
        })

        chunk = student_ids_sorted[idx: idx + size]
        for sid in chunk:
            section_enroll_rows.append({
                "section_id": section_id,
                "student_id": sid
            })

        idx += size

    return sections_rows, section_enroll_rows

def generate_sections():
    enrollments_path = OUTPUT_DIR / "enrollments.csv"
    if not enrollments_path.exists():
        raise FileNotFoundError("Missing outputs/enrollments.csv. Run 01_load_data.py first.")

    enrollments_df = pd.read_csv(enrollments_path)
    enrollments_df["student_id"] = enrollments_df["student_id"].astype(str).str.strip()
    enrollments_df["course_id"] = enrollments_df["course_id"].astype(str).str.strip()

    # Unique enrollments
    enrollments_df = enrollments_df.drop_duplicates(["course_id", "student_id"])

    sections_rows_all = []
    section_enroll_rows_all = []

    counts = (
        enrollments_df.groupby("course_id")["student_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="enrolled")
    )

    # Build sections for each course
    for _, row in counts.iterrows():
        course_id = row["course_id"]
        student_ids = enrollments_df[enrollments_df["course_id"] == course_id]["student_id"].tolist()

        sec_rows, sec_enroll_rows = section_course(course_id, student_ids)
        sections_rows_all.extend(sec_rows)
        section_enroll_rows_all.extend(sec_enroll_rows)

    sections_df = pd.DataFrame(sections_rows_all)
    section_enrollments_df = pd.DataFrame(section_enroll_rows_all)

    # Save
    sections_df.to_csv(OUTPUT_DIR / "sections.csv", index=False)
    section_enrollments_df.to_csv(OUTPUT_DIR / "section_enrollments.csv", index=False)

    # Summary prints
    print("\n✅ Sectioning complete")
    print("Total courses:", len(counts))
    print("Total sections:", len(sections_df))
    print("Total section enrollments:", len(section_enrollments_df))

    split_courses = sections_df.groupby("course_id")["section_id"].nunique()
    print("\nCourses needing split (>1 section):", (split_courses > 1).sum())

    print("\nTop 10 largest courses:")
    print(counts.head(10).to_string(index=False))

    print("\nSection size distribution (value counts):")
    print(sections_df["size"].value_counts().sort_index().to_string())

if __name__ == "__main__":
    generate_sections()
