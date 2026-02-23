import pandas as pd
from pathlib import Path

# Option A: relative path (your file inside data folder)
XLSX_PATH = Path("data") / "WAI_data.xlsx"
OUTPUT_DIR = Path("outputs")


def norm_col(c: str) -> str:
    """Normalize column names for robust matching."""
    return str(c).strip().lower().replace("\n", " ").replace("  ", " ")


def load_excel(xlsx_path: Path):
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel not found at: {xlsx_path.resolve()}")

    xl = pd.ExcelFile(xlsx_path)

    courses_rows = []
    enroll_rows = []
    students_map = {}  # student_id -> student_name

    for sheet in xl.sheet_names:
        # Raw read to capture B1 faculty and B2 course name
        raw = xl.parse(sheet_name=sheet, header=None)

        # Based on your format:
        # A1 is label "Faculty Name"
        # B1 is actual faculty string
        # B2 is course name
        faculty = str(raw.iat[0, 1]).strip() if raw.shape[1] > 1 else ""
        course_name = str(raw.iat[1, 1]).strip() if raw.shape[1] > 1 else ""

        # Student table starts with header on Row 3 (0-index row 2)
        df = xl.parse(sheet_name=sheet, header=2)
        df.columns = [norm_col(c) for c in df.columns]

        # Detect student id and student name columns robustly
        sid_candidates = [c for c in df.columns if ("student id" in c) or (c == "sid") or (c == "id")]
        name_candidates = [c for c in df.columns if ("student name" in c) or (c == "name")]

        if not sid_candidates or not name_candidates:
            print(f"\n[DEBUG] Sheet: {sheet}")
            print("Columns found:", df.columns.tolist())
            raise ValueError("Could not detect Student ID / Student Name columns. Check headers in Row 3.")

        sid_col = sid_candidates[0]
        name_col = name_candidates[0]

        roster = df[[sid_col, name_col]].copy()
        roster.columns = ["student_id", "student_name"]
        roster = roster.dropna(subset=["student_id"])

        roster["student_id"] = roster["student_id"].astype(str).str.strip()
        roster["student_name"] = roster["student_name"].astype(str).str.strip()

        # Remove junk rows
        roster = roster[roster["student_id"].str.len() > 0]
        roster = roster[~roster["student_id"].str.lower().str.contains("student")]

        # Course row
        courses_rows.append(
            {
                "course_id": sheet.strip(),
                "course_name": course_name,
                "faculty_raw": faculty,  # raw faculty string (may include premid/postmid)
            }
        )

        # Enrollment rows + student map
        for sid, sname in zip(roster["student_id"], roster["student_name"]):
            students_map.setdefault(sid, sname)
            enroll_rows.append({"course_id": sheet.strip(), "student_id": sid})

    courses_df = pd.DataFrame(courses_rows).drop_duplicates("course_id").reset_index(drop=True)
    students_df = (
        pd.DataFrame([{"student_id": k, "student_name": v} for k, v in students_map.items()])
        .sort_values("student_id")
        .reset_index(drop=True)
    )
    enrollments_df = pd.DataFrame(enroll_rows).drop_duplicates().reset_index(drop=True)

    return courses_df, students_df, enrollments_df


def basic_queries(courses_df, students_df, enrollments_df):
    print("\n✅ LOADED SUCCESSFULLY")
    print("Courses:", len(courses_df))
    print("Unique Students:", students_df["student_id"].nunique())
    print("Enrollments:", len(enrollments_df))

    counts = (
        enrollments_df.groupby("course_id")["student_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="enrolled")
    )

    print("\nTop 10 courses by enrollment:")
    print(counts.head(10).to_string(index=False))

    split = counts[counts["enrolled"] > 70].copy()
    print("\nCourses needing split (>70):", len(split))
    if len(split) > 0:
        print(split.to_string(index=False))

    fac_load = (
        courses_df.groupby("faculty_raw")["course_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="num_courses")
    )

    print("\nTop 10 faculty by number of courses:")
    print(fac_load.head(10).to_string(index=False))

    return counts, split, fac_load


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    courses_df, students_df, enrollments_df = load_excel(XLSX_PATH)

    # Queries + prints
    counts, split, fac_load = basic_queries(courses_df, students_df, enrollments_df)

    # Save outputs for later use
    courses_df.to_csv(OUTPUT_DIR / "courses.csv", index=False)
    students_df.to_csv(OUTPUT_DIR / "students.csv", index=False)
    enrollments_df.to_csv(OUTPUT_DIR / "enrollments.csv", index=False)
    counts.to_csv(OUTPUT_DIR / "course_enrollment_counts.csv", index=False)

    print("\nSaved CSVs into outputs/: courses.csv, students.csv, enrollments.csv, course_enrollment_counts.csv")
