import pandas as pd
from pathlib import Path

OUTPUT = Path("outputs")

def main():

    print("Loading data...")

    courses = pd.read_csv(OUTPUT / "courses.csv")
    sections = pd.read_csv(OUTPUT / "sections.csv")

    # Clean
    courses["course_id"] = courses["course_id"].astype(str).str.strip()
    courses["faculty_raw"] = courses["faculty_raw"].astype(str).str.strip()
    sections["course_id"] = sections["course_id"].astype(str).str.strip()
    sections["section_id"] = sections["section_id"].astype(str).str.strip()

    # Merge to get faculty per section
    merged = sections.merge(
        courses[["course_id", "faculty_raw"]],
        on="course_id",
        how="left"
    )

    print("\n=== FACULTY COURSE COUNT ===")
    faculty_course_count = (
        courses.groupby("faculty_raw")["course_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    print(faculty_course_count.to_string())

    print("\n=== FACULTY SECTION COUNT ===")
    faculty_section_count = (
        merged.groupby("faculty_raw")["section_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    print(faculty_section_count.to_string())

    print("\n=== DETAILED FACULTY → COURSES ===")
    faculty_courses = (
        courses.groupby("faculty_raw")["course_id"]
        .apply(list)
        .sort_index()
    )

    for fac, course_list in faculty_courses.items():
        print(f"\n{fac}")
        for c in course_list:
            print(f"  - {c}")

if __name__ == "__main__":
    main()
