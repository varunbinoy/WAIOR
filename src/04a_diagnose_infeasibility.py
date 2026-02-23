import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs")
SESSIONS_PER_SECTION = 20
SLOTS_PER_WEEK = 40
WEEKS = 10
MAX_SESSIONS_PER_STUDENT = SLOTS_PER_WEEK * WEEKS  # 400

def main():
    section_enroll = pd.read_csv(OUTPUT_DIR / "section_enrollments.csv")
    sections = pd.read_csv(OUTPUT_DIR / "sections.csv")
    courses = pd.read_csv(OUTPUT_DIR / "courses.csv")
    slots = pd.read_csv(OUTPUT_DIR / "slots.csv")

    # Student -> number of sections enrolled
    stud_sec_counts = section_enroll.groupby("student_id")["section_id"].nunique().sort_values(ascending=False)
    n_students = stud_sec_counts.shape[0]

    # Required sessions per student if they attend all enrolled sections
    stud_required_sessions = stud_sec_counts * SESSIONS_PER_SECTION

    print("\n=== BASIC COUNTS ===")
    print("Students:", n_students)
    print("Sections:", sections.shape[0])
    print("Courses:", courses.shape[0])
    print("Slots total:", slots.shape[0], "(max sessions per student =", MAX_SESSIONS_PER_STUDENT, ")")

    print("\n=== TOP 20 STUDENTS by #sections enrolled ===")
    top = pd.DataFrame({
        "sections_enrolled": stud_sec_counts.head(20),
        "required_sessions": stud_required_sessions.head(20)
    })
    print(top.to_string())

    # Hard infeasibility check: any student needs > 400 sessions
    impossible = stud_required_sessions[stud_required_sessions > MAX_SESSIONS_PER_STUDENT]
    print("\n=== HARD IMPOSSIBILITY CHECK (required_sessions > 400) ===")
    if len(impossible) == 0:
        print("✅ No student exceeds 400 required sessions. (Good sign)")
    else:
        print(f"❌ {len(impossible)} students exceed 400 sessions -> model is impossible with student-level conflicts.")
        print(impossible.head(30).to_string())

    # Also show distribution
    print("\n=== DISTRIBUTION: sections enrolled per student ===")
    print(stud_sec_counts.describe().to_string())

    # How dense is overlap? If most students are in most sections, then everything can't overlap.
    # Compute average #students per section and overlap hint.
    sec_sizes = section_enroll.groupby("section_id")["student_id"].nunique().sort_values(ascending=False)
    print("\n=== SECTION SIZE SUMMARY ===")
    print(sec_sizes.describe().to_string())
    print("\nLargest sections:")
    print(sec_sizes.head(10).to_string())

if __name__ == "__main__":
    main()
