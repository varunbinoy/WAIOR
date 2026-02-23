import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs")

def main():
    sections_path = OUTPUT_DIR / "sections.csv"
    enroll_path = OUTPUT_DIR / "section_enrollments.csv"

    print("\n=== VERIFYING SECTION FILES ===")

    if not sections_path.exists():
        print("❌ sections.csv NOT found")
        return

    if not enroll_path.exists():
        print("❌ section_enrollments.csv NOT found")
        return

    sections = pd.read_csv(sections_path)
    enroll = pd.read_csv(enroll_path)

    print("\nSections file loaded from:", sections_path.resolve())
    print("Total sections:", len(sections))
    print("Unique courses:", sections["course_id"].nunique())

    if "size" in sections.columns:
        print("\nSection size stats:")
        print("Min size:", sections["size"].min())
        print("Max size:", sections["size"].max())
        print("Distribution:")
        print(sections["size"].value_counts().sort_index())
    else:
        print("\n⚠️ 'size' column not found in sections.csv")

    print("\nFirst 5 sections:")
    print(sections.head())

    print("\nTotal section enrollments:", len(enroll))
    print("Unique students:", enroll["student_id"].nunique())

    print("\n=== END VERIFY ===\n")

if __name__ == "__main__":
    main()
