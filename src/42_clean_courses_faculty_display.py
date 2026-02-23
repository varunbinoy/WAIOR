import pandas as pd
from pathlib import Path

OUTPUT = Path("outputs")

def main():
    courses = pd.read_csv(OUTPUT / "courses.csv")
    courses["course_id"] = courses["course_id"].astype(str).str.strip()
    courses["faculty_raw"] = courses["faculty_raw"].astype(str).str.strip()

    # Add two explicit columns (does NOT overwrite faculty_raw)
    courses["faculty_premid"] = courses["faculty_raw"]
    courses["faculty_postmid"] = courses["faculty_raw"]

    # Only for DTI
    mask = courses["course_id"] == "DTI"
    courses.loc[mask, "faculty_premid"] = "Prof. Rohit Kumar"
    courses.loc[mask, "faculty_postmid"] = "Prof. Rogers"

    out = OUTPUT / "courses_faculty_split_view.csv"
    courses.to_csv(out, index=False)
    print("Saved:", out)

if __name__ == "__main__":
    main()
