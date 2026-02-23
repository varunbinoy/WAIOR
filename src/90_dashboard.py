# src/91_faculty_load_heatmap.py
# Faculty load heatmap: rows = faculty, cols = week (1..10), values = #sessions scheduled

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("outputs")
DASH = OUT / "dashboard"
DASH.mkdir(parents=True, exist_ok=True)

# ---- configure: point to your term schedule export ----
# Change this if your schedule file name is different
SCHEDULE_CANDIDATES = [
    "term_schedule.csv",
    "term_schedule_floor18.csv",
    "term_schedule_floor18_cap20_facsplit.csv",
    "schedule.csv",
    "timetable.csv",
]

def find_schedule():
    for name in SCHEDULE_CANDIDATES:
        p = OUT / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find term schedule file in outputs/. "
        "Expected one of: " + ", ".join(SCHEDULE_CANDIDATES) +
        "\nTip: run your term export script (e.g., python src/45_term_schedule_floor18_export.py) "
        "and check what filename it creates in outputs/."
    )

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def get_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None

def parse_faculty_for_course(course_id: str, week: int, faculty_raw: str) -> str:
    """
    Handles DTI split case:
    'Prof. Rohit Kumar (Premid) + Prof. Rojers Puthur Josrph (Postmid)'
    Weeks 1-5 -> Rohit; Weeks 6-10 -> Rojers
    Otherwise, return faculty_raw (cleaned).
    """
    fr = (faculty_raw or "").strip()

    # If course is DTI and has the combined string, split by '+'
    if str(course_id).strip().upper() == "DTI" and "+" in fr and ("Premid" in fr or "Postmid" in fr):
        parts = [p.strip() for p in fr.split("+")]
        premid = parts[0].strip()
        postmid = parts[1].strip() if len(parts) > 1 else premid

        # pick by week boundary (your spec: first 10 sessions vs last 10;
        # operationally we map as Week 1-5 vs 6-10)
        return premid if week <= 5 else postmid

    return fr if fr else "Unknown Faculty"

def main():
    # --- load schedule ---
    sched_path = find_schedule()
    sched = pd.read_csv(sched_path)
    sched = normalize_cols(sched)

    # Required: week + course_id (or section_id with sections map)
    week_col = get_col(sched, ["week", "wk"])
    course_col = get_col(sched, ["course_id", "course", "coursecode"])
    section_col = get_col(sched, ["section_id", "section"])

    if week_col is None:
        raise KeyError(f"Schedule file missing 'week' column. Columns found: {list(sched.columns)}")

    # --- load mapping course->faculty_raw ---
    courses_path = OUT / "courses.csv"
    if not courses_path.exists():
        raise FileNotFoundError("Missing outputs/courses.csv (needed for faculty mapping).")

    courses = pd.read_csv(courses_path)
    courses = normalize_cols(courses)

    # Your courses.csv has faculty_raw
    if "faculty_raw" not in courses.columns:
        raise KeyError(f"outputs/courses.csv missing faculty_raw. Columns found: {list(courses.columns)}")

    # If schedule doesn't have course_id but has section_id, map via sections.csv
    if course_col is None:
        if section_col is None:
            raise KeyError(
                "Schedule must have either course_id OR section_id.\n"
                f"Columns found: {list(sched.columns)}"
            )
        sections_path = OUT / "sections.csv"
        if not sections_path.exists():
            raise FileNotFoundError("Missing outputs/sections.csv (needed to map section->course).")
        sections = pd.read_csv(sections_path)
        sections = normalize_cols(sections)
        if "section_id" not in sections.columns or "course_id" not in sections.columns:
            raise KeyError("sections.csv must have section_id and course_id.")

        sched = sched.merge(sections[["section_id", "course_id"]], on="section_id", how="left")
        course_col = "course_id"

    # Build faculty column in schedule
    course_to_faculty = dict(zip(courses["course_id"].astype(str).str.strip(), courses["faculty_raw"].astype(str)))

    def faculty_for_row(row):
        week = int(row[week_col])
        cid = str(row[course_col]).strip()
        fraw = course_to_faculty.get(cid, "")
        return parse_faculty_for_course(cid, week, fraw)

    sched["faculty"] = sched.apply(faculty_for_row, axis=1)

    # Each row is one scheduled session (should be true for your term schedule export)
    # If your file has a 'sessions' column (aggregated), we handle that too.
    sessions_col = get_col(sched, ["sessions", "session_count"])
    if sessions_col:
        sched["w"] = sched[sessions_col].astype(int)
    else:
        sched["w"] = 1

    # Aggregate: faculty x week
    heat = (
        sched.groupby(["faculty", sched[week_col].astype(int)])["w"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    # Ensure columns 1..10 exist
    for w in range(1, 11):
        if w not in heat.columns:
            heat[w] = 0
    heat = heat[sorted(heat.columns)]

    # --- plot heatmap using matplotlib imshow ---
    plt.figure(figsize=(14, max(6, 0.35 * len(heat))))
    plt.imshow(heat.values, aspect="auto")

    plt.title("Faculty Load Heatmap (Sessions per Week)")
    plt.xlabel("Week")
    plt.ylabel("Faculty")

    # ticks
    plt.xticks(range(0, 10), [str(i) for i in range(1, 11)])
    plt.yticks(range(len(heat.index)), heat.index)

    # annotate numbers (looks elite if kept subtle)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = int(heat.values[i, j])
            if val > 0:
                plt.text(j, i, str(val), ha="center", va="center", fontsize=8)

    plt.colorbar(label="Sessions")
    out_path = DASH / "faculty_load_heatmap.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved heatmap: {out_path}")
    print(f"Schedule used: {sched_path}")
    print(f"Heatmap shape: {heat.shape[0]} faculty x {heat.shape[1]} weeks")

if __name__ == "__main__":
    main()
