import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs")
BOTTLENECK = ["DWDV", "BMS", "BV", "SMTI"]

enroll = pd.read_csv(OUTPUT_DIR / "enrollments.csv")
enroll["course_id"] = enroll["course_id"].astype(str).str.strip()
enroll["student_id"] = enroll["student_id"].astype(str).str.strip()

counts = enroll.groupby("course_id")["student_id"].nunique().sort_values(ascending=False)

print("\n=== Bottleneck course enrollments ===")
for c in BOTTLENECK:
    n = int(counts.get(c, 0))
    print(f"{c}: {n}")
