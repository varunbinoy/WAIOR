import pandas as pd

df = pd.read_csv("outputs/section_enrollments.csv")
print("Unique students:", df["student_id"].nunique())
