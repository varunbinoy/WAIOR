import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs")

WEEKS = range(1, 11)

# Mon–Sat: 6 slots
WEEKDAY_SLOTS = [
    ("S1", "09:00", "10:30"),
    ("S2", "10:45", "12:15"),
    ("S3", "12:30", "14:00"),
    ("S4", "14:45", "16:15"),
    ("S5", "16:30", "18:00"),
    ("S6", "18:15", "19:45"),
]

# Sunday: 4 slots
SUNDAY_SLOTS = [
    ("S1", "09:00", "10:30"),
    ("S2", "10:45", "12:15"),
    ("S3", "12:30", "14:00"),
    ("S4", "15:30", "17:00"),
]

DAYS_MON_SAT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
DAYS_SUN = ["Sun"]

ROOMS_W1_4 = 10
ROOMS_W5_10 = 4

def room_cap(week: int) -> int:
    return ROOMS_W1_4 if week <= 4 else ROOMS_W5_10

def build_slots():
    rows = []
    for week in WEEKS:
        # Mon–Sat
        for day in DAYS_MON_SAT:
            for slot_code, start, end in WEEKDAY_SLOTS:
                rows.append({
                    "slot_id": f"W{week}_{day}_{slot_code}",
                    "week": week,
                    "day": day,
                    "slot_code": slot_code,
                    "start": start,
                    "end": end,
                    "room_capacity": room_cap(week),
                    "is_sunday": 0
                })
        # Sunday
        for day in DAYS_SUN:
            for slot_code, start, end in SUNDAY_SLOTS:
                rows.append({
                    "slot_id": f"W{week}_{day}_{slot_code}",
                    "week": week,
                    "day": day,
                    "slot_code": slot_code,
                    "start": start,
                    "end": end,
                    "room_capacity": room_cap(week),
                    "is_sunday": 1
                })

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / "slots.csv", index=False)

    print("\n✅ slots.csv rebuilt (PAN-IIM shock ON)")
    print("Total slots:", len(df))
    print("Slots per week (should be 40):")
    print(df.groupby("week")["slot_id"].nunique().to_string())
    print("\nRoom capacity check:")
    print("Week 1 caps:", sorted(df[df["week"] == 1]["room_capacity"].unique().tolist()))
    print("Week 5 caps:", sorted(df[df["week"] == 5]["room_capacity"].unique().tolist()))

if __name__ == "__main__":
    build_slots()
