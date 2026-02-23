import pandas as pd
from pathlib import Path

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# 7 slots per contingent day (added 20:00–21:30)
SLOTS = [
    ("09:00", "10:30"),
    ("10:45", "12:15"),
    ("12:30", "14:00"),
    ("14:45", "16:15"),
    ("16:30", "18:00"),
    ("18:15", "19:45"),
    ("20:00", "21:30"),
]

ROOM_CAP = 10
MAX_CONTINGENT_DAYS = 10  # allow up to 10; solver will minimize

def main():
    rows = []
    slot_id = 0

    for day_idx in range(1, MAX_CONTINGENT_DAYS + 1):
        day_label = f"C{day_idx}"
        for (start, end) in SLOTS:
            slot_id += 1
            rows.append({
                "c_slot_id": f"{day_label}_S{slot_id}",  # unique id
                "c_day": day_label,
                "slot_in_day": len(rows) % 7 + 1,
                "start": start,
                "end": end,
                "room_cap": ROOM_CAP
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "contingent_slots.csv", index=False)

    print("✅ outputs/contingent_slots.csv created")
    print("Total contingent slots:", len(df))
    print("Slots per day:", df.groupby("c_day")["c_slot_id"].count().head())

if __name__ == "__main__":
    main()
