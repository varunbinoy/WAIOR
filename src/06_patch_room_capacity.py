import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs")

def main():
    path = OUTPUT_DIR / "slots.csv"
    df = pd.read_csv(path)

    df["room_capacity"] = 10
    df.to_csv(path, index=False)

    print("✅ Patched slots.csv: room_capacity set to 10 for all weeks.")
    print("Unique capacities now:", sorted(df["room_capacity"].unique().tolist()))

if __name__ == "__main__":
    main()
