"""Random stratum: one headline per day of the 24-month test window.

The outlet rotates daily (ROTABORSA, DUNYA, BORSAGUNDEM, ...), so outlet is not
confounded with time. If the scheduled outlet has no headline that day, the
next one in the rotation is used and the row is marked.

Usage:
    python scripts/02_sample_random.py
"""

from pathlib import Path

import pandas as pd

TEST_START = "2024-08-23"  # 24 months ending on the last day in the snapshot
TEST_END = "2026-08-22"
DAY_STEP = 1
ROTATION = ["ROTABORSA", "DUNYA", "BORSAGUNDEM"]
SEED = 20261029

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def sample_random_stratum(snapshot: pd.DataFrame) -> pd.DataFrame:
    by_day_source = {key: grp for key, grp in snapshot.groupby(["date", "source"])}
    days = pd.date_range(TEST_START, TEST_END, freq=f"{DAY_STEP}D")

    rows = []
    for i, day in enumerate(days):
        date = day.strftime("%Y-%m-%d")
        scheduled = ROTATION[i % len(ROTATION)]
        # Try the scheduled outlet first, then the others in rotation order.
        order = ROTATION[i % len(ROTATION):] + ROTATION[: i % len(ROTATION)]
        for source in order:
            candidates = by_day_source.get((date, source))
            if candidates is not None:
                row = candidates.sample(1, random_state=SEED + i).iloc[0].to_dict()
                row["scheduled_source"] = scheduled
                rows.append(row)
                break
    sample = pd.DataFrame(rows)
    sample.insert(0, "stratum", "random")
    return sample


def main() -> None:
    snapshot = pd.read_csv(DATA_DIR / "snapshot.csv")
    sample = sample_random_stratum(snapshot)
    sample.to_csv(DATA_DIR / "sample_random.csv", index=False)

    fallbacks = (sample["source"] != sample["scheduled_source"]).sum()
    print(f"{len(sample)} headlines, {fallbacks} used a fallback outlet")
    print(sample["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
