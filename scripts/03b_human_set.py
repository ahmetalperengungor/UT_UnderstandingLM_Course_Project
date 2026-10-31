"""Mark the core set to label first, and add a second few-shot pool draw.

Core set ("human_set" in data/items_key.csv): all 120 lexicon and burst test
items plus a random N_RANDOM of the 730 random-stratum items. The few-shot runs
use this set. The first pool draw had only one positive, so a second draw of
lexicon and burst headlines from the pool period is appended to the sheet.
Existing rows and labels are not changed.

Usage:
    python scripts/03b_human_set.py
"""

import csv
import importlib
from pathlib import Path

import pandas as pd

from lexicon import lexicon_score

build = importlib.import_module("03_build_label_sheets")

N_RANDOM = 180
N_POOL_LEXICON_PER_SOURCE = 4
N_POOL_BURST = 8
SEED = 20261030

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SHEET = ROOT / "labeling" / "label_sheet.csv"
SHEET_FIELDS = ["item", "title", "label", "direction", "note"]


def extend_pool(key: pd.DataFrame) -> pd.DataFrame:
    snapshot = pd.read_csv(DATA_DIR / "snapshot.csv")
    snapshot["lexicon_score"] = snapshot["title"].map(lexicon_score)
    pool_df = snapshot[snapshot["date"].between(build.POOL_START, build.POOL_END)]
    pool_df = pool_df[~pool_df["id"].isin(key["id"])]

    lex = build.sample_lexicon(pool_df, N_POOL_LEXICON_PER_SOURCE, SEED)
    burst = build.sample_bursts(pool_df[~pool_df["id"].isin(lex["id"])], N_POOL_BURST, SEED)
    new = pd.concat([lex, burst], ignore_index=True).sample(frac=1, random_state=SEED)
    new = new.assign(set="pool", pool_draw=2)
    new.insert(0, "item", range(key["item"].max() + 1, key["item"].max() + 1 + len(new)))
    return new


def main() -> None:
    key = pd.read_csv(DATA_DIR / "items_key.csv")
    if "human_set" in key:
        print("human set already chosen; nothing to do")
        return

    key["pool_draw"] = key["set"].eq("pool").map({True: 1, False: pd.NA})
    new_pool = extend_pool(key)
    key = pd.concat([key, new_pool[key.columns.drop("human_set", errors="ignore")]],
                    ignore_index=True)

    random_test = key[(key["set"] == "test") & (key["stratum"] == "random")]
    chosen = set(random_test.sample(N_RANDOM, random_state=SEED)["item"])
    key["human_set"] = (
        key["set"].eq("pool")
        | (key["set"].eq("test") & key["stratum"].isin(["lexicon", "burst"]))
        | key["item"].isin(chosen)
    )
    key["pool_draw"] = key["pool_draw"].astype("Int64")
    key.to_csv(DATA_DIR / "items_key.csv", index=False)

    with SHEET.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
        for row in new_pool.itertuples():
            writer.writerow({"item": row.item, "title": row.title, "label": "",
                             "direction": "", "note": ""})

    human = key[key["human_set"]]
    print(human.groupby(["set", "stratum"]).size().to_string())
    print(f"human set: {len(human)} items; new pool items {new_pool['item'].min()}-"
          f"{new_pool['item'].max()}")


if __name__ == "__main__":
    main()
