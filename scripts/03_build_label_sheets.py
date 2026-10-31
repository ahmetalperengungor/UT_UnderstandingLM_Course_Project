"""Assemble the test set and few-shot pool and write the blinded labelling sheet.

Test set (2024-08-23 to 2026-08-22):
  random   one headline per day (02_sample_random.py)
  lexicon  headlines matching the keyword lexicon, equal numbers per outlet
  burst    one headline from days on which one company appears in at least
           BURST_MIN_HEADLINES headlines (independent of the lexicon and LLMs)

Few-shot pool: the same strata from 2024-01-01 to 2024-08-22, before the test
window.

Outputs: labeling/label_sheet.csv (shuffled, title only), data/items_key.csv

Usage:
    python scripts/03_build_label_sheets.py
"""

from pathlib import Path

import pandas as pd

from lexicon import lexicon_score

TEST_START, TEST_END = "2024-08-23", "2026-08-22"
POOL_START, POOL_END = "2024-01-01", "2024-08-22"
SOURCES = ["ROTABORSA", "DUNYA", "BORSAGUNDEM"]
BURST_MIN_HEADLINES = 4

N_LEXICON_PER_SOURCE_TEST = 20
N_BURST_TEST = 60
N_LEXICON_PER_SOURCE_POOL = 3
N_BURST_POOL = 5
N_RANDOM_POOL = 6
SEED = 20261029

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LABEL_DIR = ROOT / "labeling"


def sample_lexicon(df: pd.DataFrame, per_source: int, seed: int) -> pd.DataFrame:
    hits = df[df["lexicon_score"] > 0]
    parts = [
        hits[hits["source"] == source].sample(per_source, random_state=seed)
        for source in SOURCES
    ]
    return pd.concat(parts).assign(stratum="lexicon")


def sample_bursts(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Pick n random burst days, then one random headline from each."""
    tagged = df.dropna(subset=["ticker"])
    sizes = tagged.groupby(["date", "ticker"]).size()
    bursts = sizes[sizes >= BURST_MIN_HEADLINES].index.to_frame(index=False)
    chosen = bursts.sample(min(n, len(bursts)), random_state=seed)
    rows = tagged.merge(chosen, on=["date", "ticker"])
    rows = rows.groupby(["date", "ticker"]).sample(1, random_state=seed)
    return rows.assign(stratum="burst")


def build_set(df, n_lex, n_burst, random_part, seed, exclude_ids):
    """Combine strata, dropping headlines already drawn by an earlier stratum."""
    df = df[~df["id"].isin(exclude_ids)]
    random_ids = set(random_part["id"])
    lex = sample_lexicon(df[~df["id"].isin(random_ids)], n_lex, seed)
    burst = sample_bursts(df[~df["id"].isin(random_ids | set(lex["id"]))], n_burst, seed)
    return pd.concat([random_part, lex, burst], ignore_index=True)


def main() -> None:
    snapshot = pd.read_csv(DATA_DIR / "snapshot.csv")
    snapshot["lexicon_score"] = snapshot["title"].map(lexicon_score)
    test_df = snapshot[snapshot["date"].between(TEST_START, TEST_END)]
    pool_df = snapshot[snapshot["date"].between(POOL_START, POOL_END)]

    random_test = pd.read_csv(DATA_DIR / "sample_random.csv")
    random_test = random_test.drop(columns="scheduled_source")
    random_test["lexicon_score"] = random_test["title"].map(lexicon_score)
    test = build_set(
        test_df, N_LEXICON_PER_SOURCE_TEST, N_BURST_TEST, random_test, SEED, set()
    ).assign(set="test")

    random_pool = pool_df.sample(N_RANDOM_POOL, random_state=SEED).assign(stratum="random")
    pool = build_set(
        pool_df, N_LEXICON_PER_SOURCE_POOL, N_BURST_POOL, random_pool, SEED, set()
    ).assign(set="pool")

    items = pd.concat([test, pool], ignore_index=True)
    assert items["id"].is_unique, "a headline was drawn twice"
    # Shuffle so the annotator cannot infer set or stratum from the order.
    items = items.sample(frac=1, random_state=SEED).reset_index(drop=True)
    items.insert(0, "item", range(1, len(items) + 1))

    key_cols = ["item", "id", "set", "stratum", "date", "time", "source",
                "ticker", "lexicon_score", "title", "url"]
    items[key_cols].to_csv(DATA_DIR / "items_key.csv", index=False)

    LABEL_DIR.mkdir(exist_ok=True)
    blank = {"label": "", "direction": "", "note": ""}
    # utf-8-sig so Excel and Numbers show Turkish characters correctly.
    items[["item", "title"]].assign(**blank).to_csv(
        LABEL_DIR / "label_sheet.csv", index=False, encoding="utf-8-sig"
    )

    print(items.groupby(["set", "stratum"]).size().to_string())
    print(f"total {len(items)}")


if __name__ == "__main__":
    main()
