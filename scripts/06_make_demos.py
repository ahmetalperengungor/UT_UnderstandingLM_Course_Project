"""Build the few-shot example sets from the labelled pool.

Three sets (seeds 1-3) of five pool headlines, two labelled high-impact and
three not. Each set is shown with three kinds of answers on the same headlines:
gold (my labels), self (the model's own zero-shot answers) and random
(Min et al., 2022).

Output: demos/<model>__<lang>.json

Usage:
    python scripts/06_make_demos.py --model qwen2.5-3b
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_POSITIVE, N_NEGATIVE = 2, 3
SEEDS = [1, 2, 3]
THRESHOLD = 0.5

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEMO_DIR = ROOT / "demos"
PRED_DIR = ROOT / "results" / "predictions"


def answer(flag: bool) -> str:
    return "Yes" if flag else "No"


def load_pool(labels_path: Path) -> pd.DataFrame:
    key = pd.read_csv(DATA_DIR / "items_key.csv")
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    translations = pd.read_csv(DATA_DIR / "translations.csv")
    pool = key[key["set"] == "pool"][["item"]]
    pool = pool.merge(labels[["item", "label"]], on="item")
    pool = pool.merge(translations[["item", "title", "title_en"]], on="item")
    if pool["label"].isna().any():
        raise SystemExit("Label all few-shot pool items in labeling/label_sheet.csv first.")
    pool["label"] = pool["label"].astype(int)
    if pool["label"].sum() < N_POSITIVE or (1 - pool["label"]).sum() < N_NEGATIVE:
        raise SystemExit("The pool has too few positives or negatives for an example set.")
    return pool


def make_sets(pool: pd.DataFrame, model: str, lang: str) -> list[dict]:
    zero_shot = pd.read_csv(PRED_DIR / model / f"{lang}__zero-shot.csv")
    pool = pool.merge(zero_shot[["item", "p_yes"]], on="item")
    headline_col = "title" if lang == "tr" else "title_en"

    sets = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        pos = pool[pool["label"] == 1].sample(N_POSITIVE, random_state=seed)
        neg = pool[pool["label"] == 0].sample(N_NEGATIVE, random_state=seed)
        chosen = pd.concat([pos, neg]).sample(frac=1, random_state=seed)
        random_labels = rng.random(len(chosen)) < 0.5

        sources = {
            "gold": chosen["label"] == 1,
            "self": chosen["p_yes"] > THRESHOLD,
            "random": pd.Series(random_labels, index=chosen.index),
        }
        for source, flags in sources.items():
            examples = [
                {"item": int(row["item"]), "headline": row[headline_col], "answer": answer(flag)}
                for (_, row), flag in zip(chosen.iterrows(), flags)
            ]
            sets.append({"name": f"{source}-s{seed}", "examples": examples})
    return sets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build few-shot example sets.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang", nargs="+", choices=["tr", "en"], default=["tr", "en"])
    parser.add_argument("--labels", type=Path, default=ROOT / "labeling" / "label_sheet.csv")
    parser.add_argument("--out", type=Path, default=DEMO_DIR)
    args = parser.parse_args()

    pool = load_pool(args.labels)
    args.out.mkdir(parents=True, exist_ok=True)
    for lang in args.lang:
        sets = make_sets(pool, args.model, lang)
        path = args.out / f"{args.model}__{lang}.json"
        path.write_text(json.dumps(sets, indent=2, ensure_ascii=False) + "\n")
        print(f"{path.name}: {len(sets)} example sets")


if __name__ == "__main__":
    main()
