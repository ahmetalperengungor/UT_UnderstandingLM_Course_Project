"""Evaluate all predictions against the labels and run the McNemar tests.

A few-shot condition predicts by majority vote over its three example sets and
scores by the mean P(Yes). The keyword lexicon is a baseline (score = number of
matching patterns).

Tests (exact McNemar on per-item correctness, Holm-corrected per family):
    RQ1   Turkish vs English (MT and self-translated)
    RQ2a  zero-shot vs gold examples
    RQ2b  gold vs random and gold vs self answers

Outputs in results/: metrics.csv, tests.csv, recall_by_direction.csv.
Metrics are given on every labelled item a condition scored ("all") and on the
items scored by every condition ("common"), each also for the random stratum.

Usage:
    python scripts/07_evaluate.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from lexicon import lexicon_score

THRESHOLD = 0.5
N_BOOTSTRAP = 1000
FEW_SHOT_SOURCES = ["gold", "self", "random"]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PRED_DIR = ROOT / "results" / "predictions"
OUT_DIR = ROOT / "results"


def load_gold(labels_path: Path) -> pd.DataFrame:
    key = pd.read_csv(DATA_DIR / "items_key.csv")
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    test = key[key["set"] == "test"]
    gold = test.merge(labels[["item", "label", "direction"]], on="item")
    missing = gold["label"].isna().sum()
    if missing:
        print(f"note: {missing} test items are unlabelled and left out")
    gold = gold.dropna(subset=["label"])
    gold["label"] = gold["label"].astype(int)
    return gold


def model_conditions(runs: dict[str, pd.DataFrame], lang: str):
    """Yield (condition, frame with item/score/pred) for one model and language."""
    zero_shot = runs.get(f"{lang}__zero-shot")
    if zero_shot is not None:
        yield "zero-shot", pd.DataFrame({
            "item": zero_shot["item"],
            "score": zero_shot["p_yes"],
            "pred": zero_shot["p_yes"] > THRESHOLD,
        })
    for source in FEW_SHOT_SOURCES:
        seeds = [df for name, df in runs.items() if name.startswith(f"{lang}__{source}-s")]
        if not seeds:
            continue
        stacked = pd.concat(seeds)
        by_item = stacked.assign(yes=stacked["p_yes"] > THRESHOLD).groupby("item")
        complete = (by_item.size() == len(seeds)).values  # scored with every example set
        yield source, pd.DataFrame({
            "item": by_item.size().index[complete],
            "score": by_item["p_yes"].mean().values[complete],
            "pred": (by_item["yes"].mean() > 0.5).values[complete],
        })


def load_conditions(gold: pd.DataFrame) -> pd.DataFrame:
    """One row per (system, lang, condition, item) with score and prediction."""
    frames = []
    for model_dir in sorted(PRED_DIR.iterdir()):
        runs = {p.stem: pd.read_csv(p) for p in model_dir.glob("*.csv")}
        for lang in ("tr", "en", "en-self"):
            for condition, df in model_conditions(runs, lang):
                frames.append(df.assign(system=model_dir.name, lang=lang, condition=condition))

    lexicon = gold[["item"]].assign(
        system="lexicon", lang="tr", condition="keywords",
        score=gold["title"].map(lexicon_score).values,
    )
    lexicon["pred"] = lexicon["score"] > 0
    frames.append(lexicon)

    preds = pd.concat(frames, ignore_index=True)[
        ["system", "lang", "condition", "item", "score", "pred"]
    ]
    preds = preds.merge(gold[["item", "label"]], on="item")
    preds["pred"] = preds["pred"].astype(int)
    preds["correct"] = (preds["pred"] == preds["label"]).astype(int)
    return preds


def bootstrap_ci(y, p, metric, rng) -> tuple[float, float]:
    n = len(y)
    values = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if y[idx].min() != y[idx].max():
            values.append(metric(y[idx], p[idx]))
    return tuple(np.percentile(values, [2.5, 97.5]))


def summarise(df: pd.DataFrame, rng) -> dict:
    y, p, s = df["label"].to_numpy(), df["pred"].to_numpy(), df["score"].to_numpy()
    f1 = lambda a, b: f1_score(a, b, zero_division=0)
    f1_lo, f1_hi = bootstrap_ci(y, p, f1, rng)
    k_lo, k_hi = bootstrap_ci(y, p, cohen_kappa_score, rng)
    return {
        "n": len(df),
        "positives": int(y.sum()),
        "predicted_positive": int(p.sum()),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "f1": f1(y, p),
        "f1_ci_low": f1_lo,
        "f1_ci_high": f1_hi,
        "kappa": cohen_kappa_score(y, p),
        "kappa_ci_low": k_lo,
        "kappa_ci_high": k_hi,
        "auroc": roc_auc_score(y, s),
        "auprc": average_precision_score(y, s),
        "accuracy": (y == p).mean(),
    }


def common_items(preds: pd.DataFrame) -> set:
    """Items scored in every condition (the few-shot runs may cover fewer)."""
    groups = preds.groupby(["system", "lang", "condition"])["item"]
    return set.intersection(*(set(items) for _, items in groups))


def metrics_table(preds: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Metrics per condition on every item it scored ("all", "random"), and on
    the items common to all conditions ("common", "common-random")."""
    rng = np.random.default_rng(0)
    random_items = set(gold.loc[gold["stratum"] == "random", "item"])
    common = common_items(preds)
    rows = []
    for (system, lang, condition), df in preds.groupby(["system", "lang", "condition"]):
        shared = df[df["item"].isin(common)]
        subsets = {
            "all": df,
            "random": df[df["item"].isin(random_items)],
            "common": shared,
            "common-random": shared[shared["item"].isin(random_items)],
        }
        for subset, part in subsets.items():
            if part["label"].nunique() < 2:
                continue
            rows.append({"system": system, "lang": lang, "condition": condition,
                         "subset": subset, **summarise(part, rng)})
    return pd.DataFrame(rows)


def mcnemar_test(preds, system, a, b) -> dict | None:
    """a and b are (lang, condition) pairs; compares per-item correctness."""
    left = preds[(preds["system"] == system) & (preds["lang"] == a[0]) & (preds["condition"] == a[1])]
    right = preds[(preds["system"] == system) & (preds["lang"] == b[0]) & (preds["condition"] == b[1])]
    if left.empty or right.empty:
        return None
    paired = left[["item", "correct"]].merge(right[["item", "correct"]], on="item")
    table = pd.crosstab(paired["correct_x"], paired["correct_y"]).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0
    )
    result = mcnemar(table.to_numpy(), exact=True)
    return {
        "system": system,
        "a": f"{a[0]}/{a[1]}",
        "b": f"{b[0]}/{b[1]}",
        "n": len(paired),
        "acc_a": paired["correct_x"].mean(),
        "acc_b": paired["correct_y"].mean(),
        "only_a_correct": int(table.loc[1, 0]),
        "only_b_correct": int(table.loc[0, 1]),
        "p_value": result.pvalue,
    }


def tests_table(preds: pd.DataFrame) -> pd.DataFrame:
    # Zero-shot pairs use every labelled item; pairs with a few-shot condition
    # use the items scored in every condition, like the few-shot metrics.
    zero_shot = preds["condition"].isin(["zero-shot", "keywords"])
    preds = preds[zero_shot | preds["item"].isin(common_items(preds))]
    models = [s for s in preds["system"].unique() if s != "lexicon"]
    families = {
        "RQ1": [(m, ("tr", c), ("en", c)) for m in models
                for c in ["zero-shot", *FEW_SHOT_SOURCES]]
               + [(m, a, ("en-self", "zero-shot")) for m in models
                  for a in (("tr", "zero-shot"), ("en", "zero-shot"))],
        "RQ2a": [(m, (l, "zero-shot"), (l, "gold")) for m in models for l in ("tr", "en")],
        "RQ2b": [(m, (l, "gold"), (l, other)) for m in models for l in ("tr", "en")
                 for other in ("random", "self")],
    }
    rows = []
    for rq, comparisons in families.items():
        results = [r for r in (mcnemar_test(preds, *c) for c in comparisons) if r]
        if not results:
            continue
        adjusted = multipletests([r["p_value"] for r in results], method="holm")[1]
        rows += [{"rq": rq, **r, "p_holm": p} for r, p in zip(results, adjusted)]
    return pd.DataFrame(rows)


def recall_by_direction(preds: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    positives = gold[(gold["label"] == 1) & gold["direction"].isin(["up", "down"])]
    merged = preds.merge(positives[["item", "direction"]], on="item")
    table = merged.groupby(["system", "lang", "condition", "direction"])["pred"].agg(
        recall="mean", n="size"
    )
    return table.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions and run tests.")
    parser.add_argument("--labels", type=Path, default=ROOT / "labeling" / "label_sheet.csv")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    gold = load_gold(args.labels)
    preds = load_conditions(gold)
    args.out.mkdir(parents=True, exist_ok=True)

    metrics = metrics_table(preds, gold)
    metrics.to_csv(args.out / "metrics.csv", index=False)
    tests_table(preds).to_csv(args.out / "tests.csv", index=False)
    recall_by_direction(preds, gold).to_csv(args.out / "recall_by_direction.csv", index=False)

    print(f"{len(gold)} labelled test items; {len(common_items(preds))} scored in every condition")
    view = metrics[metrics["subset"] == "all"]
    print(view[["system", "lang", "condition", "n", "precision", "recall", "f1", "kappa", "auprc"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
