"""Export headlines to frontier LLMs and import their Yes/No answers.

The frontier models are used through agent interfaces, so each run gets its own
folder frontier/<labeller>_<lang>/ with only the renumbered headlines and the
instructions; the model writes answers.csv there. The mapping back to items
(frontier/mapping.csv) stays outside the run folders.

Usage:
    python scripts/09_frontier.py export sonnet
    python scripts/09_frontier.py import sonnet --model-id claude-sonnet-5.5
"""

import argparse
import importlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FRONTIER_DIR = ROOT / "frontier"
PRED_DIR = ROOT / "results" / "predictions"
LANGS = {"tr": "title", "en": "title_en"}

INSTRUCTION = importlib.import_module("05_score").INSTRUCTION

TEMPLATE = """# Task

{instruction}

## What to do

1. Read `headlines.csv` in this folder. It has {n} rows with columns `n` and
   `headline`. The headlines are in {language}.
2. Decide for **every** headline yourself, reading it and judging it: `Yes` if it
   is high-impact, `No` if not. Do not write a program, keyword rule or any other
   automatic method to make the decisions. You may use code only to read and
   write files.
3. Write `answers.csv` in this folder with exactly two columns, `n,answer`, one
   row per headline, in the same order, with `answer` either `Yes` or `No`.
4. Work in batches (for example 100 headlines at a time) and append to
   `answers.csv` as you go, so no headline is skipped. At the end, check that it
   has {n} rows plus the header and that every `n` from 1 to {n} appears once.

## Rules

- Work only inside this folder. Do not open, read, create or change any file
  outside it.
- Do not look anything up and do not use any other information about these
  headlines. Judge each one on its own text.
- Do not edit `headlines.csv` or this file.
"""


def export(labellers: list[str]) -> None:
    items = pd.read_csv(DATA_DIR / "items_key.csv")
    translations = pd.read_csv(DATA_DIR / "translations.csv")
    test = items.loc[items["set"] == "test", ["item"]].merge(translations, on="item")
    test = test.sort_values("item").reset_index(drop=True)
    test.insert(0, "n", range(1, len(test) + 1))

    FRONTIER_DIR.mkdir(exist_ok=True)
    test[["n", "item"]].to_csv(FRONTIER_DIR / "mapping.csv", index=False)

    for labeller in labellers:
        for lang, column in LANGS.items():
            run_dir = FRONTIER_DIR / f"{labeller}_{lang}"
            run_dir.mkdir(exist_ok=True)
            test[["n", column]].rename(columns={column: "headline"}).to_csv(
                run_dir / "headlines.csv", index=False
            )
            language = "Turkish" if lang == "tr" else "English (machine-translated from Turkish)"
            (run_dir / "INSTRUCTIONS.md").write_text(
                TEMPLATE.format(instruction=INSTRUCTION, n=len(test), language=language)
            )
            print(f"wrote {run_dir.relative_to(ROOT)}")


def import_answers(labeller: str, model_id: str) -> None:
    mapping = pd.read_csv(FRONTIER_DIR / "mapping.csv")
    out_dir = PRED_DIR / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in LANGS:
        path = FRONTIER_DIR / f"{labeller}_{lang}" / "answers.csv"
        if not path.exists():
            print(f"missing {path.relative_to(ROOT)}")
            continue
        answers = pd.read_csv(path)
        answers["answer"] = answers["answer"].astype(str).str.strip().str.capitalize()
        problems = []
        if sorted(answers["n"]) != list(range(1, len(mapping) + 1)):
            problems.append("n does not cover 1..N exactly once")
        if not answers["answer"].isin(["Yes", "No"]).all():
            problems.append("answers other than Yes/No")
        if problems:
            raise SystemExit(f"{path.relative_to(ROOT)}: " + "; ".join(problems))

        result = mapping.merge(answers, on="n")
        result["p_yes"] = (result["answer"] == "Yes").astype(float)
        result[["item", "p_yes"]].assign(log_odds=float("nan"), answer_mass=float("nan")).to_csv(
            out_dir / f"{lang}__zero-shot.csv", index=False
        )
        print(f"{model_id} {lang}: {int(result['p_yes'].sum())} of {len(result)} labelled Yes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frontier-model labelling runs.")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("labellers", nargs="+")
    imp = sub.add_parser("import")
    imp.add_argument("labeller")
    imp.add_argument("--model-id", required=True, help="exact model name, used as folder name")
    args = parser.parse_args()

    if args.command == "export":
        export(args.labellers)
    else:
        import_answers(args.labeller, args.model_id)


if __name__ == "__main__":
    main()
