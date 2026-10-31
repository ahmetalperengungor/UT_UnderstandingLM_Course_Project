"""Write the LaTeX results tables to paper/tables/ from results/metrics.csv.

Usage:
    python scripts/10_tables.py
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "paper" / "tables"

MODELS = {"qwen2.5-3b": "Qwen2.5-3B", "llama3.2-3b": "Llama-3.2-3B", "gemma3-4b": "Gemma-3-4B"}
FRONTIER = {"claude-opus-5.5": "Claude Opus 5.5", "claude-sonnet-5.5": "Claude Sonnet 5.5",
            "gemini-flash-3.8": "Gemini Flash 3.8"}
LANGS = {"tr": "TR", "en": "EN (MT)", "en-self": "EN (self)"}
CONDITIONS = ["zero-shot", "gold", "self", "random"]


def fmt(x: float) -> str:
    """Two decimals without the leading zero, as is usual in ACL tables."""
    if pd.isna(x):
        return "--"
    text = f"{abs(x):.2f}".removeprefix("0")
    return f"$-${text}" if x < 0 else text


def zero_shot_table(metrics: pd.DataFrame) -> str:
    df = metrics[(metrics["subset"] == "all")
                 & metrics["condition"].isin(["zero-shot", "keywords"])]
    systems = [*MODELS, *sorted(s for s in df["system"].unique()
                                if s not in MODELS and s != "lexicon"), "lexicon"]
    n = int(df["n"].max())
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"System & Input & F1 [95\% CI] & $\kappa$ & AUPRC \\", r"\midrule",
    ]
    for system in systems:
        rows = df[df["system"] == system]
        name = MODELS.get(system) or FRONTIER.get(system, "Keyword lexicon")
        for i, lang in enumerate(l for l in LANGS if l in set(rows["lang"])):
            r = rows[rows["lang"] == lang].iloc[0]
            # Frontier LLMs give Yes/No answers only, so they have no ranking.
            auprc = "--" if system not in MODELS and system != "lexicon" else fmt(r["auprc"])
            lines.append(
                f"{name if i == 0 else ''} & {LANGS[lang]} & {fmt(r['f1'])} "
                f"[{fmt(r['f1_ci_low'])}, {fmt(r['f1_ci_high'])}] & {fmt(r['kappa'])} & {auprc} \\\\"
            )
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    positives = int(df["positives"].max())
    lines += [
        r"\end{tabular}",
        rf"\caption{{Zero-shot results on all {n} labelled test headlines ({positives} "
        r"high-impact). The frontier models give Yes/No answers only, so they have no AUPRC.}",
        r"\label{tab:zero-shot}", r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def few_shot_table(metrics: pd.DataFrame) -> str:
    df = metrics[(metrics["subset"] == "common") & metrics["system"].isin(MODELS)]
    n = int(df["n"].max())
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\begin{tabular}{ll" + "rr" * len(CONDITIONS) + "}", r"\toprule",
        r" & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{c.capitalize()}}}" for c in CONDITIONS) + r" \\",
        " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(CONDITIONS))),
        r"Model & Input & " + " & ".join(["F1 & AUPRC"] * len(CONDITIONS)) + r" \\",
        r"\midrule",
    ]
    for system, name in MODELS.items():
        for i, lang in enumerate(["tr", "en"]):
            cells = []
            for condition in CONDITIONS:
                r = df[(df["system"] == system) & (df["lang"] == lang) & (df["condition"] == condition)]
                cells += [fmt(r["f1"].iloc[0]), fmt(r["auprc"].iloc[0])] if len(r) else ["--", "--"]
            lines.append(f"{name if i == 0 else ''} & {LANGS[lang]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    positives = int(df["positives"].max())
    lines += [
        r"\end{tabular}",
        rf"\caption{{Prompting conditions on the {n} test headlines scored in every condition "
        rf"({positives} high-impact). Few-shot predictions are the majority vote over three "
        r"example sets; AUPRC uses the mean $p$.}",
        r"\label{tab:few-shot}", r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write LaTeX results tables.")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    metrics = pd.read_csv(args.results / "metrics.csv")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "zero_shot.tex").write_text(zero_shot_table(metrics))
    (args.out / "few_shot.tex").write_text(few_shot_table(metrics))
    print(f"wrote {args.out / 'zero_shot.tex'} and {args.out / 'few_shot.tex'}")


if __name__ == "__main__":
    main()
