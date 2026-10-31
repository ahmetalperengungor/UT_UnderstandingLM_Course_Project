"""Self-translation (Etxaniz et al., 2024): each model translates the headlines.

4-shot plain-text prompt ("Turkish: ... / English: ...") with the first four
FLORES-200 dev pairs of at most 100 characters, greedy decoding, stopping at the
end of the line. Company names are not masked, as in the paper.

Output: data/self_translations/<model>.csv

Usage:
    python scripts/04b_self_translate.py --model qwen2.5-3b
"""

import argparse
import importlib
import time
from pathlib import Path

import pandas as pd
from mlx_lm import load, stream_generate

MODELS = importlib.import_module("05_score").MODELS

# FLORES-200 dev, lines 4-7 (Turkish tur_Latn, English eng_Latn).
FLORES_EXAMPLES = [
    ("Pilot, Filo Lideri Dilokrit Pattavee olarak belirlendi.",
     "The pilot was identified as Squadron Leader Dilokrit Pattavee."),
    ("Yerel basında bir havaalanı yangın söndürme aracının olaya müdahale ederken devrildiği belirtildi.",
     "Local media reports an airport fire vehicle rolled over while responding."),
    ("28 yaşındaki Vidal, üç sezon önce Sevilla'dan Barça'ya transfer olmuştu.",
     "28-year-old Vidal had joined Barça three seasons ago, from Sevilla."),
    ("Katalan başkentine taşındığından beri Vidal kulüp adına 49 maça çıktı.",
     "Since moving to the Catalan-capital, Vidal had played 49 games for the club."),
]
MAX_TOKENS = 80

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "self_translations"


def build_prompt(headline: str) -> str:
    lines = []
    for turkish, english in FLORES_EXAMPLES:
        lines += [f"Turkish: {turkish}", f"English: {english}"]
    lines += [f"Turkish: {headline}", "English:"]
    return "\n".join(lines)


def translate(model, tokenizer, headline: str) -> str:
    """Greedy decoding until the first newline."""
    text = ""
    for chunk in stream_generate(model, tokenizer, build_prompt(headline), max_tokens=MAX_TOKENS):
        text += chunk.text
        if "\n" in text:
            break
    return text.split("\n")[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate headlines with the LLM itself.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--limit", type=int, help="translate only the first N items (testing)")
    args = parser.parse_args()

    items = pd.read_csv(DATA_DIR / "translations.csv")[["item", "title"]]
    if args.limit:
        items = items.head(args.limit)

    model, tokenizer = load(MODELS[args.model])
    start = time.time()
    items["title_en"] = [translate(model, tokenizer, t) for t in items["title"]]
    print(f"{args.model}: {len(items)} headlines in {time.time() - start:.0f}s, "
          f"{(items['title_en'] == '').sum()} empty")

    if args.limit:
        for row in items.itertuples():
            print(f"  {row.title}\n  -> {row.title_en}")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items[["item", "title_en"]].to_csv(OUT_DIR / f"{args.model}.csv", index=False)


if __name__ == "__main__":
    main()
