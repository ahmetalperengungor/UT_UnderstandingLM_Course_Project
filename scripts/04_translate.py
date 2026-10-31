"""Translate the headlines into English with opus-mt-tc-big-tr-en.

Company names are replaced by placeholders (ENT1, ENT2, ...) before translation
and restored afterwards, so that "Şok Marketler" does not become "Shock
Markets". Names come from company_names.py and from the capitalised words
before a ticker in parentheses. Only items missing from data/translations.csv
are translated. The first run also writes labeling/translation_check.csv, a
sample of 50 test translations for a manual adequacy check.

Usage:
    python scripts/04_translate.py
    python scripts/04_translate.py --spot-check-only
"""

import re
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer

from company_names import ALIASES

MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-tr-en"
N_SPOT_CHECK = 50
SEED = 20261029
BATCH_SIZE = 32
NUM_BEAMS = 4

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LABEL_DIR = ROOT / "labeling"

UPPER = "A-ZÇĞİÖŞÜ"
NOT_LETTER_AFTER = r"(?![\w])"
NAME_BEFORE_TICKER = re.compile(
    rf"(?:[{UPPER}0-9][\w.&-]*\s){{0,3}}[{UPPER}0-9][\w.&-]*"
    rf"(?=(?:['’]\w+)?\s*\([{UPPER}0-9]{{3,6}}\))"
)
LEADING_ABLATIVE = re.compile(r"(?:\w+(?:dan|den|tan|ten)\s)+(?=\S)")
ALIAS_PATTERNS = [
    re.compile(rf"(?<![\w]){re.escape(alias)}{NOT_LETTER_AFTER}")
    for alias in sorted(
        {a for aliases in ALIASES.values() for a in aliases if not a.isupper()},
        key=len,
        reverse=True,  # "Koza Altın" before "Koza"
    )
]


def mask_names(title: str) -> tuple[str, list[str]]:
    """Replace company names with ENT1, ENT2, ... and return the names."""
    spans = [m.span() for p in ALIAS_PATTERNS for m in p.finditer(title)]
    for m in NAME_BEFORE_TICKER.finditer(title):
        # "Bakanlıktan Türk Traktör'e (TTRAK)": a leading ablative word
        # ("from the Ministry") is not part of the name.
        lead = LEADING_ABLATIVE.match(m.group())
        spans.append((m.start() + (lead.end() if lead else 0), m.end()))

    # Keep the longest span wherever two overlap.
    chosen = []
    for start, end in sorted(spans, key=lambda s: (s[0], -(s[1] - s[0]))):
        if chosen and start < chosen[-1][1]:
            continue
        chosen.append((start, end))

    names, masked, last = [], [], 0
    for start, end in chosen:
        names.append(title[start:end])
        masked.append(title[last:start] + f"ENT{len(names)}")
        last = end
    masked.append(title[last:])
    return "".join(masked), names


def restore_names(text: str, names: list[str]) -> tuple[str, bool]:
    """Put the original names back; report whether every placeholder survived."""
    complete = True
    for i in range(len(names), 0, -1):  # ENT10 before ENT1
        if f"ENT{i}" not in text:
            complete = False
        text = text.replace(f"ENT{i}", names[i - 1])
    return text, complete


def translate(texts: list[str]) -> list[str]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(device).eval()

    out = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = tokenizer(texts[start:start + BATCH_SIZE], return_tensors="pt", padding=True)
        with torch.no_grad():
            generated = model.generate(**batch.to(device), num_beams=NUM_BEAMS, max_new_tokens=128)
        out.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return out


def write_spot_check(translations: pd.DataFrame) -> None:
    key = pd.read_csv(DATA_DIR / "items_key.csv")
    test_items = key.loc[key["set"] == "test", "item"]
    sample = translations[translations["item"].isin(test_items)]
    sample = sample.sample(N_SPOT_CHECK, random_state=SEED).sort_values("item")
    sample[["item", "title", "title_en"]].assign(adequate="", note="").to_csv(
        LABEL_DIR / "translation_check.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    if "--spot-check-only" in sys.argv:
        write_spot_check(pd.read_csv(DATA_DIR / "translations.csv"))
        return

    items = pd.read_csv(DATA_DIR / "items_key.csv")
    out_path = DATA_DIR / "translations.csv"
    done = pd.read_csv(out_path) if out_path.exists() else None
    if done is not None:  # translate only items added since (e.g. the second pool draw)
        items = items[~items["item"].isin(done["item"])].reset_index(drop=True)
        if items.empty:
            print("all items already translated")
            return
    masked = items["title"].map(mask_names)
    items["title_masked"] = masked.str[0]
    names = masked.str[1]

    raw = translate(items["title_masked"].tolist())
    restored = [restore_names(text, n) for text, n in zip(raw, names)]
    items["title_en"] = [text for text, _ in restored]
    items["names_restored"] = [ok for _, ok in restored]

    cols = ["item", "id", "title", "title_masked", "title_en", "names_restored"]
    if done is None:
        items[cols].to_csv(out_path, index=False)
        write_spot_check(items[cols])
    else:
        pd.concat([done, items[cols]], ignore_index=True).to_csv(out_path, index=False)

    n_masked = (names.str.len() > 0).sum()
    n_lost = (~items["names_restored"]).sum()
    print(f"{len(items)} headlines, {n_masked} with masked names, {n_lost} lost a placeholder")


if __name__ == "__main__":
    main()
