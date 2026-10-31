"""Score headlines with local LLMs (MLX, 4-bit) as P(Yes) / (P(Yes) + P(No)).

The score is read from the first answer token; nothing is generated.
Instructions are always in English; the headline is Turkish ("tr"), MT English
("en") or the model's own translation ("en-self"). Few-shot runs put five
example headlines from demos/<model>__<lang>.json before the target. The shared
prompt prefix is computed once and kept in the KV cache.

Output: results/predictions/<model>/<lang>__<run>.csv
(item, p_yes, log_odds, answer_mass). Items already in a file are skipped.

Usage:
    python scripts/05_score.py --model qwen2.5-3b --lang tr en
    python scripts/05_score.py --model qwen2.5-3b --few-shot --human-set
"""

import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from mlx_lm import load
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache

MODELS = {
    "qwen2.5-3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "llama3.2-3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "gemma3-4b": "mlx-community/gemma-3-text-4b-it-4bit",
}

INSTRUCTION = (
    "You classify headlines from Turkish financial news. A headline is high-impact "
    "if it reports a concrete, new event that would noticeably move the share price "
    "of a specific listed company or the Turkish stock market, in either direction. "
    "Commentary, forecasts, price reports, lists of stocks, routine disclosures and "
    "small events are not high-impact.\n\n"
    "Is the headline high-impact? Answer with Yes or No."
)
TARGET = "@@HEADLINE@@"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEMO_DIR = ROOT / "demos"
PRED_DIR = ROOT / "results" / "predictions"


def build_prompt(examples: list[tuple[str, str]]) -> str:
    """User message with optional examples; TARGET marks the headline slot."""
    parts = [INSTRUCTION, ""]
    if examples:
        parts.append("Examples:")
        for headline, answer in examples:
            parts += [f"Headline: {headline}", f"Answer: {answer}", ""]
    parts.append(f"Headline: {TARGET}")
    return "\n".join(parts)


def answer_token_ids(tokenizer, word: str) -> list[int]:
    """First-token ids of the answer word with and without a leading space."""
    ids = {tokenizer.encode(v, add_special_tokens=False)[0] for v in (word, " " + word)}
    return sorted(ids)


class Scorer:
    def __init__(self, model_name: str):
        self.model, self.tokenizer = load(MODELS[model_name])
        self.yes_ids = answer_token_ids(self.tokenizer, "Yes")
        self.no_ids = answer_token_ids(self.tokenizer, "No")

    def render(self, user_message: str) -> str:
        messages = [{"role": "user", "content": user_message}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def last_logprobs(self, ids: list[int], cache) -> mx.array:
        logits = self.model(mx.array(ids)[None], cache=cache)[0, -1]
        return logits - mx.logsumexp(logits)

    def score_run(self, prompt: str, headlines: list[str], answer_prefix: str = "") -> pd.DataFrame:
        # With examples, models copy their "Answer:" line before answering, so
        # few-shot replies are started with it and Yes/No is the next token.
        rendered = self.render(prompt) + answer_prefix
        # The prefix stops before the space that precedes the headline, because
        # most tokenizers attach a leading space to the following word.
        prefix_ids = self.encode(rendered.split(" " + TARGET)[0])
        cache = make_prompt_cache(self.model)
        self.last_logprobs(prefix_ids, cache)
        use_cache = can_trim_prompt_cache(cache)

        rows = []
        for headline in headlines:
            ids = self.encode(rendered.replace(TARGET, headline))
            if use_cache and ids[: len(prefix_ids)] == prefix_ids:
                suffix = ids[len(prefix_ids):]
                logprobs = self.last_logprobs(suffix, cache)
                trim_prompt_cache(cache, len(suffix))
            else:
                logprobs = self.last_logprobs(ids, make_prompt_cache(self.model))
            log_yes = mx.logsumexp(logprobs[mx.array(self.yes_ids)])
            log_no = mx.logsumexp(logprobs[mx.array(self.no_ids)])
            rows.append((log_yes.item(), log_no.item()))

        df = pd.DataFrame(rows, columns=["log_yes", "log_no"])
        df["log_odds"] = df["log_yes"] - df["log_no"]
        df["p_yes"] = 1 / (1 + np.exp(-df["log_odds"]))
        # Low values mean the model wanted to start its reply with another token.
        df["answer_mass"] = np.exp(df["log_yes"]) + np.exp(df["log_no"])
        return df[["p_yes", "log_odds", "answer_mass"]]


LANGS = ["tr", "en", "en-self"]


def load_headlines(lang: str, model_name: str) -> pd.DataFrame:
    if lang == "en-self":
        items = pd.read_csv(DATA_DIR / "self_translations" / f"{model_name}.csv")
        return items[["item"]].assign(headline=items["title_en"].fillna(""))
    items = pd.read_csv(DATA_DIR / "translations.csv")
    column = "title" if lang == "tr" else "title_en"
    return items[["item"]].assign(headline=items[column])


def runs_for(model_name: str, lang: str, demo_dir: Path | None) -> dict[str, list]:
    if demo_dir is None:
        return {"zero-shot": []}
    demo_file = demo_dir / f"{model_name}__{lang}.json"
    sets = json.loads(demo_file.read_text())
    return {s["name"]: [(e["headline"], e["answer"]) for e in s["examples"]] for s in sets}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score headlines with an MLX model.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--lang", nargs="+", choices=LANGS, default=["tr", "en"])
    parser.add_argument("--few-shot", action="store_true")
    parser.add_argument("--demo-dir", type=Path, default=DEMO_DIR)
    parser.add_argument("--limit", type=int, help="score only the first N items (testing)")
    parser.add_argument("--human-set", action="store_true",
                        help="score only the human-set items (03b); rerun without it for the rest")
    args = parser.parse_args()

    scorer = Scorer(args.model)
    out_dir = PRED_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_dir = args.demo_dir if args.few_shot else None
    for lang in args.lang:
        items = load_headlines(lang, args.model)
        if args.human_set:
            key = pd.read_csv(DATA_DIR / "items_key.csv")
            items = items[items["item"].isin(key.loc[key["human_set"], "item"])]
        if args.limit:
            items = items.head(args.limit)
        for run, examples in runs_for(args.model, lang, demo_dir).items():
            out_path = out_dir / f"{lang}__{run}.csv"
            todo, done = items, None
            if out_path.exists() and not args.limit:
                # Score only items added since the file was written.
                done = pd.read_csv(out_path)
                todo = items[~items["item"].isin(done["item"])]
                if todo.empty:
                    print(f"skip {out_path.name}")
                    continue
            start = time.time()
            answer_prefix = "Answer:" if examples else ""
            scores = scorer.score_run(build_prompt(examples), todo["headline"].tolist(),
                                      answer_prefix)
            result = pd.concat([todo[["item"]].reset_index(drop=True), scores], axis=1)
            if not args.limit:
                # Write, then rename, so stopping the run can't leave a partial file.
                tmp_path = out_path.with_suffix(".tmp")
                pd.concat([done, result], ignore_index=True).to_csv(tmp_path, index=False)
                os.replace(tmp_path, out_path)
            print(f"{args.model} {lang} {run}: {len(todo)} items in {time.time() - start:.0f}s, "
                  f"mean answer mass {result['answer_mass'].mean():.2f}")


if __name__ == "__main__":
    main()
