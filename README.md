# Prompting small LLMs for Turkish financial event detection

Final project for *Understanding Language Models*  
Ahmet Alperen Güngör · Matriculation number 7264397

## Abstract

In this project I aim to test whether small open LLMs (3 to 4B parameters) manage to
detect high impact events in Turkish financial news headlines better when the headline
is translated into English, and when we perform in-context learning (the prompt
contains five examples). On 850 headlines from two years of news that I have scraped
and labelled as a native Turkish speaker, I try to make three comparisons:

1. Turkish input against English input from an external translation system or from the
   model itself;
2. zero-shot against few-shot prompting;
3. the small models against a keyword lexicon and three frontier LLMs.

I found that neither translation nor examples helped consistently: both mainly shifted
how often a model answered YES, while its ranking ability (AUPRC) didn't change much.
The Claude models agreed with my labels best but were much stricter than me, and the best
small model matched the lexicon on the full set and exceeded it on the random sample.

Details about the data, methodology and results can be found in the paper
([`paper.pdf`](paper.pdf)).

## Research questions

Small open LLMs (3–4B, 4-bit, run locally) decide whether a Turkish financial news
headline reports a **high-impact event**. I compare:

- **RQ1**: the Turkish headline vs. an English translation (external MT, or the model's
  own translation as in Etxaniz et al., 2024);
- **RQ2**: zero-shot vs. five examples in the prompt, and whether the example answers are
  my labels, the model's own answers or random (after Min et al., 2022);
- **RQ3**: agreement with my labels, compared with a keyword lexicon and three frontier
  LLMs (Gemini 3.8 Flash, Claude Opus 5.5 and Sonnet 5.5).

## Repository layout

```
scripts/
  01_build_snapshot.py     cleaned headline snapshot from the news database
  02_sample_random.py      random test stratum: one headline per day, outlet rotating
  03_build_label_sheets.py lexicon and burst strata, few-shot pool, blinded sheet
  03b_human_set.py         core set (labelled first) and second pool draw
  04_translate.py          Turkish -> English (opus-mt-tc-big-tr-en, names masked)
  04b_self_translate.py    each LLM translates the headlines itself
  05_score.py              LLM scoring: P(Yes) for every headline and condition
  06_make_demos.py         few-shot example sets (gold / self / random answers)
  07_evaluate.py           metrics, bootstrap CIs, McNemar tests
  08_figures.py            figures
  09_frontier.py           blinded labelling runs for the frontier LLMs
  10_tables.py             LaTeX results tables
  label.py                 terminal labelling tool
  lexicon.py               keyword lexicon
  company_names.py         BIST company names, protected during translation
labeling/                  guideline, labels, translation check sheet
data/                      item key, translations, self-translations, snapshot info
demos/                     few-shot example sets per model and language
frontier/                  frontier LLM run folders (instructions, headlines, answers)
results/                   predictions, metrics, tests, figures
paper.pdf                  the paper
proposal.tex               original project proposal
```

## Running

Requires Apple silicon for the LLMs (MLX). Python 3.12.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate

# data (only needed to rebuild from the database)
python scripts/01_build_snapshot.py --db path/to/news_database.sqlite
python scripts/02_sample_random.py
python scripts/03_build_label_sheets.py
python scripts/03b_human_set.py

python scripts/label.py --human-set
python scripts/04_translate.py

for m in qwen2.5-3b llama3.2-3b gemma3-4b; do
  python scripts/04b_self_translate.py --model $m
  python scripts/05_score.py --model $m --lang tr en en-self
  python scripts/06_make_demos.py --model $m
  python scripts/05_score.py --model $m --few-shot --human-set
done

# frontier LLMs: answers come from agent sessions, so only the import is scripted
# (09_frontier.py export <labeller> writes the run folders for a new session)
python scripts/09_frontier.py import claude --model-id claude-opus-5.5
python scripts/09_frontier.py import sonnet --model-id claude-sonnet-5.5
python scripts/09_frontier.py import gemini --model-id gemini-flash-3.8

python scripts/07_evaluate.py
python scripts/08_figures.py
python scripts/10_tables.py
```

Models are downloaded from the Hugging Face Hub on first use:
`mlx-community/Qwen2.5-3B-Instruct-4bit`, `mlx-community/Llama-3.2-3B-Instruct-4bit`,
`mlx-community/gemma-3-text-4b-it-4bit` and `Helsinki-NLP/opus-mt-tc-big-tr-en`.

