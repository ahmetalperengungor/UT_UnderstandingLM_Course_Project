# Task

You classify headlines from Turkish financial news. A headline is high-impact if it reports a concrete, new event that would noticeably move the share price of a specific listed company or the Turkish stock market, in either direction. Commentary, forecasts, price reports, lists of stocks, routine disclosures and small events are not high-impact.

Is the headline high-impact? Answer with Yes or No.

## What to do

1. Read `headlines.csv` in this folder. It has 850 rows with columns `n` and
   `headline`. The headlines are in English (machine-translated from Turkish).
2. Decide for **every** headline yourself, reading it and judging it: `Yes` if it
   is high-impact, `No` if not. Do not write a program, keyword rule or any other
   automatic method to make the decisions. You may use code only to read and
   write files.
3. Write `answers.csv` in this folder with exactly two columns, `n,answer`, one
   row per headline, in the same order, with `answer` either `Yes` or `No`.
4. Work in batches (for example 100 headlines at a time) and append to
   `answers.csv` as you go, so no headline is skipped. At the end, check that it
   has 850 rows plus the header and that every `n` from 1 to 850 appears once.

## Rules

- Work only inside this folder. Do not open, read, create or change any file
  outside it.
- Do not look anything up and do not use any other information about these
  headlines. Judge each one on its own text.
- Do not edit `headlines.csv` or this file.
