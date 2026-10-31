# Labelling guideline: high-impact financial headlines

## Task
You see one Turkish financial news **headline** at a time. Decide whether it reports a
**high-impact event**.

- `label = 1`: high-impact
- `label = 0`: not high-impact

Judge **only what the headline states**. Don't look up the article, and don't use
what you know happened afterwards (e.g. how the price moved).

## Definition
A headline is high-impact when **both** hold:

1. **It reports a concrete, new event**: something that happened or was decided, not
   commentary, a forecast, a price report or a list.
2. **A reasonable investor would expect it to move prices noticeably on the day**:
   either a specific listed company's shares (roughly several percent) or the broad
   Turkish market. **Either direction counts:** a large positive event (big contract,
   acquisition, upgrade) is as much a `1` as a large negative one (arrest, fire,
   downgrade).

## Typical positives (1)
| Category | Examples |
|---|---|
| Legal action against a firm, its executives or major political figures | arrest, detention, raid, indictment, trustee (kayyum) appointment, asset seizure |
| Regulatory sanction or trading halt | trading ban, suspension of trading, licence revocation, large competition fine |
| Corporate distress | bankruptcy, konkordato filing, default, liquidation, mass layoffs, major recall |
| Disaster or operational shock | factory fire or explosion, earthquake hitting operations, production halt, cyber attack |
| Credit rating change | downgrade or upgrade of a company, a bank or Türkiye |
| Major corporate action | acquisition, merger, takeover bid, "dev" contract or order material to the firm, CEO or chair resignation |
| Macro or political shock | unexpected rate decision, war or military attack, coup, terror attack, snap election, sanctions on Türkiye or Turkish firms |

## Typical negatives (0)
- Routine or scheduled disclosures: quarterly results without a stated surprise, regular
  dividend decisions, general assembly notices, investor presentations
- Analyst target prices, recommendations, "which stocks to buy" pieces
- Market reports ("Borsa günü yükselişle tamamladı"), price levels for gold, FX or fuel
- Lists and roundups (most traded stocks, broker flows, funds)
- Commentary, interviews, opinions, expectations ("…bekleniyor", "…olur mu?")
- Events abroad with no plausible link to Turkish listed companies or the Turkish market
- Small events: a minor order or a small fine relative to the firm, a 4.0 earthquake with no damage

## Borderline rules
- **Tone doesn't matter, only impact.** A neutral or dull headline with no event is a
  confident `0`. A dry headline reporting a big event ("Şirkete kayyum
  atandı") is `1`.
- **"Announced" is not "changed".** "Kredi derecelendirme notu açıklandı", "temettü
  kararını açıkladı", "bilançosunu açıkladı" report a routine disclosure → 0, unless the
  headline itself states a change or a surprise ("notunu düşürdü", "rekor zarar",
  "sürpriz temettü").
- **Surprise words count.** "Sürpriz temettü", "beklenmedik faiz artışı" → lean 1.
- **Scale matters.** "Dev sözleşme" or an amount large for the company → 1. A routine order → 0.
- **Already-known story with nothing new** (a follow-up, a reaction, a court date) → 0.
  A **new step** in it (an arrest, a verdict, an indictment) → 1.
- **Foreign events:** 1 only if they plausibly move Turkish markets (major war escalation,
  a Fed surprise, sanctions involving Türkiye).
- **A price move is not an event.** "Hisse tavan yaptı", "4 yılın en düşük seviyesi" → 0;
  if the headline states the cause, judge the cause.
- **Forecasts stay 0, whoever makes them.** "S&P: indirim beklemiyoruz" → 0; a rating
  action ("notunu yükseltti") → 1.
- **Registrations and formalities** ("… tescil edildi", a higher authorised capital
  ceiling) → 0.
- **Scheduled statistics and flows** (PMI, GDP, inflation, fund flows, broker
  buying) → 0, unless the headline states a surprise.
- **Mergers:** a merger decision at a small or mid-sized firm, or with a named outside
  partner → 1; an application, or an internal merger at a large firm → 0.
- **Still unsure:** pick the more likely label. Use `note` for a short reason if useful.

## Direction (positives only)
For every `1`, also record which way you expect prices to move: `up`, `down`, or `unclear`
when the direction can't be read from the headline (e.g. a CEO resignation). This is
secondary information; the research questions use only the binary label.

## Sheet format
`item, title, label, direction, note`. Fill `label` with 0 or 1 for every row, and
`direction` for every 1; `note` is optional. In the labelling tool
(`scripts/label.py`) one key does both: `1` = up, `2` = down, `3` = unclear, `0` = not
high-impact. The order is random, and the sheet doesn't show where an item came
from.
