"""Build a cleaned, frozen headline snapshot from the news database (read-only).

Keeps three outlets (ROTABORSA, DUNYA, BORSAGUNDEM) and removes
1. recurring roundups: titles that repeat at least TEMPLATE_MIN_REPEATS times in
   one outlet once numbers and dates are stripped. Titles naming a ticker are
   exempt, since a company can have the same kind of event in different years;
2. titles too short to judge;
3. duplicates within or across outlets (the earliest copy is kept).

Usage:
    python scripts/01_build_snapshot.py --db PATH/TO/news_database.sqlite
"""

import argparse
import hashlib
import html
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

SOURCES = ["ROTABORSA", "DUNYA", "BORSAGUNDEM"]
# Covers the 24-month test window plus earlier months for the few-shot pool.
SNAPSHOT_START = "2024-01-01"
SNAPSHOT_END = "2026-08-22"  # last day in the database
TIMEZONE = "Europe/Istanbul"
# Daily roundups repeat dozens to hundreds of times. Real events can recur a
# few times (a company's yearly dividend decision, a central bank holding
# rates), so the threshold is set high: a leftover roundup only adds an easy
# negative, while a removed event would bias the test set.
TEMPLATE_MIN_REPEATS = 10
MIN_TITLE_CHARS = 20
MIN_TITLE_WORDS = 4

TURKISH_MONTHS = (
    "ocak şubat mart nisan mayıs haziran temmuz ağustos eylül ekim kasım aralık"
).split()
TURKISH_WEEKDAYS = "pazartesi salı çarşamba perşembe cuma cumartesi pazar".split()
DATE_WORDS = re.compile(r"\b(" + "|".join(TURKISH_MONTHS + TURKISH_WEEKDAYS) + r")\b")
TICKER_IN_TITLE = re.compile(r"\([A-ZÇĞİÖŞÜ0-9]{3,6}\)")  # e.g. "(TUPRS)"

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT.parent / "workspace" / "data" / "news_database.sqlite"
OUT_DIR = ROOT / "data"


def turkish_lower(text: str) -> str:
    """Lowercase with the Turkish dotted/dotless i rules."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def clean_title(title: str) -> str:
    """Unescape HTML entities, normalise quotes and collapse whitespace."""
    title = html.unescape(title or "")
    title = title.replace("’", "'").replace("‘", "'")
    title = title.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", title).strip()


def duplicate_key(title: str) -> str:
    """Lowercased title without punctuation, used to find exact duplicates."""
    return re.sub(r"[^\w]+", " ", turkish_lower(title)).strip()


def template_key(title: str) -> str:
    """Title with numbers and date words removed, used to find roundups.

    "Göstergeleri Al sinyali veren hisseler (14 Şubat)" and the same headline
    with another date map to the same key.
    """
    key = duplicate_key(title)
    key = re.sub(r"\d+", " ", key)
    key = DATE_WORDS.sub(" ", key)
    return re.sub(r"\s+", " ", key).strip()


def load_rows(db_path: Path) -> pd.DataFrame:
    """Read all rows of the three outlets from the database, read-only."""
    uri = f"file:{db_path}?mode=ro"
    placeholders = ",".join("?" for _ in SOURCES)
    query = (
        "SELECT id, source, published_at, title, url, primary_ticker "
        f"FROM news_records WHERE source IN ({placeholders})"
    )
    with sqlite3.connect(uri, uri=True) as conn:
        return pd.read_sql_query(query, conn, params=SOURCES)


def build(db_path: Path) -> tuple[pd.DataFrame, dict]:
    df = load_rows(db_path)
    stats = {"rows_read": len(df)}

    local = pd.to_datetime(df["published_at"], utc=True, format="ISO8601")
    local = local.dt.tz_convert(TIMEZONE)
    df["date"] = local.dt.strftime("%Y-%m-%d")
    df["time"] = local.dt.strftime("%H:%M")
    df["_ts"] = local
    df["title"] = df["title"].map(clean_title)
    df["dup_key"] = df["title"].map(duplicate_key)
    df["tpl_key"] = df["title"].map(template_key)

    # Template detection uses each outlet's full history, so a roundup that
    # appears only once inside the window is still recognised.
    repeats = df.groupby(["source", "tpl_key"])["id"].transform("size")
    df = df[(df["date"] >= SNAPSHOT_START) & (df["date"] <= SNAPSHOT_END)]
    stats["rows_in_period"] = len(df)

    names_ticker = df["title"].str.contains(TICKER_IN_TITLE)
    df = df[(repeats.loc[df.index] < TEMPLATE_MIN_REPEATS) | names_ticker]
    stats["after_template_filter"] = len(df)

    words = df["title"].str.split().str.len()
    df = df[(df["title"].str.len() >= MIN_TITLE_CHARS) & (words >= MIN_TITLE_WORDS)]
    stats["after_length_filter"] = len(df)

    df = df.sort_values(["_ts", "id"]).drop_duplicates("dup_key", keep="first")
    stats["after_dedup"] = len(df)

    ticker = df["primary_ticker"].where(df["primary_ticker"] != "NON")
    snapshot = pd.DataFrame(
        {
            "id": df["id"],
            "date": df["date"],
            "time": df["time"],
            "source": df["source"],
            "title": df["title"],
            "ticker": ticker,
            "url": df["url"],
        }
    ).reset_index(drop=True)
    stats["per_source"] = snapshot["source"].value_counts().to_dict()
    return snapshot, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    snapshot, stats = build(args.db)
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "snapshot.csv"
    snapshot.to_csv(out_path, index=False)

    info = {
        "sources": SOURCES,
        "period": [SNAPSHOT_START, SNAPSHOT_END],
        "timezone": TIMEZONE,
        "filters": {
            "template_min_repeats": TEMPLATE_MIN_REPEATS,
            "min_title_chars": MIN_TITLE_CHARS,
            "min_title_words": MIN_TITLE_WORDS,
        },
        "row_counts": stats,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    (OUT_DIR / "snapshot_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
