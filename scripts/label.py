"""Terminal labeller for labeling/label_sheet.csv (standard library only).

Keys: 1 high-impact (up), 2 high-impact (down), 3 high-impact (unclear),
0 not high-impact, n note, b back, f skip, j next unlabelled, q quit.
Saves after every key. --human-set shows the pool first, then the core set,
then the remaining test items.

Usage:
    python scripts/label.py --human-set
"""

import argparse
import csv
import os
import shutil
import sys
import termios
import textwrap
import time
import tty
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHEET = ROOT / "labeling" / "label_sheet.csv"
ITEMS_KEY = ROOT / "data" / "items_key.csv"
FIELDS = ["item", "title", "label", "direction", "note"]
DIRECTION_KEYS = {"1": "up", "2": "down", "3": "unclear"}

BOLD, DIM, GREEN, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"
REMINDER = (
    "High-impact = a concrete NEW event that would noticeably move a company's shares "
    "or the Turkish market.  1 = up, 2 = down, 3 = direction unclear.  0 = routine disclosure, commentary, forecast, price report, list, "
    "or too small.  Judge only the headline."
)


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:  # sheets made before the direction column existed
        row.setdefault("direction", "")
    return rows


def save(path: Path, rows: list[dict]) -> None:
    """Write to a temporary file, then rename, so a crash can't corrupt the sheet."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def next_unlabelled(rows: list[dict], start: int) -> int:
    for i in range(start, len(rows)):
        if rows[i]["label"] == "":
            return i
    return len(rows)


def render(rows: list[dict], i: int, started: float, done_at_start: int) -> None:
    width = min(shutil.get_terminal_size().columns, 100)
    done = sum(r["label"] != "" for r in rows)
    pace = ""
    labelled_now = done - done_at_start
    if labelled_now >= 5:
        per_item = (time.time() - started) / labelled_now
        remaining = (len(rows) - done) * per_item / 60
        pace = f"  ·  {per_item:.0f}s/item  ·  ~{remaining:.0f} min left"

    print("\033[2J\033[H", end="")
    print(f"{DIM}labelled {done}/{len(rows)}{pace}{RESET}\n")
    row = rows[i]
    print(f"{DIM}item {row['item']}{RESET}\n")
    for line in textwrap.wrap(row["title"], width - 4):
        print(f"  {BOLD}{line}{RESET}")
    print()

    if row["label"] == "1":
        current = f"{GREEN}high-impact ({row['direction']}){RESET}"
    elif row["label"] == "0":
        current = f"{RED}not high-impact{RESET}"
    else:
        current = f"{DIM}unlabelled{RESET}"
    note = f"  {DIM}note: {row['note']}{RESET}" if row["note"] else ""
    print(f"  current: {current}{note}\n")
    print(f"{DIM}" + "\n".join(textwrap.wrap(REMINDER, width)) + f"{RESET}\n")
    print(f"{DIM}[1] up  [2] down  [3] unclear  [0] no  [n] note  [b] back  [f] skip  "
          f"[j] next unlabelled  [q] quit{RESET}")


def items_where(column: str, value: str) -> set[str]:
    with ITEMS_KEY.open(encoding="utf-8") as f:
        return {row["item"] for row in csv.DictReader(f) if row.get(column) == value}


def main() -> None:
    parser = argparse.ArgumentParser(description="Label headlines from the keyboard.")
    parser.add_argument("sheet", nargs="?", type=Path, default=DEFAULT_SHEET)
    parser.add_argument("--pool-first", action="store_true",
                        help="show the few-shot pool items before the test items")
    parser.add_argument("--human-set", action="store_true",
                        help="pool first, then the human-set test items, then the rest")
    args = parser.parse_args()

    path = args.sheet
    saved = load(path)
    # Navigation order; the rows are shared with `saved`, which keeps file order.
    rows = saved
    if args.pool_first or args.human_set:
        pool = items_where("set", "pool")
        rows = [r for r in rows if r["item"] in pool] + [r for r in rows if r["item"] not in pool]
    if args.human_set:
        chosen = items_where("human_set", "True")
        rows = [r for r in rows if r["item"] in chosen] + [r for r in rows if r["item"] not in chosen]
    i = next_unlabelled(rows, 0)
    if i == len(rows):
        print("Every item is labelled.")
        i = 0
    started, done_at_start = time.time(), sum(r["label"] != "" for r in rows)

    while True:
        render(rows, i, started, done_at_start)
        key = read_key().lower()
        if key in ("0", "1", "2", "3"):
            rows[i]["label"] = "0" if key == "0" else "1"
            rows[i]["direction"] = DIRECTION_KEYS.get(key, "")
            save(path, saved)
            if i == len(rows) - 1:
                if next_unlabelled(rows, 0) == len(rows):
                    break
            else:
                i += 1
        elif key == "n":
            print("\nnote (Enter to save): ", end="", flush=True)
            rows[i]["note"] = input().strip()
            save(path, saved)
        elif key == "b":
            i = max(i - 1, 0)
        elif key == "f":
            i = min(i + 1, len(rows) - 1)
        elif key == "j":
            i = min(next_unlabelled(rows, i + 1), len(rows) - 1)
        elif key in ("q", "\x03"):  # q or Ctrl-C
            break

    done = sum(r["label"] != "" for r in rows)
    print(f"\nSaved to {path}  ({done}/{len(rows)} labelled)")


if __name__ == "__main__":
    main()
