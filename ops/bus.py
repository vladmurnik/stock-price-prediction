#!/usr/bin/env python
"""Agent message bus: workers talk to the orchestrator without polluting the user chat.

Usage (from any agent or shell):

    python ops/bus.py send --from quant-data --type progress --task t12 \
        --text "RSS feeds healthy, 412 items, 0 duplicates"
    python ops/bus.py send --from quant-model --type question --task t10 \
        --text "A3 needs openjev artifact; proceed without it?"
    python ops/bus.py read --to orchestrator --unread
    python ops/bus.py read --task t10 --limit 20
    python ops/bus.py ack --id 7
    python ops/bus.py render          # writes ops/chat.md

Storage: ops/bus/messages.jsonl, one JSON object per line. Append-only, so a crash
never loses earlier traffic. Nothing here is written to reports/ or to the user's chat.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUS = ROOT / "bus" / "messages.jsonl"
CHAT = ROOT / "chat.md"
TYPES = ("progress", "question", "answer", "result", "blocked", "handoff")
MAX_TEXT = 4000


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_all() -> list[dict]:
    if not BUS.exists():
        return []
    out: list[dict] = []
    for line in BUS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _append(message: dict) -> None:
    BUS.parent.mkdir(parents=True, exist_ok=True)
    with BUS.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(message, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cmd_send(args: argparse.Namespace) -> int:
    text = (args.text or "").strip()
    if not text:
        print("empty --text", file=sys.stderr)
        return 2
    messages = _read_all()
    record = {
        "id": (messages[-1]["id"] + 1) if messages else 1,
        "ts": _now(),
        "from": args.sender,
        "to": args.to,
        "type": args.type,
        "task": args.task or "",
        "text": text[:MAX_TEXT],
        "acked": False,
    }
    _append(record)
    print(f"[{record['id']}] {record['from']} -> {record['to']} ({record['type']}): {text[:120]}")
    return 0


def _select(messages: list[dict], args: argparse.Namespace) -> list[dict]:
    out = messages
    if args.to:
        out = [m for m in out if m.get("to") == args.to]
    if args.frm:
        out = [m for m in out if m.get("from") == args.frm]
    if args.task:
        out = [m for m in out if m.get("task") == args.task]
    if args.type:
        out = [m for m in out if m.get("type") == args.type]
    if getattr(args, "unread", False):
        out = [m for m in out if not m.get("acked")]
    return out[-args.limit :]


def cmd_read(args: argparse.Namespace) -> int:
    selected = _select(_read_all(), args)
    if not selected:
        print("no messages")
        return 0
    for m in selected:
        flag = " " if m.get("acked") else "*"
        task = f" [{m['task']}]" if m.get("task") else ""
        print(f"{flag} #{m['id']} {m['ts']} {m['from']} -> {m['to']} ({m['type']}){task}")
        print(f"    {m['text']}")
    return 0


def cmd_ack(args: argparse.Namespace) -> int:
    messages = _read_all()
    hit = 0
    for m in messages:
        if m["id"] == args.id:
            m["acked"] = True
            hit += 1
    if not hit:
        print(f"no message #{args.id}", file=sys.stderr)
        return 2
    BUS.write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in messages), encoding="utf-8"
    )
    print(f"acked #{args.id}")
    return 0


def cmd_render(_args: argparse.Namespace) -> int:
    messages = _read_all()
    lines = [
        "# Agent bus",
        "",
        "Автоматически собирается из `ops/bus/messages.jsonl` (`python ops/bus.py render`).",
        "Здесь виден прямой диалог агентов с оркестратором: progress, вопросы, результаты.",
        "",
        "| # | время (UTC) | от | кому | тип | задача | сообщение |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in messages[-300:]:
        text = m["text"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {m['id']} | {m['ts'][11:19]} | {m['from']} | {m['to']} | "
            f"{m['type']} | {m.get('task', '')} | {text} |"
        )
    CHAT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"rendered {len(messages)} message(s) -> {CHAT}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    send = sub.add_parser("send", help="append a message to the bus")
    send.add_argument("--from", dest="sender", required=True)
    send.add_argument("--to", default="orchestrator")
    send.add_argument("--type", default="progress", choices=TYPES)
    send.add_argument("--task", default="")
    send.add_argument("--text", required=True)
    send.set_defaults(func=cmd_send)

    read = sub.add_parser("read", help="print messages")
    read.add_argument("--to")
    read.add_argument("--from", dest="frm")
    read.add_argument("--task")
    read.add_argument("--type")
    read.add_argument("--unread", action="store_true")
    read.add_argument("--limit", type=int, default=30)
    read.set_defaults(func=cmd_read)

    ack = sub.add_parser("ack", help="mark a message as read")
    ack.add_argument("--id", type=int, required=True)
    ack.set_defaults(func=cmd_ack)

    render = sub.add_parser("render", help="rebuild ops/chat.md")
    render.set_defaults(func=cmd_render)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
