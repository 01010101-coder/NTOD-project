#!/usr/bin/env python3
"""Агенты skvoznoi_project в реальном времени.

  python3.11 .opencode/queue/agents.py          # один снимок
  python3.11 .opencode/queue/agents.py --watch  # обновление каждые 3 с (Ctrl+C — выход)

Источники:
- opencode (apply, deepseek flash): ~/.local/share/opencode/opencode.db, сессии «apply <change> #N»;
- Claude Code (propose / verify, Opus): ~/.claude/projects/*skvoznoi-project/*/subagents/,
  описание субагента начинается с «propose <change>» или «verify <change>»;
- прогресс: openspec/changes/<change>/tasks.md, вердикт: .opencode/queue/verify/<change>.md;
- очередь: процесс ocq.py и хвост её лога .opencode/queue/logs/ocq.log.
"""

import glob
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
CHANGES = PROJ / "openspec/changes"
DB = Path.home() / ".local/share/opencode/opencode.db"
CLAUDE = Path.home() / ".claude/projects"
UTF8 = "utf-8"
ACTIVE_SEC = 90  # opencode-сессия «работает», если обновлялась за последние N секунд
CLAUDE_IDLE = 600  # субагент без записей дольше — считаем зависшим/оборванным
RECENT = 1800  # показывать завершённых агентов за последние N секунд

IDS = [
    ("C01", "setup-project-skeleton"),
    ("C02", "add-sqlite-storage"),
    ("C03", "add-city-zones-model"),
    ("C04", "add-demand-generator"),
    ("C05", "add-driver-supply-simulation"),
    ("C06", "add-simulation-events"),
    ("C07", "add-simulation-engine-and-export"),
    ("C08", "add-zone-metrics-aggregation"),
    ("C09", "add-dataset-preparation"),
    ("C10", "add-baseline-models"),
    ("C11", "add-lstm-forecaster"),
    ("C12", "add-model-evaluation"),
    ("C13", "add-inference-service"),
    ("C14", "add-surge-pricing-policy"),
    ("C15", "add-streamlit-dashboard-skeleton"),
    ("C16", "add-city-zone-map"),
    ("C17", "add-time-control-panel"),
    ("C18", "add-forecast-vs-actual-charts"),
]
NAME2ID = {n: c for c, n in IDS}

BOLD, DIM, GREEN, YELLOW, RED, CYAN, RESET = (
    ("\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m")
    if sys.stdout.isatty()
    else ("",) * 7
)


def read(path):
    return Path(path).read_text(encoding=UTF8, errors="ignore")


def age(sec):
    sec = max(0, sec)
    if sec < 120:
        return f"{int(sec)}с"
    if sec < 7200:
        return f"{int(sec // 60)}м"
    return f"{int(sec // 3600)}ч"


def short(text, n):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'^\]?(\[\w+\] )?cd "?[^";&]*"?\s*(?:&&|;)\s*', r"\1", text)
    text = text.replace(str(PROJ) + "/", "")
    return text if len(text) <= n else text[: n - 1] + "…"


# ---------- прогресс по файлам ----------


def change_dir(name):
    arch = sorted(glob.glob(f"{CHANGES}/archive/*-{name}"))
    return (Path(arch[-1]), True) if arch else (CHANGES / name, False)


def groups_of(path):
    """[(номер, сделано, всего)] по группам tasks.md."""
    out = []
    for block in re.split(r"(?m)^(?=## )", read(path)):
        m = re.match(r"## (\d+)", block)
        if m:
            done = len(re.findall(r"^\s*- \[x\]", block, re.M | re.I))
            total = len(re.findall(r"^\s*- \[[ xX]\]", block, re.M))
            out.append((int(m.group(1)), done, total))
    return out


def verdict(name):
    path = HERE / "verify" / f"{name}.md"
    if not path.exists():
        return None
    m = re.search(r"ВЕРДИКТ:\s*(PASS|FAIL)", read(path))
    return m.group(1) if m else "?"


# ---------- opencode ----------


def oc_last_action(con, sid):
    rows = con.execute(
        "select data from session_message where session_id=? order by seq desc limit 15", (sid,)
    ).fetchall()
    for (data,) in rows:
        try:
            content = json.loads(data).get("content") or []
        except json.JSONDecodeError:
            continue
        for item in reversed(content):
            kind = item.get("type")
            if kind == "tool":
                inp = item.get("input") or item.get("state", {}).get("input") or {}
                arg = str(
                    inp.get("command") or inp.get("filePath") or inp.get("path")
                    or inp.get("pattern") or inp.get("description") or ""
                )
                arg = re.sub(r'^cd "[^"]*" && ', "", arg)
                return f"[{item.get('name') or item.get('tool')}] {arg}"
            text = (item.get("text") or "").strip()
            if text:
                return ("💭 " if kind == "reasoning" else "") + text
    return ""


def opencode_agents(now):
    if not DB.exists():
        return []
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute(
        "select id, title, time_updated, model, tokens_input, tokens_cache_read, "
        "tokens_output + tokens_reasoning, idle_outcome, "
        "(select count(*) from session_message m where m.session_id=s.id) "
        "from session_v2 s where parent_id is null and directory=? and time_updated > ? "
        "order by time_updated desc",
        (str(PROJ), (now - 6 * 3600) * 1000),
    ).fetchall()
    agents = []
    for sid, title, upd, model, tin, tcache, tout, outcome, steps in rows:
        m = re.match(r"(\w+) (\S+)(?: #(\d+))?", title or "")
        if not m:
            continue
        msgs = [
            json.loads(d)
            for (d,) in con.execute(
                "select data from session_message where session_id=? order by seq desc limit 2", (sid,)
            )
        ]
        pending = bool(msgs) and "completed" not in msgs[0].get("time", {})
        done_msg = next((x for x in msgs if "completed" in x.get("time", {})), None)
        kinds = {c.get("type") for c in (done_msg or {}).get("content") or []}
        finished = done_msg is not None and "text" in kinds and "tool" not in kinds
        quiet = now - upd / 1000
        alive = not finished and (quiet < ACTIVE_SEC or (pending and quiet < 900))
        try:
            model = json.loads(model).get("id") or model
        except (TypeError, ValueError, AttributeError):
            pass
        agents.append(
            {
                "src": "opencode",
                "kind": m.group(1),
                "change": m.group(2),
                "group": m.group(3),
                "model": str(model or "").split("/")[-1],
                "alive": alive,
                "state": "работает" if alive else ("готово" if finished else (outcome or "стоит")),
                "quiet": quiet,
                "steps": steps,
                "tokens": f"{tin / 1e3:.0f}k/{tcache / 1e6:.1f}M/{tout / 1e3:.0f}k",
                "action": oc_last_action(con, sid) if alive else "",
            }
        )
    con.close()
    return agents


# ---------- Claude Code ----------


def tail_lines(path, size=200_000):
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - size))
        return f.read().decode(UTF8, errors="ignore").splitlines()


def claude_last_action(path):
    steps = 0
    for line in reversed(tail_lines(path)):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "assistant":
            continue
        content = rec.get("message", {}).get("content") or []
        for item in reversed(content):
            if item.get("type") == "tool_use":
                inp = item.get("input") or {}
                arg = str(
                    inp.get("command") or inp.get("file_path") or inp.get("pattern")
                    or inp.get("description") or inp.get("skill") or ""
                )
                return f"[{item.get('name')}] {arg}"
            if item.get("type") == "text" and item.get("text", "").strip():
                return item["text"].strip()
        steps += 1
    return ""


def claude_agents(now):
    global STOPPED
    path = HERE / "stopped_agents.txt"
    STOPPED = set(read(path).split()) if path.exists() else set()
    agents = []
    for meta_path in glob.glob(f"{CLAUDE}/*skvoznoi-project/*/subagents/*.meta.json"):
        try:
            meta = json.loads(read(meta_path))
        except ValueError:
            continue
        m = re.match(r"(propose|verify|review)\s+(\S+)", meta.get("description", ""))
        if not m:
            continue
        log = meta_path.replace(".meta.json", ".jsonl")
        mtime = os.path.getmtime(log) if os.path.exists(log) else os.path.getmtime(meta_path)
        quiet = now - mtime
        if quiet > 6 * 3600:
            continue
        aid = os.path.basename(meta_path).replace("agent-", "").replace(".meta.json", "")
        stopped = aid in STOPPED
        finished = bool(meta.get("finished"))
        alive = not finished and not stopped and quiet < CLAUDE_IDLE
        steps = sum(1 for _ in open(log, encoding=UTF8, errors="ignore")) if os.path.exists(log) else 0
        agents.append(
            {
                "src": "claude",
                "kind": m.group(1),
                "change": m.group(2),
                "group": None,
                "model": meta.get("model") or "opus",
                "alive": alive,
                "state": "работает" if alive else ("готово" if finished else ("остановлен" if stopped else "оборван?")),
                "quiet": quiet,
                "steps": steps,
                "tokens": "",
                "action": claude_last_action(log) if alive and os.path.exists(log) else "",
            }
        )
    return agents


# ---------- очередь ----------


def queue_info():
    try:
        out = subprocess.run(["pgrep", "-f", "ocq.py"], capture_output=True, text=True).stdout.split()
    except OSError:
        out = []
    out = [p for p in out if p != str(os.getpid())]
    log = HERE / "logs" / "ocq.log"
    tail = [ln for ln in read(log).splitlines() if ln.strip()][-6:] if log.exists() else []
    return out, tail


# ---------- вывод ----------


def stage_of(name, arch, d, groups, verd, active):
    kinds = {a["kind"] for a in active}
    if arch:
        return f"{GREEN}архив{RESET}", 5
    if "verify" in kinds:
        return f"{CYAN}verify…{RESET}", 13
    if verd == "PASS":
        return f"{GREEN}verify PASS{RESET}", 11
    if "propose" in kinds or "review" in kinds:
        return f"{CYAN}propose…{RESET}", 14
    if "apply" in kinds:
        return f"{CYAN}apply…{RESET}", 12
    if verd == "FAIL" and groups and any(dn < t for _, dn, t in groups):
        return f"{YELLOW}правки{RESET}", 6
    if groups and all(dn == t for _, dn, t in groups):
        return f"{YELLOW}ждёт verify{RESET}", 11
    if groups and any(dn for _, dn, _ in groups):
        return f"{YELLOW}apply стоит{RESET}", 11
    if groups:
        return "спека готова", 12
    if (d / "proposal.md").exists():
        return f"{YELLOW}спека неполна{RESET}", 13
    return f"{DIM}ожидает{RESET}", 7


def group_bar(groups, running):
    cells = []
    for num, done, total in groups:
        if str(num) in running:
            cells.append(f"{CYAN}▶{num}{RESET}")
        elif total and done == total:
            cells.append(f"{GREEN}✓{num}{RESET}")
        elif done:
            cells.append(f"{YELLOW}◐{num}{RESET}")
        else:
            cells.append(f"{DIM}·{num}{RESET}")
    return " ".join(cells)


def render():
    now = time.time()
    agents = opencode_agents(now) + claude_agents(now)
    alive = [a for a in agents if a["alive"]]
    lines = [f"{BOLD}Агенты skvoznoi_project — {time.strftime('%H:%M:%S')}{RESET}", ""]
    lines.append(f"{BOLD}{'ID':4} {'CHANGE':33} {'ЭТАП':14} {'ЗАДАЧИ':7} ГРУППЫ tasks.md{RESET}")
    for cid, name in IDS:
        d, arch = change_dir(name)
        tasks = d / "tasks.md"
        groups = groups_of(tasks) if tasks.exists() else []
        active = [a for a in alive if a["change"] == name]
        verd = verdict(name)
        stage, vis = stage_of(name, arch, d, groups, verd, active)
        done = sum(dn for _, dn, _ in groups)
        total = sum(t for _, _, t in groups)
        prog = f"{done}/{total}" if groups else "—"
        running = {a["group"] for a in active if a["group"]}
        bar = "" if arch else group_bar(groups, running)
        if verd == "FAIL" and not arch:
            bar += f"  {RED}verify: FAIL{RESET}"
        pad = " " * max(1, 15 - len(re.sub(r"\033\[\d+m", "", stage)))
        lines.append(f"{cid:4} {name:33} {stage}{pad}{prog:7} {bar}")

    limit = int(os.environ.get("OCQ_JOBS", "4"))
    color = RED if len(alive) > limit else (YELLOW if len(alive) == limit else GREEN)
    lines += ["", f"{BOLD}Работают сейчас: {color}{len(alive)}/{limit}{RESET}{BOLD} слотов{RESET}"]
    if not alive:
        lines.append(f"  {DIM}никого{RESET}")
    for a in sorted(alive, key=lambda a: (a["src"], a["change"])):
        cid = NAME2ID.get(a["change"], "   ")
        title = f"{a['kind']} {a['change']}" + (f" #{a['group']}" if a["group"] else "")
        tok = f"  {DIM}{a['tokens']}{RESET}" if a["tokens"] else ""
        lines.append(
            f"  {CYAN}▶{RESET} {cid} {a['src']:8} {a['model'][:18]:18} {title:46} "
            f"{a['steps']:>4} шаг, {age(a['quiet'])} назад{tok}"
        )
        if a["action"]:
            lines.append(f"       {DIM}{short(a['action'], 140)}{RESET}")

    recent = [a for a in agents if not a["alive"] and a["quiet"] < RECENT]
    if recent:
        lines += ["", f"{BOLD}Завершились за {RECENT // 60} мин{RESET}"]
        for a in sorted(recent, key=lambda a: a["quiet"])[:8]:
            title = f"{a['kind']} {a['change']}" + (f" #{a['group']}" if a["group"] else "")
            mark = GREEN + "■" if a["state"] == "готово" else YELLOW + "■"
            lines.append(f"  {mark}{RESET} {a['src']:8} {title:46} {a['state']:10} {age(a['quiet'])} назад")

    pids, tail = queue_info()
    lines += ["", f"{BOLD}Очередь apply (ocq.py){RESET}: " + (f"{GREEN}работает{RESET} pid {' '.join(pids)}" if pids else f"{YELLOW}не запущена{RESET}")]
    lines += [f"  {DIM}{short(t, 150)}{RESET}" for t in tail]
    lines.append(f"\n{DIM}▶ работает  ✓ группа сделана  ◐ частично  · не начата{RESET}")
    return "\n".join(lines)


def main():
    if "--watch" not in sys.argv:
        print(render())
        return
    try:
        while True:
            frame = render()
            print("\033[H\033[2J" + frame + f"\n{DIM}обновление каждые 3 с, выход — Ctrl+C{RESET}", flush=True)
            time.sleep(3)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
