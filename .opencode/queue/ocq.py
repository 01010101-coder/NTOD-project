#!/usr/bin/env python3
"""Очередь apply-агентов opencode.

  python3.11 .opencode/queue/ocq.py            # запустить очередь
  python3.11 .opencode/queue/ocq.py --dry-run  # показать, что будет сделано
  python3.11 .opencode/queue/ocq.py --status   # статус задач из jobs.txt

Конвейер change: propose (агент Claude, Opus) -> apply (эта очередь, deepseek flash)
-> verify (агент Claude, Opus) -> archive.

jobs.txt: строки `<kind> <change> [после: dep1, dep2]`, kind = apply | propose.
Зависимость — заголовок другой строки jobs.txt или внешний этап:
  `propose <change>` — готов tasks.md;
  `verify <change>`  — в verify/<change>.md первая строка `ВЕРДИКТ: PASS` (или change в архиве).
jobs.txt перечитывается на лету: новые строки подхватываются без перезапуска.
Очередь не завершается, пока есть невыполненные строки (ждёт спеки и verify).

Промпт: prompts/<kind>-<change>.md, иначе шаблон prompts/_<kind>.md.
Подстановки: {change}, {context} (куски плана и контракта про этот change),
для apply ещё {group} (текущая группа задач из tasks.md).

Экономия лимитов:
- apply идёт по группам задач tasks.md, каждая группа — новая короткая сессия
  «apply <change> #N»; группы, дописанные в tasks.md после verify, подхватываются;
- нужные разделы docs/development-plan.md и docs/pipeline-contract.md вставляются в промпт;
- не больше OCQ_JOBS агентов одновременно (по умолчанию 4) ВСЕГО — вместе с работающими
  субагентами Claude (propose/verify/review); пауза OCQ_GAP сек между стартами;
- прерванная сессия продолжается через `-s <id>`, а не начинается заново;
- при упоре в лимит вся очередь встаёт на паузу (10 мин, затем x2, максимум 60).
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
CHANGES = PROJECT / "openspec/changes"
VERIFY = HERE / "verify"
DB = Path.home() / ".local/share/opencode/opencode.db"
LOGS = HERE / "logs"
UTF8 = "utf-8"

MODELS = {
    "propose": os.environ.get("OCQ_PROPOSE_MODEL", "opencode-go/deepseek-v4-pro"),
    "apply": os.environ.get("OCQ_APPLY_MODEL", "opencode-go/deepseek-v4.1-flash"),
}
MAX_JOBS = int(os.environ.get("OCQ_JOBS", "4"))
GAP = int(os.environ.get("OCQ_GAP", "30"))
BACKOFF_START, BACKOFF_MAX = 600, 3600
MAX_ATTEMPTS = 3  # на одну сессию, не считая пауз по лимиту
CLAUDE_IDLE = 600  # субагент Claude без записей дольше — не считается работающим

LIMIT_RE = re.compile(
    r"rate.?limit|429|quota|usage limit|limit (?:reached|exceeded)|too many requests"
    r"|insufficient (?:balance|credit)|лимит",
    re.I,
)
RESUME_MSG = (
    "Сессия была прервана (лимит/обрыв). Продолжи с того места, где остановился: "
    "сверься с файлами на диске и tasks.md, не переделывай готовое. Условия исходного "
    "задания в силе, в том числе: отмечай [x] в tasks.md сразу после каждой задачи. "
    "В конце — краткий отчёт."
)


def read(path):
    return Path(path).read_text(encoding=UTF8, errors="ignore")


def parse_jobs():
    jobs = []
    for raw in read(HERE / "jobs.txt").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        head, _, deps = line.partition("после:")
        kind, change = head.split()[:2]
        jobs.append(
            {
                "kind": kind,
                "change": change,
                "title": f"{kind} {change}",
                "deps": [d.strip() for d in deps.split(",") if d.strip()],
            }
        )
    return jobs


def last_session(title):
    """(id, idle_outcome) последней корневой сессии с таким заголовком в проекте."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        row = con.execute(
            "select id, idle_outcome from session_v2 where title=? and parent_id is null "
            "and directory=? order by time_created desc limit 1",
            (title, str(PROJECT)),
        ).fetchone()
    finally:
        con.close()
    return row or (None, None)


def session_running(title):
    """Сессия с таким заголовком ещё идёт (нет итога и недавно обновлялась)."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        row = con.execute(
            "select idle_outcome, time_updated from session_v2 where title=? and parent_id is null "
            "and directory=? order by time_created desc limit 1",
            (title, str(PROJECT)),
        ).fetchone()
    finally:
        con.close()
    return bool(row) and row[0] is None and time.time() - row[1] / 1000 < 300


def task_groups(change):
    """[(номер, текст группы, есть ли невыполненные)] из tasks.md."""
    path = CHANGES / change / "tasks.md"
    if not path.exists():
        return []
    groups = []
    for block in re.split(r"(?m)^(?=## )", read(path)):
        m = re.match(r"## (\d+)", block)
        if m:
            groups.append((int(m.group(1)), block.strip(), "- [ ]" in block))
    return groups


def archived(change):
    return not (CHANGES / change).exists() and any((CHANGES / "archive").glob(f"*-{change}"))


def verdict(change):
    """PASS / FAIL / None по отчёту verify/<change>.md."""
    path = VERIFY / f"{change}.md"
    if not path.exists():
        return None
    m = re.search(r"ВЕРДИКТ:\s*(PASS|FAIL)", read(path))
    return m.group(1) if m else None


def stage_done(dep):
    """Внешний этап конвейера, которого нет в jobs.txt."""
    kind, _, change = dep.partition(" ")
    if archived(change):
        return True
    if kind == "verify":
        return verdict(change) == "PASS"
    if kind == "propose":
        return (CHANGES / change / "tasks.md").exists()
    if kind == "apply":
        groups = task_groups(change)
        return bool(groups) and not any(o for _, _, o in groups)
    return False


def stopped_agents():
    """id субагентов Claude, остановленных вручную (флаг finished им не ставится)."""
    path = HERE / "stopped_agents.txt"
    return set(read(path).split()) if path.exists() else set()


def claude_alive():
    """Сколько субагентов Claude (propose/verify/review) сейчас работает — они делят лимит OCQ_JOBS."""
    n = 0
    for meta in (Path.home() / ".claude/projects").glob("*skvoznoi-project/*/subagents/*.meta.json"):
        log = meta.with_name(meta.name.replace(".meta.json", ".jsonl"))
        try:
            info = json.loads(read(meta))
            fresh = time.time() - (log if log.exists() else meta).stat().st_mtime < CLAUDE_IDLE
        except (OSError, ValueError):
            continue
        stopped = meta.name.replace("agent-", "").replace(".meta.json", "") in stopped_agents()
        if fresh and not stopped and not info.get("finished") and re.match(r"(propose|verify|review) ", info.get("description", "")):
            n += 1
    return n


def is_done(job):
    return stage_done(job["title"])


def md_section(path, heading_re):
    """Раздел markdown от заголовка, совпавшего с heading_re, до заголовка того же или выше уровня."""
    lines = read(PROJECT / path).splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"(#+) ", line)
        if m and re.search(heading_re, line):
            level = len(m.group(1))
            out = [line]
            for nxt in lines[i + 1 :]:
                n = re.match(r"(#+) ", nxt)
                if n and len(n.group(1)) <= level:
                    break
                out.append(nxt)
            return "\n".join(out).strip()
    return ""


def context_for(change):
    plan = "docs/development-plan.md"
    contract = "docs/pipeline-contract.md"
    contract_text = read(PROJECT / contract)
    parts = [
        f"--- {plan}, раздел 2 ---\n" + md_section(plan, r"^## 2\."),
        f"--- {plan}, раздел change ---\n" + md_section(plan, rf"`{change}`"),
        f"--- {contract}, общие правила ---\n" + contract_text.split("\n## ", 1)[0].strip(),
        f"--- {contract}, интерфейс change ---\n" + md_section(contract, rf"\b{change}\b"),
    ]
    return "\n\n".join(p for p in parts if not p.endswith("---\n"))


def prompt_for(job, group=""):
    own = HERE / "prompts" / f"{job['kind']}-{job['change']}.md"
    tpl = own if own.exists() else HERE / "prompts" / f"_{job['kind']}.md"
    return read(tpl).format(change=job["change"], context=context_for(job["change"]), group=group)


class Queue:
    def __init__(self, dry):
        self.dry = dry
        self.jobs, self.state = [], {}
        self.lock = threading.Lock()
        self.pause_until = 0.0
        self.backoff = BACKOFF_START
        self.last_start = 0.0
        self.waiting = set()
        self.reload()

    def log(self, msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    def reload(self):
        """Перечитать jobs.txt; готовые apply с новыми открытыми группами вернуть в работу."""
        self.jobs = parse_jobs()
        for j in self.jobs:
            st = self.state.get(j["title"])
            if st is None:
                self.state[j["title"]] = "done" if is_done(j) else "pending"
            elif st == "done" and j["kind"] == "apply" and not is_done(j):
                self.state[j["title"]] = "pending"
                self.log(f"↺ {j['title']}: в tasks.md появились новые задачи")

    def dep_ok(self, dep):
        st = self.state.get(dep)
        return st == "done" if st is not None else stage_done(dep)

    def ready(self):
        for j in self.jobs:
            if self.state[j["title"]] != "pending":
                continue
            if any(self.state.get(d) == "failed" for d in j["deps"]):
                self.state[j["title"]] = "failed"
                self.log(f"✗ {j['title']}: зависимость провалилась")
                continue
            blockers = [d for d in j["deps"] if not self.dep_ok(d)]
            if j["kind"] == "apply" and not stage_done(f"propose {j['change']}"):
                blockers.append(f"propose {j['change']}")
            if not blockers:
                self.waiting.discard(j["title"])
                return j
            if j["title"] not in self.waiting:
                self.waiting.add(j["title"])
                self.log(f"… {j['title']}: ждёт {', '.join(blockers)}")
        return None

    def wait_turn(self):
        """Ждать конца паузы по лимиту и интервала между стартами."""
        while True:
            with self.lock:
                now = time.time()
                if now >= self.pause_until and now - self.last_start >= GAP:
                    self.last_start = now
                    return
            time.sleep(5)

    def session(self, title, model, prompt, check):
        """Довести одну сессию до check() == True. Возвращает True при успехе."""
        attempts = 0
        LOGS.mkdir(exist_ok=True)
        logfile = LOGS / f"{title.replace(' ', '_').replace('#', '')}.log"
        while attempts < MAX_ATTEMPTS:
            self.wait_turn()
            while session_running(title):  # клиент от прошлого запуска очереди ещё работает
                time.sleep(15)
            sid, outcome = last_session(title)
            cmd = ["opencode", "run", "--auto", "-m", model, "--title", title]
            if sid and outcome != "succeeded":
                cmd += ["-s", sid, RESUME_MSG]
                how = f"продолжаю {sid} (было: {outcome})"
            else:
                cmd.append(prompt)
                how = "новая сессия"
            self.log(f"▶ {title}: {how}")
            with open(logfile, "a", encoding=UTF8) as f:
                f.write(f"\n===== {time.ctime()} {how}\n")
                f.flush()
                proc = subprocess.run(cmd, cwd=PROJECT, stdout=f, stderr=subprocess.STDOUT)
            _, outcome = last_session(title)
            if check():
                with self.lock:
                    self.backoff = BACKOFF_START
                self.log(f"✓ {title}")
                return True
            tail = read(logfile)[-4000:]
            if LIMIT_RE.search(tail) or outcome == "interrupted":
                with self.lock:
                    self.pause_until = max(self.pause_until, time.time() + self.backoff)
                    self.log(f"⏸ {title}: похоже на лимит, пауза {self.backoff // 60} мин")
                    self.backoff = min(self.backoff * 2, BACKOFF_MAX)
                continue
            attempts += 1
            self.log(f"↻ {title}: код {proc.returncode}, итог {outcome}, проверка не прошла ({attempts}/{MAX_ATTEMPTS})")
        self.log(f"✗ {title}: смотри {logfile}")
        return False

    def run_job(self, job):
        change, model = job["change"], MODELS[job["kind"]]
        if job["kind"] == "propose":
            ok = self.session(job["title"], model, prompt_for(job), lambda: is_done(job))
        else:
            ok = True
            while ok and not is_done(job):
                num, text, _ = next(g for g in task_groups(change) if g[2])
                title = f"{job['title']} #{num}"

                def group_done(num=num):
                    return not any(o for n, _, o in task_groups(change) if n == num)

                ok = self.session(title, model, prompt_for(job, text), group_done)
        with self.lock:
            self.state[job["title"]] = "done" if ok else "failed"
        if ok and job["kind"] == "apply":
            self.log(f"✅ {job['title']}: все группы выполнены, ждёт verify")

    def plan_dry(self):
        for j in self.jobs:
            if self.state[j["title"]] == "done":
                self.log(f"[dry] {j['title']}: уже готово")
            elif j["kind"] == "propose":
                sid, outcome = last_session(j["title"])
                self.log(f"[dry] {j['title']}: " + (f"продолжить {sid} ({outcome})" if sid else "новая"))
            else:
                groups = [n for n, _, o in task_groups(j["change"]) if o]
                deps = [d for d in j["deps"] if not self.dep_ok(d)]
                self.log(
                    f"[dry] {j['title']}: группы {groups or '(ждёт propose)'}"
                    + (f", ждёт {', '.join(deps)}" if deps else "")
                )

    def loop(self):
        if self.dry:
            return self.plan_dry()
        threads = []
        while True:
            threads = [t for t in threads if t.is_alive()]
            with self.lock:
                self.reload()
                job = self.ready() if len(threads) + claude_alive() < MAX_JOBS else None
                if job:
                    self.state[job["title"]] = "running"
                left = any(s in ("pending", "running") for s in self.state.values())
            if job:
                t = threading.Thread(target=self.run_job, args=(job,))
                t.start()
                threads.append(t)
            elif not threads and not left:
                break
            time.sleep(5)
        self.log("Очередь пуста. " + ", ".join(f"{k}: {v}" for k, v in self.state.items()))


def main():
    if "--status" in sys.argv:
        for j in parse_jobs():
            if j["kind"] == "apply":
                groups = task_groups(j["change"])
                left = [n for n, _, o in groups if o]
                if archived(j["change"]):
                    print(f"{j['title']:45} в архиве")
                    continue
                info = "нет tasks.md" if not groups else (f"открыты группы {left}" if left else "готово")
                info += f", verify: {verdict(j['change']) or '—'}"
            else:
                info = "готово" if is_done(j) else (last_session(j["title"])[1] or "—")
            print(f"{j['title']:45} {info}")
        return
    Queue(dry="--dry-run" in sys.argv).loop()


if __name__ == "__main__":
    main()
