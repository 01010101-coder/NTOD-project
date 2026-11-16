"""Тесты CLI генерации истории change ``m2-loop`` (задачи 5.1–5.2, §8 плана M2).

Smoke 1 день (задача 5.1): subprocess ``.venv/bin/python -m simulator --days 1
--seed 42 --db <tmp>/a.db --csv <tmp>/csv`` завершается с exit 0 не дольше
производной границы бюджета §5 с запасом (≤ 10 с, design, решение 9), обе
таблицы наполнены (для 1 дня — ровно 1728 строк метрик и 288 уникальных ``ts``),
CSV-файлы равны таблицам плюс заголовок. Guard и формат вывода (задача 5.2):
повторный запуск в непустую БД — exit 2 с упоминанием ``trips`` и без новых
строк (A8), stdout — ровно 4 строки шаблона §4, stderr содержит прогресс
машинного формата, дефолты ``--seed``/``--db`` берутся из конфига, а явные
флаги их переопределяют. Все прогоны пишут в ``tmp_path`` — артефактов в
репозитории тесты не оставляют.
"""

import re
import sqlite3
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
TIME_LIMIT_SEC = 10  # производная граница бюджета §5 (30 дней ≤ 60 с) с запасом

# Шаблоны сводки §4 (числа иллюстративны; regex скрипта приёмки допускает [0-9.]+).
SUMMARY_PATTERNS = [
    r"sim: days=\d+ seed=\d+ db=\S+",
    r"trips: total=\d+ completed=\d+ completed_share=\d+\.\d% "
    r"unfulfilled=\d+ unfulfilled_share=\d+\.\d%",
    r"wait: avg_wait_min=\d+\.\d\d deficit_share=\d+\.\d%",
    r"wall: \d+\.\ds",
]
PROGRESS_PATTERN = r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] trips=\d+ elapsed=\d+\.\d+s"


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    """Запустить CLI subprocess'ом из корня проекта."""

    return subprocess.run(
        [str(PYTHON), "-m", "simulator", *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )


def table_counts(db: Path) -> tuple[int, int, int]:
    """(строк trips, строк zone_metrics_5min, уникальных ts) в БД."""

    conn = sqlite3.connect(db)
    try:
        trips = conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
        metrics = conn.execute("SELECT COUNT(*) FROM zone_metrics_5min").fetchone()[0]
        stamps = conn.execute("SELECT COUNT(DISTINCT ts) FROM zone_metrics_5min").fetchone()[0]
    finally:
        conn.close()
    return trips, metrics, stamps


def test_cli_smoke(tmp_path):
    """Smoke 1 день: exit 0, таблицы наполнены, CSV = таблица + заголовок, ≤ 10 с."""

    db = tmp_path / "a.db"
    csv_dir = tmp_path / "csv"
    started = time.monotonic()
    result = run_cli(["--days", "1", "--seed", "42", "--db", str(db), "--csv", str(csv_dir)])
    elapsed = time.monotonic() - started

    assert result.returncode == 0, result.stderr
    assert elapsed <= TIME_LIMIT_SEC, f"прогон занял {elapsed:.1f} с > {TIME_LIMIT_SEC} с"
    assert result.stdout.rstrip("\n").count("\n") == 3  # ровно 4 строки сводки

    trips, metrics, stamps = table_counts(db)
    assert trips > 0
    assert metrics == 1728  # 1 день: 288 окон × 6 зон
    assert stamps == 288

    # CSV-экспорт обеих таблиц: строк в каждом файле = строк таблицы + 1 (A9).
    assert (csv_dir / "trips.csv").is_file()
    assert (csv_dir / "zone_metrics_5min.csv").is_file()
    assert sum(1 for _ in open(csv_dir / "trips.csv", encoding="utf-8")) == trips + 1
    assert sum(1 for _ in open(csv_dir / "zone_metrics_5min.csv", encoding="utf-8")) == metrics + 1


def test_cli_guard_rejects_nonempty_trips(tmp_path):
    """Повторный запуск в непустую БД: exit 2, «trips» в stderr, новых строк нет (A8)."""

    db = tmp_path / "a.db"
    first = run_cli(["--days", "1", "--seed", "42", "--db", str(db)])
    assert first.returncode == 0
    trips_before, _metrics, _stamps = table_counts(db)

    second = run_cli(["--days", "1", "--seed", "42", "--db", str(db)])
    assert second.returncode == 2
    assert "trips" in second.stderr
    assert "генерация требует пустой таблицы trips" in second.stderr
    trips_after, _metrics, _stamps = table_counts(db)
    assert trips_after == trips_before  # генерация не выполнялась


def test_cli_summary_progress_and_flags(tmp_path):
    """stdout — ровно 4 строки шаблона §4; stderr — прогресс; флаги переопределяют."""

    db = tmp_path / "override.db"
    result = run_cli(["--days", "1", "--seed", "42", "--db", str(db)])
    assert result.returncode == 0

    lines = result.stdout.rstrip("\n").split("\n")
    assert len(lines) == 4
    for line, pattern in zip(lines, SUMMARY_PATTERNS):
        assert re.fullmatch(pattern, line), (line, pattern)
    # Явные флаги отражены в первой строке сводки (days, seed, db прогона).
    assert f"days=1 seed=42 db={db}" == lines[0].split(" ", 1)[1]

    progress = re.findall(PROGRESS_PATTERN, result.stderr)
    assert len(progress) >= 1


def test_cli_summary_cancel_line(tmp_path):
    """С включёнными отказами stdout — 5 строк: ``cancel:`` перед ``wall:`` (4.2)."""

    config_path = tmp_path / "cancel.yaml"
    db = tmp_path / "cancel.db"
    config_path.write_text(
        "seed: 42\n"
        f"storage:\n  db_path: {db}\n"
        "drivers:\n"
        "  dispatch:\n"
        "    cancel_grace_min: 0\n"
        "    cancel_prob_per_min: 0.5\n",
        encoding="utf-8",
    )
    result = run_cli(["--days", "1", "--config", str(config_path)])
    assert result.returncode == 0, result.stderr

    lines = result.stdout.rstrip("\n").split("\n")
    assert len(lines) == 5
    cancel_match = re.fullmatch(r"cancel: cancelled=(\d+) cancelled_share=(\d+\.\d)%", lines[3])
    assert cancel_match, lines[3]
    assert int(cancel_match.group(1)) > 0, "сценарий без отказов не проверен"
    assert re.fullmatch(r"wall: \d+\.\ds", lines[4])

    # total второй строки включает отказы.
    total_match = re.fullmatch(
        r"trips: total=(\d+) completed=(\d+) completed_share=\d+\.\d% "
        r"unfulfilled=(\d+) unfulfilled_share=\d+\.\d%",
        lines[1],
    )
    assert total_match, lines[1]
    total, completed, unfulfilled = map(int, total_match.groups())
    assert total == completed + unfulfilled + int(cancel_match.group(1))


def test_cli_defaults_from_config(tmp_path):
    """Дефолты --seed/--db берутся из конфига, если флаги не заданы (спека §4)."""

    config_path = tmp_path / "custom.yaml"
    default_db = tmp_path / "defaults.db"
    config_path.write_text(f"seed: 7\nstorage:\n  db_path: {default_db}\n", encoding="utf-8")

    result = run_cli(["--days", "1", "--config", str(config_path)])
    assert result.returncode == 0, result.stderr
    first_line = result.stdout.splitlines()[0]
    assert first_line == f"sim: days=1 seed=7 db={default_db}"
    assert default_db.is_file()  # БД создана по пути из конфига

    # Явные флаги переопределяют значения конфига.
    override_db = tmp_path / "override.db"
    result = run_cli(
        ["--days", "1", "--config", str(config_path), "--seed", "42", "--db", str(override_db)]
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == f"sim: days=1 seed=42 db={override_db}"
