"""Сквозная проверка смен (change ``world-driver-shifts``, задача 6.1).

Два мира одной эпохи (7 суток, seed 42) во временных БД (не в ``data/``):
``drivers.count = 240`` со сменами против ``drivers.count = 160`` без смен.
Средняя доля парка на смене профиля по умолчанию ≈ 0.68, поэтому 240 со
сменами и 160 без смен дают сопоставимый средний парк — сравнение честное.
Скрипт печатает почасовую загрузку (busy / (free + busy) по строкам
``zone_metrics_5min``) и потери (сумма ``unfulfilled``) по часам, затем итог:
ночная загрузка 02–05 ч и доля потерь вечернего спада предложения.

Запуск: ``.venv/bin/python scripts/shifts_experiment.py``. Базы создаются в
``tempfile.TemporaryDirectory`` и удаляются после прогона.
"""

import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path

from common.config import AppConfig, load_config
from simulator.__main__ import SqliteTripWriter
from simulator.engine import Engine
from storage.db import connect
from storage.schema import init_db

DAYS = 7
SEED = 42
SHIFT_COUNT = 240
BASE_COUNT = 160
NIGHT_HOURS = range(2, 5)  # бакеты 02:00–05:00
EVENING_HOURS = range(17, 22)  # вечерний спад предложения

REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def run_world(conn: sqlite3.Connection, count: int, shifts: bool) -> AppConfig:
    """Прогнать 7 суток в готовую временную БД; вернуть конфиг мира."""

    cfg = load_config(REFERENCE_CONFIG).model_copy(deep=True)
    cfg.drivers.count = count
    cfg.drivers.shifts.enabled = shifts
    Engine(cfg, seed=SEED).run(DAYS, SqliteTripWriter(conn))
    return cfg


def hourly_rows(conn: sqlite3.Connection) -> dict[int, dict[str, int]]:
    """Свернуть строки ``zone_metrics_5min`` по часам суток.

    Загрузка хранится двумя способами: пул ``busy / (free + busy)`` по часу и
    средняя по строкам (зона × 5-минутное окно) — первая взвешивает зоны по
    размеру парка, вторая читается как «типичная загрузка зоны».
    """

    totals: dict[int, dict[str, int]] = defaultdict(
        lambda: {
            "free": 0,
            "busy": 0,
            "unfulfilled": 0,
            "requests": 0,
            "completed": 0,
            "load_sum": 0.0,
            "rows": 0,
        }
    )
    rows = conn.execute(
        "SELECT ts, free_drivers, busy_drivers, unfulfilled, requests, completed"
        " FROM zone_metrics_5min"
    )
    for ts, free, busy, unfulfilled, requests, completed in rows:
        hour = int(ts[11:13])
        bucket = totals[hour]
        bucket["free"] += free
        bucket["busy"] += busy
        bucket["unfulfilled"] += unfulfilled
        bucket["requests"] += requests
        bucket["completed"] += completed
        bucket["load_sum"] += 100.0 * busy / max(free + busy, 1)
        bucket["rows"] += 1
    return dict(totals)


def load_percent(row: dict[str, int]) -> float:
    """Загрузка парка на смене: busy / (free + busy) в процентах."""

    total = row["free"] + row["busy"]
    return 100.0 * row["busy"] / total if total else 0.0


def zone_load_percent(row: dict[str, int]) -> float:
    """Средняя по зонам и 5-минуткам загрузка часа в процентах."""

    return row["load_sum"] / row["rows"] if row["rows"] else 0.0


def main() -> None:
    """Прогнать два мира, напечатать почасовую таблицу и итог одной строкой."""

    results: dict[str, dict[int, dict[str, int]]] = {}
    with tempfile.TemporaryDirectory(prefix="shifts-experiment-") as tmp:
        for label, count, shifts in (
            ("240 со сменами", SHIFT_COUNT, True),
            ("160 без смен", BASE_COUNT, False),
        ):
            conn = connect(str(Path(tmp) / f"world_{count}.db"))
            init_db(conn)
            run_world(conn, count, shifts)
            results[label] = hourly_rows(conn)
            conn.close()

    header = f"{'мир':<16}{'час':>4}{'загрузка':>10}{'ср. по зонам':>14}{'потери':>9}{'заказы':>9}"
    print(header)
    print("-" * len(header))
    for label, totals in results.items():
        for hour in range(24):
            row = totals[hour]
            print(
                f"{label:<16}{hour:>4}{load_percent(row):>9.1f}%"
                f"{zone_load_percent(row):>13.1f}%{row['unfulfilled']:>9}{row['requests']:>9}"
            )

    shifting = results["240 со сменами"]
    baseline = results["160 без смен"]
    night_on = sum(zone_load_percent(shifting[h]) for h in NIGHT_HOURS) / len(NIGHT_HOURS)
    night_off = sum(zone_load_percent(baseline[h]) for h in NIGHT_HOURS) / len(NIGHT_HOURS)
    night_pool_on = sum(load_percent(shifting[h]) for h in NIGHT_HOURS) / len(NIGHT_HOURS)
    night_pool_off = sum(load_percent(baseline[h]) for h in NIGHT_HOURS) / len(NIGHT_HOURS)
    evening_losses = sum(shifting[h]["unfulfilled"] for h in EVENING_HOURS)
    total_losses = sum(row["unfulfilled"] for row in shifting.values())
    print("-" * len(header))
    print(
        f"итог: ночная загрузка 02–05 ч со сменами {night_on:.1f}% по зонам "
        f"(пул {night_pool_on:.1f}%) против {night_off:.1f}% по зонам "
        f"(пул {night_pool_off:.1f}%) без смен; потери вечером 17–22 ч "
        f"{100.0 * evening_losses / total_losses if total_losses else 0.0:.1f}% всех потерь"
    )


if __name__ == "__main__":
    main()
