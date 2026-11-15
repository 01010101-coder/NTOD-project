"""CLI генерации истории M2: ``python -m simulator --days N [...]`` (§4 плана M2).

Аргументы: ``--days`` (обязателен), ``--config`` (по умолчанию ``config.yaml``),
``--seed``/``--db`` (по умолчанию из конфига, флаги переопределяют), ``--csv DIR``
(каталог создаётся, обе таблицы выгружаются через ``storage.csv_export``).
Guard до прогона: непустая таблица ``trips`` в целевой БД — сообщение в stderr и
exit 2 (``trip_id`` назначает SQLite при вставке батчей в порядке создания,
§10.11, поэтому генерация требует пустой таблицы). Прогресс — раз в сим-час в
stderr машинного формата ``[YYYY-MM-DD HH:MM] trips=N elapsed=X.Xs``; сводка —
в stdout ровно 4 строки формата §4. Sqlite-реализация шва ``TripWriter``
(§10.12), guard-сообщение и печати прогресса/сводки живут в общем модуле
``simulator.writers`` (change ``m6-rng-branches``) и переиспользуются отсюда
импортом — тот же формат §4 использует CLI ``simulator.branches``; логики мира
в модуле нет.
"""

import argparse
import os
import sys

from common.config import AppConfig, load_config
from simulator.engine import Engine
from simulator.writers import GUARD_MESSAGE, SqliteTripWriter, print_progress, print_summary
from storage.csv_export import export_table
from storage.db import connect
from storage.schema import init_db


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Аргументы CLI §4: ``--days`` обязателен, остальное — с дефолтами."""

    parser = argparse.ArgumentParser(
        prog="simulator",
        description="Генерация истории города такси в SQLite (веха M2, §4 плана).",
    )
    parser.add_argument("--days", type=int, required=True, help="число суток генерации")
    parser.add_argument("--config", default="config.yaml", help="путь к конфигу YAML")
    parser.add_argument("--seed", type=int, default=None, help="seed (по умолчанию из конфига)")
    parser.add_argument("--db", default=None, help="путь БД (по умолчанию из конфига)")
    parser.add_argument(
        "--csv", default=None, help="каталог выгрузки trips.csv и zone_metrics_5min.csv"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI: guard → генерация → сводка → CSV; exit 0/2 (§4)."""

    args = _parse_args(argv)
    cfg: AppConfig = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    db_path = cfg.storage.db_path if args.db is None else args.db

    # Guard §4 до прогона: connect/init_db создают новую БД со схемой M1,
    # непустая таблица trips — генерация не выполняется (exit 2, A8).
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = connect(db_path)
    init_db(conn)
    existing = conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    if existing > 0:
        print(GUARD_MESSAGE, file=sys.stderr)
        return 2

    summary = Engine(cfg, seed=seed).run(args.days, SqliteTripWriter(conn), progress=print_progress)
    print_summary(summary, db_path)

    if args.csv:
        os.makedirs(args.csv, exist_ok=True)
        export_table(conn, "trips", os.path.join(args.csv, "trips.csv"))
        export_table(conn, "zone_metrics_5min", os.path.join(args.csv, "zone_metrics_5min.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
