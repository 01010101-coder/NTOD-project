"""Демо-скрипт слоя хранения: ``python -m storage.demo``.

Создаёт временную БД и CSV внутри ``tempfile.TemporaryDirectory`` — эталонная
``data/taxi.db`` из ``config.yaml`` не затрагивается (решение design.md 8,
демо безопасно для повторного запуска и CI). Поток работы: ``init_db`` → по
одной строке в каждую таблицу контракта через репозиторий → чтение выборками →
``export_table`` для трёх таблиц → печать сводки (путь БД, число строк, пути
CSV). Сообщения идут через ``common.logging_setup`` (stderr); код возврата 0.
"""

import logging
import os
import tempfile
from datetime import datetime

from common.logging_setup import setup_logging
from storage.csv_export import export_table
from storage.db import connect
from storage.repo import (
    insert_forecasts,
    insert_trips,
    insert_zone_metrics,
    list_forecasts,
    list_trips,
    list_zone_metrics,
)
from storage.schema import TABLES, init_db

_log = logging.getLogger(__name__)


def main() -> int:
    """Выполнить демо-цикл записи/чтения/экспорта; вернуть код возврата процесса."""
    setup_logging()
    with tempfile.TemporaryDirectory(prefix="storage_demo_") as tmp:
        db_path = os.path.join(tmp, "demo.db")
        conn = connect(db_path)
        init_db(conn)

        # По одной строке в каждую таблицу контракта через репозиторий.
        trip_ids = insert_trips(
            conn,
            [
                {
                    "created_at": datetime(2026, 1, 5, 8, 0),
                    "origin_zone": 1,
                    "dest_zone": 2,
                    "assigned_at": None,
                    "status": "completed",
                    "price_multiplier": 1.2,
                }
            ],
        )
        insert_zone_metrics(
            conn,
            [
                {
                    "ts": datetime(2026, 1, 5, 8, 0),
                    "zone_id": 1,
                    "requests": 10,
                    "completed": 8,
                    "unfulfilled": 2,
                    "free_drivers": 3,
                    "busy_drivers": 5,
                    "avg_wait_min": 4.5,
                    "supply_demand_ratio": 0.3,
                    "is_rain": 0,
                    "event": "",
                    "price_multiplier": 1.2,
                }
            ],
        )
        insert_forecasts(
            conn,
            [
                {
                    "ts": datetime(2026, 1, 5, 8, 0),
                    "zone_id": 1,
                    "horizon_min": 15,
                    "pred_requests": 12.0,
                    "pred_free_drivers": 2.5,
                    "deficit_prob": 0.7,
                    "model_version": "demo",
                }
            ],
        )

        # Чтение записанного обратно выборками слоя доступа.
        counts = {
            "trips": len(list_trips(conn)),
            "zone_metrics_5min": len(list_zone_metrics(conn)),
            "forecasts": len(list_forecasts(conn)),
        }

        # Выгрузка всех трёх таблиц в CSV рядом с БД.
        csv_paths = {}
        for table in TABLES:
            path = os.path.join(tmp, f"{table}.csv")
            export_table(conn, table, path)
            csv_paths[table] = path
        conn.close()

        # Сводка демо: путь БД, число строк каждой таблицы, пути CSV.
        _log.info("Демо слоя хранения: trip_id=%s", trip_ids)
        _log.info("БД: %s", db_path)
        _log.info(
            "Строк записано и прочитано: %s",
            ", ".join(f"{name}={count}" for name, count in counts.items()),
        )
        for table, path in csv_paths.items():
            _log.info("CSV %s: %s", table, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
