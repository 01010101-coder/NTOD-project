"""Контроль целостности эпохи M3 (change ``m3-dataset``, задача 1.3).

Сравнивает md5 CSV-выгрузок ``trips`` и ``zone_metrics_5min`` первых
``ml.train_days`` дней БД эпохи (``ml.db``, по умолчанию ``data/m3_epoch.db``)
с выгрузками эталона ``data/taxi.db``. Префикс эпохи обязан совпадать побитово
(детерминизм A7 плана M2): единственный 37-дневный прогон пишет те же строки,
что 30-дневный эталон, плюс тест-продолжение. Запуск — ``make epoch-check``
(ручная цель, вне ``make test``: БД в CI нет).

Выгрузки обеих сторон делаются одним ``storage.csv_export.export_table`` во
временные файлы, поэтому формат байт-в-байт одинаковый; префикс эпохи
копируется во временную БД (``ATTACH`` + ``INSERT … SELECT … WHERE``) с
сохранением естественного порядка строк. Результат — PASS/FAIL по каждой
таблице; exit-код равен числу провалов (прецедент ``scripts/accept_m2.sh``).
"""

import hashlib
import tempfile
from datetime import timedelta
from pathlib import Path

from common.config import load_config
from common.timeutil import from_iso, to_iso
from storage.csv_export import export_table
from storage.db import connect
from storage.schema import TABLE_TRIPS, TABLE_ZONE_METRICS, init_db

TAXI_DB = "data/taxi.db"
"""Эталон 30-дневного прогона (seed 42) — префикс эпохи обязан ему совпадать."""


def _md5(path: Path) -> str:
    """md5-хеш содержимого файла (шестнадцатеричная строка)."""

    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_prefix(epoch_db: str, cutoff: str, out_dir: Path) -> dict[str, Path]:
    """Выгрузить строки эпохи со временем < ``cutoff`` в CSV через ``export_table``.

    Префикс копируется во временную БД со схемой контракта (порядок вставки =
    естественный порядок сканирования, как в эталоне), после чего таблицы
    выгружаются тем же кодом, что и эталонные стороны.
    """

    prefix_db = out_dir / "epoch-prefix.db"
    conn = connect(str(prefix_db))
    init_db(conn)
    conn.execute("ATTACH DATABASE ? AS epoch", (epoch_db,))
    conn.execute(
        f"INSERT INTO main.{TABLE_TRIPS} SELECT * FROM epoch.{TABLE_TRIPS} WHERE created_at < ?",
        (cutoff,),
    )
    conn.execute(
        f"INSERT INTO main.{TABLE_ZONE_METRICS} SELECT * FROM epoch.{TABLE_ZONE_METRICS} "
        "WHERE ts < ?",
        (cutoff,),
    )
    paths: dict[str, Path] = {}
    for table in (TABLE_TRIPS, TABLE_ZONE_METRICS):
        out = out_dir / f"epoch-{table}.csv"
        export_table(conn, table, str(out))
        paths[table] = out
    conn.close()
    return paths


def main() -> int:
    """Сравнить md5 префикса эпохи и эталона; exit — число несовпавших таблиц."""

    cfg = load_config("config.yaml")
    epoch_db = cfg.ml.db
    if not Path(epoch_db).exists():
        print(f"FAIL epoch-check: БД эпохи {epoch_db} не найдена — сначала make epoch")
        return 1
    if not Path(TAXI_DB).exists():
        print(f"FAIL epoch-check: эталон {TAXI_DB} не найден")
        return 1

    sim_start = from_iso(cfg.engine.sim_start + " 00:00:00")
    cutoff = to_iso(sim_start + timedelta(days=cfg.ml.train_days))

    fails = 0
    with tempfile.TemporaryDirectory(prefix="epoch-check-") as tmp:
        out_dir = Path(tmp)
        export_table(connect(TAXI_DB), TABLE_TRIPS, str(out_dir / "taxi-trips.csv"))
        export_table(connect(TAXI_DB), TABLE_ZONE_METRICS, str(out_dir / "taxi-metrics.csv"))
        prefix_paths = _export_prefix(epoch_db, cutoff, out_dir)
        taxi_paths = {
            TABLE_TRIPS: out_dir / "taxi-trips.csv",
            TABLE_ZONE_METRICS: out_dir / "taxi-metrics.csv",
        }
        for table in (TABLE_TRIPS, TABLE_ZONE_METRICS):
            epoch_hash = _md5(prefix_paths[table])
            taxi_hash = _md5(taxi_paths[table])
            if epoch_hash == taxi_hash:
                print(f"PASS {table}: md5 совпадают ({epoch_hash})")
            else:
                print(
                    f"FAIL {table}: эпоха {epoch_hash} != эталон {taxi_hash} (префикс < {cutoff})"
                )
                fails += 1
    if fails == 0:
        print(f"epoch-check: префикс {cfg.ml.train_days} дней {epoch_db} побитово = {TAXI_DB}")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
