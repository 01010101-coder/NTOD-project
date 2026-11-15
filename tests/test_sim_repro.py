"""Тесты детерминизма change ``m2-loop`` (задача 4.3, §8 плана M2).

``test_reproducibility`` — краеугольный тест спеки simulator-loop: два
независимых полных прогона движка по 7 дней с одним seed 42 (полная неделя от
эталонного старта-понедельника: будни и выходные, «пятничный вечер» пт+сб,
дожди и surge-инъекции успевают разыграться — proposal, вопрос 4) дают побайтово
одинаковые таблицы: ``trips`` сравниваются по порядку вставки (trip_id — alias
rowid, §10.11), ``zone_metrics_5min`` — по ключу ``(ts, zone_id)``. Прогоны
пишут в файловые БД в ``tmp_path`` через sqlite-writer из ``__main__.py`` —
тот же путь записи, что и у CLI. Соседний тест: seed 43 даёт отличный префикс
``trips`` (сценарий «Другой seed даёт другую историю»).
"""

from pathlib import Path

from common.config import load_config
from simulator.__main__ import SqliteTripWriter
from simulator.engine import Engine

REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"

DAYS = 7  # полная неделя от sim_start (proposal, вопрос 4)
SEED = 42
PREFIX_ROWS = 100  # длина сравниваемого префикса trips у другого seed


def _run_to_db(tmp_path: Path, seed: int, db_name: str):
    """Полный прогон ``DAYS`` суток с ``seed`` в файловую БД через sqlite-writer."""

    from storage.db import connect
    from storage.schema import init_db

    cfg = load_config(REFERENCE_CONFIG)
    conn = connect(str(tmp_path / db_name))
    init_db(conn)
    summary = Engine(cfg, seed=seed).run(DAYS, SqliteTripWriter(conn))
    return conn, summary


def test_reproducibility(tmp_path):
    """Два прогона одного seed дают идентичные trips и zone_metrics_5min."""

    conn_a, summary_a = _run_to_db(tmp_path, SEED, "a.db")
    conn_b, summary_b = _run_to_db(tmp_path, SEED, "b.db")

    assert summary_a.trips_total == summary_b.trips_total > 0
    assert summary_a.completed == summary_b.completed
    assert summary_a.unfulfilled == summary_b.unfulfilled

    trips_a = conn_a.execute("SELECT * FROM trips ORDER BY trip_id").fetchall()
    trips_b = conn_b.execute("SELECT * FROM trips ORDER BY trip_id").fetchall()
    assert len(trips_a) == len(trips_b) == summary_a.trips_total
    assert [tuple(row) for row in trips_a] == [tuple(row) for row in trips_b]

    metrics_a = conn_a.execute("SELECT * FROM zone_metrics_5min ORDER BY ts, zone_id").fetchall()
    metrics_b = conn_b.execute("SELECT * FROM zone_metrics_5min ORDER BY ts, zone_id").fetchall()
    assert len(metrics_a) == len(metrics_b) == DAYS * 288 * 6
    assert [tuple(row) for row in metrics_a] == [tuple(row) for row in metrics_b]


def test_different_seed_differs(tmp_path):
    """Другой seed даёт другую историю: префикс trips по trip_id не идентичен."""

    conn_a, _summary_a = _run_to_db(tmp_path, SEED, "a.db")
    conn_c, _summary_c = _run_to_db(tmp_path, 43, "c.db")

    trips_a = [
        tuple(row)
        for row in conn_a.execute("SELECT * FROM trips ORDER BY trip_id").fetchall()[:PREFIX_ROWS]
    ]
    trips_c = [
        tuple(row)
        for row in conn_c.execute("SELECT * FROM trips ORDER BY trip_id").fetchall()[:PREFIX_ROWS]
    ]
    assert trips_a != trips_c
