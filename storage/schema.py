"""DDL-схема SQLite: три таблицы контракта данных и три индекса.

Схема фиксирована планом M1 (§6, «точная, из контракта openspec/config.yaml»):
``trips``, ``zone_metrics_5min`` и ``forecasts`` создаются сразу, чтобы вехи
M2+ не меняли схему. Время хранится текстом «YYYY-MM-DD HH:MM:SS» (секунды 00).
Change ``m3-inference`` дополняет PK ``forecasts`` колонкой ``model_version``
(было ``(ts, zone_id, horizon_min)`` — схема M1 на одну модель): четыре версии
вехи M3 (два наива и две сети) лежат в таблице рядом на одних парах
(ts, зона, горизонт). Для БД, созданных до change'а,
:func:`ensure_forecasts_schema` пересоздаёт пустую таблицу старой формы
(непустая — отказ: по инварианту A5 M2 ``forecasts`` в существующих БД проекта
пуста).
"""

import sqlite3

from storage.db import transaction

TABLE_TRIPS = "trips"
"""Имя таблицы поездок (контракт данных, план M1 §6)."""

TABLE_ZONE_METRICS = "zone_metrics_5min"
"""Имя таблицы 5-минутных метрик зон."""

TABLE_FORECASTS = "forecasts"
"""Имя таблицы прогнозов."""

TABLES = (TABLE_TRIPS, TABLE_ZONE_METRICS, TABLE_FORECASTS)
"""Белый список имён таблиц контракта (решение design.md 6): SQL для выборок и
экспорта строится подстановкой имени только из этого кортежа, все значения —
через параметры ``?``; неизвестное имя отклоняется вызывающим кодом."""

# Полный DDL схемы. Каждый объект объявлен с IF NOT EXISTS, поэтому исполнение
# скрипта идемпотентно: повторный вызов не пересоздаёт таблицы и не трогает
# данные (решение design.md 1).
_DDL_TABLES = """
CREATE TABLE IF NOT EXISTS trips (
    trip_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    origin_zone INTEGER NOT NULL,
    dest_zone INTEGER NOT NULL,
    assigned_at TEXT,
    pickup_at TEXT,
    dropoff_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('completed', 'cancelled', 'unfulfilled')),
    price_multiplier REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS zone_metrics_5min (
    ts TEXT NOT NULL,
    zone_id INTEGER NOT NULL,
    requests INTEGER NOT NULL,
    completed INTEGER NOT NULL,
    unfulfilled INTEGER NOT NULL,
    free_drivers INTEGER NOT NULL,
    busy_drivers INTEGER NOT NULL,
    avg_wait_min REAL,
    supply_demand_ratio REAL NOT NULL,
    is_rain INTEGER NOT NULL,
    event TEXT NOT NULL,
    price_multiplier REAL NOT NULL,
    PRIMARY KEY (ts, zone_id)
);
"""

# Фрагмент forecasts вынесен из общего скрипта: ensure_forecasts_schema
# пересоздаёт таблицу тем же текстом, что и init_db (одна форма DDL на проект).
_DDL_FORECASTS = """
CREATE TABLE IF NOT EXISTS forecasts (
    ts TEXT NOT NULL,
    zone_id INTEGER NOT NULL,
    horizon_min INTEGER NOT NULL,
    pred_requests REAL NOT NULL,
    pred_free_drivers REAL NOT NULL,
    deficit_prob REAL NOT NULL,
    model_version TEXT NOT NULL,
    PRIMARY KEY (ts, zone_id, horizon_min, model_version)
);
"""

_DDL_FORECASTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_forecasts_model_version ON forecasts (model_version);
"""

DDL = (
    _DDL_TABLES
    + _DDL_FORECASTS
    + """
CREATE INDEX IF NOT EXISTS idx_trips_created_at ON trips (created_at);
CREATE INDEX IF NOT EXISTS idx_trips_origin_zone ON trips (origin_zone);
"""
    + _DDL_FORECASTS_INDEX
)

FORECASTS_NEW_PK = ("ts", "zone_id", "horizon_min", "model_version")
"""PK таблицы forecasts с версии change ``m3-inference`` (несколько версий моделей)."""

FORECASTS_LEGACY_PK = ("ts", "zone_id", "horizon_min")
"""PK таблицы forecasts схемы M1 (одна модель) — форма для миграции."""


def _forecasts_pk(conn: sqlite3.Connection) -> tuple[str, ...] | None:
    """PK таблицы forecasts по ``PRAGMA table_info``; ``None`` — таблицы нет.

    Доступ по позициям колонок прагмы (1 — имя, 5 — порядковый номер в PK):
    работает и с ``sqlite3.Row``, и с кортежами «сырых» соединений.
    """

    rows = conn.execute(f"PRAGMA table_info({TABLE_FORECASTS})").fetchall()
    if not rows:
        return None
    members = sorted((row[5], row[1]) for row in rows if row[5])
    return tuple(name for _, name in members)


def ensure_forecasts_schema(conn: sqlite3.Connection) -> None:
    """Привести ``forecasts`` к PK с ``model_version`` на существующей БД (идемпотентно).

    Таблицы нет или PK уже новой формы — no-op; старая форма и пустая таблица —
    DROP + CREATE в одной транзакции (вместе с индексом версии, который падает
    вместе с таблицей); старая и непустая — ``RuntimeError``: по инварианту A5 M2
    в существующих БД проекта ``forecasts`` пуста, непустая таблица означает
    чужие данные, которые молча терять нельзя. Иной PK — тоже отказ: схема не
    из этого проекта.
    """

    pk = _forecasts_pk(conn)
    if pk is None or pk == FORECASTS_NEW_PK:
        return
    if pk != FORECASTS_LEGACY_PK:
        raise RuntimeError(
            f"неизвестная форма PK таблицы forecasts: {pk} — ожидается "
            f"{FORECASTS_LEGACY_PK} (схема M1) или {FORECASTS_NEW_PK}"
        )
    n_rows = conn.execute(f"SELECT COUNT(*) FROM {TABLE_FORECASTS}").fetchone()[0]
    if n_rows:
        raise RuntimeError(
            f"таблица forecasts старой схемы (PK без model_version) содержит {n_rows} строк —"
            f" миграция отклонена: очистите таблицу (DELETE FROM {TABLE_FORECASTS})"
            " или пересоздайте БД"
        )
    with transaction(conn):
        conn.execute(f"DROP TABLE {TABLE_FORECASTS}")
        conn.execute(_DDL_FORECASTS)
        conn.execute(_DDL_FORECASTS_INDEX)


def init_db(conn: sqlite3.Connection) -> None:
    """Создать схему БД, если её ещё нет, и привести ``forecasts`` к текущему PK.

    Исполняет весь DDL одним ``executescript`` (благодаря IF NOT EXISTS повторный
    вызов на существующей БД не пересоздаёт таблицы и сохраняет строки), затем
    мигрирует таблицу ``forecasts`` старой формы — пустая пересоздаётся с PK
    ``(ts, zone_id, horizon_min, model_version)`` (change ``m3-inference``).
    """
    conn.executescript(DDL)
    ensure_forecasts_schema(conn)
    conn.commit()
