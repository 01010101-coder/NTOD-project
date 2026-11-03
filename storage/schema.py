"""DDL-схема SQLite: три таблицы контракта данных и три индекса.

Схема фиксирована планом M1 (§6, «точная, из контракта openspec/config.yaml»):
``trips``, ``zone_metrics_5min`` и ``forecasts`` создаются сразу, чтобы вехи
M2+ не меняли схему. Время хранится текстом «YYYY-MM-DD HH:MM:SS» (секунды 00);
``forecasts`` до вехи M3 остаётся пустой.
"""

import sqlite3

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
DDL = """
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

CREATE TABLE IF NOT EXISTS forecasts (
    ts TEXT NOT NULL,
    zone_id INTEGER NOT NULL,
    horizon_min INTEGER NOT NULL,
    pred_requests REAL NOT NULL,
    pred_free_drivers REAL NOT NULL,
    deficit_prob REAL NOT NULL,
    model_version TEXT NOT NULL,
    PRIMARY KEY (ts, zone_id, horizon_min)
);

CREATE INDEX IF NOT EXISTS idx_trips_created_at ON trips (created_at);
CREATE INDEX IF NOT EXISTS idx_trips_origin_zone ON trips (origin_zone);
CREATE INDEX IF NOT EXISTS idx_forecasts_model_version ON forecasts (model_version);
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Создать схему БД, если её ещё нет (идемпотентно).

    Исполняет весь DDL одним ``executescript``; благодаря IF NOT EXISTS в каждом
    объекте повторный вызов на существующей БД завершается без ошибок, не
    пересоздаёт таблицы и сохраняет ранее записанные строки.
    """
    conn.executescript(DDL)
    conn.commit()
