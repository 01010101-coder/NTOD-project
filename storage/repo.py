"""Слой доступа к данным: пакетная вставка и выборки для таблиц контракта.

По паре функций на таблицу (решение design.md 4); вставляемые строки — dict,
выборки возвращают ``sqlite3.Row`` с доступом к колонкам по именам. Время на
входе — наивный ``datetime``, конвертация только через ``common.timeutil.to_iso``
(решение design.md 5): лексикографический порядок ISO-текста «YYYY-MM-DD HH:MM:SS»
совпадает с хронологическим, поэтому фильтры диапазона сравнивают тексты.
Имена таблиц в SQL — константы белого списка из ``storage.schema``
(решение design.md 6), все значения — через параметры ``?``.
"""

import sqlite3
from datetime import datetime

from common.timeutil import to_iso
from storage.db import transaction
from storage.schema import TABLE_FORECASTS, TABLE_TRIPS, TABLE_ZONE_METRICS

# trip_id не указывается в INSERT: SQLite генерирует его как alias rowid.
_INSERT_TRIPS = f"""
INSERT INTO {TABLE_TRIPS}
    (created_at, origin_zone, dest_zone, assigned_at, pickup_at, dropoff_at,
     status, price_multiplier)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_ZONE_METRICS = f"""
INSERT INTO {TABLE_ZONE_METRICS}
    (ts, zone_id, requests, completed, unfulfilled, free_drivers, busy_drivers,
     avg_wait_min, supply_demand_ratio, is_rain, event, price_multiplier)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_FORECASTS = f"""
INSERT INTO {TABLE_FORECASTS}
    (ts, zone_id, horizon_min, pred_requests, pred_free_drivers, deficit_prob,
     model_version)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""


def _iso_or_none(value: datetime | None) -> str | None:
    """datetime в ISO-текст; None проходит насквозь (NULL-допустимые колонки времени)."""
    return None if value is None else to_iso(value)


def _select(
    conn: sqlite3.Connection,
    table: str,
    time_col: str,
    zone_col: str,
    *,
    zone: int | None = None,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    eq_filters: dict[str, object] | None = None,
    order_by: str,
) -> list[sqlite3.Row]:
    """Выборка ``SELECT *`` с опциональными фильтрами по зоне и диапазону времени.

    Имена колонок приходят только из внутренних литералов вызовов ниже, значения
    — параметрами ``?``. Границы диапазона включаются (сценарий «Фильтр по
    диапазону времени» specs); без фильтров возвращаются все строки таблицы.
    """
    sql = f"SELECT * FROM {table}"
    clauses: list[str] = []
    params: list[object] = []
    if zone is not None:
        clauses.append(f"{zone_col} = ?")
        params.append(zone)
    if time_from is not None:
        clauses.append(f"{time_col} >= ?")
        params.append(to_iso(time_from))
    if time_to is not None:
        clauses.append(f"{time_col} <= ?")
        params.append(to_iso(time_to))
    for col, value in (eq_filters or {}).items():
        if value is not None:
            clauses.append(f"{col} = ?")
            params.append(value)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += f" ORDER BY {order_by}"
    return conn.execute(sql, params).fetchall()


def insert_trips(conn: sqlite3.Connection, rows: list[dict]) -> list[int]:
    """Вставить батч строк ``trips`` одной операцией ``executemany`` (одна транзакция).

    ``trip_id`` в строках не передаётся — SQLite генерирует его; возвращается
    список сгенерированных id (из ``lastrowid`` последней строки минус длина
    батча); пустой батч даёт пустой список. Обязательные ключи строки:
    ``created_at`` (datetime), ``origin_zone``, ``dest_zone``, ``status``,
    ``price_multiplier``; необязательные (None, если нет): ``assigned_at``,
    ``pickup_at``, ``dropoff_at``.
    """
    if not rows:
        return []
    params = [
        (
            to_iso(row["created_at"]),
            row["origin_zone"],
            row["dest_zone"],
            _iso_or_none(row.get("assigned_at")),
            _iso_or_none(row.get("pickup_at")),
            _iso_or_none(row.get("dropoff_at")),
            row["status"],
            row["price_multiplier"],
        )
        for row in rows
    ]
    with transaction(conn):
        conn.executemany(_INSERT_TRIPS, params)
        # cursor.lastrowid после executemany равен None (Python 3.11), поэтому
        # «последний сгенерированный id» берём SQL-функцией last_insert_rowid():
        # это rowid последней вставленной в данном соединении строки.
        last_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return list(range(last_id - len(rows) + 1, last_id + 1))


def list_trips(
    conn: sqlite3.Connection,
    zone: int | None = None,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
) -> list[sqlite3.Row]:
    """Выбрать строки ``trips``; фильтры — по зоне происхождения и диапазону ``created_at``."""
    return _select(
        conn,
        TABLE_TRIPS,
        "created_at",
        "origin_zone",
        zone=zone,
        time_from=time_from,
        time_to=time_to,
        order_by="trip_id",
    )


def insert_zone_metrics(conn: sqlite3.Connection, rows: list[dict]) -> None:
    """Вставить батч строк ``zone_metrics_5min`` одной операцией (одна транзакция).

    PK ``(ts, zone_id)`` задаёт вызывающий код; ключи строки: ``ts`` (datetime),
    ``zone_id``, ``requests``, ``completed``, ``unfulfilled``, ``free_drivers``,
    ``busy_drivers``, ``avg_wait_min`` (может быть None), ``supply_demand_ratio``,
    ``is_rain``, ``event``, ``price_multiplier``.
    """
    if not rows:
        return
    params = [
        (
            to_iso(row["ts"]),
            row["zone_id"],
            row["requests"],
            row["completed"],
            row["unfulfilled"],
            row["free_drivers"],
            row["busy_drivers"],
            row.get("avg_wait_min"),
            row["supply_demand_ratio"],
            row["is_rain"],
            row["event"],
            row["price_multiplier"],
        )
        for row in rows
    ]
    with transaction(conn):
        conn.executemany(_INSERT_ZONE_METRICS, params)


def list_zone_metrics(
    conn: sqlite3.Connection,
    zone: int | None = None,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
) -> list[sqlite3.Row]:
    """Выбрать строки ``zone_metrics_5min``; фильтры — по зоне и диапазону ``ts``."""
    return _select(
        conn,
        TABLE_ZONE_METRICS,
        "ts",
        "zone_id",
        zone=zone,
        time_from=time_from,
        time_to=time_to,
        order_by="ts, zone_id",
    )


def insert_forecasts(conn: sqlite3.Connection, rows: list[dict]) -> None:
    """Вставить батч строк ``forecasts`` одной операцией (одна транзакция).

    PK ``(ts, zone_id, horizon_min, model_version)`` задаёт вызывающий код;
    ключи строки: ``ts`` (datetime), ``zone_id``, ``horizon_min``,
    ``pred_requests``, ``pred_free_drivers``, ``deficit_prob``, ``model_version``.
    """
    if not rows:
        return
    params = [
        (
            to_iso(row["ts"]),
            row["zone_id"],
            row["horizon_min"],
            row["pred_requests"],
            row["pred_free_drivers"],
            row["deficit_prob"],
            row["model_version"],
        )
        for row in rows
    ]
    with transaction(conn):
        conn.executemany(_INSERT_FORECASTS, params)


def delete_forecasts(conn: sqlite3.Connection, model_version: str) -> int:
    """Удалить все строки одной версии модели (одна транзакция); вернуть число удалённых.

    Повторный прогон инференса (change ``m3-inference``) вызывает перед вставкой:
    замена строго своих строк — идемпотентность вместо падения на PK; строки
    чужих ``model_version`` и другие таблицы не затрагиваются.
    """

    with transaction(conn):
        cursor = conn.execute(
            f"DELETE FROM {TABLE_FORECASTS} WHERE model_version = ?", (model_version,)
        )
        return cursor.rowcount


def list_forecasts(
    conn: sqlite3.Connection,
    zone: int | None = None,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    model_version: str | None = None,
) -> list[sqlite3.Row]:
    """Выбрать строки ``forecasts``; фильтры — по зоне, диапазону ``ts`` и версии модели."""
    return _select(
        conn,
        TABLE_FORECASTS,
        "ts",
        "zone_id",
        zone=zone,
        time_from=time_from,
        time_to=time_to,
        eq_filters={"model_version": model_version},
        order_by="ts, zone_id, horizon_min",
    )
