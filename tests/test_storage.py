"""Тесты storage/: схема, прагмы, транзакции, репозиторий, CSV-экспорт, демо-скрипт.

Покрывают сценарии specs change'а m1-storage (capability data-storage):
«Инициализация пустой БД создаёт схему», «Недопустимый статус trips
отклоняется», «Повторный init без ошибок», «Прагмы применены», «Коммит при
успехе», «Rollback при исключении», roundtrip всех трёх таблиц, «Формат
хранения проверен напрямую в БД», «CSV содержит те же строки» и «Демо-скрипт
завершается успехом» (решение design.md 9: tmp_path и «сырое» соединение).
"""

import csv
import sqlite3
import subprocess
import sys
from datetime import datetime

import pytest

from common.timeutil import from_iso, to_iso
from storage.csv_export import export_table
from storage.db import connect, transaction
from storage.repo import (
    insert_forecasts,
    insert_trips,
    insert_zone_metrics,
    list_forecasts,
    list_trips,
    list_zone_metrics,
)
from storage.schema import init_db

CONTRACT_TABLES = {"trips", "zone_metrics_5min", "forecasts"}
"""Ровно этот набор таблиц должен создать init_db (план M1 §6)."""

INDEX_NAMES = {"idx_trips_created_at", "idx_trips_origin_zone", "idx_forecasts_model_version"}
"""Именованные индексы change'а (без sqlite_autoindex_*)."""

TRIPS_COLUMNS = [
    "trip_id",
    "created_at",
    "origin_zone",
    "dest_zone",
    "assigned_at",
    "pickup_at",
    "dropoff_at",
    "status",
    "price_multiplier",
]
"""Колонки trips в порядке схемы §6 (9 штук)."""

TRIP = {
    "created_at": datetime(2026, 1, 5, 8, 0),
    "origin_zone": 1,
    "dest_zone": 2,
    "assigned_at": None,  # заказ не назначен: NULL должен пережить roundtrip
    "pickup_at": datetime(2026, 1, 5, 8, 3),
    "dropoff_at": datetime(2026, 1, 5, 8, 25),
    "status": "completed",
    "price_multiplier": 1.2,
}

METRIC = {
    "ts": datetime(2026, 1, 5, 8, 0),
    "zone_id": 1,
    "requests": 10,
    "completed": 8,
    "unfulfilled": 2,
    "free_drivers": 3,
    "busy_drivers": 5,
    "avg_wait_min": None,  # никто не назначен в интервале: NULL должен пережить roundtrip
    "supply_demand_ratio": 0.3,
    "is_rain": 0,
    "event": "",
    "price_multiplier": 1.2,
}

FORECAST = {
    "ts": datetime(2026, 1, 5, 8, 0),
    "zone_id": 1,
    "horizon_min": 15,
    "pred_requests": 12.0,
    "pred_free_drivers": 2.5,
    "deficit_prob": 0.7,
    "model_version": "v0",
}

# «Сырая» вставка trips: ровные значения колонок без trip_id (для тестов схемы
# и транзакций, где репозиторий не используется, чтобы не порождать вложенные
# транзакции внутри with transaction(conn):).
_RAW_TRIP_INSERT = (
    "INSERT INTO trips (created_at, origin_zone, dest_zone, status, price_multiplier) "
    "VALUES ('2026-01-05 08:00:00', 1, 2, 'completed', 1.0)"
)


@pytest.fixture()
def db_path(tmp_path):
    """Путь временного файла БД теста."""
    return str(tmp_path / "test.db")


@pytest.fixture()
def conn(db_path):
    """Соединение с временной БД: connect + init_db; закрывается после теста."""
    connection = connect(db_path)
    init_db(connection)
    yield connection
    connection.close()


def _iso_or_none(value: datetime | None) -> str | None:
    """datetime в ISO-текст; None проходит насквозь (NULL-допустимые колонки времени)."""
    return None if value is None else to_iso(value)


def _stored(row: dict) -> dict:
    """Ожидаемый вид записи в БД: все datetime-значения конвертированы в ISO-текст."""
    return {
        key: _iso_or_none(value) if isinstance(value, datetime) else value
        for key, value in row.items()
    }


def _row_values(row: sqlite3.Row) -> dict:
    """sqlite3.Row в dict по именам колонок (для сравнения с записанным)."""
    return {name: row[name] for name in row.keys()}


# ---------------------------------------------------------------- схема (5.1)


def test_init_creates_contract_schema(tmp_path) -> None:
    """Сценарий «Инициализация пустой БД создаёт схему»: три таблицы, три индекса,
    9 колонок trips."""
    connection = connect(str(tmp_path / "fresh.db"))
    init_db(connection)
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    indices = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )
    }
    assert tables == CONTRACT_TABLES
    assert indices == INDEX_NAMES
    columns = [row[1] for row in connection.execute("PRAGMA table_info(trips)")]
    assert columns == TRIPS_COLUMNS
    connection.close()


def test_repeated_init_preserves_rows(conn) -> None:
    """Сценарий «Повторный init без ошибок»: повторный init_db сохраняет записанные строки."""
    conn.execute(_RAW_TRIP_INSERT)
    init_db(conn)  # повторный вызов не должен падать и не должен пересоздавать таблицы
    assert conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0] == 1


def test_invalid_status_rejected(conn) -> None:
    """Сценарий «Недопустимый статус trips отклоняется»: CHECK violation, строка не записана."""
    bad_insert = _RAW_TRIP_INSERT.replace("'completed'", "'done'")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(bad_insert)
    assert conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0] == 0


def test_connection_pragmas(conn) -> None:
    """Сценарий «Прагмы применены»: foreign_keys=1, journal_mode=wal."""
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_transaction_commit_persists(conn, db_path) -> None:
    """Сценарий «Коммит при успехе»: строка видна тому же и новому («сырому») соединению."""
    with transaction(conn):
        conn.execute(_RAW_TRIP_INSERT)
    assert conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0] == 1
    fresh = sqlite3.connect(db_path)
    try:
        assert fresh.execute("SELECT COUNT(*) FROM trips").fetchone()[0] == 1
    finally:
        fresh.close()


def test_transaction_rollback_on_exception(conn) -> None:
    """Сценарий «Rollback при исключении»: ValueError пробрасывается, откаченных строк нет."""
    with pytest.raises(ValueError, match="boom"):
        with transaction(conn):
            conn.execute(_RAW_TRIP_INSERT)
            raise ValueError("boom")
    assert conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0] == 0


# ------------------------------------------------- roundtrip и время (5.2)


def test_roundtrip_trips(conn) -> None:
    """Roundtrip trips: все поля равны записанным, включая NULL assigned_at; id совпадают."""
    ids = insert_trips(conn, [dict(TRIP)])
    rows = list_trips(conn)
    assert len(rows) == 1
    assert rows[0]["trip_id"] == ids[0]
    assert _row_values(rows[0]) == {**_stored(TRIP), "trip_id": ids[0]}


def test_roundtrip_zone_metrics(conn) -> None:
    """Roundtrip zone_metrics_5min: все поля равны записанным, включая NULL avg_wait_min."""
    insert_zone_metrics(conn, [dict(METRIC)])
    rows = list_zone_metrics(conn)
    assert len(rows) == 1
    assert _row_values(rows[0]) == _stored(METRIC)


def test_roundtrip_forecasts(conn) -> None:
    """Roundtrip forecasts: все поля равны записанным; PK задаёт вызывающий код."""
    insert_forecasts(conn, [dict(FORECAST)])
    rows = list_forecasts(conn)
    assert len(rows) == 1
    assert _row_values(rows[0]) == _stored(FORECAST)


def test_time_fields_stored_as_iso_text(conn, db_path) -> None:
    """Сценарий «Формат хранения проверен напрямую в БД»: второе «сырое» соединение
    читает текст «YYYY-MM-DD HH:MM:SS» с секундами 00; NULL-колонка — NULL."""
    insert_trips(conn, [dict(TRIP)])
    insert_zone_metrics(conn, [dict(METRIC)])
    insert_forecasts(conn, [dict(FORECAST)])

    raw = sqlite3.connect(db_path)  # чтение напрямую, минуя слой хранения
    try:
        created_at, pickup_at, dropoff_at = raw.execute(
            "SELECT created_at, pickup_at, dropoff_at FROM trips"
        ).fetchone()
        metric_ts = [row[0] for row in raw.execute("SELECT ts FROM zone_metrics_5min")]
        forecast_ts = [row[0] for row in raw.execute("SELECT ts FROM forecasts")]
        assigned_at = raw.execute("SELECT assigned_at FROM trips").fetchone()[0]
    finally:
        raw.close()

    for value in (created_at, pickup_at, dropoff_at, *metric_ts, *forecast_ts):
        assert isinstance(value, str)
        from_iso(value)  # строгое соответствие «YYYY-MM-DD HH:MM:SS», иначе ValueError
        assert value.endswith(":00")
    assert assigned_at is None


# ------------------------------------------------------ CSV и демо (5.3)


def test_csv_matches_selection(conn, tmp_path) -> None:
    """Сценарий «CSV содержит те же строки»: заголовок — колонки таблицы, строки и
    NULL-поля (пустое поле на месте assigned_at) совпадают с выборкой."""
    insert_trips(conn, [dict(TRIP)])
    out_path = tmp_path / "trips.csv"
    export_table(conn, "trips", str(out_path))

    parsed = list(csv.reader(out_path.read_text(encoding="utf-8").splitlines()))
    header, data = parsed[0], parsed[1:]
    expected_header = [row[1] for row in conn.execute("PRAGMA table_info(trips)")]
    assert header == expected_header

    selection = [_row_values(row) for row in list_trips(conn)]
    assert len(data) == len(selection)
    for csv_row, stored in zip(data, selection, strict=True):
        assert csv_row == [
            "" if stored[column] is None else str(stored[column]) for column in header
        ]
    assert data[0][header.index("assigned_at")] == ""


def test_demo_script_succeeds() -> None:
    """Сценарий «Демо-скрипт завершается успехом»: код возврата 0, сводка и пути CSV в stderr."""
    result = subprocess.run(
        [sys.executable, "-m", "storage.demo"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Строк записано и прочитано: trips=1, zone_metrics_5min=1, forecasts=1" in result.stderr
    assert "CSV trips:" in result.stderr
