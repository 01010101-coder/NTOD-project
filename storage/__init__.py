"""Слой хранения: SQLite-схема, соединения и транзакции, доступ к данным, CSV-экспорт.

Публичный API реэкспортируется для удобного ``from storage import ...``
(решение design.md 10): схема и идемпотентная инициализация — ``schema``,
соединение с прагмами и транзакции — ``db``, вставки/выборки — ``repo``,
выгрузка таблиц в CSV — ``csv_export``, демо-скрипт — ``python -m storage.demo``.
Модули пакета импортируют только ``db``/``schema``, поэтому циклических
импортов нет.
"""

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

__all__ = [
    "connect",
    "export_table",
    "init_db",
    "insert_forecasts",
    "insert_trips",
    "insert_zone_metrics",
    "list_forecasts",
    "list_trips",
    "list_zone_metrics",
    "transaction",
]
