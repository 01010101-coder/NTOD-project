"""Общая sqlite-подложка прогонов миров (change ``m6-rng-branches``, design решение 3).

Единственное место жизни sqlite-реализации шва ``TripWriter`` (§10.12) поверх
``storage.db``/``storage.repo``, guard-сообщения непустой БД и печатей
прогресса/сводки формата §4 M2. Модуль переиспользуют оба CLI генерации
миров — ``python -m simulator`` и ``python -m simulator.branches`` — без
копирования формата вывода. Логики мира в модуле нет: движок отдаёт
строки-дикты, writer вставляет их в SQLite, функции печатают отчёт прогона.
"""

import sys

from common.timeutil import from_iso, to_iso
from storage.repo import insert_trips, insert_zone_metrics

GUARD_MESSAGE = "генерация требует пустой таблицы trips — укажите новый --db"
_PROGRESS_FORMAT = "[{stamp}] trips={count} elapsed={sec:.1f}s"


class SqliteTripWriter:
    """sqlite-реализация ``TripWriter`` (§10.12) поверх ``storage.repo``.

    Единственный смысл — вставка батчей строк движка в SQLite через
    ``storage.repo.insert_trips``/``insert_zone_metrics`` (транзакции задаёт
    repo). Логики мира нет: движок отдаёт времена строками ISO «YYYY-MM-DD
    HH:MM:SS» (решение 5 design.md change ``m2-loop``), а repo принимает
    ``datetime`` — writer конвертирует ``from_iso``, NULL проходят насквозь.
    """

    def __init__(self, conn) -> None:
        """Сохранить открытое соединение ``storage.db.connect``."""

        self._conn = conn

    def write_trips(self, rows: list[dict]) -> None:
        """Вставить батч строк ``trips`` одной транзакцией."""

        converted = [
            {
                "created_at": from_iso(row["created_at"]),
                "origin_zone": row["origin_zone"],
                "dest_zone": row["dest_zone"],
                "assigned_at": (
                    from_iso(row["assigned_at"]) if row["assigned_at"] is not None else None
                ),
                "pickup_at": from_iso(row["pickup_at"]) if row["pickup_at"] is not None else None,
                "dropoff_at": (
                    from_iso(row["dropoff_at"]) if row["dropoff_at"] is not None else None
                ),
                "status": row["status"],
                "price_multiplier": row["price_multiplier"],
            }
            for row in rows
        ]
        insert_trips(self._conn, converted)

    def write_metrics(self, rows: list[dict]) -> None:
        """Вставить батч строк ``zone_metrics_5min`` одной транзакцией.

        Ключи строк буфера агрегации совпадают с ожиданиями repo (``ts`` —
        наивный ``datetime``), конвертация не нужна.
        """

        insert_zone_metrics(self._conn, rows)


def print_progress(t, trips_so_far: int, elapsed_sec: float) -> None:
    """Callback движка: машинная строка прогресса раз в сим-час в stderr (A4)."""

    print(
        _PROGRESS_FORMAT.format(stamp=to_iso(t)[:16], count=trips_so_far, sec=elapsed_sec),
        file=sys.stderr,
    )


def print_summary(summary, db_path: str) -> None:
    """Сводка прогона: ровно 4 строки формата §4 в stdout (A4)."""

    total = summary.trips_total
    completed_share = 100.0 * summary.completed / total if total else 0.0
    print(f"sim: days={summary.days} seed={summary.seed} db={db_path}")
    print(
        f"trips: total={total} completed={summary.completed} "
        f"completed_share={completed_share:.1f}% unfulfilled={summary.unfulfilled} "
        f"unfulfilled_share={summary.unfulfilled_share:.1f}%"
    )
    print(
        f"wait: avg_wait_min={summary.avg_wait_min:.2f} deficit_share={summary.deficit_share:.1f}%"
    )
    print(f"wall: {summary.wall_sec:.1f}s")
