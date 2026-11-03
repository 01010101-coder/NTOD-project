"""Общие функции работы со временем проекта: наивный ``datetime`` ↔ ISO-текст.

Единственная точка преобразования времени (решение плана M1 §7.4): время в
проекте хранится с минутной точностью как текст «YYYY-MM-DD HH:MM:SS»,
секунды всегда 00. Таймзоны не используются (решение §7.2).
"""

from datetime import datetime, timedelta

ISO_FORMAT = "%Y-%m-%d %H:%M:%S"
"""Шаблон strftime/strptime канонического текста времени проекта."""


def to_iso(dt: datetime) -> str:
    """Отформатировать наивный ``datetime`` в текст «YYYY-MM-DD HH:MM:SS».

    Секунды и микросекунды входного значения усекаются до нуля (не округляются):
    инвариант «секунды всегда 00» выполняется даже для «грязного» входа
    (решение design.md 2). Результат годится для колонок времени SQLite-схемы.
    """
    normalized = dt.replace(second=0, microsecond=0)
    return normalized.strftime(ISO_FORMAT)


def from_iso(value: str) -> datetime:
    """Строго разобрать текст «YYYY-MM-DD HH:MM:SS» в наивный ``datetime``.

    Разбор выполняет ``datetime.strptime`` с шаблоном ``"%Y-%m-%d %H:%M:%S"``
    (решение design.md 3). Строка, не соответствующая формату (например,
    с ``T``-разделителем, только дата или произвольный текст), приводит
    к ``ValueError`` с указанием ожидаемого формата. Уступка ``strptime``:
    числа без ведущих нулей (``2026-3-1 9:05:00``) тоже разбираются;
    канонический вид текста всегда даёт :func:`to_iso`.
    """
    try:
        return datetime.strptime(value, ISO_FORMAT)
    except ValueError as exc:
        raise ValueError(
            f"Ожидалось время в формате «YYYY-MM-DD HH:MM:SS», получено {value!r}: {exc}"
        ) from exc


def add_minutes(dt: datetime, n: int) -> datetime:
    """Прибавить ``n`` минут к наивному ``datetime``, сохранив минутную точность.

    Секунды и микросекунды результата усекаются до нуля (решение design.md 4).
    Отрицательные ``n`` разрешены — откат во времени нужен для горизонтов
    прогноза (``ts − horizon``). Границы часа, суток, месяца и года
    обрабатывает арифметика ``datetime``.
    """
    shifted = dt + timedelta(minutes=n)
    return shifted.replace(second=0, microsecond=0)


def bucket5(ts: datetime) -> datetime:
    """Привязать момент ``ts`` к началу содержащего его 5-минутного интервала.

    Метка вычисляется модулем минут: минуты 00–04 дают ``:00``, 05–09 — ``:05``,
    …, 55–59 — ``:55``; секунды и микросекунды метки всегда 00 (решение
    design.md 5). Возвращается наивный ``datetime``; ISO-метка для колонки
    ``zone_metrics_5min.ts`` получается как ``to_iso(bucket5(ts))``. Формула
    без ``timestamp()`` — не вовлекает локальную таймзону.
    """
    return ts.replace(minute=ts.minute - ts.minute % 5, second=0, microsecond=0)
