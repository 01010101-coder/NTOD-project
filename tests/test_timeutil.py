"""Тесты common/timeutil.py: преобразования наивный ``datetime`` ↔ ISO-текст.

Покрывают сценарии «Roundtrip ISO», «Секунды и микросекунды отбрасываются»,
«Некорректный текст отклоняется», «Прибавление в пределах часа», «Переход
через час и полночь», «Границы минут внутри часа», «Переход часа» и «Переход
полуночи и суток» из specs change'а m1-common (capability common-utilities).
"""

from datetime import datetime

import pytest

from common.timeutil import add_minutes, bucket5, from_iso, to_iso

# Параметризация roundtrip: обычная дата, последняя и первая минуты суток,
# 1 января и 29 февраля високосного 2028 года.
ROUNDTRIP_CASES = [
    ("2026-03-10 09:07:00", datetime(2026, 3, 10, 9, 7)),
    ("2026-03-10 23:59:00", datetime(2026, 3, 10, 23, 59)),
    ("2026-03-10 00:00:00", datetime(2026, 3, 10, 0, 0)),
    ("2026-01-01 12:05:00", datetime(2026, 1, 1, 12, 5)),
    ("2028-02-29 06:30:00", datetime(2028, 2, 29, 6, 30)),
]

# Строки не формата «YYYY-MM-DD HH:MM:SS», которые from_iso обязан отклонить.
INVALID_TEXTS = [
    "2026-03-10T09:07:00",  # T-разделитель вместо пробела
    "2026-03-10",  # только дата, без времени
    "",  # пустая строка
    "мусор",  # произвольный текст
]

# Границы минут внутри часа: каждая минута 0–59 отображается на начало своего
# 5-минутного бакета (:00–:04 → :00, :05–:09 → :05, …, :55–:59 → :55).
MINUTE_BOUNDARY_CASES = [
    (datetime(2026, 3, 10, 10, m), datetime(2026, 3, 10, 10, m - m % 5)) for m in range(60)
]

# Границы add_minutes: в пределах часа, через час, полночь, месяц и год;
# последний случай — отрицательное n (откат на горизонт прогноза).
ADD_MINUTES_CASES = [
    (datetime(2026, 3, 10, 9, 12), 6, datetime(2026, 3, 10, 9, 18)),
    (datetime(2026, 3, 10, 9, 58), 5, datetime(2026, 3, 10, 10, 3)),
    (datetime(2026, 3, 10, 23, 59), 2, datetime(2026, 3, 11, 0, 1)),
    (datetime(2026, 3, 31, 23, 50), 15, datetime(2026, 4, 1, 0, 5)),
    (datetime(2026, 12, 31, 23, 55), 10, datetime(2027, 1, 1, 0, 5)),
    (datetime(2026, 3, 10, 10, 0), -15, datetime(2026, 3, 10, 9, 45)),
]


@pytest.mark.parametrize(("iso_text", "expected_dt"), ROUNDTRIP_CASES)
def test_roundtrip_iso(iso_text: str, expected_dt: datetime) -> None:
    """Сценарий «Roundtrip ISO»: текст ↔ datetime взаимно обратимы."""
    assert to_iso(from_iso(iso_text)) == iso_text
    assert from_iso(to_iso(expected_dt)) == expected_dt


def test_roundtrip_truncates_seconds() -> None:
    """Roundtrip от «грязного» datetime возвращает его с усечёнными секундами."""
    dt = datetime(2026, 3, 10, 9, 7, 31, 500_000)
    assert to_iso(dt) == "2026-03-10 09:07:00"
    assert from_iso(to_iso(dt)) == datetime(2026, 3, 10, 9, 7)


def test_seconds_and_microseconds_dropped() -> None:
    """Сценарий «Секунды и микросекунды отбрасываются»: секунды 00, без микросекунд."""
    dt = datetime(2026, 3, 10, 10, 7, 31, 500_000)
    text = to_iso(dt)
    assert text == "2026-03-10 10:07:00"
    parsed = from_iso(text)
    assert (parsed.second, parsed.microsecond) == (0, 0)


@pytest.mark.parametrize("bad_text", INVALID_TEXTS)
def test_invalid_text_rejected(bad_text: str) -> None:
    """Сценарий «Некорректный текст отклоняется»: ValueError называет ожидаемый формат."""
    with pytest.raises(ValueError, match="YYYY-MM-DD HH:MM:SS"):
        from_iso(bad_text)


@pytest.mark.parametrize(("dt", "n", "expected"), ADD_MINUTES_CASES)
def test_add_minutes_crosses_boundaries(dt: datetime, n: int, expected: datetime) -> None:
    """Сценарии «Прибавление в пределах часа» и «Переход через час и полночь»:
    границы часа, суток, месяца и года, отрицательные n."""
    result = add_minutes(dt, n)
    assert result == expected
    assert (result.second, result.microsecond) == (0, 0)


def test_add_minutes_truncates_result_seconds() -> None:
    """Секунды и микросекунды результата усекаются до нуля (решение design.md 4)."""
    result = add_minutes(datetime(2026, 3, 10, 9, 12, 30, 250_000), 6)
    assert result == datetime(2026, 3, 10, 9, 18)
    assert (result.second, result.microsecond) == (0, 0)


@pytest.mark.parametrize(("dt", "expected"), MINUTE_BOUNDARY_CASES)
def test_bucket5_minute_boundaries(dt: datetime, expected: datetime) -> None:
    """Сценарий «Границы минут внутри часа»: :00–:04 → :00, :05–:09 → :05, …, :55–:59 → :55."""
    assert bucket5(dt) == expected


@pytest.mark.parametrize(
    ("dt", "expected"),
    [
        (datetime(2026, 3, 10, 10, 56), datetime(2026, 3, 10, 10, 55)),
        (datetime(2026, 3, 10, 10, 59), datetime(2026, 3, 10, 10, 55)),
        (datetime(2026, 3, 10, 11, 0), datetime(2026, 3, 10, 11, 0)),
        (datetime(2026, 3, 10, 11, 1), datetime(2026, 3, 10, 11, 0)),
    ],
)
def test_bucket5_hour_crossing(dt: datetime, expected: datetime) -> None:
    """Сценарий «Переход часа»: 10:56, 10:59 → 10:55; 11:00, 11:01 → 11:00."""
    assert bucket5(dt) == expected


def test_bucket5_midnight_and_day_crossing() -> None:
    """Сценарий «Переход полуночи и суток»: 23:58:31 → 23:55 того же дня; 00:03 → 00:00 своего."""
    assert bucket5(datetime(2026, 3, 10, 23, 58, 31)) == datetime(2026, 3, 10, 23, 55)
    assert bucket5(datetime(2026, 3, 11, 0, 3)) == datetime(2026, 3, 11, 0, 0)


def test_bucket5_label_is_valid_iso_text() -> None:
    """ISO-метка бакета через to_iso валидна: разбирается from_iso, секунды 00."""
    dt = datetime(2026, 3, 10, 10, 7, 31, 500_000)
    bucket = bucket5(dt)
    text = to_iso(bucket)
    assert text == "2026-03-10 10:05:00"
    parsed = from_iso(text)
    assert parsed == bucket
    assert (parsed.second, parsed.microsecond) == (0, 0)
