"""Тесты цели числа водителей на смене (change ``world-driver-shifts``, задача 2.1).

Покрывают спеку simulator-loop: правило дня ``weekday() < 5``, линейную
интерполяцию доли внутри часа и округление ``floor(count × s + 0.5)``.
Функция чистая — движок и парк не нужны.
"""

from datetime import datetime

from common.config import AppConfig
from simulator.shifts import online_target

# 2026-11-02 — понедельник (sim_start эталона), 2026-11-07 — суббота.
MONDAY = datetime(2026, 11, 2)
SATURDAY = datetime(2026, 11, 7)


def drivers_cfg(count: int = 300):
    """Секция ``drivers`` с дефолтными профилями смен и заданным размером парка."""

    return AppConfig.model_validate(
        {"drivers": {"count": count, "shifts": {"enabled": True}}}
    ).drivers


def test_night_target_is_third_of_fleet() -> None:
    """03:00 будней: доля 0.33 в 03:00 и 04:00 — 99 из 300."""

    assert online_target(drivers_cfg(300), MONDAY.replace(hour=3)) == 99


def test_mid_hour_is_mean_of_neighbours() -> None:
    """Середина часа — среднее соседних долей: 03:30 при 0.33/0.33."""

    assert online_target(drivers_cfg(300), MONDAY.replace(hour=3, minute=30)) == 99
    # 07:30: (0.70 + 0.80) / 2 = 0.75; 1000 × 0.75 = 750.
    assert online_target(drivers_cfg(1000), MONDAY.replace(hour=7, minute=30)) == 750


def test_late_hour_interpolates_to_next_day_start() -> None:
    """23:30 интерполирует к 00:00 того же профиля: (0.45 + 0.40) / 2."""

    # 300 × 0.425 = 127.5 → floor(127.5 + 0.5) = 128.
    assert online_target(drivers_cfg(300), MONDAY.replace(hour=23, minute=30)) == 128


def test_saturday_uses_weekend_profile() -> None:
    """Суббота берёт профиль выходных: 03:00 — доля 0.35, а не 0.33 будней."""

    assert online_target(drivers_cfg(300), SATURDAY.replace(hour=3)) == 105
    assert online_target(drivers_cfg(300), MONDAY.replace(hour=3)) == 99


def test_rounding_is_floor_plus_half() -> None:
    """Округление ``floor(count × s + 0.5)``: 10 × 0.35 = 3.5 → 4, 10 × 0.34 → 3."""

    assert online_target(drivers_cfg(10), MONDAY.replace(hour=1)) == 4  # 0.35
    assert online_target(drivers_cfg(10), MONDAY.replace(hour=2)) == 3  # 0.33
