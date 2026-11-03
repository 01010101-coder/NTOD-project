"""Парк водителей симулятора M2: состояния, старт, длительности и счётчики по зонам.

Публичный контракт — приложение Б плана M2 (docs/plan-m2-engine.md): строковый
enum ``DriverState`` (``free``/``to_pickup``/``on_trip``), изменяемый dataclass
``Driver`` и класс ``Fleet``. Старт парка детерминирован без rng — водитель
``i`` в зоне ``i mod Z`` (§10.14, §3.5); разыгрываний в модуле нет вовсе
(§10.9). Длительности подачи и поездки считаются по одной формуле
``ceil(путь × k)``, минимум 1 минута (§3.2, §10.5). Счётчики
``free_by_zone``/``busy_by_zone`` поставляют снапшот под агрегацию §3.8.
"""

import enum
import math
from dataclasses import dataclass

import numpy as np


class DriverState(enum.Enum):
    """Состояние водителя (§3.5): свободен, едет за заказом, везёт пассажира."""

    FREE = "free"
    TO_PICKUP = "to_pickup"
    ON_TRIP = "on_trip"


@dataclass
class Driver:
    """Водитель парка: изменяемая запись приложения Б с ровно пятью полями."""

    id: int
    zone: int
    state: DriverState
    trip_id: int | None  # внутренний id заказа; None у свободного водителя
    remaining_min: int  # до посадки (TO_PICKUP) или до высадки (ON_TRIP)


def travel_minutes(path_min: int, k: float) -> int:
    """Длительность подачи или поездки: ceil(путь × k), минимум 1 минута (§3.2, §10.5).

    Подача считается так же, как поездка (§10.5); минимум 1 минута покрывает и
    подачу в свою зону (``path_min = 0``). ``k`` — множитель времени поездок
    (например, дождевой); вызывается движком при диспатче и назначении.
    """
    return max(1, math.ceil(path_min * k))


class Fleet:
    """Парк водителей: единственный источник движения машин симулятора (§3.5).

    Хранение — плоский список ``Driver`` по возрастанию ``id``; все итерации —
    по возрастанию ``id`` (§3.7). Зона занятого водителя меняется ровно один
    раз — при высадке, на зону назначения заказа; до высадки занятый числится
    в зоне, где был свободен на момент назначения.
    """

    def __init__(self, count: int, zone_count: int) -> None:
        """Детерминированный старт парка без rng (§10.14): водитель i в зоне i mod Z."""
        self._zone_count = zone_count
        self._drivers = [
            Driver(
                id=i,
                zone=i % zone_count,
                state=DriverState.FREE,
                trip_id=None,
                remaining_min=0,
            )
            for i in range(count)
        ]

    def free_by_zone(self) -> np.ndarray:
        """Число свободных водителей по зонам: массив длины Z, dtype int (снапшот §3.8)."""
        counts = np.zeros(self._zone_count, dtype=int)
        for driver in self._drivers:
            if driver.state is DriverState.FREE:
                counts[driver.zone] += 1
        return counts

    def busy_by_zone(self) -> np.ndarray:
        """Число занятых водителей (to_pickup и on_trip) по зонам: массив длины Z, dtype int."""
        counts = np.zeros(self._zone_count, dtype=int)
        for driver in self._drivers:
            if driver.state is not DriverState.FREE:
                counts[driver.zone] += 1
        return counts
