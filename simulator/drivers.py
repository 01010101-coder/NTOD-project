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
        # Приватный канал сведений о поездке (design.md, решение 6): план
        # (зона назначения, длительность) на водителя, заполняется движком
        # при назначении через _plan_trip; публичные сигнатуры Б и поля
        # Driver не расширяются.
        self._trip_plans: dict[int, tuple[int, int]] = {}

    def nearest_free(self, zone: int, path_row: np.ndarray) -> int | None:
        """Свободный водитель с минимальным временем подачи; при равенстве — меньший id (§3.5).

        Один проход по водителям по возрастанию ``id``; время подачи водителя
        есть ``path_row[driver.zone]`` — строка матрицы кратчайших путей для
        зоны посадки ``zone``. Кандидат лучше только при строгом ``<`` — при
        равенстве остаётся водитель с меньшим ``id``. Свободных нет — ``None``.
        """
        best_id: int | None = None
        best_time = math.inf
        for driver in self._drivers:
            if driver.state is not DriverState.FREE:
                continue
            candidate = int(path_row[driver.zone])
            if candidate < best_time:
                best_id, best_time = driver.id, candidate
        return best_id

    def assign(self, driver_id: int, trip_id: int, pickup_min: int) -> None:
        """Назначить свободному водителю заказ (§3.5): to_pickup, trip_id, remaining_min.

        ``remaining_min = pickup_min`` — время подачи, посчитанное движком по
        ``travel_minutes``; занятый водитель больше не возвращается
        ``nearest_free`` и числится ``busy_by_zone()`` в зоне, где был свободен.
        """
        driver = self._drivers[driver_id]
        driver.state = DriverState.TO_PICKUP
        driver.trip_id = trip_id
        driver.remaining_min = pickup_min

    def _plan_trip(self, driver_id: int, dest: int, trip_min: int) -> None:
        """Сообщить парку план поездки: зона назначения и длительность (приватный канал).

        Заполняется движком сразу после ``assign``: приложение Б не несёт ни
        зону назначения заказа (нужна ``tick()`` для высадки), ни длительность
        поездки (нужна как ``remaining_min`` в ``ON_TRIP``); публичные
        сигнатуры Б и поля ``Driver`` не расширяются (design.md, решение 6).
        """
        self._trip_plans[driver_id] = (dest, trip_min)

    def tick(self) -> list[tuple[int, int]]:
        """Одна сим-минута (§3.7 шаг 4): тики счётчиков, посадки и высадки занятых.

        Один проход по занятым по возрастанию ``id``, ``remaining_min -= 1``;
        исчерпание у ``TO_PICKUP`` — посадка (``ON_TRIP``), у ``ON_TRIP`` —
        высадка: ``FREE``, зона = зона назначения из плана поездки, ``trip_id
        = None``. Переход ровно один за тик (правило «минимум 1 минута»,
        §10.5): у только что пересевшего в машину счётчик >= 1 и в этом же
        тике не уменьшается второй раз. Возвращает высадки минуты — пары
        ``(driver_id, зона_высадки)`` по возрастанию ``driver_id``.
        """
        dropoffs: list[tuple[int, int]] = []
        for driver in self._drivers:
            if driver.state is DriverState.FREE:
                continue
            driver.remaining_min -= 1
            if driver.remaining_min > 0:
                continue
            if driver.state is DriverState.TO_PICKUP:
                # Посадка: длительность поездки берётся из плана, не тикает в этом же вызове.
                driver.state = DriverState.ON_TRIP
                driver.remaining_min = self._trip_plans[driver.id][1]
            else:
                # Высадка: водитель свободен в зоне назначения заказа (§3.5).
                dest = self._trip_plans.pop(driver.id)[0]
                driver.state = DriverState.FREE
                driver.zone = dest
                driver.trip_id = None
                driver.remaining_min = 0
                dropoffs.append((driver.id, dest))
        return dropoffs

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
