"""Типы событийного потока наблюдения change ``live-engine`` (спека simulator-live).

Иммутабельные записи минуты мира для шва ``on_minute`` у ``Engine.run``:
``MinuteRecord`` собирает среду минуты (дождь, фактический тариф по зонам,
снапшот свободного парка) и события минуты, сгруппированные по типу. События
самодостаточны для клиента анимации — механика мира не дублируется:
назначение несёт зону водителя в момент назначения (в ``trips`` её нет —
главный пробел, который закрывает поток) и моменты ``pickup_at``/``dropoff_at``,
вычисленные движком в момент назначения; сквозная нить событий — внутренний
id заказа (порядок создания, совпадает с ``trip_id`` вставки, §10.11).
Все типы — frozen dataclass с кортежами: запись минуты неизменяема, её
безопасно хранить в событийном логе и передавать колбэку (design.md, решение 5).
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SpawnEvent:
    """Создание заказа в минуту записи (шаг 2 цикла §3.7).

    ``internal_id`` — внутренний id заказа движка; ``origin``/``dest`` — зоны
    происхождения и назначения; ``price_multiplier`` — множитель тарифа зоны
    происхождения на момент создания заказа.
    """

    internal_id: int
    origin: int
    dest: int
    price_multiplier: float


@dataclass(frozen=True)
class AssignEvent:
    """Назначение заказа водителю в минуту записи (шаг 3 цикла §3.7).

    ``driver_zone`` — зона водителя в момент назначения (до исключения из
    зеркала свободных). ``pickup_at``/``dropoff_at`` — сим-времена, известные
    в момент назначения: посадка — минута ``t + pickup_min − 1``, высадка —
    ``pickup_at + trip_min`` (инвариант механики ``Fleet.tick``); клиент
    планирует обе фазы движения сразу, не повторяя формулы тика.
    """

    internal_id: int
    driver_id: int
    driver_zone: int
    origin: int
    dest: int
    pickup_at: datetime
    dropoff_at: datetime


@dataclass(frozen=True)
class DropoffEvent:
    """Высадка в минуту записи (шаг 4 цикла §3.7): заказ довезён.

    ``driver_id`` — освободившийся водитель; ``dest_zone`` — зона высадки, в
    которой водитель свободен с этой минуты.
    """

    internal_id: int
    driver_id: int
    dest_zone: int


@dataclass(frozen=True)
class ExpiryEvent:
    """Истечение неназначенного заказа в минуту записи (шаг 5 цикла §3.7)."""

    internal_id: int
    origin: int


@dataclass(frozen=True)
class CancelEvent:
    """Отказ пассажира от долгой подачи в минуту записи (шаг 3 цикла §3.7).

    ``pickup_min`` — подача отвергнутого водителя: заказ терминален, водитель
    остаётся свободным. События идут в порядке обхода очереди диспатча
    (change ``world-dispatch-radius``).
    """

    internal_id: int
    origin: int
    pickup_min: int


@dataclass(frozen=True)
class MinuteRecord:
    """Запись одной сим-минуты: среда на минуту и события минуты.

    ``t`` — сим-минута записи (наивный datetime, секунды 00); ``is_rain`` —
    дождь в минуту ``t``; ``price_mult`` — фактический множитель тарифа каждой
    зоны на минуту ``t`` кортежем по возрастанию ``zone_id`` (при шве
    ``price_fn`` — его значение, иначе тариф среды); ``free_by_zone`` —
    снапшот числа свободных водителей по зонам на конец минуты кортежем по
    возрастанию ``zone_id``. События минуты упорядочены по порядку обработки
    цикла; в записях дренажа хвоста ``spawns``/``assignments`` пусты.
    ``cancellations`` — отказы пассажиров (change ``world-dispatch-radius``),
    поле добавлено последним с дефолтом: существующие конструкторы не ломаются.
    ``online_drivers`` — число водителей на смене (в любом состоянии, кроме
    ``offline``, включая помеченных к уходу) на конец минуты (change
    ``world-driver-shifts``); поле тоже добавлено последним с дефолтом.
    ``part_online`` — число подработчиков на линии на конец минуты (change
    ``world-elastic-supply``); поле добавлено последним с дефолтом 0, при
    выключенном вызове оно равно 0.
    """

    t: datetime
    is_rain: bool
    price_mult: tuple[float, ...]
    free_by_zone: tuple[int, ...]
    spawns: tuple[SpawnEvent, ...]
    assignments: tuple[AssignEvent, ...]
    dropoffs: tuple[DropoffEvent, ...]
    expiries: tuple[ExpiryEvent, ...]
    cancellations: tuple[CancelEvent, ...] = ()
    online_drivers: int = 0
    part_online: int = 0
