"""Тесты парка водителей change'а m2-drivers (строка №3 §7 плана M2).

Покрывают сценарии спеки simulator-drivers: «Публичный API», «Детерминированный
старт парка», «Диспатч — ближайший свободный водитель с тай-брейком по меньшему
id», «Длительности — ceil с минимумом 1 минута», «Назначение и движение —
машина состояний с высадкой в зоне назначения», «Счётчики свободных и занятых
по зонам». Проверки строки №3 §7: ближайший водитель получает заказ; при
равенстве — меньший id; после dropoff водитель free в dest; длительность =
``ceil(путь × k)``; старт парка: водитель i в зоне i mod Z. Тесты
детерминированы, rng не используется; эталонная матрица path_min берётся из
``build_city`` закрытого change'а m2-city, синтетические строки path_row —
фиксированные ``np.array``.
"""

from dataclasses import fields
from pathlib import Path

import numpy as np

from common.config import load_config
from simulator.city import build_city
from simulator.drivers import Driver, DriverState, Fleet, travel_minutes

# Эталонный конфиг ищется относительно файла теста, не от cwd (как в M1-тестах
# и в тестах change'а m2-city).
REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"

# Эталонная матрица кратчайших путей (change m2-city): диспатч по эталонному
# графу и его симметричным зонам; это данные тестов, не магические числа мира.
REFERENCE_PATH_MIN = build_city(load_config(REFERENCE_CONFIG)).path_min


# --- Публичный API (спека «Публичный API», задача 5.1) ---


def test_api_public_names_and_enum_values() -> None:
    """Импорт публичных имён приложения Б и строковые значения enum."""

    def _importable() -> tuple[type, type, type]:
        return Driver, DriverState, Fleet

    assert _importable() == (Driver, DriverState, Fleet)
    assert DriverState.FREE.value == "free"
    assert DriverState.TO_PICKUP.value == "to_pickup"
    assert DriverState.ON_TRIP.value == "on_trip"


def test_api_driver_has_exactly_five_fields() -> None:
    """У ``Driver`` ровно пять полей приложения Б, все изменяемы (dataclass не заморожен)."""
    assert [f.name for f in fields(Driver)] == [
        "id",
        "zone",
        "state",
        "trip_id",
        "remaining_min",
    ]
    driver = Driver(id=0, zone=1, state=DriverState.FREE, trip_id=None, remaining_min=0)
    driver.zone = 2
    driver.trip_id = 7
    driver.remaining_min = 3
    assert (driver.zone, driver.trip_id, driver.remaining_min) == (2, 7, 3)


def test_api_deterministic_no_rng() -> None:
    """«Отсутствие случайности в модуле»: без seed одинаковые вызовы — одинаковый результат."""

    def run() -> tuple[list[int], list[int], list[list[tuple[int, int]]], list[tuple]]:
        fleet = Fleet(200, 6)
        fleet.assign(0, 7, 2)
        fleet._plan_trip(0, 4, 3)
        fleet.assign(1, 8, 1)
        fleet._plan_trip(1, 2, 2)
        drops = [fleet.tick() for _ in range(6)]
        snapshot = [(d.id, d.zone, d.state, d.trip_id, d.remaining_min) for d in fleet._drivers]
        return fleet.free_by_zone().tolist(), fleet.busy_by_zone().tolist(), drops, snapshot

    assert run() == run()


# --- Детерминированный старт парка (спека «Детерминированный старт парка», задача 5.1) ---


def test_start_layout_reference_fleet() -> None:
    """Раскладка эталонного парка: водитель 0/5/6/199 в зонах 0/5/0/1, все free."""
    fleet = Fleet(200, 6)
    assert [fleet._drivers[i].zone for i in (0, 5, 6, 199)] == [0, 5, 0, 1]
    for driver in fleet._drivers:
        assert driver.state is DriverState.FREE
        assert driver.trip_id is None
        assert driver.remaining_min == 0


def test_start_fewer_drivers_than_zones() -> None:
    """Парк меньше числа зон: водители 0, 1, 2 — в зонах 0, 1, 2."""
    fleet = Fleet(3, 6)
    assert [driver.zone for driver in fleet._drivers] == [0, 1, 2]


def test_start_free_counters() -> None:
    """Счётчики свободных на старте: [34, 34, 33, 33, 33, 33], занятых нет, сумма 200."""
    fleet = Fleet(200, 6)
    free, busy = fleet.free_by_zone(), fleet.busy_by_zone()
    assert free.tolist() == [34, 34, 33, 33, 33, 33]
    assert busy.tolist() == [0] * 6
    assert int(free.sum()) == 200


# --- Диспатч (спека «Диспатч», задача 5.2) ---


def test_dispatch_synthetic_nearest_wins() -> None:
    """Ближайший получает заказ на синтетической path_row, даже если его id больше."""
    fleet = Fleet(2, 6)  # водители 0 и 1 в зонах 0 и 1 (старт i mod Z)
    path_row = np.array([5, 2, 9, 9, 9, 9])  # до зоны посадки: из зоны 1 быстрее (2 < 5)
    assert fleet.nearest_free(4, path_row) == 1


def test_dispatch_reference_matrix_nearest_wins() -> None:
    """Ближайший получает заказ на эталонной матрице build_city(эталонный config.yaml)."""
    fleet = Fleet(2, 6)  # зоны 0 и 1
    assert fleet.nearest_free(4, REFERENCE_PATH_MIN[4]) == 1  # 14 < 25
    assert fleet.nearest_free(0, REFERENCE_PATH_MIN[0]) == 0  # 0 < 18


def test_dispatch_tie_same_zone_smaller_id() -> None:
    """Равенство времён подачи (два водителя в одной зоне) — меньший id."""
    fleet = Fleet(7, 6)  # водители 0 и 6 оба в зоне 0 (старт i mod Z)
    path_row = np.array([3, 9, 9, 9, 9, 9])
    assert fleet.nearest_free(1, path_row) == 0


def test_dispatch_tie_reference_symmetric_zones() -> None:
    """Симметричные зоны эталонного графа: равное время подачи — меньший id."""
    fleet = Fleet(2, 6)  # зоны 0 и 1
    assert REFERENCE_PATH_MIN[2][0] == REFERENCE_PATH_MIN[2][1] == 9
    assert fleet.nearest_free(2, REFERENCE_PATH_MIN[2]) == 0
    assert REFERENCE_PATH_MIN[3][0] == REFERENCE_PATH_MIN[3][1] == 15
    assert fleet.nearest_free(3, REFERENCE_PATH_MIN[3]) == 0


def test_dispatch_no_free_returns_none() -> None:
    """Нет свободных водителей — None; пустой парк — тоже None."""
    fleet = Fleet(2, 6)
    fleet.assign(0, 1, 2)
    fleet.assign(1, 2, 2)
    assert fleet.nearest_free(0, np.array([1, 1, 1, 1, 1, 1])) is None
    assert Fleet(0, 6).nearest_free(0, np.array([1, 1, 1, 1, 1, 1])) is None


def test_dispatch_busy_not_considered() -> None:
    """Занятые (to_pickup и on_trip) не рассматриваются диспатчем."""
    fleet = Fleet(3, 6)  # водители 0, 1, 2 в зонах 0, 1, 2
    fleet.assign(0, 1, 1)  # подача 1 минута: посадка в первом же тике
    fleet._plan_trip(0, 4, 3)
    path_row = np.array([0, 5, 6, 9, 9, 9])  # ближе всех занятый водитель 0 (0 минут)
    assert fleet._drivers[0].state is DriverState.TO_PICKUP
    assert fleet.nearest_free(0, path_row) == 1
    fleet.tick()  # посадка водителя 0 — теперь он on_trip
    assert fleet._drivers[0].state is DriverState.ON_TRIP
    assert fleet.nearest_free(0, path_row) == 1


# --- Назначение, движение и счётчики (спеки «Назначение и движение», «Счётчики», задача 5.3) ---


def test_tick_assign_moves_to_pickup() -> None:
    """Назначение переводит водителя в to_pickup с remaining = pickup_min."""
    fleet = Fleet(2, 6)
    fleet.assign(0, 7, 3)
    driver = fleet._drivers[0]
    assert driver.state is DriverState.TO_PICKUP
    assert driver.trip_id == 7
    assert driver.remaining_min == 3
    # занятый учитывается busy_by_zone() в зоне, где был свободен, и не диспатчится
    assert fleet.busy_by_zone().tolist() == [1, 0, 0, 0, 0, 0]
    assert fleet.free_by_zone().tolist() == [0, 1, 0, 0, 0, 0]
    assert fleet.nearest_free(0, np.array([1, 1, 1, 1, 1, 1])) == 1


def test_cycle_full_from_assign_to_dropoff() -> None:
    """Полный цикл: подача 2, поездка 3 — пятый тик возвращает (id, dest), водитель free в dest."""
    fleet = Fleet(2, 6)
    fleet.assign(0, 7, 2)
    fleet._plan_trip(0, 4, 3)
    assert fleet.tick() == []
    assert fleet._drivers[0].state is DriverState.TO_PICKUP
    assert fleet.tick() == []
    assert fleet._drivers[0].state is DriverState.ON_TRIP  # посадка после двух минут подачи
    assert fleet.tick() == []
    assert fleet.tick() == []
    assert fleet.tick() == [(0, 4)]
    driver = fleet._drivers[0]
    assert driver.state is DriverState.FREE
    assert driver.zone == 4  # свободен в зоне назначения заказа
    assert driver.trip_id is None


def test_tick_dropoffs_of_one_minute_ascending_id() -> None:
    """Высадки одной минуты идут по возрастанию id, независимо от порядка назначения."""
    fleet = Fleet(4, 6)  # водители 0..3 в зонах 0..3
    fleet.assign(2, 1, 1)
    fleet._plan_trip(2, 3, 1)
    fleet.assign(0, 2, 1)
    fleet._plan_trip(0, 5, 1)
    fleet.assign(1, 3, 1)
    fleet._plan_trip(1, 2, 1)
    assert fleet.tick() == []  # первый тик — только посадки, без высадок
    assert fleet.tick() == [(0, 5), (1, 2), (2, 3)]


def test_tick_transitions_not_chained() -> None:
    """Переходы не сцепляются: после посадки счётчик не тикает второй раз в том же вызове."""
    fleet = Fleet(1, 6)
    fleet.assign(0, 1, 1)
    fleet._plan_trip(0, 4, 3)
    assert fleet.tick() == []  # посадка: to_pickup → on_trip
    driver = fleet._drivers[0]
    assert driver.state is DriverState.ON_TRIP
    assert driver.remaining_min == 3  # длительность поездки не уменьшилась в этом же тике
    assert fleet.tick() == []
    assert fleet.tick() == []
    assert fleet.tick() == [(0, 4)]  # высадка ровно через trip_min тиков после посадки


def test_zones_counter_snapshots_along_cycle() -> None:
    """Снапшоты счётчиков вдоль цикла: занятый в зоне до назначения, после высадки — в dest."""
    fleet = Fleet(200, 6)
    assert fleet.free_by_zone().tolist() == [34, 34, 33, 33, 33, 33]
    assert fleet.busy_by_zone().tolist() == [0] * 6
    fleet.assign(0, 1, 2)
    fleet._plan_trip(0, 4, 3)
    assert fleet.free_by_zone().tolist() == [33, 34, 33, 33, 33, 33]
    assert fleet.busy_by_zone().tolist() == [1, 0, 0, 0, 0, 0]
    for _ in range(5):
        fleet.tick()
    assert fleet.free_by_zone().tolist() == [33, 34, 33, 33, 34, 33]
    assert fleet.busy_by_zone().tolist() == [0] * 6


def test_counters_sum_constant_along_cycle() -> None:
    """Сумма free + busy равна размеру парка вдоль произвольной последовательности."""
    fleet = Fleet(200, 6)
    fleet.assign(0, 1, 2)
    fleet._plan_trip(0, 4, 3)
    fleet.assign(7, 2, 1)
    fleet._plan_trip(7, 0, 2)
    for _ in range(8):
        fleet.tick()
        assert int(fleet.free_by_zone().sum() + fleet.busy_by_zone().sum()) == 200
    fleet.assign(3, 3, 1)
    fleet._plan_trip(3, 5, 4)
    for _ in range(6):
        fleet.tick()
        assert int(fleet.free_by_zone().sum() + fleet.busy_by_zone().sum()) == 200


# --- Длительности (спека «Длительности», задача 5.4) ---


def test_travel_minutes_ceil_with_rain_multiplier() -> None:
    """Округление вверх: ceil(9 × 1.4) = 13 — эталонный travel_mult дождя."""
    assert travel_minutes(9, 1.4) == 13


def test_travel_minutes_whole_product_not_rounded() -> None:
    """Целое произведение не округляется."""
    assert travel_minutes(9, 1.0) == 9
    assert travel_minutes(15, 1.0) == 15


def test_travel_minutes_minimum_one_minute() -> None:
    """Минимум 1 минута: путь 0 (водитель уже в зоне посадки) → 1 при любом множителе."""
    assert travel_minutes(0, 1.4) == 1
    assert travel_minutes(0, 1.0) == 1
