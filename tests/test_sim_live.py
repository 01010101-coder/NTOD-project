"""Тесты шва наблюдения change ``live-engine`` (спека simulator-live).

``test_live_types_frozen`` — типы потока конструируются и неизменяемы (задача
1.1). ``test_first_record_matches_world_start`` — первая запись короткого
прогона: t = sim_start 00:00:00, векторы длины Z, события минуты 0 согласованы
со стартом парка ``i mod Z`` (задача 2.1). ``test_drain_records`` — минуты за
границей основной фазы излучают записи дренажа только с высадками/истечениями
до терминального состояния всех заказов (задача 2.2). ``test_observer_purity``
— краеугольный инвариант: прогон с наблюдателем и без на одном seed даёт
побайтово одинаковые ``trips``/``zone_metrics_5min`` и равные ``Summary``,
два прогона с наблюдателем — равные последовательности записей (задача 3.1).
``test_events_match_trips`` сверяет поток с таблицей ``trips`` по цепочкам
spawn → assignment → dropoff/expiry (задача 3.2); ``test_free_park_invariant``
— сумма ``free_by_zone`` = парк минус активные поездки, восстановленные из
потока (задача 3.3); ``test_minute_rhythm`` — ритм записей по минутам
(задача 3.4). Прогоны пишут в БД в памяти через sqlite-writer из
``__main__.py`` — тот же путь записи, что и у CLI.
"""

import dataclasses
from collections import Counter
from datetime import datetime

import pytest

from common.config import AppConfig
from common.timeutil import add_minutes, from_iso, to_iso
from simulator.__main__ import SqliteTripWriter
from simulator.engine import Engine
from simulator.live import (
    AssignEvent,
    DropoffEvent,
    ExpiryEvent,
    MinuteRecord,
    SpawnEvent,
)
from storage.db import connect
from storage.schema import init_db

SIM_START = "2026-11-02"
SIM_START_DT = from_iso(f"{SIM_START} 00:00:00")

# Две синтетические зоны с одной дорогой 30 мин — как в тестах цикла
# (tests/test_sim_loop.py): файлы тестов самостоятельны, поэтому мир копируется.
MICRO_ZONES = [
    {
        "id": 0,
        "name": "a",
        "archetype": "residential",
        "attract_origin": 1.0,
        "attract_dest": 1.0,
        "map": {"x": 0.0, "y": 0.0},
    },
    {
        "id": 1,
        "name": "b",
        "archetype": "business",
        "attract_origin": 1.0,
        "attract_dest": 1.0,
        "map": {"x": 10.0, "y": 0.0},
    },
]


def micro_config(**overrides) -> AppConfig:
    """Синтетический мир: 2 зоны, дорога 30 мин, без случайных событий.

    ``scale=0`` отключает спавн (λ = 0), шансы дождя и инъекции нулевые —
    мир полностью рукописен; спавн включается секцией ``population``.
    """

    base = {
        "seed": 7,
        "city": {
            "zones": MICRO_ZONES,
            "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 30.0}],
        },
        "population": {"scale": 0.0},
        "drivers": {"count": 1, "wait_limit_min": 6},
        "weather": {
            "rain_day_chance": 0.0,
            "episode_start_h": [0, 0],
            "episode_duration_h": [1, 1],
        },
        "pricing": {"surge_injection": {"day_chance": 0.0}},
    }
    base.update(overrides)
    return AppConfig.model_validate(base)


def run_with_records(cfg: AppConfig, days: int) -> tuple[object, list[MinuteRecord], object]:
    """Прогон с наблюдателем: строки — в БД в памяти, записи — в список."""

    conn = connect(":memory:")
    init_db(conn)
    records: list[MinuteRecord] = []
    summary = Engine(cfg).run(days, SqliteTripWriter(conn), on_minute=records.append)
    return summary, records, conn


def test_live_types_frozen():
    """Типы потока конструируются; присвоивание поля frozen dataclass падает."""

    t0 = from_iso("2026-11-02 00:00:00")
    spawn = SpawnEvent(internal_id=1, origin=0, dest=1, price_multiplier=1.5)
    assignment = AssignEvent(
        internal_id=1,
        driver_id=3,
        driver_zone=0,
        origin=0,
        dest=1,
        pickup_at=add_minutes(t0, 2),
        dropoff_at=add_minutes(t0, 7),
    )
    dropoff = DropoffEvent(internal_id=1, driver_id=3, dest_zone=1)
    expiry = ExpiryEvent(internal_id=2, origin=0)
    record = MinuteRecord(
        t=t0,
        is_rain=True,
        price_mult=(1.0, 1.5),
        free_by_zone=(1, 0),
        spawns=(spawn,),
        assignments=(assignment,),
        dropoffs=(dropoff,),
        expiries=(expiry,),
    )
    assert record.t == t0 and record.is_rain is True
    assert record.assignments[0].pickup_at == add_minutes(t0, 2)
    assert record.dropoffs[0].dest_zone == 1
    assert record.expiries == (expiry,)
    for obj, field, value in (
        (spawn, "origin", 1),
        (assignment, "driver_zone", 1),
        (dropoff, "dest_zone", 0),
        (expiry, "origin", 1),
        (record, "is_rain", False),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, field, value)


def test_first_record_matches_world_start():
    """Первая запись: t = sim_start 00:00:00, векторы длины Z, старт парка согласован."""

    # Мир без спавна: событий нет, свободный парк — стартовая раскладка
    # ``i mod Z`` (4 водителя, 2 зоны → по 2), тариф 1.0, дождя нет.
    _summary, records, _conn = run_with_records(
        micro_config(drivers={"count": 4, "wait_limit_min": 6}), 1 / 1440
    )
    assert len(records) == 1
    first = records[0]
    assert first.t == SIM_START_DT
    assert first.is_rain is False
    assert first.price_mult == (1.0, 1.0)
    assert first.free_by_zone == (2, 2)
    assert first.spawns == ()
    assert first.assignments == ()
    assert first.dropoffs == ()
    assert first.expiries == ()

    # Мир со спавном: зоны водителей первых назначений согласованы со стартом
    # парка ``i mod Z`` — пока в зоне происхождения есть свободные, ближайший
    # водитель из неё (путь 0 < 30), иначе из другой свободной зоны; снапшот
    # на конец минуты = старт минус назначения минуты (30-минутная дорога
    # исключает высадки в ту же минуту, лимит ожидания — истечения).
    busy_cfg = micro_config(population={"scale": 50.0}, drivers={"count": 4, "wait_limit_min": 6})
    _summary, records, _conn = run_with_records(busy_cfg, 1 / 1440)
    first = records[0]
    assert first.t == SIM_START_DT
    assert len(first.price_mult) == len(first.free_by_zone) == 2
    free = [2, 2]  # водители 0, 2 в зоне 0; водители 1, 3 в зоне 1
    for assignment in first.assignments:
        assert 0 <= assignment.driver_zone < 2
        if free[assignment.origin] > 0:
            expected_zone = assignment.origin
        else:
            expected_zone = 1 - assignment.origin
        assert assignment.driver_zone == expected_zone
        free[assignment.driver_zone] -= 1
    assert list(first.free_by_zone) == free


def test_drain_records():
    """Записи дренажа продолжаются за границей суток до терминального состояния."""

    cfg = micro_config(population={"scale": 50.0}, drivers={"count": 1, "wait_limit_min": 6})
    summary, records, _conn = run_with_records(cfg, 1)
    boundary = add_minutes(SIM_START_DT, 1440)
    assert len(records) > 1440, "хвост активен, записи дренажа отсутствуют"
    assert records[1439].t == add_minutes(SIM_START_DT, 1439)
    assert records[1440].t == boundary  # первая запись дренажа — минута после границы
    drain = records[1440:]
    for record in drain:
        assert record.spawns == () and record.assignments == ()

    # Активные на границе заказы (ожидающие и в пути) закрываются ровно
    # событиями дренажа; последняя запись — минута закрытия последнего заказа.
    active_at_boundary = set()
    main_dropoffs = 0
    main_expiries = 0
    for record in records[:1440]:
        active_at_boundary.update(spawn.internal_id for spawn in record.spawns)
        active_at_boundary.difference_update(drop.internal_id for drop in record.dropoffs)
        active_at_boundary.difference_update(exp.internal_id for exp in record.expiries)
        main_dropoffs += len(record.dropoffs)
        main_expiries += len(record.expiries)
    assert active_at_boundary, "хвост пуст — сценарий дренажа не проверен"
    terminal_in_drain = {
        event.internal_id for record in drain for event in (*record.dropoffs, *record.expiries)
    }
    assert active_at_boundary == terminal_in_drain
    assert sum(len(record.dropoffs) for record in drain) == summary.completed - main_dropoffs
    assert sum(len(record.expiries) for record in drain) == summary.unfulfilled - main_expiries
    assert drain[-1].dropoffs or drain[-1].expiries


def test_observer_purity():
    """Наблюдатель — чистое чтение: таблицы и Summary не меняются, записи детерминированы."""

    cfg = micro_config(population={"scale": 50.0}, drivers={"count": 3, "wait_limit_min": 6})

    def run(*, collect: bool):
        conn = connect(":memory:")
        init_db(conn)
        records: list[MinuteRecord] = []
        summary = Engine(cfg).run(
            1, SqliteTripWriter(conn), on_minute=(records.append if collect else None)
        )
        return summary, records, conn

    summary_plain, _, conn_plain = run(collect=False)
    summary_observed, records_a, conn_observed = run(collect=True)
    summary_again, records_b, _conn_again = run(collect=True)

    # wall_sec — wall-замер прогона, единственное законно различающееся поле.
    summary_plain = dataclasses.replace(summary_plain, wall_sec=0.0)
    summary_observed = dataclasses.replace(summary_observed, wall_sec=0.0)
    summary_again = dataclasses.replace(summary_again, wall_sec=0.0)
    assert summary_plain == summary_observed == summary_again

    trips_plain = [tuple(row) for row in conn_plain.execute("SELECT * FROM trips ORDER BY trip_id")]
    trips_observed = [
        tuple(row) for row in conn_observed.execute("SELECT * FROM trips ORDER BY trip_id")
    ]
    assert trips_plain and trips_plain == trips_observed

    metrics_plain = [
        tuple(row)
        for row in conn_plain.execute("SELECT * FROM zone_metrics_5min ORDER BY ts, zone_id")
    ]
    metrics_observed = [
        tuple(row)
        for row in conn_observed.execute("SELECT * FROM zone_metrics_5min ORDER BY ts, zone_id")
    ]
    assert len(metrics_plain) == 288 * 2  # сутки × 288 окон × 2 зоны
    assert metrics_plain == metrics_observed

    assert records_a and records_a == records_b


def test_events_match_trips():
    """Каждой строке trips соответствует цепочка событий с согласованными временами и зонами."""

    cfg = micro_config(population={"scale": 50.0}, drivers={"count": 3, "wait_limit_min": 6})
    summary, records, conn = run_with_records(cfg, 1)
    assert summary.trips_total > 0

    spawns: dict[int, tuple[datetime, SpawnEvent]] = {}
    assignments: dict[int, tuple[datetime, AssignEvent]] = {}
    dropoffs: dict[int, tuple[datetime, DropoffEvent]] = {}
    expiries: dict[int, tuple[datetime, ExpiryEvent]] = {}
    for record in records:
        for event in record.spawns:
            spawns[event.internal_id] = (record.t, event)
        for event in record.assignments:
            assignments[event.internal_id] = (record.t, event)
        for event in record.dropoffs:
            dropoffs[event.internal_id] = (record.t, event)
        for event in record.expiries:
            expiries[event.internal_id] = (record.t, event)

    # Строка trips каждого заказа перестраивается из его цепочки событий;
    # таблица сверяется мультимножеством: соответствие цепочки строке — по
    # значениям полей, не по trip_id (SQLite нумерует вставки в порядке
    # терминации по батчам выгрузки, а не в порядке создания заказов).
    rebuilt: Counter[tuple] = Counter()
    for internal_id, (created_at, spawn) in spawns.items():
        assigned_iso = pickup_iso = dropoff_iso = None
        if internal_id in assignments:
            assert internal_id in dropoffs and internal_id not in expiries
            assign_t, assignment = assignments[internal_id]
            dropoff_t, dropoff = dropoffs[internal_id]
            assert (assignment.origin, assignment.dest) == (spawn.origin, spawn.dest)
            assert dropoff.driver_id == assignment.driver_id
            assert dropoff.dest_zone == spawn.dest
            # t высадки равна dropoff_at, предсказанному в assignment.
            assert dropoff_t == assignment.dropoff_at
            status = "completed"
            assigned_iso, pickup_iso = to_iso(assign_t), to_iso(assignment.pickup_at)
            dropoff_iso = to_iso(dropoff_t)
        else:
            assert internal_id in expiries and internal_id not in dropoffs
            expiry_t, expiry = expiries[internal_id]
            assert expiry.origin == spawn.origin
            assert expiry_t == add_minutes(created_at, cfg.drivers.wait_limit_min)
            status = "unfulfilled"
        rebuilt[
            (
                to_iso(created_at),
                spawn.origin,
                spawn.dest,
                assigned_iso,
                pickup_iso,
                dropoff_iso,
                status,
                spawn.price_multiplier,
            )
        ] += 1

    stored = Counter(
        (
            row["created_at"],
            row["origin_zone"],
            row["dest_zone"],
            row["assigned_at"],
            row["pickup_at"],
            row["dropoff_at"],
            row["status"],
            row["price_multiplier"],
        )
        for row in conn.execute("SELECT * FROM trips")
    )
    assert rebuilt == stored

    # Полный охват событий равен итогам прогона; лишних заказов в потоке нет.
    assert len(spawns) == summary.trips_total
    assert len(assignments) == len(dropoffs) == summary.completed
    assert len(expiries) == summary.unfulfilled


def test_free_park_invariant():
    """Сумма free_by_zone каждой записи = парк минус активные поездки на конец минуты."""

    cfg = micro_config(population={"scale": 50.0}, drivers={"count": 3, "wait_limit_min": 6})
    _summary, records, _conn = run_with_records(cfg, 1)
    assert records
    active: set[int] = set()
    for record in records:
        active.update(assignment.internal_id for assignment in record.assignments)
        active.difference_update(dropoff.internal_id for dropoff in record.dropoffs)
        assert sum(record.free_by_zone) == cfg.drivers.count - len(active)


def test_minute_rhythm():
    """Сутки с наблюдателем: ровно days × 1440 записей основной фазы, t + 1 минута, секунды 00."""

    cfg = micro_config(population={"scale": 50.0}, drivers={"count": 3, "wait_limit_min": 6})
    _summary, records, _conn = run_with_records(cfg, 1)
    assert len(records) > 1440  # хвост активен: есть и основная фаза, и дренаж
    assert records[0].t == SIM_START_DT
    assert records[1439].t == add_minutes(SIM_START_DT, 1439)
    for record in records:
        assert record.t.second == 0 and record.t.microsecond == 0
    for prev, following in zip(records, records[1:]):
        assert add_minutes(prev.t, 1) == following.t
    for record in records[1440:]:
        assert record.spawns == () and record.assignments == ()
