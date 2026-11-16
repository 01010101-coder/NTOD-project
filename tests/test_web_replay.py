"""Тесты событийного лога change ``replay-server`` (``app/replay.py``, задачи 2.1–2.3).

``test_log_fills_progressively`` — прогон движка через ``ReplayLog``: записи
доступны читателю по мере прогона (срез ``[0, filled)`` в середине run'а),
после — счётчик ``filled`` покрывает основную фазу и дренаж, метрики
упорядочены по ``(ts, zone_id)`` (задача 2.1). ``test_snapshot_matches_manual_
replay`` — снимок минуты совпадает с активными поездками, восстановленными
вручную из потока записей; тихий мир на старте даёт пустой список (задача
2.2). ``test_background_fill_and_determinism`` — фоновый прогон в потоке:
``filled`` растёт во время прогона и доходит до ``days × 1440`` (+ дренаж),
повторный прогон того же seed даёт идентичные записи и метрики (задача 2.3).
"""

import time

from app.replay import ReplayLog, snapshot_at, start_fill
from app.server import record_to_dict
from common.config import AppConfig
from common.timeutil import add_minutes, from_iso
from simulator.engine import Engine

SIM_START_DT = from_iso("2026-11-02 00:00:00")

# Две синтетические зоны с одной дорогой 30 мин — как в тестах шва наблюдения
# (tests/test_sim_live.py): файлы тестов самостоятельны, поэтому мир копируется.
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

    ``scale=0`` отключает спавн (λ = 0), шансы дождя и инъекции нулевые;
    спавн включается секцией ``population``. Секция ``app`` серверу не нужна —
    seed/days передаются явно.
    """

    base = {
        "seed": 7,
        "city": {
            "zones": MICRO_ZONES,
            "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 30.0}],
        },
        "population": {"scale": 0.0},
        "drivers": {"count": 3, "wait_limit_min": 6},
        "weather": {
            "rain_day_chance": 0.0,
            "episode_start_h": [0, 0],
            "episode_duration_h": [1, 1],
        },
        "pricing": {"surge_injection": {"day_chance": 0.0}},
    }
    base.update(overrides)
    return AppConfig.model_validate(base)


def fill_sync(log: ReplayLog, cfg: AppConfig, days: float):
    """Синхронный прогон в лог (фон тестируется отдельно) — возвращает Summary."""

    return Engine(cfg, seed=cfg.seed).run(days, log.metrics, on_minute=log.append)


def test_log_fills_progressively():
    """Записи читаются по мере прогона; итог покрывает фазу и дренаж; метрики упорядочены (2.1)."""

    cfg = micro_config(population={"scale": 50.0})
    log = ReplayLog()
    checked_mid_run = False

    def observing(record) -> None:
        """Колбэк шва: аппенд в лог и контрольное чтение на 100-й записи.

        Чтение происходит, пока ``Engine.run`` ещё не вернулся — то есть
        «по мере прогона», как требует спека прогрессивной готовности.
        """

        nonlocal checked_mid_run
        log.append(record)
        if log.filled == 100:
            mid = log.read()
            assert len(mid) == 100
            assert mid[-1].t == add_minutes(SIM_START_DT, 99)
            checked_mid_run = True

    summary = Engine(cfg, seed=cfg.seed).run(0.5, log.metrics, on_minute=observing)
    assert checked_mid_run
    assert log.filled >= 720  # основная фаза (0.5 суток) плюс дренаж хвоста
    assert len(log.read()) == log.filled
    assert log.record(0).t == SIM_START_DT
    for index in (1, log.filled // 2, log.filled - 1):
        assert log.record(index).t == add_minutes(SIM_START_DT, index)

    # Метрики: порядок (ts, zone_id); 720 минут закрывают 144 окна × 2 зоны.
    rows = log.metrics_rows()
    assert len(rows) == 144 * 2
    stamps = [row["ts"] for row in rows]
    assert stamps == sorted(stamps)
    for start in range(0, len(rows), 2):
        assert [rows[start]["zone_id"], rows[start + 1]["zone_id"]] == [0, 1]
    assert summary.trips_total > 0


def test_snapshot_matches_manual_replay():
    """Снимок минуты = активные поездки и ожидание, построенные вручную (2.2, city-canvas 1.1)."""

    cfg = micro_config(population={"scale": 50.0})
    log = ReplayLog()
    fill_sync(log, cfg, 1)
    assert log.filled > 1440

    minute = 500
    record = log.record(minute)
    # Ручное восстановление: назначения ≤ t минус высадки ≤ t; спавны ≤ t
    # минус назначенные и истёкшие; минуты записей — assigned_at/created_at.
    active_manual: dict[int, dict] = {}
    pending_manual: dict[int, dict] = {}
    for past in log.read(0, minute + 1):
        for event in past.spawns:
            pending_manual[event.internal_id] = {
                "internal_id": event.internal_id,
                "origin": event.origin,
                "created_at": past.t,
            }
        for event in past.assignments:
            active_manual[event.internal_id] = {
                "internal_id": event.internal_id,
                "driver_id": event.driver_id,
                "driver_zone": event.driver_zone,
                "origin": event.origin,
                "dest": event.dest,
                "assigned_at": past.t,
                "pickup_at": event.pickup_at,
                "dropoff_at": event.dropoff_at,
            }
            pending_manual.pop(event.internal_id, None)
        for event in past.dropoffs:
            active_manual.pop(event.internal_id, None)
        for event in past.expiries:
            pending_manual.pop(event.internal_id, None)

    snapshot = snapshot_at(log, minute)
    assert snapshot["t"] == record.t
    assert snapshot["is_rain"] == record.is_rain
    assert snapshot["price_mult"] == list(record.price_mult)
    assert snapshot["free_by_zone"] == list(record.free_by_zone)
    assert snapshot["active_trips"] == [active_manual[key] for key in sorted(active_manual)]
    assert snapshot["active_trips"], "сценарий без активных поездок не проверен"
    assert snapshot["pending_orders"] == [pending_manual[key] for key in sorted(pending_manual)]
    # Оба списка упорядочены по internal_id (порядок создания заказов).
    trips_ids = [trip["internal_id"] for trip in snapshot["active_trips"]]
    pending_ids = [order["internal_id"] for order in snapshot["pending_orders"]]
    assert trips_ids == sorted(trips_ids)
    assert pending_ids == sorted(pending_ids)
    # Фаза «к пассажиру» неотрицательна: посадка не раньше минуты назначения
    # (pickup_at = t + pickup_min − 1, подача ≥ 1 минуты).
    assert all(trip["pickup_at"] >= trip["assigned_at"] for trip in snapshot["active_trips"])

    # Инвариант парка: свободные = парк минус активные (перекрёстная сверка).
    assert sum(snapshot["free_by_zone"]) == cfg.drivers.count - len(snapshot["active_trips"])

    # Пустая минута «ночи старта» для тихого мира: ни назначений, ни активных.
    quiet = ReplayLog()
    fill_sync(quiet, micro_config(), 1 / 1440)
    empty = snapshot_at(quiet, 0)
    assert empty["active_trips"] == []
    assert empty["pending_orders"] == []
    assert empty["free_by_zone"] == list(quiet.record(0).free_by_zone)


def test_cancellations_serialized_and_removed_from_pending():
    """Отказы сериализуются; снимок минуты отказа не держит заказ в ожидании (5.2)."""

    cfg = micro_config(
        city={
            "zones": [MICRO_ZONES[0], {**MICRO_ZONES[1], "attract_origin": 0.0}],
            "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 5.0}],
        },
        population={"scale": 50.0},
        drivers={
            "count": 3,
            "wait_limit_min": 6,
            "dispatch": {
                "cancel_grace_min": 0,
                "cancel_prob_per_min": 1.0,
                "cancel_max_prob": 1.0,
            },
        },
    )
    log = ReplayLog()
    fill_sync(log, cfg, 30 / 1440)

    index = next(i for i in range(log.filled) if log.record(i).cancellations)
    record = log.record(index)
    payload = record_to_dict(record)
    expected = [
        {
            "internal_id": event.internal_id,
            "origin": event.origin,
            "pickup_min": event.pickup_min,
        }
        for event in record.cancellations
    ]
    assert payload["cancellations"] == expected
    assert expected and all(item["pickup_min"] >= 1 for item in expected)

    cancelled_ids = {item["internal_id"] for item in expected}
    snapshot = snapshot_at(log, index)
    pending_ids = {order["internal_id"] for order in snapshot["pending_orders"]}
    assert cancelled_ids.isdisjoint(pending_ids)

    # При выключенных отказах список сериализуется пустым.
    quiet = ReplayLog()
    fill_sync(quiet, micro_config(population={"scale": 50.0}), 5 / 1440)
    assert record_to_dict(quiet.record(4))["cancellations"] == []


def test_background_fill_and_determinism():
    """Фоновый прогон: filled растёт во время прогона, итог детерминирован (2.3)."""

    days = 0.5
    cfg = micro_config(population={"scale": 50.0})

    log_a = ReplayLog()
    run_a = start_fill(log_a, cfg, seed=cfg.seed, days=days)
    # Прогрессивность: вскоре после старта записи уже читаются, итогов ещё нет.
    deadline = time.monotonic() + 30.0
    while log_a.filled < 60 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert log_a.filled >= 60, "лог не пополняется во время прогона"
    assert run_a.summary is None and run_a.error is None

    run_a.thread.join(timeout=60.0)
    assert run_a.done
    assert run_a.summary is not None and run_a.error is None
    assert log_a.filled >= int(days * 1440)
    # Дренаж, если хвост был активен: записи за границей без спавна/диспатча.
    drain = log_a.read(int(days * 1440))
    for record in drain:
        assert record.spawns == () and record.assignments == ()
    assert run_a.summary.trips_total > 0

    # Детерминизм лога: тот же seed — записи и метрики совпадают один в один.
    log_b = ReplayLog()
    run_b = start_fill(log_b, cfg, seed=cfg.seed, days=days)
    run_b.thread.join(timeout=60.0)
    assert run_b.done
    assert log_b.filled == log_a.filled
    assert log_b.read() == log_a.read()
    assert log_b.metrics_rows() == log_a.metrics_rows()

    # Метрики лога равны строкам синхронного прогона того же seed.
    log_sync = ReplayLog()
    summary_sync = fill_sync(log_sync, cfg, days)
    assert log_sync.metrics_rows() == log_a.metrics_rows()
    assert summary_sync.completed == run_a.summary.completed
