"""Тесты цикла и агрегации change ``m2-loop`` (задачи 4.1–4.2, §8 плана M2).

``test_aggregation_math`` — микросценарий задачи 4.1: на синтетических мирах
(``AppConfig.model_validate`` с управляемыми профилями, парком и множителями)
движок прогоняет ≥ 2 пяти-минутных окон, и все 11 колонок строк
``zone_metrics_5min`` по каждой зоне сверяются с рукописными ожиданиями §3.8
(счётчики событий мира — дополнительно по формулам из строк ``trips``).
Остальные тесты — порядок шагов §3.7 и семантика колонок на микромирах (задача
4.2): назначение в минуту создания (wait 0), истечение ровно в
``created_at + drivers.wait_limit_min``, спасение назначением (§10.3),
``avg_wait_min`` NULL в окне без назначений, ``is_rain`` при дожде в одной
минуте окна, поминутное среднее ``price_multiplier`` при форс-инъекции (шов
``price_fn`` §10.13 — ступенчатый тариф) и приоритет «пятничного вечера» над
surge-инъекцией в метке ``event``.
"""

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pytest

from common.config import AppConfig, DispatchCfg, load_config
from common.timeutil import add_minutes, bucket5, from_iso
from simulator.aggregate import AggregateBuffer, _is_deficit
from simulator.drivers import DriverState
from simulator.engine import Engine, _cancel_prob
from simulator.shifts import online_target

SIM_START = "2026-11-02"
SIM_START_DT = from_iso(f"{SIM_START} 00:00:00")

# Эталонный конфиг ищется относительно файла теста, не от cwd (прецедент
# M1/m2-city). Хэши эталонного прогона (задача 1.1, design решение 8) сняты на
# коде до change'а ``m6-supply-response``; при выключенном отклике прогон
# обязан оставаться побитово прежним.
REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
REFERENCE_DAYS = 1
REFERENCE_SEED = 42
REFERENCE_TRIPS_SHA256 = "a73d2cb9bcc26420fa25053b8bd55004a7d2710ed0bf1495880b236ff97ae0d8"
REFERENCE_METRICS_SHA256 = "c1fb13e0aecac96b47430e136a230bc5cb62d1a6713a01f9d11b73a197967014"


def _rows_digest(rows: list[dict]) -> str:
    """sha256 канонического JSON-представления упорядоченных строк таблицы.

    ``json.dumps`` с ``sort_keys`` и ``default=str`` даёт стабильный текст:
    ключи отсортированы, ``datetime`` приводится к ISO-подобной строке.
    """

    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reference_config(*, enabled: bool, **reposition_overrides) -> AppConfig:
    """Эталонный конфиг с заданным флагом отклика и переопределениями правила."""

    cfg = load_config(REFERENCE_CONFIG).model_copy(deep=True)
    cfg.drivers.reposition.enabled = enabled
    for field, value in reposition_overrides.items():
        setattr(cfg.drivers.reposition, field, value)
    return cfg


def _free_zone5_mean(cfg: AppConfig, price_fn, minutes: int) -> float:
    """Среднее ``free_drivers`` зоны 5 по закрытым бакетам прогона."""

    writer = StubWriter()
    Engine(cfg, seed=REFERENCE_SEED, price_fn=price_fn).run(minutes / 1440, writer)
    rows = [row for row in writer.metrics if row["zone_id"] == 5]
    assert rows, "у зоны 5 нет ни одного закрытого бакета"
    return sum(row["free_drivers"] for row in rows) / len(rows)


# Две синтетические зоны с равными весами и одной дорогой; архетипы есть в
# дефолтных профилях конфига (m2-city), поэтому demand.profiles не задаётся.
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


class StubWriter:
    """Writer-заглушка: копит строки обеих таблиц в памяти (шов §10.12)."""

    def __init__(self) -> None:
        self.trips: list[dict] = []
        self.metrics: list[dict] = []

    def write_trips(self, rows: list[dict]) -> None:
        self.trips.extend(rows)

    def write_metrics(self, rows: list[dict]) -> None:
        self.metrics.extend(rows)


def micro_config(**overrides) -> AppConfig:
    """Синтетический мир: 2 зоны, дорога 30 мин, парк 1, без случайных событий.

    ``scale=0`` отключает спавн (λ = 0), шансы дождя и инъекции нулевые —
    мир полностью рукописен. ``overrides`` замещают секции верхнего уровня;
    дорога настраивается секцией ``city`` целиком.
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


def run_minutes(cfg: AppConfig, minutes: int) -> StubWriter:
    """Прогнать ``minutes`` сим-минут (дробный ``days`` — микропрогоны проверок)."""

    writer = StubWriter()
    Engine(cfg).run(minutes / 1440, writer)
    return writer


def metrics_by_key(writer: StubWriter) -> dict[tuple[str, int], dict]:
    """Индекс строк метрик по (ISO-метке окна, зоне)."""

    return {(row["ts"].strftime("%Y-%m-%d %H:%M"), row["zone_id"]): row for row in writer.metrics}


def test_aggregation_math():
    """Все 11 колонок §3.8 по каждой зоне — рукописные ожидания (задача 4.1)."""

    # Мир A: спавн отключён, парк 5 (старт i mod 2 → свободные [3, 2]), событий
    # нет — все счётчики нули, тариф 1.0. Три окна: 00:00, 00:05, 00:10.
    writer_a = run_minutes(micro_config(drivers={"count": 5, "wait_limit_min": 6}), 15)
    stamps = ["2026-11-02 00:00", "2026-11-02 00:05", "2026-11-02 00:10"]
    rows_a = metrics_by_key(writer_a)
    assert len(rows_a) == 3 * 2
    for stamp in stamps:
        for zone_id, free in ((0, 3), (1, 2)):
            row = rows_a[(stamp, zone_id)]
            assert row["ts"] == datetime.fromisoformat(stamp)
            assert row["zone_id"] == zone_id
            assert row["requests"] == 0
            assert row["completed"] == 0
            assert row["unfulfilled"] == 0
            assert row["free_drivers"] == free
            assert row["busy_drivers"] == 0
            assert row["avg_wait_min"] is None
            assert row["supply_demand_ratio"] == free / max(0, 1)
            assert row["is_rain"] == 0
            assert row["event"] == ""
            assert row["price_multiplier"] == 1.0

    # Мир B: форс-дождь с 00:00 на 1 ч — дождь был в любой момент каждого окна.
    rainy = micro_config(
        weather={
            "rain_day_chance": 1.0,
            "episode_start_h": [0, 0],
            "episode_duration_h": [1, 1],
        }
    )
    rows_b = metrics_by_key(run_minutes(rainy, 15))
    assert len(rows_b) == 3 * 2
    for row in rows_b.values():
        assert row["is_rain"] == 1

    # Мир C: ступенчатый тариф через шов price_fn (§10.13): 1.5 на минутах
    # m % 7 < 3, иначе 1.0. Поминутные средние окон рукописны: смена ступени
    # попадает внутрь окон — (3×1.5+2×1.0)/5 = 1.3, (2×1.0+3×1.5)/5 = 1.3,
    # (4×1.0+1×1.5)/5 = 1.1, (2×1.5+3×1.0)/5 = 1.2.
    stepped = Engine(micro_config(), price_fn=lambda zone_id, t: 1.5 if t.minute % 7 < 3 else 1.0)
    writer_c = StubWriter()
    stepped.run(20 / 1440, writer_c)
    rows_c = metrics_by_key(writer_c)
    assert len(rows_c) == 4 * 2
    expected_prices = {
        "2026-11-02 00:00": 1.3,
        "2026-11-02 00:05": 1.3,
        "2026-11-02 00:10": 1.1,
        "2026-11-02 00:15": 1.2,
    }
    for (stamp, _zone_id), row in rows_c.items():
        assert row["price_multiplier"] == expected_prices[stamp]

    # Мир D: живые счётчики при нулевом парке — все заказы истекают; requests и
    # unfulfilled каждого окна сверяются с формулами §3.8 поверх строк trips
    # (created_at в интервале; истечение в bucket5(created_at + wait_limit)),
    # остальные колонки рукописны (парк пуст — free/busy 0, назначений нет).
    busy_world = micro_config(
        city={
            "zones": MICRO_ZONES,
            "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 5.0}],
        },
        population={"scale": 10.0},
        drivers={"count": 0, "wait_limit_min": 6},
    )
    writer_d = run_minutes(busy_world, 15)
    assert all(trip["status"] == "unfulfilled" for trip in writer_d.trips)
    created_counts = Counter(
        (bucket5(from_iso(trip["created_at"])).strftime("%Y-%m-%d %H:%M"), trip["origin_zone"])
        for trip in writer_d.trips
    )
    expiry_counts = Counter(
        (
            bucket5(add_minutes(from_iso(trip["created_at"]), 6)).strftime("%Y-%m-%d %H:%M"),
            trip["origin_zone"],
        )
        for trip in writer_d.trips
    )
    for row in writer_d.metrics:
        key = (row["ts"].strftime("%Y-%m-%d %H:%M"), row["zone_id"])
        assert row["requests"] == created_counts[key]
        assert row["unfulfilled"] == expiry_counts[key]
        assert row["completed"] == 0
        assert row["free_drivers"] == 0 and row["busy_drivers"] == 0
        assert row["avg_wait_min"] is None
        assert row["supply_demand_ratio"] == 0.0
        assert row["is_rain"] == 0 and row["event"] == ""
        assert row["price_multiplier"] == 1.0


def test_assignment_same_minute_wait_zero():
    """Заказ, созданный при свободном водителе, назначается в минуту создания (§3.7)."""

    writer = run_minutes(micro_config(population={"scale": 50.0}), 15)
    assigned = [
        trip
        for trip in writer.trips
        if trip["status"] == "completed"
        and trip["created_at"] == "2026-11-02 00:00:00"
        and trip["assigned_at"] == "2026-11-02 00:00:00"
    ]
    assert assigned, "нет заказа, назначенного в минуту создания"
    assert assigned[0]["created_at"] <= assigned[0]["assigned_at"] <= assigned[0]["pickup_at"]


def test_expiry_exactly_at_wait_limit():
    """С нулевым парком каждый заказ истекает ровно в created_at + wait_limit_min."""

    world = micro_config(
        city={
            "zones": MICRO_ZONES,
            "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 5.0}],
        },
        population={"scale": 10.0},
        drivers={"count": 0, "wait_limit_min": 6},
    )
    writer = run_minutes(world, 15)
    assert writer.trips
    assert all(
        trip["status"] == "unfulfilled"
        and trip["assigned_at"] is None
        and trip["pickup_at"] is None
        and trip["dropoff_at"] is None
        for trip in writer.trips
    )
    # Каждое истечение попадает в окно bucket5(created_at + 6) и ни в какие
    # другие; сверяются истечения основной фазы — заказы, созданные в последние
    # минуты окна, истекают в дренаже хвоста, а метрики за хвост не пишутся
    # (design, решение 5).
    main_phase = [
        trip
        for trip in writer.trips
        if bucket5(add_minutes(from_iso(trip["created_at"]), 6)) < from_iso("2026-11-02 00:15:00")
    ]
    expected = Counter(
        bucket5(add_minutes(from_iso(trip["created_at"]), 6)).strftime("%Y-%m-%d %H:%M")
        for trip in main_phase
    )
    got = Counter(
        row["ts"].strftime("%Y-%m-%d %H:%M")
        for row in writer.metrics
        for _ in range(row["unfulfilled"])
    )
    assert expected == got
    # В окно самой минуты создания истечения попасть не может: лимит ≥ 1 минуты.
    assert sum(row["unfulfilled"] for row in writer.metrics if row["ts"] == SIM_START_DT) == 0


def test_assignment_saves_from_expiry():
    """Назначение на лимитной минуте спасает заказ от истечения (§10.3).

    Один водитель с короткой дорогой (5 мин): занят первые 5 минут, освобождается
    на минуте 00:05; ожидающий заказ минуты 00:00 диспатч назначает на минуте
    00:06 — в саму минуту истечения, что происходит только после диспатча.
    """

    world = micro_config(
        city={
            "zones": MICRO_ZONES,
            "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 5.0}],
        },
        population={"scale": 50.0},
        drivers={"count": 1, "wait_limit_min": 6},
    )
    writer = run_minutes(world, 30)
    saved = [
        trip
        for trip in writer.trips
        if trip["status"] == "completed"
        and trip["created_at"] == "2026-11-02 00:00:00"
        and trip["assigned_at"] == "2026-11-02 00:06:00"
    ]
    assert saved, "назначение на лимитной минуте не спасло заказ"
    # При единственном водителе часть заказов той же минуты спасения не получает.
    assert any(
        trip["status"] == "unfulfilled" and trip["created_at"] == "2026-11-02 00:00:00"
        for trip in writer.trips
    )


def test_avg_wait_null_without_assignments():
    """В окне без назначений avg_wait_min = NULL; supply_demand_ratio вычислен."""

    world = micro_config(
        city={
            "zones": MICRO_ZONES,
            "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 5.0}],
        },
        population={"scale": 10.0},
        drivers={"count": 0, "wait_limit_min": 6},
    )
    writer = run_minutes(world, 15)
    assert writer.trips  # заказы есть, назначений нет
    for row in writer.metrics:
        assert row["avg_wait_min"] is None
        assert row["supply_demand_ratio"] == row["free_drivers"] / max(row["requests"], 1)


def test_is_rain_any_minute_in_window():
    """is_rain — ИЛИ по минутам: одна дождливая минута окна даёт 1, ни одной — 0."""

    buffer = AggregateBuffer(2)
    for minute, is_rain in ((1, False), (2, False), (3, True), (4, False), (5, False)):
        buffer.set_env(0, is_rain, "", 1.0)
        buffer.set_env(1, False, "", 1.0)
    row0, row1 = buffer.flush(SIM_START_DT)
    assert row0["is_rain"] == 1 and row1["is_rain"] == 0

    # Через движок: эпизод [00:00, 01:00) — окна 00:00..00:55 дождливые,
    # окна после 01:00 — сухие («при отсутствии дождя во всём интервале — 0»).
    world = micro_config(
        weather={
            "rain_day_chance": 1.0,
            "episode_start_h": [0, 0],
            "episode_duration_h": [1, 1],
        }
    )
    writer = run_minutes(world, 70)  # 14 окон: 12 в дожде, 2 после него
    for row in writer.metrics:
        expected = 1 if row["ts"] < from_iso("2026-11-02 01:00:00") else 0
        assert row["is_rain"] == expected


def test_price_multiplier_minute_average():
    """price_multiplier окна — среднее поминутных значений (шов price_fn §10.13)."""

    stepped = Engine(micro_config(), price_fn=lambda zone_id, t: 1.5 if t.minute % 7 < 3 else 1.0)
    writer = StubWriter()
    stepped.run(20 / 1440, writer)
    expected = {
        "2026-11-02 00:00": 1.3,
        "2026-11-02 00:05": 1.3,
        "2026-11-02 00:10": 1.1,
        "2026-11-02 00:15": 1.2,
    }
    for row in writer.metrics:
        assert row["price_multiplier"] == expected[row["ts"].strftime("%Y-%m-%d %H:%M")]


def test_event_priority_friday_over_surge():
    """Метка event: «пятничный вечер» приоритетнее surge-инъекции (первое сработавшее).

    «Пятничный вечер» активен все дни и часы для business-архетипа (обе зоны —
    business); инъекция форсируется на весь день с множителем 2.0 — она точно
    разыграна (day_chance = 1.0), колонка price_multiplier показывает её зону,
    но метка event во всех строках обеих зон остаётся friday_evening.
    """

    world = micro_config(
        city={
            "zones": [
                {**MICRO_ZONES[0], "archetype": "business"},
                {**MICRO_ZONES[1], "archetype": "business"},
            ],
            "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 5.0}],
        },
        events={
            "friday_evening": {
                "days": [0, 1, 2, 3, 4, 5, 6],
                "hours": [0, 23],
                "archetypes": ["business"],
                "demand_mult": 1.3,
            }
        },
        pricing={
            "surge_injection": {
                "day_chance": 1.0,
                "duration_min": [1440, 1440],
                "multiplier": [2.0, 2.0],
            }
        },
    )
    # Прогон целых суток: инъекция стартует U(сутки) и длится весь остаток дня,
    # поэтому её след (price_multiplier > 1.0 в зоне действия) гарантированно
    # виден в метриках — окно минуты старта всегда содержит минуту инъекции.
    writer = run_minutes(world, 1440)
    assert any(row["price_multiplier"] > 1.0 for row in writer.metrics), (
        "форс-инъекция не разыграна — приоритет не проверен"
    )
    assert all(row["event"] == "friday_evening" for row in writer.metrics)


def test_reference_run_bitwise_compatibility() -> None:
    """Эталон совместимости (задача 1.1, design решение 8).

    Прогон 1 сутки seed 42 на эталонном конфиге обязан давать те же строки
    ``trips`` и ``zone_metrics_5min``, что и код до change'а
    ``m6-supply-response``: хэши сняты на неизменённом коде и записаны
    литералами. Сценарий «Эталонный прогон без отклика» спеки simulator-loop.
    """

    cfg = load_config(REFERENCE_CONFIG)
    writer = StubWriter()
    summary = Engine(cfg, seed=REFERENCE_SEED).run(REFERENCE_DAYS, writer)

    assert summary.trips_total == summary.completed + summary.unfulfilled > 0
    assert _rows_digest(writer.trips) == REFERENCE_TRIPS_SHA256
    assert _rows_digest(writer.metrics) == REFERENCE_METRICS_SHA256


# --- Отклик предложения в минутном цикле (change m6-supply-response, задача 5.2) ---


def test_response_pulls_free_drivers_to_surcharge() -> None:
    """Сценарий «Свободные подтягиваются к надбавке»: 2 часа, зона 5 дороже."""

    price_fn = lambda zone_id, _t: 1.3 if zone_id == 5 else 1.0  # noqa: E731
    off = _reference_config(enabled=False)
    on = _reference_config(enabled=True)

    assert _free_zone5_mean(on, price_fn, 120) > _free_zone5_mean(off, price_fn, 120)


def test_response_does_not_consume_rng() -> None:
    """Сценарий «Отклик не тратит rng»: дождь и порядок заказов не сдвигаются."""

    price_fn = lambda zone_id, _t: 1.3 if zone_id == 5 else 1.0  # noqa: E731
    minutes = 120

    def snapshot(cfg: AppConfig) -> list:
        records: list = []
        Engine(cfg, seed=REFERENCE_SEED, price_fn=price_fn).run(
            minutes / 1440, StubWriter(), on_minute=records.append
        )
        return records[:minutes]  # основная фаза: по записи на минуту, без дренажа

    off = snapshot(_reference_config(enabled=False))
    on = snapshot(_reference_config(enabled=True))

    assert [record.is_rain for record in on] == [record.is_rain for record in off]
    spawns = lambda records: [  # noqa: E731
        (record.t, spawn.origin, spawn.dest) for record in records for spawn in record.spawns
    ]
    assert spawns(on) == spawns(off)
    # Прогон с откликом действительно переезжал: состояние парка по зонам
    # отличается, иначе тест проверял бы пустую ветку правила.
    assert any(a.free_by_zone != b.free_by_zone for a, b in zip(on, off))


def test_relocation_only_on_decision_minutes() -> None:
    """Сценарий «Решения только на минутах решения»: первый переезд на k = 5."""

    cfg = _reference_config(enabled=True, every_min=5)
    cfg.population.scale = 0.0  # без заказов свободные не двигаются сами
    cfg.weather.rain_day_chance = 0.0
    # Минута 0 нейтральна; надбавка действует с минуты 1, но решение — только k % 5 == 0.
    price_fn = lambda zone_id, t: 1.3 if zone_id == 5 and t.minute >= 1 else 1.0  # noqa: E731

    records: list = []
    Engine(cfg, seed=REFERENCE_SEED, price_fn=price_fn).run(
        6 / 1440, StubWriter(), on_minute=records.append
    )

    assert len(records) == 6
    initial = records[0].free_by_zone
    assert all(record.free_by_zone == initial for record in records[:5])
    assert records[5].free_by_zone != initial


def test_response_counters_sum_invariant() -> None:
    """Инвариант ``free_drivers + busy_drivers = drivers.count`` на каждом бакете."""

    cfg = _reference_config(enabled=True)
    writer = StubWriter()
    Engine(cfg, seed=REFERENCE_SEED, price_fn=lambda zone_id, _t: 1.3 if zone_id == 5 else 1.0).run(
        180 / 1440, writer
    )

    totals: dict[datetime, list[int]] = defaultdict(lambda: [0, 0])
    for row in writer.metrics:
        totals[row["ts"]][0] += row["free_drivers"]
        totals[row["ts"]][1] += row["busy_drivers"]

    assert totals
    for free, busy in totals.values():
        assert free + busy == cfg.drivers.count


# --- Радиус поиска и отказ пассажира (change world-dispatch-radius) ---


def pickup_minutes(trip: dict) -> int:
    """Фактическая подача строки ``trips`` в минутах: ``pickup_at − assigned_at + 1``."""

    return (
        int((from_iso(trip["pickup_at"]) - from_iso(trip["assigned_at"])).total_seconds() // 60) + 1
    )


def _reference_with_dispatch(**overrides) -> AppConfig:
    """Эталонный конфиг с переопределениями подсекции ``drivers.dispatch``."""

    cfg = load_config(REFERENCE_CONFIG).model_copy(deep=True)
    for field, value in overrides.items():
        setattr(cfg.drivers.dispatch, field, value)
    return cfg


def test_driver_outside_radius_not_assigned() -> None:
    """Свободная машина за радиусом не назначается, заказ истекает, водитель свободен."""

    # Зона 0 — только стартовая зона единственного водителя (спроса нет); заказы
    # рождаются в зоне 1, путь 0→1 = 30 минут > радиуса 10.
    world = micro_config(
        city={
            "zones": [{**MICRO_ZONES[0], "attract_origin": 0.0}, MICRO_ZONES[1]],
            "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 30.0}],
        },
        population={"scale": 50.0},
        drivers={
            "count": 1,
            "wait_limit_min": 6,
            "dispatch": {"max_pickup_min": 10},
        },
    )
    writer = run_minutes(world, 20)

    assert writer.trips
    assert all(trip["status"] == "unfulfilled" for trip in writer.trips)
    # Водитель не поехал: в зоне 0 он свободен на каждом снапшоте.
    assert all(row["free_drivers"] == 1 for row in writer.metrics if row["zone_id"] == 0)


def test_rain_narrows_radius() -> None:
    """Дождь сужает радиус: путь 9 минут × 1.4 = 13 > 10 — назначения нет."""

    def world(*, rain: bool) -> AppConfig:
        return micro_config(
            city={
                "zones": [{**MICRO_ZONES[0], "attract_origin": 0.0}, MICRO_ZONES[1]],
                "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 9.0}],
            },
            population={"scale": 50.0},
            drivers={
                "count": 1,
                "wait_limit_min": 6,
                "dispatch": {"max_pickup_min": 10},
            },
            weather={
                "rain_day_chance": 1.0 if rain else 0.0,
                "episode_start_h": [0, 0],
                "episode_duration_h": [1, 1],
            },
        )

    dry = run_minutes(world(rain=False), 20)
    completed_dry = [trip for trip in dry.trips if trip["status"] == "completed"]
    assert completed_dry
    assert all(pickup_minutes(trip) == 9 for trip in completed_dry)

    rainy = run_minutes(world(rain=True), 20)
    assert rainy.trips
    assert all(trip["status"] == "unfulfilled" for trip in rainy.trips)


def test_order_outside_radius_does_not_block_queue() -> None:
    """Заказ вне радиуса не мешает назначению следующего заказа той же минуты."""

    # Зона 0 — стартовая зона водителя; зона 1 дальняя (30 мин), зона 2 близкая
    # (5 мин). Внутренние id идут по возрастанию зоны, поэтому дальний заказ
    # стоит в очереди первым, близкий — вторым.
    world = micro_config(
        city={
            "zones": [
                {**MICRO_ZONES[0], "attract_origin": 0.0},
                MICRO_ZONES[1],
                {
                    "id": 2,
                    "name": "c",
                    "archetype": "mixed",
                    "attract_origin": 1.0,
                    "attract_dest": 1.0,
                    "map": {"x": 20.0, "y": 0.0},
                },
            ],
            "roads": [
                {"from_zone": 0, "to_zone": 1, "minutes": 30.0},
                {"from_zone": 0, "to_zone": 2, "minutes": 5.0},
            ],
        },
        population={"scale": 100.0},
        drivers={
            "count": 1,
            "wait_limit_min": 6,
            "dispatch": {"max_pickup_min": 10},
        },
    )
    writer = run_minutes(world, 20)

    waited = [trip for trip in writer.trips if trip["status"] == "unfulfilled"]
    assigned = [
        trip for trip in writer.trips if trip["status"] == "completed" and trip["origin_zone"] == 2
    ]
    assert waited and assigned, "сценарий очереди не воспроизведён"
    # Дальний заказ был в очереди в минуту назначения близкого: назначение
    # близкого не заблокировано первым заказом.
    assert any(
        from_iso(far["created_at"])
        <= from_iso(near["assigned_at"])
        <= add_minutes(from_iso(far["created_at"]), world.drivers.wait_limit_min)
        for far in waited
        for near in assigned
    )


def test_completed_pickup_within_radius() -> None:
    """Инвариант 1 суток seed 42 с радиусом 10: подача всех ``completed`` ≤ 10 минут."""

    cfg = _reference_with_dispatch(max_pickup_min=10)
    writer = StubWriter()
    Engine(cfg, seed=REFERENCE_SEED).run(1, writer)

    completed = [trip for trip in writer.trips if trip["status"] == "completed"]
    assert completed
    assert all(pickup_minutes(trip) <= 10 for trip in completed)


def test_cancel_prob_formula() -> None:
    """Вероятность отказа: 0 в пределах терпения, линейный рост, потолок."""

    cfg = DispatchCfg(cancel_grace_min=5, cancel_prob_per_min=0.1, cancel_max_prob=0.9)
    assert _cancel_prob(2, cfg) == 0.0
    assert _cancel_prob(5, cfg) == 0.0
    assert _cancel_prob(8, cfg) == pytest.approx(0.3)
    assert _cancel_prob(20, cfg) == pytest.approx(0.9)
    assert _cancel_prob(20, DispatchCfg()) == 0.0


def test_pickup_within_grace_not_drawn() -> None:
    """Подача в пределах терпения не разыгрывается: p = 0, заказ назначается."""

    world = micro_config(
        city={
            "zones": [{**MICRO_ZONES[0], "attract_origin": 0.0}, MICRO_ZONES[1]],
            "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 5.0}],
        },
        population={"scale": 50.0},
        drivers={
            "count": 1,
            "wait_limit_min": 6,
            "dispatch": {
                "cancel_grace_min": 5,
                "cancel_prob_per_min": 0.1,
                "cancel_max_prob": 0.9,
            },
        },
    )
    writer = StubWriter()
    summary = Engine(world).run(20 / 1440, writer)

    assert summary.cancelled == 0
    assert not any(trip["status"] == "cancelled" for trip in writer.trips)
    completed = [trip for trip in writer.trips if trip["status"] == "completed"]
    assert completed
    assert all(pickup_minutes(trip) == 5 for trip in completed)


def test_cancellation_frees_driver_for_next_order() -> None:
    """Отказ освобождает водителя: следующие заказы той же минуты получают его же."""

    world = micro_config(
        city={
            "zones": [MICRO_ZONES[0], {**MICRO_ZONES[1], "attract_origin": 0.0}],
            "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 5.0}],
        },
        population={"scale": 100.0},
        drivers={
            "count": 1,
            "wait_limit_min": 6,
            "dispatch": {
                "cancel_grace_min": 0,
                "cancel_prob_per_min": 1.0,
                "cancel_max_prob": 1.0,
            },
        },
    )
    records: list = []
    writer = StubWriter()
    summary = Engine(world).run(10 / 1440, writer, on_minute=records.append)

    assert summary.cancelled == len(writer.trips) > 0
    for trip in writer.trips:
        assert trip["status"] == "cancelled"
        assert trip["assigned_at"] is None
        assert trip["pickup_at"] is None
        assert trip["dropoff_at"] is None
    # Водитель не занят ни минуты: в каждой записи он свободен в зоне 0.
    assert all(record.free_by_zone == (1, 0) for record in records)
    # В одной минуте отказано нескольким заказам — водитель предложен повторно.
    assert max(len(record.cancellations) for record in records) >= 2


def test_cancellations_do_not_shift_spawn_or_env() -> None:
    """Отказы не сдвигают спавн и среду: поминутные is_rain, тариф и заказы равны."""

    minutes = 300

    def snapshot(cfg: AppConfig) -> list:
        records: list = []
        Engine(cfg, seed=REFERENCE_SEED).run(minutes / 1440, StubWriter(), on_minute=records.append)
        return records[:minutes]

    off = snapshot(_reference_with_dispatch(cancel_prob_per_min=0.0))
    on = snapshot(_reference_with_dispatch(cancel_prob_per_min=0.1))

    assert any(record.cancellations for record in on), "отказы не разыграны"
    assert [record.is_rain for record in on] == [record.is_rain for record in off]
    assert [record.price_mult for record in on] == [record.price_mult for record in off]

    def spawns(records: list) -> list:
        return [
            (record.t, spawn.origin, spawn.dest) for record in records for spawn in record.spawns
        ]

    assert spawns(on) == spawns(off)


def test_cancellations_deterministic() -> None:
    """Два прогона одного seed с отказами побайтово равны."""

    cfg = _reference_with_dispatch(cancel_prob_per_min=0.1)
    writer_a = StubWriter()
    summary_a = Engine(cfg, seed=REFERENCE_SEED).run(0.25, writer_a)
    writer_b = StubWriter()
    summary_b = Engine(cfg, seed=REFERENCE_SEED).run(0.25, writer_b)

    assert summary_a.cancelled == summary_b.cancelled > 0
    assert _rows_digest(writer_a.trips) == _rows_digest(writer_b.trips)
    assert _rows_digest(writer_a.metrics) == _rows_digest(writer_b.metrics)


def test_summary_counts_cancelled() -> None:
    """Summary сверяется со счётом статусов в строках; при дефолтах отказов нет."""

    cfg = _reference_with_dispatch(cancel_prob_per_min=0.1)
    writer = StubWriter()
    summary = Engine(cfg, seed=REFERENCE_SEED).run(0.25, writer)

    statuses = Counter(trip["status"] for trip in writer.trips)
    assert summary.cancelled == statuses["cancelled"] > 0
    assert summary.trips_total == summary.completed + summary.unfulfilled + summary.cancelled
    assert summary.cancelled_share == pytest.approx(100.0 * summary.cancelled / summary.trips_total)

    default = Engine(load_config(REFERENCE_CONFIG), seed=REFERENCE_SEED).run(0.05, StubWriter())
    assert default.cancelled == 0
    assert default.cancelled_share == 0.0


def test_cancellations_count_in_unfulfilled() -> None:
    """Отказы входят в unfulfilled агрегата и дают дефицит строки окна."""

    # Зона 0: водитель рядом (подача 1 мин > терпения 0) — каждый заказ
    # отменяется, водитель остаётся свободным. Зона 1: путь 30 > радиуса 10 —
    # заказы не назначаются и истекают. В окне есть и истечения, и отказы.
    world = micro_config(
        city={
            "zones": MICRO_ZONES,
            "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 30.0}],
        },
        population={"scale": 100.0},
        drivers={
            "count": 1,
            "wait_limit_min": 6,
            "dispatch": {
                "max_pickup_min": 10,
                "cancel_grace_min": 0,
                "cancel_prob_per_min": 1.0,
                "cancel_max_prob": 1.0,
            },
        },
    )
    records: list = []
    writer = StubWriter()
    Engine(world).run(20 / 1440, writer, on_minute=records.append)

    written = {(row["ts"], row["zone_id"]) for row in writer.metrics}
    cancellations: Counter = Counter()
    expiries: Counter = Counter()
    for record in records:
        window = bucket5(record.t)
        if (window, 0) not in written:
            continue
        for event in record.cancellations:
            cancellations[(window, event.origin)] += 1
        for event in record.expiries:
            expiries[(window, event.origin)] += 1

    both = [
        window for window, _zone in written if cancellations[(window, 0)] and expiries[(window, 1)]
    ]
    assert both, "нет окна с истечением и отказом одновременно"
    rows = {(row["ts"], row["zone_id"]): row for row in writer.metrics}
    window = both[0]
    assert rows[(window, 0)]["unfulfilled"] == cancellations[(window, 0)]
    assert rows[(window, 1)]["unfulfilled"] == expiries[(window, 1)]
    assert _is_deficit(rows[(window, 0)])
    assert _is_deficit(rows[(window, 1)])


# --- Смены водителей в минутном цикле (change world-driver-shifts, задача 4) ---


def _shift_profile(value: float) -> list[float]:
    """Ровный профиль доли парка на смене: 24 одинаковых значения."""

    return [value] * 24


def _silent_shift_config(*, count: int, share: float, zones: int = 6) -> AppConfig:
    """Эталонный конфиг без спавна и дождя, парк ``count`` и ровный профиль смен."""

    cfg = load_config(REFERENCE_CONFIG).model_copy(deep=True)
    cfg.drivers.count = count
    cfg.drivers.shifts.enabled = True
    cfg.drivers.shifts.online_share.weekday = _shift_profile(share)
    cfg.drivers.shifts.online_share.weekend = _shift_profile(share)
    cfg.population.scale = 0.0
    cfg.weather.rain_day_chance = 0.0
    return cfg


def _micro_shift_config(**overrides) -> AppConfig:
    """Микромир 2 зон со включёнными сменами (профиль по умолчанию)."""

    drivers = {"count": 20, "wait_limit_min": 6, "shifts": {"enabled": True}}
    drivers.update(overrides.pop("drivers", {}))
    return micro_config(drivers=drivers, **overrides)


def test_shifts_night_third_of_fleet() -> None:
    """Сценарий «Ночью на смене треть парка»: 03:00 будней, 99 из 300."""

    cfg = load_config(REFERENCE_CONFIG).model_copy(deep=True)
    cfg.drivers.count = 300
    cfg.drivers.shifts.enabled = True
    cfg.population.scale = 0.0
    cfg.weather.rain_day_chance = 0.0

    records: list = []
    Engine(cfg, seed=REFERENCE_SEED).run(181 / 1440, StubWriter(), on_minute=records.append)

    night = records[180]
    assert night.t == from_iso("2026-11-02 03:00:00")
    assert night.online_drivers == 99


def test_shifts_free_drivers_leave_first() -> None:
    """Сценарий «Свободные уходят первыми»: 20 свободных, цель 15 — ушли id 15..19."""

    cfg = _silent_shift_config(count=20, share=0.75)
    engine = Engine(cfg, seed=REFERENCE_SEED)
    records: list = []
    engine.run(1 / 1440, StubWriter(), on_minute=records.append)

    assert records[0].online_drivers == 15
    offline = {
        driver.id for driver in engine._fleet._drivers if driver.state is DriverState.OFFLINE
    }
    assert offline == {15, 16, 17, 18, 19}
    assert engine._fleet.leaving_count() == 0  # все ушли свободными, меток нет


def test_shifts_online_driver_home_zone_same_minute() -> None:
    """Сценарий «Выход на смену в домашней зоне»: водитель 13 → зона 1."""

    cfg = _silent_shift_config(count=20, share=1.0)
    engine = Engine(cfg, seed=REFERENCE_SEED)
    fleet = engine._fleet
    # Водители 1 и 7 зоны 1 заняты — после выхода 13 он ближайший свободный.
    for driver_id in (1, 7):
        fleet.assign(driver_id, 100 + driver_id, 5)
        fleet._plan_trip(driver_id, 3, 5)
        engine._free_ids[1].remove(driver_id)
    # Водитель 13 вне смены: цель 20 при 19 на смене впускает его.
    fleet.go_offline(13)
    engine._free_ids[1].remove(13)

    engine._apply_shifts(from_iso("2026-11-02 00:01:00"))

    driver = fleet._drivers[13]
    assert driver.state is DriverState.FREE
    assert driver.zone == 1  # 13 mod 6
    assert 13 in engine._free_ids[1]
    assert engine._nearest_free_id(1) == 13  # доступен диспатчу этой же минуты


def test_shifts_online_follows_target() -> None:
    """Сценарий «Число на смене следует цели»: 2 суток, поминутно online − метки = T(t)."""

    cfg = _micro_shift_config(population={"scale": 50.0})
    engine = Engine(cfg, seed=7)
    observed: list[tuple[object, int]] = []
    engine.run(
        2,
        StubWriter(),
        on_minute=lambda record: observed.append((record, engine._fleet.leaving_count())),
    )

    main_phase = observed[: 2 * 1440]
    assert len(main_phase) == 2 * 1440
    for record, leaving in main_phase:
        target = online_target(cfg.drivers, record.t)
        assert record.online_drivers - leaving == target
        assert record.online_drivers >= target


def test_shifts_do_not_consume_rng() -> None:
    """Сценарий «Смены не тратят rng»: дождь и последовательность спавна совпадают."""

    def snapshot(enabled: bool) -> tuple[list, list, list]:
        cfg = _micro_shift_config(
            population={"scale": 50.0},
            drivers={"count": 10, "shifts": {"enabled": enabled}},
        )
        records: list = []
        Engine(cfg, seed=7).run(2 / 24, StubWriter(), on_minute=records.append)
        main_phase = records[: 2 * 60]
        rain = [record.is_rain for record in main_phase]
        spawns = [
            (record.t, spawn.origin, spawn.dest, spawn.price_multiplier)
            for record in main_phase
            for spawn in record.spawns
        ]
        online = [record.online_drivers for record in main_phase]
        return rain, spawns, online

    rain_off, spawns_off, online_off = snapshot(enabled=False)
    rain_on, spawns_on, online_on = snapshot(enabled=True)

    assert rain_on == rain_off
    assert spawns_on == spawns_off
    # Ветка смен действительно работала: парк на смене менялся.
    assert set(online_on) != set(online_off)


def test_shifts_deterministic() -> None:
    """Сценарий «Смены детерминированы»: два прогона одного seed равны побайтово."""

    cfg = _micro_shift_config(population={"scale": 50.0}, drivers={"count": 10})

    def run() -> tuple[list[dict], list[dict]]:
        writer = StubWriter()
        Engine(cfg, seed=7).run(1, writer)
        return writer.trips, writer.metrics

    trips_a, metrics_a = run()
    trips_b, metrics_b = run()
    assert trips_a and metrics_a
    assert trips_a == trips_b
    assert metrics_a == metrics_b


def test_shifts_night_load_higher() -> None:
    """Сценарий «Ночная загрузка становится реалистичной»: 7 суток seed 42, 02–05 ч."""

    def night_load(enabled: bool) -> float:
        cfg = load_config(REFERENCE_CONFIG).model_copy(deep=True)
        cfg.drivers.count = 60
        cfg.drivers.shifts.enabled = enabled
        writer = StubWriter()
        Engine(cfg, seed=REFERENCE_SEED).run(7, writer)
        buckets = [row for row in writer.metrics if 2 <= row["ts"].hour < 5]
        assert buckets
        return sum(
            row["busy_drivers"] / max(row["free_drivers"] + row["busy_drivers"], 1)
            for row in buckets
        ) / len(buckets)

    assert night_load(enabled=True) > night_load(enabled=False)
