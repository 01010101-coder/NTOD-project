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

from common.config import AppConfig, load_config
from common.timeutil import add_minutes, bucket5, from_iso
from simulator.aggregate import AggregateBuffer
from simulator.engine import Engine

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
