"""Тесты simulator/policy.py и швов движка change'а m6-policy.

Математика политик (задачи 2.1–2.2): сценарии «Таблица ступеней по прил. А»,
«Сдвиг сдвигает границу», «Клиппинг p_eff в [0, 1]», «Ступень компонуется
с дождём», «Потолок суммарного тарифа», «Неактивная политика не меняет тариф»,
«Дефицит сейчас → реактивный множитель», «Спокойная зона → множитель 1.00» и
«Нет закрытых бакетов — прогрев» спеки simulator-policy.

Швы движка (задача 3.1): «Наблюдатель получает строки закрытых бакетов» и
«Наблюдение метрик не меняет мир» (спека simulator-loop) — содержимое строк
наблюдателя равно строкам БД прогона, прогон с наблюдателем и без побайтово
одинаков; «world_tariff отдаёт тариф среды без политики» — значения совпадают
с автономной средой тех же seed, при заданном price_fn и без него, чтение во
время прогона мир не меняет.

LiveRowSource (задача 3.2): «Эквивалентность источнику поверх БД того же
прогона» — окна, точечные метрики, dropoff_counts и build_window совпадают
с SqliteRowSource поверх БД того же прогона; «Неполное окно в прогреве» —
None, как в M3.

Контур (задачи 4.1–4.4): «Стаб-модель с фиксированной вероятностью» — таблица
ступеней 0.20/0.45/0.65/0.90 → 1.00/1.15/1.30/1.50; «Множитель держится в
пределах бакета» — счётчик вызовов стаба = 1 на бакет-зону; «Доля срабатываний
стаб-модели» — 1.0 / 0.0; «Нет артефакта модели» и smoke на мини-артефакте
M3 (deficit_prob ∈ [0, 1], ступень применена); «Будущее не меняет признаки
контура» и повторная генерация ветки со стаб-политикой побайтова; «Префикс
до активации равен a-base» с активацией на последнем дне.
"""

import dataclasses
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from common.config import AppConfig, MlCfg, PolicyCfg, load_config
from common.timeutil import add_minutes, from_iso, to_iso
from ml.dataset import SqliteRowSource, bucket_index, build_window, from_bucket_index
from ml.train import model_version_for, prepare_training_data, save_model_artifacts, train_model
from simulator.__main__ import SqliteTripWriter
from simulator.engine import Engine
from simulator.env import WorldEnv
from simulator.live import AssignEvent, MinuteRecord
from simulator.policy import (
    LiveRowSource,
    NetworkPredictor,
    PolicyContour,
    compose_tariff,
    has_deficit,
    model_shift,
    reactive_multiplier,
    stair_multiplier,
)
from storage.db import connect
from storage.repo import insert_zone_metrics
from storage.schema import init_db

# Ступени прил. А — дефолты модели конфига (эталон = дефолты).
DEFAULT_STAIRS = PolicyCfg().stairs

REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
SIM_START = "2026-11-02"
SIM_START_DT = from_iso(f"{SIM_START} 00:00:00")


def test_stair_table_appendix_a() -> None:
    """Таблица прил. А при сдвиге 0: 0.39/0.40/0.60/0.80/0.95 → 1.00/1.15/1.30/1.50/1.50."""

    probs = (0.39, 0.40, 0.60, 0.80, 0.95)
    assert [stair_multiplier(p, 0.0, DEFAULT_STAIRS) for p in probs] == [
        1.00,
        1.15,
        1.30,
        1.50,
        1.50,
    ]


def test_stair_order_does_not_matter() -> None:
    """Порядок списка ступеней не влияет на результат (решение 9 design.md)."""

    for p in (0.0, 0.39, 0.40, 0.59, 0.60, 0.79, 0.80, 1.0):
        assert stair_multiplier(p, 0.0, list(reversed(DEFAULT_STAIRS))) == stair_multiplier(
            p, 0.0, DEFAULT_STAIRS
        )


def test_shift_moves_boundary() -> None:
    """Сдвиг 0.05 двигает границу: 0.37 + 0.05 = 0.42 → ступень 0.40; без сдвига — 1.00."""

    assert stair_multiplier(0.37, 0.05, DEFAULT_STAIRS) == 1.15
    assert stair_multiplier(0.37, 0.0, DEFAULT_STAIRS) == 1.00


def test_model_shift_default_zero() -> None:
    """Модель без записи в ``shift`` получает δ = 0.0 (требование секции policy)."""

    assert model_shift("mlp", {"mlp": -0.05, "lstm": 0.0}) == -0.05
    assert model_shift("transformer", {"mlp": -0.05}) == 0.0


def test_p_eff_clipped_to_unit_interval() -> None:
    """Клиппинг p_eff в [0, 1]: сумма выше 1 и ниже 0 границ не переполняет."""

    assert stair_multiplier(0.98, 0.10, DEFAULT_STAIRS) == 1.50  # p_eff = 1.0
    assert stair_multiplier(0.02, -0.10, DEFAULT_STAIRS) == 1.00  # p_eff = 0.0


def test_compose_with_rain() -> None:
    """Ступень компонуется с дождём умножением: 1.5 × 1.30 = 1.95, а не подмена ×1.30."""

    assert compose_tariff(1.5, 1.30, 2.0) == pytest.approx(1.95)


def test_total_cap() -> None:
    """Потолок суммарного тарифа: 1.5 × 1.50 = 2.25 → итог 2.0."""

    assert compose_tariff(1.5, 1.50, 2.0) == 2.0


def test_inactive_policy_inert() -> None:
    """Множитель 1.00 — тариф мира как есть, без клиппа (мир выше потолка не режется)."""

    assert compose_tariff(2.5, 1.0, 2.0) == 2.5
    assert compose_tariff(1.0, 1.0, 2.0) == 1.0


def _row(**overrides: Any) -> dict:
    """Рукописная строка метрик закрытого бакета (ключи — как у ``AggregateBuffer.flush``)."""

    row: dict = {
        "ts": "2026-11-02 00:05:00",
        "zone_id": 0,
        "requests": 4,
        "completed": 2,
        "unfulfilled": 0,
        "free_drivers": 5,
        "busy_drivers": 3,
        "avg_wait_min": None,
        "supply_demand_ratio": 1.25,
        "is_rain": 0,
        "event": "",
        "price_multiplier": 1.0,
    }
    row.update(overrides)
    return row


def test_has_deficit_rule_branches() -> None:
    """Правило контракта: каждая из двух ветвей — дефицит, спокойная строка — нет."""

    assert has_deficit(_row(unfulfilled=1)) is True  # ветвь «истёкшие заказы»
    assert has_deficit(_row(requests=10, free_drivers=4)) is True  # 0.4 < 0.5
    assert has_deficit(_row(requests=2, free_drivers=5)) is False


def test_reactive_deficit_by_unfulfilled() -> None:
    """Ветвь «истёкшие заказы»: ``unfulfilled > 0`` при хорошем обеспечении — ×reactive_mult."""

    row = _row(unfulfilled=2, requests=2, free_drivers=10)
    assert reactive_multiplier(row, 1.30) == 1.30


def test_reactive_deficit_by_ratio() -> None:
    """Ветвь «низкое обеспечение»: 1/4 = 0.25 < 0.5 — ×reactive_mult."""

    row = _row(requests=4, free_drivers=1)
    assert reactive_multiplier(row, 1.30) == 1.30


def test_reactive_boundary_ratio_exact_half() -> None:
    """Граница ровно 0.5 дефицитом не считается (порог строгий): 2/4 → 1.00."""

    row = _row(requests=4, free_drivers=2)
    assert reactive_multiplier(row, 1.30) == 1.00


def test_reactive_calm_zone() -> None:
    """Спокойная зона без дефицита по правилу контракта — множитель 1.00."""

    assert reactive_multiplier(_row(requests=2, free_drivers=5), 1.30) == 1.00


def test_reactive_warmup_no_rows() -> None:
    """Нет закрытых бакетов (прогрев мира) — правило не срабатывает: множитель 1.00."""

    assert reactive_multiplier(None, 1.30) == 1.00


def test_reactive_composes_with_world() -> None:
    """Итоговый тариф реактивной политики — композиция: min(1.5 × 1.30, 2.0) = 1.95."""

    mult = reactive_multiplier(_row(unfulfilled=1), 1.30)
    assert compose_tariff(1.5, mult, 2.0) == pytest.approx(1.95)


# --- Группа 3: швы движка и LiveRowSource -----------------------------------
#
# Две синтетические зоны с одной дорогой 30 мин — стиль tests/test_sim_live.py:
# файлы тестов самостоятельны, поэтому мир копируется.

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


def micro_config(**overrides: Any) -> AppConfig:
    """Синтетический мир: 2 зоны, дорога 30 мин, спавн и форс-события управляются.

    Шансы дождя и инъекции нулевые — мир рукописен; ``overrides`` замещают
    секции верхнего уровня (``population`` включает спавн, ``weather``/``pricing``
    включают форс-эпизоды — среда в строках метрик становится нетривиальной).
    """

    base: dict = {
        "seed": 7,
        "city": {
            "zones": MICRO_ZONES,
            "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 30.0}],
        },
        "population": {"scale": 0.0},
        "drivers": {"count": 4, "wait_limit_min": 6},
        "weather": {
            "rain_day_chance": 0.0,
            "episode_start_h": [0, 0],
            "episode_duration_h": [1, 1],
        },
        "pricing": {"surge_injection": {"day_chance": 0.0}},
    }
    base.update(overrides)
    return AppConfig.model_validate(base)


# Форс-события мира эквивалентности: дождь с 00:00 на 1 ч и инъекция ×1.5 на
# весь день — колонки среды (is_rain, event, price_multiplier) нетривиальны.
LIVE_WORLD = micro_config(
    population={"scale": 50.0},
    weather={
        "rain_day_chance": 1.0,
        "episode_start_h": [0, 0],
        "episode_duration_h": [1, 1],
    },
    pricing={
        "surge_injection": {
            "day_chance": 1.0,
            "duration_min": [1440, 1440],
            "multiplier": [1.5, 1.5],
        }
    },
)


def _metric_columns(observed: dict) -> tuple:
    """11 колонок наблюдённой строки в порядке SELECT БД (ts — в ISO)."""

    return (
        to_iso(observed["ts"]),
        observed["zone_id"],
        observed["requests"],
        observed["completed"],
        observed["unfulfilled"],
        observed["free_drivers"],
        observed["busy_drivers"],
        observed["avg_wait_min"],
        observed["supply_demand_ratio"],
        observed["is_rain"],
        observed["event"],
        observed["price_multiplier"],
    )


_METRIC_SELECT = (
    "SELECT ts, zone_id, requests, completed, unfulfilled, free_drivers, busy_drivers,"
    " avg_wait_min, supply_demand_ratio, is_rain, event, price_multiplier"
    " FROM zone_metrics_5min ORDER BY ts, zone_id"
)


def test_metrics_observer_matches_db(tmp_path: Path) -> None:
    """Наблюдатель получил дни × 288 × Z строк; содержимое равно строкам БД прогона."""

    cfg = load_config(REFERENCE_CONFIG)
    conn = connect(str(tmp_path / "run.db"))
    init_db(conn)
    observed: list[dict] = []
    Engine(cfg, seed=42).run(1, SqliteTripWriter(conn), on_metrics_row=observed.append)

    zone_count = len(cfg.city.zones)
    assert len(observed) == 288 * zone_count
    # Порядок вызова — по возрастанию zone_id внутри каждого бакета.
    assert [row["zone_id"] for row in observed[:zone_count]] == list(range(zone_count))
    stored = conn.execute(_METRIC_SELECT).fetchall()
    assert len(stored) == len(observed)
    for row, db_row in zip(observed, stored):
        assert _metric_columns(row) == tuple(db_row)


def test_metrics_observer_purity() -> None:
    """Наблюдение метрик не меняет мир: таблицы и Summary прогонов совпадают."""

    cfg = micro_config(population={"scale": 50.0})

    def run(*, collect: bool):
        conn = connect(":memory:")
        init_db(conn)
        rows: list[dict] = []
        summary = Engine(cfg).run(
            1, SqliteTripWriter(conn), on_metrics_row=(rows.append if collect else None)
        )
        return summary, rows, conn

    summary_plain, _, conn_plain = run(collect=False)
    summary_observed, rows, conn_observed = run(collect=True)
    assert rows  # наблюдатель получил строки

    summary_plain = dataclasses.replace(summary_plain, wall_sec=0.0)
    summary_observed = dataclasses.replace(summary_observed, wall_sec=0.0)
    assert summary_plain == summary_observed

    trips_plain = [tuple(row) for row in conn_plain.execute("SELECT * FROM trips ORDER BY trip_id")]
    trips_observed = [
        tuple(row) for row in conn_observed.execute("SELECT * FROM trips ORDER BY trip_id")
    ]
    assert trips_plain and trips_plain == trips_observed
    metrics_plain = [tuple(row) for row in conn_plain.execute(_METRIC_SELECT)]
    metrics_observed = [tuple(row) for row in conn_observed.execute(_METRIC_SELECT)]
    assert metrics_plain == metrics_observed


def test_world_tariff_matches_env_and_independent_of_price_fn() -> None:
    """world_tariff — тариф среды: одинаков у price_fn и без, равен автономной среде, чистое.

    Три прогона одного seed: с подменой тарифа ×3.0 (наблюдатель минут опрашивает
    ``world_tariff`` в самом прогоне), с той же подменой без опроса и базовый без
    price_fn. Автономная среда воспроизводит экзогенный поток — ``SeedSequence
    (seed).spawn(2)`` даёт второй поток среде, ``advance_day`` — по разу на сутки.
    """

    cfg = micro_config(
        population={"scale": 50.0},
        pricing={
            "surge_injection": {
                "day_chance": 1.0,
                "duration_min": [1440, 1440],
                "multiplier": [2.0, 2.0],
            }
        },
    )

    def write_conn() -> object:
        conn = connect(":memory:")
        init_db(conn)
        return conn

    engine_priced = Engine(cfg, price_fn=lambda zone_id, t: 3.0)

    def probe(record) -> None:
        for zone in (0, 1):
            engine_priced.world_tariff(zone, record.t)

    conn_probed = write_conn()
    engine_priced.run(1, SqliteTripWriter(conn_probed), on_minute=probe)
    conn_priced = write_conn()
    Engine(cfg, price_fn=lambda zone_id, t: 3.0).run(1, SqliteTripWriter(conn_priced))
    conn_plain = write_conn()
    engine_plain = Engine(cfg)
    engine_plain.run(1, SqliteTripWriter(conn_plain))

    # Экзогенный поток: второй ребёнок SeedSequence, розыгрыш по разу на сутки.
    _endo_seq, exo_seq = np.random.SeedSequence(cfg.seed).spawn(2)
    env = WorldEnv(cfg, np.random.default_rng(exo_seq))
    env.advance_day(0)

    minutes = [add_minutes(SIM_START_DT, m) for m in range(1440)]
    probed_values: list[float] = []
    for zone in (0, 1):
        for t in minutes:
            expected = env.price_multiplier(zone, t)
            assert engine_priced.world_tariff(zone, t) == expected
            assert engine_plain.world_tariff(zone, t) == expected
            probed_values.append(expected)
    assert any(value > 1.0 for value in probed_values), "инъекция не разыграна"

    # Чтение во время прогона (пробующий наблюдатель) не изменило таблицы.
    trips_a = [tuple(r) for r in conn_probed.execute("SELECT * FROM trips ORDER BY trip_id")]
    trips_b = [tuple(r) for r in conn_priced.execute("SELECT * FROM trips ORDER BY trip_id")]
    assert trips_a and trips_a == trips_b

    # Подмена тарифа видна в заказах; без подмены тариф заказа — тариф среды.
    priced = conn_probed.execute("SELECT price_multiplier FROM trips").fetchall()
    assert all(row[0] == 3.0 for row in priced)
    for origin, created_at, mult in conn_plain.execute(
        "SELECT origin_zone, created_at, price_multiplier FROM trips"
    ):
        assert mult == env.price_multiplier(origin, from_iso(created_at))


def test_live_source_equivalent_to_sqlite(tmp_path: Path) -> None:
    """Окна, точечные метрики, dropoff_counts и build_window равны SqliteRowSource."""

    live = LiveRowSource()
    db_path = tmp_path / "run.db"
    conn = connect(str(db_path))
    init_db(conn)
    Engine(LIVE_WORLD).run(
        1, SqliteTripWriter(conn), on_minute=live.on_minute, on_metrics_row=live.on_metrics_row
    )
    sqlite = SqliteRowSource(str(db_path))

    first, last = live.time_range()
    assert (first, last) == sqlite.time_range()
    first_idx = bucket_index(first)
    last_idx = bucket_index(last)
    stamps = [from_bucket_index(idx) for idx in range(first_idx, last_idx + 1)]

    for zone in (0, 1):
        for ts in stamps:
            live_at, db_at = live.metrics_at(zone, ts), sqlite.metrics_at(zone, ts)
            assert (live_at is None) == (db_at is None)
            assert live_at is None or live_at == db_at
            live_win = live.metrics_window(zone, ts, 36)
            db_win = sqlite.metrics_window(zone, ts, 36)
            assert (live_win is None) == (db_win is None)
            if live_win is not None:
                assert np.array_equal(live_win, db_win)

    # dropoff_counts: сетка (зона, окно, cutoff) — включая cutoff между границами.
    for zone in (0, 1):
        for after_idx in range(first_idx, last_idx, 13):
            for span in (1, 3):
                after = from_bucket_index(after_idx)
                until = from_bucket_index(after_idx + span)
                cutoffs = (
                    after,
                    add_minutes(after, 1),
                    from_bucket_index(after_idx + span // 2),
                    until,
                    add_minutes(until, 2),
                )
                for cutoff in cutoffs:
                    assert live.dropoff_counts(zone, after, until, cutoff) == (
                        sqlite.dropoff_counts(zone, after, until, cutoff)
                    )

    # build_window: сборщик признаков M3 над обоими источниками даёт одно и то же.
    checked = 0
    for zone in (0, 1):
        for ts in stamps[::11]:
            live_window = build_window(live, zone, ts, window_hours=3)
            db_window = build_window(sqlite, zone, ts, window_hours=3)
            assert (live_window is None) == (db_window is None)
            if live_window is not None:
                assert np.array_equal(live_window, db_window)
                checked += 1
    assert checked > 0, "все окна в прогреве — эквивалентность не проверена"


def test_live_source_partial_window_warmup_none() -> None:
    """Неполное окно в прогреве — None (как в M3); границы растут с миром."""

    live = LiveRowSource()
    with pytest.raises(ValueError):  # пустой источник границ не имеет
        live.time_range()

    Engine(micro_config(population={"scale": 50.0})).run(
        40 / 1440, StubWriter(), on_minute=live.on_minute, on_metrics_row=live.on_metrics_row
    )
    assert live.time_range() == (SIM_START_DT, add_minutes(SIM_START_DT, 35))  # 00:00..00:35

    ts_last = add_minutes(SIM_START_DT, 35)
    assert live.metrics_at(0, ts_last) is not None  # последний бакет виден
    assert live.metrics_window(0, ts_last, 8) is not None  # ровно вся история — окно есть
    assert live.metrics_window(0, ts_last, 9) is None  # истории меньше окна — None
    assert build_window(live, 0, ts_last, window_hours=3) is None  # сборщик M3 не строит пример
    full = live.metrics_window(1, ts_last, 8)
    assert full is not None and full.shape == (8, 10)


# --- Группа 4: контур инференса и политики веток ----------------------------


class StubPredictor:
    """Стаб-прогнозист: фиксированные вероятности по зонам + счётчик вызовов."""

    def __init__(self, probs: dict[int, float]) -> None:
        self._probs = dict(probs)
        self.calls: Counter[tuple[int, datetime]] = Counter()

    def predict(self, zone_id: int, ts: datetime) -> float | None:
        self.calls[(zone_id, ts)] += 1
        return self._probs.get(zone_id)


def _run_with_contour(
    cfg: AppConfig,
    predictor: StubPredictor,
    conn: object,
    *,
    days: float = 1,
    model: str = "",
) -> PolicyContour:
    """Прогон мира с контуром — привязка, как её сделает generate_branch (решение 5):
    ``price`` в движок при построении, ``bind`` после, наблюдатели контура в run."""

    contour = PolicyContour(cfg, predictor, LiveRowSource(), model=model)
    engine = Engine(cfg, price_fn=contour.price)
    contour.bind(engine)
    engine.run(
        days,
        SqliteTripWriter(conn),
        on_minute=contour.on_minute,
        on_metrics_row=contour.on_metrics_row,
    )
    return contour


def test_contour_stairs_by_stub() -> None:
    """Стаб-модель: фикс. p по зонам 0.20/0.45/0.65/0.90 → 1.00/1.15/1.30/1.50 (задача 4.1)."""

    cfg = AppConfig.model_validate({"ml": {"train_days": 0}})  # активация с первой минуты
    stub = StubPredictor({0: 0.20, 1: 0.45, 2: 0.65, 3: 0.90, 4: 0.20, 5: 0.45})
    contour = PolicyContour(cfg, stub, LiveRowSource())
    contour.bind(Engine(cfg))
    t = add_minutes(SIM_START_DT, 7)
    expected = {0: 1.00, 1: 1.15, 2: 1.30, 3: 1.50, 4: 1.00, 5: 1.15}
    for zone in range(6):
        # мир без инъекций — тариф среды 1.0, итог равен множителю ступени
        assert contour.price(zone, t) == expected[zone]


def test_contour_multiplier_cached_per_bucket() -> None:
    """Кэш на бакет: первая и последняя минуты бакета — одно вычисление стаба."""

    cfg = micro_config(ml={"train_days": 0})
    stub = StubPredictor({0: 0.90, 1: 0.90})
    contour = PolicyContour(cfg, stub, LiveRowSource())
    contour.bind(Engine(cfg))
    first, last = add_minutes(SIM_START_DT, 1), add_minutes(SIM_START_DT, 4)  # бакет 00:00
    assert contour.price(0, first) == contour.price(0, last)
    assert sum(n for (zone, _ts), n in stub.calls.items() if zone == 0) == 1
    contour.price(0, add_minutes(SIM_START_DT, 6))  # следующий бакет — новое вычисление
    assert sum(n for (zone, _ts), n in stub.calls.items() if zone == 0) == 2
    contour.price(1, first)  # зоны считаются независимо
    assert sum(n for (zone, _ts), n in stub.calls.items() if zone == 1) == 1


def test_contour_fire_diagnostics() -> None:
    """Доля срабатываний: 1.0 при p выше всех порогов, 0.0 при p ниже всех (задача 4.1)."""

    cfg = micro_config(ml={"train_days": 0})
    for p, fired in ((0.90, True), (0.20, False)):
        stub = StubPredictor({0: p, 1: p})
        contour = PolicyContour(cfg, stub, LiveRowSource())
        contour.bind(Engine(cfg))
        for bucket in range(2):  # 2 бакета × 2 зоны = 4 слота
            for zone in (0, 1):
                contour.price(zone, add_minutes(SIM_START_DT, bucket * 5 + 2))
        assert contour.fire_slots == 4
        assert contour.fire_count == (4 if fired else 0)
        assert contour.fire_rate == (1.0 if fired else 0.0)


def test_network_predictor_missing_artifact(tmp_path: Path) -> None:
    """Пустой каталог артефактов — FileNotFoundError с именем семейства (задача 4.2)."""

    with pytest.raises(FileNotFoundError, match="mlp"):
        NetworkPredictor(LiveRowSource(), AppConfig(), family="mlp", artifacts_root=tmp_path)


def _ml_cfg() -> MlCfg:
    """Тестовые гиперпараметры: роли 2/1/1 дня, окно 1 ч, малая сеть (приём test_ml_infer)."""

    return MlCfg(
        train_days=2,
        val_days=1,
        test_days=1,
        window_hours=1,
        mlp_hidden=[16, 128],
        lstm_hidden=8,
        dropout=0.0,
        lr=0.01,
        batch_size=128,
        max_epochs=4,
        patience=4,
        seed=42,
    )


def _fill_pattern_db(db_path: Path) -> None:
    """Мини-эпоха с закономерностью: 3 дня × 288 бакетов, зоны 0/1 (приём test_ml_infer)."""

    conn = connect(str(db_path))
    init_db(conn)
    rows: list[dict] = []
    t = SIM_START_DT
    end = add_minutes(SIM_START_DT, 3 * 1440 - 5)
    while t <= end:
        j = int((t - SIM_START_DT).total_seconds()) // 300
        for zone_id in (0, 1):
            requests = 3 + int(round(3.0 * np.sin(2.0 * np.pi * (j % 96) / 96.0))) + zone_id
            free_drivers = 4 + (j % 3) + zone_id
            unfulfilled = 1 if requests >= 6 else 0
            rows.append(
                {
                    "ts": t,
                    "zone_id": zone_id,
                    "requests": requests,
                    "completed": requests - unfulfilled,
                    "unfulfilled": unfulfilled,
                    "free_drivers": free_drivers,
                    "busy_drivers": j % 4,
                    "avg_wait_min": (j % 5) * 0.5,
                    "supply_demand_ratio": free_drivers / max(requests, 1),
                    "is_rain": 1 if j % 29 == 0 else 0,
                    "event": "",
                    "price_multiplier": 1.0,
                }
            )
        t = add_minutes(t, 5)
    insert_zone_metrics(conn, rows)
    conn.close()


def test_network_predictor_smoke(tmp_path: Path) -> None:
    """Мини-артефакт M3: deficit_prob ∈ [0, 1], контур применяет ступень (задача 4.2)."""

    pattern_db = tmp_path / "pattern.db"
    _fill_pattern_db(pattern_db)
    ml_cfg = _ml_cfg()
    data = prepare_training_data(pattern_db, ml_cfg, zones=list(range(6)), start=SIM_START_DT)
    result = train_model("mlp", data, ml_cfg)
    version = model_version_for("mlp", today=date(2026, 1, 1))
    save_model_artifacts(
        tmp_path / "artifacts" / version,
        model_version=version,
        arch="mlp",
        model=result.model,
        data=data,
        result=result,
        ml_cfg=ml_cfg,
    )

    # Живой мир — эталонный (6 зон, как при обучении: ёмкость эмбеддинга),
    # окно активации с первой минуты (train_days = 0).
    cfg = load_config(REFERENCE_CONFIG)
    cfg = cfg.model_copy(update={"ml": cfg.ml.model_copy(update={"train_days": 0})})

    live = LiveRowSource()
    predictor = NetworkPredictor(live, cfg, family="mlp", artifacts_root=tmp_path / "artifacts")
    contour = PolicyContour(cfg, predictor, live, model="mlp")
    engine = Engine(cfg, seed=42, price_fn=contour.price)
    contour.bind(engine)
    conn = connect(str(tmp_path / "run.db"))
    init_db(conn)
    engine.run(
        1,
        SqliteTripWriter(conn),
        on_minute=contour.on_minute,
        on_metrics_row=contour.on_metrics_row,
    )

    assert contour.fire_slots >= 288 * 6  # каждый бакет-зону суток контур посчитал
    assert 0.0 <= contour.fire_rate <= 1.0
    closed_ts = add_minutes(SIM_START_DT, 595)  # закрытый бакет для момента 10:00
    p = predictor.predict(0, closed_ts)
    assert p is not None and 0.0 <= p <= 1.0
    t = add_minutes(SIM_START_DT, 600)
    # Сдвиг контура — из конфига (после калибровки m6-worlds в эталоне ≠ 0):
    # ожидание считается с тем же δ, что применяет контур.
    expected = compose_tariff(
        engine.world_tariff(0, t),
        stair_multiplier(p, model_shift("mlp", cfg.policy.shift), cfg.policy.stairs),
        cfg.policy.total_cap,
    )
    assert contour.price(0, t) == expected  # ступень применена к тому же прогнозу сети


def _bucket_row(idx: int, zone: int) -> dict:
    """Рукописная строка метрик бакета ``idx`` (для ручного питания источника)."""

    return {
        "ts": from_bucket_index(idx),
        "zone_id": zone,
        "requests": 4 + zone,
        "completed": 2,
        "unfulfilled": 0,
        "free_drivers": 5,
        "busy_drivers": 3,
        "avg_wait_min": None,
        "supply_demand_ratio": 1.25,
        "is_rain": 0,
        "event": "",
        "price_multiplier": 1.0,
    }


def _minute_record(minute: int, dest: int, dropoff_bucket: int) -> MinuteRecord:
    """Запись минуты с одним назначением: минута назначения, зона и бакет высадки."""

    t = add_minutes(SIM_START_DT, minute)
    assignment = AssignEvent(
        internal_id=minute,
        driver_id=0,
        driver_zone=0,
        origin=0,
        dest=dest,
        pickup_at=t,
        dropoff_at=from_bucket_index(dropoff_bucket),
    )
    return MinuteRecord(
        t=t,
        is_rain=False,
        price_mult=(1.0, 1.0),
        free_by_zone=(1, 1),
        spawns=(),
        assignments=(assignment,),
        dropoffs=(),
        expiries=(),
    )


def test_live_source_future_records_do_not_change_windows() -> None:
    """Записи после ts не меняют окна и dropoff_counts на ts (анти-лик, задача 4.3)."""

    live = LiveRowSource()
    ts = from_bucket_index(23)  # момент прогноза: окно 12 бакетов = 12..23
    # Прошлое: бакеты 0..23 по двум зонам + назначения с высадками в бакетах 22/23.
    for idx in range(24):
        for zone in (0, 1):
            live.on_metrics_row(_bucket_row(idx, zone))
    live.on_minute(_minute_record(5 * 20, dest=0, dropoff_bucket=22))
    live.on_minute(_minute_record(5 * 21, dest=1, dropoff_bucket=23))
    windows = {zone: live.metrics_window(zone, ts, 12) for zone in (0, 1)}
    cutoffs = (ts, add_minutes(ts, -1), from_bucket_index(20))
    counts = {
        (zone, cutoff): live.dropoff_counts(zone, from_bucket_index(15), ts, cutoff)
        for zone in (0, 1)
        for cutoff in cutoffs
    }

    # Будущее: новые бакеты + назначения ПОСЛЕ ts, в т.ч. с высадкой ВНУТРИ
    # старого окна (бакет 20) — по моменту назначения они не должны считаться.
    for idx in range(24, 30):
        for zone in (0, 1):
            live.on_metrics_row(_bucket_row(idx, zone))
    live.on_minute(_minute_record(5 * 30, dest=0, dropoff_bucket=20))
    live.on_minute(_minute_record(5 * 24, dest=1, dropoff_bucket=26))

    for zone in (0, 1):
        assert np.array_equal(live.metrics_window(zone, ts, 12), windows[zone])
        for cutoff in cutoffs:
            assert (
                live.dropoff_counts(zone, from_bucket_index(15), ts, cutoff)
                == counts[(zone, cutoff)]
            )
    # Новые назначения реально поданы: с будущим cutoff окно (15, 23] считает
    # и прошлое назначение бакета 22, и новое назначение бакета 20 — строго
    # больше замороженного счёта, который их обеих исключал по моменту ts.
    assert (
        live.dropoff_counts(0, from_bucket_index(15), ts, add_minutes(SIM_START_DT, 200))
        > counts[(0, ts)]
    )


def test_contour_regeneration_identical(tmp_path: Path) -> None:
    """Повторная генерация ветки со стаб-политикой — побайтово те же таблицы (задача 4.3)."""

    cfg = micro_config(population={"scale": 50.0}, ml={"train_days": 0})
    conns = []
    for name in ("a.db", "b.db"):
        conn = connect(str(tmp_path / name))
        init_db(conn)
        _run_with_contour(cfg, StubPredictor({0: 0.65, 1: 0.65}), conn)
        conns.append(conn)

    trips = [
        [tuple(row) for row in conn.execute("SELECT * FROM trips ORDER BY trip_id")]
        for conn in conns
    ]
    assert trips[0] and trips[0] == trips[1]
    metrics = [
        [tuple(row) for row in conn.execute("SELECT * FROM zone_metrics_5min ORDER BY ts, zone_id")]
        for conn in conns
    ]
    assert metrics[0] == metrics[1]


def test_activation_prefix_equals_a_base(tmp_path: Path) -> None:
    """До границы активации таблицы равны a-base, после — различаются (задача 4.4)."""

    cfg = micro_config(population={"scale": 50.0}, ml={"train_days": 1})  # активация день 2
    boundary_iso = "2026-11-03 00:00:00"

    conn_a = connect(str(tmp_path / "a.db"))
    init_db(conn_a)
    Engine(cfg).run(2, SqliteTripWriter(conn_a))

    conn_b = connect(str(tmp_path / "b.db"))
    init_db(conn_b)
    _run_with_contour(cfg, StubPredictor({0: 0.90, 1: 0.90}), conn_b, days=2)

    metrics_a = conn_a.execute("SELECT * FROM zone_metrics_5min ORDER BY ts, zone_id").fetchall()
    metrics_b = conn_b.execute("SELECT * FROM zone_metrics_5min ORDER BY ts, zone_id").fetchall()
    prefix_a = [row for row in metrics_a if row[0] < boundary_iso]
    prefix_b = [row for row in metrics_b if row[0] < boundary_iso]
    assert prefix_a and prefix_a == prefix_b  # дни 1 — побитово равны
    suffix_a = [row for row in metrics_a if row[0] >= boundary_iso]
    suffix_b = [row for row in metrics_b if row[0] >= boundary_iso]
    assert suffix_a != suffix_b  # день 2 — политика меняет тариф и мир

    trips_a = conn_a.execute("SELECT * FROM trips ORDER BY trip_id").fetchall()
    trips_b = conn_b.execute("SELECT * FROM trips ORDER BY trip_id").fetchall()
    prefix_trips_a = [row for row in trips_a if row[1] < boundary_iso]
    prefix_trips_b = [row for row in trips_b if row[1] < boundary_iso]
    assert prefix_trips_a and prefix_trips_a == prefix_trips_b
    assert trips_a != trips_b


def test_negative_shift_keeps_upper_stairs_reachable() -> None:
    """Отрицательный δ растягивает шкалу, а не срезает верх: p = 1 → верхняя ступень.

    Раньше ``p + δ`` при δ ≈ −0.44 упиралось в 0.56, и ступени ×1.3/×1.5 были
    недостижимы. Растяжение сохраняет порядок вероятностей и порог первой ступени.
    """

    delta = -0.44
    assert stair_multiplier(1.0, delta, DEFAULT_STAIRS) == 1.50
    assert stair_multiplier(0.2, delta, DEFAULT_STAIRS) == 1.00
    probs = [0.5, 0.7, 0.8, 0.95, 0.99]
    mults = [stair_multiplier(p, delta, DEFAULT_STAIRS) for p in probs]
    assert mults == sorted(mults)
    assert {1.15, 1.30, 1.50} <= set(mults)
