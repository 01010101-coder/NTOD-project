"""Тесты модели спроса change'а m2-demand (строка №2 §7 плана M2).

Закрывают сценарии спеки simulator-demand: «λ без факторов равна base_rate»,
«Все множители формулы входят в λ», «Дождь повышает спрос на
rain.demand_mult», «Пятничный вечер применяется только к заданным архетипам»,
«Вне окна события множитель события не применяется», «Удвоение тарифа при
α=1 режет поток вдвое», «α=0 отключает эластичность», «Детерминизм при
фиксированном seed», «Разыгрывание всегда, даже при λ=0», «Число заказов
равно сумме пуассоновских розыгрышей», «Поток тяготеет к привлекательным
целям», «Затухание по dest_decay_min», «Нулевой парк — все заказы unfulfilled
ровно через лимит», «Назначение спасает от истечения». Точные равенства — там,
где статистика не нужна (design.md, решение 6); статистические проверки — с
фиксированными seed, большими выборками и широкими коридорами. Магические
числа тестов — параметры конфигов, не кода.
"""

import dataclasses
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from common.config import AppConfig, load_config
from simulator.city import CitySpec, ZoneInfo, build_city
from simulator.demand import (
    TripRequest,
    _is_expired,
    _pick_destination,
    lambda_rate,
    spawn_requests,
)

# Эталонный конфиг ищется относительно файла теста, не от cwd (как в M1-тестах).
REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"

# Нейтральная сим-минута: понедельник (будни) 16:00, профиль residential = 1.0.
NO_FACTOR_MINUTE = datetime(2026, 11, 2, 16, 0)


def _mutated_cfg(mutate: Callable[[dict], None]) -> AppConfig:
    """Конфиг с точечной правкой payload — источник синтетических конфигов."""
    payload = AppConfig().model_dump()
    mutate(payload)
    return AppConfig.model_validate(payload)


def _rate(city: CitySpec, zone: ZoneInfo, t: datetime, cfg: AppConfig, **overrides) -> float:
    """``lambda_rate`` с нейтральными факторами: без дождя, события и тарифа."""
    params = {"is_rain": False, "event_mult": 1.0, "price_mult": 1.0, **overrides}
    return lambda_rate(city, zone, t, cfg=cfg, **params)


def _spawn(city: CitySpec, t: datetime, cfg: AppConfig, seed: int, pm=None) -> list[TripRequest]:
    """Спавн одной минуты с фиксированным seed и нейтральными факторами."""
    mults = np.ones(len(city.zones)) if pm is None else np.asarray(pm, dtype=float)
    return spawn_requests(
        np.random.default_rng(seed),
        city,
        t,
        is_rain=False,
        event_mult=1.0,
        price_mult_by_zone=mults,
        cfg=cfg,
    )


def _origin_only_cfg(origin_id: int = 0, pull: float = 20000.0) -> AppConfig:
    """Эталонный город, порождающий пассажиров только из одной зоны."""

    def zero_other_origins(payload: dict) -> None:
        for row in payload["city"]["zones"]:
            row["attract_origin"] = pull if row["id"] == origin_id else 0.0

    return _mutated_cfg(zero_other_origins)


def _small_city_cfg(origin_pull: float = 0.0, decay: int = 15) -> AppConfig:
    """Три жилые зоны 0—(5 мин)—1—(10 мин)—2, профиль весь 1.0.

    path_min: 0→1 = 5 (ближняя цель), 0→2 = 15 (дальняя); attract_dest у обеих
    целей одинаковый (1.0) — управляемая арена для тестов эластичности и
    затухания. ``origin_pull > 0`` делает единственным источником зону 0
    (λ остальных зон = 0); ``decay`` задаёт demand.dest_decay_min.
    """

    def replace_city(payload: dict) -> None:
        payload["city"]["zones"] = [
            {
                "id": zone_id,
                "name": f"zone-{zone_id}",
                "archetype": "residential",
                "attract_origin": 1.0,
                "attract_dest": 1.0,
                "map": {"x": 100.0 * zone_id, "y": 0.0},
            }
            for zone_id in range(3)
        ]
        payload["city"]["roads"] = [
            {"from_zone": 0, "to_zone": 1, "minutes": 5.0},
            {"from_zone": 1, "to_zone": 2, "minutes": 10.0},
        ]
        payload["demand"]["profiles"]["residential"] = {
            "weekday": [1.0] * 24,
            "weekend": [1.0] * 24,
        }
        payload["demand"]["dest_decay_min"] = decay
        if origin_pull:
            for row in payload["city"]["zones"]:
                row["attract_origin"] = origin_pull if row["id"] == 0 else 0.0

    return _mutated_cfg(replace_city)


# --- λ: базовая интенсивность и мультипликативность (задача 3.1) --------------


@pytest.mark.parametrize(
    "cfg", [load_config(REFERENCE_CONFIG), AppConfig()], ids=["reference", "defaults"]
)
def test_lambda_base_rate_exact(cfg: AppConfig) -> None:
    """λ без факторов равна base_rate точно (спека: сценарий 1; решение 6)."""
    city = build_city(cfg)
    zone = city.zones[0]  # residential, attract_origin = 1.0
    assert _rate(city, zone, NO_FACTOR_MINUTE, cfg) == cfg.demand.base_rate == 2.0


def test_lambda_each_multiplier_multiplicative() -> None:
    """Каждый множитель формулы §3.3 мультипликативен при поочерёдной активации."""
    reference = load_config(REFERENCE_CONFIG)
    city = build_city(reference)
    zone0 = city.zones[0]
    event = reference.events.friday_evening

    # профиль часа: понедельник 08:00, residential weekday[8]
    hour8 = datetime(2026, 11, 2, 8, 0)
    expected = reference.demand.base_rate * city.profiles["residential"].weekday[8]
    assert _rate(city, zone0, hour8, reference) == pytest.approx(expected)

    # attract_origin: зона с весом источника 1.5 при нейтральном профиле
    pull_cfg = _mutated_cfg(lambda p: p["city"]["zones"][0].update({"attract_origin": 1.5}))
    pull_city = build_city(pull_cfg)
    expected = pull_cfg.demand.base_rate * 1.5
    assert _rate(pull_city, pull_city.zones[0], NO_FACTOR_MINUTE, pull_cfg) == pytest.approx(
        expected
    )

    # population.scale: масштаб всего города
    scale_cfg = _mutated_cfg(lambda p: p["population"].update({"scale": 2.0}))
    scale_city = build_city(scale_cfg)
    expected = scale_cfg.demand.base_rate * scale_cfg.population.scale
    assert _rate(scale_city, scale_city.zones[0], NO_FACTOR_MINUTE, scale_cfg) == pytest.approx(
        expected
    )

    # дождь: is_rain=True × weather.demand_mult
    expected = reference.demand.base_rate * reference.weather.demand_mult
    assert _rate(city, zone0, NO_FACTOR_MINUTE, reference, is_rain=True) == pytest.approx(expected)

    # событие: пятница 18:00, business в archetypes → × event_mult
    friday18 = datetime(2026, 11, 6, 18, 0)
    business = city.zones[2]
    expected = (
        reference.demand.base_rate
        * city.profiles["business"].weekday[18]
        * business.attract_origin
        * event.demand_mult
    )
    assert _rate(
        city, business, friday18, reference, event_mult=event.demand_mult
    ) == pytest.approx(expected)

    # тариф: price_mult = 2.0 при α = 1 → × 2.0**(-1)
    expected = reference.demand.base_rate * 2.0 ** (-reference.pricing.elasticity_alpha)
    assert _rate(city, zone0, NO_FACTOR_MINUTE, reference, price_mult=2.0) == pytest.approx(
        expected
    )


def test_lambda_rain_ratio_matches_config() -> None:
    """Отношение λ(дождь)/λ(сухо) равно weather.demand_mult (спека: сценарий 3)."""
    cfg = load_config(REFERENCE_CONFIG)
    city = build_city(cfg)
    zone = city.zones[0]
    for t in (NO_FACTOR_MINUTE, datetime(2026, 11, 2, 8, 0)):
        dry = _rate(city, zone, t, cfg)
        wet = _rate(city, zone, t, cfg, is_rain=True)
        assert wet / dry == pytest.approx(cfg.weather.demand_mult)


def test_lambda_alpha_zero_disables_elasticity() -> None:
    """α = 0: множитель m**(-0) = 1, λ не зависит от тарифа (спека: сценарий 5)."""
    cfg = _mutated_cfg(lambda p: p["pricing"].update({"elasticity_alpha": 0.0}))
    city = build_city(cfg)
    zone = city.zones[0]
    neutral = _rate(city, zone, NO_FACTOR_MINUTE, cfg)
    assert _rate(city, zone, NO_FACTOR_MINUTE, cfg, price_mult=3.0) == neutral


# --- λ: пятничный вечер (задача 3.1, решение 3 design.md) ---------------------


def test_lambda_friday_evening_only_config_archetypes() -> None:
    """Множитель события — только архетипам из events.friday_evening.archetypes."""
    cfg = load_config(REFERENCE_CONFIG)
    city = build_city(cfg)
    event = cfg.events.friday_evening
    friday18 = datetime(2026, 11, 6, 18, 0)
    for zone in city.zones:
        with_event = _rate(city, zone, friday18, cfg, event_mult=event.demand_mult)
        without = _rate(city, zone, friday18, cfg)
        if zone.archetype in event.archetypes:
            assert with_event == pytest.approx(without * event.demand_mult), zone
        else:
            assert with_event == without, zone

    # список архетипов читается из конфига: замена на residential меняет зоны-цели
    flipped = _mutated_cfg(
        lambda p: p["events"]["friday_evening"].update({"archetypes": ["residential"]})
    )
    flipped_city = build_city(flipped)
    flipped_event = flipped.events.friday_evening
    with_event = _rate(
        flipped_city,
        flipped_city.zones[0],
        friday18,
        flipped,
        event_mult=flipped_event.demand_mult,
    )
    without = _rate(flipped_city, flipped_city.zones[0], friday18, flipped)
    assert with_event == pytest.approx(without * flipped_event.demand_mult)


def test_lambda_friday_evening_window_bounds() -> None:
    """Окно события включительно по дням и часам; вне окна множителя нет."""
    cfg = load_config(REFERENCE_CONFIG)
    city = build_city(cfg)
    event = cfg.events.friday_evening
    business = city.zones[2]
    assert event.days == [4, 5] and event.hours == [17, 23]
    applied = (
        datetime(2026, 11, 6, 17, 0),  # пятница, левая граница часов
        datetime(2026, 11, 6, 23, 0),  # пятница, правая граница часов
        datetime(2026, 11, 7, 18, 0),  # суббота (день 5 в days)
    )
    skipped = (
        datetime(2026, 11, 6, 16, 0),  # пятница, час ниже окна
        datetime(2026, 11, 2, 18, 0),  # понедельник, день вне days
        datetime(2026, 11, 5, 18, 0),  # четверг, день вне days
    )
    for t in applied:
        with_event = _rate(city, business, t, cfg, event_mult=event.demand_mult)
        without = _rate(city, business, t, cfg)
        assert with_event == pytest.approx(without * event.demand_mult), t
    for t in skipped:
        assert _rate(city, business, t, cfg, event_mult=event.demand_mult) == _rate(
            city, business, t, cfg
        ), t


# --- эластичность спроса по тарифу (задача 3.2) --------------------------------


def _total_spawned(cfg: AppConfig, price_mult: float, minutes: int, seed: int) -> int:
    """Суммарный объём заказов за серию из ``minutes`` минут с фиксированным seed."""
    city = build_city(cfg)
    mults = np.full(len(city.zones), price_mult)
    rng = np.random.default_rng(seed)
    total = 0
    for i in range(minutes):
        t = datetime(2026, 11, 2, 8, 0) + timedelta(minutes=i)
        total += len(
            spawn_requests(
                rng, city, t, is_rain=False, event_mult=1.0, price_mult_by_zone=mults, cfg=cfg
            )
        )
    return total


def test_elasticity_halves_flow() -> None:
    """Удвоение тарифа при α=1 режет поток ~вдвое (коридор [0.4, 0.6])."""
    cfg = _small_city_cfg()
    assert cfg.pricing.elasticity_alpha == 1.0
    minutes = 2000
    cheap = _total_spawned(cfg, 1.0, minutes, seed=2026)
    dear = _total_spawned(cfg, 2.0, minutes, seed=2026)
    assert cheap > 0 and dear > 0
    ratio = dear / cheap
    assert 0.4 <= ratio <= 0.6, (cheap, dear, ratio)


# --- выбор назначения P(B|A) (задача 3.3) ---------------------------------------


def _dest_shares(orders: list[TripRequest]) -> dict[int, float]:
    """Эмпирические доли назначений заказов одной серии спавна."""
    counter = Counter(order.dest for order in orders)
    total = sum(counter.values())
    return {dest: count / total for dest, count in counter.items()}


def test_dest_attraction_weights() -> None:
    """Доли назначений тяготеют к нормированным весам attract_dest × затухание."""
    cfg = _origin_only_cfg()
    city = build_city(cfg)
    orders = _spawn(city, datetime(2026, 11, 2, 12, 0), cfg, seed=11)
    assert len(orders) >= 20000
    assert all(order.dest != order.origin for order in orders)

    weights = {
        zone.id: zone.attract_dest
        * float(np.exp(-city.path_min[0, zone.id] / cfg.demand.dest_decay_min))
        for zone in city.zones
        if zone.id != 0
    }
    total = sum(weights.values())
    shares = _dest_shares(orders)
    for dest, weight in weights.items():
        assert shares[dest] == pytest.approx(weight / total, abs=0.02), dest

    # привлекательная близкая зона (center, attract 1.8) строго выше
    # непривлекательной дальней (sleep-2, attract 0.6)
    assert shares[2] > shares[1]


def test_dest_decay_convergence() -> None:
    """Увеличение dest_decay_min сближает доли дальней и ближней целей к весам."""
    ratios: dict[int, float] = {}
    for decay in (15, 60):
        cfg = _small_city_cfg(origin_pull=20000.0, decay=decay)
        city = build_city(cfg)
        orders = _spawn(city, datetime(2026, 11, 2, 12, 0), cfg, seed=13)
        assert len(orders) >= 20000
        shares = _dest_shares(orders)
        assert shares[1] > shares[2]  # ближняя цель получает больше дальней
        near_weight = float(np.exp(-5.0 / decay))
        far_weight = float(np.exp(-15.0 / decay))
        ratios[decay] = shares[1] / shares[2]
        # эмпирическое отношение долей сходится к отношению весов
        assert ratios[decay] == pytest.approx(near_weight / far_weight, rel=0.05), decay
    assert ratios[60] < ratios[15]  # увеличение decay ослабило дальнее отталкивание


# --- спавн: порядок разыгрываний и поля заказа (задача 3.4) ---------------------


def test_spawn_deterministic() -> None:
    """Один seed и параметры — попарно равные списки заказов (спека: сценарий 1)."""
    cfg = load_config(REFERENCE_CONFIG)
    city = build_city(cfg)
    t = datetime(2026, 11, 2, 12, 0)
    first = _spawn(city, t, cfg, seed=7)
    second = _spawn(city, t, cfg, seed=7)
    assert len(first) == len(second) > 0
    assert first == second  # frozen-dataclass: равенство по всем полям


def test_spawn_zero_rate_advances_rng() -> None:
    """λ=0: розыгрыш выполняется для всех зон, поток rng продвинут одинаково."""
    cfg = load_config(REFERENCE_CONFIG)
    city = build_city(cfg)
    zero_cfg = _mutated_cfg(lambda p: p["population"].update({"scale": 0.0}))
    zero_city = build_city(zero_cfg)
    t = datetime(2026, 11, 2, 12, 0)

    assert _spawn(zero_city, t, zero_cfg, seed=7) == []

    # «до» пустого прогона и «после» него на одинаково инициализированных rng
    before = _spawn(city, t, cfg, seed=7)
    rng_after = np.random.default_rng(7)
    empty = spawn_requests(
        rng_after,
        zero_city,
        t,
        is_rain=False,
        event_mult=1.0,
        price_mult_by_zone=np.ones(len(zero_city.zones)),
        cfg=zero_cfg,
    )
    assert empty == []
    after = spawn_requests(
        rng_after,
        city,
        t,
        is_rain=False,
        event_mult=1.0,
        price_mult_by_zone=np.ones(len(city.zones)),
        cfg=cfg,
    )
    assert after == before


def test_spawn_count_matches_poisson() -> None:
    """Заказы по зонам = пуассоновские разыгрыши в порядке решения 2 design.md."""
    cfg = load_config(REFERENCE_CONFIG)
    city = build_city(cfg)
    t = datetime(2026, 11, 2, 12, 0)

    # повторяем документированный порядок: пуассон зоны → розыгрыши назначений
    replay = np.random.default_rng(7)
    expected: Counter[int] = Counter()
    for zone in city.zones:
        rate = lambda_rate(city, zone, t, is_rain=False, event_mult=1.0, price_mult=1.0, cfg=cfg)
        count = int(replay.poisson(rate))
        for _ in range(count):
            _pick_destination(replay, city, zone, cfg)
        expected[zone.id] = count

    orders = _spawn(city, t, cfg, seed=7)
    assert Counter(order.origin for order in orders) == expected


def test_spawn_order_fields() -> None:
    """created_at = t и price_multiplier = множителю зоны происхождения (§3.4)."""
    cfg = load_config(REFERENCE_CONFIG)
    city = build_city(cfg)
    t = datetime(2026, 11, 2, 12, 0)
    mults = [0.8, 1.1, 1.3, 0.9, 1.2, 1.0]
    orders = _spawn(city, t, cfg, seed=7, pm=mults)
    assert orders
    for order in orders:
        assert order.created_at == t
        assert order.price_multiplier == mults[order.origin]


# --- лимит ожидания заказа (задача 3.5, решение 5 design.md, §10.3) -------------


def test_wait_limit_zero_fleet_exact_expiry() -> None:
    """Нулевой парк: заказы истекают ровно в created_at + wait_limit_min."""
    cfg = load_config(REFERENCE_CONFIG)
    assert cfg.drivers.wait_limit_min == 6
    city = build_city(cfg)
    orders: list[TripRequest] = []
    for i in range(12):
        orders.extend(_spawn(city, datetime(2026, 11, 2, 8, 0) + timedelta(minutes=i), cfg, seed=5))
    assert orders
    for order in orders:
        for late in range(6):  # +0 … +5 — ещё не истёк
            moment = order.created_at + timedelta(minutes=late)
            assert not _is_expired(order, moment, cfg, assigned=False), (order, moment)
        limit = order.created_at + timedelta(minutes=cfg.drivers.wait_limit_min)
        assert _is_expired(order, limit, cfg, assigned=False), order


def test_wait_limit_assignment_rescues() -> None:
    """Назначение спасает: предикат помечает истёкшим только неназначенных."""
    cfg = load_config(REFERENCE_CONFIG)
    order = TripRequest(datetime(2026, 11, 2, 8, 0), 0, 2, 1.0)
    for late in (6, 7, 60):  # даже при подаче дольше лимита
        moment = order.created_at + timedelta(minutes=late)
        assert not _is_expired(order, moment, cfg, assigned=True)
        assert _is_expired(order, moment, cfg, assigned=False)


# --- контракт TripRequest (приложение Б; проверка задачи 1.1) -------------------


def test_trip_request_frozen_fields() -> None:
    """Поля — по порядку приложения Б; объект неизменяем (frozen)."""
    order = TripRequest(datetime(2026, 11, 2, 8, 0), 0, 2, 1.3)
    assert [f.name for f in dataclasses.fields(TripRequest)] == [
        "created_at",
        "origin",
        "dest",
        "price_multiplier",
    ]
    assert order.price_multiplier == 1.3
    with pytest.raises(dataclasses.FrozenInstanceError):
        order.origin = 1  # type: ignore[misc]
