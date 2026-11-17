"""Тесты среды мира change'а m2-env (строка №4 §7 плана M2).

Покрывают сценарии спеки simulator-env на уровне публичного API ``WorldEnv``
(proposal.md, вопрос 1): «Форс-дождь (шанс 1.0) — спрос выше и поездки
дольше», «День без дождя», «Пятница даёт горб в заданных архетипах», «Вне
дней, часов или архетипов события нет», «Приоритет «пятничного вечера» над
инъекцией», «Форс-инъекция — price_multiplier > 1 в заданном окне», «День без
инъекции», «Дождь и «пятничный вечер» тариф не меняют», «Детерминизм эпизодов
по seed», «Запросы не сдвигают поток rng». Композитные проверки строки №4 §7
(множители к λ и длительностям поездок, колонка ``price_multiplier``
агрегата) завершаются в тестах m2-demand/m2-drivers/m2-loop — здесь среда
проверяется по её API и каноническим значениям приложения А (design.md,
решение 10). Форс-режимы задаются правкой значений конфига, без новых
параметров API; окна эпизодов находятся по наблюдаемому состоянию
(proposal.md, вопрос 2). Change ``world-demand-shocks`` добавил сценарии
«Среднее множителя около 1», «Персистентность компоненты», «Детерминизм шоков
по seed», «Чтение не тратит rng» и «Шоки не сдвигают дождь и инъекции».
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from common.config import AppConfig, load_config
from simulator.city import ZoneInfo, build_city
from simulator.env import WorldEnv

# Эталонный конфиг ищется относительно файла теста, не от cwd (как в M1-тестах).
REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"

# Канонический seed мира — поле seed эталонного конфига (приложение А).
SEED = 42

# Seed приоритетного теста: инъекция пятницы попадает в вечерний час
# business-зоны; пересечение проверяется ассертами самого теста.
PRIORITY_SEED = 2

# Сим-даты эталонного старта «2026-11-02» (понедельник, нумерация Mon=0).
DAY0 = datetime(2026, 11, 2)
DAY1 = DAY0 + timedelta(days=1)
FRIDAY = DAY0 + timedelta(days=4)
SATURDAY = DAY0 + timedelta(days=5)

# Длина скана эпизодов в минутах: самый поздний конец интервала — эпизод
# дождя со стартом 22:00 и длительностью 3 ч (25:00) и окно инъекции с
# минутой старта 23:59 и длительностью 60 мин (24:59); скан покрывает
# интервалы целиком, включая перешедшие через полночь.
SCAN_MINUTES = 25 * 60


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    """Непрерывные участки истинных значений в минутных флагах: [начало, конец)."""

    spans: list[tuple[int, int]] = []
    lo: int | None = None
    for minute, flag in enumerate([*flags, False]):
        if flag and lo is None:
            lo = minute
        elif not flag and lo is not None:
            spans.append((lo, minute))
            lo = None
    return spans


def _forced_config() -> AppConfig:
    """Эталонный конфиг с форс-режимами: дождь и инъекция каждый день (шанс 1.0)."""

    cfg = load_config(REFERENCE_CONFIG)
    cfg.weather.rain_day_chance = 1.0
    cfg.pricing.surge_injection.day_chance = 1.0
    return cfg


def _observable(env: WorldEnv, zones: tuple[ZoneInfo, ...], day_start: datetime, minutes: int):
    """Наблюдаемое состояние среды: флаги дождя, тариф и метки по зонам и минутам.

    Возвращает кортеж ``(флаги дождя, сетка price_multiplier, сетка event_label)``
    для сравнения состояний двух сред без обращения к приватным полям.
    """

    stamps = [day_start + timedelta(minutes=m) for m in range(minutes)]
    rain = tuple(env.is_rain(t) for t in stamps)
    prices = tuple(tuple(env.price_multiplier(zone.id, t) for t in stamps) for zone in zones)
    labels = tuple(tuple(env.event_label(zone, t) for t in stamps) for zone in zones)
    return rain, prices, labels


# --- форс-дождь и день без дождя (спека «Дождь», задача 4.1) -----------------


def test_forced_rain_episode_window_and_bounds() -> None:
    """Форс-дождь (шанс 1.0): эпизод 1–3 ч со стартом 6:00–22:00, вне его дождя нет."""

    cfg = load_config(REFERENCE_CONFIG)
    cfg.weather.rain_day_chance = 1.0
    env = WorldEnv(cfg, np.random.default_rng(SEED))
    env.advance_day(0)
    runs = _runs([env.is_rain(DAY0 + timedelta(minutes=m)) for m in range(SCAN_MINUTES)])
    assert len(runs) == 1, runs  # ровно один эпизод: вне его is_rain ложен
    lo, hi = runs[0]
    assert 60 <= hi - lo <= 180  # длительность эпизода 1–3 ч
    assert 6 * 60 <= lo <= 22 * 60  # час старта в замкнутом окне 6:00–22:00
    # полуоткрытый интервал эпизода [начало, конец)
    assert not env.is_rain(DAY0 + timedelta(minutes=lo - 1))
    assert env.is_rain(DAY0 + timedelta(minutes=lo))
    assert env.is_rain(DAY0 + timedelta(minutes=hi - 1))
    assert not env.is_rain(DAY0 + timedelta(minutes=hi))


def test_forced_rain_config_factors() -> None:
    """Форс-дождь: конфиг даёт потребителям множители спроса и времени поездок 1.4.

    Применение множителей к λ и длительностям поездок — изменения
    m2-demand/m2-drivers (proposal.md, вопрос 1); среда даёт состояние.
    """

    cfg = load_config(REFERENCE_CONFIG)
    assert cfg.weather.demand_mult == 1.4
    assert cfg.weather.travel_mult == 1.4


def test_rain_episode_survives_midnight() -> None:
    """Эпизод через полночь: is_rain истинен в минутах следующих суток (решение 6).

    Окно розыгрыша форсится конфигом (старт 22:00, длительность 3 ч), чтобы
    эпизод гарантированно перешёл через полночь; хвост не стирается розыгрышем
    следующих суток — «розыгрыш следующего дня этого не меняет».
    """

    cfg = load_config(REFERENCE_CONFIG)
    cfg.weather.rain_day_chance = 1.0
    cfg.weather.episode_start_h = [22, 22]
    cfg.weather.episode_duration_h = [3, 3]
    env = WorldEnv(cfg, np.random.default_rng(SEED))
    env.advance_day(0)
    assert not env.is_rain(datetime(2026, 11, 2, 21, 59))
    assert env.is_rain(datetime(2026, 11, 2, 22, 0))
    env.advance_day(1)
    assert env.is_rain(datetime(2026, 11, 3, 0, 30))  # хвост предыдущих суток
    assert not env.is_rain(datetime(2026, 11, 3, 1, 0))  # конец интервала исключающий
    assert not env.is_rain(datetime(2026, 11, 3, 12, 0))  # между хвостом и новым эпизодом
    assert env.is_rain(datetime(2026, 11, 3, 22, 30))  # новый эпизод суток 1


def test_no_rain_day() -> None:
    """День без дождя (шанс 0.0): is_rain ложен для каждой минуты суток."""

    cfg = load_config(REFERENCE_CONFIG)
    cfg.weather.rain_day_chance = 0.0
    env = WorldEnv(cfg, np.random.default_rng(SEED))
    env.advance_day(0)
    assert not any(env.is_rain(DAY0 + timedelta(minutes=m)) for m in range(1440))


# --- форс-инъекция и тариф (спека «Surge-инъекции и price_multiplier», 4.2) ---


def test_forced_injection_single_window() -> None:
    """Форс-инъекция (шанс 1.0): ровно одна зона с окном 30–60 мин и множителем 1.2–1.6.

    Окно находится по наблюдаемому состоянию — минуты, где price_multiplier > 1
    (proposal.md, вопрос 2); вне окна и в других зонах множитель ровно 1.0.
    """

    cfg = load_config(REFERENCE_CONFIG)
    cfg.pricing.surge_injection.day_chance = 1.0
    env = WorldEnv(cfg, np.random.default_rng(SEED))
    env.advance_day(0)
    windows: list[tuple[int, tuple[int, int]]] = []
    for zone_id in range(len(cfg.city.zones)):
        runs = _runs(
            [
                env.price_multiplier(zone_id, DAY0 + timedelta(minutes=m)) > 1
                for m in range(SCAN_MINUTES)
            ]
        )
        assert len(runs) <= 1, (zone_id, runs)
        if runs:
            windows.append((zone_id, runs[0]))
    assert len(windows) == 1, windows
    surge_zone, (lo, hi) = windows[0]
    assert 30 <= hi - lo <= 60  # длительность окна 30–60 минут
    mults = {env.price_multiplier(surge_zone, DAY0 + timedelta(minutes=m)) for m in range(lo, hi)}
    assert len(mults) == 1  # множитель постоянен внутри окна
    assert 1.2 <= mults.pop() <= 1.6
    # вне окна и в других зонах множитель ровно 1.0
    assert all(
        env.price_multiplier(zone_id, DAY0 + timedelta(minutes=m)) == 1.0
        for zone_id in range(len(cfg.city.zones))
        for m in range(SCAN_MINUTES)
        if zone_id != surge_zone or not lo <= m < hi
    )


def test_no_injection_day_tariff_is_flat() -> None:
    """День без инъекции (шанс 0.0): price_multiplier равен 1.0 для всех зон и минут."""

    cfg = load_config(REFERENCE_CONFIG)
    cfg.pricing.surge_injection.day_chance = 0.0
    env = WorldEnv(cfg, np.random.default_rng(SEED))
    env.advance_day(0)
    assert all(
        env.price_multiplier(zone_id, DAY0 + timedelta(minutes=m)) == 1.0
        for zone_id in range(len(cfg.city.zones))
        for m in range(1440)
    )


def test_rain_and_friday_do_not_change_injection_tariff() -> None:
    """Дождь и «пятничный вечер» тариф не меняют: множитель 1.0 вне surge-окон (§3.6)."""

    cfg = load_config(REFERENCE_CONFIG)
    cfg.weather.rain_day_chance = 1.0  # дождь форсирован
    cfg.pricing.surge_injection.day_chance = 0.0
    city = build_city(cfg)
    env = WorldEnv(cfg, np.random.default_rng(SEED))
    env.advance_day(0)
    # окно дождя и все прочие минуты суток 0: тариф ровно 1.0
    assert all(
        env.price_multiplier(zone.id, DAY0 + timedelta(minutes=m)) == 1.0
        for zone in city.zones
        for m in range(1440)
    )
    # «пятничный вечер» активен в business-зоне, но тариф не меняет
    friday_evening = FRIDAY.replace(hour=18)
    assert env.event_label(city.zones[2], friday_evening) == "friday_evening"
    assert env.price_multiplier(city.zones[2].id, friday_evening) == 1.0


# --- «пятничный вечер» (спека «Пятничный вечер», задача 4.3) -----------------


def test_friday_evening_in_configured_archetypes() -> None:
    """Пятница и суббота 17:00–23:59 для архетипов business/mixed: 'friday_evening'."""

    cfg = load_config(REFERENCE_CONFIG)
    city = build_city(cfg)
    env = WorldEnv(cfg, np.random.default_rng(SEED))
    event_zones = [
        zone for zone in city.zones if zone.archetype in cfg.events.friday_evening.archetypes
    ]
    assert {zone.archetype for zone in event_zones} == {"business", "mixed"}
    for day in (FRIDAY, SATURDAY):
        for hour in range(17, 24):
            for minute in (0, 30, 59):
                stamp = day.replace(hour=hour, minute=minute)
                for zone in event_zones:
                    assert env.event_label(zone, stamp) == "friday_evening"


def test_no_friday_event_outside_days_hours_archetypes() -> None:
    """Вне дней, часов или архетипов события нет: четверг, полдень, residential/airport.

    Инъекций в среде нет (advance_day не вызывался), поэтому вне события
    метка пустая (спека: «если surge-инъекция в этой зоне не активна»).
    """

    cfg = load_config(REFERENCE_CONFIG)
    city = build_city(cfg)
    env = WorldEnv(cfg, np.random.default_rng(SEED))
    thursday = DAY0 + timedelta(days=3)
    assert env.event_label(city.zones[2], thursday.replace(hour=18)) == ""
    assert env.event_label(city.zones[2], FRIDAY.replace(hour=12)) == ""
    assert env.event_label(city.zones[2], FRIDAY.replace(hour=16, minute=59)) == ""
    assert env.event_label(city.zones[2], SATURDAY.replace(hour=16, minute=59)) == ""
    assert env.event_label(city.zones[0], FRIDAY.replace(hour=18)) == ""  # residential
    assert env.event_label(city.zones[4], FRIDAY.replace(hour=18)) == ""  # airport


def test_friday_evening_priority_over_injection() -> None:
    """Одновременная инъекция и «пятничный вечер»: 'friday_evening' — первое сработавшее."""

    cfg = load_config(REFERENCE_CONFIG)
    cfg.pricing.surge_injection.day_chance = 1.0
    city = build_city(cfg)
    env = WorldEnv(cfg, np.random.default_rng(PRIORITY_SEED))
    env.advance_day(4)  # пятница
    # окно инъекции находится по наблюдаемому состоянию (proposal.md, вопрос 2)
    windows = {
        zone: runs[0]
        for zone in city.zones
        if (
            runs := _runs(
                [
                    env.price_multiplier(zone.id, FRIDAY + timedelta(minutes=m)) > 1
                    for m in range(1440)
                ]
            )
        )
    }
    assert len(windows) == 1, windows
    ((zone, (lo, _hi)),) = windows.items()
    assert zone.archetype in cfg.events.friday_evening.archetypes
    probe = max(FRIDAY + timedelta(minutes=lo), FRIDAY + timedelta(hours=17))
    assert probe < FRIDAY + timedelta(days=1)  # окно пересекает вечер пятницы
    assert env.price_multiplier(zone.id, probe) > 1  # инъекция активна в точке проба
    assert env.event_label(zone, probe) == "friday_evening"  # приоритет §3.8


def test_friday_evening_demand_multiplier() -> None:
    """«Горб» спроса ×1.3: конфиг даёт events.friday_evening.demand_mult = 1.3.

    Применение множителя к λ — изменение m2-demand (proposal.md, вопрос 1);
    канонические дни/часы/архетипы — приложение А.
    """

    cfg = load_config(REFERENCE_CONFIG)
    assert cfg.events.friday_evening.demand_mult == 1.3
    assert cfg.events.friday_evening.days == [4, 5]
    assert cfg.events.friday_evening.hours == [17, 23]
    assert cfg.events.friday_evening.archetypes == ["business", "mixed"]


# --- поток rng (спека «Нескачущий rng-поток», §10.13, задача 4.4) -------------


def test_rng_determinism_of_episodes() -> None:
    """Детерминизм эпизодов по seed: одинаковые seed — одинаковые эпизоды."""

    city = build_city(_forced_config())
    first = WorldEnv(_forced_config(), np.random.default_rng(SEED))
    second = WorldEnv(_forced_config(), np.random.default_rng(SEED))
    for day in (0, 1):
        first.advance_day(day)
        second.advance_day(day)
        day_start = DAY0 + timedelta(days=day)
        assert _observable(first, city.zones, day_start, SCAN_MINUTES) == _observable(
            second, city.zones, day_start, SCAN_MINUTES
        )


def test_queries_do_not_shift_rng_stream() -> None:
    """Запросы не сдвигают поток rng (§10.13): опрос минут не меняет следующие сутки.

    Две среды с одинаковым seed выполняют advance_day(0); первая опрашивает
    is_rain/event_label/price_multiplier по всем минутам, вторая не опрашивает
    ничего; после advance_day(1) состояния совпадают.
    """

    city = build_city(_forced_config())
    polled = WorldEnv(_forced_config(), np.random.default_rng(SEED))
    silent = WorldEnv(_forced_config(), np.random.default_rng(SEED))
    polled.advance_day(0)
    silent.advance_day(0)
    for minute in range(1440):
        stamp = DAY0 + timedelta(minutes=minute)
        polled.is_rain(stamp)
        for zone in city.zones:
            polled.event_label(zone, stamp)
            polled.price_multiplier(zone.id, stamp)
    polled.advance_day(1)
    silent.advance_day(1)
    assert _observable(polled, city.zones, DAY1, 1440) == _observable(
        silent, city.zones, DAY1, 1440
    )


# --- шоки спроса (change world-demand-shocks, спека simulator-env) --------------

SHOCK_SEED = 2026
"""Seed потока шоков в тестах; отдельный от SEED, чтобы не путать потоки."""


def _shock_env(**overrides) -> tuple[WorldEnv, AppConfig]:
    """Среда эталонного конфига с включёнными шоками и переопределениями полей."""

    cfg = load_config(REFERENCE_CONFIG)
    cfg.demand.shocks.enabled = True
    for field, value in overrides.items():
        setattr(cfg.demand.shocks, field, value)
    env = WorldEnv(cfg, np.random.default_rng(SEED), np.random.default_rng(SHOCK_SEED))
    return env, cfg


def _shock_series(env: WorldEnv, cfg: AppConfig, minutes: int) -> np.ndarray:
    """Поминутные множители всех зон: массив ``(минуты, зоны)`` в порядке id."""

    series = np.empty((minutes, len(cfg.city.zones)), dtype=float)
    for minute in range(minutes):
        stamp = DAY0 + timedelta(minutes=minute)
        env.advance_minute(stamp)
        series[minute] = [env.demand_factor(zone.id, stamp) for zone in cfg.city.zones]
    return series


def test_shocks_mean_near_one() -> None:
    """Сценарий «Среднее множителя около 1»: 60 сим-суток, дефолтные σ, [0.95, 1.05]."""

    env, cfg = _shock_env()
    series = _shock_series(env, cfg, 60 * 1440)
    mean = float(series.mean())
    assert 0.95 <= mean <= 1.05, mean


def test_shocks_persistence_component() -> None:
    """Сценарий «Персистентность компоненты»: лаг 60 при τ=240 даёт ACF [0.70, 0.85]."""

    env, cfg = _shock_env(zone_sigma=0.0, day_sigma=0.0, city_tau_min=240)
    series = _shock_series(env, cfg, 60 * 1440)
    log_factor = np.log(series[:, 0])
    acf = float(np.corrcoef(log_factor[:-60], log_factor[60:])[0, 1])
    assert 0.70 <= acf <= 0.85, acf


def test_shocks_deterministic_by_seed() -> None:
    """Сценарий «Детерминизм шоков по seed»: одинаковые генераторы — одинаковый путь."""

    first, cfg = _shock_env()
    second, _ = _shock_env()
    assert np.array_equal(_shock_series(first, cfg, 2 * 1440), _shock_series(second, cfg, 2 * 1440))


def test_shock_reads_do_not_consume_rng() -> None:
    """Сценарий «Чтение не тратит rng»: опрос demand_factor не сдвигает путь шоков."""

    polled, cfg = _shock_env()
    silent, _ = _shock_env()
    expected = _shock_series(polled, cfg, 1440)
    observed = np.empty_like(expected)
    for minute in range(1440):
        stamp = DAY0 + timedelta(minutes=minute)
        silent.advance_minute(stamp)
        observed[minute] = [silent.demand_factor(zone.id, stamp) for zone in cfg.city.zones]
        for _ in range(3):  # повторные чтения той же минуты тоже не тратят rng
            for zone in cfg.city.zones:
                silent.demand_factor(zone.id, stamp)
    assert np.array_equal(expected, observed)


def test_shocks_do_not_shift_rain_and_injections() -> None:
    """Сценарий «Шоки не сдвигают дождь и инъекции»: 30 суток, эпизоды совпадают."""

    cfg = _forced_config()
    cfg.demand.shocks.enabled = True
    with_shocks = WorldEnv(cfg, np.random.default_rng(SEED), np.random.default_rng(SHOCK_SEED))
    cfg_off = _forced_config()
    cfg_off.demand.shocks.enabled = False
    without_shocks = WorldEnv(cfg_off, np.random.default_rng(SEED))
    city = build_city(cfg)
    for day in range(30):
        with_shocks.advance_day(day)
        without_shocks.advance_day(day)
        day_start = DAY0 + timedelta(days=day)
        assert _observable(with_shocks, city.zones, day_start, SCAN_MINUTES) == _observable(
            without_shocks, city.zones, day_start, SCAN_MINUTES
        )


def test_disabled_shocks_factor_is_one_without_generator() -> None:
    """При выключенных шоках множитель 1.0, генератор шоков не требуется."""

    cfg = load_config(REFERENCE_CONFIG)
    env = WorldEnv(cfg, np.random.default_rng(SEED))
    stamp = DAY0 + timedelta(minutes=13)
    env.advance_minute(stamp)
    assert all(env.demand_factor(zone.id, stamp) == 1.0 for zone in cfg.city.zones)
