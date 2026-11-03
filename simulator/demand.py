"""Модель спроса симулятора: заказы, интенсивность λ, спавн (§3.3–3.4 плана M2).

Публичный контракт — приложение Б плана M2: frozen-dataclass ``TripRequest``,
функции ``lambda_rate`` (мультипликативная интенсивность заказов/мин со всеми
множителями мира) и ``spawn_requests`` (пуассоновский спавн заказов по минутам
с детерминированным порядком разыгрываний и выбором назначения
``P(B | A) ∝ attract_dest(B) × exp(−path_min[A][B] / dest_decay_min)``, B ≠ A).
Модуль также несёт семантику лимита ожидания §3.4, порождающую статус
``unfulfilled``. Среда (дождь, событие, тариф) входит готовыми параметрами
``is_rain`` / ``event_mult`` / ``price_mult`` — случайные события модуль сам
не разыгрывает (design.md, Non-Goals).
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from common.config import AppConfig
from common.timeutil import add_minutes
from simulator.city import CitySpec, ZoneInfo


@dataclass(frozen=True)
class TripRequest:
    """Заказ: время создания, зона происхождения, назначение, множитель тарифа.

    Порядок полей зафиксирован приложением Б плана M2; объект неизменяем
    (frozen). ``price_multiplier`` — множитель тарифа зоны происхождения на
    момент создания заказа (§3.4 плана M2); назначение всегда отличается от
    происхождения (§3.3 плана M2).
    """

    created_at: datetime
    origin: int
    dest: int
    price_multiplier: float


def _friday_evening_active(archetype: str, t: datetime, cfg: AppConfig) -> bool:
    """Активен ли «пятничный вечер» для архетипа в момент ``t`` (решение 3 design.md).

    Все три условия — из конфига ``cfg.events.friday_evening``: день недели
    ``t.weekday()`` входит в ``days``, час ``t.hour`` попадает в диапазон
    ``hours`` включительно, архетип зоны входит в ``archetypes``. Сам множитель
    события приходит готовым параметром ``event_mult`` (proposal, вопрос 2):
    скалярная сигнатура приложения Б не выражает архетип-специфичность, поэтому
    условие проверяет ``demand.py``, а параметр остаётся швом подмены значения,
    не условия.
    """
    event = cfg.events.friday_evening
    return (
        t.weekday() in event.days
        and event.hours[0] <= t.hour <= event.hours[1]
        and archetype in event.archetypes
    )


def lambda_rate(
    city: CitySpec,
    zone: ZoneInfo,
    t: datetime,
    *,
    is_rain: bool,
    event_mult: float,
    price_mult: float,
    cfg: AppConfig,
) -> float:
    """Интенсивность спроса заказа/мин для зоны в момент ``t`` (§3.3 плана M2).

    Мультипликативная формула дословно: ``base_rate × профиль_часа(архетип,
    час, будни/выходные) × attract_origin × population.scale × (дождь? ×
    weather.demand_mult) × event_mult × price_mult**(−elasticity_alpha)``.
    Источники множителей — design.md, решение 1: профиль из
    ``city.profiles[zone.archetype]`` по часу ``t.hour`` и строке по дню недели
    (``t.weekday() < 5`` — будни, иначе выходные); дождь только при
    ``is_rain=True``; ``event_mult`` применяется к зоне только при активном
    для её архетипа «пятничном вечере» (решение 3 design.md); эластичность —
    множитель ``m**(−α)`` внутри λ (решение §10.7 плана M2), а не отдельной
    фильтрацией пассажиров.
    """
    profile = city.profiles[zone.archetype]
    hours = profile.weekday if t.weekday() < 5 else profile.weekend
    rate = cfg.demand.base_rate
    rate *= hours[t.hour]
    rate *= zone.attract_origin
    rate *= cfg.population.scale
    if is_rain:
        rate *= cfg.weather.demand_mult
    if _friday_evening_active(zone.archetype, t, cfg):
        rate *= event_mult
    return rate * price_mult ** (-cfg.pricing.elasticity_alpha)


def spawn_requests(
    rng: np.random.Generator,
    city: CitySpec,
    t: datetime,
    *,
    is_rain: bool,
    event_mult: float,
    price_mult_by_zone: np.ndarray,
    cfg: AppConfig,
) -> list[TripRequest]:
    """Пуассоновский спавн заказов за минуту ``t`` (§3.3 плана M2).

    Зоны обходятся по возрастанию ``zone_id`` (``CitySpec.zones`` сортирован —
    контракт приложения Б); для каждой вызывается ``lambda_rate`` с
    ``price_mult`` своей зоны — формула не дублируется (design.md, решение 1),
    и розыгрыш ``rng.poisson(λ)`` выполняется всегда, даже при λ=0 (решение
    §10.9 плана M2 — устойчивость потока ``rng`` к рефакторингу). Порядок
    разыгрываний: пуассон зоны, затем для каждого пассажира по порядку — один
    розыгрыш назначения и немедленное создание заказа (design.md, решение 2).
    Одинаковый seed и параметры дают идентичный список заказов.
    """
    requests: list[TripRequest] = []
    for zone in city.zones:
        rate = lambda_rate(
            city,
            zone,
            t,
            is_rain=is_rain,
            event_mult=event_mult,
            price_mult=float(price_mult_by_zone[zone.id]),
            cfg=cfg,
        )
        count = int(rng.poisson(rate))
        for _ in range(count):
            dest = _pick_destination(rng, city, zone, cfg)
            requests.append(
                TripRequest(
                    created_at=t,
                    origin=zone.id,
                    dest=dest,
                    price_multiplier=float(price_mult_by_zone[zone.id]),
                )
            )
    return requests


def _pick_destination(
    rng: np.random.Generator, city: CitySpec, origin: ZoneInfo, cfg: AppConfig
) -> int:
    """Выбрать назначение заказа: ``P(B|A) ∝ attract_dest(B) × exp(−path/decay)``.

    Кандидаты — все зоны B ≠ A в порядке возрастания id (§3.3 плана M2); веса
    ``attract_dest × exp(−path_min[A][B] / demand.dest_decay_min)`` нормируются
    на сумму, выбор — один вызов ``rng.choice`` на пассажира (design.md,
    решение 4: механизм выбора после слияния — часть мира для данного seed).
    На эталоне показатели экспоненты в диапазоне ``[exp(−1.4), exp(−1/3)]`` —
    переполнений нет.
    """
    candidates = [zone for zone in city.zones if zone.id != origin.id]
    ids = np.array([zone.id for zone in candidates])
    weights = np.array(
        [
            zone.attract_dest
            * np.exp(-city.path_min[origin.id, zone.id] / cfg.demand.dest_decay_min)
            for zone in candidates
        ]
    )
    return int(rng.choice(ids, p=weights / weights.sum()))


def _is_expired(trip: TripRequest, t: datetime, cfg: AppConfig, *, assigned: bool) -> bool:
    """Истёк ли заказ к минуте ``t``: лимит ожидания §3.4 плана M2.

    Неназначенный заказ истекает ровно в минуту ``created_at +
    drivers.wait_limit_min``: минуты цикл обрабатывает ровно по одному разу,
    поэтому граница ``t ≥`` лимитная минута даёт истечение «ровно в лимитную
    минуту» (design.md, решение 5). Назначение спасает от истечения даже при
    долгой подаче — истечение применяется только к неназначенным заказам
    (решение §10.3 плана M2); правило живёт в одном месте и переиспользуется
    движком (``m2-loop``, шаг 5 цикла §3.7). Публичный контракт модуля
    приложением Б не расширяется (proposal, вопрос 3); поля статусов у
    ``TripRequest`` нет — статус возникает при записи строки trips движком.
    """
    if assigned:
        return False
    return t >= add_minutes(trip.created_at, cfg.drivers.wait_limit_min)
