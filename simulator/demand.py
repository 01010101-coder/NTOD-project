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

from common.config import AppConfig
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
