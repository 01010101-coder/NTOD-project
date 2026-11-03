"""Модель города симулятора: зоны-архетипы, профили спроса и кратчайшие пути.

Публичный контракт — приложение Б плана M2: frozen-dataclass-типы ``ZoneInfo``,
``HourProfile``, ``CitySpec`` и функция ``build_city``. Координаты карты из
``ZoneCfg.map`` в модель города не попадают (§3.1 плана M2: движок их не читает,
они нужны карте M5). Профили строятся только по архетипам, реально
встречающимся в зонах; матрица кратчайших путей ``path_min`` считается при
построении алгоритмом Флойда по двунаправленному дорожному графу и приводится
к целому dtype (§3.2, решение 9 design.md change'а m2-city).
"""

from dataclasses import dataclass

import numpy as np

from common.config import AppConfig, RoadCfg


@dataclass(frozen=True)
class ZoneInfo:
    """Зона города: идентификатор, имя, архетип и веса источника/цели."""

    id: int
    name: str
    archetype: str
    attract_origin: float
    attract_dest: float


@dataclass(frozen=True)
class HourProfile:
    """Суточный профиль спроса архетипа: множители на часы 00–23."""

    weekday: np.ndarray  # длины 24
    weekend: np.ndarray  # длины 24


@dataclass(frozen=True)
class CitySpec:
    """Спецификация города, общей для всех модулей движка."""

    zones: tuple[ZoneInfo, ...]  # сортированы по id, id = 0..Z-1 без пропусков
    path_min: np.ndarray  # (Z, Z), int, кратчайшие пути в минутах
    profiles: dict[str, HourProfile]  # по архетипам из зон


def build_city(cfg: AppConfig) -> CitySpec:
    """Собрать спецификацию города из конфигурации (приложение Б плана M2).

    Зоны сортируются по ``id`` без координат карты; идентификаторы должны быть
    уникальны и образовывать ряд ``0..Z-1`` без пропусков, дорожный граф —
    связным. Нарушение любого условия — ``ValueError`` с понятным текстом,
    спецификация не возвращается (решение 10 design.md).
    """
    zones = sorted(
        (
            ZoneInfo(
                id=zone.id,
                name=zone.name,
                archetype=zone.archetype,
                attract_origin=zone.attract_origin,
                attract_dest=zone.attract_dest,
            )
            for zone in cfg.city.zones
        ),
        key=lambda zone: zone.id,
    )
    _check_zone_ids(zones)
    path_min = _shortest_paths(len(zones), cfg.city.roads)
    profiles = {
        archetype: HourProfile(
            weekday=np.asarray(cfg.demand.profiles[archetype]["weekday"], dtype=float),
            weekend=np.asarray(cfg.demand.profiles[archetype]["weekend"], dtype=float),
        )
        for archetype in sorted({zone.archetype for zone in zones})
    }
    return CitySpec(zones=tuple(zones), path_min=path_min, profiles=profiles)


def _check_zone_ids(zones: list[ZoneInfo]) -> None:
    """Проверить, что id зон уникальны и образуют 0..Z-1 без пропусков.

    Контракт приложения Б («сортированы по id, id = 0..Z-1 без пропусков»):
    иначе матрица ``path_min`` не определима и индексация зон по id ломается.
    """
    ids = [zone.id for zone in zones]
    if len(set(ids)) != len(ids):
        duplicates = sorted({zone_id for zone_id in ids if ids.count(zone_id) > 1})
        raise ValueError(f"id зон должны быть уникальны, дубликаты: {duplicates}")
    if ids != list(range(len(ids))):
        raise ValueError(
            f"id зон должны образовывать ряд 0..{len(ids) - 1} без пропусков, получено: {ids}"
        )


def _shortest_paths(zone_count: int, roads: list[RoadCfg]) -> np.ndarray:
    """Матрица кратчайших путей Флойда по двунаправленным рёбрам-минутам (§3.2).

    Ребро прокладывает путь в обе стороны; отсутствие ребра — бесконечность,
    диагональ — нули. Несвязный граф (пара зон без пути между ними) —
    ``ValueError``: матрица не определена. Итог приводится к целому dtype —
    контракт приложения Б; для эталона минуты целые, дробные вне эталона
    не рассматриваются (§6 плана M2).
    """
    dist = np.full((zone_count, zone_count), np.inf)
    np.fill_diagonal(dist, 0.0)
    for road in roads:
        forward = dist[road.from_zone, road.to_zone]
        backward = dist[road.to_zone, road.from_zone]
        dist[road.from_zone, road.to_zone] = min(forward, road.minutes)
        dist[road.to_zone, road.from_zone] = min(backward, road.minutes)
    for middle in range(zone_count):
        np.minimum(dist, dist[:, middle, None] + dist[None, middle, :], out=dist)
    finite = np.isfinite(dist)
    if not finite.all():
        origin, target = np.argwhere(~finite)[0]
        raise ValueError(
            f"дорожный граф не связный: нет пути между зонами {int(origin)} и {int(target)}"
        )
    return dist.astype(np.int64)
