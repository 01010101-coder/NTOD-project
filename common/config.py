"""Pydantic-модели конфигурации проекта.

Каркас механики (``extra="forbid"``, ``load_config``, частичная загрузка с
дозаполнением дефолтами) задан в M1; change ``m2-city`` привёл состав ключей и
дефолты к канону §6 плана M2: дефолты моделей равны каноническим значениям
приложения А этого плана, поэтому пустой конфиг грузится и валиден, а
эталонный ``config.yaml`` в корне репо равен конфигу целиком из дефолтов.
У каждой секции — полные значения по умолчанию: конфиг грузится без файла и
из файла, задающего лишь часть секций («старые конфиги грузятся после
добавления новых секций»). Неизвестные ключи запрещены (``extra="forbid"``):
опечатка в конфиге даёт понятную ошибку с путём до ключа, а не молчаливое
игнорирование.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _CfgBase(BaseModel):
    """База всех моделей конфига: неизвестные ключи запрещены."""

    model_config = ConfigDict(extra="forbid")


class StorageCfg(_CfgBase):
    """Хранилище: путь к файлу базы SQLite (схему создаёт ``m1-storage``)."""

    db_path: str = "data/taxi.db"


# Канонический мир — приложение А плана M2 (§6: дефолты моделей равны
# каноническим значениям). Числа — стартовая точка; финальные фиксируются
# калибровкой (change ``m2-calibration``) и обновляются здесь же и в
# ``config.yaml``.
_DEFAULT_ZONE_ROWS: list[dict[str, Any]] = [
    {
        "id": 0,
        "name": "sleep-1",
        "archetype": "residential",
        "attract_origin": 1.0,
        "attract_dest": 0.6,
        "map": {"x": 180.0, "y": 180.0},
    },
    {
        "id": 1,
        "name": "sleep-2",
        "archetype": "residential",
        "attract_origin": 1.0,
        "attract_dest": 0.6,
        "map": {"x": 820.0, "y": 180.0},
    },
    {
        "id": 2,
        "name": "center",
        "archetype": "business",
        "attract_origin": 1.3,
        "attract_dest": 1.8,
        "map": {"x": 500.0, "y": 400.0},
    },
    {
        "id": 3,
        "name": "office",
        "archetype": "business",
        "attract_origin": 1.2,
        "attract_dest": 1.5,
        "map": {"x": 500.0, "y": 620.0},
    },
    {
        "id": 4,
        "name": "airport",
        "archetype": "airport",
        "attract_origin": 0.6,
        "attract_dest": 1.2,
        "map": {"x": 900.0, "y": 620.0},
    },
    {
        "id": 5,
        "name": "lounge",
        "archetype": "mixed",
        "attract_origin": 1.1,
        "attract_dest": 1.4,
        "map": {"x": 180.0, "y": 620.0},
    },
]

# Двунаправленные рёбра дорожного графа, время в минутах.
_DEFAULT_ROAD_ROWS: list[dict[str, Any]] = [
    {"from_zone": 0, "to_zone": 2, "minutes": 9.0},
    {"from_zone": 1, "to_zone": 2, "minutes": 9.0},
    {"from_zone": 2, "to_zone": 3, "minutes": 6.0},
    {"from_zone": 3, "to_zone": 4, "minutes": 10.0},
    {"from_zone": 2, "to_zone": 5, "minutes": 8.0},
    {"from_zone": 0, "to_zone": 5, "minutes": 5.0},
    {"from_zone": 1, "to_zone": 4, "minutes": 14.0},
]

# Суточные профили спроса: архетип -> будни/выходные -> строка 24 множителей на
# часы 00–23. Числа записаны строками по 24 значения, чтобы каждый профиль
# читался одной строкой, как в приложении А (ruff format разворачивает длинные
# списки по одному элементу на строку); разбор строки — в ``_hours``.
_PROFILE_ROWS: dict[str, dict[str, str]] = {
    "residential": {
        "weekday": "0.2 0.15 0.1 0.1 0.1 0.2 0.6 1.2 1.6 1.3 0.9 0.8 "
        "0.8 0.8 0.8 0.9 1.0 1.2 1.4 1.3 1.1 0.9 0.6 0.35",
        "weekend": "0.3 0.25 0.2 0.15 0.1 0.15 0.3 0.5 0.7 0.9 1.0 1.1 "
        "1.1 1.1 1.1 1.1 1.1 1.2 1.3 1.3 1.2 1.0 0.8 0.5",
    },
    "business": {
        "weekday": "0.15 0.1 0.1 0.1 0.15 0.3 0.7 1.3 1.7 1.5 1.2 1.1 "
        "1.1 1.1 1.1 1.2 1.3 1.6 1.7 1.4 1.0 0.7 0.4 0.25",
        "weekend": "0.2 0.15 0.1 0.1 0.1 0.15 0.25 0.4 0.6 0.8 0.9 1.0 "
        "1.0 1.0 1.0 1.0 1.0 1.1 1.2 1.2 1.1 0.9 0.6 0.4",
    },
    "airport": {
        "weekday": "0.7 0.7 0.7 0.8 0.8 0.9 1.0 1.1 1.1 1.1 1.1 1.0 "
        "1.0 1.0 1.0 1.0 1.0 1.1 1.1 1.1 1.1 1.0 0.9 0.8",
        "weekend": "0.8 0.75 0.7 0.7 0.8 0.9 0.95 1.0 1.05 1.1 1.1 1.0 "
        "1.0 1.0 1.0 1.0 1.0 1.05 1.1 1.1 1.1 1.0 0.95 0.85",
    },
    "mixed": {
        "weekday": "0.5 0.4 0.3 0.2 0.15 0.15 0.25 0.4 0.6 0.8 0.9 0.9 "
        "0.9 0.9 1.0 1.0 1.1 1.2 1.3 1.4 1.5 1.5 1.3 0.9",
        "weekend": "0.8 0.7 0.55 0.4 0.25 0.2 0.25 0.35 0.5 0.7 0.85 0.95 "
        "1.0 1.0 1.05 1.1 1.15 1.25 1.35 1.45 1.6 1.6 1.4 1.1",
    },
}


class MapCfg(_CfgBase):
    """Координаты зоны на карте (нужны карте M5; движок их не читает)."""

    x: float
    y: float


class ZoneCfg(_CfgBase):
    """Зона города: имя, архетип и веса привлекательности источника/цели."""

    id: int
    name: str
    archetype: str
    attract_origin: float
    attract_dest: float
    map: MapCfg


class RoadCfg(_CfgBase):
    """Дорога дорожного графа: время в пути между зонами в минутах."""

    from_zone: int
    to_zone: int
    minutes: float


def _default_zones() -> list[ZoneCfg]:
    """Шесть зон эталона — приложение А плана M2 (каждый вызов — новые объекты)."""

    return [ZoneCfg(**row) for row in _DEFAULT_ZONE_ROWS]


def _default_roads() -> list[RoadCfg]:
    """Семь дорог эталона — приложение А плана M2 (каждый вызов — новые объекты)."""

    return [RoadCfg(**row) for row in _DEFAULT_ROAD_ROWS]


def _hours(row: str) -> list[float]:
    """Разобрать строку 24 множителей профиля в список чисел."""

    return [float(value) for value in row.split()]


def _default_profiles() -> dict[str, dict[str, list[float]]]:
    """Профили четырёх архетипов — приложение А (каждый вызов — новые объекты)."""

    return {
        archetype: {kind: _hours(row) for kind, row in kinds.items()}
        for archetype, kinds in _PROFILE_ROWS.items()
    }


class CityCfg(_CfgBase):
    """Город: зоны-архетипы с весами источника/цели и дорожный граф."""

    zones: list[ZoneCfg] = Field(default_factory=_default_zones)
    roads: list[RoadCfg] = Field(default_factory=_default_roads)


class PopulationCfg(_CfgBase):
    """Масштаб интенсивности спроса по всему городу."""

    scale: float = 1.0


class DemandCfg(_CfgBase):
    """Спрос: базовая интенсивность, суточные профили по архетипам и затухание."""

    base_rate: float = 2.0
    profiles: dict[str, dict[str, list[float]]] = Field(default_factory=_default_profiles)
    dest_decay_min: int = 15

    @field_validator("profiles")
    @classmethod
    def _check_profile_keys(
        cls, value: dict[str, dict[str, list[float]]]
    ) -> dict[str, dict[str, list[float]]]:
        """Жёсткая структура профиля: ровно ключи ``weekday`` и ``weekend``."""

        for archetype, profile in value.items():
            if set(profile) != {"weekday", "weekend"}:
                raise ValueError(
                    f"профиль архетипа {archetype!r} должен содержать ровно ключи "
                    f"'weekday' и 'weekend', получено: {sorted(profile)}"
                )
        return value


class DriversCfg(_CfgBase):
    """Парк водителей: размер и лимит ожидания назначения."""

    count: int = 200
    wait_limit_min: int = 6


class WeatherCfg(_CfgBase):
    """Погода: шанс дождевого дня, случайный эпизод, множители спроса и времени."""

    rain_day_chance: float = 0.15
    episode_start_h: list[int] = Field(default_factory=lambda: [6, 22])
    episode_duration_h: list[int] = Field(default_factory=lambda: [1, 3])
    demand_mult: float = 1.4
    travel_mult: float = 1.4


class FridayEveningCfg(_CfgBase):
    """Событие «пятничный вечер»: дни, часы (включительно), архетипы, множитель."""

    days: list[int] = Field(default_factory=lambda: [4, 5])
    hours: list[int] = Field(default_factory=lambda: [17, 23])
    archetypes: list[str] = Field(default_factory=lambda: ["business", "mixed"])
    demand_mult: float = 1.3


class EventsCfg(_CfgBase):
    """События: пока только «пятничный вечер»."""

    friday_evening: FridayEveningCfg = Field(default_factory=FridayEveningCfg)


class SurgeInjectionCfg(_CfgBase):
    """Обучающие surge-инъекции: случайный шанс в день, длительность, множитель."""

    day_chance: float = 0.10
    duration_min: list[int] = Field(default_factory=lambda: [30, 60])
    multiplier: list[float] = Field(default_factory=lambda: [1.2, 1.6])


class PricingCfg(_CfgBase):
    """Прайсинг: эластичность, формула выручки (§10.15) и surge-инъекции."""

    elasticity_alpha: float = 1.0
    base_boarding: float = 100.0
    per_min_rub: float = 12.0
    surge_injection: SurgeInjectionCfg = Field(default_factory=SurgeInjectionCfg)


class EngineCfg(_CfgBase):
    """Движок симуляции: шаг агрегации метрик и старт сим-часов (понедельник)."""

    agg_step_min: int = 5
    sim_start: str = "2026-11-02"


class AppConfig(_CfgBase):
    """Корневая конфигурация проекта: секции канона §6 плана M2."""

    seed: int = 42
    storage: StorageCfg = Field(default_factory=StorageCfg)
    city: CityCfg = Field(default_factory=CityCfg)
    population: PopulationCfg = Field(default_factory=PopulationCfg)
    demand: DemandCfg = Field(default_factory=DemandCfg)
    drivers: DriversCfg = Field(default_factory=DriversCfg)
    weather: WeatherCfg = Field(default_factory=WeatherCfg)
    events: EventsCfg = Field(default_factory=EventsCfg)
    pricing: PricingCfg = Field(default_factory=PricingCfg)
    engine: EngineCfg = Field(default_factory=EngineCfg)


def load_config(path: str | Path) -> AppConfig:
    """Загрузить конфигурацию из YAML-файла и валидировать моделью ``AppConfig``.

    Разбор выполняет ``yaml.safe_load`` — произвольные YAML-объекты конфигу не
    нужны (решение design.md 5). Пустой файл (``safe_load`` даёт ``None``) и
    пустой словарь дают конфиг целиком из дефолтов; файл, задающий лишь часть
    секций, дозаполняется дефолтами pydantic-моделей. Значение YAML, не
    являющееся отображением секций (список, скаляр), отклоняется ``ValueError``
    с текстом «конфиг должен быть словарём секций» ещё до валидации pydantic —
    иначе падение было бы внутренним ``TypeError`` без объяснения.
    """
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(
            f"конфиг должен быть словарём секций, получено {type(data).__name__}: {data!r}"
        )
    return AppConfig.model_validate(data)
