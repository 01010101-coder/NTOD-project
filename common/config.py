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
# каноническим значениям). Профили спроса, drivers.count и dest_decay_min —
# финальные числа калибровки (change ``m2-calibration``), синхронно внесённые
# в ``config.yaml``; прочие числа — стартовые значения приложения А.
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
# часы 00–23. Значения — финальные после калибровки ``m2-calibration`` (равны
# ``config.yaml``). Числа записаны строками по 24 значения, чтобы каждый профиль
# читался одной строкой, как в приложении А (ruff format разворачивает длинные
# списки по одному элементу на строку); разбор строки — в ``_hours``.
_PROFILE_ROWS: dict[str, dict[str, str]] = {
    "residential": {
        "weekday": "0.1 0.08 0.07 0.07 0.07 0.1 0.2 0.35 0.42 0.35 0.28 0.25 "
        "0.25 0.25 0.25 0.28 0.32 0.38 0.42 0.38 0.35 0.32 0.2 0.14",
        "weekend": "0.14 0.12 0.1 0.08 0.08 0.1 0.18 0.25 0.32 0.35 0.38 0.38 "
        "0.38 0.38 0.38 0.38 0.42 0.45 0.5 0.45 0.42 0.38 0.28 0.2",
    },
    "business": {
        "weekday": "0.2 0.15 0.12 0.12 0.15 0.25 0.45 0.6 0.7 0.65 0.55 0.55 "
        "0.55 0.55 0.55 0.6 0.7 1.0 1.3 1.4 1.35 1.2 0.7 0.4",
        "weekend": "0.25 0.2 0.15 0.15 0.15 0.2 0.3 0.4 0.5 0.55 0.6 0.6 "
        "0.6 0.6 0.6 0.65 0.75 0.9 1.0 1.05 1.0 0.9 0.6 0.4",
    },
    "airport": {
        "weekday": "0.5 0.5 0.5 0.55 0.55 0.6 0.65 0.7 0.7 0.7 0.65 0.6 "
        "0.6 0.6 0.6 0.6 0.65 0.7 0.75 0.8 0.8 0.75 0.65 0.55",
        "weekend": "0.55 0.5 0.5 0.5 0.55 0.6 0.65 0.7 0.75 0.75 0.7 0.65 "
        "0.65 0.65 0.65 0.65 0.7 0.75 0.8 0.8 0.8 0.7 0.65 0.6",
    },
    "mixed": {
        "weekday": "0.3 0.25 0.2 0.2 0.2 0.25 0.35 0.45 0.55 0.55 0.55 0.55 "
        "0.55 0.6 0.6 0.65 0.75 0.9 1.0 1.05 1.1 1.0 0.8 0.5",
        "weekend": "0.35 0.3 0.25 0.25 0.25 0.3 0.4 0.5 0.6 0.65 0.65 0.65 "
        "0.65 0.7 0.7 0.75 0.85 0.95 1.05 1.1 1.15 1.05 0.85 0.6",
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
    dest_decay_min: int = 12  # финальное число калибровки m2-calibration

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


class RepositionCfg(_CfgBase):
    """Отклик предложения на тариф (change ``m6-supply-response``): переезды свободных.

    Свободные водители раз в ``every_min`` минут переезжают в зону с приростом
    тарифа не ниже ``min_gain`` на пути не длиннее ``max_reach_min``; из зоны
    уезжает доля ``min(max_share, sensitivity × gain)``. При ``enabled = false``
    (дефолт) движок работает побитово как до change'а.
    """

    enabled: bool = False
    every_min: int = Field(default=5, ge=1)
    min_gain: float = Field(default=0.10, gt=0)
    sensitivity: float = Field(default=1.0, gt=0)
    max_share: float = Field(default=0.3, gt=0, le=1.0)
    max_reach_min: int = Field(default=12, ge=1)


class DispatchCfg(_CfgBase):
    """Диспатч (change ``world-dispatch-radius``): радиус поиска и отказ пассажира.

    ``max_pickup_min`` — максимальная длительность подачи в минутах; ``None``
    (дефолт) — без ограничения. ``cancel_grace_min`` — подача, которую пассажир
    терпит без риска отказа; ``cancel_prob_per_min`` — прирост вероятности
    отказа за каждую минуту сверх терпения (``0.0`` — отказов нет);
    ``cancel_max_prob`` — потолок вероятности. Дефолты выключают оба механизма,
    поэтому мир побитово совпадает с поведением до change'а.
    """

    max_pickup_min: int | None = Field(default=None, ge=1)
    cancel_grace_min: int = Field(default=5, ge=0)
    cancel_prob_per_min: float = Field(default=0.0, ge=0.0)
    cancel_max_prob: float = Field(default=0.9, ge=0.0, le=1.0)


# Почасовые доли парка на смене (change ``world-driver-shifts``): элемент h —
# доля ``drivers.count`` на смене в момент h:00. Значения — стартовые числа
# спеки ``config-loading``, калибруются отдельно (design.md, риск 1).
_SHIFT_WEEKDAY: list[float] = [
    0.40,
    0.35,
    0.33,
    0.33,
    0.33,
    0.40,
    0.55,
    0.70,
    0.80,
    0.80,
    0.75,
    0.70,
    0.70,
    0.70,
    0.70,
    0.75,
    0.80,
    0.85,
    0.85,
    0.80,
    0.72,
    0.65,
    0.55,
    0.45,
]

_SHIFT_WEEKEND: list[float] = [
    0.50,
    0.45,
    0.40,
    0.35,
    0.33,
    0.33,
    0.40,
    0.50,
    0.60,
    0.65,
    0.70,
    0.70,
    0.70,
    0.70,
    0.70,
    0.72,
    0.75,
    0.80,
    0.80,
    0.78,
    0.72,
    0.68,
    0.62,
    0.55,
]


class OnlineShareCfg(_CfgBase):
    """Почасовая доля парка на смене: будни и выходные — ровно 24 значения в (0, 1].

    Профиль — стартовое значение спеки (change ``world-driver-shifts``);
    значения калибруются отдельно. Валидация длины и диапазона даёт ошибку с
    путём до поля профиля (``drivers.shifts.online_share.weekday``).
    """

    weekday: list[float] = Field(default_factory=lambda: list(_SHIFT_WEEKDAY))
    weekend: list[float] = Field(default_factory=lambda: list(_SHIFT_WEEKEND))

    @field_validator("weekday", "weekend")
    @classmethod
    def _check_profile(cls, value: list[float]) -> list[float]:
        """Профиль обязан содержать ровно 24 доли, каждая в (0, 1]."""

        if len(value) != 24:
            raise ValueError(
                f"профиль доли парка на смене должен содержать ровно 24 значения, "
                f"получено {len(value)}"
            )
        for hour, share in enumerate(value):
            if not 0.0 < share <= 1.0:
                raise ValueError(
                    f"доля парка на смене в час {hour} должна лежать в (0, 1], получено {share}"
                )
        return value


class ShiftsCfg(_CfgBase):
    """Смены водителей (change ``world-driver-shifts``): флаг и профиль по часам.

    При ``enabled = false`` (дефолт) шаг смен в движке пропускается целиком, и
    мир побитово совпадает с поведением до change'а.
    """

    enabled: bool = False
    online_share: OnlineShareCfg = Field(default_factory=OnlineShareCfg)


class DriversCfg(_CfgBase):
    """Парк водителей: размер, лимит ожидания назначения и отклик на тариф."""

    count: int = 380  # финальное число калибровки m2-calibration
    wait_limit_min: int = 6
    reposition: RepositionCfg = Field(default_factory=RepositionCfg)
    dispatch: DispatchCfg = Field(default_factory=DispatchCfg)
    shifts: ShiftsCfg = Field(default_factory=ShiftsCfg)


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


class AppCfg(_CfgBase):
    """Сервер воспроизведения (change ``replay-server``): адрес, порт и дни лога."""

    host: str = "127.0.0.1"
    port: int = 8000
    days: int = 30


class MlCfg(_CfgBase):
    """Машинное обучение (change ``m3-dataset``): эпоха данных и гиперпараметры.

    Дефолты — приложение А плана M3 (docs/plan-m3-ml.md): ``db`` — БД единой
    эпохи (37-дневный прогон мира), ``train_days``/``val_days``/``test_days`` —
    роли целых дней (train — дни 1–28, val — хвост train 29–30 для ранней
    остановки, test — продолжение мира 31–37), ``window_hours`` — окно истории
    признаков (финальное число свипа 3/6/12 change ``m3-train-cli``:
    победило окно 3 — val MAE спроса 1.806 против 1.848/1.843, синхронно с
    config.yaml, как ``dest_decay_min`` после m2-calibration);
    гиперпараметры обучения заводятся один раз на все changes вехи M3.
    """

    db: str = "data/m3_epoch.db"
    train_days: int = 30
    val_days: int = 2
    test_days: int = 7
    window_hours: int = 3
    zone_embed_dim: int = 8
    mlp_hidden: list[int] = Field(default_factory=lambda: [256, 128])
    lstm_hidden: int = 64
    lstm_layers: int = 1
    dropout: float = 0.1
    lr: float = 0.001
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 8
    cls_weight: float = 1.0
    seed: int = 42  # отдельный от мира seed обучения


# Стартовые числа политики — приложение А плана M5+M6 (docs/plan-m5-m6-policy.md);
# калибровка δ-сдвигов и частот срабатываний — change ``m6-worlds``.
_DEFAULT_STAIR_ROWS: list[dict[str, float]] = [
    {"p_min": 0.80, "mult": 1.50},
    {"p_min": 0.60, "mult": 1.30},
    {"p_min": 0.40, "mult": 1.15},
]

# δ-сдвиги ступеней по семействам моделей; модель без записи в ``shift``
# получает сдвиг 0.0 (сценарий «Значения из файла переопределяют дефолты»).
# Значения — результат δ-калибровки по растягивающей формуле change
# ``policy-stretch-calibration`` (artifacts/policy-calibration.json: val-дни
# 29–30 эпохи M3, цель 0.10, частоты после 0.098/0.100), синхронно с
# config.yaml — прецедент dest_decay_min/drivers.count после m2-calibration.
_DEFAULT_SHIFT: dict[str, float] = {"mlp": -0.719, "lstm": -0.688}


class PolicyStairCfg(_CfgBase):
    """Ступень политики цены: порог вероятности дефицита и множитель тарифа."""

    p_min: float
    mult: float

    @field_validator("p_min")
    @classmethod
    def _check_p_min(cls, value: float) -> float:
        """Порог ступени обязан лежать в открытом интервале (0, 1)."""

        if not 0.0 < value < 1.0:
            raise ValueError(f"p_min должен быть в интервале (0, 1), получено {value}")
        return value

    @field_validator("mult")
    @classmethod
    def _check_mult(cls, value: float) -> float:
        """Множитель ступени не может снижать тариф (нейтральный 1.0 или выше)."""

        if value < 1.0:
            raise ValueError(f"mult должен быть >= 1.0, получено {value}")
        return value


def _default_stairs() -> list[PolicyStairCfg]:
    """Три ступени приложения А плана M5+M6 (каждый вызов — новые объекты)."""

    return [PolicyStairCfg(**row) for row in _DEFAULT_STAIR_ROWS]


class PolicyCfg(_CfgBase):
    """Политика цены (change ``m6-policy``): ступени по прогнозу дефицита.

    Дефолты — приложение А плана M5+M6: ``stairs`` записаны по убыванию
    ``p_min`` (порядок списка на выбор ступени не влияет), ``shift`` —
    δ-сдвиги эффективной вероятности по семействам моделей, ``worlds_dir`` —
    каталог миров веточного генератора (потребитель — change ``m6-worlds``).
    """

    horizon_min: int = 15
    stairs: list[PolicyStairCfg] = Field(default_factory=_default_stairs)
    total_cap: float = 2.0
    reactive_mult: float = 1.30
    shift: dict[str, float] = Field(default_factory=lambda: dict(_DEFAULT_SHIFT))
    target_fire_rate: float = 0.10
    worlds_dir: str = "data/worlds"

    @field_validator("stairs")
    @classmethod
    def _check_stairs_non_empty(cls, value: list[PolicyStairCfg]) -> list[PolicyStairCfg]:
        """Список ступеней не может быть пустым — политика без ступеней бессмысленна."""

        if not value:
            raise ValueError("stairs не может быть пустым списком — задайте хотя бы одну ступень")
        return value

    @field_validator("shift")
    @classmethod
    def _fill_shift_defaults(cls, value: dict[str, float]) -> dict[str, float]:
        """Незаданная в YAML модель получает калиброванный δ-сдвиг из ``_DEFAULT_SHIFT``."""

        merged = dict(_DEFAULT_SHIFT)
        merged.update(value)
        return merged


class AppConfig(_CfgBase):
    """Корневая конфигурация проекта: секции канона §6 плана M2, ``ml`` и ``policy``."""

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
    app: AppCfg = Field(default_factory=AppCfg)
    ml: MlCfg = Field(default_factory=MlCfg)
    policy: PolicyCfg = Field(default_factory=PolicyCfg)


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
