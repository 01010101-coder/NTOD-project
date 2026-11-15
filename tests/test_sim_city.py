"""Тесты конфиг-канона и модели города change'а m2-city (строка №1 §7 плана M2).

Покрывают сценарии спек change'а: «Эталонный config.yaml валиден», «Конфиг
целиком из дефолтов» и канон контроля A11 (спека config-loading); «Публичные
типы», «Матрица кратчайших путей path_min», «Валидность города» и «Профили
спроса по архетипам» (спека simulator-city). Ключевые проверки выполняются
дважды — на эталонном config.yaml и на конфиге целиком из дефолтов: дефолты
моделей равны каноническим значениям приложения А плана M2 (§6).
"""

from collections import deque
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from common.config import AppConfig, load_config
from simulator.city import CitySpec, HourProfile, ZoneInfo, build_city

# Эталонный конфиг ищется относительно файла теста, не от cwd (как в M1-тестах).
REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"

# Архетипы эталона — приложение А плана M2.
EXPECTED_ARCHETYPES = {"residential", "business", "airport", "mixed"}

# Полная эталонная матрица кратчайших путей в минутах, сверенная вручную по
# приложению А (рёбра 0-2=9, 1-2=9, 2-3=6, 3-4=10, 2-5=8, 0-5=5, 1-4=14).
EXPECTED_PATH_MIN = np.array(
    [
        [0, 18, 9, 15, 25, 5],
        [18, 0, 9, 15, 14, 17],
        [9, 9, 0, 6, 16, 8],
        [15, 15, 6, 0, 10, 14],
        [25, 14, 16, 10, 0, 24],
        [5, 17, 8, 14, 24, 0],
    ]
)


@pytest.fixture(params=["reference", "defaults"], ids=["reference", "defaults"])
def city_cfg(request: pytest.FixtureRequest) -> AppConfig:
    """Конфигурация: эталонный config.yaml или дефолты моделей."""
    if request.param == "reference":
        return load_config(REFERENCE_CONFIG)
    return AppConfig()


@pytest.fixture(params=["reference", "defaults"], ids=["reference", "defaults"])
def city_spec(request: pytest.FixtureRequest) -> CitySpec:
    """Город: build_city из эталонного config.yaml или из дефолтов."""
    if request.param == "reference":
        return build_city(load_config(REFERENCE_CONFIG))
    return build_city(AppConfig())


# --- конфиг-канон (спека config-loading, задача 4.1) -------------------------


def test_config_reference_yaml_valid_and_equals_defaults() -> None:
    """Эталонный config.yaml валиден и равен конфигу целиком из дефолтов."""
    assert load_config(REFERENCE_CONFIG) == AppConfig()


def test_config_empty_file_loads_as_defaults(tmp_path: Path) -> None:
    """Пустой конфиг грузится и валиден: равен дефолтам (§6 плана M2)."""
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == AppConfig()


def test_config_reference_matches_canon_a11() -> None:
    """Канон контроля A11 (§11.1 плана M2) на эталонном config.yaml."""
    cfg = load_config(REFERENCE_CONFIG)
    assert len(cfg.city.zones) == 6
    assert len(cfg.city.roads) == 7
    assert cfg.demand.base_rate > 0
    assert set(cfg.demand.profiles) == EXPECTED_ARCHETYPES
    for profile in cfg.demand.profiles.values():
        assert len(profile["weekday"]) == 24
        assert len(profile["weekend"]) == 24
    assert cfg.engine.sim_start == "2026-11-02"
    assert cfg.weather.demand_mult > 1.0
    assert cfg.weather.travel_mult > 1.0
    assert 0.0 < cfg.pricing.surge_injection.day_chance <= 1.0


def test_config_defaults_match_canon_a11() -> None:
    """Дефолты моделей равны каноническим значениям приложения А (§6)."""
    cfg = AppConfig()
    assert cfg.seed == 42
    assert cfg.storage.db_path == "data/taxi.db"
    assert len(cfg.city.zones) == 6
    assert len(cfg.city.roads) == 7
    assert cfg.demand.base_rate == 2.0
    assert set(cfg.demand.profiles) == EXPECTED_ARCHETYPES
    for profile in cfg.demand.profiles.values():
        assert len(profile["weekday"]) == 24
        assert len(profile["weekend"]) == 24
    assert cfg.drivers.count == 380
    assert cfg.drivers.wait_limit_min == 6
    assert cfg.weather.rain_day_chance == 0.15
    assert cfg.events.friday_evening.days == [4, 5]
    assert cfg.events.friday_evening.hours == [17, 23]
    assert cfg.pricing.base_boarding == 100.0
    assert cfg.pricing.per_min_rub == 12.0
    assert cfg.engine.sim_start == "2026-11-02"
    assert cfg.engine.agg_step_min == 5


# --- граф дорог и path_min (спека simulator-city, задача 4.2) ----------------


def test_path_road_graph_connected(city_cfg: AppConfig) -> None:
    """Граф дорог связный: из зоны 0 обходом в ширину достижимы все зоны."""
    adjacency: dict[int, list[int]] = {}
    for road in city_cfg.city.roads:
        adjacency.setdefault(road.from_zone, []).append(road.to_zone)
        adjacency.setdefault(road.to_zone, []).append(road.from_zone)
    seen = {0}
    queue = deque([0])
    while queue:
        for neighbour in adjacency.get(queue.popleft(), []):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    assert seen == {zone.id for zone in city_cfg.city.zones}


def test_path_min_symmetric_zero_diagonal_positive_off(city_spec: CitySpec) -> None:
    """path_min симметрична, на диагонали нули, вне диагонали положительна."""
    matrix = city_spec.path_min
    assert (matrix == matrix.T).all()
    assert (np.diag(matrix) == 0).all()
    off_diagonal = matrix[~np.eye(len(matrix), dtype=bool)]
    assert (off_diagonal > 0).all()


def test_path_min_int_dtype_square_shape(city_spec: CitySpec) -> None:
    """path_min — целочисленная матрица (Z, Z); для эталона 6×6."""
    assert city_spec.path_min.dtype.kind == "i"
    assert city_spec.path_min.shape == (len(city_spec.zones), len(city_spec.zones))
    assert city_spec.path_min.shape == (6, 6)


def test_path_min_values_match_manual_reference(city_spec: CitySpec) -> None:
    """Матрица равна свертке вручную; транзит 0→3=15, прямое ребро 1→4=14."""
    assert (city_spec.path_min == EXPECTED_PATH_MIN).all()
    assert city_spec.path_min[0][3] == 15
    assert city_spec.path_min[1][4] == 14


def test_path_min_not_longer_than_direct_road(city_cfg: AppConfig) -> None:
    """Вес прямого ребра не меньше значения в матрице (сценарий спеки)."""
    spec = build_city(city_cfg)
    for road in city_cfg.city.roads:
        assert spec.path_min[road.from_zone, road.to_zone] <= road.minutes


# --- профили спроса (спека simulator-city, задача 4.3) -----------------------


def test_profiles_all_archetypes_complete(city_spec: CitySpec) -> None:
    """4 архетипа × будни/выходные, каждый массив длины 24 (эталон и дефолты)."""
    assert set(city_spec.profiles) == EXPECTED_ARCHETYPES
    for profile in city_spec.profiles.values():
        assert profile.weekday.shape == (24,)
        assert profile.weekend.shape == (24,)


def test_profiles_values_non_negative(city_spec: CitySpec) -> None:
    """Все 24×2 значений каждого архетипа неотрицательны (эталон и дефолты)."""
    for profile in city_spec.profiles.values():
        assert (profile.weekday >= 0).all()
        assert (profile.weekend >= 0).all()


def test_profiles_stored_as_numpy_arrays(city_spec: CitySpec) -> None:
    """HourProfile содержит numpy-массивы float длины 24."""
    for profile in city_spec.profiles.values():
        assert isinstance(profile, HourProfile)
        assert isinstance(profile.weekday, np.ndarray)
        assert isinstance(profile.weekend, np.ndarray)
        assert profile.weekday.dtype.kind == "f"
        assert profile.weekend.dtype.kind == "f"


def test_profiles_reference_equals_defaults() -> None:
    """Профили эталонного конфига и дефолтов совпадают (дефолты = приложение А)."""
    reference = build_city(load_config(REFERENCE_CONFIG))
    default = build_city(AppConfig())
    assert set(reference.profiles) == set(default.profiles)
    for archetype, profile in reference.profiles.items():
        other = default.profiles[archetype]
        assert np.array_equal(profile.weekday, other.weekday)
        assert np.array_equal(profile.weekend, other.weekend)


# --- зоны и валидность города (спека simulator-city, задача 4.4) -------------


def test_zone_ids_sequential_sorted_without_map(city_spec: CitySpec) -> None:
    """Зоны сортированы по id 0..Z-1 без пропусков; поля заполнены, map нет."""
    assert [zone.id for zone in city_spec.zones] == list(range(len(city_spec.zones)))
    for zone in city_spec.zones:
        assert isinstance(zone, ZoneInfo)
        assert zone.name
        assert zone.archetype in EXPECTED_ARCHETYPES
        assert zone.attract_origin > 0
        assert zone.attract_dest > 0
        assert not hasattr(zone, "map")


def test_zone_info_fields_match_contract() -> None:
    """Поля ZoneInfo — контракт приложения Б, без координат карты."""
    assert [field.name for field in fields(ZoneInfo)] == [
        "id",
        "name",
        "archetype",
        "attract_origin",
        "attract_dest",
    ]


def _cfg_with_mutated_city(mutate: Callable[[dict], None]) -> AppConfig:
    """Конфиг с правкой секции city — источник негативных сценариев build_city."""
    payload = AppConfig().model_dump()
    mutate(payload["city"])
    return AppConfig.model_validate(payload)


def test_zone_duplicate_id_rejected() -> None:
    """Дубликат zone_id — ValueError, спецификация города не возвращается."""

    def set_duplicate_id(city: dict) -> None:
        city["zones"][1]["id"] = 0

    with pytest.raises(ValueError, match="уникальны"):
        build_city(_cfg_with_mutated_city(set_duplicate_id))


def test_zone_gap_ids_rejected() -> None:
    """Пропуск идентификаторов (0, 1, 3) — ValueError, город не возвращается."""

    def drop_zones_and_gap_id(city: dict) -> None:
        del city["zones"][3:]
        city["zones"][2]["id"] = 3

    with pytest.raises(ValueError, match="без пропусков"):
        build_city(_cfg_with_mutated_city(drop_zones_and_gap_id))


def test_zone_disconnected_graph_rejected() -> None:
    """Зона без рёбер (несвязный граф) — ValueError о несвязном графе."""

    def append_isolated_zone(city: dict) -> None:
        city["zones"].append(
            {
                "id": 6,
                "name": "isolated",
                "archetype": "mixed",
                "attract_origin": 1.0,
                "attract_dest": 1.0,
                "map": {"x": 0.0, "y": 0.0},
            }
        )

    with pytest.raises(ValueError, match="не связный"):
        build_city(_cfg_with_mutated_city(append_isolated_zone))
