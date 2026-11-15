"""Тесты чистого правила отклика предложения (change ``m6-supply-response``).

Покрывают сценарии спеки ``simulator-supply-response``: переезд в соседнюю
зону с надбавкой, малый прирост, недоступная дальняя зона, выбор между
целями (путь и ``zone_id``), ровный тариф, потолок доли, детерминизм и
неизменность входа. Правило не использует rng; вход — рукописные списки
свободных ``driver_id``, тарифы и матрица путей.
"""

import numpy as np
import pytest

from common.config import RepositionCfg
from simulator.reposition import plan_relocations

ZONE_COUNT = 6


def _paths(overrides: dict[tuple[int, int], float]) -> np.ndarray:
    """Матрица путей: дальние по умолчанию (100), переопределения — по ключу."""

    matrix = np.full((ZONE_COUNT, ZONE_COUNT), 100.0)
    np.fill_diagonal(matrix, 0.0)
    for (source, target), minutes in overrides.items():
        matrix[source][target] = minutes
    return matrix


def _config(**overrides: float) -> RepositionCfg:
    """Параметры правила из спеки; ``overrides`` меняют отдельные поля."""

    params = {
        "min_gain": 0.10,
        "sensitivity": 1.0,
        "max_share": 0.3,
        "max_reach_min": 12,
    }
    params.update(overrides)
    return RepositionCfg(**params)


def _free_zone_zero(ids: list[int]) -> list[list[int]]:
    """Свободные только в зоне 0 (остальные зоны пусты)."""

    return [list(ids)] + [[] for _ in range(ZONE_COUNT - 1)]


def test_relocate_three_of_ten() -> None:
    """Сценарий «Переезд в соседнюю зону с надбавкой»: 3 из 10 водителей уезжают."""

    free = _free_zone_zero([0, 6, 12, 18, 24, 30, 36, 42, 48, 54])
    tariffs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.3]
    paths = _paths({(0, 5): 5.0})

    assert plan_relocations(free, tariffs, paths, _config()) == [
        (0, 5),
        (6, 5),
        (12, 5),
    ]


def test_small_gain_no_move() -> None:
    """Сценарий «Малый прирост не вызывает переезда»: 0.05 < min_gain 0.10."""

    free = _free_zone_zero([0, 6, 12])
    tariffs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.05]
    paths = _paths({(0, 5): 5.0})

    assert plan_relocations(free, tariffs, paths, _config()) == []


def test_far_zone_unreachable() -> None:
    """Сценарий «Дальняя зона недоступна»: путь 20 > max_reach_min 12."""

    free = _free_zone_zero([0, 6, 12])
    tariffs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.3]
    paths = _paths({(0, 5): 20.0})

    assert plan_relocations(free, tariffs, paths, _config()) == []


def test_path_at_limit_allowed() -> None:
    """Граница ``path_min == max_reach_min`` допустима (условие ≤)."""

    free = _free_zone_zero([0, 6, 12, 18, 24, 30, 36, 42, 48, 54])
    tariffs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.3]
    paths = _paths({(0, 5): 12.0})

    assert plan_relocations(free, tariffs, paths, _config()) == [
        (0, 5),
        (6, 5),
        (12, 5),
    ]


def test_tie_prefers_shorter_path() -> None:
    """Сценарий «Выбор между несколькими целями»: равный прирост — короче путь."""

    free = _free_zone_zero([0, 6, 12, 18, 24, 30, 36, 42, 48, 54])
    tariffs = [1.0, 1.3, 1.3, 1.0, 1.0, 1.0]
    paths = _paths({(0, 1): 8.0, (0, 2): 5.0})

    assert plan_relocations(free, tariffs, paths, _config()) == [
        (0, 2),
        (6, 2),
        (12, 2),
    ]


def test_tie_same_path_prefers_smaller_zone_id() -> None:
    """При одинаковом приросте и пути выбирается цель с меньшим ``zone_id``."""

    free = _free_zone_zero([0, 6, 12, 18, 24, 30, 36, 42, 48, 54])
    tariffs = [1.0, 1.3, 1.3, 1.0, 1.0, 1.0]
    paths = _paths({(0, 1): 5.0, (0, 2): 5.0})

    assert plan_relocations(free, tariffs, paths, _config()) == [
        (0, 1),
        (6, 1),
        (12, 1),
    ]


def test_uniform_tariff_no_moves() -> None:
    """Сценарий «Ровный тариф — нет движения»: пустой список."""

    free = _free_zone_zero([0, 6, 12])
    tariffs = [1.0] * ZONE_COUNT
    paths = _paths({})

    assert plan_relocations(free, tariffs, paths, _config()) == []


def test_share_capped_by_max_share() -> None:
    """Сценарий «Доля ограничена сверху»: floor(0.3 × 7) = 2."""

    free = _free_zone_zero([0, 6, 12, 18, 24, 30, 36])
    tariffs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.5]
    paths = _paths({(0, 5): 5.0})

    assert plan_relocations(free, tariffs, paths, _config()) == [(0, 5), (6, 5)]


def test_gain_threshold_tolerates_float_error() -> None:
    """Пара 1.2/1.1: float даёт 0.099999…, но gain ≥ min_gain 0.10 по спеке."""

    free = _free_zone_zero(list(range(100)))
    tariffs = [1.1, 1.1, 1.1, 1.1, 1.1, 1.2]
    paths = _paths({(0, 5): 5.0})

    result = plan_relocations(free, tariffs, paths, _config())

    assert len(result) == 10  # floor(0.1 × 100), а не 0 из-за ошибки float
    assert {target for _driver, target in result} == {5}


def test_floor_share_tolerates_float_error() -> None:
    """0.29 × 100 = 28.999…: floor по спеке — 29 водителей."""

    free = _free_zone_zero(list(range(100)))
    tariffs = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0]  # gain 1.0 — срабатывает max_share
    paths = _paths({(0, 5): 5.0})

    result = plan_relocations(free, tariffs, paths, _config(max_share=0.29))

    assert len(result) == 29


def test_gain_below_threshold_still_rejected() -> None:
    """Допуск не делает допустимым заметно меньший прирост."""

    free = _free_zone_zero(list(range(100)))
    tariffs = [1.1, 1.1, 1.1, 1.1, 1.1, 1.2]
    paths = _paths({(0, 5): 5.0})

    # 1.2 − 1.1 ≈ 0.09999999999999987 < 0.100001 − 1e-9
    assert plan_relocations(free, tariffs, paths, _config(min_gain=0.100001)) == []


def test_output_ordered_by_source_then_driver_id() -> None:
    """Список упорядочен: по возрастанию исходной зоны, внутри — по ``driver_id``."""

    free = [[2, 8, 14, 20], [1, 5, 9, 13], [], [], [], []]
    tariffs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.3]
    paths = _paths({(0, 5): 5.0, (1, 5): 5.0})

    assert plan_relocations(free, tariffs, paths, _config()) == [(2, 5), (1, 5)]


def test_repeat_call_same_and_input_unchanged() -> None:
    """Сценарий «Повторный вызов»: результат стабилен, вход не изменён."""

    free = _free_zone_zero([0, 6, 12, 18, 24, 30, 36, 42, 48, 54])
    snapshot = [list(ids) for ids in free]
    tariffs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.3]
    paths = _paths({(0, 5): 5.0})
    cfg = _config()

    first = plan_relocations(free, tariffs, paths, cfg)
    second = plan_relocations(free, tariffs, paths, cfg)

    assert first == second
    assert free == snapshot


def test_zone_count_mismatch_rejected() -> None:
    """Число списков свободных водителей обязано совпадать с числом тарифов."""

    with pytest.raises(ValueError, match="не совпадает"):
        plan_relocations([[0]], [1.0, 1.0], _paths({}), _config())
