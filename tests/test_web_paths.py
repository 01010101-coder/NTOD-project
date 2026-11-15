"""Тесты матрицы маршрутов change ``replay-server`` (``app/paths.py``, задача 3.1).

``test_routes_match_path_min`` — для всех пар зон эталонного конфига сумма
минут рёбер маршрута равна ``build_city(cfg).path_min[i][j]`` (кратчайшие
совпадают с Флойдом движка), диагональ — ``[zone]``, концы маршрута — источник
и цель. ``test_disconnected_graph_rejected`` — несвязный граф даёт
``ValueError`` с понятным текстом.
"""

import math

import pytest

from app.paths import build_routes
from common.config import AppConfig
from simulator.city import build_city


def test_routes_match_path_min():
    """Свёртка маршрута в сумму минут рёбер == path_min движка для всех пар (3.1)."""

    cfg = AppConfig()  # эталон: 6 зон, 7 двунаправленных рёбер
    routes = build_routes(cfg)
    path_min = build_city(cfg).path_min
    count = len(cfg.city.zones)
    assert len(routes) == count and all(len(row) == count for row in routes)

    # Рёбра двунаправленные — словарь минут симметричен.
    minutes: dict[tuple[int, int], float] = {}
    for road in cfg.city.roads:
        minutes[(road.from_zone, road.to_zone)] = road.minutes
        minutes[(road.to_zone, road.from_zone)] = road.minutes
    for i in range(count):
        for j in range(count):
            route = routes[i][j]
            assert route[0] == i and route[-1] == j
            if i == j:
                assert route == [i]
                continue
            # Сумма минут рёбер последовательности против матрицы Флойда движка.
            total = sum(minutes[(a, b)] for a, b in zip(route, route[1:]))
            assert math.isclose(total, float(path_min[i][j])), (i, j, route, total)

    # Пример кратчайшего с пересадкой: 0 → 4 прямого ребра нет, путь через
    # промежуточные зоны обязан существовать и быть короче любой обходной пары.
    route_04 = routes[0][4]
    assert len(route_04) >= 3
    direct = minutes.get((0, 4))
    assert direct is None


def test_disconnected_graph_rejected():
    """Несвязный граф (зона 2 без рёбер) — ``ValueError`` с текстом «не связный»."""

    cfg = AppConfig.model_validate(
        {
            "city": {
                "zones": [
                    {
                        "id": 0,
                        "name": "a",
                        "archetype": "residential",
                        "attract_origin": 1.0,
                        "attract_dest": 1.0,
                        "map": {"x": 0, "y": 0},
                    },
                    {
                        "id": 1,
                        "name": "b",
                        "archetype": "business",
                        "attract_origin": 1.0,
                        "attract_dest": 1.0,
                        "map": {"x": 1, "y": 0},
                    },
                    {
                        "id": 2,
                        "name": "c",
                        "archetype": "airport",
                        "attract_origin": 1.0,
                        "attract_dest": 1.0,
                        "map": {"x": 2, "y": 0},
                    },
                ],
                "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 5.0}],
            },
        }
    )
    with pytest.raises(ValueError, match="не связный"):
        build_routes(cfg)
