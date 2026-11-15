"""Матрица маршрутов как последовательностей зон (change ``replay-server``, задача 3.1).

Next-hop Флойд по рёбрам дорог конфигурации, независимо от движка
(design.md, решение 5): ``path_min`` движка даёт только длины, анимации
``city-canvas`` нужен путь как последовательность зон — клиент раскладывает
фазы ``driver_zone → origin`` и ``origin → dest`` по точкам зон. Результат —
квадратная матрица ``routes``: для каждой пары зон последовательность от
источника к цели, диагональ — ``[zone]``. Несвязный граф — ``ValueError`` с
понятным текстом (матрица не определена, как и в ``simulator.city``).
"""

from common.config import AppConfig


def build_routes(cfg: AppConfig) -> list[list[list[int]]]:
    """Кратчайшие маршруты всех пар зон по рёбрам конфигурации (next-hop Флойд).

    Зоны нумеруются ``0..Z-1`` (контракт города §3.1; нарушение —
    ``ValueError`` до построения матриц). Алгоритм — Флойд с матрицей
    следующих вершин: при улучшении пути ``i → k → j`` первый хоп пути ``i → j``
    становится первым хопом пути ``i → k``; восстановление идёт по хопам от
    источника до цели. Диагональ — ``[zone]``; рёбра двунаправленные,
    минуты сравниваются во float без приведения к целому.
    """

    zones = sorted(zone.id for zone in cfg.city.zones)
    if len(set(zones)) != len(zones) or zones != list(range(len(zones))):
        raise ValueError(
            f"id зон должны образовывать ряд 0..{len(zones) - 1} без пропусков, получено: {zones}"
        )
    count = len(zones)
    inf = float("inf")
    dist = [[inf] * count for _ in range(count)]
    nxt: list[list[int | None]] = [[None] * count for _ in range(count)]
    for i in range(count):
        dist[i][i] = 0.0
        nxt[i][i] = i
    for road in cfg.city.roads:
        for a, b in ((road.from_zone, road.to_zone), (road.to_zone, road.from_zone)):
            if road.minutes < dist[a][b]:
                dist[a][b] = road.minutes
                nxt[a][b] = b

    for middle in range(count):
        for i in range(count):
            di = dist[i][middle]
            if di == inf:
                continue
            for j in range(count):
                if di + dist[middle][j] < dist[i][j]:
                    dist[i][j] = di + dist[middle][j]
                    nxt[i][j] = nxt[i][middle]

    for i in range(count):
        for j in range(count):
            if dist[i][j] == inf:
                raise ValueError(f"дорожный граф не связный: нет пути между зонами {i} и {j}")

    return [[_path(i, j, nxt) for j in range(count)] for i in range(count)]


def _path(i: int, j: int, nxt: list[list[int | None]]) -> list[int]:
    """Восстановить последовательность зон ``i → j`` по матрице следующих вершин."""

    route = [i]
    current = i
    while current != j:
        following = nxt[current][j]
        if following is None:  # не достигается при проверенной связности — защита от цикла
            raise ValueError(f"маршрут {i} → {j} не восстанавливается: граф не связный")
        current = following
        route.append(current)
    return route
