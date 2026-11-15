"""Отклик предложения на тариф: чистое правило выбора переездов (change ``m6-supply-response``).

Свободные водители, оставшиеся после диспатча, раз в ``every_min`` минут решают,
переезжать ли порожняком в зону с более высоким фактическим тарифом. Правило
детерминированное и без rng (спека ``simulator-supply-response``): из исходной
зоны ``i`` в зону ``j`` возможен переезд только при приросте
``gain = tariff[j] − tariff[i] ≥ min_gain`` и пути
``path_min[i][j] ≤ max_reach_min``. У зоны не больше одной цели — максимальный
прирост, затем короткий путь, затем меньший ``zone_id``. Уезжает
``floor(min(max_share, sensitivity × gain) × free[i])`` водителей с наименьшими
``driver_id``. Все зоны решают по одному снимку входа; правило не изменяет
входные данные и не читает wall-clock.
"""

import math
from collections.abc import Sequence

import numpy as np

from common.config import RepositionCfg

# Допуск для сравнений с порогом и округления вниз (спека задаёт математику
# точными числами, а float ошибается на ~1e-16): 1.2 − 1.1 = 0.099999…, и без
# допуска это не «gain ≥ min_gain 0.10»; 0.29 × 100 = 28.999…, и floor без
# допуска даёт 28 вместо 29. При переборе граничных min_gain/max_share в
# экспериментах поведение без допуска непредсказуемо.
_FLOAT_EPS = 1e-9


def plan_relocations(
    free_ids_by_zone: Sequence[Sequence[int]],
    tariff_by_zone: Sequence[float] | np.ndarray,
    path_min: np.ndarray,
    cfg: RepositionCfg,
) -> list[tuple[int, int]]:
    """Спланировать переезды свободных водителей на одну минуту решения.

    ``free_ids_by_zone`` — ``driver_id`` свободных водителей по зонам, внутри
    зоны по возрастанию (зеркало ``Engine._free_ids``); ``tariff_by_zone`` —
    фактический тариф каждой зоны текущей минуты; ``path_min`` — матрица
    кратчайших путей ``[источник][цель]`` в минутах; ``cfg`` — параметры
    ``drivers.reposition``. Возвращает упорядоченный список пар
    ``(driver_id, zone_id_цели)``: по возрастанию исходной зоны, внутри зоны —
    по возрастанию ``driver_id``. Пустой список — движения нет.
    """

    tariffs = [float(value) for value in tariff_by_zone]
    zone_count = len(tariffs)
    if len(free_ids_by_zone) != zone_count:
        raise ValueError(
            "число списков свободных водителей "
            f"({len(free_ids_by_zone)}) не совпадает с числом тарифов ({zone_count})"
        )

    relocations: list[tuple[int, int]] = []
    for source in range(zone_count):
        free_ids = free_ids_by_zone[source]
        if not free_ids:
            continue
        # Одна цель на исходную зону: максимальный gain, затем меньший путь,
        # затем меньший zone_id. Обход по возрастанию id делает последний
        # тай-брейк бесплатным (строгое улучшение по ключу).
        best_target: int | None = None
        best_gain = 0.0
        best_path = 0.0
        for target in range(zone_count):
            if target == source:
                continue
            gain = tariffs[target] - tariffs[source]
            # Допуск: математически gain ≥ min_gain, даже если float дал
            # 0.09999999999999987 вместо 0.10 (пара 1.2/1.1).
            if gain < cfg.min_gain - _FLOAT_EPS:
                continue
            path = float(path_min[source][target])
            if path > cfg.max_reach_min:
                continue
            better = (
                best_target is None
                or gain > best_gain
                or (gain == best_gain and path < best_path)
                or (gain == best_gain and path == best_path and target < best_target)
            )
            if better:
                best_target, best_gain, best_path = target, gain, path
        if best_target is None:
            continue
        share = min(cfg.max_share, cfg.sensitivity * best_gain)
        # Допуск: share × n, равное целому по спеке (0.29 × 100), во float
        # даёт 28.999…; floor без допуска теряет водителя.
        count = math.floor(share * len(free_ids) + _FLOAT_EPS)
        for driver_id in free_ids[:count]:
            relocations.append((int(driver_id), best_target))
    return relocations
