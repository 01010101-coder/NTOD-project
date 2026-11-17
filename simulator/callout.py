"""Вызов подработчиков на линию: чистое правило (change ``world-elastic-supply``).

Правило спеки ``simulator-callout``: на минуту решения по свободному резерву и
сигналу водителям в зонах решает, кого и в какую зону вызвать. Прирост считается
относительно 1.0 — водитель вне линии сравнивает «выйти» с «не выйти», а не зону
с зоной. Доля вызываемых берётся от размера резерва на входе и одинакова для всех
зон, поэтому порядок зон влияет только на то, кому не хватило резерва. Правило
детерминированное и без rng: повторный вызов на том же входе даёт тот же список,
входные данные не меняются.
"""

import math
from collections.abc import Sequence

from common.config import CalloutCfg

# Допуск для сравнений с порогом и округления вниз — тот же приём, что в
# ``reposition.py``: float ошибается на ~1e-16, и без допуска ``gain ≥ min_gain``
# на граничных числах и ``floor`` на значениях вроде 0.09 × 100 дают другой ответ.
_FLOAT_EPS = 1e-9


def plan_callouts(
    pool_ids: Sequence[int],
    signal_by_zone: Sequence[float],
    cfg: CalloutCfg,
) -> list[tuple[int, int]]:
    """Спланировать вызов подработчиков на одну минуту решения.

    ``pool_ids`` — ``driver_id`` свободного резерва (подработчики вне линии, ещё
    не вызванные); ``signal_by_zone`` — сигнал водителям каждой зоны текущей
    минуты; ``cfg`` — параметры ``drivers.callout``. Возвращает упорядоченный
    список пар ``(driver_id, zone_id_выхода)``: по возрастанию ``zone_id``,
    внутри зоны — по возрастанию ``driver_id``. Пустой список — вызова нет.

    Для зоны ``z`` прирост ``gain = signal[z] − 1.0``; зона вызывает водителей,
    только если ``gain ≥ min_gain``. Число вызываемых —
    ``floor(min(max_share, sensitivity × gain) × P)``, где ``P`` — размер
    резерва на входе (одинаков для всех зон). Зона забирает из ещё не
    распределённого резерва водителей с наименьшими ``driver_id``; при нехватке
    резерва получает остаток, следующие зоны — ничего.
    """

    signals = [float(value) for value in signal_by_zone]
    pool = sorted(int(driver_id) for driver_id in pool_ids)
    total = len(pool)
    if total == 0:
        return []
    calls: list[tuple[int, int]] = []
    cursor = 0
    for zone_id in range(len(signals)):
        gain = signals[zone_id] - 1.0
        # Допуск: математически gain ≥ min_gain, даже если float дал 0.099999…
        if gain < cfg.min_gain - _FLOAT_EPS:
            continue
        share = min(cfg.max_share, cfg.sensitivity * gain)
        # Допуск: доля × P, равная целому по спеке, во float даёт 4.999….
        count = math.floor(share * total + _FLOAT_EPS)
        take = min(count, total - cursor)
        for driver_id in pool[cursor : cursor + take]:
            calls.append((driver_id, zone_id))
        cursor += take
        if cursor >= total:
            break
    return calls
