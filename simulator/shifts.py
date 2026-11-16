"""Цель числа водителей на смене на минуту ``t`` (change ``world-driver-shifts``).

Чистая функция спеки simulator-loop: профиль дня ``t`` (то же правило
``t.weekday() < 5``, что у профилей спроса), линейная интерполяция между
часами внутри профиля и округление ``floor(count × s + 0.5)``. Модуль не
зависит от движка и парка, rng не использует — цель детерминирована.
"""

import math
from datetime import datetime

from common.config import DriversCfg


def online_target(drivers_cfg: DriversCfg, t: datetime) -> int:
    """Цель водителей на смене в минуту ``t`` (спека simulator-loop).

    Доля ``s(t) = share[h] + (share[(h + 1) mod 24] − share[h]) × m / 60``,
    где ``h``/``m`` — час и минута ``t``; оба значения берутся из профиля дня
    ``t`` (будни при ``weekday() < 5``, иначе выходные). Цель —
    ``floor(drivers.count × s(t) + 0.5)``.
    """

    share = drivers_cfg.shifts.online_share
    profile = share.weekday if t.weekday() < 5 else share.weekend
    hour = t.hour
    next_hour = (hour + 1) % 24
    fraction = t.minute / 60.0
    s = profile[hour] + (profile[next_hour] - profile[hour]) * fraction
    return math.floor(drivers_cfg.count * s + 0.5)
