"""Среда мира симулятора M2: погода, календарь событий и тариф зон.

Публичный контракт — приложение Б плана M2 (``docs/plan-m2-engine.md``):
конструктор ``WorldEnv(cfg, rng)``, суточный шаг ``advance_day`` и запросы
``is_rain``/``event_label``/``price_multiplier``. Розыгрыши дождя и
surge-инъекций выполняются безусловно в ``advance_day`` единственным
переданным снаружи генератором (§3.6, §3.7, §10.13 плана): запросы — чистые
чтения состояния и ``rng`` не тратят, поэтому при заданном seed эпизоды
детерминированы, а ветки A/B сравнимы попарно. Эпизоды хранятся как
абсолютные интервалы сим-времени ``[начало, конец)`` без усечения полуночью
(решение 6 design.md change'а m2-env): эпизод, перешедший через полночь,
действует и в минуты следующих суток.
"""

from datetime import datetime

import numpy as np

from common.config import AppConfig
from simulator.city import ZoneInfo


class WorldEnv:
    """Среда мира: состояние погоды, календарных событий и тарифа сим-суток.

    Единственный источник случайности — ``Generator``, переданный
    конструктору (§3.7: генератор создаёт ``Engine`` и передаёт модулям);
    wall-clock время не используется (§10.10 — только сим-время снаружи).
    """

    def __init__(self, cfg: AppConfig, rng: np.random.Generator) -> None:
        """Сохранить конфигурацию и генератор; сутки стартуют без эпизодов."""
        self._cfg = cfg
        self._rng = rng
        # Эпизод дождя: абсолютный интервал [начало, конец); None — дождя нет.
        self._rain_start: datetime | None = None
        self._rain_end: datetime | None = None
        # Surge-инъекция: зона, абсолютное окно [начало, конец) и множитель;
        # None — инъекции нет (тариф всюду 1.0, §3.6).
        self._surge_zone: int | None = None
        self._surge_start: datetime | None = None
        self._surge_end: datetime | None = None
        self._surge_mult: float = 1.0

    def advance_day(self, day_index: int) -> None:
        """Перейти на сутки ``day_index`` (выполняется в 00:00 сим-суток).

        Состояние эпизодов не наследуется от предыдущих суток: каждый вызов
        начинает сутки с пустой среды. Суточные розыгрыши дождя и инъекции
        §3.6 заполняют приватные поля этим же вызовом — состояние суток
        определено одним вызовом (спека «Розыгрыш выполняется на сутки вперёд»).
        """
        self._rain_start = None
        self._rain_end = None
        self._surge_zone = None
        self._surge_start = None
        self._surge_end = None
        self._surge_mult = 1.0

    def is_rain(self, t: datetime) -> bool:
        """Сообщить, идёт ли дождь в момент ``t``; ``rng`` не тратит."""
        # Заглушка каркаса: эпизод заполняется суточным розыгрышем advance_day.
        return False

    def event_label(self, zone: ZoneInfo, t: datetime) -> str:
        """Метка события зоны ``zone`` в момент ``t`` по правилу колонки ``event`` (§3.8).

        «Первое сработавшее»: ``'friday_evening'`` | ``'surge_injection'`` | ``''``.
        """
        # Заглушка каркаса: календарь и инъекция — состояние суток и конфиг.
        return ""

    def price_multiplier(self, zone_id: int, t: datetime) -> float:
        """Множитель тарифа зоны ``zone_id`` в момент ``t``; ``rng`` не тратит.

        «Всюду 1.0, кроме обучающих инъекций» (§3.6): дождь и «пятничный
        вечер» тариф не меняют.
        """
        # Заглушка каркаса: окно инъекции заполняется розыгрышем advance_day.
        return 1.0
