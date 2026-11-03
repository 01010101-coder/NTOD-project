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

from datetime import datetime, timedelta

import numpy as np

from common.config import AppConfig
from common.timeutil import from_iso
from simulator.city import ZoneInfo


def _parse_day_start(value: str) -> datetime:
    """Разобрать ``engine.sim_start`` в наивный datetime начала суток.

    Эталонный конфиг задаёт только дату («2026-11-02»), а строгий ``from_iso``
    M1 принимает лишь полный формат «YYYY-MM-DD HH:MM:SS» (docstring
    ``common/timeutil.py``), поэтому значение без времени дополняется
    « 00:00:00» и разбирается тем же ``from_iso`` (дата суток — решение 7
    design.md; правка ``common/`` change'ом запрещена).
    """

    if " " not in value:
        return from_iso(f"{value} 00:00:00")
    return from_iso(value)


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

        Дата суток = ``engine.sim_start`` + ``day_index`` суток, день недели
        Mon=0 (решение 7 design.md). Состояние эпизодов не наследуется от
        предыдущих суток: каждый вызов начинает сутки с пустой среды и
        выполняет розыгрыши §3.6 в фиксированном порядке — дождь, затем
        инъекция; Бернулли — ``rng.random() < шанс``, при неуспехе параметры
        эпизода не разыгрываются (безусловность §10.13 — независимость от
        решений потребителя тарифа, а не от исхода Бернулли).
        """
        day_start = _parse_day_start(self._cfg.engine.sim_start) + timedelta(days=day_index)
        # Дождь: Bernoulli; при успехе — эпизод с целым часом старта по
        # замкнутому диапазону конфига и целой длительностью в часах; интервал
        # от 00:00 суток, без усечения полуночью (решение 6 design.md).
        self._rain_start = None
        self._rain_end = None
        weather = self._cfg.weather
        if self._rng.random() < weather.rain_day_chance:
            start_h = int(
                self._rng.integers(weather.episode_start_h[0], weather.episode_start_h[1] + 1)
            )
            duration_h = int(
                self._rng.integers(weather.episode_duration_h[0], weather.episode_duration_h[1] + 1)
            )
            self._rain_start = day_start + timedelta(hours=start_h)
            self._rain_end = self._rain_start + timedelta(hours=duration_h)
        # Surge-инъекция: Bernoulli; при успехе — зона, целая минута старта
        # («U(сутки)»), длительность в минутах и множитель; розыгрыш выполняется
        # после дождя — порядок §3.6 (решение 6 design.md).
        self._surge_zone = None
        self._surge_start = None
        self._surge_end = None
        self._surge_mult = 1.0
        injection = self._cfg.pricing.surge_injection
        if self._rng.random() < injection.day_chance:
            self._surge_zone = int(self._rng.integers(0, len(self._cfg.city.zones)))
            start_min = int(self._rng.integers(0, 1440))  # «U(сутки)»: минута 0..1439
            duration_min = int(
                self._rng.integers(injection.duration_min[0], injection.duration_min[1] + 1)
            )
            self._surge_start = day_start + timedelta(minutes=start_min)
            self._surge_end = self._surge_start + timedelta(minutes=duration_min)
            self._surge_mult = float(
                self._rng.uniform(injection.multiplier[0], injection.multiplier[1])
            )

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
