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
        # Живые эпизоды дождя — абсолютные интервалы [начало, конец)
        # сим-времени: хвост эпизода предыдущих суток, перешедший через
        # полночь, и эпизод разыгранных суток (решение 6 design.md).
        self._rain_spans: list[tuple[datetime, datetime]] = []
        # Живые окна surge-инъекций — та же логика абсолютных интервалов;
        # запись (начало, конец, зона, множитель); тариф всюду 1.0 вне окон
        # (§3.6). При пересечении хвоста и нового окна одной зоны приоритет
        # у окна разыгранных суток — запросы читают список с конца.
        self._surges: list[tuple[datetime, datetime, int, float]] = []

    def advance_day(self, day_index: int) -> None:
        """Перейти на сутки ``day_index`` (выполняется в 00:00 сим-суток).

        Дата суток = ``engine.sim_start`` + ``day_index`` суток, день недели
        Mon=0 (решение 7 design.md). Розыгрыши §3.6 выполняются в фиксированном
        порядке — дождь, затем инъекция; Бернулли — ``rng.random() < шанс``,
        при неуспехе параметры эпизода не разыгрываются (безусловность §10.13 —
        независимость от решений потребителя тарифа, а не от исхода Бернулли).
        Хвосты эпизодов предыдущих суток, перешедшие через полночь, сохраняются
        — «розыгрыш следующего дня этого не меняет» (решение 6 design.md);
        интервалы, закончившиеся до начала новых суток, удаляются.
        """
        day_start = _parse_day_start(self._cfg.engine.sim_start) + timedelta(days=day_index)
        # Хвосты предыдущих суток: интервалы, ещё не закончившиеся к 00:00
        # новых суток, остаются живыми, остальные удаляются.
        self._rain_spans = [span for span in self._rain_spans if span[1] > day_start]
        self._surges = [surge for surge in self._surges if surge[1] > day_start]
        # Дождь: Bernoulli; при успехе — эпизод с целым часом старта по
        # замкнутому диапазону конфига и целой длительностью в часах; интервал
        # от 00:00 суток, без усечения полуночью (решение 6 design.md).
        weather = self._cfg.weather
        if self._rng.random() < weather.rain_day_chance:
            start_h = int(
                self._rng.integers(weather.episode_start_h[0], weather.episode_start_h[1] + 1)
            )
            duration_h = int(
                self._rng.integers(weather.episode_duration_h[0], weather.episode_duration_h[1] + 1)
            )
            rain_start = day_start + timedelta(hours=start_h)
            self._rain_spans.append((rain_start, rain_start + timedelta(hours=duration_h)))
        # Surge-инъекция: Bernoulli; при успехе — зона, целая минута старта
        # («U(сутки)»), длительность в минутах и множитель; розыгрыш выполняется
        # после дождя — порядок §3.6 (решение 6 design.md).
        injection = self._cfg.pricing.surge_injection
        if self._rng.random() < injection.day_chance:
            zone_id = int(self._rng.integers(0, len(self._cfg.city.zones)))
            start_min = int(self._rng.integers(0, 1440))  # «U(сутки)»: минута 0..1439
            duration_min = int(
                self._rng.integers(injection.duration_min[0], injection.duration_min[1] + 1)
            )
            surge_start = day_start + timedelta(minutes=start_min)
            self._surges.append(
                (
                    surge_start,
                    surge_start + timedelta(minutes=duration_min),
                    zone_id,
                    float(self._rng.uniform(injection.multiplier[0], injection.multiplier[1])),
                )
            )

    def is_rain(self, t: datetime) -> bool:
        """Сообщить, идёт ли дождь в момент ``t``; ``rng`` не тратит.

        Истинен внутри абсолютного интервала живого эпизода
        ``[начало, конец)``, включая минуты после полуночи, если эпизод
        перешёл через неё (решение 6 design.md).
        """
        return any(start <= t < end for start, end in self._rain_spans)

    def event_label(self, zone: ZoneInfo, t: datetime) -> str:
        """Метка события зоны ``zone`` в момент ``t`` по правилу колонки ``event`` (§3.8).

        «Первое сработавшее»: ``'friday_evening'`` | ``'surge_injection'`` | ``''``.
        Календарная часть вычисляется из самой метки запроса ``t`` (решение 7
        design.md), инъекция — по абсолютным окнам зоны ``zone.id``.
        """
        friday = self._cfg.events.friday_evening
        if (
            t.weekday() in friday.days
            and friday.hours[0] <= t.hour <= friday.hours[1]
            and zone.archetype in friday.archetypes
        ):
            return "friday_evening"
        if any(
            surge_zone == zone.id and start <= t < end
            for start, end, surge_zone, _mult in self._surges
        ):
            return "surge_injection"
        return ""

    def price_multiplier(self, zone_id: int, t: datetime) -> float:
        """Множитель тарифа зоны ``zone_id`` в момент ``t``; ``rng`` не тратит.

        «Всюду 1.0, кроме обучающих инъекций» (§3.6): дождь и «пятничный
        вечер» тариф не меняют. Окна читаются с конца списка: при пересечении
        хвоста предыдущих суток и нового окна одной зоны действует окно
        разыгранных суток.
        """
        for start, end, surge_zone, mult in reversed(self._surges):
            if surge_zone == zone_id and start <= t < end:
                return mult
        return 1.0
