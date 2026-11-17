"""Среда мира симулятора M2: погода, календарь событий, тариф зон и шоки спроса.

Публичный контракт — приложение Б плана M2 (``docs/plan-m2-engine.md``):
конструктор ``WorldEnv(cfg, rng, shock_rng=None)``, суточный шаг ``advance_day``,
поминутный шаг шоков ``advance_minute`` и запросы
``is_rain``/``event_label``/``price_multiplier``/``demand_factor``. Розыгрыши
дождя и surge-инъекций выполняются безусловно в ``advance_day`` единственным
переданным снаружи генератором (§3.6, §3.7, §10.13 плана): запросы — чистые
чтения состояния и ``rng`` не тратят, поэтому при заданном seed эпизоды
детерминированы, а ветки A/B сравнимы попарно. Эпизоды хранятся как
абсолютные интервалы сим-времени ``[начало, конец)`` без усечения полуночью
(решение 6 design.md change'а m2-env): эпизод, перешедший через полночь,
действует и в минуты следующих суток.

Change ``world-demand-shocks`` добавил скрытый множитель спроса зоны: три
лог-нормальные компоненты (город, зона, сутки), городская и зональная —
AR(1) с поминутным шагом. Розыгрыши шоков идут отдельным генератором
``shock_rng`` (четвёртый поток ``Engine``) и не сдвигают дождь и инъекции;
при выключенных шоках генератор не используется, а ``demand_factor`` = 1.0.
"""

import math
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
    """Среда мира: состояние погоды, календарных событий, тарифа и шоков спроса.

    Случайность приходит генераторами, созданными ``Engine`` и переданными
    конструктору (§3.7): ``rng`` — суточные розыгрыши дождя и инъекций,
    ``shock_rng`` — поминутные шаги шоков спроса (change
    ``world-demand-shocks``, design.md решение 3); wall-clock время не
    используется (§10.10 — только сим-время снаружи).
    """

    def __init__(
        self,
        cfg: AppConfig,
        rng: np.random.Generator,
        shock_rng: np.random.Generator | None = None,
    ) -> None:
        """Сохранить конфигурацию и генераторы; сутки стартуют без эпизодов.

        ``shock_rng`` — отдельный генератор шоков спроса (четвёртый поток
        ``Engine``, design.md решение 3); при выключенных шоках он не нужен и
        не используется. При включённых шоках начальные значения компонент
        разыгрываются из стационарного распределения ``N(0, σ²)`` сразу, чтобы
        множитель был стационарен с первой минуты.
        """
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
        # Состояние шоков спроса (change world-demand-shocks): городская
        # компонента, компоненты зон по возрастанию zone_id и уровень суток.
        self._shocks = cfg.demand.shocks
        self._shock_rng = shock_rng
        self._shock_zone_ids = sorted(zone.id for zone in cfg.city.zones)
        self._x_city = 0.0
        self._x_zone: dict[int, float] = {zone_id: 0.0 for zone_id in self._shock_zone_ids}
        self._x_day = 0.0
        if self._shocks.enabled:
            if shock_rng is None:
                raise ValueError("включённые шоки спроса требуют отдельный генератор shock_rng")
            self._x_city = float(shock_rng.normal(0.0, self._shocks.city_sigma))
            for zone_id in self._shock_zone_ids:
                self._x_zone[zone_id] = float(shock_rng.normal(0.0, self._shocks.zone_sigma))

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

    def _ar1_step(self, value: float, sigma: float, rho: float) -> float:
        """Один шаг AR(1): ``ρ·x + σ·√(1 − ρ²)·ε`` с розыгрышем ``ε`` из потока шоков."""

        if self._shock_rng is None:  # защита типов: включённые шоки требуют генератор
            raise ValueError("генератор шоков не передан среде")
        eps = float(self._shock_rng.normal())
        return rho * value + sigma * math.sqrt(1.0 - rho * rho) * eps

    def advance_minute(self, t: datetime) -> None:
        """Шаг шоков спроса на минуту ``t`` (шаг 1 цикла при включённых шоках).

        Порядок розыгрываний фиксирован (спека simulator-env, design.md
        решение 2): на 00:00 суток сначала суточная компонента ``d``, затем
        городская ``x_c``, затем компоненты зон по возрастанию ``zone_id``.
        Шаг выполняется безусловно на каждой минуте основной фазы и не
        зависит от тарифа и решений политики; при выключенных шоках он
        ничего не делает и генератор не тратит.
        """

        if not self._shocks.enabled:
            return
        if t.hour == 0 and t.minute == 0:
            assert self._shock_rng is not None
            self._x_day = float(self._shock_rng.normal(0.0, self._shocks.day_sigma))
        rho_city = math.exp(-1.0 / self._shocks.city_tau_min)
        self._x_city = self._ar1_step(self._x_city, self._shocks.city_sigma, rho_city)
        rho_zone = math.exp(-1.0 / self._shocks.zone_tau_min)
        for zone_id in self._shock_zone_ids:
            self._x_zone[zone_id] = self._ar1_step(
                self._x_zone[zone_id], self._shocks.zone_sigma, rho_zone
            )

    def demand_factor(self, zone_id: int, t: datetime) -> float:
        """Скрытый множитель спроса зоны ``zone_id`` на текущую минуту; rng не тратит.

        ``exp(x_c + x_z + d − (σ_г² + σ_з² + σ_д²)/2)``: поправка сохраняет
        среднее множителя равным 1, поэтому калибровка уровня спроса не
        сдвигается (design.md, решение 1). Чтение чистое: ``advance_minute``
        уже сдвинул состояние, а сам запрос генератор не трогает. При
        выключенных шоках — ровно 1.0 без чтения состояния.
        """

        if not self._shocks.enabled:
            return 1.0
        correction = (
            self._shocks.city_sigma**2 + self._shocks.zone_sigma**2 + self._shocks.day_sigma**2
        ) / 2.0
        return math.exp(self._x_city + self._x_zone[zone_id] + self._x_day - correction)

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
