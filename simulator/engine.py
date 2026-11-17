"""Минутный цикл симулятора M2: ``Engine``, ``Summary`` и шов ``TripWriter`` (§3.7 плана M2).

Публичный контракт — приложение Б плана M2 (docs/plan-m2-engine.md): ``Engine``
связывает среду, спрос, диспатч и движение в детерминированный минутный цикл с
фиксированным порядком шагов (часть контракта детерминизма, §10.1), агрегирует
город через ``simulator.aggregate.AggregateBuffer`` и отдаёт строки таблиц
интерфейсу ``TripWriter`` — движок не знает про SQLite (§10.12). Вся случайность
мира идёт через ЧЕТЫРЕ дочерних ``numpy.random.Generator``, высаженных из
``np.random.SeedSequence(seed).spawn(4)`` в фиксированном порядке
[эндогенный, экзогенный, отказ пассажира, шоки спроса] (§3 плана M5+M6,
``docs/plan-m5-m6-policy.md``): эндогенный потребляют пуассоновский спавн и
выбор назначения, экзогенный — суточные розыгрыши среды ``WorldEnv`` (дождь,
surge-инъекции), третий — отказ пассажира от долгой подачи (change
``world-dispatch-radius``), четвёртый — поминутные шаги шоков спроса (change
``world-demand-shocks``); расход одного потока не сдвигает остальные, поэтому
ветки одного seed с разными политиками цены сравнимы попарно. Wall-clock читается
только по периметру ``run`` — для ``progress.elapsed_sec`` и ``Summary.wall_sec``
(§5). Шов ``price_fn`` (§10.13) подменяет числовой тариф зон, не трогая
розыгрыши среды: ``env.advance_day`` и шаг шоков вызываются безусловно,
экзогенный и шоковый потоки не сдвигаются, ветки A/B на одном seed сравнимы
попарно. Шов ``driver_signal_fn`` (change ``world-elastic-supply``) подменяет
сигнал водителям — отдельный от цены пассажиру канал, который читают только
решения водителей: отклик предложения и вызов подработчиков.
"""

import time
from bisect import insort
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from common.config import AppConfig, DispatchCfg
from common.timeutil import add_minutes, bucket5, from_iso, to_iso
from simulator.aggregate import AggregateBuffer, _is_deficit
from simulator.callout import plan_callouts
from simulator.city import build_city
from simulator.demand import TripRequest, _is_expired, spawn_requests
from simulator.drivers import DriverState, Fleet, travel_minutes
from simulator.env import WorldEnv
from simulator.live import (
    AssignEvent,
    CancelEvent,
    DropoffEvent,
    ExpiryEvent,
    MinuteRecord,
    SpawnEvent,
)
from simulator.reposition import plan_relocations
from simulator.shifts import online_target

_TERMINAL_COMPLETED = "completed"
_TERMINAL_UNFULFILLED = "unfulfilled"
_TERMINAL_CANCELLED = "cancelled"


def _cancel_prob(pickup_min: int, cfg: DispatchCfg) -> float:
    """Вероятность отказа пассажира от подачи ``pickup_min`` минут (спека simulator-loop).

    ``p = min(cancel_max_prob, cancel_prob_per_min × max(0, pickup_min −
    cancel_grace_min))``: в пределах терпения — 0 (розыгрыш не выполняется),
    дальше вероятность растёт линейно и упирается в потолок. Чистая функция —
    вынесена из шага 3, чтобы формулу можно было проверить без прогона мира.
    """

    if cfg.cancel_prob_per_min <= 0.0:
        return 0.0
    excess = max(0, pickup_min - cfg.cancel_grace_min)
    return min(cfg.cancel_max_prob, cfg.cancel_prob_per_min * excess)


class TripWriter(Protocol):
    """Шов writer'а (§10.12): движок отдаёт строки-дикты, SQLite знает только ``__main__.py``.

    Дикты подготовлены под ``storage.repo.insert_trips`` / ``insert_zone_metrics``;
    реализация writer'а живёт в ``simulator/__main__.py`` и логики мира не содержит.
    """

    def write_trips(self, rows: list[dict]) -> None:
        """Записать батч строк ``trips`` (терминальные состояния заказов)."""

    def write_metrics(self, rows: list[dict]) -> None:
        """Записать батч строк ``zone_metrics_5min`` (закрытые 5-минутные окна)."""


@dataclass(frozen=True)
class Summary:
    """Итоги прогона (приложение Б): счётчики заказов, доли, время стены.

    ``unfulfilled_share`` — доля истёкших заказов прогона в процентах;
    ``cancelled``/``cancelled_share`` — число и доля отказов пассажиров от
    долгой подачи (change ``world-dispatch-radius``); ``trips_total`` включает
    все три статуса. ``avg_wait_min`` — среднее ожидание назначения по всем
    назначенным заказам (0.0 при отсутствии назначений — proposal, вопрос 2;
    отказы не входят); ``deficit_share`` — доля дефицитных 5-минуток по
    правилу §3.8 в процентах; ``wall_sec`` — wall-время прогона (замер только
    по периметру ``run``, §5).

    ``callouts`` и ``callout_minutes`` — стоимость меры вызова подработчиков
    (change ``world-elastic-supply``, спека simulator-loop): число вызванных
    за прогон и суммарные минуты подработчиков на линии. Оба поля добавлены
    последними с дефолтом 0: при выключенном вызове они равны 0, а
    существующие конструкторы ``Summary`` не ломаются.
    """

    days: int
    seed: int
    trips_total: int
    completed: int
    unfulfilled: int
    unfulfilled_share: float
    cancelled: int
    cancelled_share: float
    avg_wait_min: float
    deficit_share: float
    wall_sec: float
    callouts: int = 0
    callout_minutes: int = 0


@dataclass
class _ActiveTrip:
    """Активный (нестерминальный) заказ во внутренней книге движка (решение 5 design.md).

    Хранит ``TripRequest``, внутренний id и времена терминации-пути: ожидание —
    ``assigned_at is None``; назначенный — ``driver_id``/``assigned_at``/``pickup_at``
    до высадки. Строка trips строится один раз, в терминальном состоянии.
    """

    internal_id: int
    trip: TripRequest
    assigned_at: datetime | None = None
    pickup_at: datetime | None = None
    driver_id: int | None = None


class Engine:
    """Движок мира: минутный цикл §3.7 в фиксированном порядке (контракт детерминизма).

    Шаги каждой сим-минуты: (1) среда; (1а) поминутное накопление среды в буфер
    агрегации (design, решение 3); (1б) смены водителей при включённом
    ``drivers.shifts`` (change ``world-driver-shifts``); (1в) вызов подработчиков
    при включённом ``drivers.callout`` (change ``world-elastic-supply``);
    (2) спавн; (3) диспатч; (3а) отклик предложения — переезды свободных к
    надбавке при включённом ``drivers.reposition`` (change
    ``m6-supply-response``); (4) движение; (5) истечения; (6) на каждой
    ``agg_step_min``-й минуте — снапшот парка и сброс окна агрегата. Батчи
    строк writer'у уходят раз в сим-час (§5). После основной фазы — дренаж
    хвоста (proposal, вопрос 1): шаги 4–5 без спавна, диспатча и агрегации до
    терминального состояния всех заказов; строки метрик за хвост не пишутся.
    """

    def __init__(
        self,
        cfg: AppConfig,
        *,
        seed: int | None = None,
        price_fn: Callable[[int, datetime], float] | None = None,
        driver_signal_fn: Callable[[int, datetime], float] | None = None,
    ) -> None:
        """Собрать мир по конфигу: четыре rng-потока (§3 плана M5+M6), среда, парк, швы.

        ``seed=None`` → ``cfg.seed``; из ``np.random.SeedSequence(seed).spawn(4)``
        создаются четыре дочерних генератора в фиксированном порядке
        [эндогенный, экзогенный, отказ пассажира, шоки спроса]: эндогенный
        передаётся ``spawn_requests`` (пуассоновский спавн, выбор назначения),
        экзогенный — ``WorldEnv`` (дождь, surge-инъекции), третий разыгрывает
        отказ от долгой подачи (change ``world-dispatch-radius``), четвёртый —
        поминутные шаги шоков спроса (change ``world-demand-shocks``). Дети
        ``spawn(n)`` определяются ``spawn_key=(i,)``, поэтому первые три ребёнка
        ``spawn(4)`` совпадают с ``spawn(3)``: включение шоков не сдвигает
        дождь, инъекции, спавн и отказы, и ветки одного seed остаются
        сравнимыми попарно. Парк стартует без rng — водитель ``i`` в зоне
        ``i mod Z`` (§10.14). Внутренние id заказов — счётчик с 1 в порядке
        создания (§10.11). ``price_fn`` — шов §10.13: подменяет числовой
        множитель тарифа зон, розыгрыши среды и шоков продолжаются.
        ``driver_signal_fn`` — шов сигнала водителям (change
        ``world-elastic-supply``): ``None`` — сигнал равен фактическому тарифу
        зоны на минуту.

        При ``drivers.callout.enabled`` парк — ``drivers.count + callout.part``:
        штатные ``[0, count)`` живут по сменам, подработчики ``[count, count +
        part)`` стоят вне линии до первой минуты и выходят только по вызову
        (design, решения 1 и 3).
        """

        self._cfg = cfg
        self._seed = cfg.seed if seed is None else seed
        endo_seq, exo_seq, cancel_seq, shock_seq = np.random.SeedSequence(self._seed).spawn(4)
        self._rng = np.random.default_rng(endo_seq)  # эндогенный: спавн (шаг 2)
        self._rng_env = np.random.default_rng(exo_seq)  # экзогенный: среда (шаг 1)
        self._rng_cancel = np.random.default_rng(cancel_seq)  # отказ пассажира (шаг 3)
        self._rng_shock = np.random.default_rng(shock_seq)  # шоки спроса (шаг 1)
        self._price_fn = price_fn
        self._driver_signal_fn = driver_signal_fn
        self._city = build_city(cfg)
        self._env = WorldEnv(cfg, self._rng_env, self._rng_shock)
        self._callout = cfg.drivers.callout
        part = self._callout.part if self._callout.enabled else 0
        self._fleet_size = cfg.drivers.count + part
        self._fleet = Fleet(self._fleet_size, len(self._city.zones))
        self._next_trip_id = 1  # внутренние id заказов: счётчик с 1 (§10.11)
        # Зеркало свободных водителей (резерв §5 — производительность): списки
        # отсортированных id по зонам. Состояние парка меняется в четырёх
        # точках, и все — в движке: назначение в шаге 3 (исключение из
        # зеркала), отправка в переезд в шаге 3а (исключение), высадки и
        # прибытия из переездов ``Fleet.tick`` в шаге 4 (возврат). Старт без
        # rng — водитель ``i`` в зоне ``i mod Z`` (§10.14), поэтому начальное
        # заполнение детерминировано. Кэш ISO-строк сим-минут: ``to_iso``
        # чистая функция времени, строки минут повторяются между заказами (§5).
        zone_count = len(self._city.zones)
        self._free_ids: list[list[int]] = [[] for _ in range(zone_count)]
        for driver_id in range(cfg.drivers.count):
            self._free_ids[driver_id % zone_count].append(driver_id)
        self._iso_cache: dict[datetime, str] = {}
        # Состояние вызова подработчиков (design, решение 3) — приватные поля
        # движка, как ``_free_ids``: новых состояний и методов у ``Fleet`` нет.
        # Резерв — подработчики вне линии, ещё не вызванные (отсортирован);
        # очередь вызванных — ``(момент выхода, зона, driver_id)``; окончания
        # сессий — ``{driver_id: момент}``; ушедшие после заказа по метке
        # ухода — множество id, ждущих возврата в резерв на шаге 1в.
        self._part_ids: tuple[int, ...] = tuple(range(cfg.drivers.count, self._fleet_size))
        self._part_reserve: list[int] = []
        self._part_called: list[tuple[datetime, int, int]] = []
        self._part_session_end: dict[int, datetime] = {}
        self._part_leaving: set[int] = set()
        for driver_id in self._part_ids:
            # До первой минуты прогона все подработчики вне линии и в резерве.
            self._fleet.go_offline(driver_id)
            self._part_reserve.append(driver_id)
        # Счётчики стоимости меры (спека simulator-loop): ведутся только в
        # основной фазе, ``run`` обнуляет их в начале.
        self._callouts = 0
        self._callout_minutes = 0

    def _tariff_mult(self, zone_id: int, t: datetime) -> float:
        """Источник тарифа (design, решение 4): ``price_fn`` или ``WorldEnv``.

        Единственная точка подмены — используется для λ-эластичности спавна,
        ``price_multiplier`` заказа и поминутного множителя агрегата.
        """

        if self._price_fn is not None:
            return self._price_fn(zone_id, t)
        return self._env.price_multiplier(zone_id, t)

    def world_tariff(self, zone_id: int, t: datetime) -> float:
        """Тариф среды без политики (change ``m6-policy``, спека simulator-loop).

        Публичное чтение ``WorldEnv.price_multiplier`` — множитель, который
        дала бы среда независимо от шва ``price_fn``; политики компонуют
        итоговый тариф умножением на него. Чтение чистое: rng не тратит,
        среду не двигает и от того, задан ли ``price_fn``, не зависит.
        """

        return self._env.price_multiplier(zone_id, t)

    def _iso(self, moment: datetime | None) -> str | None:
        """ISO-строка сим-минуты с кэшем (резерв §5): ``to_iso`` — чистая функция.

        Сим-время — наивные минуты от ``sim_start``, множество значений конечно
        (дней × 1440) и многократно повторяется между строками trips, поэтому
        кэш сокращает повторные ``strftime``; значения идентичны прямому
        ``to_iso``.
        """

        if moment is None:
            return None
        cached = self._iso_cache.get(moment)
        if cached is None:
            cached = to_iso(moment)
            self._iso_cache[moment] = cached
        return cached

    def _nearest_free_id(
        self,
        origin: int,
        *,
        max_pickup_min: int | None = None,
        travel_k: float = 1.0,
    ) -> int | None:
        """Ближайший свободный водитель — семантика ``Fleet.nearest_free`` (§3.5).

        Резерв §5: полный проход парка O(D) на каждого ожидающего заменён
        минимумом по зонам O(Z) по зеркалу ``self._free_ids``. Выбор идентичен:
        минимум времени подачи ``int(path_min[origin][z])`` среди непустых зон,
        при равенстве — меньший ``driver_id`` (списки зон отсортированы по
        возрастанию id, поэтому кандидат зоны — её первый элемент). Пропуск
        пустых зон эквивалентен пропуску занятых водителей в проходе парка.

        ``max_pickup_min`` — радиус поиска (change ``world-dispatch-radius``):
        если фактическая подача ближайшего водителя ``travel_minutes(path,
        travel_k)`` длиннее предела, водителя нет. Подача монотонна по пути,
        поэтому проверки ближайшего достаточно. Радиус rng не тратит.
        """

        best_id: int | None = None
        best_time: int | None = None
        path_row = self._city.path_min[origin]
        for zone_id, ids in enumerate(self._free_ids):
            if not ids:
                continue
            candidate = int(path_row[zone_id])
            if best_time is None or candidate < best_time:
                best_time, best_id = candidate, ids[0]
            elif candidate == best_time:
                best_id = min(best_id, ids[0])
        if best_id is None or max_pickup_min is None:
            return best_id
        if travel_minutes(int(best_time), travel_k) > max_pickup_min:
            return None
        return best_id

    def _trip_row(self, active: _ActiveTrip, status: str, dropoff_at: datetime | None) -> dict:
        """Строка-дикт ``trips`` в терминальном состоянии (§10.12, решение 5 design.md).

        Времена — ISO-строки через ``to_iso`` (контракт M1); у ``unfulfilled``
        все три времени NULL. ``price_multiplier`` — множитель зоны происхождения
        на момент создания заказа (§3.4); ключи — под ``storage.repo.insert_trips``.
        """

        return {
            "created_at": self._iso(active.trip.created_at),
            "origin_zone": active.trip.origin,
            "dest_zone": active.trip.dest,
            "assigned_at": self._iso(active.assigned_at),
            "pickup_at": self._iso(active.pickup_at),
            "dropoff_at": self._iso(dropoff_at),
            "status": status,
            "price_multiplier": active.trip.price_multiplier,
        }

    def _part_offline_count(self) -> int:
        """Число подработчиков вне линии (``OFFLINE``) — для сверки шага смен.

        Подработчики — хвост id ``[drivers.count, count + part)``; вне линии
        стоят и резерв, и вызванные, ещё не вышедшие (design, решение 3).
        """

        if not self._part_ids:
            return 0
        drivers = self._fleet._drivers
        return sum(
            1 for driver_id in self._part_ids if drivers[driver_id].state is DriverState.OFFLINE
        )

    def _part_online_count(self) -> int:
        """Число подработчиков на линии (в любом состоянии, кроме ``offline``)."""

        return len(self._part_ids) - self._part_offline_count()

    def _part_leaving_count(self) -> int:
        """Число подработчиков с меткой ухода (для сверки цели смен за O(part)).

        Считается по фактическим меткам ``Fleet``: подработчик с истёкшей
        сессией, уже ушедший по высадке, метку теряет и в это число не входит,
        хотя до шага возврата в резерв ещё лежит в ``_part_leaving``.
        """

        if not self._part_ids:
            return 0
        return sum(1 for driver_id in self._part_ids if self._fleet.is_leaving(driver_id))

    def _apply_callouts(
        self,
        t: datetime,
        k: int,
        signal_by_zone: np.ndarray,
    ) -> None:
        """Шаг 1в цикла — вызов подработчиков (спека simulator-loop, решение 5).

        Порядок внутри шага: (1) вернуть в резерв ушедших после заказа или
        переезда по метке ухода → (2) закрыть истёкшие сессии: свободный уходит
        с линии сразу, занятый получает метку ухода и уйдёт по завершении
        текущей работы → (3) выпустить вызванных, у кого наступил момент выхода
        (они доступны диспатчу этой же минуты) → (4) на минутах ``k mod
        every_min == 0`` — решения чистого правила ``plan_callouts`` по
        текущему резерву и сигналу водителям. Шаг детерминирован и rng не
        тратит. Вызывается только в основной фазе при ``drivers.callout.enabled``.
        """

        drivers = self._fleet._drivers
        # (1) Возврат ушедших: парк перевёл их в OFFLINE по метке ухода.
        for driver_id in sorted(self._part_leaving):
            if drivers[driver_id].state is DriverState.OFFLINE:
                self._part_leaving.discard(driver_id)
                insort(self._part_reserve, driver_id)
        # (2) Закрытие истёкших сессий: свободный уходит сразу, занятый — меткой.
        for driver_id in sorted(self._part_session_end):
            if t < self._part_session_end[driver_id]:
                continue
            state = drivers[driver_id].state
            if state is DriverState.OFFLINE:
                # Водителя уже нет на линии (вернулся в резерв шагом 1) — сессия закрыта.
                del self._part_session_end[driver_id]
            elif state is DriverState.FREE:
                self._free_ids[drivers[driver_id].zone].remove(driver_id)
                self._fleet.go_offline(driver_id)
                del self._part_session_end[driver_id]
                insort(self._part_reserve, driver_id)
            else:
                # Занятый доезжает текущий заказ или переезд и уходит по завершении.
                self._fleet.mark_leaving(driver_id)
                self._part_leaving.add(driver_id)
                del self._part_session_end[driver_id]
        # (3) Выход вызванных: свободен в зоне вызова, доступен диспатчу минуты.
        while self._part_called and self._part_called[0][0] <= t:
            _due, zone, driver_id = self._part_called.pop(0)
            self._fleet.go_online(driver_id, zone)
            insort(self._free_ids[zone], driver_id)
            self._part_session_end[driver_id] = add_minutes(t, self._callout.session_min)
        # (4) Решения раз в every_min минут по текущему резерву.
        if k % self._callout.every_min == 0 and self._part_reserve:
            for driver_id, zone in plan_callouts(self._part_reserve, signal_by_zone, self._callout):
                # Вызванный сразу покидает резерв, а на линию выйдет на шаге 3
                # минуты ``t + lag_min``; все вызванные уходят из резерва.
                self._part_reserve.remove(driver_id)
                insort(
                    self._part_called,
                    (add_minutes(t, self._callout.lag_min), zone, driver_id),
                )
                self._callouts += 1
        # Стоимость меры: минуты подработчиков на линии на конец шага вызова.
        self._callout_minutes += self._part_online_count()

    def _apply_shifts(self, t: datetime) -> None:
        """Шаг 1б цикла — сверка числа водителей на смене с целью минуты.

        Change ``world-driver-shifts`` (спека simulator-loop): ``O`` — водители
        на смене без метки ухода. При ``O > T(t)`` уходят свободные по
        убыванию ``id`` сразу, затем остаток набирается метками ухода занятым
        по убыванию ``id``. При ``O < T(t)`` сначала снимаются метки по
        возрастанию ``id``, затем ``offline`` выходят на смену в домашней зоне
        ``id mod Z`` и доступны диспатчу этой же минуты. Шаг детерминирован и
        rng не тратит. Вызывается только в основной фазе при
        ``drivers.shifts.enabled``.

        Считается и двигается только штатный парк ``[0, drivers.count)``
        (change ``world-elastic-supply``, решение 4): ``O = count −
        (offline_count − part_offline) − (leaving_count − part_leaving)``, где
        счётчики подработчиков ведутся движком.
        """

        count = self._cfg.drivers.count
        zone_count = len(self._city.zones)
        target = online_target(self._cfg.drivers, t)
        online = (
            count
            - (self._fleet.offline_count() - self._part_offline_count())
            - (self._fleet.leaving_count() - self._part_leaving_count())
        )
        drivers = self._fleet._drivers
        if online > target:
            excess = online - target
            # Свободные уходят первыми — по убыванию id, сразу.
            for driver_id in range(count - 1, -1, -1):
                if excess == 0:
                    break
                driver = drivers[driver_id]
                if driver.state is not DriverState.FREE:
                    continue
                zone = driver.zone
                self._fleet.go_offline(driver_id)
                self._free_ids[zone].remove(driver_id)
                excess -= 1
            # Остаток — занятым без метки: уйдут по завершении работы.
            for driver_id in range(count - 1, -1, -1):
                if excess == 0:
                    break
                driver = drivers[driver_id]
                if self._fleet.is_leaving(driver_id):
                    continue
                if driver.state in (
                    DriverState.TO_PICKUP,
                    DriverState.ON_TRIP,
                    DriverState.RELOCATING,
                ):
                    self._fleet.mark_leaving(driver_id)
                    excess -= 1
        elif online < target:
            need = target - online
            # Сначала снимаем метки — водитель передумал уходить.
            for driver_id in range(count):
                if need == 0:
                    break
                if self._fleet.is_leaving(driver_id):
                    self._fleet.unmark_leaving(driver_id)
                    need -= 1
            # Затем впускаем offline по возрастанию id в домашнюю зону id mod Z.
            for driver_id in range(count):
                if need == 0:
                    break
                if drivers[driver_id].state is not DriverState.OFFLINE:
                    continue
                zone = driver_id % zone_count
                self._fleet.go_online(driver_id, zone)
                insort(self._free_ids[zone], driver_id)
                need -= 1

    def _apply_dropoffs(
        self,
        t: datetime,
        in_flight: dict[int, _ActiveTrip],
        buffer: AggregateBuffer,
        trip_batch: list[tuple[int, dict]],
    ) -> list[_ActiveTrip]:
        """Шаг 4 цикла — движение: тик парка, высадки → ``completed`` (design, решение 5).

        ``fleet.tick()`` возвращает высадки минуты ``(driver_id, зона)``; заказ
        получает ``dropoff_at = t``, буфер — счётчик завершений зоны, строка
        trips — в батч (сортировка батча по внутреннему id делает trip_id
        SQLite «в порядке создания» внутри батча, §10.11).
        """

        landed: list[_ActiveTrip] = []
        dropoff_pairs = self._fleet.tick()
        # Прибытия из переездов (change m6-supply-response): водитель свободен в
        # зоне цели — отдельный от высадок канал, возврат в зеркало (§5).
        # Водитель с меткой ухода (change world-driver-shifts) стал offline —
        # в зеркало не возвращается.
        for driver_id, dest_zone in self._fleet.arrivals():
            if self._fleet._drivers[driver_id].state is DriverState.FREE:
                insort(self._free_ids[dest_zone], driver_id)
        for driver_id, dest_zone in dropoff_pairs:
            active = in_flight.pop(driver_id)
            # Водитель освободился в зоне высадки — возврат в зеркало (§5):
            # вторая из двух точек смены состояния парка, см. __init__.
            # Уходящий со смены не возвращается — заказ при этом completed.
            if self._fleet._drivers[driver_id].state is DriverState.FREE:
                insort(self._free_ids[dest_zone], driver_id)
            buffer.add_completion(dest_zone)
            trip_batch.append((active.internal_id, self._trip_row(active, _TERMINAL_COMPLETED, t)))
            landed.append(active)
        return landed

    def _apply_expiries(
        self,
        t: datetime,
        waiting: list[_ActiveTrip],
        buffer: AggregateBuffer,
        trip_batch: list[tuple[int, dict]],
    ) -> tuple[list[_ActiveTrip], list[_ActiveTrip]]:
        """Шаг 5 цикла — истечения неназначенных заказов (решение §10.3).

        Правило лимита — приватный предикат ``demand._is_expired`` (единый
        источник, решение 5 change ``m2-demand``); назначение «спасает» от
        истечения, поэтому проверка идёт только по ещё ожидающим. Истёкшие
        получают строку ``unfulfilled`` (NULL-времена) и покидают очередь.
        """

        still_waiting: list[_ActiveTrip] = []
        expired: list[_ActiveTrip] = []
        for active in waiting:
            if _is_expired(active.trip, t, self._cfg, assigned=False):
                buffer.add_expiry(active.trip.origin)
                trip_batch.append(
                    (active.internal_id, self._trip_row(active, _TERMINAL_UNFULFILLED, None))
                )
                expired.append(active)
            else:
                still_waiting.append(active)
        return still_waiting, expired

    def _flush_batch(
        self,
        trip_batch: list[tuple[int, dict]],
        metric_batch: list[dict],
        writer: TripWriter,
    ) -> int:
        """Выгрузить непустые батчи writer'у; вернуть число строк trips (§10.12).

        Сортировка trips по внутреннему id — trip_id SQLite назначается в
        порядке создания внутри батча (§10.11, решение 5 design.md).
        """

        written = 0
        if trip_batch:
            rows = [row for _trip_id, row in sorted(trip_batch, key=lambda pair: pair[0])]
            writer.write_trips(rows)
            written = len(rows)
            trip_batch.clear()
        if metric_batch:
            writer.write_metrics(metric_batch)
            metric_batch.clear()
        return written

    def run(
        self,
        days: int,
        writer: TripWriter,
        *,
        progress: Callable[[datetime, int, float], None] | None = None,
        on_minute: Callable[[MinuteRecord], None] | None = None,
        on_metrics_row: Callable[[dict], None] | None = None,
    ) -> Summary:
        """Выполнить прогон ``days`` сим-суток и вернуть ``Summary`` (§3.7, §5).

        ``days`` — число суток (целое; дробное значение допускается для коротких
        микропрогонов проверок — «2 часа симуляции» задачи 2.7). Строки trips
        пишутся только в терминальном состоянии: после последней сим-минуты
        выполняется дренаж хвоста (шаги 4–5 без спавна, диспатча и агрегации),
        затем финальная выгрузка непустых батчей. Wall-замер — ``time.perf_counter``
        только по периметру; механика мира wall-clock не читает (§2.2 п. 3).

        ``on_minute`` — шов наблюдения change ``live-engine`` (спека
        simulator-live): при заданном колбэке после каждой обработанной
        сим-минуты — основной фазы и дренажа хвоста — вызывается ``MinuteRecord``
        (simulator/live.py). Наблюдение — чистое чтение мира: rng не тратится,
        порядок разыгрываний, строки таблиц и ``Summary`` не меняются; при
        ``None`` сбор событий не выполняется и поведение движка как до change'а
        (design.md, решение 6).

        ``on_metrics_row`` — шов наблюдения закрытых бакетов change
        ``m6-policy`` (спека simulator-loop): при заданном колбэке на шаге 6
        (закрытие 5-минутного окна) он вызывается один раз на каждую строку
        окна — те же дикты, что уходят writer-батчу, в порядке возрастания
        ``zone_id`` (порядок сброса ``AggregateBuffer``). Наблюдение — чистое
        чтение: rng не тратится, строки таблиц и ``Summary`` не меняются; при
        ``None`` поведение как до change'а. За хвост дренажа метрики не
        закрываются — наблюдатель получает только бакеты основной фазы.
        """

        start_wall = time.perf_counter()
        total_minutes = int(round(days * 1440))
        sim_start = from_iso(self._cfg.engine.sim_start + " 00:00:00")
        agg_step = self._cfg.engine.agg_step_min
        event_mult = self._cfg.events.friday_evening.demand_mult
        zone_count = len(self._city.zones)
        buffer = AggregateBuffer(zone_count)
        # Счётчики стоимости меры относятся к прогону, а не к объекту движка.
        self._callouts = 0
        self._callout_minutes = 0
        callout_enabled = self._callout.enabled

        waiting: list[_ActiveTrip] = []  # ожидающие назначения
        in_flight: dict[int, _ActiveTrip] = {}  # назначенные, ключ — driver_id
        trip_batch: list[tuple[int, dict]] = []
        metric_batch: list[dict] = []

        completed = 0
        unfulfilled = 0
        cancelled = 0
        wait_sum = 0
        wait_count = 0
        deficit_rows = 0
        metric_rows = 0
        trips_written = 0

        for k in range(total_minutes):
            t = add_minutes(sim_start, k)
            # Шаг 1: среда — на 00:00 суток розыгрыш эпизодов на сутки (§3.6).
            if k % 1440 == 0:
                self._env.advance_day(k // 1440)
            # Шаг шоков спроса (change world-demand-shocks): при включённых
            # шоках — поминутный шаг среды и множители зон этой минуты. Шаг
            # безусловен и не зависит от тарифа и политики; при выключенных
            # шоках он пропускается, а спавн получает None (множитель 1.0).
            shock_mult_by_zone: np.ndarray | None = None
            if self._cfg.demand.shocks.enabled:
                self._env.advance_minute(t)
                shock_mult_by_zone = np.array(
                    [self._env.demand_factor(zone.id, t) for zone in self._city.zones],
                    dtype=float,
                )
            # Шаг 1а: поминутное накопление среды каждой зоны (design, решение 3).
            is_rain = self._env.is_rain(t)
            price_mult_by_zone = np.array(
                [self._tariff_mult(zone.id, t) for zone in self._city.zones], dtype=float
            )
            # Сигнал водителям (change world-elastic-supply, решение 6): без шва
            # это тот же массив, что тариф, поэтому отклик предложения побитово
            # прежний. Шов отделяет «что видят водители» от «что платит
            # пассажир»: спрос и price_multiplier заказов его не читают.
            if self._driver_signal_fn is None:
                signal_by_zone = price_mult_by_zone
            else:
                signal_by_zone = np.array(
                    [self._driver_signal_fn(zone.id, t) for zone in self._city.zones],
                    dtype=float,
                )
            for zone in self._city.zones:
                buffer.set_env(
                    zone.id,
                    is_rain,
                    self._env.event_label(zone, t),
                    float(price_mult_by_zone[zone.id]),
                )
            # Шаг 1б: смены водителей (change world-driver-shifts) — до спавна,
            # чтобы вышедшие были доступны диспатчу этой же минуты, а ушедшие
            # его не получили. При выключенных сменах шаг пропускается целиком.
            if self._cfg.drivers.shifts.enabled:
                self._apply_shifts(t)
            # Шаг 1в: вызов подработчиков (change world-elastic-supply) — после
            # смен и до спавна: вышедший по вызову доступен диспатчу этой же
            # минуты. При выключенном вызове шаг пропускается целиком.
            if callout_enabled:
                self._apply_callouts(t, k, signal_by_zone)
            # Списки событий минуты — только при наблюдателе (design, решение 6):
            # при ``on_minute=None`` ни списки, ни записи не строятся.
            if on_minute is not None:
                live_spawns: list[SpawnEvent] = []
                live_assignments: list[AssignEvent] = []
                live_dropoffs: list[DropoffEvent] = []
                live_expiries: list[ExpiryEvent] = []
                live_cancellations: list[CancelEvent] = []
            # Шаг 2: спавн — λ уже содержит все множители мира (design, решение 2).
            for trip in spawn_requests(
                self._rng,
                self._city,
                t,
                is_rain=is_rain,
                event_mult=event_mult,
                price_mult_by_zone=price_mult_by_zone,
                cfg=self._cfg,
                shock_mult_by_zone=shock_mult_by_zone,
            ):
                internal_id = self._next_trip_id
                self._next_trip_id += 1
                buffer.add_request(trip.origin)
                waiting.append(_ActiveTrip(internal_id=internal_id, trip=trip))
                if on_minute is not None:
                    live_spawns.append(
                        SpawnEvent(
                            internal_id=internal_id,
                            origin=trip.origin,
                            dest=trip.dest,
                            price_multiplier=trip.price_multiplier,
                        )
                    )
            # Шаг 3: диспатч по возрастанию (created_at, внутренний id) (§3.5).
            # Резерв производительности §5: при полном парке свободных водителей
            # нет и ``nearest_free`` вернул бы ``None`` каждому ожидающему, поэтому
            # шаг пропускается целиком. Диспатч детерминирован и rng не потребляет
            # (розыгрыши — только в среде и спавне), поэтому пропуск не меняет ни
            # поток разыгрываний, ни строки таблиц; очередь к следующей минуте
            # всё равно сортируется перед диспатчем. Размер парка — ``fleet_size``
            # (с подработчиками), иначе занятые подработчики молча выключали бы
            # поиск свободной машины (решение 8).
            if waiting and len(in_flight) < self._fleet_size:
                waiting.sort(key=lambda active: (active.trip.created_at, active.internal_id))
                still_waiting: list[_ActiveTrip] = []
                travel_k = self._cfg.weather.travel_mult if is_rain else 1.0
                dispatch = self._cfg.drivers.dispatch
                for active in waiting:
                    origin = active.trip.origin
                    # Радиус поиска (change world-dispatch-radius): водитель за
                    # радиусом не назначается, заказ остаётся ждать следующей
                    # минуты и обход очереди продолжается.
                    driver_id = self._nearest_free_id(
                        origin,
                        max_pickup_min=dispatch.max_pickup_min,
                        travel_k=travel_k,
                    )
                    if driver_id is None:
                        still_waiting.append(active)
                        continue
                    driver_zone = self._fleet._drivers[driver_id].zone
                    pickup_min = travel_minutes(
                        int(self._city.path_min[driver_zone, origin]), travel_k
                    )
                    # Отказ пассажира от долгой подачи (change world-dispatch-radius):
                    # розыгрыш — отдельный третий поток rng, только при p > 0, ровно
                    # одна равномерная величина на предложение в порядке обхода
                    # очереди. При отказе заказ терминален, водитель остаётся в
                    # зеркале, назначение/avg_wait/in_flight не трогаются.
                    cancel_p = _cancel_prob(pickup_min, dispatch)
                    if cancel_p > 0.0 and self._rng_cancel.random() < cancel_p:
                        buffer.add_expiry(origin)
                        trip_batch.append(
                            (
                                active.internal_id,
                                self._trip_row(active, _TERMINAL_CANCELLED, None),
                            )
                        )
                        cancelled += 1
                        if on_minute is not None:
                            live_cancellations.append(
                                CancelEvent(
                                    internal_id=active.internal_id,
                                    origin=origin,
                                    pickup_min=pickup_min,
                                )
                            )
                        continue
                    trip_min = travel_minutes(
                        int(self._city.path_min[origin, active.trip.dest]), travel_k
                    )
                    self._fleet.assign(driver_id, active.internal_id, pickup_min)
                    # Водитель занят — исключение из зеркала (§5): первая из двух
                    # точек смены состояния парка, см. __init__.
                    self._free_ids[driver_zone].remove(driver_id)
                    # Приватный канал плана поездки m2-drivers (design.md решение 6):
                    # зона назначения и длительность — без расширения сигнатур Б.
                    self._fleet._plan_trip(driver_id, active.trip.dest, trip_min)
                    active.assigned_at = t
                    # Фактический момент посадки по механике ``Fleet.tick``: счётчик
                    # подачи тикает уже в шаге 4 этой же минуты, поэтому посадка —
                    # минута ``t + pickup_min - 1`` (при подаче 1 минуту — та же
                    # минута, что назначение); тогда ``dropoff - pickup == trip_min``.
                    active.pickup_at = add_minutes(t, pickup_min - 1)
                    active.driver_id = driver_id
                    in_flight[driver_id] = active
                    if on_minute is not None:
                        live_assignments.append(
                            AssignEvent(
                                internal_id=active.internal_id,
                                driver_id=driver_id,
                                driver_zone=driver_zone,
                                origin=origin,
                                dest=active.trip.dest,
                                pickup_at=active.pickup_at,
                                dropoff_at=add_minutes(active.pickup_at, trip_min),
                            )
                        )
                    wait_min = int((t - active.trip.created_at).total_seconds() // 60)
                    buffer.add_assignment(origin, wait_min)
                    wait_sum += wait_min
                    wait_count += 1
                waiting = still_waiting
            # Шаг 3а: отклик предложения — переезды свободных к надбавке
            # (change m6-supply-response). Только при ``enabled`` и на минутах
            # решения; вход — зеркало свободных после диспатча и сигнал
            # водителям минуты (change world-elastic-supply: без шва
            # ``driver_signal_fn`` это фактический тариф). Правило
            # детерминировано и rng не тратит.
            reposition = self._cfg.drivers.reposition
            if reposition.enabled and k % reposition.every_min == 0:
                travel_k = self._cfg.weather.travel_mult if is_rain else 1.0
                for driver_id, dest_zone in plan_relocations(
                    self._free_ids,
                    signal_by_zone,
                    self._city.path_min,
                    reposition,
                ):
                    source_zone = self._fleet._drivers[driver_id].zone
                    minutes = travel_minutes(
                        int(self._city.path_min[source_zone, dest_zone]), travel_k
                    )
                    self._fleet.relocate(driver_id, dest_zone, minutes)
                    # Водитель уходит в переезд — исключение из зеркала (§5),
                    # третья точка смены состояния парка, см. __init__.
                    self._free_ids[source_zone].remove(driver_id)
            # Шаг 4: движение — посадки, высадки и прибытия минуты.
            landed = self._apply_dropoffs(t, in_flight, buffer, trip_batch)
            completed += len(landed)
            # Шаг 5: истечения неназначенных (после диспатча — §10.3).
            waiting, expired = self._apply_expiries(t, waiting, buffer, trip_batch)
            unfulfilled += len(expired)
            if on_minute is not None:
                live_dropoffs.extend(
                    DropoffEvent(
                        internal_id=active.internal_id,
                        driver_id=active.driver_id,
                        dest_zone=active.trip.dest,
                    )
                    for active in landed
                )
                live_expiries.extend(
                    ExpiryEvent(internal_id=active.internal_id, origin=active.trip.origin)
                    for active in expired
                )
            # Шаг 6: на каждой agg_step-й минуте — снапшот парка и сброс окна (§10.2).
            if (k + 1) % agg_step == 0:
                buffer.snapshot_drivers(self._fleet.free_by_zone(), self._fleet.busy_by_zone())
                closed_rows = buffer.flush(bucket5(t))
                metric_batch.extend(closed_rows)
                metric_rows += len(closed_rows)
                deficit_rows += sum(1 for row in closed_rows if _is_deficit(row))
                # Шов наблюдения m6-policy: наблюдателю уходят те же дикты строк,
                # что и writer-батчу, в порядке возрастания zone_id (порядок
                # сброса буфера). Чистое чтение: rng не тратится, таблицы и
                # Summary не меняются.
                if on_metrics_row is not None:
                    for row in closed_rows:
                        on_metrics_row(row)
            # Шов наблюдения live-engine: запись минуты после шагов 1–6 — среда
            # шага 1а и снапшот свободного парка на конец минуты; события минуты
            # в порядке обработки цикла. Чистое чтение: rng не тратится.
            if on_minute is not None:
                on_minute(
                    MinuteRecord(
                        t=t,
                        is_rain=is_rain,
                        price_mult=tuple(float(mult) for mult in price_mult_by_zone),
                        free_by_zone=tuple(int(count) for count in self._fleet.free_by_zone()),
                        spawns=tuple(live_spawns),
                        assignments=tuple(live_assignments),
                        dropoffs=tuple(live_dropoffs),
                        expiries=tuple(live_expiries),
                        cancellations=tuple(live_cancellations),
                        online_drivers=self._fleet_size - self._fleet.offline_count(),
                        part_online=self._part_online_count(),
                    )
                )
            # Батчи раз в сим-час — после шага 6 (§5, design решение 5);
            # progress получает границу часа (t следующей минуты — ровные :00).
            if (k + 1) % 60 == 0:
                trips_written += self._flush_batch(trip_batch, metric_batch, writer)
                if progress is not None:
                    elapsed = time.perf_counter() - start_wall
                    progress(add_minutes(t, 1), trips_written, elapsed)

        # Дренаж хвоста (proposal, вопрос 1): шаги 4–5 без спавна, диспатча и
        # агрегации до терминального состояния всех заказов; метрики не пишутся.
        # С наблюдателем каждая минута дренажа излучает запись (спека
        # simulator-live): spawns/assignments пусты, события — только высадки
        # и истечения оставшихся заказов, а среда вычисляется чистыми чтениями
        # ``env.is_rain``/``_tariff_mult`` на минуту записи (rng не тратится).
        t = add_minutes(sim_start, total_minutes - 1)
        while waiting or in_flight:
            t = add_minutes(t, 1)
            landed = self._apply_dropoffs(t, in_flight, buffer, trip_batch)
            completed += len(landed)
            waiting, expired = self._apply_expiries(t, waiting, buffer, trip_batch)
            unfulfilled += len(expired)
            if on_minute is not None:
                on_minute(
                    MinuteRecord(
                        t=t,
                        is_rain=self._env.is_rain(t),
                        price_mult=tuple(
                            float(self._tariff_mult(zone.id, t)) for zone in self._city.zones
                        ),
                        free_by_zone=tuple(int(count) for count in self._fleet.free_by_zone()),
                        spawns=(),
                        assignments=(),
                        dropoffs=tuple(
                            DropoffEvent(
                                internal_id=active.internal_id,
                                driver_id=active.driver_id,
                                dest_zone=active.trip.dest,
                            )
                            for active in landed
                        ),
                        expiries=tuple(
                            ExpiryEvent(internal_id=active.internal_id, origin=active.trip.origin)
                            for active in expired
                        ),
                        online_drivers=self._fleet_size - self._fleet.offline_count(),
                        part_online=self._part_online_count(),
                    )
                )
        # Финальная выгрузка: строки trips, терминальные в хвосте, не теряются.
        trips_written += self._flush_batch(trip_batch, metric_batch, writer)

        trips_total = completed + unfulfilled + cancelled
        return Summary(
            days=days,
            seed=self._seed,
            trips_total=trips_total,
            completed=completed,
            unfulfilled=unfulfilled,
            unfulfilled_share=100.0 * unfulfilled / trips_total if trips_total else 0.0,
            cancelled=cancelled,
            cancelled_share=100.0 * cancelled / trips_total if trips_total else 0.0,
            avg_wait_min=wait_sum / wait_count if wait_count else 0.0,
            deficit_share=100.0 * deficit_rows / metric_rows if metric_rows else 0.0,
            wall_sec=time.perf_counter() - start_wall,
            callouts=self._callouts,
            callout_minutes=self._callout_minutes,
        )
