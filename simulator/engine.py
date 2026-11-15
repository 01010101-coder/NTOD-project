"""Минутный цикл симулятора M2: ``Engine``, ``Summary`` и шов ``TripWriter`` (§3.7 плана M2).

Публичный контракт — приложение Б плана M2 (docs/plan-m2-engine.md): ``Engine``
связывает среду, спрос, диспатч и движение в детерминированный минутный цикл с
фиксированным порядком шагов (часть контракта детерминизма, §10.1), агрегирует
город через ``simulator.aggregate.AggregateBuffer`` и отдаёт строки таблиц
интерфейсу ``TripWriter`` — движок не знает про SQLite (§10.12). Вся случайность
мира идёт через ДВА дочерних ``numpy.random.Generator``, высаженных из
``np.random.SeedSequence(seed).spawn(2)`` в фиксированном порядке
[эндогенный, экзогенный] (§3 плана M5+M6, ``docs/plan-m5-m6-policy.md``):
эндогенный потребляют пуассоновский спавн и выбор назначения, экзогенный —
суточные розыгрыши среды ``WorldEnv`` (дождь, surge-инъекции); расход
эндогенного потока не сдвигает экзогенный, поэтому ветки одного seed с разными
политиками цены сравнимы попарно. Wall-clock читается
только по периметру ``run`` — для ``progress.elapsed_sec`` и ``Summary.wall_sec``
(§5). Шов ``price_fn`` (§10.13) подменяет числовой тариф зон, не трогая
розыгрыши среды: ``env.advance_day`` вызывается безусловно, экзогенный поток
не сдвигается, ветки A/B на одном seed сравнимы попарно.
"""

import time
from bisect import insort
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from common.config import AppConfig
from common.timeutil import add_minutes, bucket5, from_iso, to_iso
from simulator.aggregate import AggregateBuffer, _is_deficit
from simulator.city import build_city
from simulator.demand import TripRequest, _is_expired, spawn_requests
from simulator.drivers import Fleet, travel_minutes
from simulator.env import WorldEnv
from simulator.live import AssignEvent, DropoffEvent, ExpiryEvent, MinuteRecord, SpawnEvent
from simulator.reposition import plan_relocations

_TERMINAL_COMPLETED = "completed"
_TERMINAL_UNFULFILLED = "unfulfilled"


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
    ``avg_wait_min`` — среднее ожидание назначения по всем назначенным заказам
    (0.0 при отсутствии назначений — proposal, вопрос 2); ``deficit_share`` —
    доля дефицитных 5-минуток по правилу §3.8 в процентах; ``wall_sec`` —
    wall-время прогона (замер только по периметру ``run``, §5).
    """

    days: int
    seed: int
    trips_total: int
    completed: int
    unfulfilled: int
    unfulfilled_share: float
    avg_wait_min: float
    deficit_share: float
    wall_sec: float


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
    агрегации (design, решение 3); (2) спавн; (3) диспатч; (3а) отклик
    предложения — переезды свободных к надбавке при включённом
    ``drivers.reposition`` (change ``m6-supply-response``); (4) движение;
    (5) истечения; (6) на каждой ``agg_step_min``-й минуте — снапшот парка и
    сброс окна агрегата. Батчи строк writer'у уходят раз в сим-час (§5). После
    основной фазы — дренаж хвоста (proposal, вопрос 1): шаги 4–5 без спавна,
    диспатча и агрегации до терминального состояния всех заказов; строки метрик
    за хвост не пишутся.
    """

    def __init__(
        self,
        cfg: AppConfig,
        *,
        seed: int | None = None,
        price_fn: Callable[[int, datetime], float] | None = None,
    ) -> None:
        """Собрать мир по конфигу: два rng-потока (§3 плана M5+M6), среда, парк, шов тарифа.

        ``seed=None`` → ``cfg.seed``; из ``np.random.SeedSequence(seed).spawn(2)``
        создаются два дочерних генератора в фиксированном порядке
        [эндогенный, экзогенный]: эндогенный передаётся ``spawn_requests``
        (пуассоновский спавн, выбор назначения), экзогенный — ``WorldEnv``
        (дождь, surge-инъекции); расход одного потока не сдвигает другой, что
        обеспечивает сравнимость веток одного seed. Парк стартует без rng —
        водитель ``i`` в зоне ``i mod Z`` (§10.14). Внутренние id заказов —
        счётчик с 1 в порядке создания (§10.11). ``price_fn`` — шов §10.13:
        подменяет числовой множитель тарифа зон, розыгрыши среды продолжаются.
        """

        self._cfg = cfg
        self._seed = cfg.seed if seed is None else seed
        endo_seq, exo_seq = np.random.SeedSequence(self._seed).spawn(2)
        self._rng = np.random.default_rng(endo_seq)  # эндогенный: спавн (шаг 2)
        self._rng_env = np.random.default_rng(exo_seq)  # экзогенный: среда (шаг 1)
        self._price_fn = price_fn
        self._city = build_city(cfg)
        self._env = WorldEnv(cfg, self._rng_env)
        self._fleet = Fleet(cfg.drivers.count, len(self._city.zones))
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

    def _nearest_free_id(self, origin: int) -> int | None:
        """Ближайший свободный водитель — семантика ``Fleet.nearest_free`` (§3.5).

        Резерв §5: полный проход парка O(D) на каждого ожидающего заменён
        минимумом по зонам O(Z) по зеркалу ``self._free_ids``. Выбор идентичен:
        минимум времени подачи ``int(path_min[origin][z])`` среди непустых зон,
        при равенстве — меньший ``driver_id`` (списки зон отсортированы по
        возрастанию id, поэтому кандидат зоны — её первый элемент). Пропуск
        пустых зон эквивалентен пропуску занятых водителей в проходе парка.
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
        for driver_id, dest_zone in self._fleet.arrivals():
            insort(self._free_ids[dest_zone], driver_id)
        for driver_id, dest_zone in dropoff_pairs:
            active = in_flight.pop(driver_id)
            # Водитель освободился в зоне высадки — возврат в зеркало (§5):
            # вторая из двух точек смены состояния парка, см. __init__.
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

        waiting: list[_ActiveTrip] = []  # ожидающие назначения
        in_flight: dict[int, _ActiveTrip] = {}  # назначенные, ключ — driver_id
        trip_batch: list[tuple[int, dict]] = []
        metric_batch: list[dict] = []

        completed = 0
        unfulfilled = 0
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
            # Шаг 1а: поминутное накопление среды каждой зоны (design, решение 3).
            is_rain = self._env.is_rain(t)
            price_mult_by_zone = np.array(
                [self._tariff_mult(zone.id, t) for zone in self._city.zones], dtype=float
            )
            for zone in self._city.zones:
                buffer.set_env(
                    zone.id,
                    is_rain,
                    self._env.event_label(zone, t),
                    float(price_mult_by_zone[zone.id]),
                )
            # Списки событий минуты — только при наблюдателе (design, решение 6):
            # при ``on_minute=None`` ни списки, ни записи не строятся.
            if on_minute is not None:
                live_spawns: list[SpawnEvent] = []
                live_assignments: list[AssignEvent] = []
                live_dropoffs: list[DropoffEvent] = []
                live_expiries: list[ExpiryEvent] = []
            # Шаг 2: спавн — λ уже содержит все множители мира (design, решение 2).
            for trip in spawn_requests(
                self._rng,
                self._city,
                t,
                is_rain=is_rain,
                event_mult=event_mult,
                price_mult_by_zone=price_mult_by_zone,
                cfg=self._cfg,
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
            # всё равно сортируется перед диспатчем.
            if waiting and len(in_flight) < self._cfg.drivers.count:
                waiting.sort(key=lambda active: (active.trip.created_at, active.internal_id))
                still_waiting: list[_ActiveTrip] = []
                travel_k = self._cfg.weather.travel_mult if is_rain else 1.0
                for active in waiting:
                    origin = active.trip.origin
                    driver_id = self._nearest_free_id(origin)
                    if driver_id is None:
                        still_waiting.append(active)
                        continue
                    driver_zone = self._fleet._drivers[driver_id].zone
                    pickup_min = travel_minutes(
                        int(self._city.path_min[driver_zone, origin]), travel_k
                    )
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
            # решения; вход — зеркало свободных после диспатча и фактический
            # тариф минуты. Правило детерминировано и rng не тратит.
            reposition = self._cfg.drivers.reposition
            if reposition.enabled and k % reposition.every_min == 0:
                travel_k = self._cfg.weather.travel_mult if is_rain else 1.0
                for driver_id, dest_zone in plan_relocations(
                    self._free_ids,
                    price_mult_by_zone,
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
                    )
                )
        # Финальная выгрузка: строки trips, терминальные в хвосте, не теряются.
        trips_written += self._flush_batch(trip_batch, metric_batch, writer)

        trips_total = completed + unfulfilled
        return Summary(
            days=days,
            seed=self._seed,
            trips_total=trips_total,
            completed=completed,
            unfulfilled=unfulfilled,
            unfulfilled_share=100.0 * unfulfilled / trips_total if trips_total else 0.0,
            avg_wait_min=wait_sum / wait_count if wait_count else 0.0,
            deficit_share=100.0 * deficit_rows / metric_rows if metric_rows else 0.0,
            wall_sec=time.perf_counter() - start_wall,
        )
