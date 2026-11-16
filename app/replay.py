"""Событийный лог тихого прогона (change ``replay-server``, задачи 2.1–2.3).

Log-first (design.md, решение 1): сервер один раз тихо прогоняет мир —
``Engine`` с швом ``on_minute`` (``simulator-live``) — и дальше работает только
с логом. ``ReplayLog`` хранит append-only последовательность записей
``MinuteRecord`` и строки ``zone_metrics_5min`` того же прогона (in-memory
writer, решение 3); счётчик ``filled`` растёт монотонно, читатели видят срез
``[0, filled)`` — записи иммутабельны (frozen dataclass), ``list.append``
и присвоение int атомарны при GIL, поэтому фоновый прогон и обслуживание
не синхронизируются явно (design.md, риски).

Снимок минуты (решение 4) восстанавливает активные поездки из лога:
назначения в записях ≤ t без высадки с тем же ``internal_id`` в записях ≤ t
(истечение назначенных невозможно — механика M2), со всеми маршрутными полями
события назначения плюс минута назначения ``assigned_at`` (change
``city-canvas`` — клиент анимирует фазу «к пассажиру» без дублирования формул
движка); тем же проходом собираются неназначенные заказы ``pending_orders`` —
спавны без назначения, без истечения и без отказа пассажира к t (маркеры
ожидания после seek);
среда — значения записи t. Фоновый прогон (решение 2)
запускается в отдельном потоке при старте — обслуживание начинается немедленно,
записи и метрики пополняют лог прогрессивно.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from common.config import AppConfig
from simulator.engine import Engine
from simulator.live import MinuteRecord

if TYPE_CHECKING:
    from simulator.branches import BranchSpec


@dataclass
class MetricsCollector:
    """In-memory ``TripWriter`` фонового прогона (design.md, решение 3).

    Строки ``zone_metrics_5min`` буфера агрегации копятся списком в порядке
    закрытия окон (``(ts, zone_id)`` — окна хронологичны, зоны в окне по
    возрастанию id); строки ``trips`` серверу не нужны — восстановление мира
    идёт из записей минуты, поэтому writer их отбрасывает.
    """

    rows: list[dict] = field(default_factory=list)

    def write_trips(self, rows: list[dict]) -> None:
        """Терминальные строки ``trips`` логу не нужны — пропустить (решение 3)."""

    def write_metrics(self, rows: list[dict]) -> None:
        """Добавить батч закрытых окон ``zone_metrics_5min`` (порядок (ts, zone_id))."""

        self.rows.extend(rows)


class ReplayLog:
    """Append-only лог записей минуты и метрик тихого прогона (задача 2.1).

    Индекс списка совпадает с индексом сим-минуты: записи аппендятся швом
    ``on_minute`` подряд, от первой минуты основной фазы через дренаж хвоста.
    ``filled`` — число записей, видимых читателям; читается как атрибут
    (монотонный int, GIL-атомарно). Метрики живут в своём сборщике и отдаются
    срезом-копией списка — внешние мутации лог не трогают.
    """

    def __init__(self) -> None:
        """Создать пустой лог: без записей, счётчик заполненности 0."""

        self._records: list[MinuteRecord] = []
        self.filled = 0
        self.metrics = MetricsCollector()

    def append(self, record: MinuteRecord) -> None:
        """Добавить запись минуты (колбэк ``on_minute`` движка); ``filled`` растёт.

        Порядок важен: сначала аппенд, затем счётчик — читатель, увидевший
        новое ``filled``, гарантированно найдёт запись по индексу.
        """

        self._records.append(record)
        self.filled += 1

    def record(self, index: int) -> MinuteRecord:
        """Запись сим-минуты ``index`` (случайный доступ — seek за O(1))."""

        return self._records[index]

    def read(self, start: int = 0, stop: int | None = None) -> list[MinuteRecord]:
        """Срез записей ``[start, stop)``, зажатый в заполненный диапазон.

        ``stop=None`` означает «до заполненного края»; чтение за краем
        возвращает меньший срез, а не ошибку — лог пополняется прогрессивно.
        """

        if stop is None:
            stop = self.filled
        return self._records[start : min(stop, self.filled)]

    def metrics_rows(self) -> list[dict]:
        """Копия накопленных строк ``zone_metrics_5min`` в порядке (ts, zone_id)."""

        return list(self.metrics.rows)


def snapshot_at(log: ReplayLog, index: int) -> dict:
    """Восстановить состояние мира на конец сим-минуты ``index`` (задачи 2.2, city-canvas 1.1).

    Один проход по записям ``[0, index]``: последний словарь назначений
    ``internal_id → AssignEvent`` минус внутренние id высадок дают активные
    поездки — поля события назначения несут все маршрутные данные клиента
    (``driver_zone``, ``origin``, ``dest``, ``pickup_at``, ``dropoff_at``),
    ``assigned_at`` — сим-минута записи события назначения. Тем же проходом
    идут спавны: оставшиеся без назначения, без истечения и без отказа к
    ``index`` составляют ``pending_orders`` (``internal_id``, ``origin``, ``created_at``
    — минута спавна). Группы события записи обрабатываются в порядке цикла
    движка (спавн → назначение → высадка → истечение), поэтому состояние
    после записи — состояние конца минуты. Оба списка упорядочены по
    ``internal_id`` — порядок создания заказов (§10.11), детерминированный
    для клиента. Среда — значения записи ``index``: ``is_rain``,
    ``price_mult``, ``free_by_zone``.
    """

    assignments: dict[int, object] = {}
    assigned_at: dict[int, datetime] = {}
    landed: set[int] = set()
    spawns: dict[int, object] = {}
    spawned_at: dict[int, datetime] = {}
    for record in log.read(0, index + 1):
        for event in record.spawns:
            spawns[event.internal_id] = event
            spawned_at[event.internal_id] = record.t
        for event in record.assignments:
            assignments[event.internal_id] = event
            assigned_at[event.internal_id] = record.t
        for event in record.dropoffs:
            landed.add(event.internal_id)
        for event in record.expiries:
            spawns.pop(event.internal_id, None)
        for event in record.cancellations:
            spawns.pop(event.internal_id, None)
    record = log.record(index)
    return {
        "index": index,
        "t": record.t,
        "is_rain": record.is_rain,
        "price_mult": list(record.price_mult),
        "free_by_zone": list(record.free_by_zone),
        "active_trips": [
            {
                "internal_id": event.internal_id,
                "driver_id": event.driver_id,
                "driver_zone": event.driver_zone,
                "origin": event.origin,
                "dest": event.dest,
                "assigned_at": assigned_at[event.internal_id],
                "pickup_at": event.pickup_at,
                "dropoff_at": event.dropoff_at,
            }
            for internal_id, event in sorted(assignments.items())
            if internal_id not in landed
        ],
        "pending_orders": [
            {
                "internal_id": internal_id,
                "origin": event.origin,
                "created_at": spawned_at[internal_id],
            }
            for internal_id, event in sorted(spawns.items())
            if internal_id not in assignments
        ],
    }


@dataclass
class FillRun:
    """Фоновый прогон лога (задача 2.3): поток и итоги после завершения.

    ``thread`` — поток прогона (daemon: процесс сервера не должен висеть на
    неп-recording прогоне при выходе); ``summary``/``error`` заполняются в
    конце — ``error`` несёт исключение, если прогон упал (например, битый
    конфиг), чтобы сбой фона не стал молчаливой вечной пустотой лога.
    """

    thread: threading.Thread
    summary: object | None = field(default=None)
    error: BaseException | None = field(default=None)

    @property
    def done(self) -> bool:
        """Прогон завершён (успешно или с ошибкой)."""

        return not self.thread.is_alive()


def start_fill(
    log: ReplayLog,
    cfg: AppConfig,
    *,
    seed: int,
    days: float,
    branch: "BranchSpec | None" = None,
) -> FillRun:
    """Запустить тихий прогон движка в отдельном потоке (задача 2.3; m6-worlds-ui).

    ``seed``/``days`` приходят из CLI-флагов или конфига (флаги переопределяют,
    сборка аргументов — в ``app/__main__.py``); записи и метрики пополняют лог
    прогрессивно — колбэк ``on_minute`` и writer сборщика метрик пишут сразу.
    ``branch`` — правило ветки реестра (change ``m6-worlds-ui``, design решение 1):
    контурная фабрика строит контур до прогона (guard артефактов — исключение
    до старта), движок получает ``contour.price`` и наблюдателей контура
    (наблюдение — чистое чтение, логу оно мир не меняет); ``a-base`` и Эпоха
    (``branch=None``) — прогон без политики. Тень не поднимается: прогнозы UI
    читает из БД мира. Поток не вмешивается в обслуживание: append-only списки
    и GIL делают одновременное чтение безопасным (design.md, решение 1 и риски).
    """

    run: FillRun | None = None

    def target() -> None:
        """Тело фонового потока: прогон → Summary, исключение — в ``run.error``."""

        try:
            price_fn = None
            on_minute: Callable[[MinuteRecord], None] | None = log.append
            on_metrics_row = None
            if branch is not None and branch.contour_factory is not None:
                contour = branch.contour_factory(cfg)
                price_fn = contour.price
                contour_bind = contour.bind  # локальная ссылка до построения движка

                def on_minute_with_contour(record: MinuteRecord) -> None:
                    log.append(record)
                    contour.on_minute(record)

                on_minute = on_minute_with_contour
                on_metrics_row = contour.on_metrics_row
                engine = Engine(cfg, seed=seed, price_fn=price_fn)
                contour_bind(engine)  # поздняя привязка: тариф мира (m6-policy)
            else:
                if branch is not None and branch.price_fn_factory is not None:
                    price_fn = branch.price_fn_factory(cfg)
                engine = Engine(cfg, seed=seed, price_fn=price_fn)
            summary = engine.run(
                days, log.metrics, on_minute=on_minute, on_metrics_row=on_metrics_row
            )
        except BaseException as exc:  # noqa: BLE001 — сбой фона не должен молчать
            if run is not None:
                run.error = exc
            return
        if run is not None:
            run.summary = summary

    thread = threading.Thread(target=target, daemon=True, name="replay-fill")
    run = FillRun(thread=thread)
    thread.start()
    return run
