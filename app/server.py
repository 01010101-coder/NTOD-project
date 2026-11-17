"""FastAPI-сервер воспроизведения (change ``replay-server``, задачи 4.1–4.4, 5.1–5.4).

REST города и метрик, статика без сборки и WebSocket-протокол управления
(design.md, решения 5–7). Лог и playhead — общее состояние процесса: все
вкладки смотрят один мир и одну позицию; продвижение — серверный тик
``ReplayController.tick`` (~10 Гц), пересечённые записи уходят каждому
подключению пачками с лимитом, клиент ведёт непрерывную интерполяцию сам.
Playhead — дробное число сим-минут от старта; пересечение минуты m —
``prev < m ≤ playhead``; клиенту после снимка минуты s идут записи с s+1
(события минуты s уже внутри снимка). Чтение лога — только срезы
``[0, filled)``: фоновый прогон (``app.replay.start_fill``) пополняет лог
одновременно с обслуживанием без блокировок (design.md, риски).

Change ``m3-ml-tab`` добавляет read-only REST ``/api/ml/*`` вкладки
«Лаборатория» поверх параллельного статичного контура: БД эпохи
(``cfg.ml.db``, соединение read-only на запрос) и каталога артефактов
обучения (``artifacts_dir``).

Change ``m6-worlds-ui`` добавляет реестр миров: мир «Эпоха» (детерминированный
прогон без политики, честная метка о rng-разделении) и ветки
``simulator.branches.list_worlds``; лог каждого мира строится фоновым
прогоном движка по правилу ветки (тень не поднимается — прогнозы UI читает
из БД мира, design решение 1), активный мир переключается REST
``POST /api/worlds/{name}/switch`` (WS-сессии получают ``state``+``snapshot``
нового мира), тень-прогнозы мира отдаёт ``GET /api/worlds/{name}/forecasts``.
"""

import asyncio
import json
import sqlite3
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.compare import build_compare
from app.paths import build_routes
from app.replay import FillRun, ReplayLog, snapshot_at, start_fill
from common.config import AppConfig
from common.timeutil import from_iso, to_iso
from ml.dataset import DeficitRule, deficit_label
from simulator.branches import BRANCHES, list_worlds
from simulator.live import MinuteRecord
from storage.schema import TABLE_FORECASTS, TABLE_ZONE_METRICS

SPEED_MIN = 0.25
"""Нижняя граница скорости воспроизведения, сим-минут за реальную секунду."""

SPEED_MAX = 600.0
"""Верхняя граница скорости: быстрый проход дня (~2.4 мин/сутки)."""

BATCH_LIMIT = 200
"""Лимит записей в одной пачке ``records`` (design.md, риски — медленный клиент)."""

TICK_INTERVAL_SEC = 0.1
"""Период серверного тика продвижения playhead, ~10 Гц (design.md, решение 6)."""

POLL_INTERVAL_SEC = 0.05
"""Период опроса отправителем playhead/края лога (пачки и log-extended)."""

STATIC_DIR = Path(__file__).parent / "static"
"""Каталог статики без сборки: страница-заглушка приёмки сервера."""

ML_HORIZONS = (15, 30)
"""Горизонты прогноза контракта данных (§2): допустимые значения фильтра ``horizon``."""

DEFICIT_PROB_BAND = 0.5
"""Порог полосы дефицита на графике лаборатории: интервалы ``deficit_prob >`` порога."""

N_ERROR_BINS = 20
"""Число равных бинов гистограммы остатков (design.md m3-ml-tab, решение 5)."""

EPOCH_WORLD = "epoch"
"""Имя мира «Эпоха» в реестре миров (первая запись, design m6-worlds-ui решение 2)."""

EPOCH_LABEL = "Эпоха — прогон без политики (после rng-разделения ≠ m3_epoch)"
"""Честная метка мира Эпоха (design m6-worlds-ui решение 2: метрики и прогнозы
«Лаборатории» читаются из ``m3_epoch.db``, карта — детерминированный прогон)."""

SWITCH_FIRST_RECORD_TIMEOUT_SEC = 60.0
"""Ожидание первой записи лога мира при переключении (мелкие первые минуты)."""


@dataclass
class WorldState:
    """Построенный мир реестра: лог, контроллер, фоновый прогон (design решение 3).

    ``ReplayController`` объявлен ниже по файлу — аннотации строковые.
    """

    name: str
    label: str
    log: ReplayLog
    controller: "ReplayController"
    fill: "FillRun | None" = None


def _connect_epoch(path: str) -> sqlite3.Connection | None:
    """Read-only соединение с БД эпохи (одно на запрос) или ``None``.

    ``mode=ro``: лаборатория — читатель контура данных, схема и индексы не
    трогаются (design.md m3-ml-tab, решения 2–3). Нет файла или ошибка
    открытия — ``None``: эндпойнты отвечают пустыми структурами, а не 500
    (спека «Пустая лаборатория не ломает сервер»).
    """

    if not Path(path).exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    conn.row_factory = sqlite3.Row
    return conn


def _epoch_forecast_rows(
    db_path: str,
    *,
    zone: int | None = None,
    model: str | None = None,
    horizon: int | None = None,
    start: datetime | None = None,
    stop: datetime | None = None,
    label_rule: DeficitRule = "legacy",
) -> list[dict]:
    """Строки ``forecasts``, дополненные фактом бакета ``ts + horizon_min``.

    Соединение — LEFT JOIN по ``zone_id`` и ``datetime(ts, '+H minutes')``
    (формат времени БД совпадает с выводом ``datetime()``), порядок — по
    ``ts`` с детерминирующими вторичными ключами. Пропущенный факт-бакет
    (хвост эпохи) даёт ``null``-поля факта. ``actual_deficit`` — правило метки
    ``label_rule`` (``ml.deficit_label`` конфигурации сервера; change
    ``ml-harm-deficit-label``). Ошибка чтения (нет таблицы, чужая схема) —
    пустой список, не исключение: требование устойчивости к отсутствию
    ML-данных.
    """

    conn = _connect_epoch(db_path)
    if conn is None:
        return []
    sql = f"""
        SELECT f.ts AS ts, f.zone_id AS zone_id, f.horizon_min AS horizon_min,
               f.pred_requests AS pred_requests,
               f.pred_free_drivers AS pred_free_drivers,
               f.deficit_prob AS deficit_prob, f.model_version AS model_version,
               m.requests AS actual_requests, m.free_drivers AS actual_free_drivers,
               m.unfulfilled AS actual_unfulfilled
        FROM {TABLE_FORECASTS} f
        LEFT JOIN {TABLE_ZONE_METRICS} m
          ON m.zone_id = f.zone_id
         AND m.ts = datetime(f.ts, '+' || f.horizon_min || ' minutes')
    """
    clauses: list[str] = []
    params: list[object] = []
    if zone is not None:
        clauses.append("f.zone_id = ?")
        params.append(zone)
    if model is not None:
        clauses.append("f.model_version = ?")
        params.append(model)
    if horizon is not None:
        clauses.append("f.horizon_min = ?")
        params.append(horizon)
    if start is not None:
        clauses.append("f.ts >= ?")
        params.append(to_iso(start))
    if stop is not None:
        clauses.append("f.ts <= ?")
        params.append(to_iso(stop))
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY f.ts, f.zone_id, f.horizon_min, f.model_version"
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        conn.close()
    result = []
    for row in rows:
        if row["actual_requests"] is None:
            actual: dict[str, Any] = {
                "actual_requests": None,
                "actual_free_drivers": None,
                "actual_deficit": None,
            }
        else:
            # Правило метки — единственная реализация
            # (:func:`ml.dataset.deficit_label`), копия в сервере не плодится;
            # правило выбирает ключ ``ml.deficit_label`` конфига сервера.
            requests, free = row["actual_requests"], row["actual_free_drivers"]
            actual = {
                "actual_requests": float(requests),
                "actual_free_drivers": float(free),
                "actual_deficit": bool(
                    deficit_label(requests, free, row["actual_unfulfilled"], rule=label_rule)
                ),
            }
        result.append(
            {
                "ts": row["ts"],
                "zone_id": row["zone_id"],
                "horizon_min": row["horizon_min"],
                "pred_requests": float(row["pred_requests"]),
                "pred_free_drivers": float(row["pred_free_drivers"]),
                "deficit_prob": float(row["deficit_prob"]),
                "model_version": row["model_version"],
                **actual,
            }
        )
    return result


def _error_histogram(values: list[float]) -> dict:
    """Гистограмма остатков: 20 равных бинов по диапазону значений.

    Вырожденный диапазон (все остатки равны) расширяется на 0.5 в обе
    стороны — единственный бин вместо деления на ноль.
    """

    if not values:
        return {"bin_edges": [], "counts": []}
    lo, hi = min(values), max(values)
    if hi == lo:
        lo, hi = lo - 0.5, hi + 0.5
    width = (hi - lo) / N_ERROR_BINS
    counts = [0] * N_ERROR_BINS
    for value in values:
        counts[min(int((value - lo) / width), N_ERROR_BINS - 1)] += 1
    return {"bin_edges": [lo + i * width for i in range(N_ERROR_BINS + 1)], "counts": counts}


def _error_by_hour(rows: list[dict]) -> list[dict]:
    """MAE остатков спроса по часу суток ``ts``; часы без строк не попадают в ответ."""

    sums: dict[int, list[float]] = {}
    for row in rows:
        hour = from_iso(row["ts"]).hour
        residual = abs(row["pred_requests"] - row["actual_requests"])
        sums.setdefault(hour, [0.0, 0])
        sums[hour][0] += residual
        sums[hour][1] += 1
    return [
        {"hour": hour, "mae": total / count, "count": count}
        for hour, (total, count) in sorted(sums.items())
    ]


def _worlds_signature(cfg: AppConfig, worlds_dir: str | Path | None) -> tuple:
    """Сигнатура кэша сравнения: путь, ``st_mtime_ns`` и размер каждой БД ветки.

    Отсутствующий файл даёт ``(путь, None, None)`` — появление БД меняет ключ и
    заставляет пересчитать ответ (сценарий «Пересчёт после появления ветки»).
    """

    signature: list[tuple] = []
    for info in list_worlds(cfg, worlds_dir=worlds_dir):
        try:
            stat = info.db.stat()
            signature.append((str(info.db), stat.st_mtime_ns, stat.st_size))
        except OSError:
            signature.append((str(info.db), None, None))
    return tuple(signature)


class ReplaySession:
    """Состояние одного WS-подключения (design.md, решение 6).

    ``next_index`` — индекс следующей записи, которую сессии ещё не отправили
    (после снимка минуты s это s+1 — события минуты s уже внутри снимка);
    общий playhead сессии не имеют. ``lock`` сериализует запись ``next_index``
    между отправителем-пачками и обработчиком seek (оба — задачи одного
    event loop'а, поэтому хватает ``asyncio.Lock``).
    """

    __slots__ = ("next_index", "lock")

    def __init__(self, next_index: int) -> None:
        """Создать сессию, начинающую поток записей с ``next_index``."""

        self.next_index = next_index
        self.lock = asyncio.Lock()


class ReplayController:
    """Общий playhead воспроизведения поверх лога (design.md, решение 6).

    Хранит позицию (дробные сим-минуты от старта мира), скорость (сим-мин/сек,
    зажим ``[SPEED_MIN, SPEED_MAX]``) и флаг воспроизведения. ``tick`` двигает
    playhead на ``speed × dt`` сим-минут, не опережая заполненный край лога:
    достигнув края, стоит на месте и ждёт пополнения — «live-край».
    """

    def __init__(self, log: ReplayLog, meta: dict[str, Any], sim_start: datetime) -> None:
        """Собрать контроллер: лог, метаданные мира, старт сим-часов для seek."""

        self.log = log
        self.meta = meta
        self.sim_start = sim_start
        self.playhead = 0.0
        self.speed = 1.0
        self.playing = False
        self._last_tick_wall = time.perf_counter()

    @property
    def index(self) -> int:
        """Текущая минута playhead: floor позиции, зажатая в заполненный диапазон."""

        filled = self.log.filled
        if filled == 0:
            return 0
        return min(int(self.playhead), filled - 1)

    def tick(self, dt: float | None = None) -> None:
        """Один шаг продвижения: ``dt`` задан явно (тесты) или замерен по стене.

        Пауза и пустой лог не двигают позицию; переполнение ``dt`` после
        подвисаний процесса ограничено минутой, чтобы рывок не перепрыгивал
        сутки воспроизведения за один тик.
        """

        now = time.perf_counter()
        if dt is None:
            dt = min(now - self._last_tick_wall, 60.0)
        self._last_tick_wall = now
        if not self.playing or dt <= 0:
            return
        filled = self.log.filled
        if filled == 0:
            return
        target = min(self.playhead + self.speed * dt, float(filled - 1))
        self.playhead = max(self.playhead, target)

    def clamp_seek(self, target: datetime) -> int:
        """Индекс минуты для seek: зажим в ``[0, filled)`` (спека протокола)."""

        minutes = int((target - self.sim_start).total_seconds() // 60)
        return max(0, min(minutes, self.log.filled - 1))


def record_to_dict(record: MinuteRecord) -> dict:
    """Сериализация ``MinuteRecord`` для пачек ``records`` (design.md, решение 7).

    Прямая: dataclass → dict, кортежи → массивы, времена — ISO-минуты.
    """

    return {
        "t": to_iso(record.t),
        "is_rain": record.is_rain,
        "price_mult": list(record.price_mult),
        "free_by_zone": list(record.free_by_zone),
        "online_drivers": record.online_drivers,
        "spawns": [
            {
                "internal_id": event.internal_id,
                "origin": event.origin,
                "dest": event.dest,
                "price_multiplier": event.price_multiplier,
            }
            for event in record.spawns
        ],
        "assignments": [
            {
                "internal_id": event.internal_id,
                "driver_id": event.driver_id,
                "driver_zone": event.driver_zone,
                "origin": event.origin,
                "dest": event.dest,
                "pickup_at": to_iso(event.pickup_at),
                "dropoff_at": to_iso(event.dropoff_at),
            }
            for event in record.assignments
        ],
        "dropoffs": [
            {
                "internal_id": event.internal_id,
                "driver_id": event.driver_id,
                "dest_zone": event.dest_zone,
            }
            for event in record.dropoffs
        ],
        "expiries": [
            {"internal_id": event.internal_id, "origin": event.origin} for event in record.expiries
        ],
        "cancellations": [
            {
                "internal_id": event.internal_id,
                "origin": event.origin,
                "pickup_min": event.pickup_min,
            }
            for event in record.cancellations
        ],
    }


def state_payload(controller: ReplayController) -> dict:
    """Сообщение ``state``: позиция, скорость, playing, диапазон лога, мир."""

    filled = controller.log.filled
    index = controller.index
    if filled:
        filled_range = {
            "count": filled,
            "first": to_iso(controller.log.record(0).t),
            "last": to_iso(controller.log.record(filled - 1).t),
        }
    else:
        filled_range = {"count": 0, "first": None, "last": None}
    return {
        "type": "state",
        "index": index,
        "position": controller.playhead,
        "t": to_iso(controller.log.record(index).t) if filled else None,
        "speed": controller.speed,
        "playing": controller.playing,
        "filled": filled_range,
        "world": dict(controller.meta),
    }


def snapshot_payload(controller: ReplayController) -> dict:
    """Сообщение ``snapshot``: среда, активные поездки и ожидание минуты playhead."""

    snap = snapshot_at(controller.log, controller.index)
    return {
        "type": "snapshot",
        "index": snap["index"],
        "t": to_iso(snap["t"]),
        "env": {
            "is_rain": snap["is_rain"],
            "price_mult": snap["price_mult"],
            "free_by_zone": snap["free_by_zone"],
        },
        "active_trips": [
            {
                "internal_id": trip["internal_id"],
                "driver_id": trip["driver_id"],
                "driver_zone": trip["driver_zone"],
                "origin": trip["origin"],
                "dest": trip["dest"],
                "assigned_at": to_iso(trip["assigned_at"]),
                "pickup_at": to_iso(trip["pickup_at"]),
                "dropoff_at": to_iso(trip["dropoff_at"]),
            }
            for trip in snap["active_trips"]
        ],
        "pending_orders": [
            {
                "internal_id": order["internal_id"],
                "origin": order["origin"],
                "created_at": to_iso(order["created_at"]),
            }
            for order in snap["pending_orders"]
        ],
    }


async def _tick_loop(app_state: Any, interval: float) -> None:
    """Фоновая задача lifespan'а: серверный тик продвижения ~10 Гц.

    Активный контроллер читается из ``app_state.replay`` на каждом тике —
    переключение мира (change ``m6-worlds-ui``) подменяет объект целиком.
    """

    while True:
        app_state.replay.tick()
        await asyncio.sleep(interval)


async def _sender_loop(
    websocket: WebSocket,
    session: ReplaySession,
    app_state: Any,
) -> None:
    """Отправка пачек ``records`` и сигналов ``log-extended`` одному клиенту.

    Цикл-опрос (каждые ``POLL_INTERVAL_SEC``): неотправленные пересечённые
    записи уходят пачками с лимитом ``BATCH_LIMIT`` до полного догоняния
    playhead; рост заполненного края — сигналом ``log-extended``. Запись
    ``next_index`` защищена блокировкой сессии — seek в приёмном цикле не
    теряет перестановку потока. Исключение транспорта (клиент ушёл) просто
    завершает отправителя: приёмный цикл заметит disconnect и уберёт задачу.
    Активный мир может смениться между итерациями (``switch``): отправитель
    замечает подмену контроллера и переворачивает порог ``notified`` на
    заполненный край нового лога — клиент уже получил ``state``+``snapshot``
    от обработчика переключения.
    """

    controller = app_state.replay
    notified = controller.log.filled
    while True:
        if controller is not app_state.replay:
            controller = app_state.replay
            notified = controller.log.filled
        filled = controller.log.filled
        sent = False
        if filled:
            upto = min(int(controller.playhead), filled - 1)
            current = session.next_index
            if current <= upto:
                stop = min(current + BATCH_LIMIT, upto + 1)
                batch = controller.log.read(current, stop)
                async with session.lock:
                    if session.next_index == current:
                        await websocket.send_json(
                            {
                                "type": "records",
                                "from_index": current,
                                "records": [record_to_dict(record) for record in batch],
                            }
                        )
                        session.next_index = stop
                        sent = True
                if sent:
                    continue  # остаток очереди — без ожидания интервала
        if filled > notified:
            await websocket.send_json(
                {
                    "type": "log-extended",
                    "filled": filled,
                    "last": to_iso(controller.log.record(filled - 1).t),
                }
            )
            notified = filled
        await asyncio.sleep(POLL_INTERVAL_SEC)


async def _handle_command(
    websocket: WebSocket,
    session: ReplaySession,
    app_state: Any,
    raw: str,
) -> None:
    """Разобрать команду клиента и ответить (спека протокола, задачи 5.1–5.3).

    Команды применяются к активному миру ``app_state.replay`` — после
    переключения мира (change ``m6-worlds-ui``) новые команды сразу его.
    """

    controller = app_state.replay

    try:
        message = json.loads(raw)
        cmd = message["cmd"]
    except (json.JSONDecodeError, TypeError, KeyError):
        await websocket.send_json(
            {"type": "error", "message": "команда должна быть JSON-объектом с ключом 'cmd'"}
        )
        return

    if cmd == "play":
        controller.playing = True
    elif cmd == "pause":
        controller.playing = False
    elif cmd == "speed":
        value = message.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            await websocket.send_json(
                {"type": "error", "message": "'speed' требует числовое поле 'value'"}
            )
            return
        controller.speed = min(max(float(value), SPEED_MIN), SPEED_MAX)
    elif cmd == "seek":
        filled = controller.log.filled
        try:
            target = from_iso(message["t"])
        except (KeyError, TypeError, ValueError):
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "'seek' требует время 't' формата 'YYYY-MM-DD HH:MM:SS'",
                }
            )
            return
        if filled == 0:
            await websocket.send_json({"type": "error", "message": "лог ещё пуст"})
            return
        index = controller.clamp_seek(target)
        controller.playhead = float(index)
        async with session.lock:
            session.next_index = index + 1  # события минуты index уже в снимке
        await websocket.send_json(state_payload(controller))
        await websocket.send_json(snapshot_payload(controller))
        return
    else:
        await websocket.send_json({"type": "error", "message": f"неизвестная команда {cmd!r}"})
        return
    await websocket.send_json(state_payload(controller))


def create_app(
    cfg: AppConfig,
    *,
    seed: int | None = None,
    days: float | None = None,
    tick_interval: float | None = TICK_INTERVAL_SEC,
    artifacts_dir: Path = Path("artifacts"),
    worlds_dir: str | Path | None = None,
) -> FastAPI:
    """Собрать FastAPI-приложение сервера (задачи 4.1–4.4, 5.1–5.4; m6-worlds-ui).

    Фоновый прогон лога стартует немедленно (design.md, решение 2) — до и
    независимо от lifespan'а. ``tick_interval=None`` отключает фоновый тик —
    тесты двигают playhead вручную через ``app.state.replay.tick``;
    production-запуск (``app/__main__.py``) использует тик по умолчанию.
    ``artifacts_dir`` — каталог артефактов обучения для ``/api/ml/*``:
    дефолт — cwd-конвенция CLI ``python -m ml train``; тесты подставляют
    tmp_path (design.md m3-ml-tab, решение 1). БД эпохи читается из
    ``cfg.ml.db`` (change ``m3-dataset``). Change ``m6-worlds-ui``: стартовый
    мир — «Эпоха» длиной ``ml.train_days + ml.test_days`` суток (``days``
    переопределяет, в т.ч. для тестов), ветки реестра строятся лениво при
    переключении; ``worlds_dir=None`` — ``policy.worlds_dir`` конфига.
    """

    seed = cfg.seed if seed is None else seed
    days = cfg.ml.train_days + cfg.ml.test_days if days is None else days
    sim_start = from_iso(f"{cfg.engine.sim_start} 00:00:00")

    def build_world(name: str, label: str) -> WorldState:
        """Собрать мир: лог + контроллер + фоновый прогон правила ветки/Эпохи.

        Эпоха и ``a-base`` — прогон без политики; контурные ветки — фабрика
        контура реестра (guard артефактов поднимает ``FileNotFoundError`` до
        старта прогона); тень не поднимается — прогнозы UI читает из БД мира
        (design m6-worlds-ui, решение 1).
        """

        log = ReplayLog()
        meta = {
            "name": name,
            "label": label,
            "seed": seed,
            "days": days,
            "zones": len(cfg.city.zones),
            "sim_start": cfg.engine.sim_start,
        }
        controller = ReplayController(log, meta, sim_start)
        fill = start_fill(log, cfg, seed=seed, days=days, branch=BRANCHES.get(name))
        return WorldState(name=name, label=label, log=log, controller=controller, fill=fill)

    built: dict[str, WorldState] = {EPOCH_WORLD: build_world(EPOCH_WORLD, EPOCH_LABEL)}
    routes = build_routes(cfg)
    # Кэш сравнения веток (change m6-compare-tab, design решение 7): ответ
    # считается при первом запросе, ключ — сигнатуры файлов БД всех веток.
    compare_cache: dict[str, Any] = {"key": None, "value": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Запустить и остановить фоновый тик продвижения playhead."""

        task = None
        if tick_interval is not None:
            task = asyncio.create_task(_tick_loop(app.state, tick_interval))
        yield
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(lifespan=lifespan)
    app.state.replay = built[EPOCH_WORLD].controller
    app.state.fill = built[EPOCH_WORLD].fill
    # WS-сессии активного мира: переключение рассылает им state+snapshot.
    ws_sessions: set[tuple[WebSocket, ReplaySession]] = set()

    @app.get("/api/city")
    def api_city() -> dict:
        """REST города (задача 4.1): зоны с map-координатами, рёбра, маршруты."""

        return {
            "zones": [
                {
                    "id": zone.id,
                    "name": zone.name,
                    "archetype": zone.archetype,
                    "x": zone.map.x,
                    "y": zone.map.y,
                }
                for zone in sorted(cfg.city.zones, key=lambda zone: zone.id)
            ],
            "roads": [
                {"from": road.from_zone, "to": road.to_zone, "minutes": road.minutes}
                for road in cfg.city.roads
            ],
            "routes": routes,
        }

    @app.get("/api/metrics")
    def api_metrics(
        zone: int | None = Query(default=None),
        frm: str | None = Query(default=None, alias="from"),
        to: str | None = Query(default=None),
    ) -> dict:
        """REST метрик (задача 4.2): строки ``zone_metrics_5min`` с фильтрами.

        Доступная часть частично заполненного лога отдаётся как есть; порядок
        строк — (ts, zone_id), естественный порядок закрытия окон прогона.
        """

        start = from_iso(frm) if frm is not None else None
        stop = from_iso(to) if to is not None else None
        keys = (
            "zone_id",
            "requests",
            "completed",
            "unfulfilled",
            "free_drivers",
            "busy_drivers",
            "avg_wait_min",
            "supply_demand_ratio",
            "is_rain",
            "event",
            "price_multiplier",
        )
        metrics = []
        for row in app.state.replay.log.metrics_rows():
            if zone is not None and row["zone_id"] != zone:
                continue
            if start is not None and row["ts"] < start:
                continue
            if stop is not None and row["ts"] > stop:
                continue
            metrics.append({"ts": to_iso(row["ts"]), **{key: row[key] for key in keys}})
        return {"metrics": metrics}

    @app.get("/api/ml/models")
    def api_ml_models() -> dict:
        """Список моделей вкладки «Лаборатория» (задача 2.1): наивы + сети.

        Наивы — ``SELECT DISTINCT model_version`` из ``forecasts`` (артефактов
        у них нет), сети — каталоги ``artifacts/*/`` с валидными
        ``metrics.json`` и непустым ``loss.json`` (кривая вшита в ответ).
        Слияние по имени версии: версия с артефактами становится ``kind="nn"``,
        остальные — ``kind="naive"``. Порядок: наивы, затем сети по имени.
        Нет ни прогнозов, ни артефактов — пустой список с кодом 200.
        """

        naive: list[dict] = []
        conn = _connect_epoch(cfg.ml.db)
        if conn is not None:
            try:
                rows = conn.execute(
                    f"SELECT DISTINCT model_version FROM {TABLE_FORECASTS} ORDER BY model_version"
                ).fetchall()
            except sqlite3.Error:
                rows = []
            finally:
                conn.close()
            naive = [{"model_version": row["model_version"], "kind": "naive"} for row in rows]

        networks: list[dict] = []
        root = Path(artifacts_dir)
        if root.is_dir():
            for metrics_path in sorted(root.glob("*/metrics.json")):
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    loss = json.loads(
                        (metrics_path.parent / "loss.json").read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(metrics, dict) or not isinstance(loss, list) or not loss:
                    continue
                networks.append(
                    {
                        "model_version": str(
                            metrics.get("model_version", metrics_path.parent.name)
                        ),
                        "kind": "nn",
                        "arch": metrics.get("arch"),
                        "window_hours": metrics.get("window_hours"),
                        "best_epoch": metrics.get("best_epoch"),
                        "loss": loss,
                    }
                )
        networks.sort(key=lambda entry: entry["model_version"])
        with_artifacts = {entry["model_version"] for entry in networks}
        naive = [entry for entry in naive if entry["model_version"] not in with_artifacts]
        return {"models": naive + networks}

    @app.get("/api/ml/summary")
    def api_ml_summary() -> dict:
        """Сводная метрик вехи (задача 2.2): passthrough последнего ``summary-*.json``.

        Файл — с максимальной датой в имени (лексикография ISO-даты совпадает
        с хронологией); содержимое отдаётся без пересчёта: сводка CLI —
        канонический источник (design.md m3-ml-tab, решение 4). Нет файла или
        он не читается — 404 с понятным сообщением.
        """

        root = Path(artifacts_dir)
        candidates = sorted(root.glob("summary-*.json")) if root.is_dir() else []
        if not candidates:
            raise HTTPException(
                status_code=404,
                detail="сводка метрик не найдена: нет artifacts/summary-*.json "
                "(прогоните python -m ml infer)",
            )
        latest = candidates[-1]
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=404, detail=f"сводка {latest.name} не читается: {exc}"
            ) from exc

    @app.get("/api/ml/forecasts")
    def api_ml_forecasts(
        zone: int | None = Query(default=None),
        model: str | None = Query(default=None),
        horizon: int | None = Query(default=None),
        frm: str | None = Query(default=None, alias="from"),
        to: str | None = Query(default=None),
    ) -> dict:
        """Прогнозы, соединённые с фактом (задача 3.1), с фильтрами и порядком ts.

        Фильтры применяются строго (все заданные — одновременно); границы
        ``from``/``to`` включаются, как в ``/api/metrics``. Неизвестная зона
        или модель — пустой список с кодом 200.
        """

        if horizon is not None and horizon not in ML_HORIZONS:
            raise HTTPException(
                status_code=422, detail="horizon: допустимы горизонты 15 и 30 минут"
            )
        start = from_iso(frm) if frm is not None else None
        stop = from_iso(to) if to is not None else None
        rows = _epoch_forecast_rows(
            cfg.ml.db,
            zone=zone,
            model=model,
            horizon=horizon,
            start=start,
            stop=stop,
            label_rule=cfg.ml.deficit_label,
        )
        return {"forecasts": rows}

    @app.get("/api/ml/errors")
    def api_ml_errors(
        zone: int | None = Query(default=None),
        model: str | None = Query(default=None),
        horizon: int | None = Query(default=None),
    ) -> dict:
        """Ошибки прогноза спроса (задача 3.2): остатки, гистограмма, MAE по часу.

        Остаток — ``pred_requests − actual_requests`` по строкам с фактом
        (те же, что отдаёт ``/api/ml/forecasts``); строки без факта
        пропускаются. ``by_hour`` относит модуль ошибки к часу суток ``ts``.
        """

        if horizon is not None and horizon not in ML_HORIZONS:
            raise HTTPException(
                status_code=422, detail="horizon: допустимы горизонты 15 и 30 минут"
            )
        rows = [
            row
            for row in _epoch_forecast_rows(
                cfg.ml.db,
                zone=zone,
                model=model,
                horizon=horizon,
                label_rule=cfg.ml.deficit_label,
            )
            if row["actual_requests"] is not None
        ]
        residuals = [
            {
                "ts": row["ts"],
                "residual": row["pred_requests"] - row["actual_requests"],
            }
            for row in rows
        ]
        return {
            "residuals": residuals,
            "histogram": _error_histogram([entry["residual"] for entry in residuals]),
            "by_hour": _error_by_hour(rows),
        }

    # ── Сравнение веток (change m6-compare-tab, design решения 1, 7, 8) ──

    @app.get("/api/ml/compare")
    def api_ml_compare() -> dict:
        """Исходы тест-недели по веткам, дельты к A, ряд выручки и точность (2.1).

        Ответ целиком считает :func:`app.compare.build_compare`; сервер лишь
        кэширует его в замыкании ``create_app`` по ключу (путь, ``st_mtime_ns``,
        ``st_size``) всех БД веток. Повторный запрос без изменения файлов не
        пересчитывает (первый запрос — самая дорогая часть: источник A);
        появление/изменение БД ветки меняет ключ и запускает пересчёт.
        """

        key = _worlds_signature(cfg, worlds_dir)
        if compare_cache["value"] is None or compare_cache["key"] != key:
            compare_cache["value"] = build_compare(cfg, worlds_dir)
            compare_cache["key"] = key
        return compare_cache["value"]

    # ── Реестр миров (change m6-worlds-ui, design решения 1–4) ──
    def world_db(name: str) -> Path | None:
        """БД мира для прогнозов: Эпоха — ``ml.db``, ветка — файл реестра."""

        if name == EPOCH_WORLD:
            return Path(cfg.ml.db)
        for info in list_worlds(cfg, worlds_dir=worlds_dir):
            if info.name == name:
                return info.db
        return None

    @app.get("/api/worlds")
    def api_worlds() -> dict:
        """REST перечня миров (спека m6-worlds-ui): Эпоха первой + ветки реестра."""

        active = app.state.replay.meta.get("name")
        entries = [
            {
                "name": EPOCH_WORLD,
                "label": EPOCH_LABEL,
                "exists": True,
                "active": active == EPOCH_WORLD,
            }
        ]
        for info in list_worlds(cfg, worlds_dir=worlds_dir):
            entries.append(
                {
                    "name": info.name,
                    "label": info.label,
                    "exists": info.exists,
                    "active": active == info.name,
                }
            )
        return {"worlds": entries}

    @app.post("/api/worlds/{name}/switch")
    async def api_world_switch(name: str) -> dict:
        """Переключение активного мира (design решение 3): позиция — на старт.

        Мир строится лениво при первом выборе (фоновый прогон правила ветки);
        обработчик ждёт первую запись лога (мелкие первые минуты), затем всем
        WS-сессиям уходят ``state``+``snapshot`` нового мира, поток записей
        каждой переставляется на снимок. Неизвестное имя — 404; ветка без
        файла БД или без артефактов модели — 409, активный мир не меняется.
        """

        if name != EPOCH_WORLD and name not in BRANCHES:
            raise HTTPException(status_code=404, detail=f"неизвестный мир {name!r}")
        if name != EPOCH_WORLD:
            db = world_db(name)
            if db is None or not db.exists():
                raise HTTPException(status_code=409, detail=f"мир {name!r} не сгенерирован")
        state = built.get(name)
        if state is None:
            try:
                label = EPOCH_LABEL if name == EPOCH_WORLD else BRANCHES[name].label
                state = build_world(name, label)
            except FileNotFoundError as exc:
                raise HTTPException(
                    status_code=409, detail=f"мир {name!r} недоступен: {exc}"
                ) from exc
            built[name] = state
        if state.controller is app.state.replay:
            return state_payload(app.state.replay)
        waited = 0.0
        while state.log.filled == 0 and waited < SWITCH_FIRST_RECORD_TIMEOUT_SEC:
            if state.fill is not None and state.fill.error is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"прогон мира {name!r} упал: {state.fill.error}",
                )
            await asyncio.sleep(POLL_INTERVAL_SEC)
            waited += POLL_INTERVAL_SEC
        # Позиция — на старт мира (design решение 3): повторный выбор мира
        # тоже начинает с начала, а не с прошлой позиции контроллера.
        state.controller.playhead = 0.0
        app.state.replay = state.controller
        app.state.fill = state.fill
        for websocket, session in list(ws_sessions):
            session.next_index = state.controller.index + 1
            await websocket.send_json(state_payload(state.controller))
            await websocket.send_json(snapshot_payload(state.controller))
        return state_payload(state.controller)

    @app.get("/api/worlds/{name}/forecasts")
    def api_world_forecasts(
        name: str,
        zone: int | None = None,
        horizon: int = cfg.policy.horizon_min,
        date_from: str | None = Query(default=None, alias="from"),
        date_to: str | None = Query(default=None, alias="to"),
    ) -> dict:
        """Тень-прогнозы БД мира (design решение 4): строки с границы активации.

        Ветки — ``model_version`` = семейство (``mlp``/``lstm``), Эпоха — полные
        версии артефактов; соединение read-only, читаются только записанные
        строки. ``from``/``to`` — ISO-границы окна (карта запрашивает бакет).
        """

        if name != EPOCH_WORLD and name not in BRANCHES:
            raise HTTPException(status_code=404, detail=f"неизвестный мир {name!r}")
        if horizon not in ML_HORIZONS:
            raise HTTPException(status_code=422, detail=f"horizon ожидается из {ML_HORIZONS}")
        db = world_db(name)
        if db is None:
            return {"world": name, "rows": []}
        conn = _connect_epoch(str(db))
        if conn is None:
            return {"world": name, "rows": []}
        query = (
            f"SELECT ts, zone_id, horizon_min, deficit_prob, model_version"
            f" FROM {TABLE_FORECASTS} WHERE horizon_min = ?"
        )
        params: list[Any] = [horizon]
        if zone is not None:
            query += " AND zone_id = ?"
            params.append(zone)
        if date_from is not None:
            query += " AND ts >= ?"
            params.append(date_from)
        if date_to is not None:
            query += " AND ts <= ?"
            params.append(date_to)
        query += " ORDER BY ts, zone_id, model_version"
        try:
            rows = [
                {
                    "ts": row["ts"],
                    "zone_id": row["zone_id"],
                    "horizon_min": row["horizon_min"],
                    "deficit_prob": row["deficit_prob"],
                    "model_version": row["model_version"],
                }
                for row in conn.execute(query, params)
            ]
        except sqlite3.Error:
            rows = []  # нет таблицы/чужая схема — пустой список, не 500
        finally:
            conn.close()
        return {"world": name, "rows": rows}

    @app.get("/")
    def index_page() -> FileResponse:
        """Статика без сборки (задача 4.3): страница-заглушка приёмки сервера."""

        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.websocket("/ws/replay")
    async def ws_replay(websocket: WebSocket) -> None:
        """WS-протокол воспроизведения (задачи 5.1–5.4; m6-worlds-ui).

        Connect: дождаться первых записей активного мира, отправить ``state`` +
        ``snapshot`` минуты playhead, запустить отправителя пачек и цикл
        команд. Каждая сессия ведёт свою позицию потока ``next_index``
        поверх общего playhead — повторный connect на ту же минуту получает
        записи заново, seek переставляет поток. Команды и поток относятся к
        активному миру ``app.state.replay``: переключение мира рассылает сессиям
        ``state``+``snapshot`` и переставляет их потоки (REST ``switch``).
        """

        await websocket.accept()
        controller = app.state.replay
        while controller.log.filled == 0 and not websocket.client_state.disconnected:
            controller = app.state.replay
            await asyncio.sleep(POLL_INTERVAL_SEC)
        session = ReplaySession(next_index=controller.index + 1)
        await websocket.send_json(state_payload(controller))
        await websocket.send_json(snapshot_payload(controller))
        ws_sessions.add((websocket, session))
        sender = asyncio.create_task(_sender_loop(websocket, session, app.state))
        try:
            while True:
                raw = await websocket.receive_text()
                await _handle_command(websocket, session, app.state, raw)
        except WebSocketDisconnect:
            pass
        finally:
            ws_sessions.discard((websocket, session))
            sender.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await sender

    return app
