"""Датасет M3: окна признаков, таргеты, роли дней и нормализация (§3–§4 плана M3).

Слоистая структура (решение 2 design.md change ``m3-dataset``): сборка примеров
зависит только от протокола :class:`RowSource` — узкого интерфейса источника
строк (окно метрик, точечная метрика, расписание высадок). Единственная
реализация здесь — :class:`SqliteRowSource` поверх записанной истории; M6
подключит источник поверх живого движка без переписывания сборки.

Гарантии отсутствия утечки будущего (главный класс тестов §9 плана M3):
признаки примера (зона, ts) зависят только от строк с бакетом ≤ ts; ret-признаки
— только от поездок, назначенных не позднее ts (высадка такой поездки в будущем
легальна — это уже известное расписание); таргеты читаются из бакетов ts+H —
это метки, а не вход; нормализация обучается только на train-роли
(:class:`DatasetScalers`).

Время — наивные ``datetime``, конвертации через :mod:`common.timeutil`;
сравнения по минутным индексам от фиксированного начала :data:`_TIME_ORIGIN`
(без ``timestamp()`` — не вовлекает локальную таймзону).
"""

import pickle
import sqlite3
from bisect import bisect_right
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, NamedTuple, Protocol

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common.config import AppConfig
from common.timeutil import add_minutes, from_iso

DEFICIT_RATIO_LIMIT = 0.5
"""Порог доли свободных водителей в правиле дефицита (контракт данных, §3.8 плана M2)."""

RET_HORIZONS_MIN = (15, 30)
"""Горизонты контракта данных: шаг агрегации 5 минут, прогноз на 15 и 30 минут."""

SCALERS_ARTIFACT = Path("artifacts/dataset/scalers.pkl")
"""Путь сериализации нормализаторов (решение 6 design.md; ``artifacts/`` gitignored)."""

METRIC_FEATURES = (
    "requests",
    "completed",
    "unfulfilled",
    "free_drivers",
    "busy_drivers",
    "avg_wait_min",
    "supply_demand_ratio",
)
"""Счётные метрики зоны из ``zone_metrics_5min`` — первые 7 колонок источника."""

ENV_FEATURES = ("is_rain", "is_event", "price_multiplier")
"""Среда из ``zone_metrics_5min``; текстовая колонка ``event`` кодируется флагом 0/1
(значения симулятора: ``''`` | ``'friday_evening'`` | ``'surge_injection'``; тип
события частично восстановим по ``price_multiplier`` инъекции и календарю)."""

CALENDAR_FEATURES = (
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
    "is_weekend",
)
"""Календарные производные момента шага: sin/cos часа суток, sin/cos дня недели
(пн = 0), флаг выходного (сб/вс; пятница — будний день, как в профилях спроса
симулятора ``t.weekday() < 5``)."""

ROW_MANIFEST = tuple(METRIC_FEATURES) + tuple(ENV_FEATURES)
"""10 числовых колонок строки источника (порядок фиксирует матрицу (steps, 10))."""

FEATURE_MANIFEST = (
    tuple(METRIC_FEATURES)
    + tuple(CALENDAR_FEATURES)
    + tuple(ENV_FEATURES)
    + tuple(f"ret_{h}" for h in RET_HORIZONS_MIN)
)
"""Манифест 17 признаков на шаг окна (решение 4 design.md) — контракт для
m3-model/m3-inference: изменение набора признаков = правка этого кортежа."""

N_FEATURES = len(FEATURE_MANIFEST)
"""Число признаков на шаг окна (17)."""

# Смещения групп признаков в манифесте: метрики | календарь | среда | ret.
_METRICS_END = len(METRIC_FEATURES)
_CALENDAR_SLICE = slice(_METRICS_END, _METRICS_END + len(CALENDAR_FEATURES))
_ENV_SLICE = slice(_CALENDAR_SLICE.stop, _CALENDAR_SLICE.stop + len(ENV_FEATURES))
_RET_SLICE = slice(_ENV_SLICE.stop, _ENV_SLICE.stop + len(RET_HORIZONS_MIN))
# Среда в строке источника лежит сразу после метрик (ROW_MANIFEST = метрики + среда).
_ROW_ENV_SLICE = slice(_METRICS_END, len(ROW_MANIFEST))

# Индексы нужных колонок в строке источника (ROW_MANIFEST).
_REQUESTS_COL = ROW_MANIFEST.index("requests")
_UNFULFILLED_COL = ROW_MANIFEST.index("unfulfilled")
_FREE_DRIVERS_COL = ROW_MANIFEST.index("free_drivers")

_TIME_ORIGIN = datetime(2000, 1, 1)
"""Фиксированное начало минутных индексов (произвольная константа модуля)."""

DayRole = Literal["train", "val", "test"]
"""Роль целого дня сим-эпохи; ``None`` — время вне ролей."""

DeficitRule = Literal["legacy", "harm"]
"""Правило метки дефицита ML-контура (change ``ml-harm-deficit-label``).

``legacy`` — правило контракта данных (:data:`DEFICIT_RATIO_LIMIT` или
``unfulfilled > 0``), ``harm`` — только вред: ``unfulfilled > 0`` (истёкшие
заказы и отказы пассажиров от долгой подачи). Правило выбирается ключом
``ml.deficit_label`` и применяется одинаково во всём ML-контуре; показатели
мира (``Summary.deficit_share``, калибровка) и реактивная политика от него не
зависят.
"""


def bucket_index(ts: datetime, agg_step_min: int = 5) -> int:
    """Номер 5-минутного бакета, содержащего момент ``ts`` (метка начала бакета)."""

    minutes = int((ts - _TIME_ORIGIN).total_seconds()) // 60
    return minutes // agg_step_min


def from_bucket_index(idx: int, agg_step_min: int = 5) -> datetime:
    """Начало бакета номер ``idx`` как наивный ``datetime``."""

    return _TIME_ORIGIN + timedelta(minutes=idx * agg_step_min)


def _minute_index(ts: datetime) -> int:
    """Минутный индекс момента ``ts`` (для сравнений «назначено ≤ ts» через bisect)."""

    return int((ts - _TIME_ORIGIN).total_seconds()) // 60


def deficit_label(
    requests: float,
    free_drivers: float,
    unfulfilled: float,
    rule: DeficitRule = "legacy",
) -> bool:
    """Метка дефицита строки метрик по правилу ``rule`` (change ``ml-harm-deficit-label``).

    ``legacy`` (дефолт) — правило контракта данных: низкое обеспечение **или**
    истёкшие заказы — ``free_drivers / max(requests, 1) < 0.5`` или
    ``unfulfilled > 0`` (то же, что ``Summary.deficit_share`` движка); порог
    строгий, граница ровно 0.5 дефицитом не считается. ``harm`` — только вред:
    ``unfulfilled > 0`` (колонка считает и истёкшие заказы, и отказы пассажиров
    от долгой подачи). Значение вне двух допустимых — ``ValueError``: опечатка
    в коде не должна превращаться в тихое ``legacy``.
    """

    if rule == "legacy":
        return free_drivers / max(requests, 1) < DEFICIT_RATIO_LIMIT or unfulfilled > 0
    if rule == "harm":
        return unfulfilled > 0
    raise ValueError(f"неизвестное правило метки дефицита {rule!r}: ожидается 'legacy' или 'harm'")


def world_start(cfg: AppConfig) -> datetime:
    """Старт мира: ``engine.sim_start`` + 00:00 (тот же момент, что в движке)."""

    return from_iso(cfg.engine.sim_start + " 00:00:00")


def day_role(
    ts: datetime,
    start: datetime,
    *,
    train_days: int,
    val_days: int,
    test_days: int,
) -> DayRole | None:
    """Роль целого дня для момента ``ts`` (решение 1 design.md — чистая функция).

    Дни нумеруются от старта мира: train — дни 1..``train_days − val_days``,
    val — хвостовые ``val_days`` дней train-эпохи (ранняя остановка), test —
    ``test_days`` дней-продолжение сразу после train-эпохи. Моменты до старта
    и после test-эпохи дают ``None``.
    """

    day = (ts - start) // timedelta(days=1)
    if day < 0:  # до старта мира: floor-деление даёт −1, в роли не попадает
        return None
    if day < train_days - val_days:
        return "train"
    if day < train_days:
        return "val"
    if day < train_days + test_days:
        return "test"
    return None


def _calendar_row(t: datetime) -> np.ndarray:
    """Календарные производные момента ``t`` — 5 значений :data:`CALENDAR_FEATURES`."""

    hour = t.hour + t.minute / 60.0
    dow = t.weekday()
    angle_hour = 2.0 * np.pi * hour / 24.0
    angle_dow = 2.0 * np.pi * dow / 7.0
    return np.array(
        [
            np.sin(angle_hour),
            np.cos(angle_hour),
            np.sin(angle_dow),
            np.cos(angle_dow),
            1.0 if dow >= 5 else 0.0,
        ],
        dtype=float,
    )


def calendar_matrix(ts: datetime, steps: int, agg_step_min: int = 5) -> np.ndarray:
    """Календарь всех шагов окна: (steps, 5); последний шаг — момент ``ts``."""

    return np.stack(
        [_calendar_row(add_minutes(ts, -k * agg_step_min)) for k in range(steps - 1, -1, -1)]
    )


class MetricPoint(NamedTuple):
    """Точечные метрики бакета, нужные таргетам: спрос, свободные водители, сток."""

    requests: float
    free_drivers: float
    unfulfilled: float


class RowSource(Protocol):
    """Источник строк истории для сборки примеров (решение 2 design.md).

    Достаточно трёх запросов: окно метрик, точечная метрика и расписание
    высадок. Все времена — наивные ``datetime`` на границах бакетов.
    """

    def metrics_window(self, zone_id: int, ts: datetime, steps: int) -> np.ndarray | None:
        """Метрики ``steps`` бакетов, заканчивающихся ``ts``: (steps, 10) по
        :data:`ROW_MANIFEST`; ``None`` — если хотя бы одного бакета нет."""
        ...

    def metrics_at(self, zone_id: int, ts: datetime) -> MetricPoint | None:
        """Точечные метрики бакета ``ts`` или ``None``, если строки нет."""
        ...

    def dropoff_counts(
        self, zone_id: int, after: datetime, until: datetime, assigned_until: datetime
    ) -> int:
        """Число завершённых поездок с ``dest_zone = zone_id``, высадка которых
        попадает в бакеты ``(after, until]`` (правая граница включительно,
        левая — нет), а назначение — не позднее ``assigned_until``."""
        ...

    def time_range(self) -> tuple[datetime, datetime]:
        """Первый и последний бакет метрик источника (границы эпохи)."""
        ...


class SqliteRowSource:
    """:class:`RowSource` поверх записанной истории SQLite (БД эпохи M3).

    История загружается в память один раз при построении (решение 3 design.md):
    метрики — дикт строк по ключу (zone_id, бакет), что терпимо к «дырам»
    рукописных мини-БД; для ret-признаков — индекс высадок по парам
    (dest_zone, бакет высадки) с отсортированными минутами ``assigned_at``
    завершённых поездок (отменённые и невыполненные не высаживаются и в индекс
    не попадают). Соединение закрывается сразу после загрузки.
    """

    def __init__(self, db_path: str | Path, *, agg_step_min: int = 5) -> None:
        """Загрузить ``zone_metrics_5min`` и расписание высадок из БД ``db_path``."""

        self._step = agg_step_min
        conn = sqlite3.connect(str(db_path))
        try:
            metrics = pd.read_sql_query(
                "SELECT ts, zone_id, requests, completed, unfulfilled, free_drivers,"
                " busy_drivers, avg_wait_min, supply_demand_ratio, is_rain, event,"
                " price_multiplier FROM zone_metrics_5min",
                conn,
            )
            dropoffs = conn.execute(
                "SELECT dest_zone, dropoff_at, assigned_at FROM trips"
                " WHERE status = 'completed'"
                " AND dropoff_at IS NOT NULL AND assigned_at IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()

        self._metrics: dict[tuple[int, int], np.ndarray] = {}
        first_idx: int | None = None
        last_idx: int | None = None
        for row in metrics.itertuples(index=False):
            idx = bucket_index(from_iso(row.ts), agg_step_min)
            # NULL читается как 0.0 (окно без назначенных заказов). Тонкость:
            # pandas в колонке со смесью чисел и NULL (dtype float64) отдаёт
            # NULL как NaN, и проверка ``is None`` его не ловит — нужна явная
            # NaN-проверка (реальная эпоха M3: 4474 таких строк).
            wait = row.avg_wait_min
            vector = np.array(
                [
                    row.requests,
                    row.completed,
                    row.unfulfilled,
                    row.free_drivers,
                    row.busy_drivers,
                    0.0 if wait is None or wait != wait else wait,
                    row.supply_demand_ratio,
                    float(row.is_rain),
                    1.0 if row.event else 0.0,
                    row.price_multiplier,
                ],
                dtype=float,
            )
            self._metrics[(int(row.zone_id), idx)] = vector
            if first_idx is None or idx < first_idx:
                first_idx = idx
            if last_idx is None or idx > last_idx:
                last_idx = idx
        self._range = (
            from_bucket_index(first_idx, agg_step_min),
            from_bucket_index(last_idx, agg_step_min),
        )

        self._dropoffs: dict[tuple[int, int], list[int]] = {}
        for dest_zone, dropoff_at, assigned_at in dropoffs:
            key = (int(dest_zone), bucket_index(from_iso(dropoff_at), agg_step_min))
            self._dropoffs.setdefault(key, []).append(_minute_index(from_iso(assigned_at)))
        for assigned in self._dropoffs.values():
            assigned.sort()

    def metrics_window(self, zone_id: int, ts: datetime, steps: int) -> np.ndarray | None:
        """Окно метрик: строки ровно запрошенных бакетов, ``None`` при дыре."""

        last = bucket_index(ts, self._step)
        window = np.empty((steps, len(ROW_MANIFEST)), dtype=float)
        for k in range(steps):
            row = self._metrics.get((zone_id, last - (steps - 1 - k)))
            if row is None:
                return None
            window[k] = row
        return window

    def metrics_at(self, zone_id: int, ts: datetime) -> MetricPoint | None:
        """Точечные метрики бакета ``ts`` или ``None``."""

        row = self._metrics.get((zone_id, bucket_index(ts, self._step)))
        if row is None:
            return None
        return MetricPoint(
            float(row[_REQUESTS_COL]),
            float(row[_FREE_DRIVERS_COL]),
            float(row[_UNFULFILLED_COL]),
        )

    def dropoff_counts(
        self, zone_id: int, after: datetime, until: datetime, assigned_until: datetime
    ) -> int:
        """Число завершённых высадок в бакетах ``(after, until]`` с назначением ≤ cutoff."""

        limit = _minute_index(assigned_until)
        total = 0
        for idx in range(bucket_index(after, self._step) + 1, bucket_index(until, self._step) + 1):
            assigned = self._dropoffs.get((zone_id, idx))
            if assigned:
                total += bisect_right(assigned, limit)
        return total

    def time_range(self) -> tuple[datetime, datetime]:
        """Первый и последний бакет метрик БД."""

        return self._range


def build_window(
    source: RowSource,
    zone_id: int,
    ts: datetime,
    *,
    window_hours: float,
    agg_step_min: int = 5,
) -> np.ndarray | None:
    """Окно признаков примера (зона, ts): (steps, 17), последний шаг — бакет ts.

    ``steps = window_hours × (60 / agg_step_min)``; шаги покрывают
    ``[ts − (steps−1)·шаг, ts]``. Состав шага: метрики и среда источника,
    календарь самого момента шага и ret_15/ret_30 на момент ts (одни и те же на
    всех шагах — это знание «по состоянию на ts»). Истории меньше окна (дыра в
    бакетах) — ``None``: пример не строится.
    """

    steps = int(window_hours * 60 // agg_step_min)
    raw = source.metrics_window(zone_id, ts, steps)
    if raw is None:
        return None
    window = np.empty((steps, N_FEATURES), dtype=float)
    window[:, :_METRICS_END] = raw[:, :_METRICS_END]
    window[:, _CALENDAR_SLICE] = calendar_matrix(ts, steps, agg_step_min)
    window[:, _ENV_SLICE] = raw[:, _ROW_ENV_SLICE]
    for offset, horizon in enumerate(RET_HORIZONS_MIN):
        ret = source.dropoff_counts(zone_id, ts, add_minutes(ts, horizon), ts)
        window[:, _RET_SLICE.start + offset] = ret
    return window


class Targets(NamedTuple):
    """Таргеты примера (зона, ts) по горизонтам :data:`RET_HORIZONS_MIN`."""

    requests: tuple[float, ...]
    free_drivers: tuple[float, ...]
    deficit: tuple[bool, ...]


def build_targets(
    source: RowSource,
    zone_id: int,
    ts: datetime,
    *,
    horizons_min: Sequence[int] = RET_HORIZONS_MIN,
    label_rule: DeficitRule = "legacy",
) -> Targets | None:
    """Таргеты из точечных бакетов ts+H; нет бакета горизонта — ``None``.

    Значения — requests и free_drivers строки метрик ровно бакета ts+H; метка
    дефицита — :func:`deficit_label` по правилу ``label_rule`` (``ml.deficit_label``)
    на той же строке; признаки окна от правила не зависят.
    """

    requests: list[float] = []
    free_drivers: list[float] = []
    deficit: list[bool] = []
    for horizon in horizons_min:
        point = source.metrics_at(zone_id, add_minutes(ts, horizon))
        if point is None:
            return None
        requests.append(point.requests)
        free_drivers.append(point.free_drivers)
        deficit.append(
            deficit_label(point.requests, point.free_drivers, point.unfulfilled, rule=label_rule)
        )
    return Targets(tuple(requests), tuple(free_drivers), tuple(deficit))


class Examples(NamedTuple):
    """Примеры одной роли: сырые окна, таргеты-массивы и ключи (зона, ts)."""

    windows: np.ndarray  # (n, steps, N_FEATURES)
    requests: np.ndarray  # (n, horizons)
    free_drivers: np.ndarray  # (n, horizons)
    deficit: np.ndarray  # (n, horizons), 0.0/1.0 — метки не нормализуются
    keys: list[tuple[int, datetime]]


def collect_examples(
    source: RowSource,
    *,
    zones: Sequence[int],
    start: datetime,
    train_days: int,
    val_days: int,
    test_days: int,
    window_hours: float,
    agg_step_min: int = 5,
    horizons_min: Sequence[int] = RET_HORIZONS_MIN,
    label_rule: DeficitRule = "legacy",
) -> dict[str, Examples]:
    """Собрать примеры по всем (зона, ts) эпохи и разложить по ролям дней.

    Перебирает бакеты, где у примера есть и полная история окна, и оба таргета
    (границы берутся из :meth:`RowSource.time_range`); моменты вне ролей
    (до train/после test) пропускаются. Пустая роль получает массивы формы
    (0, …), совместимые с fit/transform. Роли считает :func:`day_role`; метка
    дефицита таргетов — правило ``label_rule`` (``ml.deficit_label``), окна и
    регрессионные таргеты от него не зависят.
    """

    steps = int(window_hours * 60 // agg_step_min)
    max_horizon = max(horizons_min)
    first_ts, last_ts = source.time_range()
    # Первый ts с полной историей окна и последний ts, для которого существует
    # бакет ts+max_horizon (иначе таргет недоступен).
    from_idx = bucket_index(first_ts, agg_step_min) + (steps - 1)
    until_idx = bucket_index(last_ts, agg_step_min) - (max_horizon // agg_step_min)

    windows: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    requests: dict[str, list[tuple[float, ...]]] = {"train": [], "val": [], "test": []}
    free_drivers: dict[str, list[tuple[float, ...]]] = {"train": [], "val": [], "test": []}
    deficit: dict[str, list[tuple[bool, ...]]] = {"train": [], "val": [], "test": []}
    keys: dict[str, list[tuple[int, datetime]]] = {"train": [], "val": [], "test": []}

    for idx in range(from_idx, until_idx + 1):
        ts = from_bucket_index(idx, agg_step_min)
        role = day_role(ts, start, train_days=train_days, val_days=val_days, test_days=test_days)
        if role is None:
            continue
        for zone_id in zones:
            window = build_window(
                source, zone_id, ts, window_hours=window_hours, agg_step_min=agg_step_min
            )
            targets = build_targets(
                source, zone_id, ts, horizons_min=horizons_min, label_rule=label_rule
            )
            if window is None or targets is None:
                continue
            windows[role].append(window)
            requests[role].append(targets.requests)
            free_drivers[role].append(targets.free_drivers)
            deficit[role].append(targets.deficit)
            keys[role].append((zone_id, ts))

    horizons = len(horizons_min)
    return {
        role: Examples(
            windows=np.stack(windows[role]) if windows[role] else np.empty((0, steps, N_FEATURES)),
            requests=(
                np.array(requests[role], dtype=float) if requests[role] else np.empty((0, horizons))
            ),
            free_drivers=(
                np.array(free_drivers[role], dtype=float)
                if free_drivers[role]
                else np.empty((0, horizons))
            ),
            deficit=(
                np.array(deficit[role], dtype=float) if deficit[role] else np.empty((0, horizons))
            ),
            keys=keys[role],
        )
        for role in ("train", "val", "test")
    }


class DatasetScalers:
    """Нормализация датасета: признаки и регрессионные таргеты (§4 плана M3).

    Три ``StandardScaler``: на 17 признаков (статистики по всем шагам окна
    совместно — решение 5 design.md), на requests и free_drivers (по одному на
    величину, оба горизонта пулом). Метки дефицита не нормализуются. Fit —
    только на train-примерах; сериализация pickle в :data:`SCALERS_ARTIFACT`.
    """

    def __init__(self) -> None:
        """Создать необученные scaler'ы (после ``fit`` — обученные)."""

        self.features = StandardScaler()
        self.requests = StandardScaler()
        self.free_drivers = StandardScaler()

    @classmethod
    def fit(
        cls, windows: np.ndarray, requests: np.ndarray, free_drivers: np.ndarray
    ) -> "DatasetScalers":
        """Обучить scaler'ы на train-примерах (формы как в :class:`Examples`)."""

        n, steps, n_features = windows.shape
        scalers = cls()
        scalers.features.fit(windows.reshape(n * steps, n_features))
        scalers.requests.fit(requests.reshape(-1, 1))
        scalers.free_drivers.fit(free_drivers.reshape(-1, 1))
        return scalers

    def transform_windows(self, windows: np.ndarray) -> np.ndarray:
        """Нормализовать окна любой формы (…, 17), сохраняя форму."""

        shape = windows.shape
        return self.features.transform(windows.reshape(-1, shape[-1])).reshape(shape)

    def transform_targets(
        self, requests: np.ndarray, free_drivers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Нормализовать таргеты-массивы (n, horizons) той же формой."""

        return (
            self.requests.transform(requests.reshape(-1, 1)).reshape(requests.shape),
            self.free_drivers.transform(free_drivers.reshape(-1, 1)).reshape(free_drivers.shape),
        )

    def save(self, path: str | Path = SCALERS_ARTIFACT) -> None:
        """Сериализовать pickle (каталог создаётся при необходимости)."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str | Path = SCALERS_ARTIFACT) -> "DatasetScalers":
        """Восстановить scaler'ы; не тот класс в файле — ``TypeError``."""

        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"в {path} не DatasetScalers, а {type(obj).__name__}")
        return obj
