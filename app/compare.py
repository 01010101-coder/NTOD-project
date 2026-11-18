"""Сравнение веток вехи M5+M6: исходы тест-недели по БД миров (change ``m6-compare-tab``).

Чистый модуль аналитики поверх записанных БД веток (design решение 1): сервер
только кэширует и отдаёт ответ, весь расчёт проверяется ручным SQL. Три
функции:

- :func:`test_window` — полуинтервал тест-недели
  ``[world_start + ml.train_days, + ml.test_days)`` по ``created_at``/``ts``;
- :func:`branch_outcomes` — исходы одной ветки: завершённые, невыполненные и
  отменённые заказы (``unserved`` = истёкшие + отказы), ``deterred``/``saved``
  относительно A, среднее ожидание подачи (``pickup_at − created_at``
  завершённых, а не ``avg_wait_min`` агрегатов — это ожидание назначения,
  design решение 2), выручка по формуле §10.15 M2, доля дефицитных
  зона-бакетов (правило контракта, design решение 3), переключения тарифа на
  зону в сутки, часы вызванных водителей из ``meta.json`` и часовой ряд
  кумулятивной выручки;
- :func:`accuracy_vs_base` — MAE спроса и precision/recall/F1 дефицита
  тень-моделей ``mlp``/``lstm`` **против факта ветки A** (design решение 6):
  факт B/C/D искажён действием политики (self-fulfilling prophecy);
- :func:`build_compare` — сборка ответа ``GET /api/ml/compare``: ветки в
  порядке реестра ``list_worlds``, дельты к A в процентах (server-side, design
  решение 8), пустые исходы у отсутствующих БД.

БД веток открываются только на чтение (``mode=ro``), файлы миров не
изменяются. Пустая/чужая БД даёт нули и ``null`` без исключений — сценарии
спеки «Ветка без БД» и «Ни одной БД».
"""

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np

from common.config import AppConfig
from common.timeutil import from_iso, to_iso
from ml.dataset import SqliteRowSource, world_start
from ml.metrics import deficit_metrics, join_facts, regression_metrics
from simulator.branches import list_worlds
from simulator.city import build_city

BASE_BRANCH = "a-base"
"""Ветка-база сравнения: мир без политики, неискажённый факт для точности."""

SHADOW_FAMILIES = ("mlp", "lstm")
"""Семейства тень-моделей в ``forecasts`` веток (``model_version`` = семейство)."""

SHADOW_HORIZONS = (15, 30)
"""Горизонты прогноза контракта данных, по которым считается точность."""

ABSOLUTE_OUTCOMES = (
    "trips",
    "unfulfilled",
    "cancelled",
    "unserved",
    "avg_wait_min",
    "revenue",
    "deficit_share",
    "price_switches",
    "callout_hours",
)
"""Абсолютные исходы ветки — по ним строятся дельты к A (спека, change
``world-final-config``): ``unserved = unfulfilled + cancelled``, переключения
цены и часы вызова абсолютны и сравнимы между ветками."""

RELATIVE_OUTCOMES = ("deterred", "saved")
"""Исходы, уже относительные к A (отпугнуто/спасено): дельт не имеют."""

OUTCOME_KEYS = ABSOLUTE_OUTCOMES + RELATIVE_OUTCOMES
"""Все исходы записи ветки в ответе ``GET /api/ml/compare``."""


class CompareWindow(NamedTuple):
    """Полуинтервал тест-недели: ``start`` включён, ``end`` — нет."""

    start: datetime
    end: datetime


def test_window(cfg: AppConfig) -> CompareWindow:
    """Окно тест-недели: ``[world_start + ml.train_days, + ml.test_days)``.

    Правило времени — то же, что у ролей датасета M3 (:func:`ml.dataset.day_role`)
    и у активации политики: политика активна с дня ``train_days + 1``, заказы
    относятся к окну по ``created_at`` (цена фиксируется при создании, §M2).
    """

    start = world_start(cfg) + timedelta(days=cfg.ml.train_days)
    return CompareWindow(start, start + timedelta(days=cfg.ml.test_days))


def _connect_ro(db: str | Path) -> sqlite3.Connection | None:
    """Read-only соединение с БД ветки или ``None`` (нет файла / ошибка открытия)."""

    if not Path(db).exists():
        return None
    try:
        return sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:
        return None


def _has_tables(conn: sqlite3.Connection, names: Sequence[str]) -> bool:
    """Есть ли в БД все таблицы ``names`` (чужая/недописанная БД — не ошибка)."""

    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    except sqlite3.Error:
        return False
    present = {row[0] for row in rows}
    return all(name in present for name in names)


def _empty_outcomes(hours: int) -> dict:
    """Нули/``null`` для БД без данных: счётчики 0, средние ``None``, ряд нулей."""

    return {
        "trips": 0,
        "unfulfilled": 0,
        "cancelled": 0,
        "unserved": 0,
        "avg_wait_min": None,
        "revenue": 0.0,
        "deficit_share": None,
        "price_switches": None,
        "callout_hours": None,
        "deterred": None,
        "saved": None,
        "revenue_series": [0.0] * max(hours, 0),
    }


def meta_path(db: str | Path) -> Path:
    """Путь метаданных ветки: ``<имя БД без .db>.meta.json`` рядом с БД."""

    path = Path(db)
    return path.parent / f"{path.stem}.meta.json"


def read_branch_meta(db: str | Path) -> dict | None:
    """Метаданные ветки (``channel``, часы вызова) или ``None``.

    Сетка до change'а ``policy-driver-signal`` метаданных не имеет — их
    отсутствие не ошибка: поля ответа становятся ``null`` (сценарий спеки).
    """

    path = meta_path(db)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _callout_hours(meta: dict | None) -> float | None:
    """Часы вызванных подработчиков тест-недели из ``meta.json`` (``null`` без поля)."""

    if meta is None:
        return None
    minutes = meta.get("callout_minutes_test")
    if minutes is None:
        return None
    return float(minutes) / 60.0


def _trip_counts(conn: sqlite3.Connection, status: str, window: CompareWindow) -> int:
    """Число заказов окна со статусом ``status`` (по ``created_at``)."""

    row = conn.execute(
        "SELECT COUNT(*) FROM trips WHERE status = ? AND created_at >= ? AND created_at < ?",
        (status, to_iso(window.start), to_iso(window.end)),
    ).fetchone()
    return int(row[0])


def _avg_wait_min(conn: sqlite3.Connection, window: CompareWindow) -> float | None:
    """Среднее ``pickup_at − created_at`` завершённых заказов окна, минуты.

    Разность считается в SQL через ``julianday`` (время БД — целые минуты),
    ``None`` — завершённых заказов с подачей в окне нет.
    """

    row = conn.execute(
        "SELECT AVG((julianday(pickup_at) - julianday(created_at)) * 1440.0)"
        " FROM trips"
        " WHERE status = 'completed' AND pickup_at IS NOT NULL"
        " AND created_at >= ? AND created_at < ?",
        (to_iso(window.start), to_iso(window.end)),
    ).fetchone()
    return None if row[0] is None else float(row[0])


def _deficit_share(conn: sqlite3.Connection, window: CompareWindow) -> float | None:
    """Доля зона-бакетов окна с дефицитом по правилу контракта.

    Правило — ``free_drivers / max(requests, 1) < 0.5`` или ``unfulfilled > 0``
    (:func:`ml.dataset.deficit_label`); ``None`` — строк метрик в окне нет.
    """

    row = conn.execute(
        "SELECT COUNT(*),"
        " SUM(CASE WHEN free_drivers * 1.0 / MAX(requests, 1) < ? OR unfulfilled > 0"
        " THEN 1 ELSE 0 END)"
        " FROM zone_metrics_5min WHERE ts >= ? AND ts < ?",
        (0.5, to_iso(window.start), to_iso(window.end)),
    ).fetchone()
    total = int(row[0] or 0)
    if not total:
        return None
    return int(row[1] or 0) / total


def _price_switches(
    conn: sqlite3.Connection, window: CompareWindow, n_zones: int, days: int
) -> float | None:
    """Переключения тарифа на зону в сутки (change ``world-final-config``).

    Смена — разница ``price_multiplier`` между соседними бакетами одной зоны
    окна (строки упорядочены по зоне и ``ts``). Делится на (число зон × число
    суток окна); ``None`` — строк метрик в окне нет. В счёт входят и
    инъекции тарифа мира: база A даёт «фон» переключений.
    """

    rows = conn.execute(
        "SELECT zone_id, price_multiplier FROM zone_metrics_5min"
        " WHERE ts >= ? AND ts < ? ORDER BY zone_id, ts",
        (to_iso(window.start), to_iso(window.end)),
    ).fetchall()
    if not rows:
        return None
    switches = 0
    previous_zone: int | None = None
    previous_mult: float | None = None
    for zone_id, multiplier in rows:
        if zone_id == previous_zone and multiplier != previous_mult:
            switches += 1
        previous_zone, previous_mult = int(zone_id), float(multiplier)
    denominator = max(n_zones, 1) * max(days, 1)
    return switches / denominator


def _revenue_and_series(
    conn: sqlite3.Connection,
    cfg: AppConfig,
    path_min: np.ndarray,
    window: CompareWindow,
    hours: int,
) -> tuple[float, list[float]]:
    """Выручка окна и её часовой кумулятивный ряд одним проходом (design решение 4).

    Стоимость заказа — ``(pricing.base_boarding + pricing.per_min_rub ×
    path_min[origin][dest]) × price_multiplier`` (§10.15 M2); строк с
    завершённым статусом окна читаются разом (~60 тыс. на ветку — доли
    секунды). Индекс часа — ``(created_at − start) // 60`` минут; последняя
    точка ряда равна итогу по построению. Матрицы путей в БД нет — ``path_min``
    приходит из ``build_city(cfg)``.
    """

    rows = conn.execute(
        "SELECT created_at, origin_zone, dest_zone, price_multiplier FROM trips"
        " WHERE status = 'completed' AND created_at >= ? AND created_at < ?"
        " ORDER BY trip_id",
        (to_iso(window.start), to_iso(window.end)),
    )
    per_hour = [0.0] * max(hours, 0)
    revenue = 0.0
    base = cfg.pricing.base_boarding
    per_min = cfg.pricing.per_min_rub
    for created_at, origin, dest, multiplier in rows:
        price = (base + per_min * int(path_min[int(origin), int(dest)])) * float(multiplier)
        revenue += price
        if per_hour:
            hour = int((from_iso(created_at) - window.start).total_seconds() // 3600)
            if 0 <= hour < len(per_hour):
                per_hour[hour] += price
    series: list[float] = []
    running = 0.0
    for value in per_hour:
        running += value
        series.append(running)
    if series:
        series[-1] = revenue  # последняя точка = итог (одна сумма, без расхождений)
    return revenue, series


def branch_outcomes(
    db: str | Path,
    cfg: AppConfig,
    path_min: np.ndarray,
    window: CompareWindow,
) -> dict:
    """Исходы одной ветки за окно тест-недели (design решения 2–4).

    Возвращает абсолютные исходы ``trips``, ``unfulfilled``, ``cancelled``,
    ``unserved`` (= истёкшие + отказы), ``avg_wait_min``, ``revenue``,
    ``deficit_share``, ``price_switches``, ``callout_hours`` (из ``meta.json``;
    ``null`` без файла), заглушки относительных ``deterred``/``saved``
    (заполняет :func:`build_compare` по ветке A) и ``revenue_series``
    (``ml.test_days × 24`` точек кумулятивной выручки). Нет файла, пустая или
    чужая БД — нули и ``null`` без исключений; соединение read-only, файл мира
    не меняется.
    """

    hours = cfg.ml.test_days * 24
    callout_hours = _callout_hours(read_branch_meta(db))
    conn = _connect_ro(db)
    if conn is None:
        empty = _empty_outcomes(hours)
        empty["callout_hours"] = callout_hours
        return empty
    try:
        if not _has_tables(conn, ("trips", "zone_metrics_5min")):
            empty = _empty_outcomes(hours)
            empty["callout_hours"] = callout_hours
            return empty
        revenue, series = _revenue_and_series(conn, cfg, path_min, window, hours)
        unfulfilled = _trip_counts(conn, "unfulfilled", window)
        cancelled = _trip_counts(conn, "cancelled", window)
        return {
            "trips": _trip_counts(conn, "completed", window),
            "unfulfilled": unfulfilled,
            "cancelled": cancelled,
            "unserved": unfulfilled + cancelled,
            "avg_wait_min": _avg_wait_min(conn, window),
            "revenue": revenue,
            "deficit_share": _deficit_share(conn, window),
            "price_switches": _price_switches(conn, window, len(cfg.city.zones), cfg.ml.test_days),
            "callout_hours": callout_hours,
            "deterred": None,
            "saved": None,
            "revenue_series": series,
        }
    except sqlite3.Error:
        empty = _empty_outcomes(hours)
        empty["callout_hours"] = callout_hours
        return empty
    finally:
        conn.close()


def _forecast_rows(db: str | Path, window: CompareWindow) -> list[dict]:
    """Строки ``forecasts`` окна по семействам ``mlp``/``lstm`` и горизонтам 15/30.

    Только прогнозы ветки A: ``model_version`` веток — имя семейства; порядок
    строк канонический (``ts, zone_id, horizon_min, model_version``), сортировка
    :func:`ml.metrics.join_facts` его сохраняет.
    """

    conn = _connect_ro(db)
    if conn is None:
        return []
    try:
        if not _has_tables(conn, ("forecasts",)):
            return []
        placeholders = ", ".join("?" for _ in SHADOW_FAMILIES)
        horizons = ", ".join("?" for _ in SHADOW_HORIZONS)
        rows = conn.execute(
            "SELECT ts, zone_id, horizon_min, pred_requests, pred_free_drivers, deficit_prob,"
            " model_version"
            f" FROM forecasts WHERE model_version IN ({placeholders})"
            f" AND horizon_min IN ({horizons}) AND ts >= ? AND ts < ?"
            " ORDER BY ts, zone_id, horizon_min, model_version",
            (*SHADOW_FAMILIES, *SHADOW_HORIZONS, to_iso(window.start), to_iso(window.end)),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [
        {
            "ts": row[0],
            "zone_id": row[1],
            "horizon_min": row[2],
            "pred_requests": row[3],
            "pred_free_drivers": row[4],
            "deficit_prob": row[5],
            "model_version": row[6],
        }
        for row in rows
    ]


def _empty_accuracy() -> dict:
    """Пустой раздел точности: источник — ветка A, строк нет."""

    return {"source": BASE_BRANCH, "rows": []}


def accuracy_vs_base(db_a: str | Path, cfg: AppConfig, window: CompareWindow) -> dict:
    """Точность тень-моделей ``mlp``/``lstm`` против факта ветки A (design решение 6).

    Переиспользует математику M3: :class:`ml.dataset.SqliteRowSource` по БД A +
    :func:`ml.metrics.join_facts` + ``regression_metrics``/``deficit_metrics``.
    По каждому семейству и горизонту — ``mae_requests`` (MAE спроса),
    ``precision``/``recall``/``f1`` детекции дефицита. Нет БД A, нет прогнозов,
    нет метрик — пустой раздел без ошибки; факт B/C/D не читается вовсе.
    """

    rows = _forecast_rows(db_a, window)
    if not rows:
        return _empty_accuracy()
    conn = _connect_ro(db_a)
    if conn is None:
        return _empty_accuracy()
    try:
        if not _has_tables(conn, ("zone_metrics_5min",)):
            return _empty_accuracy()
        count = conn.execute("SELECT COUNT(*) FROM zone_metrics_5min").fetchone()[0]
    except sqlite3.Error:
        return _empty_accuracy()
    finally:
        conn.close()
    # SqliteRowSource строит диапазон по метрикам: пустая таблица дала бы
    # исключение, а не пустой раздел (сценарий «БД без прогнозов»).
    if not count:
        return _empty_accuracy()

    try:
        source = SqliteRowSource(db_a)
    except (sqlite3.Error, ValueError):
        return _empty_accuracy()
    result_rows: list[dict] = []
    for family in SHADOW_FAMILIES:
        family_rows = [row for row in rows if row["model_version"] == family]
        if not family_rows:
            continue
        paired = join_facts(family_rows, source)
        if not paired:
            continue
        regression = regression_metrics(paired)
        deficit = deficit_metrics(paired)
        for horizon in sorted({int(row["horizon_min"]) for row in paired}):
            metrics = deficit.get(horizon, {})
            result_rows.append(
                {
                    "family": family,
                    "horizon_min": horizon,
                    "mae_requests": regression["pred_requests"][horizon]["mae"],
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1": metrics.get("f1", 0.0),
                }
            )
    return {"source": BASE_BRANCH, "rows": result_rows}


def _delta(value: float | None, base: float | None) -> float | None:
    """Относительная дельта к A в процентах; ``None`` при нулевой/пустой базе."""

    if value is None or base is None or base == 0:
        return None
    return (value - base) / base * 100.0


def _deltas(outcomes: dict, base: dict | None) -> dict:
    """Дельты абсолютных исходов ветки к A; у относительных — ``null``.

    ``None`` у всех ключей без базы; ``deterred``/``saved`` уже относительны к
    A, поэтому дельт не имеют (спека ``web-compare-tab``).
    """

    result: dict = {}
    for key in ABSOLUTE_OUTCOMES:
        result[key] = None if base is None else _delta(outcomes.get(key), base.get(key))
    for key in RELATIVE_OUTCOMES:
        result[key] = None
    return result


def build_compare(cfg: AppConfig, worlds_dir: str | Path | None = None) -> dict:
    """Собрать ответ сравнения веток (design решения 1, 5, 8).

    Ветки идут в порядке реестра ``list_worlds`` (A, B, C, D); отсутствующая БД
    даёт ``exists=false``, ``null``-исходы и ``null``-дельты. У существующей
    ветки ``channel`` — из ``meta.json`` (``null`` у старой сетки без метаданных),
    ``deterred`` = поездки A − поездки ветки, ``saved`` = необслуженные A −
    необслуженные ветки (у A — 0). Дельты считаются только для абсолютных
    исходов и только при существующей базе A (у самой A — ``null``). Раздел
    точности — всегда против факта A; без A он пуст. Форма ответа фиксирована
    спекой ``web-compare-tab``; сервер кэширует результат по сигнатурам файлов
    БД.
    """

    window = test_window(cfg)
    path_min = build_city(cfg).path_min
    entries = list_worlds(cfg, worlds_dir=worlds_dir)

    branches: list[dict] = []
    base_outcomes: dict | None = None
    base_db: Path | None = None
    for info in entries:
        if not info.exists:
            branches.append(
                {
                    "name": info.name,
                    "label": info.label,
                    "exists": False,
                    "channel": None,
                    "outcomes": dict.fromkeys(OUTCOME_KEYS),
                    "delta": dict.fromkeys(OUTCOME_KEYS),
                    "revenue_series": None,
                }
            )
            continue
        raw = branch_outcomes(info.db, cfg, path_min, window)
        meta = read_branch_meta(info.db)
        outcomes = {key: raw[key] for key in ABSOLUTE_OUTCOMES}
        outcomes.update(dict.fromkeys(RELATIVE_OUTCOMES))
        branches.append(
            {
                "name": info.name,
                "label": info.label,
                "exists": True,
                "channel": None if meta is None else meta.get("channel"),
                "outcomes": outcomes,
                "delta": None,  # заполняется после сбора базы A
                "revenue_series": raw["revenue_series"],
            }
        )
        if info.name == BASE_BRANCH:
            base_outcomes = outcomes
            base_db = info.db

    for branch in branches:
        if not branch["exists"]:
            branch["delta"] = dict.fromkeys(OUTCOME_KEYS)
            continue
        if base_outcomes is not None:
            branch["outcomes"]["deterred"] = (
                0
                if branch["name"] == BASE_BRANCH
                else base_outcomes["trips"] - branch["outcomes"]["trips"]
            )
            branch["outcomes"]["saved"] = (
                0
                if branch["name"] == BASE_BRANCH
                else base_outcomes["unserved"] - branch["outcomes"]["unserved"]
            )
        if branch["name"] == BASE_BRANCH:
            branch["delta"] = dict.fromkeys(OUTCOME_KEYS)
        else:
            branch["delta"] = _deltas(branch["outcomes"], base_outcomes)

    accuracy = (
        accuracy_vs_base(base_db, cfg, window)
        if base_db is not None and base_outcomes is not None
        else _empty_accuracy()
    )
    return {
        "window": {"start": to_iso(window.start), "end": to_iso(window.end)},
        "branches": branches,
        "accuracy": accuracy,
    }
