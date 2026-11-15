"""Тесты ml/dataset.py (change ``m3-dataset``): источник строк, манифест и
календарь, ret-признак, окно, анти-утечка, таргеты, роли дней, scaler'ы.

Реальная эпоха (``data/m3_epoch.db``) тестами не трогается (Non-Goals design.md):
все проверки — на рукописных мини-БД в ``tmp_path``. Формулы значений сеточных
фикстур повторяются в тестах через ``_expected_row``, чтобы проверять ровно
запрошенные бакеты, а не «что-то похожее».
"""

import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from common.timeutil import add_minutes, to_iso
from ml.dataset import (
    FEATURE_MANIFEST,
    METRIC_FEATURES,
    DatasetScalers,
    MetricPoint,
    SqliteRowSource,
    bucket_index,
    build_targets,
    build_window,
    calendar_matrix,
    collect_examples,
    day_role,
    deficit_label,
)
from storage.db import connect
from storage.repo import insert_trips, insert_zone_metrics
from storage.schema import init_db

SIM_START = datetime(2026, 11, 2)
"""Старт мира дефолтного конфига (понедельник)."""

NOON = datetime(2026, 11, 3, 12, 0)
"""«Текущий момент» примеров: вторник, полдень, внутри сетки grid_db."""

Z0, Z1 = 0, 1


def _metric_row(
    ts: datetime,
    zone_id: int,
    requests: int,
    free_drivers: int,
    *,
    completed: int = 1,
    unfulfilled: int = 0,
    busy_drivers: int = 2,
    avg_wait_min: float | None = 3.0,
    is_rain: int = 0,
    event: str = "",
    price_multiplier: float = 1.0,
) -> dict:
    """Строка ``zone_metrics_5min``; ratio считается по контракту от requests/free."""

    return {
        "ts": ts,
        "zone_id": zone_id,
        "requests": requests,
        "completed": completed,
        "unfulfilled": unfulfilled,
        "free_drivers": free_drivers,
        "busy_drivers": busy_drivers,
        "avg_wait_min": avg_wait_min,
        "supply_demand_ratio": free_drivers / max(requests, 1),
        "is_rain": is_rain,
        "event": event,
        "price_multiplier": price_multiplier,
    }


def _trip_row(
    created_at: datetime,
    *,
    dest_zone: int = Z0,
    assigned_at: datetime | None = None,
    pickup_at: datetime | None = None,
    dropoff_at: datetime | None = None,
    status: str = "completed",
) -> dict:
    """Строка ``trips`` с минимальным набором колонок для краевых случаев."""

    return {
        "created_at": created_at,
        "origin_zone": Z1,
        "dest_zone": dest_zone,
        "assigned_at": assigned_at,
        "pickup_at": pickup_at,
        "dropoff_at": dropoff_at,
        "status": status,
        "price_multiplier": 1.0,
    }


def _expected_row(i: int, z: int) -> np.ndarray:
    """Формулы grid_db для бакета номер ``i`` (от SIM_START) и зоны ``z`` —
    ровно те же выражения, что в фикстуре (10 колонок ROW_MANIFEST)."""

    j = i + z
    return np.array(
        [
            j % 7,  # requests
            j % 3,  # completed
            1 if j % 11 == 0 else 0,  # unfulfilled
            (j % 4) + 1,  # free_drivers
            j % 5,  # busy_drivers
            (j % 6) * 0.5,  # avg_wait_min
            ((j % 4) + 1) / max(j % 7, 1),  # supply_demand_ratio
            1.0 if j % 17 == 0 else 0.0,  # is_rain
            1.0 if j % 23 == 0 else 0.0,  # is_event
            1.0 + (j % 3) * 0.1,  # price_multiplier
        ],
        dtype=float,
    )


@pytest.fixture
def grid_db(tmp_path: Path) -> Path:
    """Мини-БД: зоны 0/1, метрики каждый 5-минутный бакет от SIM_START до
    NOON+45 мин; значения варьируются детерминированно по (бакет, зона)."""

    path = tmp_path / "grid.db"
    conn = connect(str(path))
    init_db(conn)
    rows: list[dict] = []
    base = bucket_index(SIM_START)
    t = SIM_START
    while t <= NOON + timedelta(minutes=45):
        i = bucket_index(t) - base
        for z in (Z0, Z1):
            j = i + z
            rows.append(
                _metric_row(
                    t,
                    z,
                    requests=j % 7,
                    free_drivers=(j % 4) + 1,
                    completed=j % 3,
                    unfulfilled=1 if j % 11 == 0 else 0,
                    busy_drivers=j % 5,
                    avg_wait_min=(j % 6) * 0.5,
                    is_rain=1 if j % 17 == 0 else 0,
                    event="friday_evening" if j % 23 == 0 else "",
                    price_multiplier=1.0 + (j % 3) * 0.1,
                )
            )
        t = add_minutes(t, 5)
    insert_zone_metrics(conn, rows)
    conn.close()
    return path


# ---------------------------------------------------------------------------
# 2.1 Источник строк: SqliteRowSource
# ---------------------------------------------------------------------------


def test_row_source_returns_exact_buckets(grid_db: Path) -> None:
    """Окно метрик — ровно запрошенные бакеты, значения ровно вставленных строк."""
    source = SqliteRowSource(grid_db)
    w0 = source.metrics_window(Z0, NOON, 4)
    w1 = source.metrics_window(Z1, NOON, 4)
    assert w0.shape == (4, 10)
    assert w1.shape == (4, 10)
    # Зоны различимы: одинаковые бакеты, разные значения.
    assert not np.array_equal(w0, w1)
    base = bucket_index(SIM_START)
    i_noon = bucket_index(NOON) - base
    for k, i in enumerate(range(i_noon - 3, i_noon + 1)):
        assert np.array_equal(w0[k], _expected_row(i, Z0))
        assert np.array_equal(w1[k], _expected_row(i, Z1))


def test_row_source_missing_bucket_is_none(grid_db: Path) -> None:
    """Дыра в истории (бакеты до старта мира) — окно не строится; точечный
    запрос за краем сетки — None."""
    source = SqliteRowSource(grid_db)
    assert source.metrics_window(Z0, add_minutes(SIM_START, 10), 6) is None
    assert source.metrics_at(Z0, NOON + timedelta(hours=2)) is None


def test_row_source_point_and_range(grid_db: Path) -> None:
    """Точечные метрики — тройка (requests, free_drivers, unfulfilled);
    time_range — первый и последний бакет сетки."""
    source = SqliteRowSource(grid_db)
    base = bucket_index(SIM_START)
    i_noon = bucket_index(NOON) - base
    expected = _expected_row(i_noon, Z1)
    assert source.metrics_at(Z1, NOON) == MetricPoint(
        requests=float(expected[0]),
        free_drivers=float(expected[3]),
        unfulfilled=float(expected[2]),
    )
    assert source.time_range() == (SIM_START, NOON + timedelta(minutes=45))


def test_row_source_null_avg_wait_is_zero(tmp_path: Path) -> None:
    """NULL ``avg_wait_min`` (окно без назначенных заказов) читается как 0.0."""
    path = tmp_path / "null.db"
    conn = connect(str(path))
    init_db(conn)
    insert_zone_metrics(conn, [_metric_row(SIM_START, Z0, 1, 4, avg_wait_min=None)])
    conn.close()
    source = SqliteRowSource(path)
    assert source.metrics_window(Z0, SIM_START, 1)[0, 5] == 0.0


def test_row_source_null_avg_wait_among_numbers(tmp_path: Path) -> None:
    """NULL ``avg_wait_min`` в колонке со смесью чисел и NULL — тоже 0.0.

    pandas на такой колонке даёт dtype float64 и отдаёт NULL как NaN, поэтому
    проверка ``is None`` его не ловит (реальная эпоха M3: 4474 строк с NULL
    среди числовых значений — NaN утекал в признаки и ронял обучение).
    """
    path = tmp_path / "mixed.db"
    conn = connect(str(path))
    init_db(conn)
    insert_zone_metrics(
        conn,
        [
            _metric_row(SIM_START, Z0, 1, 4, avg_wait_min=2.5),
            _metric_row(add_minutes(SIM_START, 5), Z0, 1, 4, avg_wait_min=None),
        ],
    )
    conn.close()
    source = SqliteRowSource(path)
    window = source.metrics_window(Z0, add_minutes(SIM_START, 5), 2)
    assert window[0, 5] == 2.5
    assert window[1, 5] == 0.0


# ---------------------------------------------------------------------------
# 2.2 Манифест признаков и календарные производные
# ---------------------------------------------------------------------------


def test_feature_manifest_seventeen() -> None:
    """Манифест фиксирует 17 уникальных признаков в порядке контракта."""
    assert len(FEATURE_MANIFEST) == 17
    assert len(set(FEATURE_MANIFEST)) == 17
    assert FEATURE_MANIFEST[:7] == METRIC_FEATURES
    for name in (
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
        "is_weekend",
        "is_rain",
        "is_event",
        "price_multiplier",
        "ret_15",
        "ret_30",
    ):
        assert name in FEATURE_MANIFEST


def test_calendar_noon_and_midnight() -> None:
    """Полдень: sin часа ≈ 0, cos ≈ −1; полночь: sin ≈ 0, cos ≈ 1."""
    noon = calendar_matrix(datetime(2026, 11, 3, 12, 0), 1)[0]
    assert noon[0] == pytest.approx(0.0, abs=1e-12)
    assert noon[1] == pytest.approx(-1.0)
    midnight = calendar_matrix(datetime(2026, 11, 3, 0, 0), 1)[0]
    assert midnight[0] == pytest.approx(0.0, abs=1e-12)
    assert midnight[1] == pytest.approx(1.0)


def test_calendar_weekday_weekend() -> None:
    """Понедельник и пятница — будни (is_weekend = 0), суббота — выходной;
    sin/cos дня недели — угол 2π·dow/7."""
    monday = calendar_matrix(datetime(2026, 11, 2, 9, 0), 1)[0]
    assert monday[2] == pytest.approx(0.0, abs=1e-12)
    assert monday[3] == pytest.approx(1.0)
    assert monday[4] == 0.0
    friday = calendar_matrix(datetime(2026, 11, 6, 9, 0), 1)[0]
    assert friday[4] == 0.0  # пятница — не выходной (как в профилях симулятора)
    saturday = calendar_matrix(datetime(2026, 11, 7, 9, 0), 1)[0]
    assert saturday[4] == 1.0
    assert saturday[2] == pytest.approx(np.sin(2.0 * np.pi * 5 / 7))
    assert saturday[3] == pytest.approx(np.cos(2.0 * np.pi * 5 / 7))


# ---------------------------------------------------------------------------
# 2.3 Индекс высадок: ret_15/ret_30
# ---------------------------------------------------------------------------


@pytest.fixture
def ret_db(tmp_path: Path) -> Path:
    """Мини-БД с одной строкой метрик и девятью краевыми поездками вокруг NOON."""
    path = tmp_path / "ret.db"
    conn = connect(str(path))
    init_db(conn)
    insert_zone_metrics(conn, [_metric_row(SIM_START, Z0, 1, 4)])
    m = add_minutes  # короткий алиас: строки поездок читаются одной строкой
    insert_trips(
        conn,
        [
            # высадка ровно ts+15, назначено ровно ts — входит в ret_15
            _trip_row(m(NOON, -10), assigned_at=NOON, dropoff_at=m(NOON, 15)),
            # высадка ts+10, назначено ts — входит в ret_15 и ret_30
            _trip_row(m(NOON, -10), assigned_at=NOON, dropoff_at=m(NOON, 10)),
            # высадка ровно ts — не входит: интервал (ts, ts+H] открыт слева
            _trip_row(m(NOON, -10), assigned_at=NOON, dropoff_at=NOON),
            # высадка ts+20 — только ret_30
            _trip_row(m(NOON, -10), assigned_at=NOON, dropoff_at=m(NOON, 20)),
            # высадка ровно ts+30 — правая граница включена в ret_30
            _trip_row(m(NOON, -10), assigned_at=NOON, dropoff_at=m(NOON, 30)),
            # другая зона назначения — не считается зоне 0
            _trip_row(m(NOON, -10), dest_zone=Z1, assigned_at=NOON, dropoff_at=m(NOON, 10)),
            # назначено после ts (первый бакет границы) — не учитывается
            _trip_row(m(NOON, -5), assigned_at=m(NOON, 5), dropoff_at=m(NOON, 10)),
            # отменённая поездка не высаживается — не учитывается
            _trip_row(m(NOON, -10), assigned_at=NOON, dropoff_at=m(NOON, 10), status="cancelled"),
            # невыполненная: без назначения и высадки
            _trip_row(m(NOON, -10), status="unfulfilled"),
        ],
    )
    conn.close()
    return path


def test_dropoff_counts_edge_cases(ret_db: Path) -> None:
    """ret_H: диапазон бакетов высадки (бакет ts, бакет ts+H], назначение ≤ ts,
    только completed."""
    source = SqliteRowSource(ret_db)
    assert source.dropoff_counts(Z0, NOON, add_minutes(NOON, 15), NOON) == 2
    assert source.dropoff_counts(Z0, NOON, add_minutes(NOON, 30), NOON) == 4
    assert source.dropoff_counts(Z1, NOON, add_minutes(NOON, 15), NOON) == 1
    # Расширенный интервал (бакет ts−5, бакет ts+15] захватывает бакет ts: та же
    # поездка с высадкой ровно ts (assigned ≤ ts) теперь входит — 3 вместо 2.
    assert source.dropoff_counts(Z0, add_minutes(NOON, -5), add_minutes(NOON, 15), NOON) == 3


def test_dropoff_counts_bucket_semantics(tmp_path: Path) -> None:
    """Минуты внутри бакета не различаются (бакетная семантика R4, выровнена
    с таргетом — снапшотом конца бакета ts+H): высадка ts+2 (бакет ts) не
    считается, ts+17 (бакет ts+15) считается в ret_15; хронологический подсчёт
    по минутам дал бы обратное (2 и 0 «новых» соответственно)."""
    path = tmp_path / "bucket.db"
    conn = connect(str(path))
    init_db(conn)
    insert_zone_metrics(conn, [_metric_row(SIM_START, Z0, 1, 4)])
    m = add_minutes
    insert_trips(
        conn,
        [
            _trip_row(m(NOON, -10), assigned_at=NOON, dropoff_at=m(NOON, 2)),
            _trip_row(m(NOON, -10), assigned_at=NOON, dropoff_at=m(NOON, 17)),
            _trip_row(m(NOON, -10), assigned_at=NOON, dropoff_at=m(NOON, 20)),
        ],
    )
    conn.close()
    source = SqliteRowSource(path)
    assert source.dropoff_counts(Z0, NOON, m(NOON, 15), NOON) == 1
    assert source.dropoff_counts(Z0, NOON, m(NOON, 30), NOON) == 2


# ---------------------------------------------------------------------------
# 2.4/2.5 Окно признаков и анти-утечка
# ---------------------------------------------------------------------------


def test_window_shape_and_content(grid_db: Path) -> None:
    """Окно 6 ч: 72 шага × 17 признаков; последний шаг — бакет ts, первый —
    ts−71×5 мин; ret-колонки постоянны по шагам."""
    source = SqliteRowSource(grid_db)
    window = build_window(source, Z0, NOON, window_hours=6)
    assert window.shape == (72, 17)
    base = bucket_index(SIM_START)
    i_noon = bucket_index(NOON) - base
    # Метрики — колонки 0–6 окна; среда источника — колонки 12–14 (после календаря).
    expected = _expected_row(i_noon, Z0)
    assert np.array_equal(window[-1, :7], expected[:7])
    assert np.array_equal(window[-1, 12:15], expected[7:10])
    first_expected = _expected_row(i_noon - 71, Z0)
    assert np.array_equal(window[0, :7], first_expected[:7])
    assert np.array_equal(window[0, 12:15], first_expected[7:10])
    assert np.all(window[:, 15] == window[0, 15])
    assert np.all(window[:, 16] == window[0, 16])
    # Поездок в grid_db нет — ret-признаки нулевые.
    assert window[0, 15] == 0.0 and window[0, 16] == 0.0


def test_window_three_hours(grid_db: Path) -> None:
    """Окно 3 ч: 36 шагов, первый шаг — бакет ts−35×5 мин."""
    source = SqliteRowSource(grid_db)
    window = build_window(source, Z1, NOON, window_hours=3)
    assert window.shape == (36, 17)
    base = bucket_index(SIM_START)
    i_noon = bucket_index(NOON) - base
    expected = _expected_row(i_noon - 35, Z1)
    assert np.array_equal(window[0, :7], expected[:7])
    assert np.array_equal(window[0, 12:15], expected[7:10])


def test_window_insufficient_history(grid_db: Path) -> None:
    """Истории меньше окна — пример не строится; окно ровно в размер истории
    строится (ts = SIM_START+55 мин даёт ровно 12 бакетов для окна 1 ч)."""
    source = SqliteRowSource(grid_db)
    assert build_window(source, Z0, add_minutes(SIM_START, 30), window_hours=6) is None
    assert build_window(source, Z0, add_minutes(SIM_START, 55), window_hours=1) is not None


def test_window_ignores_future_mutation(grid_db: Path) -> None:
    """Мутация строк метрик с бакетом > ts и поездок, назначенных после ts,
    не меняет окно (зона, ts)."""
    before = build_window(SqliteRowSource(grid_db), Z0, NOON, window_hours=6)
    conn = connect(str(grid_db))
    conn.execute(
        "UPDATE zone_metrics_5min SET requests = 999, free_drivers = 0, is_rain = 1,"
        " event = 'surge_injection', price_multiplier = 2.5 WHERE ts > ?",
        (to_iso(NOON),),
    )
    insert_trips(
        conn,
        [
            # назначена после NOON, высадка в рет-интервале — в признаки попасть не должна
            _trip_row(
                add_minutes(NOON, -5),
                assigned_at=add_minutes(NOON, 1),
                dropoff_at=add_minutes(NOON, 15),
            )
        ],
    )
    conn.close()
    after = build_window(SqliteRowSource(grid_db), Z0, NOON, window_hours=6)
    assert np.array_equal(before, after)


# ---------------------------------------------------------------------------
# 3.1 Таргеты
# ---------------------------------------------------------------------------


@pytest.fixture
def targets_db(tmp_path: Path) -> Path:
    """Мини-БД с известными строками бакетов NOON+15 и NOON+30 зоны 0."""
    path = tmp_path / "targets.db"
    conn = connect(str(path))
    init_db(conn)
    insert_zone_metrics(
        conn,
        [
            # базовая строка за 40 мин до NOON — только для time_range
            _metric_row(add_minutes(NOON, -40), Z0, 1, 4, avg_wait_min=None),
            _metric_row(add_minutes(NOON, 15), Z0, 10, 3, unfulfilled=0),
            _metric_row(add_minutes(NOON, 30), Z0, 4, 10, unfulfilled=1),
        ],
    )
    conn.close()
    return path


def test_targets_from_exact_buckets(targets_db: Path) -> None:
    """Таргеты берутся из строк ровно ts+15/ts+30; метка — правило контракта."""
    source = SqliteRowSource(targets_db)
    targets = build_targets(source, Z0, NOON)
    assert targets is not None
    assert targets.requests == (10.0, 4.0)
    assert targets.free_drivers == (3.0, 10.0)
    # ts+15: 3/max(10,1)=0.3 < 0.5 → дефицит; ts+30: 10/4 > 0.5, но unfulfilled=1 → дефицит
    assert targets.deficit == (True, True)


def test_deficit_label_boundaries() -> None:
    """Границы правила: ровно 0.5 — не дефицит; 0/unfulfilled=0 — дефицит."""
    assert deficit_label(requests=4, free_drivers=2, unfulfilled=0) is False
    assert deficit_label(requests=10, free_drivers=4, unfulfilled=0) is True
    assert deficit_label(requests=0, free_drivers=0, unfulfilled=0) is True
    assert deficit_label(requests=10, free_drivers=10, unfulfilled=0) is False
    assert deficit_label(requests=10, free_drivers=10, unfulfilled=1) is True


def test_targets_missing_horizon(targets_db: Path) -> None:
    """Бакет ts+30 за краем эпохи — таргет не выдаётся."""
    conn = connect(str(targets_db))
    conn.execute("DELETE FROM zone_metrics_5min WHERE ts = ?", (to_iso(add_minutes(NOON, 30)),))
    conn.close()
    source = SqliteRowSource(targets_db)
    assert build_targets(source, Z0, NOON) is None


# ---------------------------------------------------------------------------
# 3.2 Роли дней
# ---------------------------------------------------------------------------


def test_day_role_boundaries() -> None:
    """Классификация на границах суток: train дни 1–28, val 29–30, test 31–37,
    до старта и день 38 — вне ролей (дефолты train=30, val=2, test=7)."""
    start = SIM_START

    def role(ts: datetime) -> str | None:
        return day_role(ts, start, train_days=30, val_days=2, test_days=7)

    assert role(start) == "train"
    assert role(start + timedelta(days=27, hours=23, minutes=55)) == "train"
    assert role(start + timedelta(days=28)) == "val"
    assert role(start + timedelta(days=29, hours=23, minutes=55)) == "val"
    assert role(start + timedelta(days=30)) == "test"
    assert role(start + timedelta(days=36, hours=23, minutes=55)) == "test"
    assert role(start + timedelta(days=37)) is None
    assert role(start - timedelta(minutes=5)) is None


# ---------------------------------------------------------------------------
# 3.3 Сбор примеров и scaler'ы
# ---------------------------------------------------------------------------


@pytest.fixture
def days_db(tmp_path: Path) -> Path:
    """Мини-эпоха: 3 дня × 288 бакетов зоны 0 с детерминированным шумом rng(7)."""
    path = tmp_path / "days.db"
    conn = connect(str(path))
    init_db(conn)
    rng = np.random.default_rng(7)
    rows: list[dict] = []
    t = SIM_START
    end = SIM_START + timedelta(days=3) - timedelta(minutes=5)
    while t <= end:
        rows.append(
            _metric_row(
                t,
                Z0,
                requests=int(rng.integers(0, 8)),
                free_drivers=int(rng.integers(1, 6)),
                completed=int(rng.integers(0, 4)),
                unfulfilled=1 if rng.random() < 0.2 else 0,
                busy_drivers=int(rng.integers(0, 5)),
                avg_wait_min=float(rng.integers(0, 6)) * 0.5,
                is_rain=1 if rng.random() < 0.1 else 0,
                price_multiplier=float(rng.integers(10, 13)) / 10.0,
            )
        )
        t = add_minutes(t, 5)
    insert_zone_metrics(conn, rows)
    conn.close()
    return path


def _collect(days_db: Path) -> dict[str, object]:
    """Собрать примеры мини-эпохи (окно 1 ч, роли 2+1+1 дня)."""
    return collect_examples(
        SqliteRowSource(days_db),
        zones=[Z0],
        start=SIM_START,
        train_days=2,
        val_days=1,
        test_days=1,
        window_hours=1,
    )


def test_collect_examples_splits(days_db: Path) -> None:
    """Роли не пересекаются; крайние примеры отсечены историей/таргетами."""
    splits = _collect(days_db)
    train, val, test = splits["train"], splits["val"], splits["test"]
    assert train.windows.shape[1:] == (12, 17)  # окно 1 ч = 12 шагов
    # train: день 1 без первых 11 бакетов (история); val: полный день 2;
    # test: день 3 без последних 6 бакетов (нет таргета ts+30).
    assert train.windows.shape[0] == 277
    assert val.windows.shape[0] == 288
    assert test.windows.shape[0] == 282
    assert train.keys[0] == (Z0, add_minutes(SIM_START, 55))
    assert test.keys[-1] == (Z0, SIM_START + timedelta(days=2, hours=23, minutes=25))
    # Метки дефицита — 0/1.
    assert set(np.unique(train.deficit)) <= {0.0, 1.0}
    assert train.requests.shape == (277, 2)


def test_scalers_fit_only_train(days_db: Path) -> None:
    """Мутация только val/test строк (дни 2–3) не меняет параметры scaler'ов.

    Мутация начинается с бакета 00:30 дня 2: бакеты 00:05–00:25 — таргеты
    последних train-примеров дня 1 (легальные данные train-роли), их трогать
    нельзя — иначе изменился бы сам train.
    """
    splits = _collect(days_db)
    scalers_before = DatasetScalers.fit(
        splits["train"].windows, splits["train"].requests, splits["train"].free_drivers
    )
    conn = connect(str(days_db))
    conn.execute(
        "UPDATE zone_metrics_5min SET requests = 500, free_drivers = 0, unfulfilled = 9"
        " WHERE ts >= ?",
        (to_iso(SIM_START + timedelta(days=1, minutes=30)),),
    )
    conn.close()
    splits_after = _collect(days_db)
    scalers_after = DatasetScalers.fit(
        splits_after["train"].windows,
        splits_after["train"].requests,
        splits_after["train"].free_drivers,
    )
    # val/test-примеры изменились, train-статистики — нет.
    assert not np.array_equal(splits["val"].windows, splits_after["val"].windows)
    assert np.array_equal(scalers_before.features.mean_, scalers_after.features.mean_)
    assert np.array_equal(scalers_before.features.scale_, scalers_after.features.scale_)
    assert np.array_equal(scalers_before.requests.mean_, scalers_after.requests.mean_)
    assert np.array_equal(scalers_before.requests.scale_, scalers_after.requests.scale_)
    assert np.array_equal(scalers_before.free_drivers.mean_, scalers_after.free_drivers.mean_)
    assert np.array_equal(scalers_before.free_drivers.scale_, scalers_after.free_drivers.scale_)


def test_scalers_transform_shapes(days_db: Path) -> None:
    """Нормализация сохраняет формы: окна (n, 12, 17), таргеты (n, 2)."""
    splits = _collect(days_db)
    scalers = DatasetScalers.fit(
        splits["train"].windows, splits["train"].requests, splits["train"].free_drivers
    )
    windows = scalers.transform_windows(splits["test"].windows)
    assert windows.shape == splits["test"].windows.shape
    requests, free_drivers = scalers.transform_targets(
        splits["test"].requests, splits["test"].free_drivers
    )
    assert requests.shape == splits["test"].requests.shape
    assert free_drivers.shape == splits["test"].free_drivers.shape


def test_scalers_pickle_roundtrip(tmp_path: Path) -> None:
    """Сохранение/загрузка не меняют преобразования (побитово)."""
    rng = np.random.default_rng(3)
    windows = rng.normal(size=(20, 4, 17))
    requests = rng.normal(size=(20, 2)) * 5.0 + 10.0
    free_drivers = rng.normal(size=(20, 2)) + 3.0
    scalers = DatasetScalers.fit(windows, requests, free_drivers)
    expected_windows = scalers.transform_windows(windows)
    expected_requests, expected_free = scalers.transform_targets(requests, free_drivers)

    path = tmp_path / "artifacts" / "dataset" / "scalers.pkl"
    scalers.save(path)
    loaded = DatasetScalers.load(path)
    assert np.array_equal(loaded.transform_windows(windows), expected_windows)
    requests_out, free_out = loaded.transform_targets(requests, free_drivers)
    assert np.array_equal(requests_out, expected_requests)
    assert np.array_equal(free_out, expected_free)


def test_scalers_load_rejects_foreign_object(tmp_path: Path) -> None:
    """Загрузка постороннего pickle отклоняется ``TypeError``."""
    path = tmp_path / "other.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"mean": 1.0}, fh)
    with pytest.raises(TypeError):
        DatasetScalers.load(path)
