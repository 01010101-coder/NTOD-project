"""Тесты ml/baselines.py и ml/metrics.py (change ``m3-baselines``): наивы
«как сейчас» и «сутки назад», запись через ``insert_forecasts``, MAE/RMSE,
precision/recall/F1, калибровка, сводная и рендер.

Реальная эпоха (``data/m3_epoch.db``) не трогается (Non-Goals design.md):
наивы и метрики проверяются на фейковом ``RowSource`` в памяти, запись
прогнозов — на рукописной мини-БД в ``tmp_path``. Ожидаемые числа повторяются
рукописными вычислениями (§3.1 tasks.md), а не «похожими» формулами.
"""

import json
import math
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from common.timeutil import add_minutes, to_iso
from ml.baselines import NAIVE_PERSIST_V1, NAIVE_SEASON_V1, naive_persist, naive_season
from ml.dataset import MetricPoint, SqliteRowSource, bucket_index, from_bucket_index
from ml.metrics import (
    build_summary,
    calibration_table,
    deficit_metrics,
    join_facts,
    mae_rmse,
    regression_metrics,
    render_table,
)
from storage.db import connect
from storage.repo import insert_forecasts, insert_zone_metrics, list_forecasts
from storage.schema import init_db

NOON = datetime(2026, 11, 3, 12, 0)
"""«Текущий момент» прогнозов: вторник, полдень."""

YESTERDAY_NOON = NOON - timedelta(hours=24)
"""Слот суток накануне: тот же полдень понедельника."""

Z0, Z1 = 0, 1


class FakeSource:
    """Фейковый RowSource: метрики в словаре по (зона, начало бакета).

    Наивам и метрикам достаточно ``metrics_at`` и ``time_range``; момент
    запроса нормализуется в начало содержащего его бакета (как у
    ``SqliteRowSource``), поэтому ts можно давать не на границе сетки.
    """

    def __init__(self, rows: dict[tuple[int, datetime], MetricPoint]) -> None:
        self._rows = rows

    def metrics_window(self, zone_id: int, ts: datetime, steps: int) -> np.ndarray | None:
        """Окно наивам и метрикам не нужно (решение 1 design.md)."""
        raise NotImplementedError("наивам и метрикам metrics_window не нужен")

    def metrics_at(self, zone_id: int, ts: datetime) -> MetricPoint | None:
        """Точечные метрики бакета, содержащего ``ts``, или ``None``."""

        return self._rows.get((zone_id, from_bucket_index(bucket_index(ts))))

    def dropoff_counts(
        self, zone_id: int, after: datetime, until: datetime, assigned_until: datetime
    ) -> int:
        """Высадки наивам и метрикам не нужны (ret-признак собирает датасет)."""
        raise NotImplementedError("наивам и метрикам dropoff_counts не нужен")

    def time_range(self) -> tuple[datetime, datetime]:
        """Границы эпохи: первый и последний бакет словаря."""

        times = [ts for _, ts in self._rows]
        return min(times), max(times)


def _integration_source() -> FakeSource:
    """Фикстура §3.1: 2 зоны × 4 бакета (вчера-полдень, полдень, +15, +30).

    Значения рукописные; правило дефицита на них: NOON — дефицит в обеих
    зонах (3/10 и 1/4 < 0.5), +15 — нет (6/8 и 4/5), +30 — дефицит (4/12 < 0.5,
    0/3 и unfulfilled=1), вчера — нет (8/6 и 5/2).
    """

    return FakeSource(
        {
            (Z0, YESTERDAY_NOON): MetricPoint(6.0, 8.0, 0.0),
            (Z1, YESTERDAY_NOON): MetricPoint(2.0, 5.0, 0.0),
            (Z0, NOON): MetricPoint(10.0, 3.0, 0.0),
            (Z1, NOON): MetricPoint(4.0, 1.0, 2.0),
            (Z0, add_minutes(NOON, 15)): MetricPoint(8.0, 6.0, 0.0),
            (Z1, add_minutes(NOON, 15)): MetricPoint(5.0, 4.0, 0.0),
            (Z0, add_minutes(NOON, 30)): MetricPoint(12.0, 4.0, 0.0),
            (Z1, add_minutes(NOON, 30)): MetricPoint(3.0, 0.0, 1.0),
        }
    )


def _metric_dict(ts: datetime, zone_id: int, requests: int, free_drivers: int) -> dict:
    """Строка ``zone_metrics_5min`` мини-БД (схема как в тестах m3-dataset)."""

    return {
        "ts": ts,
        "zone_id": zone_id,
        "requests": requests,
        "completed": 1,
        "unfulfilled": 0,
        "free_drivers": free_drivers,
        "busy_drivers": 2,
        "avg_wait_min": 3.0,
        "supply_demand_ratio": free_drivers / max(requests, 1),
        "is_rain": 0,
        "event": "",
        "price_multiplier": 1.0,
    }


def _forecast_row(
    ts: datetime,
    zone_id: int,
    horizon_min: int,
    pred_requests: float,
    pred_free_drivers: float,
    deficit_prob: float,
) -> dict:
    """Рукописная dict-строка контракта ``forecasts`` для метрик."""

    return {
        "ts": ts,
        "zone_id": zone_id,
        "horizon_min": horizon_min,
        "pred_requests": pred_requests,
        "pred_free_drivers": pred_free_drivers,
        "deficit_prob": deficit_prob,
        "model_version": "handwritten-v1",
    }


# ---------------------------------------------------------------------------
# 1.1 Наив «как сейчас» (naive-persist-v1)
# ---------------------------------------------------------------------------


def test_naive_persist_matches_last_observation() -> None:
    """Спека «Прогнос равен последнему наблюдению»: 3/10 < 0.5 → дефицит 1.0."""

    source = FakeSource({(Z0, NOON): MetricPoint(10.0, 3.0, 0.0)})
    rows = naive_persist(source, NOON, Z0)
    assert len(rows) == 2
    assert [r["horizon_min"] for r in rows] == [15, 30]
    for row in rows:
        assert row["ts"] == NOON
        assert row["zone_id"] == Z0
        assert row["pred_requests"] == 10.0
        assert row["pred_free_drivers"] == 3.0
        assert row["deficit_prob"] == 1.0
        assert row["model_version"] == NAIVE_PERSIST_V1


def test_naive_persist_horizons_identical() -> None:
    """Спека «Оба горизонта одинаковы»: строки различаются только horizon_min."""

    source = FakeSource({(Z0, NOON): MetricPoint(10.0, 3.0, 0.0)})
    h15, h30 = naive_persist(source, NOON, Z0)
    for key in ("pred_requests", "pred_free_drivers", "deficit_prob"):
        assert h15[key] == h30[key]


def test_naive_persist_values_shape_and_bounds() -> None:
    """Спека «Форма и границы значений»: 6/10 ≥ 0.5 → дефицита нет; значения
    неотрицательны, deficit_prob равен 0.0 или 1.0."""

    source = FakeSource({(Z0, NOON): MetricPoint(10.0, 6.0, 0.0)})
    rows = naive_persist(source, NOON, Z0)
    assert len(rows) == 2
    for row in rows:
        assert row["deficit_prob"] == 0.0
        assert row["pred_requests"] >= 0.0
        assert row["pred_free_drivers"] >= 0.0
        assert row["deficit_prob"] in (0.0, 1.0)


def test_naive_persist_deficit_via_unfulfilled() -> None:
    """Правило дефицита через сток: 9/10 ≥ 0.5, но unfulfilled = 2 → 1.0."""

    source = FakeSource({(Z0, NOON): MetricPoint(10.0, 9.0, 2.0)})
    assert all(r["deficit_prob"] == 1.0 for r in naive_persist(source, NOON, Z0))


def test_naive_persist_missing_bucket_silent() -> None:
    """Нет бакета ts — молчаливый пропуск: пустой список, без исключений."""

    assert naive_persist(FakeSource({}), NOON, Z0) == []


# ---------------------------------------------------------------------------
# 1.2 Наив «сутки назад» (naive-season-v1)
# ---------------------------------------------------------------------------


def test_naive_season_exact_on_daily_periodic() -> None:
    """Спека «Точный ответ на суточной периодике»: значения вчерашнего слота
    (7, 9, дефицита нет), и на суточно-периодичной истории ошибка ровно 0."""

    rows: dict[tuple[int, datetime], MetricPoint] = {}
    t = datetime(2026, 11, 2)
    end = datetime(2026, 11, 3, 13, 0)
    while t <= end:  # двое суток с половиной, каждый бакет одни и те же метрики
        rows[(Z0, t)] = MetricPoint(7.0, 9.0, 0.0)
        t = add_minutes(t, 5)
    source = FakeSource(rows)

    forecast = naive_season(source, NOON, Z0)
    assert [r["horizon_min"] for r in forecast] == [15, 30]
    for row in forecast:
        assert row["ts"] == NOON
        assert row["pred_requests"] == 7.0
        assert row["pred_free_drivers"] == 9.0
        assert row["deficit_prob"] == 0.0  # 9/7 ≥ 0.5 и unfulfilled = 0
        assert row["model_version"] == NAIVE_SEASON_V1
        # Ошибка = 0: прогноз совпал с фактом целевого бакета (периодика 24 ч).
        fact = source.metrics_at(Z0, add_minutes(NOON, row["horizon_min"]))
        assert row["pred_requests"] == fact.requests
        assert row["pred_free_drivers"] == fact.free_drivers


def test_naive_season_missing_yesterday_silent() -> None:
    """Спека «Вчерашнего слота нет»: пустой список — без исключений и заглушек."""

    source = FakeSource({(Z0, NOON): MetricPoint(5.0, 5.0, 0.0)})
    assert naive_season(source, NOON, Z0) == []


# ---------------------------------------------------------------------------
# 1.3 Запись наивов через storage.repo
# ---------------------------------------------------------------------------


def test_naive_rows_roundtrip_through_forecasts(tmp_path: Path) -> None:
    """1.3: наивы пишутся через ``insert_forecasts`` в мини-БД и читаются
    ``list_forecasts`` с фильтром по ``model_version`` без потерь.

    PK ``forecasts`` — (ts, zone_id, horizon_min) без ``model_version`` (схема
    плана M1), поэтому моменты прогнозов версий разведены на один бакет (NOON и
    NOON+5): прогон обеих версий по одним моментам — забота ``m3-inference``.
    """

    path = tmp_path / "mini.db"
    conn = connect(str(path))
    init_db(conn)
    noon5 = add_minutes(NOON, 5)
    for ts, req, free in (
        (YESTERDAY_NOON, 7, 9),
        (add_minutes(YESTERDAY_NOON, 5), 7, 9),
        (NOON, 10, 3),
        (noon5, 10, 3),
        (add_minutes(NOON, 30), 12, 4),
    ):
        insert_zone_metrics(conn, [_metric_dict(ts, Z0, req, free)])

    source = SqliteRowSource(path)
    persist_rows = naive_persist(source, NOON, Z0)
    season_rows = naive_season(source, noon5, Z0)
    insert_forecasts(conn, persist_rows)
    insert_forecasts(conn, season_rows)

    got_persist = list_forecasts(conn, model_version=NAIVE_PERSIST_V1)
    got_season = list_forecasts(conn, model_version=NAIVE_SEASON_V1)
    assert len(got_persist) == 2
    assert len(got_season) == 2
    assert len(list_forecasts(conn)) == 4  # фильтр по версии действительно фильтрует
    # Значения возвращаются без потерь и искажений: persist взял «как сейчас»
    # (10, 3), сезонный — вчера (7, 9); время — ISO-текст момента прогноза.
    by_h = {row["horizon_min"]: row for row in got_persist}
    assert by_h[15]["ts"] == to_iso(NOON)
    assert by_h[15]["zone_id"] == Z0
    assert by_h[15]["pred_requests"] == 10.0
    assert by_h[15]["pred_free_drivers"] == 3.0
    assert by_h[15]["deficit_prob"] == 1.0
    assert by_h[30]["pred_requests"] == 10.0
    by_h = {row["horizon_min"]: row for row in got_season}
    assert by_h[15]["ts"] == to_iso(noon5)
    assert by_h[15]["pred_requests"] == 7.0
    assert by_h[15]["pred_free_drivers"] == 9.0
    assert by_h[15]["deficit_prob"] == 0.0
    assert by_h[30]["pred_requests"] == 7.0
    conn.close()


# ---------------------------------------------------------------------------
# 2.1 Метрики регрессий: MAE и RMSE
# ---------------------------------------------------------------------------


def test_mae_rmse_handwritten() -> None:
    """Спека «Рукописный случай MAE»: отклонения 1/2/3 → MAE 2.0,
    RMSE = sqrt((1+4+9)/3)."""

    mae, rmse = mae_rmse([11.0, 8.0, 13.0], [10.0, 10.0, 10.0])
    assert mae == pytest.approx(2.0)
    assert rmse == pytest.approx(math.sqrt(14.0 / 3.0))


def test_mae_rmse_empty_input_is_zero() -> None:
    """Пустой вход — (0.0, 0.0): без NaN и исключений."""

    assert mae_rmse([], []) == (0.0, 0.0)


def test_regression_metrics_by_target_and_horizon() -> None:
    """MAE/RMSE считаются по каждому таргету и горизонту отдельно."""

    facts: dict[tuple[int, datetime], MetricPoint] = {}
    rows = []
    forecast_ts = [NOON, add_minutes(NOON, 5), add_minutes(NOON, 10)]
    pred_requests = [11.0, 8.0, 13.0]  # отклонения +1, −2, +3
    pred_free = [5.0, 4.0, 6.5]  # отклонения 0, −1, +1.5
    for ts, req, free in zip(forecast_ts, pred_requests, pred_free, strict=True):
        facts[(Z0, add_minutes(ts, 15))] = MetricPoint(10.0, 5.0, 0.0)
        facts[(Z0, add_minutes(ts, 30))] = MetricPoint(10.0, 5.0, 0.0)
        rows.append(_forecast_row(ts, Z0, 15, req, free, 0.0))
    # Один отдельный случай горизонта 30: отклонение запросов 10.
    rows.append(_forecast_row(NOON, Z0, 30, 20.0, 5.0, 0.0))

    reg = regression_metrics(join_facts(rows, FakeSource(facts)))
    assert reg["pred_requests"][15]["mae"] == pytest.approx(2.0)
    assert reg["pred_requests"][15]["rmse"] == pytest.approx(math.sqrt(14.0 / 3.0))
    assert reg["pred_free_drivers"][15]["mae"] == pytest.approx(2.5 / 3.0)
    assert reg["pred_requests"][30]["mae"] == pytest.approx(10.0)


def test_forecast_without_fact_excluded() -> None:
    """Спека «Прогноз без пары к факту»: строка исключается из расчёта —
    не падает и не подставляет 0 в отклонения."""

    facts = {(Z0, add_minutes(NOON, 15)): MetricPoint(10.0, 5.0, 0.0)}
    rows = [
        _forecast_row(NOON, Z0, 15, 11.0, 5.0, 0.0),
        # Целевой бакет NOON+5+15 в факте отсутствует → строка мимо метрик.
        _forecast_row(add_minutes(NOON, 5), Z0, 15, 50.0, 50.0, 0.0),
    ]
    paired = join_facts(rows, FakeSource(facts))
    assert len(paired) == 1
    reg = regression_metrics(paired)
    assert reg["pred_requests"][15]["mae"] == pytest.approx(1.0)


def test_join_facts_accepts_db_text_time() -> None:
    """join_facts принимает ISO-текст БД (sqlite3.Row) и наивный datetime."""

    facts = {(Z0, add_minutes(NOON, 15)): MetricPoint(10.0, 5.0, 0.0)}
    as_dt = _forecast_row(NOON, Z0, 15, 11.0, 5.0, 1.0)
    as_text = dict(as_dt, ts=to_iso(NOON))
    assert join_facts([as_dt], FakeSource(facts)) == join_facts([as_text], FakeSource(facts))


# ---------------------------------------------------------------------------
# 2.2 Метрики детекции дефицита: precision/recall/F1
# ---------------------------------------------------------------------------


def test_deficit_perfect_detection() -> None:
    """Спека «Идеальная детекция»: предсказанные метки совпали с фактическими
    во всех строках → precision = recall = F1 = 1.0."""

    paired = [
        {"horizon_min": 15, "deficit_prob": 1.0, "fact_deficit": True},
        {"horizon_min": 15, "deficit_prob": 0.0, "fact_deficit": False},
    ]
    assert deficit_metrics(paired)[15] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_deficit_no_positives_degenerate() -> None:
    """Спека «Ни одного позитива»: все 0.0 — без исключений и NaN."""

    paired = [
        {"horizon_min": 15, "deficit_prob": 0.0, "fact_deficit": False},
        {"horizon_min": 15, "deficit_prob": 0.0, "fact_deficit": False},
    ]
    metrics = deficit_metrics(paired)[15]
    assert metrics == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert all(math.isfinite(value) for value in metrics.values())


def test_deficit_handwritten_confusion() -> None:
    """Рукописная матрица ошибок: TP=1, FP=1, FN=1 → P = R = 0.5, F1 = 0.5."""

    paired = [
        {"horizon_min": 15, "deficit_prob": 1.0, "fact_deficit": True},  # TP
        {"horizon_min": 15, "deficit_prob": 1.0, "fact_deficit": False},  # FP
        {"horizon_min": 15, "deficit_prob": 0.0, "fact_deficit": True},  # FN
        {"horizon_min": 15, "deficit_prob": 0.0, "fact_deficit": False},  # TN
    ]
    metrics = deficit_metrics(paired)[15]
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_deficit_threshold_inclusive() -> None:
    """Порог включительный: deficit_prob = 0.5 — предсказанный позитив."""

    paired = [{"horizon_min": 15, "deficit_prob": 0.5, "fact_deficit": True}]
    assert deficit_metrics(paired)[15]["recall"] == 1.0


# ---------------------------------------------------------------------------
# 2.3 Таблица калибровки вероятностей
# ---------------------------------------------------------------------------


def test_calibration_ideal_diagonal() -> None:
    """Спека «Идеально калиброванные вероятности»: бин ≈0.7 сбывается ≈70%."""

    paired = [{"horizon_min": 15, "deficit_prob": 0.7, "fact_deficit": i < 7} for i in range(10)]
    table = calibration_table(paired, 15)
    assert len(table) == 1
    bin_row = table[0]
    assert bin_row["bin_low"] == pytest.approx(0.7)
    assert bin_row["bin_high"] == pytest.approx(0.8)
    assert bin_row["count"] == 10
    assert bin_row["mean_prob"] == pytest.approx(0.7)
    assert bin_row["deficit_share"] == pytest.approx(0.7)


def test_calibration_empty_bins_omitted_upper_inclusive() -> None:
    """Спека «Пустые бины»: [0.2,0.3) отсутствует; верхний бин включает 1.0;
    граница 0.1 попадает в бин [0.1, 0.2)."""

    paired = [
        {"horizon_min": 15, "deficit_prob": 0.0, "fact_deficit": False},
        {"horizon_min": 15, "deficit_prob": 0.1, "fact_deficit": True},
        {"horizon_min": 15, "deficit_prob": 0.95, "fact_deficit": True},
        {"horizon_min": 15, "deficit_prob": 1.0, "fact_deficit": True},
    ]
    table = calibration_table(paired, 15)
    assert [b["bin_low"] for b in table] == [0.0, 0.1, 0.9]
    assert table[-1]["count"] == 2
    assert table[-1]["deficit_share"] == pytest.approx(1.0)
    assert table[-1]["mean_prob"] == pytest.approx(0.975)


def test_calibration_per_horizon_independent() -> None:
    """Калибровка горизонта 30 не видит строки горизонта 15."""

    paired = [
        {"horizon_min": 15, "deficit_prob": 0.7, "fact_deficit": True},
        {"horizon_min": 30, "deficit_prob": 0.3, "fact_deficit": True},
    ]
    assert [b["bin_low"] for b in calibration_table(paired, 30)] == [0.3]


# ---------------------------------------------------------------------------
# 2.4 Сводная структура и текстовая таблица
# ---------------------------------------------------------------------------


def _collect_numbers(node: object, out: list[float]) -> None:
    """Собрать все числа вложенной структуры (для проверки типов и NaN)."""

    if isinstance(node, Mapping):
        for value in node.values():
            _collect_numbers(value, out)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _collect_numbers(value, out)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out.append(float(node))


def test_build_summary_json_compatible() -> None:
    """2.4: сводная — встроенные float/int (без numpy-скаляров), без NaN,
    и сериализуется json.dumps без ошибок."""

    source = _integration_source()
    rows = naive_persist(source, NOON, Z0) + naive_persist(source, NOON, Z1)
    summary = build_summary(NAIVE_PERSIST_V1, rows, source)

    assert summary["model_version"] == NAIVE_PERSIST_V1
    assert set(summary) == {"model_version", "regression", "deficit", "calibration"}
    assert json.dumps(summary)  # не падает — значит, все значения JSON-совместимы
    numbers: list[float] = []
    _collect_numbers(summary, numbers)
    assert numbers
    assert all(not isinstance(value, np.generic) for value in numbers)
    assert all(math.isfinite(value) for value in numbers)


def test_render_table_one_version() -> None:
    """Спека «Рендер одной строки»: имя версии и семь числовых колонок,
    все значения — числа или 0.0, без NaN."""

    source = _integration_source()
    rows = naive_season(source, NOON, Z0) + naive_season(source, NOON, Z1)
    summary = build_summary(NAIVE_SEASON_V1, rows, source)
    table = render_table([summary])

    assert NAIVE_SEASON_V1 in table
    for column in (
        "MAE_req15",
        "MAE_req30",
        "MAE_fr15",
        "MAE_fr30",
        "RMSE_req15",
        "F1_def15",
        "F1_def30",
    ):
        assert column in table
    assert "nan" not in table.lower()
    data_line = table.splitlines()[1]
    assert len(data_line.split()) == 8  # имя версии + 7 чисел
    values = data_line.split()[1:]
    assert all(token.replace(".", "").isdigit() for token in values)


def test_render_table_deterministic() -> None:
    """Одинаковый вход даёт одинаковый вывод; порядок входных строк факта
    не меняет ни значений, ни округления (решение 5 design.md)."""

    source = _integration_source()
    rows = naive_persist(source, NOON, Z0) + naive_persist(source, NOON, Z1)
    table = render_table([build_summary(NAIVE_PERSIST_V1, rows, source)])
    assert table == render_table([build_summary(NAIVE_PERSIST_V1, rows, source)])

    reversed_rows = list(reversed(rows))
    assert table == render_table([build_summary(NAIVE_PERSIST_V1, reversed_rows, source)])


def test_render_table_two_versions() -> None:
    """Рендер двух версий: шапка + строка на версию, значения согласованы
    с метриками сводных."""

    source = _integration_source()
    persist = build_summary(
        NAIVE_PERSIST_V1,
        naive_persist(source, NOON, Z0) + naive_persist(source, NOON, Z1),
        source,
    )
    season = build_summary(
        NAIVE_SEASON_V1,
        naive_season(source, NOON, Z0) + naive_season(source, NOON, Z1),
        source,
    )
    table = render_table([persist, season])
    lines = table.splitlines()
    assert len(lines) == 3
    assert lines[1].split()[0] == NAIVE_PERSIST_V1
    assert lines[2].split()[0] == NAIVE_SEASON_V1
    assert "1.500" in lines[1]  # MAE_req15 persist = 1.5 (см. сквозной тест ниже)


def test_render_table_empty_summary_zeroes() -> None:
    """Сводная без пар (нет факта) рендерится нулями — таблица печатается
    всегда, без NaN (решение 4 design.md)."""

    summary = build_summary(NAIVE_PERSIST_V1, [], FakeSource({}))
    table = render_table([summary])
    assert "nan" not in table.lower()
    data_line = table.splitlines()[1]
    assert data_line.split()[1:] == ["0.000"] * 7


# ---------------------------------------------------------------------------
# 3.1 Сквозной цикл: наивы → метрики → сводная из двух строк
# ---------------------------------------------------------------------------


def test_end_to_end_naives_metrics_summary() -> None:
    """3.1: полный цикл двух наивов на общем фейковом RowSource согласован
    с рукописными вычислениями (2 зоны, 8 бакетов истории).

    persist (зоны 0 и 1, прогноз с NOON): отклонения запросов |10−8|=2, |4−5|=1
    на +15 и |10−12|=2, |4−3|=1 на +30; свободных |3−6|=3, |1−4|=3 и |3−4|=1,
    |1−0|=1. season: прогнозы вчера (6, 8) и (2, 5); отклонения запросов
    |6−8|=2, |2−5|=3 и |6−12|=6, |2−3|=1; свободных |8−6|=2, |5−4|=1 и
    |8−4|=4, |5−0|=5.
    """

    source = _integration_source()
    persist_rows = naive_persist(source, NOON, Z0) + naive_persist(source, NOON, Z1)
    season_rows = naive_season(source, NOON, Z0) + naive_season(source, NOON, Z1)
    persist = build_summary(NAIVE_PERSIST_V1, persist_rows, source)
    season = build_summary(NAIVE_SEASON_V1, season_rows, source)

    persist_reg = persist["regression"]
    assert persist_reg["pred_requests"][15]["mae"] == pytest.approx(1.5)
    assert persist_reg["pred_requests"][15]["rmse"] == pytest.approx(math.sqrt(5.0 / 2.0))
    assert persist_reg["pred_requests"][30]["mae"] == pytest.approx(1.5)
    assert persist_reg["pred_free_drivers"][15]["mae"] == pytest.approx(3.0)
    assert persist_reg["pred_free_drivers"][30]["mae"] == pytest.approx(1.0)
    # persist предсказал дефицит в обеих зонах: на +15 факта нет (FP, P=R=0),
    # на +30 дефицит сбылся в обеих (TP, всё по 1.0).
    assert persist["deficit"][15] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert persist["deficit"][30] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    # Наивы бинаризованы: только крайние бины [0.0,0.1) и [0.9,1.0].
    cal15 = {b["bin_low"]: b for b in persist["calibration"][15]}
    cal30 = {b["bin_low"]: b for b in persist["calibration"][30]}
    assert set(cal15) == {0.9}
    assert cal15[0.9]["count"] == 2
    assert cal15[0.9]["deficit_share"] == pytest.approx(0.0)
    assert set(cal30) == {0.9}
    assert cal30[0.9]["deficit_share"] == pytest.approx(1.0)

    season_reg = season["regression"]
    assert season_reg["pred_requests"][15]["mae"] == pytest.approx(2.5)
    assert season_reg["pred_requests"][30]["mae"] == pytest.approx(3.5)
    assert season_reg["pred_free_drivers"][15]["mae"] == pytest.approx(1.5)
    assert season_reg["pred_free_drivers"][30]["mae"] == pytest.approx(4.5)
    # season дефицита не предсказывал: на +30 оба фактических дефицита
    # пропущены (FN) — все метрики 0.0, вырожденный случай без NaN.
    assert season["deficit"][15] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert season["deficit"][30] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    cal_season15 = {b["bin_low"] for b in season["calibration"][15]}
    assert cal_season15 == {0.0}  # вероятности 0.0 — только первый бин

    table = render_table([persist, season])
    lines = table.splitlines()
    assert len(lines) == 3
    assert lines[1].split()[0] == NAIVE_PERSIST_V1
    assert lines[2].split()[0] == NAIVE_SEASON_V1
    assert "nan" not in table.lower()
    assert "1.500" in lines[1] and "3.500" in lines[2]  # рукописные MAE_req30
