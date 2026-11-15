"""Метрики сравнения прогнозов вехи M3: регрессии, дефицит, калибровка, сводная.

Один расчёт для любых ``model_version`` — наивов сейчас, сетей в
``m3-inference`` (решение 3 design.md change ``m3-baselines``). Вход — строки
контракта ``forecasts`` (dict-строки наивов или ``sqlite3.Row`` из
``list_forecasts``) плюс :class:`RowSource` для факта; соединение — по
(ts, zone_id, horizon_min), факт целевого бакета — ``metrics_at(ts + H)``,
метка дефицита — правило контракта (:func:`ml.dataset.deficit_label`). Строка
без пары к факту исключается из всех метрик, без падений и подстановок нулей
(спека «Прогноз без пары к факту»).

Детерминированность: пары сортируются канонически по (ts, zone_id,
horizon_min) до всякого суммирования — порядок входных строк не влияет ни на
значения, ни на округление (решение 5 design.md). Вырожденные случаи (нет
истинных или предсказанных позитивов, нет пар) дают 0.0, не NaN и не
исключения (решение 4 design.md). Все числа на выходе — встроенные
float/int: структура :func:`build_summary` JSON-совместима для
``artifacts/summary-<дата>.json`` будущего ``m3-inference``.
"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from common.timeutil import add_minutes, from_iso
from ml.dataset import RowSource, deficit_label

DEFICIT_PROB_THRESHOLD = 0.5
"""Порог детекции: ``deficit_prob >= порог`` — «прогноз предсказал дефицит»."""

N_CALIBRATION_BINS = 10
"""Число бинов калибровки [0.0, 0.1) … [0.9, 1.0]; верхний бин включает 1.0."""

REGRESSION_TARGETS = ("pred_requests", "pred_free_drivers")
"""Таргеты регрессионных метрик; факт-аналог — то же имя с префиксом ``fact_``."""

# Колонки печатной сводной (§6 плана M3): имя → извлекатель значения из сводной;
# отсутствующие метрики (горизонт без пар) читаются как 0.0.
_SUMMARY_COLUMNS: tuple[tuple[str, Callable[[Mapping], float]], ...] = (
    (
        "MAE_req15",
        lambda s: s.get("regression", {}).get("pred_requests", {}).get(15, {}).get("mae", 0.0),
    ),
    (
        "MAE_req30",
        lambda s: s.get("regression", {}).get("pred_requests", {}).get(30, {}).get("mae", 0.0),
    ),
    (
        "MAE_fr15",
        lambda s: s.get("regression", {}).get("pred_free_drivers", {}).get(15, {}).get("mae", 0.0),
    ),
    (
        "MAE_fr30",
        lambda s: s.get("regression", {}).get("pred_free_drivers", {}).get(30, {}).get("mae", 0.0),
    ),
    (
        "RMSE_req15",
        lambda s: s.get("regression", {}).get("pred_requests", {}).get(15, {}).get("rmse", 0.0),
    ),
    (
        "F1_def15",
        lambda s: s.get("deficit", {}).get(15, {}).get("f1", 0.0),
    ),
    (
        "F1_def30",
        lambda s: s.get("deficit", {}).get(30, {}).get("f1", 0.0),
    ),
)

_COLUMN_WIDTH = 10
"""Ширина числовой колонки рендера (самое длинное имя — «RMSE_req15»)."""


def join_facts(rows: Sequence[Mapping], source: RowSource) -> list[dict]:
    """Соединить строки прогнозов с фактом целевого бакета ts + horizon_min.

    Строка без факта (бакета ts+H нет в источнике) — молчаливый пропуск.
    Время на входе — наивный ``datetime`` или ISO-текст БД. Результат
    отсортирован по (ts, zone_id, horizon_min) — канонический порядок
    суммирования, гарантирующий одинаковые биты результата при любом порядке
    входных строк.
    """
    paired: list[dict] = []
    for row in rows:
        ts = row["ts"]
        if isinstance(ts, str):
            ts = from_iso(ts)
        horizon = int(row["horizon_min"])
        zone_id = int(row["zone_id"])
        point = source.metrics_at(zone_id, add_minutes(ts, horizon))
        if point is None:
            continue
        paired.append(
            {
                "ts": ts,
                "zone_id": zone_id,
                "horizon_min": horizon,
                "pred_requests": float(row["pred_requests"]),
                "pred_free_drivers": float(row["pred_free_drivers"]),
                "deficit_prob": float(row["deficit_prob"]),
                "fact_requests": float(point.requests),
                "fact_free_drivers": float(point.free_drivers),
                "fact_deficit": bool(
                    deficit_label(point.requests, point.free_drivers, point.unfulfilled)
                ),
            }
        )
    paired.sort(key=lambda r: (r["ts"], r["zone_id"], r["horizon_min"]))
    return paired


def mae_rmse(pred: Sequence[float], fact: Sequence[float]) -> tuple[float, float]:
    """MAE и RMSE по массивам «прогноз, факт»; пустой вход — (0.0, 0.0)."""
    err = np.asarray(pred, dtype=float) - np.asarray(fact, dtype=float)
    if err.size == 0:
        return 0.0, 0.0
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err * err)))


def regression_metrics(paired: Sequence[Mapping]) -> dict[str, dict[int, dict[str, float]]]:
    """MAE/RMSE пар «прогноз, факт» по каждому таргету и горизонту отдельно."""
    horizons = sorted({int(r["horizon_min"]) for r in paired})
    result: dict[str, dict[int, dict[str, float]]] = {}
    for target in REGRESSION_TARGETS:
        fact_key = "fact_" + target.removeprefix("pred_")
        per_horizon: dict[int, dict[str, float]] = {}
        for horizon in horizons:
            selected = [r for r in paired if int(r["horizon_min"]) == horizon]
            mae, rmse = mae_rmse([r[target] for r in selected], [r[fact_key] for r in selected])
            per_horizon[horizon] = {"mae": mae, "rmse": rmse}
        result[target] = per_horizon
    return result


def deficit_metrics(paired: Sequence[Mapping]) -> dict[int, dict[str, float]]:
    """Precision/recall/F1 детекции дефицита по горизонтам.

    События: «предсказано» — ``deficit_prob >= DEFICIT_PROB_THRESHOLD``;
    «сбылось» — правило контракта на целевом бакете (поле ``fact_deficit``
    пары). Нет истинных и/или предсказанных позитивов — 0.0, без деления на
    ноль и NaN (решение 4 design.md).
    """
    horizons = sorted({int(r["horizon_min"]) for r in paired})
    result: dict[int, dict[str, float]] = {}
    for horizon in horizons:
        tp = fp = fn = 0
        for row in paired:
            if int(row["horizon_min"]) != horizon:
                continue
            predicted = float(row["deficit_prob"]) >= DEFICIT_PROB_THRESHOLD
            if predicted and row["fact_deficit"]:
                tp += 1
            elif predicted:
                fp += 1
            elif row["fact_deficit"]:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[horizon] = {"precision": precision, "recall": recall, "f1": f1}
    return result


def calibration_table(paired: Sequence[Mapping], horizon: int) -> list[dict]:
    """Таблица калибровки одного горизонта: непустые бины [0.0,0.1)…[0.9,1.0].

    По бину: ``count`` — число строк, ``mean_prob`` — средняя предсказанная
    вероятность, ``deficit_share`` — доля сбывшихся дефицитов. Верхний бин
    включает 1.0; бины без строк в таблицу не входят (спека «Пустые бины»).
    """
    rows = [r for r in paired if int(r["horizon_min"]) == int(horizon)]
    table: list[dict] = []
    for bin_no in range(N_CALIBRATION_BINS):
        low = bin_no / N_CALIBRATION_BINS
        high = (bin_no + 1) / N_CALIBRATION_BINS
        is_upper = bin_no == N_CALIBRATION_BINS - 1
        in_bin = []
        for row in rows:
            prob = float(row["deficit_prob"])
            if low <= prob < high or (is_upper and prob == high):
                in_bin.append(row)
        if not in_bin:
            continue
        table.append(
            {
                "bin_low": low,
                "bin_high": high,
                "count": len(in_bin),
                "mean_prob": float(np.mean([float(r["deficit_prob"]) for r in in_bin])),
                "deficit_share": float(
                    np.mean([1.0 if r["fact_deficit"] else 0.0 for r in in_bin])
                ),
            }
        )
    return table


def build_summary(model_version: str, rows: Sequence[Mapping], source: RowSource) -> dict:
    """Сводная одной версии модели: регрессии, дефицит и калибровка.

    Структура: ``model_version`` + таргет × горизонт → MAE/RMSE + горизонт →
    precision/recall/F1 + горизонт → бины калибровки. Все числа — встроенные
    float/int (JSON-совместимость, решение 5 design.md); горизонтов без пар —
    пустые разделы, рендер читает их как 0.0.
    """
    paired = join_facts(rows, source)
    horizons = sorted({int(r["horizon_min"]) for r in paired})
    return {
        "model_version": model_version,
        "regression": regression_metrics(paired),
        "deficit": deficit_metrics(paired),
        "calibration": {horizon: calibration_table(paired, horizon) for horizon in horizons},
    }


def render_table(summaries: Sequence[Mapping]) -> str:
    """Печатная сводная плана M3 (§6): строка на версию модели, 7 колонок.

    Колонки: MAE_req15, MAE_req30, MAE_fr15, MAE_fr30, RMSE_req15, F1_def15,
    F1_def30; формат ``%.3f``. Одинаковый вход даёт одинаковый вывод: строки
    идут в порядке ``summaries``, а порядок суммирования метрик зафиксирован
    в :func:`join_facts`.
    """
    name_width = max(
        [len("model")] + [len(str(summary.get("model_version", ""))) for summary in summaries]
    )
    lines = [
        "model".ljust(name_width)
        + "  "
        + " ".join(name.rjust(_COLUMN_WIDTH) for name, _ in _SUMMARY_COLUMNS)
    ]
    for summary in summaries:
        cells = " ".join(
            f"{float(getter(summary)):>{_COLUMN_WIDTH}.3f}" for _, getter in _SUMMARY_COLUMNS
        )
        lines.append(str(summary.get("model_version", "")).ljust(name_width) + "  " + cells)
    return "\n".join(lines)
