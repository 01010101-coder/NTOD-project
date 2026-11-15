"""Наивы-бейзлайны вехи M3: прогнозы без обучения поверх RowSource (§6 плана M3).

Оба наива — чистые функции «(источник, момент ts, зона) → строки forecasts»
(решение 1 design.md change ``m3-baselines``): без обучения, без CLI и без
собственного walk-forward — прогон по тест-эпохе и сводная из 4 строк относятся
к ``m3-inference``. Формат результата — dict-строки контракта ``forecasts``
(решение 2 design.md), пригодные и для ``storage.repo.insert_forecasts``, и для
метрик :mod:`ml.metrics`; общий источник строк «наивы = будущие сети» делает
сравнение честным с первого дня вехи.

Наив «как сейчас» (:func:`naive_persist`) повторяет последнее наблюдение —
метрики бакета ts; наив «сутки назад» (:func:`naive_season`) — наблюдение того
же слота накануне (бакет ts − 24 ч). Оба горизонта контракта (15/30 минут)
получают одно и то же значение — свойство семейства persistence/seasonal, а не
недосмотр (design.md, «Open Questions»). Вероятность дефицита бинаризуется
правилом контракта (:func:`ml.dataset.deficit_label`) в 0.0/1.0 — сознательная
простота бейзлайна (design.md, Non-Goals). Нет исходного бакета — молчаливый
пропуск пары: пустой список без исключений (спека «Вчерашнего слота нет»).
"""

from datetime import datetime

from common.timeutil import add_minutes
from ml.dataset import RET_HORIZONS_MIN, MetricPoint, RowSource, deficit_label

NAIVE_PERSIST_V1 = "naive-persist-v1"
"""Версия наива «как сейчас» (persistence) в колонке ``model_version``."""

NAIVE_SEASON_V1 = "naive-season-v1"
"""Версия сезонного наива «сутки назад» в колонке ``model_version``."""

SEASON_LAG_MIN = 24 * 60
"""Сдвиг сезонного наива назад: ровно сутки, в минутах."""


def _forecast_rows(
    model_version: str, ts: datetime, zone_id: int, point: MetricPoint
) -> list[dict]:
    """Две dict-строки контракта ``forecasts``: одни и те же значения на 15/30.

    ``ts`` — момент прогноза (не целевой бакет: цель = ts + horizon_min);
    ключи строк совпадают с параметрами ``insert_forecasts``.
    """
    deficit_prob = float(deficit_label(point.requests, point.free_drivers, point.unfulfilled))
    return [
        {
            "ts": ts,
            "zone_id": zone_id,
            "horizon_min": horizon,
            "pred_requests": float(point.requests),
            "pred_free_drivers": float(point.free_drivers),
            "deficit_prob": deficit_prob,
            "model_version": model_version,
        }
        for horizon in RET_HORIZONS_MIN
    ]


def naive_persist(source: RowSource, ts: datetime, zone_id: int) -> list[dict]:
    """Наив «как сейчас»: прогноз = метрики бакета ts той же зоны.

    Нет бакета ts — пустой список (та же конвенция молчаливого пропуска, что
    и у сезонного наива).
    """
    point = source.metrics_at(zone_id, ts)
    if point is None:
        return []
    return _forecast_rows(NAIVE_PERSIST_V1, ts, zone_id, point)


def naive_season(source: RowSource, ts: datetime, zone_id: int) -> list[dict]:
    """Наив «сутки назад»: прогноз = метрики бакета ts − 24 ч той же зоны.

    Вчерашнего слота нет — пустой список: молчаливый пропуск пары (момент,
    зона), без исключений и строк-заглушек (спека «Вчерашнего слота нет»).
    """
    point = source.metrics_at(zone_id, add_minutes(ts, -SEASON_LAG_MIN))
    if point is None:
        return []
    return _forecast_rows(NAIVE_SEASON_V1, ts, zone_id, point)
