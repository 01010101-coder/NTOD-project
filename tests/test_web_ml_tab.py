"""Тесты вкладки «Лаборатория ML» (change ``m3-ml-tab``): REST ``/api/ml/*``
на временных фикстурах и статика страницы.

Фикстура (задача 1.1) строит маленькую SQLite-эпоху через
``storage.schema.init_db`` + точечные вставки ``zone_metrics_5min`` (2 зоны ×
7 бакетов 17:00–17:30; зона 0 — обеспечена, зона 5 — дефицитная по правилу
контракта) и ``forecasts`` (2 зоны × 4 ts × 2 горизонта × 3 версии: два наива
и сеть, включая строку с несуществующим факт-бакетом — хвост 17:45), плюс
tmp-каталог артефактов (``metrics.json`` + ``loss.json`` двух сетей и два
``summary-*.json`` — проверка выбора максимальной даты). Приложение —
``create_app`` с tmp-БД в ``cfg.ml.db``, tmp-артефактами и
``tick_interval=None`` через TestClient; мир — синтетический (2 зоны,
scale=0), файл тестов самостоятелен.

REST: ``models`` — слияние наивов и сетей (порядок «наивы, затем сети»),
пустые данные — пустой список, а не ошибка; ``summary`` — passthrough файла
с максимальной датой, без файла — 404; ``forecasts`` — фильтры комбинируются,
факт берётся из бакета ts + горизонт по правилу контракта, пропущенный бакет
даёт ``null``, порядок по ts; ``errors`` — остаток рукописного случая
(прогноз 10, факт 7 → +3), час ошибки — час суток ``ts``, фильтр горизонта.
Статика: таб-бар и id секций лаборатории на странице, переключение в CSS.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.server import create_app
from common.config import AppConfig
from storage.repo import insert_forecasts, insert_zone_metrics
from storage.schema import init_db

BASE_TS = datetime(2026, 12, 5, 17, 0, 0)
"""Начало рукописной эпохи: вечер субботы (час 17 проверяет разрез by_hour)."""

METRIC_STAMPS = [BASE_TS + timedelta(minutes=5 * i) for i in range(7)]
"""Бакеты метрик 17:00…17:30: факт-цели горизонта 15 всех строк и горизонта 30
только для ts=17:00 (17:35…17:45 отсутствуют — случай пропущенного факта)."""

FORECAST_STAMPS = [BASE_TS + timedelta(minutes=5 * i) for i in range(4)]
"""Штампы прогнозов 17:00…17:15; ts=17:15 с горизонтом 30 смотрит на 17:45 —
бакета нет, поля факта строки обязаны быть ``null``."""

NAIVE_VERSIONS = ("naive-persist-v1", "naive-season-v1")
"""Наивные версии в forecasts: артефактов у них нет — ``kind="naive"``."""

NN_VERSIONS = {"mlp-v1-test": "mlp", "lstm-v1-test": "lstm"}
"""Сети из артефактов; у ``mlp-v1-test`` есть и строки прогнозов в БД."""

NN_ROWS_VERSION = "mlp-v1-test"
"""Сеть, чьи прогнозы лежат в БД (2 наива + сеть); lstm-v1-test — только артефакты."""

MICRO_ZONES = [
    {
        "id": 0,
        "name": "a",
        "archetype": "residential",
        "attract_origin": 1.0,
        "attract_dest": 1.0,
        "map": {"x": 0.0, "y": 0.0},
    },
    {
        "id": 1,
        "name": "b",
        "archetype": "business",
        "attract_origin": 1.0,
        "attract_dest": 1.0,
        "map": {"x": 10.0, "y": 0.0},
    },
]

# Метрики зон: зона 0 — обеспеченная (8/10 ≥ 0.5, без истечений → не дефицит),
# зона 5 — дефицитная (2/7 < 0.5 и unfulfilled > 0) на всех бакетах.
METRICS_BY_ZONE = {
    0: {
        "requests": 10,
        "completed": 9,
        "unfulfilled": 0,
        "free_drivers": 8,
        "busy_drivers": 2,
        "avg_wait_min": 3.0,
        "supply_demand_ratio": 0.8,
    },
    5: {
        "requests": 7,
        "completed": 4,
        "unfulfilled": 1,
        "free_drivers": 2,
        "busy_drivers": 5,
        "avg_wait_min": 9.0,
        "supply_demand_ratio": 0.29,
    },
}


def _metrics_rows() -> list[dict]:
    """Строки ``zone_metrics_5min``: обе зоны × все бакеты 17:00–17:30."""

    rows = []
    for zone_id, values in METRICS_BY_ZONE.items():
        for ts in METRIC_STAMPS:
            rows.append(
                {
                    "ts": ts,
                    "zone_id": zone_id,
                    **values,
                    "is_rain": 0,
                    "event": "",
                    "price_multiplier": 1.0 if zone_id == 0 else 1.2,
                }
            )
    return rows


def _forecast_rows() -> list[dict]:
    """Строки ``forecasts``: 2 зоны × 4 ts × 2 горизонта × 3 версии.

    Рукописный случай остатков (сеть, зона 5, ts=17:00, горизонт 15):
    прогноз 10 при факте 7 → остаток +3, час ошибки — час суток ts.
    """

    rows = []
    for zone_id in METRICS_BY_ZONE:
        for i, ts in enumerate(FORECAST_STAMPS):
            for horizon in (15, 30):
                for version in NAIVE_VERSIONS + (NN_ROWS_VERSION,):
                    if version == "naive-persist-v1":
                        pred_requests = 10.0 if zone_id == 0 else 7.0
                        pred_free = 8.0 if zone_id == 0 else 2.0
                        prob = 0.1 if zone_id == 0 else 0.9
                    elif version == "naive-season-v1":
                        pred_requests = 6.0 if zone_id == 0 else 5.0
                        pred_free = 4.0
                        prob = 0.0 if zone_id == 0 else 0.6
                    else:
                        pred_requests = 5.0 + i
                        pred_free = 3.0
                        prob = 0.8 if (zone_id == 5 and horizon == 30) else 0.3
                        if (zone_id, i, horizon) == (5, 0, 15):
                            pred_requests = 10.0
                        if (zone_id, i, horizon) == (0, 0, 15):
                            pred_requests = 11.0
                    rows.append(
                        {
                            "ts": ts,
                            "zone_id": zone_id,
                            "horizon_min": horizon,
                            "pred_requests": pred_requests,
                            "pred_free_drivers": pred_free,
                            "deficit_prob": prob,
                            "model_version": version,
                        }
                    )
    return rows


def _write_artifacts(root: Path) -> None:
    """Tmp-артефакты: каталоги двух сетей и два summary (11-01 старше 11-06)."""

    root.mkdir(parents=True)
    for version, arch in NN_VERSIONS.items():
        model_dir = root / version
        model_dir.mkdir()
        metrics = {
            "model_version": version,
            "arch": arch,
            "window_hours": 3,
            "seed": 42,
            "best_epoch": 7 if arch == "mlp" else 5,
            "best_val": 0.7,
        }
        (model_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        loss = [
            {
                "epoch": e + 1,
                "train": {"total": 1.0 - 0.1 * e},
                "val": {"total": 0.9 - 0.05 * e},
            }
            for e in range(3)
        ]
        (model_dir / "loss.json").write_text(json.dumps(loss), encoding="utf-8")

    old = {"date": "2026-11-01", "summaries": [{"model_version": "old-run"}]}
    (root / "summary-2026-11-01.json").write_text(json.dumps(old), encoding="utf-8")
    latest = {
        "date": "2026-11-06",
        "summaries": [
            {
                "model_version": "naive-persist-v1",
                "regression": {
                    "pred_requests": {
                        "15": {"mae": 1.0, "rmse": 1.2},
                        "30": {"mae": 1.1, "rmse": 1.3},
                    },
                    "pred_free_drivers": {},
                },
                "deficit": {
                    "15": {"precision": 0.5, "recall": 0.4, "f1": 0.45},
                    "30": {"precision": 0.6, "recall": 0.5, "f1": 0.55},
                },
                "calibration": {
                    "15": [
                        {
                            "bin_low": 0.8,
                            "bin_high": 0.9,
                            "count": 10,
                            "mean_prob": 0.85,
                            "deficit_share": 0.8,
                        }
                    ],
                    "30": [],
                },
            },
            {"model_version": "mlp-v1-test", "regression": {}, "deficit": {}, "calibration": {}},
        ],
    }
    (root / "summary-2026-11-06.json").write_text(json.dumps(latest), encoding="utf-8")


def lab_config(db_path: Path) -> AppConfig:
    """Синтетический мир (2 зоны, тихий) с БД эпохи в tmp_path."""

    return AppConfig.model_validate(
        {
            "seed": 7,
            "city": {
                "zones": MICRO_ZONES,
                "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 30.0}],
            },
            "population": {"scale": 0.0},
            "drivers": {"count": 3, "wait_limit_min": 6},
            "weather": {
                "rain_day_chance": 0.0,
                "episode_start_h": [0, 0],
                "episode_duration_h": [1, 1],
            },
            "pricing": {"surge_injection": {"day_chance": 0.0}},
            "ml": {"db": str(db_path)},
        }
    )


def make_lab_app(db_path: Path, artifacts_dir: Path):
    """Приложение без фонового тика и с коротким тихим прогоном мира."""

    return create_app(
        lab_config(db_path), days=0.05, tick_interval=None, artifacts_dir=artifacts_dir
    )


@pytest.fixture()
def lab(tmp_path):
    """Наполненная эпоха + артефакты + TestClient."""

    db_path = tmp_path / "epoch.db"
    conn = sqlite3.connect(db_path)
    init_db(conn)
    insert_zone_metrics(conn, _metrics_rows())
    insert_forecasts(conn, _forecast_rows())
    conn.close()
    artifacts = tmp_path / "artifacts"
    _write_artifacts(artifacts)
    with TestClient(make_lab_app(db_path, artifacts)) as client:
        yield client, artifacts


@pytest.fixture()
def lab_empty(tmp_path):
    """Пустая эпоха (схема без строк) и пустой каталог артефактов."""

    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(db_path)
    init_db(conn)
    conn.close()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    with TestClient(make_lab_app(db_path, artifacts)) as client:
        yield client


@pytest.fixture()
def lab_missing(tmp_path):
    """БД эпохи и каталог артефактов не существуют вовсе."""

    with TestClient(make_lab_app(tmp_path / "nope.db", tmp_path / "artifacts")) as client:
        yield client


def test_fixtures_build_expected_rows(tmp_path):
    """Фикстура создаёт БД и файлы; счётчики строк соответствуют вставленному (1.1)."""

    db_path = tmp_path / "epoch.db"
    conn = sqlite3.connect(db_path)
    init_db(conn)
    metrics = _metrics_rows()
    forecasts = _forecast_rows()
    insert_zone_metrics(conn, metrics)
    insert_forecasts(conn, forecasts)
    conn.close()
    assert len(metrics) == 2 * len(METRIC_STAMPS)
    assert len(forecasts) == 2 * len(FORECAST_STAMPS) * 2 * 3
    check = sqlite3.connect(db_path)
    assert check.execute("SELECT COUNT(*) FROM zone_metrics_5min").fetchone()[0] == len(metrics)
    assert check.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == len(forecasts)
    check.close()
    artifacts = tmp_path / "artifacts"
    _write_artifacts(artifacts)
    assert len(list(artifacts.glob("*/metrics.json"))) == len(NN_VERSIONS)
    assert (artifacts / "summary-2026-11-06.json").exists()


def test_models_fusion_naives_and_networks(lab):
    """4 записи: наивы без артефактов, затем сети с arch/окном/эпохой и loss (2.1)."""

    client, _ = lab
    response = client.get("/api/ml/models")
    assert response.status_code == 200
    models = response.json()["models"]
    versions = [model["model_version"] for model in models]
    # Порядок «наивы, затем сети»; mlp-v1-test есть и в forecasts, и в артефактах —
    # слияние оставляет одну запись kind="nn".
    assert versions == ["naive-persist-v1", "naive-season-v1", "lstm-v1-test", "mlp-v1-test"]
    by_version = {model["model_version"]: model for model in models}
    for naive in NAIVE_VERSIONS:
        assert by_version[naive]["kind"] == "naive"
        assert "loss" not in by_version[naive] and "arch" not in by_version[naive]
    for version, arch in NN_VERSIONS.items():
        entry = by_version[version]
        assert entry["kind"] == "nn"
        assert entry["arch"] == arch
        assert entry["window_hours"] == 3
        assert isinstance(entry["best_epoch"], int)
        assert entry["loss"] and "epoch" in entry["loss"][0]
        assert "train" in entry["loss"][0] and "val" in entry["loss"][0]


def test_models_empty_data(lab_empty):
    """Пустая БД и пустой каталог артефактов — 200 с пустым списком (2.1)."""

    assert lab_empty.get("/api/ml/models").json() == {"models": []}


def test_models_missing_db_and_artifacts(lab_missing):
    """Нет ни БД эпохи, ни каталога артефактов — 200 с пустым списком (устойчивость)."""

    assert lab_missing.get("/api/ml/models").status_code == 200
    assert lab_missing.get("/api/ml/models").json() == {"models": []}


def test_summary_passthrough_latest(lab):
    """Сводка — содержимое файла с максимальной датой, структура без изменений (2.2)."""

    client, artifacts = lab
    response = client.get("/api/ml/summary")
    assert response.status_code == 200
    expected = json.loads((artifacts / "summary-2026-11-06.json").read_text(encoding="utf-8"))
    assert response.json() == expected
    assert response.json()["date"] == "2026-11-06"
    assert len(response.json()["summaries"]) == 2


def test_summary_missing(lab_empty):
    """Без summary-*.json — 404 с понятным сообщением (2.2)."""

    response = lab_empty.get("/api/ml/summary")
    assert response.status_code == 404
    assert "сводка" in response.json()["detail"].lower()


def test_forecasts_filters_fact_and_deficit(lab):
    """Фильтры комбинируются; факт — из бакета ts+15; дефицит по правилу контракта (3.1)."""

    client, _ = lab
    rows = client.get(
        "/api/ml/forecasts",
        params={"zone": 5, "model": "mlp-v1-test", "horizon": 15},
    ).json()["forecasts"]
    assert len(rows) == len(FORECAST_STAMPS)
    stamps = [row["ts"] for row in rows]
    assert stamps == sorted(stamps)
    assert stamps == [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in FORECAST_STAMPS]
    assert all(
        row["zone_id"] == 5 and row["horizon_min"] == 15 and row["model_version"] == "mlp-v1-test"
        for row in rows
    )
    # Рукописный случай: факт бакета ts+15 той же зоны, дефицит по правилу контракта.
    head = rows[0]
    assert head["ts"] == "2026-12-05 17:00:00"
    assert head["pred_requests"] == pytest.approx(10.0)
    assert head["actual_requests"] == pytest.approx(7.0)  # requests зоны 5 в 17:15
    assert head["actual_free_drivers"] == pytest.approx(2.0)
    assert head["actual_deficit"] is True
    supplied = client.get(
        "/api/ml/forecasts", params={"zone": 0, "model": "mlp-v1-test", "horizon": 15}
    ).json()["forecasts"]
    assert supplied[0]["actual_deficit"] is False  # 8/10 ≥ 0.5 и без истечений
    assert supplied[0]["actual_requests"] == pytest.approx(10.0)


def test_forecasts_null_fact_tail(lab):
    """Хвост без факт-бакета (17:15 + 30 мин = 17:45) — поля факта null (3.1)."""

    client, _ = lab
    rows = client.get(
        "/api/ml/forecasts", params={"zone": 0, "model": "mlp-v1-test", "horizon": 30}
    ).json()["forecasts"]
    by_ts = {row["ts"]: row for row in rows}
    assert by_ts["2026-12-05 17:15:00"]["actual_requests"] is None
    assert by_ts["2026-12-05 17:15:00"]["actual_free_drivers"] is None
    assert by_ts["2026-12-05 17:15:00"]["actual_deficit"] is None
    assert by_ts["2026-12-05 17:00:00"]["actual_requests"] is not None


def test_forecasts_unknown_zone_empty(lab):
    """Несуществующая зона — 200 с пустым списком (3.1)."""

    response = lab[0].get("/api/ml/forecasts", params={"zone": 99})
    assert response.status_code == 200
    assert response.json() == {"forecasts": []}


def test_forecasts_from_to_inclusive(lab):
    """Границы from/to включаются, как в /api/metrics (3.1)."""

    client, _ = lab
    rows = client.get(
        "/api/ml/forecasts",
        params={
            "zone": 5,
            "model": "naive-persist-v1",
            "horizon": 15,
            "from": "2026-12-05 17:05:00",
            "to": "2026-12-05 17:10:00",
        },
    ).json()["forecasts"]
    assert [row["ts"] for row in rows] == [
        "2026-12-05 17:05:00",
        "2026-12-05 17:10:00",
    ]


def test_forecasts_invalid_horizon(lab):
    """horizon вне контракта {15, 30} — 422 с сообщением (3.1)."""

    response = lab[0].get("/api/ml/forecasts", params={"horizon": 20})
    assert response.status_code == 422
    assert "15 и 30" in response.json()["detail"]


def test_errors_hand_case_and_by_hour(lab):
    """Остаток прогноз 10 / факт 7 → +3; час ошибки — час суток ts (3.2)."""

    client, _ = lab
    data = client.get(
        "/api/ml/errors", params={"zone": 5, "model": "mlp-v1-test", "horizon": 15}
    ).json()
    residuals = {entry["ts"]: entry["residual"] for entry in data["residuals"]}
    assert residuals["2026-12-05 17:00:00"] == pytest.approx(3.0)  # 10 − 7
    assert residuals["2026-12-05 17:05:00"] == pytest.approx(-1.0)  # 6 − 7
    assert residuals["2026-12-05 17:10:00"] == pytest.approx(0.0)
    assert residuals["2026-12-05 17:15:00"] == pytest.approx(1.0)  # 8 − 7
    # Все ts — час 17: ровно одна запись by_hour с MAE (3+1+0+1)/4 и count 4.
    by_hour = data["by_hour"]
    assert len(by_hour) == 1
    assert by_hour[0]["hour"] == 17
    assert by_hour[0]["mae"] == pytest.approx(1.25)
    assert by_hour[0]["count"] == 4
    # Гистограмма: 20 бинов, края покрывают диапазон остатков, суммы сходятся.
    histogram = data["histogram"]
    assert len(histogram["counts"]) == 20
    assert len(histogram["bin_edges"]) == 21
    assert sum(histogram["counts"]) == 4
    assert histogram["bin_edges"][0] <= -1.0
    assert histogram["bin_edges"][-1] >= 3.0


def test_errors_horizon_filter(lab):
    """horizon=30 без модели: только строки горизонта 30 всех моделей, без факта — мимо (3.2)."""

    client, _ = lab
    data = client.get("/api/ml/errors", params={"horizon": 30}).json()
    # Факт-бакеты 17:35…17:45 отсутствуют — с фактом только строки ts=17:00
    # (цель 17:30): 2 зоны × 3 версии = 6 остатков.
    stamps = {entry["ts"] for entry in data["residuals"]}
    assert stamps == {"2026-12-05 17:00:00"}
    assert len(data["residuals"]) == 6
    assert sum(data["histogram"]["counts"]) == 6
    assert {cell["hour"] for cell in data["by_hour"]} == {17}


def test_errors_skips_rows_without_fact(lab):
    """Строки без факта пропускаются: у горизонта 30 зоны 0 остаток один (3.2)."""

    client, _ = lab
    data = client.get(
        "/api/ml/errors", params={"zone": 0, "model": "naive-persist-v1", "horizon": 30}
    ).json()
    assert [entry["ts"] for entry in data["residuals"]] == ["2026-12-05 17:00:00"]
    assert data["residuals"][0]["residual"] == pytest.approx(0.0)  # 10 − 10
    assert data["by_hour"][0]["count"] == 1


def test_errors_empty_data(lab_empty):
    """Пустая эпоха — пустые остатки/гистограмма/by_hour без ошибок 500."""

    data = lab_empty.get("/api/ml/errors").json()
    assert data["residuals"] == []
    assert data["histogram"] == {"bin_edges": [], "counts": []}
    assert data["by_hour"] == []


def test_static_page_lab_tab(lab):
    """Страница: таб-бар «Город / Лаборатория», id секций лаборатории; CSS переключает."""

    client, _ = lab
    page = client.get("/").text
    assert 'data-tab="city"' in page
    assert 'data-tab="lab"' in page
    assert 'id="tab-bar"' in page
    for section_id in (
        "lab",
        "lab-empty",
        "lab-grid",
        "model-panel",
        "summary-panel",
        "forecast-panel",
        "errors-panel",
        "calibration-panel",
    ):
        assert f'id="{section_id}"' in page, f"нет секции {section_id}"
    style = client.get("/static/style.css").text
    assert "body[data-tab" in style
    assert "#lab-grid" in style
    scripts = client.get("/static/app.js").text
    assert "/api/ml/models" in scripts
    assert "/api/ml/forecasts" in scripts


def test_city_endpoints_unaffected(lab):
    """Городской контракт не изменился: /api/city отвечает как раньше (4.1)."""

    client, _ = lab
    data = client.get("/api/city").json()
    assert [zone["id"] for zone in data["zones"]] == [0, 1]
