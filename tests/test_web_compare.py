"""Тесты сравнения веток вехи M5+M6 (change ``m6-compare-tab``).

Расчёт исходов (:mod:`app.compare`) проверяется на мини-БД в tmp: ``init_db``
схемы контракта + точечные вставки ``trips``/``zone_metrics_5min``/``forecasts``
с известными значениями внутри и вне окна тест-недели. Числа тестов — ручной
расчёт по спеке ``web-compare-tab``: выручка по формуле §10.15 M2, ожидание
``pickup_at − created_at``, доля дефицита по правилу контракта, MAE/F1
тень-моделей против факта ветки A.

Окно теста сужено до одних суток (``ml.train_days = ml.test_days = 1``):
старт мира ``2026-11-02`` → окно ``[2026-11-03 00:00, 2026-11-04 00:00)``,
часовой ряд — 24 точки. Мир синтетический (2 зоны, дорога 30 мин).
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.compare import (
    BASE_BRANCH,
    OUTCOME_KEYS,
    CompareWindow,
    accuracy_vs_base,
    branch_outcomes,
    build_compare,
)
from app.compare import (
    test_window as compare_test_window,
)
from app.server import create_app
from common.config import AppConfig
from storage.repo import insert_forecasts, insert_trips, insert_zone_metrics
from storage.schema import init_db

WINDOW_START = datetime(2026, 11, 3, 0, 0, 0)
"""Начало тест-недели: старт мира 2026-11-02 + train_days=1 (микро-конфиг)."""

WINDOW_END = datetime(2026, 11, 4, 0, 0, 0)
"""Конец окна (не включается): + test_days=1."""

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
"""Две зоны: путь 0↔1 — 30 минут, диагональ — 0 (цена = base_boarding)."""


def compare_config(**overrides) -> AppConfig:
    """Синтетический мир с суточным тест-окном и тихим прогоном."""

    base = {
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
        "ml": {"train_days": 1, "test_days": 1},
    }
    base.update(overrides)
    return AppConfig.model_validate(base)


def _trip(
    created_at: datetime,
    *,
    origin: int = 0,
    dest: int = 0,
    status: str = "completed",
    pickup_at: datetime | None = None,
    price_multiplier: float = 1.0,
) -> dict:
    """Строка ``trips`` контракта: поездка с созданием, подачей и статусом."""

    return {
        "created_at": created_at,
        "origin_zone": origin,
        "dest_zone": dest,
        "assigned_at": created_at + timedelta(minutes=2) if pickup_at else None,
        "pickup_at": pickup_at,
        "dropoff_at": pickup_at + timedelta(minutes=10) if pickup_at else None,
        "status": status,
        "price_multiplier": price_multiplier,
    }


def _metrics_row(
    ts: datetime,
    zone_id: int,
    *,
    requests: int,
    free_drivers: int,
    unfulfilled: int = 0,
) -> dict:
    """Строка ``zone_metrics_5min`` контракта с достаточными для правила полями."""

    return {
        "ts": ts,
        "zone_id": zone_id,
        "requests": requests,
        "completed": requests,
        "unfulfilled": unfulfilled,
        "free_drivers": free_drivers,
        "busy_drivers": 0,
        "avg_wait_min": 0.0,
        "supply_demand_ratio": free_drivers / max(requests, 1),
        "is_rain": 0,
        "event": "",
        "price_multiplier": 1.0,
    }


def make_db(
    path: Path,
    *,
    trips: list[dict] | None = None,
    metrics: list[dict] | None = None,
    forecasts: list[dict] | None = None,
) -> Path:
    """Создать мини-БД схемы контракта и вставить заданные строки."""

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        init_db(conn)
        insert_trips(conn, trips or [])
        insert_zone_metrics(conn, metrics or [])
        insert_forecasts(conn, forecasts or [])
    finally:
        conn.close()
    return path


def _forecast(
    ts: datetime,
    zone_id: int,
    horizon: int,
    *,
    pred_requests: float,
    prob: float,
    family: str = "mlp",
) -> dict:
    """Строка ``forecasts`` одной тень-модели с фиксированным стоком."""

    return {
        "ts": ts,
        "zone_id": zone_id,
        "horizon_min": horizon,
        "pred_requests": pred_requests,
        "pred_free_drivers": 5.0,
        "deficit_prob": prob,
        "model_version": family,
    }


# ── 1.1 Окно тест-недели и исходы ветки ──


def test_test_window_follows_ml_roles():
    """Окно — ``[world_start + train_days, + test_days)`` (задача 1.1)."""

    window = compare_test_window(compare_config())
    assert window == CompareWindow(WINDOW_START, WINDOW_END)
    wider = compare_config(ml={"train_days": 2, "test_days": 3})
    assert compare_test_window(wider) == CompareWindow(datetime(2026, 11, 4), datetime(2026, 11, 7))


def test_branch_outcomes_manual(tmp_path):
    """Исходы окна совпадают с ручным расчётом; строки вне окна не считаются (1.1)."""

    db = make_db(
        tmp_path / "a-base.db",
        trips=[
            # Внутри окна: 0→1 (460 руб.), 1→0 с наценкой 1.5 (690 руб.).
            _trip(
                datetime(2026, 11, 3, 0, 10),
                origin=0,
                dest=1,
                pickup_at=datetime(2026, 11, 3, 0, 18),
            ),
            _trip(
                datetime(2026, 11, 3, 1, 30),
                origin=1,
                dest=0,
                pickup_at=datetime(2026, 11, 3, 1, 40),
                price_multiplier=1.5,
            ),
            _trip(datetime(2026, 11, 3, 0, 20), status="unfulfilled"),
            _trip(datetime(2026, 11, 3, 0, 30), status="cancelled"),
            # Вне окна: до старта и на границе end (полуинтервал).
            _trip(
                datetime(2026, 11, 2, 23, 50),
                pickup_at=datetime(2026, 11, 2, 23, 55),
            ),
            _trip(
                datetime(2026, 11, 4, 0, 0),
                pickup_at=datetime(2026, 11, 4, 0, 10),
            ),
        ],
        metrics=[
            # В окне: обеспеченная зона, дефицитная по ratio, дефицитная по нулю
            # заказов (0 / max(0,1) < 0.5).
            _metrics_row(datetime(2026, 11, 3, 0, 0), 0, requests=10, free_drivers=8),
            _metrics_row(datetime(2026, 11, 3, 0, 0), 1, requests=10, free_drivers=4),
            _metrics_row(datetime(2026, 11, 3, 0, 5), 0, requests=0, free_drivers=0),
            # Вне окна — не в доле дефицита.
            _metrics_row(datetime(2026, 11, 4, 0, 0), 0, requests=10, free_drivers=10),
        ],
    )
    cfg = compare_config()
    window = compare_test_window(cfg)
    outcomes = branch_outcomes(db, cfg, cfg_path_min(cfg), window)

    assert outcomes["trips"] == 2
    assert outcomes["unfulfilled"] == 1
    assert outcomes["avg_wait_min"] == pytest.approx(9.0)  # (8 + 10) / 2
    assert outcomes["revenue"] == pytest.approx(460.0 + 690.0)
    assert outcomes["deficit_share"] == pytest.approx(2 / 3)

    series = outcomes["revenue_series"]
    assert len(series) == 24  # test_days × 24
    assert series == sorted(series)  # не убывает
    assert series[0] == pytest.approx(460.0)  # заказ часа 0
    assert series[1] == pytest.approx(1150.0)  # плюс заказ часа 1
    assert series[-1] == pytest.approx(outcomes["revenue"])


def cfg_path_min(cfg: AppConfig):
    """Матрица кратчайших путей конфига (``build_city`` без побочных эффектов)."""

    from simulator.city import build_city

    return build_city(cfg).path_min


def test_branch_outcomes_empty_db(tmp_path):
    """Пустая БД — нули и ``null`` без исключений, ряд из нулей (1.1)."""

    db = make_db(tmp_path / "empty.db")
    cfg = compare_config()
    outcomes = branch_outcomes(db, cfg, cfg_path_min(cfg), compare_test_window(cfg))
    assert outcomes["trips"] == 0
    assert outcomes["unfulfilled"] == 0
    assert outcomes["avg_wait_min"] is None
    assert outcomes["revenue"] == 0.0
    assert outcomes["deficit_share"] is None
    assert outcomes["revenue_series"] == [0.0] * 24


def test_branch_outcomes_missing_db(tmp_path):
    """Отсутствующий файл БД — те же нули/``null``, без исключения (1.1)."""

    cfg = compare_config()
    outcomes = branch_outcomes(
        tmp_path / "nope.db", cfg, cfg_path_min(cfg), compare_test_window(cfg)
    )
    assert outcomes["trips"] == 0 and outcomes["avg_wait_min"] is None
    assert outcomes["revenue_series"] == [0.0] * 24


# ── 1.2 Точность против A ──


def _accuracy_db(path: Path) -> Path:
    """Мини-БД A с известными прогнозами и фактом целевых бакетов.

    Факт: 00:15 зона 0 — обеспечена, зона 1 — дефицитна; 00:30 зона 0 —
    дефицитна, зона 1 — обеспечена. Прогнозы mlp: h15 z0 prob .8 (FP) и z1 .7
    (TP) → precision .5, recall 1; h30 z0 .9 (TP) и z1 .2 (TN) → F1 1.
    lstm h15: оба прогноза .1 → FN у дефицитной зоны 1, precision/recall 0.
    """

    return make_db(
        path,
        metrics=[
            _metrics_row(datetime(2026, 11, 3, 0, 15), 0, requests=10, free_drivers=8),
            _metrics_row(datetime(2026, 11, 3, 0, 15), 1, requests=10, free_drivers=2),
            _metrics_row(datetime(2026, 11, 3, 0, 30), 0, requests=10, free_drivers=4),
            _metrics_row(datetime(2026, 11, 3, 0, 30), 1, requests=10, free_drivers=8),
        ],
        forecasts=[
            _forecast(datetime(2026, 11, 3, 0, 0), 0, 15, pred_requests=12.0, prob=0.8),
            _forecast(datetime(2026, 11, 3, 0, 0), 1, 15, pred_requests=8.0, prob=0.7),
            _forecast(datetime(2026, 11, 3, 0, 0), 0, 30, pred_requests=9.0, prob=0.9),
            _forecast(datetime(2026, 11, 3, 0, 0), 1, 30, pred_requests=10.0, prob=0.2),
            _forecast(
                datetime(2026, 11, 3, 0, 0), 0, 15, pred_requests=10.0, prob=0.1, family="lstm"
            ),
            _forecast(
                datetime(2026, 11, 3, 0, 0), 1, 15, pred_requests=10.0, prob=0.1, family="lstm"
            ),
        ],
    )


def test_accuracy_vs_base_manual(tmp_path):
    """MAE/F1 каждого семейства и горизонта — ручной расчёт, источник A (1.2)."""

    db = _accuracy_db(tmp_path / "a-base.db")
    cfg = compare_config()
    accuracy = accuracy_vs_base(db, cfg, compare_test_window(cfg))

    assert accuracy["source"] == BASE_BRANCH
    by_key = {(row["family"], row["horizon_min"]): row for row in accuracy["rows"]}
    assert set(by_key) == {("mlp", 15), ("mlp", 30), ("lstm", 15)}

    mlp15 = by_key[("mlp", 15)]
    assert mlp15["mae_requests"] == pytest.approx(2.0)  # (|12−10| + |8−10|) / 2
    assert mlp15["precision"] == pytest.approx(0.5)  # TP=1, FP=1
    assert mlp15["recall"] == pytest.approx(1.0)
    assert mlp15["f1"] == pytest.approx(2 / 3)

    mlp30 = by_key[("mlp", 30)]
    assert mlp30["mae_requests"] == pytest.approx(0.5)  # (|9−10| + |10−10|) / 2
    assert mlp30["precision"] == pytest.approx(1.0)
    assert mlp30["recall"] == pytest.approx(1.0)
    assert mlp30["f1"] == pytest.approx(1.0)

    lstm15 = by_key[("lstm", 15)]
    assert lstm15["mae_requests"] == pytest.approx(0.0)
    assert lstm15["precision"] == pytest.approx(0.0)
    assert lstm15["recall"] == pytest.approx(0.0)
    assert lstm15["f1"] == pytest.approx(0.0)


def test_accuracy_empty_without_forecasts(tmp_path):
    """БД A без прогнозов — пустой раздел, без ошибки (1.2)."""

    db = make_db(
        tmp_path / "a-base.db",
        metrics=[_metrics_row(datetime(2026, 11, 3, 0, 15), 0, requests=10, free_drivers=8)],
    )
    cfg = compare_config()
    assert accuracy_vs_base(db, cfg, compare_test_window(cfg)) == {
        "source": BASE_BRANCH,
        "rows": [],
    }


def test_accuracy_empty_without_db(tmp_path):
    """Отсутствующая БД A — пустой раздел, без исключения (1.2)."""

    cfg = compare_config()
    assert accuracy_vs_base(tmp_path / "nope.db", cfg, compare_test_window(cfg))["rows"] == []


# ── 1.3 Сборка ответа и дельты к A ──


def _revenue_db(path: Path, multiplier: float) -> Path:
    """БД ветки с одним завершённым заказом: выручка = base_boarding × множитель."""

    return make_db(
        path,
        trips=[
            _trip(
                datetime(2026, 11, 3, 12, 0),
                pickup_at=datetime(2026, 11, 3, 12, 5),
                price_multiplier=multiplier,
            )
        ],
    )


def _revenue_cfg() -> AppConfig:
    """Конфиг с ценой 1000 руб. за заказ без учёта минут (ручная дельта 5 %)."""

    return compare_config(
        pricing={
            "base_boarding": 1000.0,
            "per_min_rub": 0.0,
            "surge_injection": {"day_chance": 0.0},
        }
    )


def test_build_compare_deltas_and_order(tmp_path):
    """Ветки в порядке реестра; дельта выручки 1000→1050 = 5.0; у A дельт нет (1.3)."""

    worlds = tmp_path / "worlds"
    _revenue_db(worlds / "a-base.db", 1.0)
    _revenue_db(worlds / "c-mlp.db", 1.05)
    cfg = _revenue_cfg()

    data = build_compare(cfg, worlds)
    assert data["window"] == {
        "start": "2026-11-03 00:00:00",
        "end": "2026-11-04 00:00:00",
    }
    branches = data["branches"]
    assert [branch["name"] for branch in branches] == [
        "a-base",
        "b-reactive",
        "c-mlp",
        "d-lstm",
    ]

    a_base, b_reactive, c_mlp, d_lstm = branches
    assert a_base["exists"] is True and a_base["outcomes"]["revenue"] == pytest.approx(1000.0)
    assert a_base["delta"] == dict.fromkeys(OUTCOME_KEYS)  # у A дельт нет

    assert b_reactive["exists"] is False
    assert b_reactive["outcomes"] == dict.fromkeys(OUTCOME_KEYS)  # null-исходы
    assert b_reactive["delta"] == dict.fromkeys(OUTCOME_KEYS)
    assert b_reactive["revenue_series"] is None

    assert c_mlp["outcomes"]["revenue"] == pytest.approx(1050.0)
    assert c_mlp["delta"]["revenue"] == pytest.approx(5.0)
    # База unfulfilled у A = 0 → дельта null, не Infinity (сценарий спеки).
    assert c_mlp["delta"]["unfulfilled"] is None

    assert d_lstm["exists"] is False
    assert len(a_base["revenue_series"]) == 24
    assert a_base["revenue_series"][-1] == pytest.approx(1000.0)
    assert c_mlp["revenue_series"][-1] == pytest.approx(1050.0)
    # Распределение тарифов: у отсутствующей ветки пусто, у существующей — сумма = бакеты.
    assert b_reactive["tariff_mix"] == []
    assert all(item["count"] > 0 for item in a_base["tariff_mix"])
    mults = [item["mult"] for item in c_mlp["tariff_mix"]]
    assert mults == sorted(mults)


def test_build_compare_without_base(tmp_path):
    """Без БД A — дельты и точность пусты, остальные ветки рассчитаны (1.3)."""

    worlds = tmp_path / "worlds"
    _revenue_db(worlds / "c-mlp.db", 1.05)
    cfg = _revenue_cfg()

    data = build_compare(cfg, worlds)
    by_name = {branch["name"]: branch for branch in data["branches"]}
    assert by_name["a-base"]["exists"] is False
    assert by_name["c-mlp"]["exists"] is True
    for branch in data["branches"]:
        assert branch["delta"] == dict.fromkeys(OUTCOME_KEYS)
    assert data["accuracy"] == {"source": BASE_BRANCH, "rows": []}


def test_build_compare_no_worlds(tmp_path):
    """Пустой каталог миров — все ``exists=false``, пустая точность (1.3)."""

    worlds = tmp_path / "worlds"
    worlds.mkdir()
    data = build_compare(_revenue_cfg(), worlds)
    assert all(branch["exists"] is False for branch in data["branches"])
    assert all(branch["outcomes"] == dict.fromkeys(OUTCOME_KEYS) for branch in data["branches"])
    assert data["accuracy"]["rows"] == []


def test_accuracy_ignores_branch_forecasts(tmp_path):
    """Изменение ``forecasts`` в БД C не меняет раздел точности (1.3)."""

    worlds = tmp_path / "worlds"
    _accuracy_db(worlds / "a-base.db")
    _revenue_db(worlds / "c-mlp.db", 1.05)
    cfg = compare_config()

    first = build_compare(cfg, worlds)
    assert first["accuracy"]["rows"], "точность A должна быть рассчитана"

    # Подменяем прогнозы C на заведомо другие: раздел точности обязан совпасть.
    conn = sqlite3.connect(worlds / "c-mlp.db")
    try:
        conn.execute("DELETE FROM forecasts")
        conn.commit()
        insert_forecasts(
            conn,
            [
                _forecast(
                    datetime(2026, 11, 3, 0, 0),
                    0,
                    15,
                    pred_requests=99.0,
                    prob=0.99,
                )
            ],
        )
    finally:
        conn.close()

    second = build_compare(cfg, worlds)
    assert second["accuracy"] == first["accuracy"]


# ── 2.1 REST сравнения и кэш ──


def make_client(worlds: Path, cfg: AppConfig | None = None) -> TestClient:
    """TestClient сравнения: тихий мир и короткий фоновый прогон без тика."""

    return TestClient(
        create_app(cfg or compare_config(), days=0.05, tick_interval=None, worlds_dir=worlds)
    )


def test_api_compare_response(tmp_path):
    """GET /api/ml/compare: 200, ключи ответа, дельты и порядок веток (2.1)."""

    worlds = tmp_path / "worlds"
    _revenue_db(worlds / "a-base.db", 1.0)
    _revenue_db(worlds / "c-mlp.db", 1.05)

    with make_client(worlds, _revenue_cfg()) as client:
        response = client.get("/api/ml/compare")

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"window", "branches", "accuracy"}
    assert data["window"] == {
        "start": "2026-11-03 00:00:00",
        "end": "2026-11-04 00:00:00",
    }
    assert [branch["name"] for branch in data["branches"]] == [
        "a-base",
        "b-reactive",
        "c-mlp",
        "d-lstm",
    ]
    by_name = {branch["name"]: branch for branch in data["branches"]}
    assert set(by_name["c-mlp"]) == {
        "name",
        "label",
        "exists",
        "channel",
        "outcomes",
        "delta",
        "revenue_series",
        "tariff_mix",
    }
    assert by_name["c-mlp"]["delta"]["revenue"] == pytest.approx(5.0)
    assert by_name["b-reactive"]["exists"] is False


def test_api_compare_empty_worlds(tmp_path):
    """Пустой каталог миров — 200, все ветки ``exists=false`` (2.1)."""

    worlds = tmp_path / "worlds"
    worlds.mkdir()
    with make_client(worlds) as client:
        response = client.get("/api/ml/compare")

    assert response.status_code == 200
    data = response.json()
    assert len(data["branches"]) == 4
    assert all(branch["exists"] is False for branch in data["branches"])
    assert all(branch["outcomes"] == dict.fromkeys(OUTCOME_KEYS) for branch in data["branches"])
    assert data["accuracy"] == {"source": BASE_BRANCH, "rows": []}


def test_api_compare_cache_and_rebuild(tmp_path, monkeypatch):
    """Повторный запрос не пересчитывает; новая БД ветки — пересчёт (2.1)."""

    import app.server as server_module

    worlds = tmp_path / "worlds"
    _revenue_db(worlds / "a-base.db", 1.0)

    calls = {"count": 0}
    real_build = server_module.build_compare

    def counting_build(cfg, worlds_dir=None):
        calls["count"] += 1
        return real_build(cfg, worlds_dir)

    monkeypatch.setattr(server_module, "build_compare", counting_build)

    with make_client(worlds) as client:
        first = client.get("/api/ml/compare")
        second = client.get("/api/ml/compare")
        assert first.status_code == second.status_code == 200
        assert calls["count"] == 1  # второй запрос взят из кэша
        assert {b["name"]: b["exists"] for b in second.json()["branches"]}["d-lstm"] is False

        # Появление файла d-lstm меняет сигнатуру кэша — ответ пересчитан.
        _revenue_db(worlds / "d-lstm.db", 1.0)
        third = client.get("/api/ml/compare")

    assert calls["count"] == 2
    assert {b["name"]: b["exists"] for b in third.json()["branches"]}["d-lstm"] is True


# ── 3.1–3.2 Статика вкладки «Сравнение» ──


def test_static_compare_tab(tmp_path):
    """Страница: кнопка, секция и id элементов вкладки; CSS и JS на месте (3.1–3.2)."""

    worlds = tmp_path / "worlds"
    worlds.mkdir()
    with make_client(worlds) as client:
        page = client.get("/").text
        style = client.get("/static/style.css").text
        script = client.get("/static/app.js").text

    for tab in ("city", "lab", "compare"):
        assert f'data-tab="{tab}"' in page, f"нет кнопки вкладки {tab}"
    for element_id in (
        "compare",
        "compare-empty",
        "compare-grid",
        "compare-window",
        "compare-table",
        "compare-revenue-panel",
        "compare-legend",
        "revenue-chart",
        "compare-revenue-note",
        "compare-accuracy-panel",
        "compare-accuracy",
    ):
        assert f'id="{element_id}"' in page, f"нет элемента {element_id}"
    assert "python -m simulator.branches --all" in page  # подсказка пустого состояния
    assert "против факта ветки A" in page
    assert "политика меняет факт" in page
    assert "self-fulfilling prophecy" in page

    # Девять колонок исходов и канал ветки (change world-final-config).
    for title in (
        "поездки",
        "не обслужено",
        "отпугнуто",
        "спасено",
        "переключения цены",
        "часы вызванных водителей",
        "доля дефицита",
        "канал",
    ):
        assert title in script, f"нет заголовка колонки {title}"
    assert "branch.channel" in script  # канал из meta.json в строке ветки
    assert "'better': 'more'" in script or "better: 'more'" in script  # правило подсветки
    assert "branch.delta?.[column.key]" in script  # дельты только у абсолютных исходов

    assert 'body[data-tab="compare"]' in style
    assert "#compare-grid" in style
    assert "#compare-table" in style

    assert "/api/ml/compare" in script
    assert "ensureCompareLoaded" in script
    assert "redrawCompare" in script
    assert "LabCharts.drawTimeSeries" in script


# ── Исходы финального мира: отказы, переключения цены, вызов (change world-final-config) ──


def _write_meta(db: Path, **fields) -> None:
    """Метаданные ветки рядом с БД (формат ``simulator.branches``)."""

    payload = {"branch": db.stem, "channel": None, "callout_minutes_test": 0}
    payload.update(fields)
    db.with_name(f"{db.stem}.meta.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def test_branch_outcomes_cancelled_switches_and_callout(tmp_path):
    """Отказы, `unserved`, переключения цены и часы вызова — ручной расчёт (2.1)."""

    db = make_db(
        tmp_path / "a-base.db",
        trips=[
            _trip(datetime(2026, 11, 3, 0, 10), pickup_at=datetime(2026, 11, 3, 0, 18)),
            _trip(datetime(2026, 11, 3, 1, 30), pickup_at=datetime(2026, 11, 3, 1, 40)),
            _trip(datetime(2026, 11, 3, 0, 20), status="unfulfilled"),
            _trip(datetime(2026, 11, 3, 0, 30), status="cancelled"),
        ],
        metrics=[
            # Зона 0: 1.00 → 1.15 → 1.15 — одна смена тарифа.
            _metrics_row(datetime(2026, 11, 3, 0, 0), 0, requests=10, free_drivers=8),
            _metrics_row(datetime(2026, 11, 3, 0, 5), 0, requests=10, free_drivers=8),
            _metrics_row(datetime(2026, 11, 3, 0, 10), 0, requests=10, free_drivers=8),
            # Зона 1: 1.00 → 1.00 — смен нет.
            _metrics_row(datetime(2026, 11, 3, 0, 0), 1, requests=10, free_drivers=8),
            _metrics_row(datetime(2026, 11, 3, 0, 5), 1, requests=10, free_drivers=8),
        ],
    )
    # Тариф меняется в зоне 0 на втором бакете.
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "UPDATE zone_metrics_5min SET price_multiplier = 1.15"
            " WHERE zone_id = 0 AND ts >= '2026-11-03 00:05:00'"
        )
        conn.commit()
    finally:
        conn.close()
    _write_meta(db, channel="both", callout_minutes_test=180.0)

    cfg = compare_config()
    outcomes = branch_outcomes(db, cfg, cfg_path_min(cfg), compare_test_window(cfg))
    assert outcomes["trips"] == 2
    assert outcomes["unfulfilled"] == 1
    assert outcomes["cancelled"] == 1
    assert outcomes["unserved"] == 2
    assert outcomes["price_switches"] == pytest.approx(0.5)  # 1 смена / (2 зоны × 1 сутки)
    assert outcomes["callout_hours"] == pytest.approx(3.0)  # 180 мин / 60
    assert outcomes["deterred"] is None and outcomes["saved"] is None  # заполняет A-сборка


def test_branch_outcomes_old_grid_without_meta(tmp_path):
    """Старая сетка без ``meta.json``: канал и часы вызова ``null``, прочее считается (2.1)."""

    db = make_db(
        tmp_path / "c-mlp.db",
        trips=[_trip(datetime(2026, 11, 3, 2, 0), pickup_at=datetime(2026, 11, 3, 2, 5))],
        metrics=[_metrics_row(datetime(2026, 11, 3, 0, 0), 0, requests=4, free_drivers=1)],
    )
    cfg = compare_config()
    outcomes = branch_outcomes(db, cfg, cfg_path_min(cfg), compare_test_window(cfg))
    assert outcomes["callout_hours"] is None
    assert outcomes["trips"] == 1
    assert outcomes["price_switches"] == pytest.approx(0.0)  # смен нет, строки есть


def test_build_compare_relative_outcomes_and_no_deltas(tmp_path):
    """`deterred`/`saved` считаются от A и дельт к A не имеют (2.1)."""

    worlds = tmp_path / "worlds"
    a_db = make_db(
        worlds / "a-base.db",
        trips=[
            _trip(datetime(2026, 11, 3, 0, 10), pickup_at=datetime(2026, 11, 3, 0, 15)),
            _trip(datetime(2026, 11, 3, 1, 10), pickup_at=datetime(2026, 11, 3, 1, 15)),
            _trip(datetime(2026, 11, 3, 2, 10), status="unfulfilled"),
            _trip(datetime(2026, 11, 3, 3, 10), status="unfulfilled"),
        ],
    )
    c_db = make_db(
        worlds / "c-mlp.db",
        trips=[
            _trip(datetime(2026, 11, 3, 0, 10), pickup_at=datetime(2026, 11, 3, 0, 15)),
            _trip(datetime(2026, 11, 3, 1, 10), status="unfulfilled"),
        ],
    )
    _write_meta(a_db, channel="both")
    _write_meta(c_db, channel="heatmap")

    data = build_compare(compare_config(), worlds)
    by_name = {branch["name"]: branch for branch in data["branches"]}
    assert by_name["a-base"]["channel"] == "both"
    assert by_name["c-mlp"]["channel"] == "heatmap"
    assert by_name["a-base"]["outcomes"]["deterred"] == 0
    assert by_name["a-base"]["outcomes"]["saved"] == 0
    assert by_name["c-mlp"]["outcomes"]["deterred"] == 1  # A перевёз 2, C — 1
    assert by_name["c-mlp"]["outcomes"]["saved"] == 1  # у A необслужено 2, у C — 1
    for branch in data["branches"]:
        assert branch["delta"]["deterred"] is None
        assert branch["delta"]["saved"] is None
    assert by_name["b-reactive"]["channel"] is None  # ветки нет
