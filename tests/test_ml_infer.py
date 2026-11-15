"""Тесты ml/infer.py и CLI infer (change ``m3-inference``): восстановление сети
из артефакта, walk-forward по тест-эпохе, анти-лик (мутация будущих строк не
меняет прогноз), наивы тем же набором пар, перезапись версий и сводная прогона.

Реальная эпоха не трогается: все проверки — на рукописных мини-БД в ``tmp_path``
(приём ``tests/test_ml_train.py``): 3 дня × 288 бакетов, данные в зонах 0/1,
город конфига по умолчанию — 6 зон (ёмкость эмбеддинга n_zones = 6, зоны без
данных молча дают ноль примеров, как и в обучении). Гиперпараметры тестовых
конфигов малы — обучение сходится за секунды.
"""

import json
import re
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from common.config import AppConfig, MlCfg
from ml.__main__ import main
from ml.baselines import NAIVE_PERSIST_V1, NAIVE_SEASON_V1
from ml.dataset import SqliteRowSource, collect_examples
from ml.infer import (
    load_model_artifact,
    run_inference,
    summarize_run,
    walk_forward_model,
    walk_forward_naives,
)
from ml.train import model_version_for, prepare_training_data, save_model_artifacts, train_model
from storage.db import connect
from storage.repo import insert_forecasts, insert_zone_metrics, list_forecasts
from storage.schema import init_db

SIM_START = datetime(2026, 11, 2)
"""Старт мира мини-БД (дефолт engine.sim_start конфига, понедельник)."""

CITY_ZONES = tuple(range(6))
"""Зоны города конфига по умолчанию; данные паттерна есть только в 0/1."""

_FORECAST = {
    "ts": datetime(2026, 11, 4, 12, 0),
    "zone_id": 0,
    "horizon_min": 15,
    "pred_requests": 1.0,
    "pred_free_drivers": 1.0,
    "deficit_prob": 0.5,
    "model_version": "foreign-v9",
}
"""Чужая строка forecasts (для проверок «прогон не трогает чужие версии»)."""


def _ml_cfg(**overrides: object) -> MlCfg:
    """Тестовые гиперпараметры: роли 2/1/1 дня, окно 1 ч, малая сеть."""

    base = {
        "train_days": 2,
        "val_days": 1,
        "test_days": 1,
        "window_hours": 1,
        "mlp_hidden": [16, 128],
        "lstm_hidden": 8,
        "dropout": 0.0,
        "lr": 0.01,
        "batch_size": 128,
        "max_epochs": 4,
        "patience": 4,
        "seed": 42,
    }
    base.update(overrides)
    return MlCfg(**base)


def _pattern_row(ts: datetime, j: int, zone_id: int) -> dict:
    """Строка метрик с суточной закономерностью (приём ``tests/test_ml_train.py``)."""

    requests = 3 + int(round(3.0 * np.sin(2.0 * np.pi * (j % 96) / 96.0))) + zone_id
    free_drivers = 4 + (j % 3) + zone_id
    unfulfilled = 1 if requests >= 6 else 0
    return {
        "ts": ts,
        "zone_id": zone_id,
        "requests": requests,
        "completed": requests - unfulfilled,
        "unfulfilled": unfulfilled,
        "free_drivers": free_drivers,
        "busy_drivers": j % 4,
        "avg_wait_min": (j % 5) * 0.5,
        "supply_demand_ratio": free_drivers / max(requests, 1),
        "is_rain": 1 if j % 29 == 0 else 0,
        "event": "",
        "price_multiplier": 1.0,
    }


def _fill_db(db_path: Path, *, mutate_from: datetime | None = None) -> None:
    """Наполнить БД паттерном: 3 дня × 288 бакетов, зоны 0/1. Строки с
    ``ts >= mutate_from`` записываются другими значениями (мутация будущего
    для анти-лик теста: таргеты и вся глубина после X)."""

    conn = connect(str(db_path))
    init_db(conn)
    rows: list[dict] = []
    t = SIM_START
    end = SIM_START + timedelta(days=3) - timedelta(minutes=5)
    while t <= end:
        j = int((t - SIM_START).total_seconds()) // 300
        for zone_id in (0, 1):
            row = _pattern_row(t, j, zone_id)
            if mutate_from is not None and t >= mutate_from:
                row["requests"] += 5
                row["free_drivers"] += 3
                row["completed"] = row["requests"] - row["unfulfilled"]
                row["busy_drivers"] = (j + 2) % 4
                row["avg_wait_min"] = (j % 5) * 0.5 + 0.25
                row["supply_demand_ratio"] = row["free_drivers"] / max(row["requests"], 1)
            rows.append(row)
        t = t + timedelta(minutes=5)
    insert_zone_metrics(conn, rows)
    conn.close()


@pytest.fixture
def pattern_db(tmp_path: Path) -> Path:
    """Мини-эпоха с закономерностью: 3 дня, зоны 0/1."""

    path = tmp_path / "pattern.db"
    _fill_db(path)
    return path


def _test_keys(db_path: Path) -> list[tuple[int, datetime]]:
    """Ключи test-роли сборщика датасета (те же границы, что у walk-forward)."""

    source = SqliteRowSource(db_path)
    return collect_examples(
        source,
        zones=CITY_ZONES,
        start=SIM_START,
        train_days=2,
        val_days=1,
        test_days=1,
        window_hours=1,
    )["test"].keys


def _train_and_save(artifacts_root: Path, db_path: Path, arch: str, *, day: int = 1) -> Path:
    """Обучить архитектуру на мини-эпохе и записать каталог артефакта с датой
    ``2026-01-<day>`` (управляемая «свежесть» для выбора каталога)."""

    ml_cfg = _ml_cfg()
    data = prepare_training_data(db_path, ml_cfg, zones=list(CITY_ZONES), start=SIM_START)
    result = train_model(arch, data, ml_cfg)
    version = model_version_for(arch, today=date(2026, 1, day))
    model_dir = artifacts_root / version
    save_model_artifacts(
        model_dir,
        model_version=version,
        arch=arch,
        model=result.model,
        data=data,
        result=result,
        ml_cfg=ml_cfg,
    )
    return model_dir


@pytest.fixture
def artifacts_root(pattern_db: Path, tmp_path: Path) -> Path:
    """Корень артефактов с каталогами обеих архитектур."""

    root = tmp_path / "artifacts"
    root.mkdir()
    _train_and_save(root, pattern_db, "mlp")
    _train_and_save(root, pattern_db, "lstm")
    return root


def _run(db_path: Path, artifacts_root: Path, models: str = "all") -> None:
    """Прогон без сводной: как его делает CLI (источник и соединение на прогон)."""

    cfg = AppConfig()
    source = SqliteRowSource(db_path)
    conn = connect(str(db_path))
    try:
        run_inference(source, conn, cfg=cfg, models=models, artifacts_root=artifacts_root)
    finally:
        conn.close()


def _stored_rows(db_path: Path, model_version: str) -> list[tuple]:
    """Строки версии из БД кортежами (для сравнения прогонов между собой)."""

    conn = connect(str(db_path))
    try:
        return [tuple(row) for row in list_forecasts(conn, model_version=model_version)]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 2.1 Восстановление сети из артефакта
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["mlp", "lstm"])
def test_load_model_artifact_roundtrip(pattern_db: Path, tmp_path: Path, arch: str) -> None:
    """Мини-артефакт, записанный save_model_artifacts, загружается: прогнозы
    восстановленной сети побитово равны прогнозам сети до сохранения."""
    ml_cfg = _ml_cfg()
    data = prepare_training_data(pattern_db, ml_cfg, zones=list(CITY_ZONES), start=SIM_START)
    result = train_model(arch, data, ml_cfg)
    model_dir = tmp_path / "artifacts" / model_version_for(arch, today=date(2026, 1, 1))
    save_model_artifacts(
        model_dir,
        model_version=model_dir.name,
        arch=arch,
        model=result.model,
        data=data,
        result=result,
        ml_cfg=ml_cfg,
    )

    artifact = load_model_artifact(model_dir, n_zones=len(CITY_ZONES))

    assert artifact.model_version == model_dir.name
    assert artifact.arch == arch
    assert artifact.ml_cfg == ml_cfg
    idx = np.arange(8)
    windows = torch.from_numpy(data.windows["val"][idx].astype(np.float32))
    zones = torch.from_numpy(data.zones["val"][idx].astype(np.int64))
    saved, trained = artifact.model(windows, zones), result.model(windows, zones)
    assert torch.equal(saved.pred_requests, trained.pred_requests)
    assert torch.equal(saved.pred_free_drivers, trained.pred_free_drivers)
    assert torch.equal(saved.deficit_logits, trained.deficit_logits)
    assert torch.equal(saved.deficit_prob, trained.deficit_prob)


def test_load_uses_snapshot_not_live_config(pattern_db: Path, tmp_path: Path) -> None:
    """Дрейф живого конфига (окно 2 ч вместо снимка 1 ч) не влияет на загрузку:
    сеть построена по снимку и принимает окна снимка, а не дрейфнувшего окна."""
    model_dir = _train_and_save(tmp_path / "artifacts", pattern_db, "mlp")  # окно 1 ч

    artifact = load_model_artifact(model_dir, n_zones=len(CITY_ZONES))

    assert artifact.ml_cfg.window_hours == 1  # снимок, а не живой конфиг с окном 2
    windows_snapshot = torch.zeros(2, 12, 17)  # 12 шагов = окно снимка (1 ч)
    zones = torch.zeros(2, dtype=torch.int64)
    assert artifact.model(windows_snapshot, zones).deficit_prob.shape == (2, 2)
    with pytest.raises(ValueError, match="окно"):
        artifact.model(torch.zeros(2, 24, 17), zones)  # окно дрейфнувшего конфига


# ---------------------------------------------------------------------------
# 2.2 Walk-forward одной сети
# ---------------------------------------------------------------------------


def test_walk_forward_rows(pattern_db: Path, tmp_path: Path) -> None:
    """Строки ровно по две на годную пару test-роли; последние бакеты без пары
    ts+30 пропущены; deficit_prob ∈ [0, 1]."""
    model_dir = _train_and_save(tmp_path / "artifacts", pattern_db, "mlp")
    artifact = load_model_artifact(model_dir, n_zones=len(CITY_ZONES))
    source = SqliteRowSource(pattern_db)

    rows = walk_forward_model(artifact, source, zones=list(CITY_ZONES), start=SIM_START)

    keys = _test_keys(pattern_db)
    assert len(rows) == 2 * len(keys)
    by_pair: dict[tuple[datetime, int], list[int]] = {}
    for row in rows:
        by_pair.setdefault((row["ts"], row["zone_id"]), []).append(row["horizon_min"])
    assert set(by_pair) == {(ts, zone) for zone, ts in keys}
    assert all(sorted(horizons) == [15, 30] for horizons in by_pair.values())
    assert all(0.0 <= row["deficit_prob"] <= 1.0 for row in rows)
    # Граница тест-эпохи: последний бакет без целевого ts+30 не даёт строки.
    last_bucket = SIM_START + timedelta(days=3) - timedelta(minutes=5)
    assert max(ts for _, ts in keys) == last_bucket - timedelta(minutes=30)


def test_walk_forward_physical_values(pattern_db: Path, tmp_path: Path) -> None:
    """Денормализация по образцу val_mae_requests: значения строк равны
    inverse_transform нормализованных прогнозов сети; величины физические
    (scaler таргета нетривиален, прогнозы сопоставимы с фактом метрик)."""
    model_dir = _train_and_save(tmp_path / "artifacts", pattern_db, "mlp")
    artifact = load_model_artifact(model_dir, n_zones=len(CITY_ZONES))
    source = SqliteRowSource(pattern_db)
    rows = walk_forward_model(artifact, source, zones=list(CITY_ZONES), start=SIM_START)

    assert float(artifact.scalers.requests.mean_[0]) > 1.0  # денормализация нетривиальна
    examples = collect_examples(
        SqliteRowSource(pattern_db),
        zones=CITY_ZONES,
        start=SIM_START,
        train_days=2,
        val_days=1,
        test_days=1,
        window_hours=1,
    )["test"]
    windows = artifact.scalers.transform_windows(examples.windows[:32])
    zones_t = torch.from_numpy(np.array([zone for zone, _ in examples.keys[:32]], dtype=np.int64))
    with torch.no_grad():
        output = artifact.model(torch.from_numpy(windows.astype(np.float32)), zones_t)
    expected = artifact.scalers.requests.inverse_transform(
        output.pred_requests.numpy().reshape(-1, 1)
    ).reshape(output.pred_requests.numpy().shape)
    for i in range(32):
        assert rows[2 * i]["pred_requests"] == pytest.approx(float(expected[i, 0]), abs=1e-5)
        assert rows[2 * i + 1]["pred_requests"] == pytest.approx(float(expected[i, 1]), abs=1e-5)
        assert abs(rows[2 * i]["pred_requests"]) < 100.0  # физические заказы, не z-баллы


def test_walk_forward_skips_holes(pattern_db: Path, tmp_path: Path) -> None:
    """Дыра в истории test-дня: пары с окном, накрывающим дыру, пропущены без
    исключений — набор строк совпадает с ключами сборщика на той же БД."""
    hole_db = tmp_path / "hole.db"
    shutil.copy(pattern_db, hole_db)
    hole_ts = SIM_START + timedelta(days=2, hours=1)
    conn = connect(str(hole_db))
    conn.execute(
        "DELETE FROM zone_metrics_5min WHERE ts = ? AND zone_id = 0",
        (hole_ts.strftime("%Y-%m-%d %H:%M:%S"),),
    )
    conn.close()

    model_dir = _train_and_save(tmp_path / "artifacts", pattern_db, "mlp")
    artifact = load_model_artifact(model_dir, n_zones=len(CITY_ZONES))
    rows = walk_forward_model(
        artifact, SqliteRowSource(hole_db), zones=list(CITY_ZONES), start=SIM_START
    )

    keys = _test_keys(hole_db)
    assert (hole_ts, 0) not in keys  # окно этой пары накрывает дыру
    assert len(rows) == 2 * len(keys)
    assert {(row["ts"], row["zone_id"]) for row in rows} == {(ts, z) for z, ts in keys}


# ---------------------------------------------------------------------------
# 2.3 Анти-лик: мутация будущего не меняет прогноз
# ---------------------------------------------------------------------------


def test_no_future_leak(tmp_path: Path) -> None:
    """Два источника, идентичные до X и различающиеся всеми строками после X
    (включая целевые бакеты), дают одинаковые прогнозы сети на все ts ≤ X−30."""
    base_db = tmp_path / "base.db"
    mutated_db = tmp_path / "mutated.db"
    mutation_at = SIM_START + timedelta(days=2, hours=3)  # X внутри test-дня
    _fill_db(base_db)
    _fill_db(mutated_db, mutate_from=mutation_at)

    model_dir = _train_and_save(tmp_path / "artifacts", base_db, "mlp")
    artifact = load_model_artifact(model_dir, n_zones=len(CITY_ZONES))
    rows_base = walk_forward_model(
        artifact, SqliteRowSource(base_db), zones=list(CITY_ZONES), start=SIM_START
    )
    rows_mutated = walk_forward_model(
        artifact, SqliteRowSource(mutated_db), zones=list(CITY_ZONES), start=SIM_START
    )

    def value_map(rows: list[dict]) -> dict:
        return {
            (r["ts"], r["zone_id"], r["horizon_min"]): (
                r["pred_requests"],
                r["pred_free_drivers"],
                r["deficit_prob"],
            )
            for r in rows
            if r["ts"] <= mutation_at - timedelta(minutes=30)
        }

    base_map, mutated_map = value_map(rows_base), value_map(rows_mutated)
    assert len(base_map) > 0
    assert base_map.keys() == mutated_map.keys()
    assert base_map == mutated_map  # побитово: одинаковые входы — одинаковый прогноз
    # Санити: источники действительно различаются после X.
    source_base = SqliteRowSource(base_db)
    source_mutated = SqliteRowSource(mutated_db)
    assert (
        source_base.metrics_at(0, mutation_at).requests
        != source_mutated.metrics_at(0, mutation_at).requests
    )


# ---------------------------------------------------------------------------
# 2.4 Наивы тем же проходом
# ---------------------------------------------------------------------------


def test_naives_same_pairs(pattern_db: Path) -> None:
    """Наборы (ts, zone_id) строк наивов совпадают с наборами сетей: на полной
    мини-эпохе дыр нет, у сезонного наива вчерашний слот есть всюду."""
    source = SqliteRowSource(pattern_db)
    examples = collect_examples(
        source,
        zones=CITY_ZONES,
        start=SIM_START,
        train_days=2,
        val_days=1,
        test_days=1,
        window_hours=1,
    )["test"]

    naive_rows = walk_forward_naives(source, examples)

    network_pairs = {(ts, zone) for zone, ts in examples.keys}
    assert len(examples.keys) > 0
    for version in (NAIVE_PERSIST_V1, NAIVE_SEASON_V1):
        pairs = {(row["ts"], row["zone_id"]) for row in naive_rows[version]}
        assert pairs == network_pairs
        assert len(naive_rows[version]) == 2 * len(network_pairs)


def test_naive_season_silently_skips_missing_yesterday(pattern_db: Path, tmp_path: Path) -> None:
    """Пара без вчерашнего слота: у сезонного наива нет строки и нет исключения,
    у persist и сети пара на месте (допустимое расхождение спеки)."""
    hole_db = tmp_path / "no_yesterday.db"
    shutil.copy(pattern_db, hole_db)
    target_ts = SIM_START + timedelta(days=2, hours=6)  # пара в test-дне
    yesterday = target_ts - timedelta(hours=24)
    conn = connect(str(hole_db))
    conn.execute(
        "DELETE FROM zone_metrics_5min WHERE ts = ? AND zone_id = 0",
        (yesterday.strftime("%Y-%m-%d %H:%M:%S"),),
    )
    conn.close()

    source = SqliteRowSource(hole_db)
    examples = collect_examples(
        source,
        zones=CITY_ZONES,
        start=SIM_START,
        train_days=2,
        val_days=1,
        test_days=1,
        window_hours=1,
    )["test"]

    naive_rows = walk_forward_naives(source, examples)

    persist_pairs = {(row["ts"], row["zone_id"]) for row in naive_rows[NAIVE_PERSIST_V1]}
    season_pairs = {(row["ts"], row["zone_id"]) for row in naive_rows[NAIVE_SEASON_V1]}
    assert (target_ts, 0) in persist_pairs  # слот ts есть
    assert (target_ts, 0) not in season_pairs  # вчерашнего нет — молчаливый пропуск
    assert (target_ts, 0) in {(ts, zone) for zone, ts in examples.keys}  # сеть пару строит


# ---------------------------------------------------------------------------
# 3.1/3.3 Прогон версий
# ---------------------------------------------------------------------------


def test_run_all_writes_four_versions(pattern_db: Path, artifacts_root: Path) -> None:
    """Прогон «all» на эпохе с артефактами обеих архитектур пишет четыре версии
    и возвращает счётчики строк (все — по две строки на пару test-роли)."""
    cfg = AppConfig()

    source = SqliteRowSource(pattern_db)
    conn = connect(str(pattern_db))
    try:
        versions = run_inference(source, conn, cfg=cfg, models="all", artifacts_root=artifacts_root)
    finally:
        conn.close()

    expected_pairs = len(_test_keys(pattern_db))
    assert [version.model_version for version in versions] == [
        NAIVE_PERSIST_V1,
        NAIVE_SEASON_V1,
        "mlp-v1-2026-01-01",
        "lstm-v1-2026-01-01",
    ]
    assert all(version.rows == 2 * expected_pairs for version in versions)


def test_run_is_idempotent_and_scoped(pattern_db: Path, artifacts_root: Path) -> None:
    """Повторный прогон даёт те же строки и числа без удвоения; прогон
    подмножества семейств не трогает версии вне прогона."""
    conn = connect(str(pattern_db))
    source = SqliteRowSource(pattern_db)
    try:
        run_inference(source, conn, cfg=AppConfig(), models="all", artifacts_root=artifacts_root)
        first = {
            version: _stored_rows(pattern_db, version)
            for version in (NAIVE_PERSIST_V1, NAIVE_SEASON_V1, "mlp-v1-2026-01-01")
        }
        run_inference(source, conn, cfg=AppConfig(), models="all", artifacts_root=artifacts_root)
        second = {
            version: _stored_rows(pattern_db, version)
            for version in (NAIVE_PERSIST_V1, NAIVE_SEASON_V1, "mlp-v1-2026-01-01")
        }
        assert first == second

        run_inference(source, conn, cfg=AppConfig(), models="mlp", artifacts_root=artifacts_root)
        assert _stored_rows(pattern_db, NAIVE_PERSIST_V1) == first[NAIVE_PERSIST_V1]
        assert _stored_rows(pattern_db, NAIVE_SEASON_V1) == first[NAIVE_SEASON_V1]
        assert len(_stored_rows(pattern_db, "lstm-v1-2026-01-01")) > 0  # lstm не в прогоне — цел
    finally:
        conn.close()


def test_run_colocates_four_versions(pattern_db: Path, artifacts_root: Path) -> None:
    """Сценарий «Четыре версии рядом»: на каждом ключе тест-эпохи ровно четыре
    строки с разными model_version."""
    _run(pattern_db, artifacts_root)

    conn = connect(str(pattern_db))
    try:
        total = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        keys = conn.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM forecasts GROUP BY ts, zone_id, horizon_min)"
        ).fetchone()[0]
        complete_keys = conn.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM forecasts GROUP BY ts, zone_id, horizon_min"
            " HAVING COUNT(DISTINCT model_version) = 4)"
        ).fetchone()[0]
    finally:
        conn.close()
    assert keys > 0
    assert complete_keys == keys
    assert total == 4 * keys


def test_run_picks_latest_artifact(pattern_db: Path, tmp_path: Path) -> None:
    """Несколько каталогов семейства: выбирается лексический максимум (свежий)."""
    root = tmp_path / "artifacts"
    root.mkdir()
    _train_and_save(root, pattern_db, "mlp", day=1)
    latest = _train_and_save(root, pattern_db, "mlp", day=2)

    _run(pattern_db, root, models="mlp")

    versions = {row["model_version"] for row in _stored_rows_as_dicts(pattern_db)}
    assert versions == {latest.name}


def _stored_rows_as_dicts(db_path: Path) -> list[dict]:
    conn = connect(str(db_path))
    try:
        return [dict(row) for row in list_forecasts(conn)]
    finally:
        conn.close()


def test_run_refuses_mismatched_snapshots(pattern_db: Path, tmp_path: Path) -> None:
    """Сети с расходящимися снимками ролей/окна — отказ до записи: наборы пар
    разошлись бы и сравнение версий стало бы нечестным."""
    root = tmp_path / "artifacts"
    root.mkdir()
    _train_and_save(root, pattern_db, "mlp")  # окно 1 ч
    lstm_cfg = _ml_cfg(window_hours=2)  # дрейфнувший снимок: окно 2 ч
    data = prepare_training_data(pattern_db, lstm_cfg, zones=list(CITY_ZONES), start=SIM_START)
    result = train_model("lstm", data, lstm_cfg)
    version = model_version_for("lstm", today=date(2026, 1, 1))
    save_model_artifacts(
        root / version,
        model_version=version,
        arch="lstm",
        model=result.model,
        data=data,
        result=result,
        ml_cfg=lstm_cfg,
    )

    source = SqliteRowSource(pattern_db)
    conn = connect(str(pattern_db))
    try:
        with pytest.raises(ValueError, match="снимки"):
            run_inference(source, conn, cfg=AppConfig(), models="all", artifacts_root=root)
        assert conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 0
    finally:
        conn.close()


def test_run_naive_only_uses_live_roles(pattern_db: Path, tmp_path: Path) -> None:
    """Наив-прогон без сетей берёт границы ролей из живого ml-конфига и пишет
    только две наивные версии."""
    cfg = AppConfig(ml=_ml_cfg())

    source = SqliteRowSource(pattern_db)
    conn = connect(str(pattern_db))
    try:
        versions = run_inference(
            source,
            conn,
            cfg=cfg,
            models="naive",
            artifacts_root=tmp_path / "artifacts",
        )
    finally:
        conn.close()

    assert [version.model_version for version in versions] == [
        NAIVE_PERSIST_V1,
        NAIVE_SEASON_V1,
    ]
    assert all(version.rows == 2 * len(_test_keys(pattern_db)) for version in versions)


# ---------------------------------------------------------------------------
# 3.2 Сводная прогона
# ---------------------------------------------------------------------------


def test_summarize_run_json_and_table(pattern_db: Path, artifacts_root: Path) -> None:
    """JSON-сводная существует, читается и содержит раздел каждой версии со всеми
    тремя группами метрик; печатная таблица — шапка из 7 колонок и строка на версию."""
    cfg = AppConfig()
    source = SqliteRowSource(pattern_db)
    conn = connect(str(pattern_db))
    try:
        versions = run_inference(source, conn, cfg=cfg, models="all", artifacts_root=artifacts_root)
        summary_path, table = summarize_run(conn, source, versions, artifacts_root=artifacts_root)
    finally:
        conn.close()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["date"] == date.today().isoformat()
    assert [summary["model_version"] for summary in payload["summaries"]] == [
        version.model_version for version in versions
    ]
    for summary in payload["summaries"]:
        assert set(summary) == {"model_version", "regression", "deficit", "calibration"}
        assert set(summary["regression"]) == {"pred_requests", "pred_free_drivers"}
        for target in summary["regression"].values():
            assert set(target) == {"15", "30"}  # ключи горизонтов JSON — строки
            assert all(set(metrics) == {"mae", "rmse"} for metrics in target.values())
        assert set(summary["deficit"]) == {"15", "30"}
        for metrics in summary["deficit"].values():
            assert set(metrics) == {"precision", "recall", "f1"}
        assert set(summary["calibration"]) == {"15", "30"}

    lines = table.splitlines()
    assert len(lines) == 1 + len(versions)
    for column in (
        "MAE_req15",
        "MAE_req30",
        "MAE_fr15",
        "MAE_fr30",
        "RMSE_req15",
        "F1_def15",
        "F1_def30",
    ):
        assert column in lines[0]
    for version in versions:
        assert version.model_version in table


# ---------------------------------------------------------------------------
# 4.1 CLI infer
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, db_path: Path, name: str = "config.yaml") -> Path:
    """YAML-конфиг CLI: секция ml с мини-гиперпараметрами и путём мини-БД."""

    ml = {
        "db": str(db_path),
        "train_days": 2,
        "val_days": 1,
        "test_days": 1,
        "window_hours": 1,
        "mlp_hidden": [16, 128],
        "lstm_hidden": 8,
        "dropout": 0.0,
        "lr": 0.01,
        "batch_size": 128,
        "max_epochs": 4,
        "patience": 4,
        "seed": 42,
    }
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"ml": ml}), encoding="utf-8")
    return path


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch, pattern_db: Path) -> Path:
    """cwd = tmp_path (корень артефактов — относительный) и конфиг с мини-БД."""

    monkeypatch.chdir(tmp_path)
    return _write_config(tmp_path, pattern_db)


def test_cli_infer_smoke(cli_env: Path, pattern_db: Path, capsys) -> None:
    """Smoke: train создаёт артефакты за сегодня, infer — exit 0, строки всех
    четырёх версий в БД, json-сводная и печатная таблица на месте."""
    assert main(["train", "--config", str(cli_env)]) == 0

    rc = main(["infer", "--config", str(cli_env)])
    assert rc == 0
    captured = capsys.readouterr()

    today = date.today().isoformat()
    walk_lines = re.findall(r"walk-forward: model=(\S+) rows=(\d+)", captured.out)
    assert [name for name, _ in walk_lines] == [
        "naive-persist-v1",
        "naive-season-v1",
        f"mlp-v1-{today}",
        f"lstm-v1-{today}",
    ]
    assert len({rows for _, rows in walk_lines}) == 1  # все версии на одном наборе пар

    summary_line = re.search(r"summary: (\S+)", captured.out)
    assert summary_line
    summary_path = Path(summary_line.group(1))
    assert summary_path.is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(payload["summaries"]) == 4
    assert len(captured.out.strip().splitlines()[-1].split()) == 8  # таблица: имя + 7 колонок

    rows = _stored_rows_as_dicts(pattern_db)
    assert {row["model_version"] for row in rows} == {
        "naive-persist-v1",
        "naive-season-v1",
        f"mlp-v1-{today}",
        f"lstm-v1-{today}",
    }


def test_cli_infer_missing_db(cli_env: Path, tmp_path: Path, capsys) -> None:
    """Несуществующая --db — exit 2 и сообщение о недоступной БД."""
    rc = main(["infer", "--config", str(cli_env), "--db", str(tmp_path / "missing.db")])
    assert rc == 2
    captured = capsys.readouterr()
    assert "не найдена" in captured.err
    assert captured.out == ""


def test_cli_infer_no_artifacts(cli_env: Path, pattern_db: Path, capsys) -> None:
    """--models mlp без каталогов mlp-v1-* — exit 2, forecasts не тронута."""
    conn = connect(str(pattern_db))
    insert_forecasts(conn, [{**dict(_FORECAST), "model_version": "foreign-v9"}])
    conn.close()

    rc = main(["infer", "--config", str(cli_env), "--models", "mlp"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "mlp-v1-*" in captured.err

    rows = _stored_rows_as_dicts(pattern_db)
    assert [row["model_version"] for row in rows] == ["foreign-v9"]


def test_cli_infer_nonempty_legacy_forecasts(cli_env: Path, pattern_db: Path, capsys) -> None:
    """Непустая таблица forecasts старой схемы — exit 2 с сообщением о миграции."""
    conn = connect(str(pattern_db))
    conn.execute("DROP TABLE forecasts")
    conn.execute(
        "CREATE TABLE forecasts ("
        " ts TEXT NOT NULL, zone_id INTEGER NOT NULL, horizon_min INTEGER NOT NULL,"
        " pred_requests REAL NOT NULL, pred_free_drivers REAL NOT NULL,"
        " deficit_prob REAL NOT NULL, model_version TEXT NOT NULL,"
        " PRIMARY KEY (ts, zone_id, horizon_min))"
    )
    insert_forecasts(conn, [dict(_FORECAST)])
    conn.close()

    rc = main(["infer", "--config", str(cli_env), "--models", "naive"])
    assert rc == 2
    assert "очистите" in capsys.readouterr().err
