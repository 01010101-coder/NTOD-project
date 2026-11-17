"""Тесты ml/planning.py и подкоманд ``plan-train``/``plan-eval``
(change ``ml-planning-forecast``): метки окна цели, совпадение окон с M3,
сеть и обучение, артефакты, календарь, персистентность, метрики, CLI и
прогнозист живого контура.

Реальная эпоха (``data/m3_epoch.db``) не трогается: все проверки — на
рукописных мини-БД в ``tmp_path`` (приём тестов M3). Гиперпараметры —
малые (mlp_hidden [16, 128], dropout 0.0), чтобы обучение сходилось за секунды.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from common.config import AppConfig, MlCfg
from common.timeutil import add_minutes
from ml.__main__ import main
from ml.dataset import (
    N_FEATURES,
    SqliteRowSource,
    bucket_index,
    collect_examples,
)
from ml.planning import (
    CalendarBaseline,
    PlanningPredictor,
    build_planning_model,
    collect_planning_examples,
    day_type_index,
    evaluate_planning,
    half_hour_index,
    load_planning_artifact,
    persistence_score,
    planning_pos_weight,
    planning_targets,
    predict_planning_windows,
    prepare_planning_data,
    render_planning_table,
    save_planning_artifacts,
    score_metrics,
    train_planning,
)
from ml.train import METRICS_FILE, SCALERS_FILE, WEIGHTS_FILE, model_version_for
from simulator.__main__ import SqliteTripWriter
from simulator.engine import Engine
from simulator.policy import LiveRowSource
from storage.db import connect
from storage.repo import insert_zone_metrics
from storage.schema import init_db

SIM_START = datetime(2026, 11, 2)
"""Старт мира мини-БД (дефолт engine.sim_start, понедельник)."""

ZONES = (0, 1)
"""Зоны мини-эпохи; ёмкость эмбеддинга n_zones = 2."""

ARTIFACT_FILES = frozenset({WEIGHTS_FILE, SCALERS_FILE, METRICS_FILE})
"""Набор файлов каталога артефакта планирования."""


# ---------------------------------------------------------------------------
# Общие фикстуры и помощники
# ---------------------------------------------------------------------------


def _ml_cfg(**overrides: object) -> MlCfg:
    """Тестовые гиперпараметры: малая сеть, окно 1 ч, несколько эпох."""

    base: dict[str, object] = {
        "train_days": 2,
        "val_days": 1,
        "test_days": 1,
        "window_hours": 1,
        "mlp_hidden": [16, 128],
        "lstm_hidden": 8,
        "dropout": 0.0,
        "lr": 0.01,
        "batch_size": 64,
        "max_epochs": 3,
        "patience": 4,
        "seed": 42,
    }
    base.update(overrides)
    return MlCfg(**base)


def _metric_row(
    ts: datetime,
    zone_id: int,
    *,
    requests: int = 5,
    free_drivers: int = 4,
    unfulfilled: int = 0,
) -> dict:
    """Строка ``zone_metrics_5min`` с минимальным набором значимых полей."""

    completed = max(requests - unfulfilled, 0)
    return {
        "ts": ts,
        "zone_id": zone_id,
        "requests": requests,
        "completed": completed,
        "unfulfilled": unfulfilled,
        "free_drivers": free_drivers,
        "busy_drivers": 2,
        "avg_wait_min": 3.0,
        "supply_demand_ratio": free_drivers / max(requests, 1),
        "is_rain": 0,
        "event": "",
        "price_multiplier": 1.0,
    }


def _write_db(path: Path, rows: list[dict]) -> Path:
    """Записать строки метрик в новую мини-БД с полной схемой."""

    conn = connect(str(path))
    init_db(conn)
    insert_zone_metrics(conn, rows)
    conn.close()
    return path


def _pattern_row(ts: datetime, zone_id: int, bucket: int) -> dict:
    """Строка эпохи с закономерностью: вред 1 ч каждые 4 ч, сдвиг по зонам."""

    j = bucket + 7 * zone_id
    requests = 6 + (j % 5)
    unfulfilled = 1 if j % 48 < 12 else 0
    return _metric_row(
        ts,
        zone_id,
        requests=requests,
        free_drivers=2 + (j % 4),
        unfulfilled=unfulfilled,
    )


@pytest.fixture
def pattern_db(tmp_path: Path) -> Path:
    """Мини-эпоха 5 дней × 288 бакетов, зоны 0/1 — с обоими классами вреда."""

    path = tmp_path / "pattern.db"
    rows: list[dict] = []
    for bucket in range(5 * 288):
        ts = add_minutes(SIM_START, bucket * 5)
        for zone_id in ZONES:
            rows.append(_pattern_row(ts, zone_id, bucket))
    return _write_db(path, rows)


def _train_and_save(
    db_path: Path,
    ml_cfg: MlCfg,
    artifacts_root: Path,
    family: str,
    *,
    zones: tuple[int, ...] = ZONES,
) -> tuple[Path, object, object]:
    """Подготовить данные, обучить семейство и записать артефакт версии."""

    data = prepare_planning_data(db_path, ml_cfg, zones=list(zones), start=SIM_START)
    result = train_planning(family, data, ml_cfg)
    version = model_version_for(family)
    model_dir = artifacts_root / version
    save_planning_artifacts(
        model_dir,
        model_version=version,
        family=family,
        model=result.model,
        data=data,
        result=result,
        ml_cfg=ml_cfg,
    )
    return model_dir, data, result


@pytest.fixture
def trained_pair(tmp_path: Path, pattern_db: Path) -> tuple[Path, MlCfg, Path]:
    """Артефакты обоих семейств на мини-эпохе: корень, конфиг, БД."""

    ml_cfg = _ml_cfg()
    root = tmp_path / "artifacts" / "planning"
    for family in ("mlp", "lstm"):
        _train_and_save(pattern_db, ml_cfg, root, family)
    return root, ml_cfg, pattern_db


# ---------------------------------------------------------------------------
# 2.1 Датасет планирования
# ---------------------------------------------------------------------------


def test_planning_labels_by_target_window(tmp_path: Path) -> None:
    """Сценарий «Метка по окну цели»: вред только в окне H=60 → harm_60=1, harm_120=0."""

    noon = datetime(2026, 11, 2, 12, 0)
    rows = [_metric_row(add_minutes(SIM_START, k * 5), 0, unfulfilled=0) for k in range(288)]
    harm_ts = add_minutes(noon, 60)
    rows[bucket_index(harm_ts) - bucket_index(SIM_START)] = _metric_row(harm_ts, 0, unfulfilled=2)
    source = SqliteRowSource(_write_db(tmp_path / "labels.db", rows))

    targets = planning_targets(source, 0, noon, horizons_min=(60, 120), window_min=30)

    assert targets is not None
    assert targets.labels == (1, 0)
    assert targets.harm_sums == (2.0, 0.0)


def test_planning_ratio_without_harm_gives_no_label(tmp_path: Path) -> None:
    """Сценарий «Соотношение без вреда не даёт метку»: ratio < 0.5, unfulfilled = 0 → 0."""

    noon = datetime(2026, 11, 2, 12, 0)
    rows = [
        _metric_row(add_minutes(SIM_START, k * 5), 0, requests=10, free_drivers=0)
        for k in range(288)
    ]
    source = SqliteRowSource(_write_db(tmp_path / "ratio.db", rows))

    targets = planning_targets(source, 0, noon, horizons_min=(60, 120), window_min=30)

    assert targets is not None
    assert targets.labels == (0, 0)
    assert targets.harm_sums == (0.0, 0.0)


def test_planning_targets_edge_of_epoch(tmp_path: Path) -> None:
    """Сценарий «Край эпохи»: нет последнего бакета окна H=120 → примера нет."""

    rows = [_metric_row(add_minutes(SIM_START, k * 5), 0) for k in range(288)]
    source = SqliteRowSource(_write_db(tmp_path / "edge.db", rows))
    last_ts = add_minutes(SIM_START, 287 * 5)

    edge = add_minutes(last_ts, -140)  # последний бакет окна: edge + 145 = last + 5
    assert planning_targets(source, 0, edge, horizons_min=(120,), window_min=30) is None

    inside = add_minutes(last_ts, -145)  # последний бакет окна ровно есть
    targets = planning_targets(source, 0, inside, horizons_min=(120,), window_min=30)
    assert targets is not None and targets.labels == (0,)


def test_collect_planning_windows_match_m3(pattern_db: Path) -> None:
    """Окна планирования совпадают с collect_examples M3 для тех же (зона, ts)."""

    ml_cfg = _ml_cfg(test_days=3)
    source = SqliteRowSource(pattern_db)
    planning = collect_planning_examples(
        source,
        zones=list(ZONES),
        start=SIM_START,
        train_days=ml_cfg.train_days,
        val_days=ml_cfg.val_days,
        test_days=ml_cfg.test_days,
        window_hours=ml_cfg.window_hours,
        horizons_min=ml_cfg.planning.horizons_min,
        window_min=ml_cfg.planning.window_min,
    )
    m3 = collect_examples(
        source,
        zones=list(ZONES),
        start=SIM_START,
        train_days=ml_cfg.train_days,
        val_days=ml_cfg.val_days,
        test_days=ml_cfg.test_days,
        window_hours=ml_cfg.window_hours,
    )

    checked = 0
    for role in ("train", "test"):
        planning_index = {key: i for i, key in enumerate(planning[role].keys)}
        for j, key in enumerate(m3[role].keys):
            i = planning_index.get(key)
            if i is None:
                continue
            assert np.array_equal(planning[role].windows[i], m3[role].windows[j])
            checked += 1
    assert checked > 0, "пересечение наборов пар пусто — сравнение не выполнено"
    # Планирование требует более длинного окна цели: хвост последнего дня
    # тест-роли у него короче, чем у M3.
    assert len(planning["test"].keys) < len(m3["test"].keys)


def test_collect_planning_roles_and_shapes(pattern_db: Path) -> None:
    """Роли непусты, формы контракта, метки 0/1, суммы вреда неотрицательны."""

    ml_cfg = _ml_cfg()
    splits = collect_planning_examples(
        SqliteRowSource(pattern_db),
        zones=list(ZONES),
        start=SIM_START,
        train_days=ml_cfg.train_days,
        val_days=ml_cfg.val_days,
        test_days=ml_cfg.test_days,
        window_hours=ml_cfg.window_hours,
        horizons_min=ml_cfg.planning.horizons_min,
        window_min=ml_cfg.planning.window_min,
    )
    for role in ("train", "val", "test"):
        examples = splits[role]
        assert examples.windows.shape[1:] == (12, N_FEATURES)
        assert examples.labels.shape == (examples.windows.shape[0], 2)
        assert examples.harm_sums.shape == examples.labels.shape
        assert set(np.unique(examples.labels)) == {0.0, 1.0}
        assert float(examples.harm_sums.min()) >= 0.0


# ---------------------------------------------------------------------------
# 3.1 Сеть, обучение и артефакты
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["mlp", "lstm"])
def test_planning_net_shapes(arch: str) -> None:
    """Сценарий «Формы выхода»: вероятности (n, 2) и в [0, 1] для обеих архитектур."""

    ml_cfg = _ml_cfg()
    model = build_planning_model(arch, ml_cfg, n_zones=2)
    windows = torch.from_numpy(np.random.default_rng(0).normal(size=(7, 12, N_FEATURES)))
    zones = torch.zeros(7, dtype=torch.int64)
    output = model(windows.float(), zones)

    assert output.logits.shape == (7, 2)
    assert output.prob.shape == (7, 2)
    assert torch.all(output.prob >= 0.0) and torch.all(output.prob <= 1.0)
    assert torch.allclose(output.prob, torch.sigmoid(output.logits))


def test_planning_pos_weight_per_horizon() -> None:
    """pos_weight — отношение негативов к позитивам отдельно по горизонтам."""

    labels = np.array([[1, 0], [0, 0], [1, 1], [1, 0]], dtype=float)
    weights = planning_pos_weight(labels, [60, 120])
    assert weights[0] == pytest.approx((1 - 0.75) / 0.75)
    assert weights[1] == pytest.approx((1 - 0.25) / 0.25)


def test_planning_pos_weight_degenerate_raises() -> None:
    """Вырожденная доля вреда горизонта — ValueError с номером горизонта."""

    labels = np.zeros((4, 2), dtype=float)
    with pytest.raises(ValueError, match="горизонта 60"):
        planning_pos_weight(labels, [60, 120])


def test_save_planning_artifacts_roundtrip(pattern_db: Path, tmp_path: Path) -> None:
    """Три файла каталога, восстановление сети, метрики val/test в metrics.json."""

    ml_cfg = _ml_cfg()
    root = tmp_path / "artifacts" / "planning"
    model_dir, data, result = _train_and_save(pattern_db, ml_cfg, root, "mlp")

    assert {entry.name for entry in model_dir.iterdir()} == set(ARTIFACT_FILES)
    artifact = load_planning_artifact(model_dir, n_zones=max(ZONES) + 1)
    assert artifact.model_version == model_dir.name
    assert artifact.arch == "mlp"
    assert artifact.ml_cfg.planning.horizons_min == [60, 120]

    # Артефакт на сырых окнах повторяет обученную сеть на тех же данных.
    zones = data.zones["val"][:5]
    raw = _denormalize(data, data.windows["val"][:5])
    artifact_probs = predict_planning_windows(artifact, raw, zones)
    with torch.no_grad():
        trained_probs = result.model(
            torch.from_numpy(data.windows["val"][:5].astype(np.float32)),
            torch.from_numpy(zones.astype(np.int64)),
        ).prob.numpy()
    assert artifact_probs.shape == (5, 2)
    assert np.array_equal(artifact_probs, trained_probs)

    metrics = json.loads((model_dir / METRICS_FILE).read_text(encoding="utf-8"))
    assert metrics["model_version"] == model_dir.name
    assert metrics["arch"] == "mlp"
    assert metrics["horizons_min"] == [60, 120]
    assert metrics["window_min"] == 30
    assert metrics["n_train"] == int(data.windows["train"].shape[0])
    assert metrics["best_epoch"] == result.best_epoch
    assert set(metrics["val"]) == {"60", "120"}
    assert set(metrics["test"]) == {"60", "120"}
    for role in ("val", "test"):
        for horizon in ("60", "120"):
            assert set(metrics[role][horizon]) == {"ap", "roc_auc", "harm_top15"}
            assert 0.0 <= metrics[role][horizon]["harm_top15"] <= 1.0


def _denormalize(data, normalized: np.ndarray) -> np.ndarray:
    """Обратное преобразование признаков (для повторного пути инференса)."""

    mean = data.scalers.features.mean_
    scale = data.scalers.features.scale_
    return normalized * scale + mean


def test_save_planning_artifacts_refuses_existing_dir(pattern_db: Path, tmp_path: Path) -> None:
    """Занятый каталог — FileExistsError до всякой записи."""

    ml_cfg = _ml_cfg()
    root = tmp_path / "artifacts" / "planning"
    model_dir, data, result = _train_and_save(pattern_db, ml_cfg, root, "mlp")
    with pytest.raises(FileExistsError, match="уже существует"):
        save_planning_artifacts(
            model_dir,
            model_version=model_dir.name,
            family="mlp",
            model=result.model,
            data=data,
            result=result,
            ml_cfg=ml_cfg,
        )


def test_retrain_identical_weights_and_metrics(pattern_db: Path, tmp_path: Path) -> None:
    """Сценарий «Детерминизм обучения»: равные веса и metrics.json с одним seed."""

    ml_cfg = _ml_cfg()
    runs: list[Path] = []
    for run_no in (1, 2):
        root = tmp_path / f"run{run_no}" / "planning"
        model_dir, _data, _result = _train_and_save(pattern_db, ml_cfg, root, "mlp")
        runs.append(model_dir)

    first = torch.load(runs[0] / WEIGHTS_FILE, weights_only=True)
    second = torch.load(runs[1] / WEIGHTS_FILE, weights_only=True)
    assert first["ml_cfg"] == second["ml_cfg"]
    for name, tensor in first["state_dict"].items():
        assert torch.equal(tensor, second["state_dict"][name]), name

    metrics_first = json.loads((runs[0] / METRICS_FILE).read_text(encoding="utf-8"))
    metrics_second = json.loads((runs[1] / METRICS_FILE).read_text(encoding="utf-8"))
    assert metrics_first == metrics_second


# ---------------------------------------------------------------------------
# 4.1 Бейзлайны и метрики
# ---------------------------------------------------------------------------


def test_day_type_and_half_hour_index() -> None:
    """Типы дней: пн–чт → 0, пт → 1, сб → 2, вс → 3; получас 18:29 → 36."""

    assert day_type_index(datetime(2026, 11, 2, 12, 0)) == 0  # пн
    assert day_type_index(datetime(2026, 11, 5, 12, 0)) == 0  # чт
    assert day_type_index(datetime(2026, 11, 6, 12, 0)) == 1  # пт
    assert day_type_index(datetime(2026, 11, 7, 12, 0)) == 2  # сб
    assert day_type_index(datetime(2026, 11, 8, 12, 0)) == 3  # вс
    assert half_hour_index(datetime(2026, 11, 2, 18, 29)) == 36
    assert half_hour_index(datetime(2026, 11, 2, 0, 0)) == 0
    assert half_hour_index(datetime(2026, 11, 2, 23, 59)) == 47


def test_calendar_manual_mini_epoch() -> None:
    """Сценарий «Календарь на ручной мини-эпохе»: 3 позитива из 4 → 0.75."""

    cell_times = [datetime(2026, 11, 2, 18, 0), datetime(2026, 11, 2, 18, 5)]
    cell_times += [datetime(2026, 11, 3, 18, 10), datetime(2026, 11, 3, 18, 15)]
    other_times = [datetime(2026, 11, 2, 10, 0), datetime(2026, 11, 2, 10, 5)]
    other_times += [datetime(2026, 11, 3, 10, 10), datetime(2026, 11, 3, 10, 15)]
    keys = [(0, ts) for ts in cell_times + other_times]
    labels = np.zeros((8, 2), dtype=float)
    labels[:4, 0] = [1.0, 1.0, 1.0, 0.0]

    calendar = CalendarBaseline.fit(keys, labels)

    # Ячейка (зона 0, пн–чт, получас 36): 3 из 4 → 0.75 по H=60, 0 по H=120.
    score = calendar.score(0, datetime(2026, 11, 9, 18, 20))
    assert score[0] == pytest.approx(0.75)
    assert score[1] == pytest.approx(0.0)
    # Пустая ячейка — общая частота train (3/8 по H=60, 0 по H=120).
    fallback = calendar.score(0, datetime(2026, 11, 9, 3, 0))
    assert fallback[0] == pytest.approx(3.0 / 8.0)
    assert fallback[1] == pytest.approx(0.0)


def test_persistence_score_last_hour(tmp_path: Path) -> None:
    """Персистентность: потери за последний час до ts включительно, будущее не влияет."""

    noon = datetime(2026, 11, 2, 12, 0)
    rows = []
    for k in range(288):
        ts = add_minutes(SIM_START, k * 5)
        if noon >= ts > add_minutes(noon, -60):
            unfulfilled = 1
        elif ts > noon:
            unfulfilled = 5  # будущее не должно попасть в бейзлайн
        else:
            unfulfilled = 0
        rows.append(_metric_row(ts, 0, unfulfilled=unfulfilled))
    source = SqliteRowSource(_write_db(tmp_path / "persist.db", rows))

    assert persistence_score(source, 0, noon) == pytest.approx(12.0)
    assert persistence_score(source, 0, add_minutes(noon, -60)) == pytest.approx(0.0)


def test_score_metrics_budget_concentration() -> None:
    """AP/ROC-AUC определены, доля вреда топ-15 % следует за концентрацией вреда."""

    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    scores = np.linspace(0.0, 1.0, 10)
    harms_high = np.zeros(10)
    harms_high[8:] = 1.0  # весь вред — у двух самых высоких скоров
    harms_low = np.zeros(10)
    harms_low[:2] = 1.0  # весь вред — у двух самых низких скоров

    high = score_metrics(labels, scores, harms_high)
    low = score_metrics(labels, scores, harms_low)
    assert high["harm_top15"] == pytest.approx(1.0)
    assert low["harm_top15"] == pytest.approx(0.0)
    for metrics in (high, low):
        assert 0.0 <= metrics["ap"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_score_metrics_degenerate_returns_zeros() -> None:
    """Один класс меток и нулевой вред — нули без NaN и исключений."""

    metrics = score_metrics(np.zeros(4), np.linspace(0, 1, 4), np.zeros(4))
    assert metrics == {"ap": 0.0, "roc_auc": 0.0, "harm_top15": 0.0}


def test_evaluate_table_rows(trained_pair) -> None:
    """Сценарий «Таблица оценки»: четыре модели на каждый горизонт."""

    root, ml_cfg, db_path = trained_pair
    rows = evaluate_planning(
        SqliteRowSource(db_path),
        ml_cfg,
        zones=list(ZONES),
        start=SIM_START,
        artifacts_root=root,
    )

    combinations = {(row["model"], row["horizon_min"]) for row in rows}
    assert combinations == {
        (model, horizon)
        for model in ("календарь", "персистентность", "mlp", "lstm")
        for horizon in (60, 120)
    }
    for row in rows:
        assert set(row) == {"model", "horizon_min", "ap", "roc_auc", "harm_top15"}
        assert 0.0 <= float(row["ap"]) <= 1.0
        assert 0.0 <= float(row["roc_auc"]) <= 1.0
        assert 0.0 <= float(row["harm_top15"]) <= 1.0

    table = render_planning_table(rows)
    for model in ("календарь", "персистентность", "mlp", "lstm"):
        assert model in table
    assert "harm_top15" in table


def test_evaluate_horizon_snapshot_mismatch(trained_pair, tmp_path: Path) -> None:
    """Горизонты артефакта, расходящиеся с конфигом, — ValueError."""

    root, ml_cfg, db_path = trained_pair
    narrower = _ml_cfg(planning={"horizons_min": [60], "window_min": 30})
    with pytest.raises(ValueError, match="расходятся"):
        evaluate_planning(
            SqliteRowSource(db_path),
            narrower,
            zones=list(ZONES),
            start=SIM_START,
            artifacts_root=root,
        )


# ---------------------------------------------------------------------------
# 5.1 CLI plan-train / plan-eval
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, db_path: Path, name: str = "config.yaml", **overrides) -> Path:
    """YAML-конфиг CLI: мини-эпоха и мини-гиперпараметры (как в тестах M3)."""

    ml: dict[str, object] = {
        "db": str(db_path),
        "train_days": 2,
        "val_days": 1,
        "test_days": 1,
        "window_hours": 1,
        "mlp_hidden": [16, 128],
        "lstm_hidden": 8,
        "dropout": 0.0,
        "lr": 0.01,
        "batch_size": 64,
        "max_epochs": 3,
        "patience": 4,
        "seed": 42,
        "planning": {
            "horizons_min": [60, 120],
            "window_min": 30,
            "artifacts_root": "artifacts/planning",
        },
    }
    ml.update(overrides)
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"ml": ml}), encoding="utf-8")
    return path


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch, pattern_db: Path) -> Path:
    """cwd = tmp_path (корень артефактов относительный) и конфиг с мини-БД."""

    monkeypatch.chdir(tmp_path)
    return _write_config(tmp_path, pattern_db)


def test_cli_plan_train_and_eval(cli_env: Path, capsys) -> None:
    """Сценарий «Обучение и оценка на малой эпохе»: два каталога и таблица."""

    from datetime import date

    for family in ("mlp", "lstm"):
        assert main(["plan-train", "--config", str(cli_env), "--model", family]) == 0
    captured = capsys.readouterr()
    for family in ("mlp", "lstm"):
        version = f"{family}-v1-{date.today().isoformat()}"
        model_dir = Path("artifacts") / "planning" / version
        assert model_dir.is_dir()
        assert {entry.name for entry in model_dir.iterdir()} == set(ARTIFACT_FILES)
        assert f"model_version={version}" in captured.out
    assert "epoch 1/3 model=mlp" in captured.err

    assert main(["plan-eval", "--config", str(cli_env)]) == 0
    table = capsys.readouterr().out
    for model in ("календарь", "персистентность", "mlp", "lstm"):
        assert table.count(model) == 2, table
    assert "horizon" in table and "AP" in table and "harm_top15" in table
    assert "60" in table and "120" in table


def test_cli_plan_train_missing_db(cli_env: Path, tmp_path: Path, capsys) -> None:
    """Нет БД эпохи — exit 2, сообщение в stderr, артефактов нет."""

    rc = main(
        ["plan-train", "--config", str(cli_env), "--model", "mlp", "--db", str(tmp_path / "no.db")]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "не найдена" in captured.err
    assert captured.out == ""
    assert not Path("artifacts").exists()


def test_cli_plan_eval_without_artifact(cli_env: Path, capsys) -> None:
    """Сценарий «Нет артефакта»: plan-eval — exit 2 и сообщение в stderr."""

    rc = main(["plan-eval", "--config", str(cli_env)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "нет каталогов артефактов" in captured.err
    assert captured.out == ""


def test_cli_plan_eval_missing_db(cli_env: Path, tmp_path: Path, capsys) -> None:
    """plan-eval без БД эпохи — exit 2 и сообщение о недоступной БД."""

    rc = main(["plan-eval", "--config", str(cli_env), "--db", str(tmp_path / "no.db")])
    assert rc == 2
    assert "не найдена" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 6.1 Прогнозист живого контура
# ---------------------------------------------------------------------------

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


def _micro_cfg(planning_root: Path) -> AppConfig:
    """Синтетический мир из двух зон; корень артефактов — каталог теста."""

    return AppConfig.model_validate(
        {
            "seed": 7,
            "city": {
                "zones": MICRO_ZONES,
                "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 30.0}],
            },
            "population": {"scale": 50.0},
            "drivers": {"count": 4, "wait_limit_min": 6},
            "weather": {
                "rain_day_chance": 0.0,
                "episode_start_h": [0, 0],
                "episode_duration_h": [1, 1],
            },
            "pricing": {"surge_injection": {"day_chance": 0.0}},
            "ml": {"planning": {"artifacts_root": str(planning_root)}},
        }
    )


@pytest.fixture
def live_run(tmp_path: Path, pattern_db: Path) -> tuple[AppConfig, LiveRowSource, Path]:
    """Артефакт MLP на мини-эпохе + суточный прогон микромира с живым источником."""

    root = tmp_path / "artifacts" / "planning"
    _train_and_save(pattern_db, _ml_cfg(), root, "mlp")
    cfg = _micro_cfg(root)
    live = LiveRowSource()
    db_path = tmp_path / "run.db"
    conn = connect(str(db_path))
    init_db(conn)
    try:
        Engine(cfg).run(
            1,
            SqliteTripWriter(conn),
            on_minute=live.on_minute,
            on_metrics_row=live.on_metrics_row,
        )
    finally:
        conn.close()
    return cfg, live, db_path


def test_predictor_live_equals_offline(live_run) -> None:
    """Сценарий «Совпадение живого и офлайн-прогноза» на одних (зона, ts)."""

    cfg, live, db_path = live_run
    sqlite = SqliteRowSource(str(db_path))
    live_predictor = PlanningPredictor(live, cfg, family="mlp", horizon_min=60)
    db_predictor = PlanningPredictor(sqlite, cfg, family="mlp", horizon_min=60)

    last = live.time_range()[1]
    stamps = [last, add_minutes(last, -30), add_minutes(last, -90)]
    checked = 0
    for zone_id in (0, 1):
        for ts in stamps:
            live_prob = live_predictor.predict(zone_id, ts)
            db_prob = db_predictor.predict(zone_id, ts)
            assert live_prob is not None and db_prob is not None
            assert live_prob == db_prob
            checked += 1
    assert checked == 6


def test_predictor_horizon_outside_artifact(live_run) -> None:
    """Сценарий «Горизонт вне артефакта»: ошибка называет H и горизонты артефакта."""

    cfg, live, _db_path = live_run
    with pytest.raises(ValueError, match=r"90.*\[60, 120\]"):
        PlanningPredictor(live, cfg, family="mlp", horizon_min=90)


def test_predictor_warmup_returns_none(live_run) -> None:
    """Неполное окно (прогрев) — None, без исключений."""

    cfg, _live, db_path = live_run
    empty = LiveRowSource()
    predictor = PlanningPredictor(empty, cfg, family="mlp", horizon_min=60)
    assert predictor.predict(0, SqliteRowSource(str(db_path)).time_range()[0]) is None
