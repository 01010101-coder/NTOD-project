"""Тесты ml/train.py и ml/__main__.py (change ``m3-train-cli``): подготовка
данных обучения, цикл с ранней остановкой, артефакты версии модели,
детерминизм seed, CLI train/sweep.

Реальная эпоха (``data/m3_epoch.db``) не трогается (§4 tasks.md — отдельные
шаги с выполнением): все проверки — на рукописных мини-БД в ``tmp_path``
(приём ``tests/test_ml_dataset.py``). Гиперпараметры тестовых конфигов —
малые (mlp_hidden [16, 128], dropout 0.0 как в ``test_ml_model.py``), чтобы
обучение в тестах сходилось за секунды.
"""

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from common.config import MlCfg
from ml.__main__ import main
from ml.dataset import DatasetScalers
from ml.model import build_model
from ml.train import (
    LOSS_FILE,
    METRICS_FILE,
    SCALERS_FILE,
    WEIGHTS_FILE,
    model_version_for,
    prepare_training_data,
    save_model_artifacts,
    train_model,
    val_mae_requests,
)
from storage.db import connect
from storage.repo import insert_zone_metrics
from storage.schema import init_db

SIM_START = datetime(2026, 11, 2)
"""Старт мира мини-БД (дефолт engine.sim_start конфига, понедельник)."""

ZONES = (0, 1)
"""Зоны мини-эпохи; ёмкость эмбеддинга n_zones = 2."""

ARTIFACT_FILES = frozenset({WEIGHTS_FILE, SCALERS_FILE, LOSS_FILE, METRICS_FILE})
"""Полный набор файлов каталога версии модели."""


def _ml_cfg(**overrides: object) -> MlCfg:
    """Тестовые гиперпараметры обучения: малая сеть и несколько эпох."""

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


def _prepare(db_path: Path, **overrides: object):
    """Подготовка данных мини-эпохи с тестовыми гиперпараметрами."""

    return prepare_training_data(db_path, _ml_cfg(**overrides), zones=list(ZONES), start=SIM_START)


def _pattern_row(ts: datetime, j: int, zone_id: int) -> dict:
    """Строка метрик с закономерностью: суточная синусоида спроса (кодируется
    календарными признаками окна), дефицит — у высоких бакетов (обоих классов
    меток поровну не бывает, но оба класса представлены)."""

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


@pytest.fixture
def pattern_db(tmp_path: Path) -> Path:
    """Мини-эпоха с закономерностью: 3 дня × 288 бакетов, зоны 0/1."""

    path = tmp_path / "pattern.db"
    conn = connect(str(path))
    init_db(conn)
    rows: list[dict] = []
    t = SIM_START
    end = SIM_START + timedelta(days=3) - timedelta(minutes=5)
    while t <= end:
        j = int((t - SIM_START).total_seconds()) // 300
        for zone_id in ZONES:
            rows.append(_pattern_row(t, j, zone_id))
        t = t + timedelta(minutes=5)
    insert_zone_metrics(conn, rows)
    conn.close()
    return path


@pytest.fixture
def noise_db(tmp_path: Path) -> Path:
    """Мини-эпоха со случайным шумом rng(7): закономерности нет, val-loss
    выходит на плато и деградирует — для ранней остановки и лучшей эпохи."""

    path = tmp_path / "noise.db"
    conn = connect(str(path))
    init_db(conn)
    rng = np.random.default_rng(7)
    rows = []
    t = SIM_START
    end = SIM_START + timedelta(days=3) - timedelta(minutes=5)
    while t <= end:
        rows.append(
            {
                "ts": t,
                "zone_id": 0,
                "requests": int(rng.integers(0, 8)),
                "completed": int(rng.integers(0, 4)),
                "unfulfilled": 1 if rng.random() < 0.2 else 0,
                "free_drivers": int(rng.integers(1, 6)),
                "busy_drivers": int(rng.integers(0, 5)),
                "avg_wait_min": float(rng.integers(0, 6)) * 0.5,
                "supply_demand_ratio": 1.0,
                "is_rain": 1 if rng.random() < 0.1 else 0,
                "event": "",
                "price_multiplier": 1.0,
            }
        )
        t = t + timedelta(minutes=5)
    insert_zone_metrics(conn, rows)
    conn.close()
    return path


def _write_config(tmp_path: Path, db_path: Path, name: str = "config.yaml", **overrides) -> Path:
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
    ml.update(overrides)
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"ml": ml}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1.1 Подготовка данных обучения
# ---------------------------------------------------------------------------


def test_prepare_roles_and_shapes(pattern_db: Path) -> None:
    """Роли непусты, формы соответствуют контракту Examples (окно 1 ч → 12
    шагов), train-нормализация даёт среднее ≈ 0; сырые таргеты не тронуты."""
    data = _prepare(pattern_db)
    assert data.windows["train"].shape[1:] == (12, 17)
    for role in ("train", "val", "test"):
        assert data.windows[role].shape[0] > 0
        assert data.zones[role].dtype == np.int64
        assert data.requests[role].shape == (data.windows[role].shape[0], 2)
    assert abs(float(data.windows["train"].mean())) < 1e-9
    assert float(data.raw_requests["val"].min()) >= 0.0
    assert data.n_zones == max(ZONES) + 1


def test_prepare_pos_weight(pattern_db: Path) -> None:
    """pos_weight по train-меткам: дефицит — меньшинство, вес > 1 и конечен."""
    data = _prepare(pattern_db)
    share = data.deficit["train"].mean()
    assert 0.0 < share < 1.0
    assert data.pos_weight == pytest.approx((1.0 - share) / share)


def test_prepare_pos_weight_harm_rule(noise_db: Path) -> None:
    """При ``deficit_label: harm`` метки и pos_weight считаются по вреду:
    окна не меняются, метки отличаются, вес = негативы/позитивы harm train-роли."""
    legacy = _prepare(noise_db)
    harm = _prepare(noise_db, deficit_label="harm")
    assert np.array_equal(harm.windows["train"], legacy.windows["train"])
    assert not np.array_equal(harm.deficit["train"], legacy.deficit["train"])
    share = float(harm.deficit["train"].mean())
    assert 0.0 < share < 1.0
    assert harm.pos_weight == pytest.approx((1.0 - share) / share)


def test_prepare_empty_role_raises(pattern_db: Path) -> None:
    """Пустая роль val (val_days=0) и пустая роль train (train_days=0) —
    ValueError с именем роли до построения сети."""
    with pytest.raises(ValueError, match="роль val пуста"):
        _prepare(pattern_db, train_days=1, val_days=0)
    with pytest.raises(ValueError, match="роль train пуста"):
        _prepare(pattern_db, train_days=0, val_days=0)


# ---------------------------------------------------------------------------
# 1.2 Обучение одной архитектуры
# ---------------------------------------------------------------------------


def test_train_loss_decreases(pattern_db: Path) -> None:
    """На мини-эпохе с закономерностью train-loss последней эпохи меньше
    первой (сценарий «Loss убывает на обучении»); кап не досрочно."""
    data = _prepare(pattern_db)
    result = train_model("mlp", data, _ml_cfg(max_epochs=5))
    assert result.history[0]["train"]["total"] > result.history[-1]["train"]["total"]
    assert [entry["epoch"] for entry in result.history] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("arch", ["mlp", "lstm"])
def test_train_both_architectures(pattern_db: Path, arch: str) -> None:
    """Обе архитектуры проходят один и тот же цикл: history, best_epoch ≥ 1."""
    data = _prepare(pattern_db)
    result = train_model(arch, data, _ml_cfg(max_epochs=3))
    assert result.best_epoch in (1, 2, 3)
    assert result.wall_sec > 0.0
    assert set(result.best_val) == {"total", "mse_requests", "mse_free", "bce"}


def test_train_early_stopping(noise_db: Path) -> None:
    """patience=1 при max_epochs=30: остановка до капа, лучшая эпоха —
    аргминимум val-total по history (сценарий «Ранняя остановка»)."""
    data = _prepare(noise_db)
    result = train_model("mlp", data, _ml_cfg(patience=1, max_epochs=30))
    assert len(result.history) < 30
    val_totals = [entry["val"]["total"] for entry in result.history]
    assert result.best_epoch == val_totals.index(min(val_totals)) + 1


def test_train_restores_best_epoch_state(noise_db: Path) -> None:
    """После деградации val сеть соответствует эпохе минимума val-loss
    (сценарий «Сохраняется лучшая эпоха»): повторное обучение с капом
    best_epoch детерминированно даёт то же состояние (та же траектория —
    тот же seed и тот же порядок перестановок)."""
    data = _prepare(noise_db)
    long_run = train_model("mlp", data, _ml_cfg(max_epochs=40, patience=40))
    assert long_run.best_epoch < len(long_run.history), "тест предполагает деградацию val"
    rerun = train_model("mlp", data, _ml_cfg(max_epochs=long_run.best_epoch, patience=40))
    for name, tensor in rerun.model.state_dict().items():
        assert torch.equal(tensor, long_run.model.state_dict()[name]), name


# ---------------------------------------------------------------------------
# 1.3 Артефакты версии модели
# ---------------------------------------------------------------------------


def test_save_artifacts_and_roundtrip(pattern_db: Path, tmp_path: Path) -> None:
    """Четыре файла созданы и читаются: восстановленная из weights.pt сеть
    даёт побитово те же выходы на фиксированном батче, scaler'ы — то же
    преобразование; loss.json и metrics.json согласованы с обучением."""
    ml_cfg = _ml_cfg(max_epochs=2)
    data = _prepare(pattern_db)
    result = train_model("mlp", data, ml_cfg)

    version = model_version_for("mlp", today=date(2026, 11, 6))
    model_dir = tmp_path / "artifacts" / version
    save_model_artifacts(
        model_dir,
        model_version=version,
        arch="mlp",
        model=result.model,
        data=data,
        result=result,
        ml_cfg=ml_cfg,
    )
    assert {entry.name for entry in model_dir.iterdir()} == set(ARTIFACT_FILES)

    blob = torch.load(model_dir / WEIGHTS_FILE, weights_only=True)
    assert blob["arch"] == "mlp"
    assert blob["ml_cfg"]["seed"] == ml_cfg.seed
    restored = build_model(blob["arch"], MlCfg(**blob["ml_cfg"]), n_zones=data.n_zones)
    restored.load_state_dict(blob["state_dict"])
    restored.eval()

    idx = np.arange(8)
    windows = torch.from_numpy(data.windows["val"][idx].astype(np.float32))
    zones = torch.from_numpy(data.zones["val"][idx].astype(np.int64))
    saved_out = restored(windows, zones)
    trained_out = result.model(windows, zones)
    assert torch.equal(saved_out.pred_requests, trained_out.pred_requests)
    assert torch.equal(saved_out.pred_free_drivers, trained_out.pred_free_drivers)
    assert torch.equal(saved_out.deficit_logits, trained_out.deficit_logits)
    assert torch.equal(saved_out.deficit_prob, trained_out.deficit_prob)

    scalers = DatasetScalers.load(model_dir / SCALERS_FILE)
    probe = np.random.default_rng(0).normal(size=(3, 12, 17))
    assert np.array_equal(scalers.transform_windows(probe), data.scalers.transform_windows(probe))

    history = json.loads((model_dir / LOSS_FILE).read_text(encoding="utf-8"))
    assert [entry["epoch"] for entry in history] == [1, 2]
    for entry in history:
        assert set(entry["train"]) == {"total", "mse_requests", "mse_free", "bce"}
        assert set(entry["val"]) == {"total", "mse_requests", "mse_free", "bce"}

    metrics = json.loads((model_dir / METRICS_FILE).read_text(encoding="utf-8"))
    assert metrics["model_version"] == version
    assert metrics["arch"] == "mlp"
    assert metrics["window_hours"] == 1
    assert metrics["seed"] == ml_cfg.seed
    assert metrics["n_train"] == int(data.windows["train"].shape[0])
    assert metrics["n_val"] == int(data.windows["val"].shape[0])
    assert metrics["best_epoch"] == (min(range(2), key=lambda i: history[i]["val"]["total"]) + 1)
    assert metrics["best_val"]["total"] == min(entry["val"]["total"] for entry in history)
    assert metrics["ml_cfg"] == blob["ml_cfg"]
    assert metrics["wall_sec"] >= 0.0


def test_artifacts_record_harm_label(pattern_db: Path, tmp_path: Path) -> None:
    """Сеть, обученная при ``harm``: снимок и metrics.json фиксируют правило,
    pos_weight посчитан по меткам harm train-роли (сценарий «Правило метки в артефакте»)."""
    ml_cfg = _ml_cfg(max_epochs=2, deficit_label="harm")
    data = _prepare(pattern_db, deficit_label="harm")
    result = train_model("mlp", data, ml_cfg)
    version = model_version_for("mlp", today=date(2026, 11, 7))
    model_dir = tmp_path / "artifacts" / version
    save_model_artifacts(
        model_dir,
        model_version=version,
        arch="mlp",
        model=result.model,
        data=data,
        result=result,
        ml_cfg=ml_cfg,
    )

    blob = torch.load(model_dir / WEIGHTS_FILE, weights_only=True)
    assert blob["ml_cfg"]["deficit_label"] == "harm"
    metrics = json.loads((model_dir / METRICS_FILE).read_text(encoding="utf-8"))
    assert metrics["deficit_label"] == "harm"
    share = float(data.deficit["train"].mean())
    assert metrics["pos_weight"] == pytest.approx((1.0 - share) / share)


def test_save_artifacts_refuses_existing_dir(pattern_db: Path, tmp_path: Path) -> None:
    """Занятый каталог версии — FileExistsError до всякой записи."""
    ml_cfg = _ml_cfg(max_epochs=1)
    data = _prepare(pattern_db)
    result = train_model("mlp", data, ml_cfg)
    model_dir = tmp_path / "artifacts" / model_version_for("mlp")
    model_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="уже существует"):
        save_model_artifacts(
            model_dir,
            model_version=model_dir.name,
            arch="mlp",
            model=result.model,
            data=data,
            result=result,
            ml_cfg=ml_cfg,
        )
    assert not (model_dir / WEIGHTS_FILE).exists()


# ---------------------------------------------------------------------------
# 2.1 Детерминизм повторного обучения
# ---------------------------------------------------------------------------


def test_retrain_identical_artifacts(pattern_db: Path, tmp_path: Path, monkeypatch) -> None:
    """Два запуска с одним seed в разных корнях артефактов дают loss.json и
    metrics.json, совпадающие значением в значение (wall_sec — таймер, из
    сравнения исключается)."""
    ml_cfg = _ml_cfg(max_epochs=3)
    runs: list[Path] = []
    for run_no in (1, 2):
        root = tmp_path / f"run{run_no}"
        root.mkdir()
        monkeypatch.chdir(root)
        data = prepare_training_data(pattern_db, ml_cfg, zones=list(ZONES), start=SIM_START)
        result = train_model("mlp", data, ml_cfg)
        version = model_version_for("mlp")
        save_model_artifacts(
            root / "artifacts" / version,
            model_version=version,
            arch="mlp",
            model=result.model,
            data=data,
            result=result,
            ml_cfg=ml_cfg,
        )
        runs.append(root / "artifacts" / version)

    loss_first = json.loads((runs[0] / LOSS_FILE).read_text(encoding="utf-8"))
    loss_second = json.loads((runs[1] / LOSS_FILE).read_text(encoding="utf-8"))
    assert loss_first == loss_second

    metrics_first = json.loads((runs[0] / METRICS_FILE).read_text(encoding="utf-8"))
    metrics_second = json.loads((runs[1] / METRICS_FILE).read_text(encoding="utf-8"))
    metrics_first.pop("wall_sec")
    metrics_second.pop("wall_sec")
    assert metrics_first == metrics_second


# ---------------------------------------------------------------------------
# 3.1/3.2 CLI train/sweep
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch, pattern_db: Path) -> Path:
    """cwd = tmp_path (корень артефактов — относительный) и конфиг с мини-БД."""

    monkeypatch.chdir(tmp_path)
    return _write_config(tmp_path, pattern_db)


def test_cli_train_smoke(cli_env: Path, capsys) -> None:
    """Smoke обеих архитектур: exit 0, два каталога по четыре файла, строки
    `done: model_version=… artifacts=… val=…` в stdout и прогресс эпох в stderr."""
    rc = main(["train", "--config", str(cli_env)])
    assert rc == 0
    captured = capsys.readouterr()

    today = date.today().isoformat()
    done_lines = [line for line in captured.out.splitlines() if line.startswith("done:")]
    assert len(done_lines) == 2
    for arch in ("mlp", "lstm"):
        version = f"{arch}-v1-{today}"
        match = re.search(
            rf"done: model_version={version} artifacts=\S+ val=(\d+\.\d+)", captured.out
        )
        assert match, captured.out
        model_dir = Path("artifacts") / version
        assert model_dir.is_dir()
        assert {entry.name for entry in model_dir.iterdir()} == set(ARTIFACT_FILES)
    assert re.search(
        r"epoch 1/4 arch=mlp loss=\d+\.\d+ val=\d+\.\d+ patience_left=\d+", captured.err
    )
    assert "arch=lstm" in captured.err


def test_cli_train_missing_db(cli_env: Path, tmp_path: Path, capsys) -> None:
    """Несуществующий --db — exit 2 и сообщение о недоступной БД, без артефактов."""
    rc = main(["train", "--config", str(cli_env), "--db", str(tmp_path / "missing.db")])
    assert rc == 2
    captured = capsys.readouterr()
    assert "не найдена" in captured.err
    assert captured.out == ""
    assert not Path("artifacts").exists()


def test_cli_train_occupied_dir(cli_env: Path, tmp_path: Path, capsys) -> None:
    """Занятый каталог версии — exit 2 до обучения, каталог не тронут."""
    version = model_version_for("mlp")
    occupied = Path("artifacts") / version
    occupied.mkdir(parents=True)
    sentinel = occupied / WEIGHTS_FILE
    sentinel.write_bytes(b"sentinel")

    rc = main(["train", "--config", str(cli_env), "--arch", "mlp"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "уже существует" in captured.err
    assert sentinel.read_bytes() == b"sentinel"


def test_cli_train_empty_role(cli_env: Path, tmp_path: Path, capsys) -> None:
    """Пустая роль (val_days=0) — exit 2 с сообщением об ошибке."""
    empty_cfg = _write_config(
        tmp_path,
        tmp_path / "pattern.db",
        name="empty.yaml",
        train_days=1,
        val_days=0,
    )
    rc = main(["train", "--config", str(empty_cfg), "--arch", "mlp"])
    assert rc == 2
    assert "роль val пуста" in capsys.readouterr().err


def test_cli_sweep(cli_env: Path, tmp_path: Path, capsys) -> None:
    """Свип --windows 1,2: json со строкой на каждое окно, winner — минимум
    val MAE, в корне артефактов ровно один новый файл (без каталогов моделей)."""
    rc = main(["sweep", "--config", str(cli_env), "--windows", "1,2"])
    assert rc == 0
    captured = capsys.readouterr()

    artifacts = Path("artifacts")
    assert [entry.name for entry in sorted(artifacts.iterdir())] == ["window-sweep.json"]

    sweep = json.loads((artifacts / "window-sweep.json").read_text(encoding="utf-8"))
    assert [row["window_hours"] for row in sweep["windows"]] == [1, 2]
    for row in sweep["windows"]:
        assert set(row) == {"window_hours", "val_mae_requests", "best_val_total", "best_epoch"}
        assert row["val_mae_requests"] >= 0.0
    best = min(sweep["windows"], key=lambda row: row["val_mae_requests"])
    assert sweep["winner"] == best["window_hours"]
    assert "winner" in captured.out


def test_val_mae_requests_physical_scale(pattern_db: Path) -> None:
    """MAE свипа считается в физических заказах: сырой val-таргет лежит в
    диапазоне паттерна (0…8 заказов), MAE не превышает размаха таргета."""
    data = _prepare(pattern_db)
    result = train_model("mlp", data, _ml_cfg(max_epochs=2))
    raw_spread = float(data.raw_requests["val"].max() - data.raw_requests["val"].min())
    mae = val_mae_requests(result.model, data)
    assert 0.0 <= mae <= raw_spread
