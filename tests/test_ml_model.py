"""Тесты ml/model.py (change ``m3-model``): туловища MLP/LSTM, multi-task
головки, loss, детерминизм seed и совместимость с примерами ``ml-dataset``.

Реальная эпоха (``data/m3_epoch.db``) не трогается: совместимость с датасетом
проверяется на рукописной мини-БД (приём ``tests/test_ml_dataset.py``) — сбор
``collect_examples``, нормализация ``DatasetScalers``, конвертация numpy →
float32/int64 tensor на границе вызывающего (решение 6 design.md). Детерминизм
dropout в тестах шагов исключён ``dropout=0.0`` (сравнение чистой сходимости);
взаимозаменяемость архитектур — параметризацией по ``arch``.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from common.config import MlCfg
from common.timeutil import add_minutes
from ml.dataset import RET_HORIZONS_MIN, DatasetScalers, SqliteRowSource, collect_examples
from ml.model import (
    TRUNK_OUTPUT_DIM,
    LSTMTrunk,
    MLPTrunk,
    ModelOutput,
    MultitaskLoss,
    build_model,
    deficit_pos_weight,
    set_global_seed,
)
from storage.db import connect
from storage.repo import insert_zone_metrics
from storage.schema import init_db

SIM_START = datetime(2026, 11, 2)
"""Старт мира мини-БД (дефолт конфига, понедельник)."""

N_ZONES = 4
"""Ёмкость эмбеддинга в синтетических тестах."""


def _batch(n: int = 5, steps: int = 12, seed: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Синтетический батч: окна (n, steps, 17) float32 и зоны (n,) int64."""

    rng = np.random.default_rng(seed)
    windows = torch.from_numpy(rng.normal(size=(n, steps, 17)).astype(np.float32))
    zones = torch.from_numpy(rng.integers(0, N_ZONES, size=n).astype(np.int64))
    return windows, zones


# ---------------------------------------------------------------------------
# 1.1/1.2/1.3 Туловища: контракт «окно + зона → R»
# ---------------------------------------------------------------------------


def test_mlp_trunk_forward_shapes() -> None:
    """MLP: (n, steps, 17) + зоны (n,) → (n, 128) на конфиге с окном 1 ч."""
    set_global_seed(0)
    trunk = MLPTrunk(MlCfg(window_hours=1), n_zones=N_ZONES)
    windows, zones = _batch()
    assert trunk(windows, zones).shape == (5, TRUNK_OUTPUT_DIM)


def test_mlp_trunk_default_config_layers() -> None:
    """Дефолтный конфиг (окно 3 ч → 36 шагов, победитель свипа
    ``m3-train-cli``): слои ровно по
    ``ml.mlp_hidden=[256, 128]`` без лишней финальной проекции; dropout из
    ``ml.dropout`` (сценарий «построение по гиперпараметрам конфига»)."""
    set_global_seed(0)
    cfg = MlCfg()
    trunk = MLPTrunk(cfg, n_zones=N_ZONES)
    assert isinstance(trunk.net[0], nn.Linear)
    assert trunk.net[0].weight.shape == (256, 36 * 17 + cfg.zone_embed_dim)
    assert isinstance(trunk.net[1], nn.ReLU)
    assert isinstance(trunk.net[2], nn.Dropout)
    assert trunk.net[2].p == cfg.dropout
    assert isinstance(trunk.net[-1], nn.Linear)
    assert trunk.net[-1].weight.shape == (TRUNK_OUTPUT_DIM, 256)
    windows, zones = _batch(n=2, steps=36)
    assert trunk(windows, zones).shape == (2, TRUNK_OUTPUT_DIM)


def test_mlp_trunk_rejects_mismatched_hidden() -> None:
    """``mlp_hidden[-1] ≠ 128`` (и пустой список) — ValueError с сообщением о
    TRUNK_OUTPUT_DIM (контракт общих головок не может быть нарушен тихо)."""
    with pytest.raises(ValueError, match="128"):
        MLPTrunk(MlCfg(window_hours=1, mlp_hidden=[256]), n_zones=N_ZONES)
    with pytest.raises(ValueError, match="128"):
        MLPTrunk(MlCfg(window_hours=1, mlp_hidden=[]), n_zones=N_ZONES)


@pytest.mark.parametrize("lstm_layers", [1, 2])
def test_lstm_trunk_forward_shapes(lstm_layers: int) -> None:
    """LSTM: (n, steps, 17) + зоны (n,) → (n, 128), в том числе при двух слоях;
    размеры рекуррентной части — из ml.lstm_hidden/ml.lstm_layers."""
    set_global_seed(0)
    cfg = MlCfg(window_hours=1, lstm_layers=lstm_layers)
    trunk = LSTMTrunk(cfg, n_zones=N_ZONES)
    assert trunk.lstm.num_layers == lstm_layers
    assert trunk.lstm.hidden_size == cfg.lstm_hidden
    windows, zones = _batch()
    assert trunk(windows, zones).shape == (5, TRUNK_OUTPUT_DIM)


def test_build_model_rejects_unknown_arch() -> None:
    """Фабрика принимает только arch ∈ {"mlp", "lstm"}."""
    with pytest.raises(ValueError, match="lstm"):
        build_model("transformer", MlCfg(window_hours=1), n_zones=N_ZONES)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2.1 MultiTaskNet: шесть величин контракта forecasts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["mlp", "lstm"])
def test_multitask_net_output_shapes(arch: str) -> None:
    """Сценарий взаимозаменяемости: смена туловища не меняет форм — у обеих
    архитектур выходы (n, 2) ×3, вероятности в [0, 1], колонки — по числу
    горизонтов 15/30."""
    set_global_seed(0)
    net = build_model(arch, MlCfg(window_hours=1), n_zones=N_ZONES)  # type: ignore[arg-type]
    net.eval()
    assert net.head_requests.in_features == TRUNK_OUTPUT_DIM
    assert net.head_requests.out_features == len(RET_HORIZONS_MIN)
    windows, zones = _batch(n=6)
    with torch.no_grad():
        out = net(windows, zones)
    assert isinstance(out, ModelOutput)
    assert out.pred_requests.shape == (6, 2)
    assert out.pred_free_drivers.shape == (6, 2)
    assert out.deficit_logits.shape == (6, 2)
    assert out.deficit_prob.shape == (6, 2)
    assert out.deficit_prob.min().item() >= 0.0
    assert out.deficit_prob.max().item() <= 1.0


@pytest.mark.parametrize("arch", ["mlp", "lstm"])
def test_zone_id_changes_output(arch: str) -> None:
    """Сценарий «зона влияет на прогноз»: одинаковое окно с разными zone_id
    даёт разные выходы (чистый forward — в eval(), без масок dropout)."""
    set_global_seed(0)
    net = build_model(arch, MlCfg(window_hours=1), n_zones=2)  # type: ignore[arg-type]
    net.eval()
    window = torch.zeros(1, 12, 17)
    zones = torch.tensor([0, 1], dtype=torch.int64)
    with torch.no_grad():
        out = net(window.repeat(2, 1, 1), zones)
    assert not torch.allclose(out.pred_requests[0], out.pred_requests[1])


# ---------------------------------------------------------------------------
# 2.2 MultitaskLoss и deficit_pos_weight
# ---------------------------------------------------------------------------


def test_loss_zero_mse_on_perfect_regression() -> None:
    """Идеальные регрессии: оба MSE-компонента нулевые, total — только
    взвешенный BCE-член (сценарий спеки)."""
    y_req = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    y_free = torch.tensor([[2.0, 1.0], [-1.0, 0.0]])
    logits = torch.zeros(2, 2)
    out = ModelOutput(
        pred_requests=y_req.clone(),
        pred_free_drivers=y_free.clone(),
        deficit_logits=logits,
        deficit_prob=torch.sigmoid(logits),
    )
    comps = MultitaskLoss(cls_weight=1.0, pos_weight=1.0)(out, y_req, y_free, torch.ones(2, 2))
    assert comps.mse_requests.item() == 0.0
    assert comps.mse_free.item() == 0.0
    assert torch.allclose(comps.total, comps.bce)


def test_loss_cls_weight_zero_excludes_bce() -> None:
    """``cls_weight = 0``: total не содержит BCE-члена и равен сумме двух MSE
    (сценарий спеки); bce-компонент — нулевой заглушки."""
    set_global_seed(0)
    out = ModelOutput(
        pred_requests=torch.randn(3, 2),
        pred_free_drivers=torch.randn(3, 2),
        deficit_logits=torch.randn(3, 2),
        deficit_prob=torch.sigmoid(torch.randn(3, 2)),
    )
    y_req, y_free = torch.randn(3, 2), torch.randn(3, 2)
    comps = MultitaskLoss(cls_weight=0.0, pos_weight=4.0)(out, y_req, y_free, torch.ones(3, 2))
    assert torch.equal(comps.total, comps.mse_requests + comps.mse_free)
    assert comps.bce.item() == 0.0


def test_deficit_pos_weight_from_share() -> None:
    """``pos_weight = (1 − p) / p`` по фактической доле дефицитных бакетов:
    p = 0.25 → 3; p = 0.5 → 1 (сценарий спеки)."""
    labels = np.zeros((4, 2))
    labels[0, 0] = 1.0
    labels[1, 0] = 1.0
    assert deficit_pos_weight(labels) == pytest.approx(3.0)
    assert deficit_pos_weight(np.array([[1.0, 0.0], [0.0, 1.0]])) == 1.0


@pytest.mark.parametrize("labels", [np.zeros((3, 2)), np.ones((3, 2))])
def test_deficit_pos_weight_degenerate(labels: np.ndarray) -> None:
    """Доля дефицита 0 или 1 — ValueError (мир без дефицита не обучаем)."""
    with pytest.raises(ValueError):
        deficit_pos_weight(labels)


# ---------------------------------------------------------------------------
# 3.1 Smoke-обучение: loss убывает
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["mlp", "lstm"])
def test_smoke_training_loss_decreases(arch: str) -> None:
    """Несколько шагов Adam по фиксированному батчу с закономерной зависимостью
    таргетов от входа (средний признак окна): loss после шагов строго меньше
    исходного (сценарий спеки «loss убывает»)."""
    set_global_seed(11)
    cfg = MlCfg(window_hours=1, dropout=0.0)  # чистая сходимость без масок dropout
    n = 16
    rng = np.random.default_rng(5)
    windows = torch.from_numpy(rng.normal(size=(n, 12, 17)).astype(np.float32))
    zones = torch.from_numpy((np.arange(n) % N_ZONES).astype(np.int64))
    y_req = windows[:, :, 0].mean(dim=1, keepdim=True).repeat(1, 2)
    y_free = windows[:, :, 1].mean(dim=1, keepdim=True).repeat(1, 2)
    y_def = torch.tensor([[1.0, 0.0]]).repeat(n, 1)  # доля дефицита 0.5 → pos_weight 1
    net = build_model(arch, cfg, n_zones=N_ZONES)  # type: ignore[arg-type]
    loss_fn = MultitaskLoss(cls_weight=1.0, pos_weight=deficit_pos_weight(y_def.numpy()))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        comps = loss_fn(net(windows, zones), y_req, y_free, y_def)
        comps.total.backward()
        opt.step()
        losses.append(comps.total.item())
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# 3.2 Детерминизм seed
# ---------------------------------------------------------------------------


def _train_steps(arch: str, seed: int, n_steps: int = 5) -> tuple[dict, list]:
    """Построить сеть с фиксированным seed и прогнать одинаковые шаги обучения
    на одинаковом батче; вернуть начальные веса (state_dict) и loss каждого шага."""
    set_global_seed(seed)
    net = build_model(arch, MlCfg(window_hours=1, dropout=0.0), n_zones=N_ZONES)  # type: ignore[arg-type]
    init = {k: v.clone() for k, v in net.state_dict().items()}
    rng = np.random.default_rng(7)  # батч от собственного rng — не зависит от seed сети
    windows = torch.from_numpy(rng.normal(size=(6, 12, 17)).astype(np.float32))
    zones = torch.from_numpy(rng.integers(0, N_ZONES, size=6).astype(np.int64))
    y_req = torch.from_numpy(rng.normal(size=(6, 2)).astype(np.float32))
    y_free = torch.from_numpy(rng.normal(size=(6, 2)).astype(np.float32))
    y_def = torch.from_numpy(rng.integers(0, 2, size=(6, 2)).astype(np.float32))
    loss_fn = MultitaskLoss(cls_weight=1.0, pos_weight=2.0)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    losses = []
    for _ in range(n_steps):
        opt.zero_grad()
        comps = loss_fn(net(windows, zones), y_req, y_free, y_def)
        comps.total.backward()
        opt.step()
        losses.append(comps.total.detach())
    return init, losses


@pytest.mark.parametrize("arch", ["mlp", "lstm"])
def test_same_seed_identical_init_and_losses(arch: str) -> None:
    """Один seed: идентичные начальные веса и идентичные значения loss на
    каждом шаге — точное равенство тензоров (сценарий спеки)."""
    init_a, losses_a = _train_steps(arch, seed=42)
    init_b, losses_b = _train_steps(arch, seed=42)
    assert set(init_a) == set(init_b)
    for key in init_a:
        assert torch.equal(init_a[key], init_b[key])
    assert all(torch.equal(a, b) for a, b in zip(losses_a, losses_b))


@pytest.mark.parametrize("arch", ["mlp", "lstm"])
def test_different_seeds_different_init(arch: str) -> None:
    """Разные seed: начальные веса различаются (сценарий спеки)."""
    init_a, _ = _train_steps(arch, seed=42)
    init_b, _ = _train_steps(arch, seed=43)
    assert any(not torch.equal(init_a[key], init_b[key]) for key in init_a)


# ---------------------------------------------------------------------------
# 3.3 Совместимость с ml-dataset без переупаковки
# ---------------------------------------------------------------------------


def _metric_row(ts: datetime, requests: int, free_drivers: int) -> dict:
    """Строка ``zone_metrics_5min`` мини-БД (supply_demand_ratio — по контракту)."""

    return {
        "ts": ts,
        "zone_id": 0,
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


@pytest.mark.parametrize("arch", ["mlp", "lstm"])
def test_dataset_examples_forward_and_loss(arch: str, tmp_path: Path) -> None:
    """Сценарий «вход совместим с примерами датасета»: мини-БД →
    ``collect_examples`` → ``DatasetScalers`` → numpy → float32/int64 tensor →
    forward + MultitaskLoss без переупаковки."""
    # Мини-эпоха 3 дня × 288 бакетов зоны 0 (приём days_db из tests/test_ml_dataset.py).
    path = tmp_path / "days.db"
    conn = connect(str(path))
    init_db(conn)
    rng = np.random.default_rng(7)
    rows: list[dict] = []
    t = SIM_START
    end = SIM_START + timedelta(days=3) - timedelta(minutes=5)
    while t <= end:
        rows.append(_metric_row(t, int(rng.integers(0, 8)), int(rng.integers(1, 6))))
        t = add_minutes(t, 5)
    insert_zone_metrics(conn, rows)
    conn.close()

    splits = collect_examples(
        SqliteRowSource(path),
        zones=[0],
        start=SIM_START,
        train_days=2,
        val_days=1,
        test_days=1,
        window_hours=1,
    )
    train, test = splits["train"], splits["test"]
    assert train.windows.shape[1:] == (12, 17)
    scalers = DatasetScalers.fit(train.windows, train.requests, train.free_drivers)

    cfg = MlCfg(window_hours=1)
    set_global_seed(cfg.seed)
    net = build_model(arch, cfg, n_zones=1)  # type: ignore[arg-type]
    loss_fn = MultitaskLoss(cls_weight=cfg.cls_weight, pos_weight=deficit_pos_weight(train.deficit))

    windows = torch.from_numpy(scalers.transform_windows(test.windows).astype(np.float32))
    zones = torch.tensor([zone for zone, _ in test.keys], dtype=torch.int64)
    y_req_np, y_free_np = scalers.transform_targets(test.requests, test.free_drivers)
    y_req = torch.from_numpy(y_req_np.astype(np.float32))
    y_free = torch.from_numpy(y_free_np.astype(np.float32))
    y_def = torch.from_numpy(test.deficit.astype(np.float32))

    out = net(windows, zones)
    comps = loss_fn(out, y_req, y_free, y_def)
    assert out.pred_requests.shape == y_req.shape == (len(test.keys), 2)
    assert out.pred_free_drivers.shape == y_free.shape == (len(test.keys), 2)
    assert out.deficit_prob.min().item() >= 0.0
    assert out.deficit_prob.max().item() <= 1.0
    assert torch.isfinite(comps.total)
    comps.total.backward()  # граф loss корректен — backward по test-батчу проходит
