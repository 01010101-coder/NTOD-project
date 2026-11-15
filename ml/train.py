"""Цикл обучения вехи M3 (change ``m3-train-cli``): подготовка данных, обучение
архитектур с ранней остановкой, артефакты версии модели.

Тонкая склейка поверх ``ml/`` (решение 1 design.md): сбор примеров —
:func:`ml.dataset.collect_examples`, нормализация — :class:`ml.dataset.DatasetScalers`
(fit только на train-роли), loss — :class:`ml.model.MultitaskLoss`, построение
сети — :func:`ml.model.build_model`. Детерминизм — из двух независимых rng
(решение 2 design.md): инициализация весов и маски dropout —
:func:`ml.model.set_global_seed(ml.seed)`, перестановки train-батчей —
``np.random.default_rng(ml.seed)``. Формат каталога ``artifacts/{model_version}/``
— контракт ``m3-inference`` (решение 4 design.md): запись артефактов
(:func:`save_model_artifacts`) отделяется от обучения (:func:`train_model`),
чтобы свип окна учил без расплодивания каталогов моделей.
"""

import json
import math
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from common.config import MlCfg
from ml.dataset import DatasetScalers, SqliteRowSource, collect_examples
from ml.model import (
    Arch,
    LossComponents,
    MultitaskLoss,
    MultiTaskNet,
    build_model,
    deficit_pos_weight,
    set_global_seed,
)

ARTIFACTS_ROOT = Path("artifacts")
"""Корень артефактов обучения — относительный от cwd (решение 8 design.md)."""

WEIGHTS_FILE = "weights.pt"
SCALERS_FILE = "scalers.pkl"
LOSS_FILE = "loss.json"
METRICS_FILE = "metrics.json"
"""Четыре файла каталога версии модели — контракт ``m3-inference`` (задача 1.3)."""

SWEEP_BATCH_SIZE = 4096
"""Батч инференса свипа (без градиентов — можно крупнее обучающего)."""

_ROLES = ("train", "val", "test")
_COMPONENT_KEYS = ("total", "mse_requests", "mse_free", "bce")
"""Компоненты loss в порядке :class:`ml.model.LossComponents` — ключи loss.json."""


class PreparedData(NamedTuple):
    """Данные обучения по ролям: нормализованные массивы + scaler'ы и pos_weight.

    Окна и регрессионные таргеты — в нормализованном пространстве (масштабы
    обучены на train-роли); метки дефицита и ``raw_requests`` — сырые (метки
    не нормализуются, сырой спрос нужен свипу для MAE в физических заказах).
    """

    windows: dict[str, np.ndarray]  # role → (n, steps, 17), нормализовано
    zones: dict[str, np.ndarray]  # role → (n,), int64
    requests: dict[str, np.ndarray]  # role → (n, 2), нормализовано
    free_drivers: dict[str, np.ndarray]  # role → (n, 2), нормализовано
    deficit: dict[str, np.ndarray]  # role → (n, 2), 0.0/1.0
    raw_requests: dict[str, np.ndarray]  # role → (n, 2), физические заказы
    scalers: DatasetScalers
    pos_weight: float
    n_zones: int  # ёмкость эмбеддинга: max(zones) + 1 по списку зон города


class TrainResult(NamedTuple):
    """Итог обучения одной архитектуры: сеть эпохи лучшего val-loss, кривая
    обучения (компоненты train/val по эпохам), номер и компоненты лучшей эпохи,
    wall-время."""

    model: MultiTaskNet
    history: list[dict[str, object]]
    best_epoch: int
    best_val: dict[str, float]
    wall_sec: float


def prepare_training_data(
    db_path: str | Path,
    ml_cfg: MlCfg,
    *,
    zones: Sequence[int],
    start: datetime,
) -> PreparedData:
    """Собрать обучающие данные из БД эпохи (задача 1.1).

    Роли дней — ``ml.train_days/val_days/test_days`` от ``start`` (:func:
    `ml.dataset.day_role`), окна — ``ml.window_hours``. Scaler'ы обучаются на
    train-роли и нормализуют окна и регрессионные таргеты всех ролей;
    ``pos_weight`` BCE считается по train-меткам дефицита. Пустая роль train
    или val — ``ValueError`` до построения сети.
    """

    splits = collect_examples(
        SqliteRowSource(db_path),
        zones=zones,
        start=start,
        train_days=ml_cfg.train_days,
        val_days=ml_cfg.val_days,
        test_days=ml_cfg.test_days,
        window_hours=ml_cfg.window_hours,
    )
    for role in ("train", "val"):
        if splits[role].windows.shape[0] == 0:
            raise ValueError(
                f"роль {role} пуста (0 примеров) — проверьте БД {db_path} и границы"
                f" ролей ml.train_days={ml_cfg.train_days}, ml.val_days={ml_cfg.val_days}"
            )
    scalers = DatasetScalers.fit(
        splits["train"].windows, splits["train"].requests, splits["train"].free_drivers
    )
    windows = {role: scalers.transform_windows(splits[role].windows) for role in _ROLES}
    normalized = {
        role: scalers.transform_targets(splits[role].requests, splits[role].free_drivers)
        for role in _ROLES
    }
    return PreparedData(
        windows=windows,
        zones={
            role: np.array([zone_id for zone_id, _ in splits[role].keys], dtype=np.int64)
            for role in _ROLES
        },
        requests={role: normalized[role][0] for role in _ROLES},
        free_drivers={role: normalized[role][1] for role in _ROLES},
        deficit={role: splits[role].deficit for role in _ROLES},
        raw_requests={role: splits[role].requests for role in _ROLES},
        scalers=scalers,
        pos_weight=deficit_pos_weight(splits["train"].deficit),
        n_zones=max(zones) + 1,
    )


def _batches(n: int, batch_size: int, order: np.ndarray | None) -> list[np.ndarray]:
    """Срезы-батчи по ``order`` (``None`` — натуральный порядок без перемешивания)."""

    seq = np.arange(n) if order is None else order
    return [seq[start : start + batch_size] for start in range(0, n, batch_size)]


def _batch_tensors(data: PreparedData, role: str, idx: np.ndarray) -> tuple[torch.Tensor, ...]:
    """Конвертация numpy → tensor одного батча на границе вызывающего (решение 6
    design.md change ``m3-model``): окна float32, зоны int64, таргеты float32."""

    return (
        torch.from_numpy(data.windows[role][idx].astype(np.float32)),
        torch.from_numpy(data.zones[role][idx].astype(np.int64)),
        torch.from_numpy(data.requests[role][idx].astype(np.float32)),
        torch.from_numpy(data.free_drivers[role][idx].astype(np.float32)),
        torch.from_numpy(data.deficit[role][idx].astype(np.float32)),
    )


def _components_float(comps: LossComponents) -> np.ndarray:
    """Компоненты loss массивом в порядке :data:`_COMPONENT_KEYS`."""

    return np.array(
        [
            float(comps.total.detach()),
            float(comps.mse_requests.detach()),
            float(comps.mse_free.detach()),
            float(comps.bce.detach()),
        ]
    )


def _component_dict(sums: np.ndarray, n_batches: int) -> dict[str, float]:
    """Средние по батчам компоненты — dict для loss.json/history."""

    return {
        key: float(value)
        for key, value in zip(_COMPONENT_KEYS, (sums / n_batches).tolist(), strict=True)
    }


def _pass_epoch(
    model: MultiTaskNet,
    loss_fn: MultitaskLoss,
    data: PreparedData,
    role: str,
    batch_size: int,
    order: np.ndarray | None,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[np.ndarray, int]:
    """Один проход по роли: суммы компонент loss и число батчей.

    С ``optimizer`` — обучающий проход (zero_grad → backward → step), без —
    чистый подсчёт без градиентов (вызывается внутри ``torch.no_grad()``).
    """

    sums = np.zeros(len(_COMPONENT_KEYS))
    n_batches = 0
    for idx in _batches(data.windows[role].shape[0], batch_size, order):
        windows, zones, y_requests, y_free, y_deficit = _batch_tensors(data, role, idx)
        if optimizer is not None:
            optimizer.zero_grad()
        comps = loss_fn(model(windows, zones), y_requests, y_free, y_deficit)
        if optimizer is not None:
            comps.total.backward()
            optimizer.step()
        sums += _components_float(comps)
        n_batches += 1
    return sums, n_batches


def train_model(
    arch: Arch,
    data: PreparedData,
    ml_cfg: MlCfg,
    *,
    on_epoch: Callable[[int, float, float, int], None] | None = None,
) -> TrainResult:
    """Обучить одну архитектуру (задача 1.2).

    ``set_global_seed(ml.seed)`` → ``build_model`` → ручной батчинг
    перестановками из ``np.random.default_rng(ml.seed)``; оптимизатор Adam
    (``ml.lr``), loss — ``MultitaskLoss(ml.cls_weight, pos_weight train-роли)``.
    Ранняя остановка — по среднему по батчам val ``total`` без перемешивания:
    улучшение строгое, ``ml.patience`` эпох подряд без улучшения — стоп, кап
    ``ml.max_epochs``. Сеть в результате — глубокая копия state_dict эпохи
    лучшего val-loss (даже если после были хуже эпохи). ``on_epoch(epoch,
    train_total, val_total, patience_left)`` — колбэк прогресса CLI.
    """

    set_global_seed(ml_cfg.seed)
    model = build_model(arch, ml_cfg, n_zones=data.n_zones)
    loss_fn = MultitaskLoss(ml_cfg.cls_weight, data.pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=ml_cfg.lr)
    rng = np.random.default_rng(ml_cfg.seed)

    history: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_total = math.inf
    best_epoch = 0
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, ml_cfg.max_epochs + 1):
        perm = rng.permutation(data.windows["train"].shape[0])
        model.train()
        train_sums, train_batches = _pass_epoch(
            model, loss_fn, data, "train", ml_cfg.batch_size, perm, optimizer
        )
        model.eval()
        with torch.no_grad():
            val_sums, val_batches = _pass_epoch(
                model, loss_fn, data, "val", ml_cfg.batch_size, None, None
            )
        train_comps = _component_dict(train_sums, train_batches)
        val_comps = _component_dict(val_sums, val_batches)
        history.append({"epoch": epoch, "train": train_comps, "val": val_comps})

        val_total = val_comps["total"]
        if val_total < best_total:
            best_total, best_epoch, stale = val_total, epoch, 0
            best_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }
        else:
            stale += 1
        if on_epoch is not None:
            on_epoch(epoch, train_comps["total"], val_total, ml_cfg.patience - stale)
        if stale >= ml_cfg.patience:
            break

    if best_state is None:
        # NaN в данных или слишком большой ml.lr: ни одна эпоха не улучшила
        # val-loss (NaN < x всегда ложно) — обучение не состоялось.
        raise ValueError(
            "обучение не состоялось: ни одна эпоха не улучшила val-loss —"
            " проверьте данные на NaN и гиперпараметры ml.lr/ml.max_epochs"
        )
    model.load_state_dict(best_state)
    model.eval()
    best_val = history[best_epoch - 1]["val"]
    return TrainResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val=dict(best_val),
        wall_sec=time.perf_counter() - started,
    )


def model_version_for(arch: Arch, *, today: date | None = None) -> str:
    """``model_version = {arch}-v1-{локальная дата запуска}`` (решение 5 design.md)."""

    day = date.today() if today is None else today
    return f"{arch}-v1-{day.isoformat()}"


def save_model_artifacts(
    model_dir: str | Path,
    *,
    model_version: str,
    arch: Arch,
    model: MultiTaskNet,
    data: PreparedData,
    result: TrainResult,
    ml_cfg: MlCfg,
) -> None:
    """Записать каталог версии модели (задача 1.3): weights.pt ({arch,
    state_dict, ml_cfg: снимок}), scalers.pkl, loss.json (кривая по эпохам),
    metrics.json (итоги обучения). Существующий каталог — ``FileExistsError``
    до всякой записи: повтор в ту же дату — явный отказ (решение 4 design.md).
    """

    model_dir = Path(model_dir)
    if model_dir.exists():
        raise FileExistsError(f"каталог модели уже существует: {model_dir}")
    model_dir.mkdir(parents=True)
    torch.save(
        {"arch": arch, "state_dict": model.state_dict(), "ml_cfg": ml_cfg.model_dump()},
        model_dir / WEIGHTS_FILE,
    )
    data.scalers.save(model_dir / SCALERS_FILE)
    with open(model_dir / LOSS_FILE, "w", encoding="utf-8") as fh:
        json.dump(result.history, fh, ensure_ascii=False, indent=2)
    metrics = {
        "model_version": model_version,
        "arch": arch,
        "window_hours": ml_cfg.window_hours,
        "seed": ml_cfg.seed,
        "pos_weight": data.pos_weight,
        "n_train": int(data.windows["train"].shape[0]),
        "n_val": int(data.windows["val"].shape[0]),
        "best_epoch": result.best_epoch,
        "best_val": result.best_val,
        "ml_cfg": ml_cfg.model_dump(),
        "wall_sec": result.wall_sec,
    }
    with open(model_dir / METRICS_FILE, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)


def val_mae_requests(model: MultiTaskNet, data: PreparedData) -> float:
    """Val MAE прогноза спроса в физических заказах — критерий свипа окна
    (решение 6 design.md): прогнозы в нормализованном пространстве проходят
    обратное преобразование scaler'а таргетов, MAE против сырых val-таргетов,
    среднее по горизонтам.
    """

    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for idx in _batches(data.windows["val"].shape[0], SWEEP_BATCH_SIZE, None):
            windows, zones = _batch_tensors(data, "val", idx)[:2]
            preds.append(model(windows, zones).pred_requests.numpy())
    pred = np.concatenate(preds)
    phys = data.scalers.requests.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
    return float(np.abs(phys - data.raw_requests["val"]).mean(axis=0).mean())
