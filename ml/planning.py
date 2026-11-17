"""Прогноз вреда для планирования (change ``ml-planning-forecast``).

Отдельная capability от контракта ``forecasts`` (решение 1 design.md): цель —
вред (``unfulfilled > 0``) хотя бы в одном бакете окна ``[ts + H, ts + H +
window_min)``, горизонты H — ``ml.planning.horizons_min`` (дефолт 1 и 2 ч).
Признаки — то же окно 17 признаков ``ml-dataset`` (:func:`ml.dataset.build_window`),
поэтому анти-лик гарантии M3 наследуются без изменений; сети — те же туловища
:class:`ml.model.MLPTrunk` / :class:`ml.model.LSTMTrunk` с одной головой на
горизонты (решение 3 design.md).

Модуль содержит полный конвейер: сбор примеров по ролям дней
(:func:`collect_planning_examples`), обучение BCE с ``pos_weight`` по горизонтам
(:func:`train_planning`), артефакты версии
(:func:`save_planning_artifacts`), бейзлайны календаря и персистентности и
таблицу метрик (:func:`evaluate_planning`) и прогнозист живого контура
(:class:`PlanningPredictor`). Артефакты лежат в отдельном корне
``ml.planning.artifacts_root`` (дефолт ``artifacts/planning``), чтобы
``ml.infer.latest_artifact_dir`` M3 их не подхватывал.
"""

import json
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F

from common.config import AppConfig, MlCfg
from common.timeutil import add_minutes
from ml.dataset import (
    N_FEATURES,
    DatasetScalers,
    RowSource,
    SqliteRowSource,
    bucket_index,
    build_window,
    day_role,
    from_bucket_index,
)
from ml.model import TRUNK_OUTPUT_DIM, Arch, LSTMTrunk, MLPTrunk, set_global_seed
from ml.train import METRICS_FILE, SCALERS_FILE, WEIGHTS_FILE

PLANNING_INFER_BATCH_SIZE = 4096
"""Батч инференса планирования (без градиентов — крупнее обучающего, как в M3)."""

TOP_HARM_SHARE = 0.15
"""Бюджет вызова: метрика «доля вреда в топ-15 % скоров» (спека ml-planning)."""

TOP_HARM_PERCENTILE = 100.0 * (1.0 - TOP_HARM_SHARE)
"""Перцентиль порога бюджета: 85-й — выше него попадают лучшие 15 % скоров."""

PERSIST_WINDOW_MIN = 60
"""Окно персистентного бейзлайна: потери зоны за последний час до ts."""

PLANNING_MODELS = ("календарь", "персистентность", "mlp", "lstm")
"""Порядок моделей в таблице оценки (спека «Таблица оценки»)."""

_ROLES = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Датасет планирования
# ---------------------------------------------------------------------------


class PlanningTargets(NamedTuple):
    """Метки примера по горизонтам: 0/1 и суммы ``unfulfilled`` окон цели."""

    labels: tuple[int, ...]
    harm_sums: tuple[float, ...]


def planning_targets(
    source: RowSource,
    zone_id: int,
    ts: datetime,
    *,
    horizons_min: Sequence[int],
    window_min: int,
    agg_step_min: int = 5,
) -> PlanningTargets | None:
    """Метки ``harm_H`` окна ``[ts + H, ts + H + window_min)`` (задача 2.1).

    Метка горизонта H равна 1, если хотя бы в одном бакете окна
    ``unfulfilled > 0``; правило дефицита контракта (соотношение свободных
    водителей) намеренно не участвует (решение 2 design.md). Сумма
    ``unfulfilled`` окна возвращается рядом — она нужна метрике бюджета вызова.
    Нет хотя бы одного бакета окна (край эпохи или дыра) — ``None``: пример не
    выдаётся.
    """

    if not horizons_min:
        raise ValueError("горизонты планирования не могут быть пустыми")
    if window_min <= 0 or window_min % agg_step_min != 0:
        raise ValueError(
            f"window_min={window_min} обязан быть положительным и кратным шагу"
            f" агрегации {agg_step_min}"
        )
    steps = window_min // agg_step_min
    labels: list[int] = []
    harm_sums: list[float] = []
    for horizon in horizons_min:
        total = 0.0
        harm = False
        for k in range(steps):
            point = source.metrics_at(zone_id, add_minutes(ts, horizon + k * agg_step_min))
            if point is None:
                return None
            total += float(point.unfulfilled)
            if point.unfulfilled > 0:
                harm = True
        labels.append(1 if harm else 0)
        harm_sums.append(total)
    return PlanningTargets(tuple(labels), tuple(harm_sums))


class PlanningExamples(NamedTuple):
    """Примеры одной роли: окна признаков, метки, суммы вреда и ключи (зона, ts)."""

    windows: np.ndarray  # (n, steps, N_FEATURES), сырые признаки
    labels: np.ndarray  # (n, n_horizons), 0.0/1.0 — метки не нормализуются
    harm_sums: np.ndarray  # (n, n_horizons), суммы unfulfilled окон цели
    keys: list[tuple[int, datetime]]


def collect_planning_examples(
    source: RowSource,
    *,
    zones: Sequence[int],
    start: datetime,
    train_days: int,
    val_days: int,
    test_days: int,
    window_hours: float,
    horizons_min: Sequence[int],
    window_min: int,
    agg_step_min: int = 5,
) -> dict[str, PlanningExamples]:
    """Собрать примеры планирования по всем (зона, ts) эпохи по ролям дней.

    Окна — тем же :func:`ml.dataset.build_window`, что и датасет M3 (признаки
    совпадают для тех же пар (зона, ts)); роли дней — :func:`ml.dataset.day_role`.
    Первый ts — с полной историей окна, последний — такой, что последний бакет
    цели ``ts + max(H) + window_min − шаг`` ещё существует; примеры с дырой в
    цели пропускаются (спека «Край эпохи»). Пустая роль получает массивы формы
    (0, …), совместимые с fit/transform.
    """

    horizons = tuple(int(horizon) for horizon in horizons_min)
    if not horizons:
        raise ValueError("горизонты планирования не могут быть пустыми")
    steps = int(window_hours * 60 // agg_step_min)
    if steps <= 0:
        raise ValueError(f"window_hours={window_hours} даёт пустое окно истории")
    if window_min % agg_step_min != 0:
        raise ValueError(f"window_min={window_min} не кратен шагу агрегации {agg_step_min}")
    max_horizon = max(horizons)

    first_ts, last_ts = source.time_range()
    from_idx = bucket_index(first_ts, agg_step_min) + (steps - 1)
    until_idx = bucket_index(last_ts, agg_step_min) - (max_horizon + window_min) // agg_step_min + 1

    windows: dict[str, list[np.ndarray]] = {role: [] for role in _ROLES}
    labels: dict[str, list[tuple[int, ...]]] = {role: [] for role in _ROLES}
    harm_sums: dict[str, list[tuple[float, ...]]] = {role: [] for role in _ROLES}
    keys: dict[str, list[tuple[int, datetime]]] = {role: [] for role in _ROLES}

    for idx in range(from_idx, until_idx + 1):
        ts = from_bucket_index(idx, agg_step_min)
        role = day_role(ts, start, train_days=train_days, val_days=val_days, test_days=test_days)
        if role is None:
            continue
        for zone_id in zones:
            window = build_window(
                source, zone_id, ts, window_hours=window_hours, agg_step_min=agg_step_min
            )
            if window is None:
                continue
            targets = planning_targets(
                source,
                zone_id,
                ts,
                horizons_min=horizons,
                window_min=window_min,
                agg_step_min=agg_step_min,
            )
            if targets is None:
                continue
            windows[role].append(window)
            labels[role].append(targets.labels)
            harm_sums[role].append(targets.harm_sums)
            keys[role].append((zone_id, ts))

    n_horizons = len(horizons)
    return {
        role: PlanningExamples(
            windows=(
                np.stack(windows[role])
                if windows[role]
                else np.empty((0, steps, N_FEATURES), dtype=float)
            ),
            labels=(
                np.array(labels[role], dtype=float)
                if labels[role]
                else np.empty((0, n_horizons), dtype=float)
            ),
            harm_sums=(
                np.array(harm_sums[role], dtype=float)
                if harm_sums[role]
                else np.empty((0, n_horizons), dtype=float)
            ),
            keys=keys[role],
        )
        for role in _ROLES
    }


class PlanningData(NamedTuple):
    """Данные обучения планирования: нормализованные окна и сырые метки по ролям."""

    windows: dict[str, np.ndarray]  # role → (n, steps, 17), нормализовано
    zones: dict[str, np.ndarray]  # role → (n,), int64
    labels: dict[str, np.ndarray]  # role → (n, n_horizons), 0.0/1.0
    harm_sums: dict[str, np.ndarray]  # role → (n, n_horizons)
    keys: dict[str, list[tuple[int, datetime]]]  # role → ключи (зона, ts)
    scalers: DatasetScalers  # обучены только scaler признаков (fit на train)
    pos_weight: np.ndarray  # (n_horizons,), отношение негативов к позитивам
    n_zones: int  # ёмкость эмбеддинга зоны: max(zones) + 1


def prepare_planning_data(
    db_path: str | Path,
    ml_cfg: MlCfg,
    *,
    zones: Sequence[int],
    start: datetime,
) -> PlanningData:
    """Собрать данные обучения планирования из БД эпохи (задача 2.1).

    Роли дней и окно — ``ml`` конфига, горизонты и окно цели — подсекция
    ``ml.planning``. Нормализуются только признаки (``DatasetScalers.features``,
    fit на train): метки бинарны и не нормализуются, поэтому регрессионные
    scaler'ы M3 остаются необученными и в планировании не используются. Пустая
    роль train или val — ``ValueError`` до построения сети; ``pos_weight``
    считается по train-меткам каждого горизонта.
    """

    planning = ml_cfg.planning
    splits = collect_planning_examples(
        SqliteRowSource(db_path),
        zones=zones,
        start=start,
        train_days=ml_cfg.train_days,
        val_days=ml_cfg.val_days,
        test_days=ml_cfg.test_days,
        window_hours=ml_cfg.window_hours,
        horizons_min=planning.horizons_min,
        window_min=planning.window_min,
    )
    for role in ("train", "val"):
        if splits[role].windows.shape[0] == 0:
            raise ValueError(
                f"роль {role} пуста (0 примеров) — проверьте БД {db_path} и границы"
                f" ролей ml.train_days={ml_cfg.train_days}, ml.val_days={ml_cfg.val_days}"
            )
    scalers = DatasetScalers()
    scalers.features.fit(splits["train"].windows.reshape(-1, N_FEATURES))
    return PlanningData(
        windows={role: scalers.transform_windows(splits[role].windows) for role in _ROLES},
        zones={
            role: np.array([zone_id for zone_id, _ in splits[role].keys], dtype=np.int64)
            for role in _ROLES
        },
        labels={role: splits[role].labels for role in _ROLES},
        harm_sums={role: splits[role].harm_sums for role in _ROLES},
        keys={role: splits[role].keys for role in _ROLES},
        scalers=scalers,
        pos_weight=planning_pos_weight(splits["train"].labels, planning.horizons_min),
        n_zones=max(zones) + 1,
    )


def planning_pos_weight(labels: np.ndarray, horizons_min: Sequence[int]) -> np.ndarray:
    """``pos_weight`` BCE по каждому горизонту: ``(1 − p) / p`` по train-меткам.

    ``p`` — доля вреда горизонта на train. Вырожденная доля (0 или 1) —
    ``ValueError``: сигнал ошибки данных/конфигурации, а не тихий fallback
    (прецедент :func:`ml.model.deficit_pos_weight`).
    """

    labels = np.asarray(labels, dtype=float)
    if labels.ndim != 2 or labels.shape[1] != len(horizons_min):
        raise ValueError(
            f"метки обязаны иметь форму (n, {len(horizons_min)}), получено {labels.shape}"
        )
    if labels.shape[0] == 0:
        raise ValueError("нет train-примеров: pos_weight не определён")
    shares = labels.mean(axis=0)
    weights = np.empty(len(horizons_min), dtype=float)
    for k, (horizon, share) in enumerate(zip(horizons_min, shares, strict=True)):
        share = float(share)
        if not 0.0 < share < 1.0:
            raise ValueError(
                f"доля вреда горизонта {horizon} на train равна {share}:"
                f" pos_weight = (1 − p)/p не определён"
            )
        weights[k] = (1.0 - share) / share
    return weights


# ---------------------------------------------------------------------------
# Сеть планирования и обучение
# ---------------------------------------------------------------------------


class PlanningOutput(NamedTuple):
    """Выход сети планирования: логиты и вероятности формы (n, n_horizons)."""

    logits: Tensor
    prob: Tensor


class PlanningNet(nn.Module):
    """Туловище MLP/LSTM M3 + одна голова ``Linear(TRUNK_OUTPUT_DIM, n_horizons)``.

    Порядок колонок — ``ml.planning.horizons_min``; на выходе — логиты для
    численно стабильного BCE-with-logits и вероятности после сигмоиды (та же
    конвенция двух представлений, что у :class:`ml.model.MultiTaskNet`).
    """

    def __init__(self, trunk: nn.Module, n_horizons: int) -> None:
        super().__init__()
        if n_horizons < 1:
            raise ValueError(f"n_horizons обязан быть >= 1, получено {n_horizons}")
        self.trunk = trunk
        self.head = nn.Linear(TRUNK_OUTPUT_DIM, n_horizons)

    def forward(self, windows: Tensor, zone_ids: Tensor) -> PlanningOutput:
        """Окна ``(n, steps, 17)`` и зоны ``(n,)`` → логиты и вероятности."""

        logits = self.head(self.trunk(windows, zone_ids))
        return PlanningOutput(logits=logits, prob=torch.sigmoid(logits))


def build_planning_model(arch: Arch, cfg: MlCfg, n_zones: int) -> PlanningNet:
    """Собрать ``PlanningNet`` вокруг туловища выбранной архитектуры.

    ``n_zones`` — ёмкость обучаемого эмбеддинга зоны; гиперпараметры туловища —
    секция ``ml`` (те же, что у M3). Seed фиксирует вызывающий через
    :func:`ml.model.set_global_seed` перед построением.
    """

    if arch == "mlp":
        trunk: nn.Module = MLPTrunk(cfg, n_zones)
    elif arch == "lstm":
        trunk = LSTMTrunk(cfg, n_zones)
    else:
        raise ValueError(f"неизвестная архитектура {arch!r}: ожидается 'mlp' или 'lstm'")
    return PlanningNet(trunk, len(cfg.planning.horizons_min))


class PlanningTrainResult(NamedTuple):
    """Итог обучения: сеть лучшей эпохи, кривая по эпохам, номер и loss лучшей эпохи."""

    model: PlanningNet
    history: list[dict[str, object]]
    best_epoch: int
    best_val: dict[str, float]


def _batches(n: int, batch_size: int, order: np.ndarray | None) -> list[np.ndarray]:
    """Срезы-батчи по ``order`` (``None`` — натуральный порядок без перемешивания)."""

    seq = np.arange(n) if order is None else order
    return [seq[start : start + batch_size] for start in range(0, n, batch_size)]


def _batch_tensors(data: PlanningData, role: str, idx: np.ndarray) -> tuple[Tensor, Tensor, Tensor]:
    """numpy → tensor одного батча на границе вызывающего (окна float32, зоны int64)."""

    return (
        torch.from_numpy(data.windows[role][idx].astype(np.float32)),
        torch.from_numpy(data.zones[role][idx].astype(np.int64)),
        torch.from_numpy(data.labels[role][idx].astype(np.float32)),
    )


def _pass_epoch(
    model: PlanningNet,
    data: PlanningData,
    role: str,
    batch_size: int,
    order: np.ndarray | None,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """Один проход по роли: средний BCE батчей (с ``pos_weight`` по горизонтам)."""

    pos_weight = torch.as_tensor(data.pos_weight, dtype=torch.float32)
    losses: list[float] = []
    for idx in _batches(data.windows[role].shape[0], batch_size, order):
        windows, zones, labels = _batch_tensors(data, role, idx)
        if optimizer is not None:
            optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            model(windows, zones).logits, labels, pos_weight=pos_weight
        )
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


def train_planning(
    arch: Arch,
    data: PlanningData,
    ml_cfg: MlCfg,
    *,
    on_epoch: Callable[[int, float, float, int], None] | None = None,
) -> PlanningTrainResult:
    """Обучить сеть планирования (задача 3.1): BCE, ранняя остановка по val.

    ``set_global_seed(ml.seed)`` → ``build_planning_model`` → батчи
    перестановками ``np.random.default_rng(ml.seed)``, Adam (``ml.lr``),
    loss — BCE-with-logits с ``pos_weight`` по горизонтам. Ранняя остановка —
    по среднему val BCE (улучшение строгое, ``ml.patience`` эпох подряд без
    улучшения, кап ``ml.max_epochs``); в результате — состояние лучшей эпохи.
    ``on_epoch(epoch, train_bce, val_bce, patience_left)`` — колбэк прогресса CLI.
    """

    set_global_seed(ml_cfg.seed)
    model = build_planning_model(arch, ml_cfg, n_zones=data.n_zones)
    optimizer = torch.optim.Adam(model.parameters(), lr=ml_cfg.lr)
    rng = np.random.default_rng(ml_cfg.seed)

    history: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_total = math.inf
    best_epoch = 0
    stale = 0

    for epoch in range(1, ml_cfg.max_epochs + 1):
        perm = rng.permutation(data.windows["train"].shape[0])
        model.train()
        train_bce = _pass_epoch(model, data, "train", ml_cfg.batch_size, perm, optimizer)
        model.eval()
        with torch.no_grad():
            val_bce = _pass_epoch(model, data, "val", ml_cfg.batch_size, None, None)
        entry = {
            "epoch": epoch,
            "train": {"bce": train_bce},
            "val": {"bce": val_bce},
        }
        history.append(entry)

        if val_bce < best_total:
            best_total, best_epoch, stale = val_bce, epoch, 0
            best_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }
        else:
            stale += 1
        if on_epoch is not None:
            on_epoch(epoch, train_bce, val_bce, ml_cfg.patience - stale)
        if stale >= ml_cfg.patience:
            break

    if best_state is None:
        # NaN в данных или слишком большой ml.lr: ни одна эпоха не улучшила val.
        raise ValueError(
            "обучение не состоялось: ни одна эпоха не улучшила val-loss —"
            " проверьте данные на NaN и гиперпараметры ml.lr/ml.max_epochs"
        )
    model.load_state_dict(best_state)
    model.eval()
    return PlanningTrainResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val=dict(history[best_epoch - 1]["val"]),
    )


# ---------------------------------------------------------------------------
# Артефакты версии
# ---------------------------------------------------------------------------


class PlanningArtifact(NamedTuple):
    """Восстановленный артефакт: сеть в ``eval``-режиме, scaler'ы и снимок ``ml``."""

    model_version: str
    arch: Arch
    ml_cfg: MlCfg
    model: PlanningNet
    scalers: DatasetScalers


def latest_planning_artifact_dir(artifacts_root: str | Path, family: str) -> Path:
    """Самый свежий каталог семейства: лексический максимум ``{family}-v1-*``."""

    candidates = sorted(
        path for path in Path(artifacts_root).glob(f"{family}-v1-*") if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(
            f"нет каталогов артефактов {family}-v1-* в {artifacts_root} — прогоните"
            f" python -m ml plan-train --model {family}"
        )
    return candidates[-1]


def load_planning_artifact(model_dir: str | Path, *, n_zones: int) -> PlanningArtifact:
    """Восстановить сеть планирования из каталога артефакта версии.

    Архитектура, окно и горизонты — из снимка ``ml_cfg`` в ``weights.pt`` (не из
    живого конфига: защита от дрейфа, прецедент ``ml.infer``); нормализация —
    scaler'ы каталога; ``model_version`` — имя каталога. ``n_zones`` (ёмкость
    эмбеддинга) — параметр окружения, берётся из города конфига.
    """

    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"каталог артефакта не найден: {model_dir}")
    blob = torch.load(model_dir / WEIGHTS_FILE, weights_only=True)
    ml_cfg = MlCfg(**blob["ml_cfg"])
    model = build_planning_model(blob["arch"], ml_cfg, n_zones=n_zones)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return PlanningArtifact(
        model_version=model_dir.name,
        arch=blob["arch"],
        ml_cfg=ml_cfg,
        model=model,
        scalers=DatasetScalers.load(model_dir / SCALERS_FILE),
    )


def predict_planning_windows(
    artifact: PlanningArtifact, windows: np.ndarray, zones: Sequence[int]
) -> np.ndarray:
    """Батч-прогон сети планирования на сырых окнах: (n, n_horizons) вероятностей.

    Окна нормализуются scaler'ами артефакта; прогноз — батчи без градиентов в
    ``eval``-режиме сети (общий путь walk-forward оценки и живого контура).
    """

    n = windows.shape[0]
    zone_arr = np.asarray(zones, dtype=np.int64)
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for lo in range(0, n, PLANNING_INFER_BATCH_SIZE):
            batch = slice(lo, lo + PLANNING_INFER_BATCH_SIZE)
            windows_t = torch.from_numpy(
                artifact.scalers.transform_windows(windows[batch]).astype(np.float32)
            )
            zones_t = torch.from_numpy(zone_arr[batch])
            probs.append(artifact.model(windows_t, zones_t).prob.numpy())
    if not probs:
        return np.empty((0, len(artifact.ml_cfg.planning.horizons_min)))
    return np.concatenate(probs)


def _role_metrics(
    model: PlanningNet,
    labels: np.ndarray,
    harm_sums: np.ndarray,
    windows: np.ndarray,
    zones: np.ndarray,
    horizons_min: Sequence[int],
) -> dict[str, dict[str, float]]:
    """AP, ROC-AUC и доля вреда топ-15 % сети на роли — для ``metrics.json``.

    ``windows`` — уже нормализованные окна роли (как в :class:`PlanningData`).
    """

    if windows.shape[0] == 0:
        return {}
    probs = _predict_normalized(model, windows, zones)
    return {
        str(horizon): score_metrics(labels[:, k], probs[:, k], harm_sums[:, k])
        for k, horizon in enumerate(horizons_min)
    }


def _predict_normalized(model: PlanningNet, windows: np.ndarray, zones: np.ndarray) -> np.ndarray:
    """Вероятности (n, n_horizons) по уже нормализованным окнам."""

    probs: list[np.ndarray] = []
    with torch.no_grad():
        for lo in range(0, windows.shape[0], PLANNING_INFER_BATCH_SIZE):
            batch = slice(lo, lo + PLANNING_INFER_BATCH_SIZE)
            windows_t = torch.from_numpy(windows[batch].astype(np.float32))
            zones_t = torch.from_numpy(np.asarray(zones[batch], dtype=np.int64))
            probs.append(model(windows_t, zones_t).prob.numpy())
    if not probs:
        return np.empty((0, 0))
    return np.concatenate(probs)


def save_planning_artifacts(
    model_dir: str | Path,
    *,
    model_version: str,
    family: Arch,
    model: PlanningNet,
    data: PlanningData,
    result: PlanningTrainResult,
    ml_cfg: MlCfg,
) -> None:
    """Записать каталог артефакта (задача 3.1): weights.pt, scalers.pkl, metrics.json.

    ``weights.pt`` хранит архитектуру, ``state_dict`` и снимок секции ``ml``
    (включая ``planning``); ``metrics.json`` — метрики val и test (AP, ROC-AUC,
    доля вреда в топ-15 %) по горизонтам и итоги обучения. Зависимых от времени
    полей (wall time) в ``metrics.json`` нет: два обучения с одним seed дают
    побайтово равный файл. Существующий каталог — ``FileExistsError`` до записи.
    """

    model_dir = Path(model_dir)
    if model_dir.exists():
        raise FileExistsError(f"каталог артефакта уже существует: {model_dir}")
    model_dir.mkdir(parents=True)
    torch.save(
        {"arch": family, "state_dict": model.state_dict(), "ml_cfg": ml_cfg.model_dump()},
        model_dir / WEIGHTS_FILE,
    )
    data.scalers.save(model_dir / SCALERS_FILE)
    horizons = list(ml_cfg.planning.horizons_min)
    metrics: dict[str, object] = {
        "model_version": model_version,
        "arch": family,
        "horizons_min": horizons,
        "window_min": ml_cfg.planning.window_min,
        "window_hours": ml_cfg.window_hours,
        "seed": ml_cfg.seed,
        "pos_weight": [float(weight) for weight in data.pos_weight],
        "n_train": int(data.windows["train"].shape[0]),
        "n_val": int(data.windows["val"].shape[0]),
        "n_test": int(data.windows["test"].shape[0]),
        "best_epoch": result.best_epoch,
        "best_val": result.best_val,
        "ml_cfg": ml_cfg.model_dump(),
    }
    for role in ("val", "test"):
        metrics[role] = _role_metrics(
            model,
            data.labels[role],
            data.harm_sums[role],
            data.windows[role],
            data.zones[role],
            horizons,
        )
    with open(model_dir / METRICS_FILE, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Метрики и бейзлайны
# ---------------------------------------------------------------------------


def score_metrics(
    labels: np.ndarray, scores: np.ndarray, harm_sums: np.ndarray
) -> dict[str, float]:
    """AP, ROC-AUC и доля вреда топ-15 % для одного горизонта.

    Доля вреда — сумма ``unfulfilled`` окон цели у примеров, чей скор выше
    85-го перцентиля скоров, делённая на общую сумму вреда; при нулевом общем
    вреде — 0.0. Вырожденные случаи (нет примеров, один класс меток) дают
    AP = ROC-AUC = 0.0, без NaN и исключений (прецедент :mod:`ml.metrics`).
    """

    labels = np.asarray(labels, dtype=float).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    harms = np.asarray(harm_sums, dtype=float).ravel()
    if not (labels.size == scores.size == harms.size):
        raise ValueError(
            f"метки, скоры и суммы вреда обязаны совпадать по длине, получено"
            f" {labels.size}, {scores.size}, {harms.size}"
        )
    if labels.size == 0 or np.all(labels == labels[0]):
        ap = roc_auc = 0.0
    else:
        ap = float(average_precision_score(labels, scores))
        roc_auc = float(roc_auc_score(labels, scores))
    total = float(harms.sum())
    harm_top15 = 0.0
    if total > 0.0:
        threshold = float(np.percentile(scores, TOP_HARM_PERCENTILE))
        harm_top15 = float(harms[scores > threshold].sum() / total)
    return {"ap": ap, "roc_auc": roc_auc, "harm_top15": harm_top15}


def day_type_index(ts: datetime) -> int:
    """Тип дня календаря планирования: 0 — пн–чт, 1 — пт, 2 — сб, 3 — вс."""

    dow = ts.weekday()
    return 0 if dow <= 3 else dow - 3


def half_hour_index(ts: datetime) -> int:
    """Номер получаса суток момента ``ts`` (0…47)."""

    return ts.hour * 2 + ts.minute // 30


class CalendarBaseline:
    """Календарь (решение 4 design.md): частота вреда в «зона × тип дня × получас».

    Ячейки обучаются на train-примерах отдельно по каждому горизонту; пустая
    ячейка заменяется общей частотой вреда train-роли (спека ml-planning).
    """

    def __init__(
        self,
        sums: dict[tuple[int, int, int], np.ndarray],
        counts: dict[tuple[int, int, int], int],
        fallback: np.ndarray,
    ) -> None:
        self._sums = sums
        self._counts = counts
        self._fallback = fallback

    @classmethod
    def fit(cls, keys: Sequence[tuple[int, datetime]], labels: np.ndarray) -> "CalendarBaseline":
        """Обучить таблицу частот по train-примерам ``(зона, ts) → метки``."""

        labels = np.asarray(labels, dtype=float)
        if labels.ndim != 2:
            raise ValueError(f"метки обязаны иметь форму (n, n_horizons), получено {labels.shape}")
        n_horizons = labels.shape[1]
        sums: dict[tuple[int, int, int], np.ndarray] = {}
        counts: dict[tuple[int, int, int], int] = {}
        for (zone_id, ts), row in zip(keys, labels, strict=True):
            cell = (int(zone_id), day_type_index(ts), half_hour_index(ts))
            if cell not in sums:
                sums[cell] = np.zeros(n_horizons, dtype=float)
                counts[cell] = 0
            sums[cell] += row
            counts[cell] += 1
        fallback = labels.mean(axis=0) if labels.shape[0] else np.zeros(n_horizons, dtype=float)
        return cls(sums, counts, fallback)

    def score(self, zone_id: int, ts: datetime) -> np.ndarray:
        """Скор ячейки момента ``ts``; пустая ячейка — общая частота train."""

        cell = (int(zone_id), day_type_index(ts), half_hour_index(ts))
        count = self._counts.get(cell, 0)
        if count == 0:
            return self._fallback.copy()
        return self._sums[cell] / count


def persistence_score(
    source: RowSource, zone_id: int, ts: datetime, *, agg_step_min: int = 5
) -> float:
    """Персистентность: число потерь зоны за последний час до ``ts`` включительно.

    Показывает, насколько быстро обесценивается текущее состояние (design.md,
    решение 4). Отсутствующие бакеты часа в сумму не входят; читаются только
    бакеты ≤ ts — гарантия отсутствия утечки будущего.
    """

    steps = PERSIST_WINDOW_MIN // agg_step_min
    total = 0.0
    for k in range(steps):
        point = source.metrics_at(zone_id, add_minutes(ts, -k * agg_step_min))
        if point is not None:
            total += float(point.unfulfilled)
    return total


def evaluate_planning(
    source: RowSource,
    ml_cfg: MlCfg,
    *,
    zones: Sequence[int],
    start: datetime,
    artifacts_root: str | Path,
) -> list[dict[str, object]]:
    """Таблица оценки планирования на test-роли эпохи (задача 4.1).

    Бейзлайны: календарь (fit на train-днях) и персистентность (потери за
    последний час); сети — свежайшие артефакты ``mlp`` и ``lstm`` из
    ``artifacts_root``. Для каждого горизонта — строка каждой из четырёх
    моделей с AP, ROC-AUC и долей вреда в топ-15 % скоров. Расхождение
    горизонтов снимка артефакта и живого конфига — ``ValueError`` (колонки
    таблицы разошлись бы).
    """

    plans = list(ml_cfg.planning.horizons_min)
    splits = collect_planning_examples(
        source,
        zones=zones,
        start=start,
        train_days=ml_cfg.train_days,
        val_days=ml_cfg.val_days,
        test_days=ml_cfg.test_days,
        window_hours=ml_cfg.window_hours,
        horizons_min=plans,
        window_min=ml_cfg.planning.window_min,
    )
    test = splits["test"]
    if test.windows.shape[0] == 0:
        raise ValueError(
            "роль test пуста (0 примеров) — проверьте БД эпохи и границы ролей"
            f" ml.train_days={ml_cfg.train_days}, ml.test_days={ml_cfg.test_days}"
        )

    calendar = CalendarBaseline.fit(splits["train"].keys, splits["train"].labels)
    scores: dict[str, np.ndarray] = {
        "календарь": np.stack([calendar.score(zone_id, ts) for zone_id, ts in test.keys]),
        "персистентность": np.repeat(
            np.array(
                [persistence_score(source, zone_id, ts) for zone_id, ts in test.keys],
                dtype=float,
            )[:, np.newaxis],
            len(plans),
            axis=1,
        ),
    }

    n_zones = max(zones) + 1
    for family in ("mlp", "lstm"):
        artifact = load_planning_artifact(
            latest_planning_artifact_dir(artifacts_root, family), n_zones=n_zones
        )
        snapshot = list(artifact.ml_cfg.planning.horizons_min)
        if snapshot != plans:
            raise ValueError(
                f"горизонты артефакта {artifact.model_version} {snapshot} расходятся"
                f" с ml.planning.horizons_min {plans} — переобучите сети"
            )
        scores[family] = predict_planning_windows(
            artifact, test.windows, [zone_id for zone_id, _ in test.keys]
        )

    rows: list[dict[str, object]] = []
    for k, horizon in enumerate(plans):
        for name in PLANNING_MODELS:
            metrics = score_metrics(test.labels[:, k], scores[name][:, k], test.harm_sums[:, k])
            rows.append(
                {
                    "model": name,
                    "horizon_min": horizon,
                    "ap": metrics["ap"],
                    "roc_auc": metrics["roc_auc"],
                    "harm_top15": metrics["harm_top15"],
                }
            )
    return rows


def render_planning_table(rows: Sequence[Mapping[str, object]]) -> str:
    """Печатная таблица оценки: строка на «модель × горизонт».

    Колонки: AP, ROC_AUC, harm_top15 (формат ``%.3f``); порядок строк — входной
    (горизонт, затем модели). Ширина колонки модели — по самому длинному имени.
    """

    name_width = max([len("model")] + [len(str(row.get("model", ""))) for row in rows])
    header = (
        "model".ljust(name_width)
        + "  "
        + "horizon".rjust(7)
        + "  "
        + "AP".rjust(7)
        + "  "
        + "ROC_AUC".rjust(7)
        + "  "
        + "harm_top15".rjust(10)
    )
    lines = [header]
    for row in rows:
        lines.append(
            str(row.get("model", "")).ljust(name_width)
            + "  "
            + str(row.get("horizon_min", "")).rjust(7)
            + "  "
            + f"{float(row.get('ap', 0.0)):>7.3f}"
            + "  "
            + f"{float(row.get('roc_auc', 0.0)):>7.3f}"
            + "  "
            + f"{float(row.get('harm_top15', 0.0)):>10.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Прогнозист живого контура
# ---------------------------------------------------------------------------


class PlanningPredictor:
    """Прогнозист планирования: сеть семейства + окно признаков над источником.

    Стежайший артефакт семейства берётся из ``ml.planning.artifacts_root``
    (или явного ``artifacts_root``); горизонт H обязан входить в горизонты
    снимка артефакта — иначе ``ValueError`` при построении с именем H и
    списком горизонтов. Окно строится тем же :func:`ml.dataset.build_window`,
    что при обучении (анти-лик гарантии M3 наследуются), неполное окно
    (прогрев) — ``None``.
    """

    def __init__(
        self,
        source: RowSource,
        cfg: AppConfig,
        *,
        family: str,
        horizon_min: int,
        artifacts_root: str | Path | None = None,
    ) -> None:
        root = cfg.ml.planning.artifacts_root if artifacts_root is None else artifacts_root
        model_dir = latest_planning_artifact_dir(root, family)
        n_zones = max(zone.id for zone in cfg.city.zones) + 1
        artifact = load_planning_artifact(model_dir, n_zones=n_zones)
        horizons = list(artifact.ml_cfg.planning.horizons_min)
        if horizon_min not in horizons:
            raise ValueError(
                f"horizon_min={horizon_min} вне горизонтов артефакта {horizons}"
                f" ({artifact.model_version})"
            )
        self._artifact = artifact
        self._source = source
        self._horizon_idx = horizons.index(horizon_min)

    def predict(self, zone_id: int, ts: datetime) -> float | None:
        """Вероятность вреда горизонта H для (зона, ts) или ``None`` без окна."""

        window = build_window(
            self._source, zone_id, ts, window_hours=self._artifact.ml_cfg.window_hours
        )
        if window is None:
            return None
        probs = predict_planning_windows(self._artifact, window[np.newaxis], [zone_id])
        return float(probs[0, self._horizon_idx])


__all__ = [
    "CalendarBaseline",
    "PlanningArtifact",
    "PlanningData",
    "PlanningExamples",
    "PlanningNet",
    "PlanningOutput",
    "PlanningPredictor",
    "PlanningTargets",
    "PlanningTrainResult",
    "build_planning_model",
    "collect_planning_examples",
    "day_type_index",
    "evaluate_planning",
    "half_hour_index",
    "latest_planning_artifact_dir",
    "load_planning_artifact",
    "persistence_score",
    "planning_pos_weight",
    "planning_targets",
    "predict_planning_windows",
    "prepare_planning_data",
    "render_planning_table",
    "save_planning_artifacts",
    "score_metrics",
    "train_planning",
]
