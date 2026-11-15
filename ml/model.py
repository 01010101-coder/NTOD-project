"""Модели вехи M3: туловища MLP/LSTM, multi-task головки, loss и seed (§5 плана M3).

Контракт туловища (решение 1 design.md change ``m3-model``) — сигнатура, а не
абстрактный класс: оба туловища — ``nn.Module`` с ``forward(окна, зоны) → R``
фиксированной размерности :data:`TRUNK_OUTPUT_DIM`; взаимозаменяемость
проверяется параметризованным тестом, а не иерархией типов (прецедент —
лёгкие интерфейсы ``ml-dataset``). Сеть предсказывает в нормализованном
пространстве таргетов (scaler'ы ``ml-dataset``, fit на train); конвертация
numpy → tensor — на границе вызывающего, один раз в батче (решение 6 design.md).
Цикл обучения с ранней остановкой — ``m3-train-cli``, запись ``forecasts`` —
``m3-inference``: этот модуль даёт им всё без доработок модели.
"""

from typing import Literal, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from common.config import MlCfg
from ml.dataset import N_FEATURES, RET_HORIZONS_MIN

TRUNK_OUTPUT_DIM = 128
"""Размер представления туловища R: «окно + зона → R» (план M3 §5); последний
слой ``ml.mlp_hidden`` обязан равняться ему — контракт общих головок."""

AGG_STEP_MIN = 5
"""Шаг агрегации датасета в минутах; формула шагов окна совпадает с ``ml.dataset``."""

Arch = Literal["mlp", "lstm"]
"""Архитектура туловища; смена архитектуры меняет только туловище (спека)."""


def set_global_seed(seed: int) -> None:
    """Фиксировать torch- и numpy-RNG (решение 7 design.md): вызывается перед
    построением сети; одинаковый seed даёт идентичные инициализации весов и
    одинаковые маски dropout при одинаковой последовательности вызовов."""

    torch.manual_seed(seed)
    np.random.seed(seed)


def _window_steps(cfg: MlCfg) -> int:
    """Число шагов окна: та же формула, что в ``ml.dataset.build_window``."""

    return int(cfg.window_hours * 60 // AGG_STEP_MIN)


def _check_window(windows: Tensor, steps: int | None) -> None:
    """Форма окна: ``(n, steps, N_FEATURES)``; ``steps is None`` — не проверяется
    (LSTM не зависит от числа шагов при построении)."""

    if (
        windows.dim() != 3
        or windows.shape[2] != N_FEATURES
        or (steps is not None and windows.shape[1] != steps)
    ):
        expected = f"(n, {steps if steps is not None else 'steps'}, {N_FEATURES})"
        raise ValueError(f"окно обязано иметь форму {expected}, получено {tuple(windows.shape)}")


class ModelOutput(NamedTuple):
    """Выход сети на батч: колонки всех величин в порядке
    :data:`RET_HORIZONS_MIN` (15, 30) — контракт ``forecasts``.

    Регрессии — в нормализованном пространстве таргетов (обратное преобразование
    scaler'ов при инференсе — ``m3-inference``). Дефицит — две репрезентации
    осознанно (решение 4 design.md): логиты для численно стабильного
    BCE-with-logits, вероятности после сигмоиды для инференса/метрик.
    """

    pred_requests: Tensor
    pred_free_drivers: Tensor
    deficit_logits: Tensor
    deficit_prob: Tensor


class MLPTrunk(nn.Module):
    """MLP-туловище (основная архитектура, тема курса): flatten окна + конкат
    эмбеддинга зоны → плотные слои по ``ml.mlp_hidden`` с ReLU/Dropout → R.

    При дефолте ``[256, 128]`` — ``Linear(steps·17 + 8 → 256) → ReLU →
    Dropout → Linear(256 → 128)`` без лишней финальной проекции (решение 2
    design.md). Последний элемент ``mlp_hidden`` обязан равняться
    :data:`TRUNK_OUTPUT_DIM` — иначе ``ValueError``: контракт общих головок.
    """

    def __init__(self, cfg: MlCfg, n_zones: int) -> None:
        super().__init__()
        hidden = list(cfg.mlp_hidden)
        if not hidden or hidden[-1] != TRUNK_OUTPUT_DIM:
            raise ValueError(
                f"последний слой ml.mlp_hidden обязан равняться TRUNK_OUTPUT_DIM"
                f" ({TRUNK_OUTPUT_DIM}), получено {cfg.mlp_hidden}"
            )
        self._steps = _window_steps(cfg)
        self.zone_embed = nn.Embedding(n_zones, cfg.zone_embed_dim)
        in_dim = self._steps * N_FEATURES + cfg.zone_embed_dim
        layers: list[nn.Module] = []
        prev_dim = in_dim
        for dim in hidden[:-1]:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(cfg.dropout)])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, hidden[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, windows: Tensor, zone_ids: Tensor) -> Tensor:
        """Окна ``(n, steps, 17)`` float32 и зоны ``(n,)`` int64 → R ``(n, 128)``."""

        _check_window(windows, self._steps)
        flat = windows.flatten(start_dim=1)
        emb = self.zone_embed(zone_ids)
        return self.net(torch.cat((flat, emb), dim=1))


class LSTMTrunk(nn.Module):
    """LSTM-туловище (сравнение, буква ТЗ): эмбеддинг зоны конкатенируется к
    каждому шагу окна → ``LSTM(hidden=ml.lstm_hidden, layers=ml.lstm_layers,
    batch_first=True)`` → последнее скрытое состояние → ``Linear(lstm_hidden,
    128)`` (решение 3 design.md); нелинейности после проекции нет — «сырое» R.
    """

    def __init__(self, cfg: MlCfg, n_zones: int) -> None:
        super().__init__()
        self.zone_embed = nn.Embedding(n_zones, cfg.zone_embed_dim)
        self.lstm = nn.LSTM(
            input_size=N_FEATURES + cfg.zone_embed_dim,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(cfg.lstm_hidden, TRUNK_OUTPUT_DIM)

    def forward(self, windows: Tensor, zone_ids: Tensor) -> Tensor:
        """Окна ``(n, steps, 17)`` float32 и зоны ``(n,)`` int64 → R ``(n, 128)``."""

        _check_window(windows, None)
        emb = self.zone_embed(zone_ids)  # (n, zone_embed_dim)
        x = torch.cat((windows, emb.unsqueeze(1).expand(-1, windows.shape[1], -1)), dim=2)
        out, _ = self.lstm(x)  # (n, steps, lstm_hidden)
        return self.proj(out[:, -1, :])


class MultiTaskNet(nn.Module):
    """Сеть вокруг туловища: общие головки ``Linear(R, 2)`` ×3 — спрос, сток и
    логиты дефицита (**одна** BCE-головка с двумя выходами, решение 4 design.md).
    Порядок колонок всех выходов — :data:`RET_HORIZONS_MIN` (15, 30)."""

    def __init__(self, trunk: nn.Module) -> None:
        super().__init__()
        self.trunk = trunk
        self.head_requests = nn.Linear(TRUNK_OUTPUT_DIM, len(RET_HORIZONS_MIN))
        self.head_free_drivers = nn.Linear(TRUNK_OUTPUT_DIM, len(RET_HORIZONS_MIN))
        self.head_deficit = nn.Linear(TRUNK_OUTPUT_DIM, len(RET_HORIZONS_MIN))

    def forward(self, windows: Tensor, zone_ids: Tensor) -> ModelOutput:
        """Окна и зоны туловищу → R; головки → все шесть величин контракта."""

        r = self.trunk(windows, zone_ids)
        logits = self.head_deficit(r)
        return ModelOutput(
            pred_requests=self.head_requests(r),
            pred_free_drivers=self.head_free_drivers(r),
            deficit_logits=logits,
            deficit_prob=torch.sigmoid(logits),
        )


class LossComponents(NamedTuple):
    """Компоненты loss раздельно (решение 5 design.md) — для кривых обучения и
    ``metrics.json``; ``total`` — дифференцируемая взвешенная сумма."""

    total: Tensor
    mse_requests: Tensor
    mse_free: Tensor
    bce: Tensor


def deficit_pos_weight(train_deficit: np.ndarray) -> float:
    """``pos_weight`` BCE: отношение долей недефицитных к дефицитным бакетам
    train-роли — ``(1 − p) / p`` по фактической доле дефицита; вырожденные доли
    (мир без дефицита или весь в дефиците) — ``ValueError``: сигнал ошибки
    конфигурации, а не тихий fallback. Вызывается из ``m3-train-cli``."""

    labels = np.asarray(train_deficit, dtype=float)
    share = float(labels.mean())
    if not 0.0 < share < 1.0:
        raise ValueError(
            f"доля дефицитных бакетов train равна {share}: pos_weight = (1 − p)/p не определён"
        )
    return (1.0 - share) / share


class MultitaskLoss(nn.Module):
    """Multi-task loss (решение 5 design.md): ``total = MSE(спрос) + MSE(сток)
    + ml.cls_weight × BCEWithLogits(логиты, метки, pos_weight)`` — оба MSE в
    нормализованном пространстве таргетов. При ``cls_weight = 0`` BCE-член из
    total не входит (и не вычисляется): total равен сумме двух MSE."""

    def __init__(self, cls_weight: float, pos_weight: float) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.pos_weight = pos_weight

    def forward(
        self, output: ModelOutput, y_requests: Tensor, y_free: Tensor, y_deficit: Tensor
    ) -> LossComponents:
        """Компоненты по выходу сети и таргетам батча (метки — 0/1, (n, 2))."""

        mse_requests = F.mse_loss(output.pred_requests, y_requests)
        mse_free = F.mse_loss(output.pred_free_drivers, y_free)
        if self.cls_weight > 0:
            pos_weight = torch.as_tensor(self.pos_weight, dtype=y_deficit.dtype)
            bce = F.binary_cross_entropy_with_logits(
                output.deficit_logits, y_deficit, pos_weight=pos_weight
            )
            total = mse_requests + mse_free + self.cls_weight * bce
        else:
            bce = torch.zeros((), dtype=mse_requests.dtype)
            total = mse_requests + mse_free
        return LossComponents(total, mse_requests, mse_free, bce)


def build_model(arch: Arch, cfg: MlCfg, n_zones: int) -> MultiTaskNet:
    """Собрать ``MultiTaskNet`` вокруг туловища выбранной архитектуры.

    ``n_zones`` — число районов города (ёмкость обучаемого эмбеддинга зоны);
    размеры слоёв — гиперпараметры ``cfg`` (приложение А плана M3). Фиксацию
    seed делает вызывающий через :func:`set_global_seed` перед построением
    (решение 7 design.md).
    """

    if arch == "mlp":
        trunk: nn.Module = MLPTrunk(cfg, n_zones)
    elif arch == "lstm":
        trunk = LSTMTrunk(cfg, n_zones)
    else:
        raise ValueError(f"неизвестная архитектура {arch!r}: ожидается 'mlp' или 'lstm'")
    return MultiTaskNet(trunk)
