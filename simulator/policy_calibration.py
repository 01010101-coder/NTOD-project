"""Калибровка честности политики (changes ``m6-worlds``, ``policy-driver-signal``).

Запуск: ``python -m simulator.policy_calibration [--config config.yaml] [--db …]
[--artifacts-root …]`` — δ-калибровка цены; ``--signal`` — калибровка порога
сигнала водителям (см. ниже).

δ-калибровка. На валидационных днях эпохи M3 (хвост train-роли:
дни ``train_days − val_days + 1 .. train_days``, при эталоне — 29–30) обе сети из
свежайших каталогов артефактов семейств ``mlp``/``lstm`` прогоняются walk-forward
инференсом (тот же сборщик val-примеров, что у M3, без ретрейна и перекалибровки —
§2 плана M5+M6) — получается набор вероятностей ``deficit_prob`` горизонта
``policy.horizon_min`` по парам (зона, момент). Для каждого семейства δ-сдвиг
подбирается чистой функцией :func:`fit_delta` (бинарный поиск по монотонной доле
срабатываний) так, чтобы частота «любая ступень активна» ≈ ``policy.target_fire_rate``
в допуске ±1 п.п.; недостижимая цель (дискретный скачок доли, риск §8.4) фиксируется
честно — наилучшие δ и признак недостижения, без Platt/изотонной калибровки.

Калибровка порога сигнала (change ``policy-driver-signal``). По тем же val-дням
эпохи (роли — из снимка артефактов планирования) сети планирования обоих
семейств дают вероятности вреда на горизонте ``policy.signal.horizon_min`` по
парам (зона, момент). Для каждого семейства порог подбирается чистой функцией
:func:`fit_threshold` так, чтобы доля пар с вероятностью не ниже порога была
≈ ``policy.signal.target_fire_rate`` в допуске ±1 п.п. Результат —
``artifacts/signal-calibration.json`` и точечное обновление строки
``threshold:`` подсекции ``policy.signal`` ``config.yaml``.

Модуль в ``simulator/``, а не ``ml/`` (design решение 1): калибровка потребляет
``policy.stairs``/``target_fire_rate`` — предметную область политики, а ``ml`` не
знает про политику (решение 1 design ``m6-policy``). Направление зависимостей —
simulator импортирует ml публично, не наоборот.

Результат δ-калибровки — ``artifacts/policy-calibration.json`` (δ, частоты до/после,
target, допуск, факт достижения, имена каталогов артефактов, границы окна,
горизонт) и точечное обновление строки ``shift:`` секции ``policy:``
``config.yaml`` (design решение 2: PyYAML не сохраняет комментарии, поэтому
правится текст строки, а не перезаписывается YAML). Guard'ы до всякой записи:
отсутствие каталога любого семейства или строки ``shift:`` в конфиге — ошибка без
создания json и без правки конфига (CLI — exit 2). Процедура детерминирована:
повторный запуск даёт тот же json и те же δ.
"""

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

from common.config import AppConfig, PolicyStairCfg, load_config
from ml.dataset import (
    N_FEATURES,
    RET_HORIZONS_MIN,
    SqliteRowSource,
    bucket_index,
    build_window,
    day_role,
    from_bucket_index,
    world_start,
)
from ml.infer import (
    collect_role_examples,
    latest_artifact_dir,
    load_model_artifact,
    predict_windows,
)
from ml.planning import (
    latest_planning_artifact_dir,
    load_planning_artifact,
    planning_targets,
    predict_planning_windows,
)

TOLERANCE = 0.01
"""Допуск частоты срабатываний: ±1 п.п. от цели (спека)."""

JSON_NAME = "policy-calibration.json"
"""Имя json-артефакта δ-калибровки в корне каталога артефактов."""

SIGNAL_JSON_NAME = "signal-calibration.json"
"""Имя json-артефакта калибровки порога сигнала (спека policy-driver-signal)."""

FAMILIES = ("mlp", "lstm")
"""Семейства моделей калибровки — обе ветки C/D вехи M5+M6."""

_DELTA_BOUND = 1.0
"""Граница поиска δ ∈ [−1, 1]: больший сдвиг бессмыслен (p_eff клипится в [0, 1])."""

_THRESHOLD_BOUND = 1.0
"""Граница поиска порога вероятности ∈ [0, 1]."""

_MAX_ITER = 100
"""Предел итераций бинарного поиска (запас на дискретный скачок доли, риск §8.4)."""

_SECTION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):")
"""Верхнеуровневая секция YAML-файла (строка без отступа, оканчивающаяся «:»)."""

_SHIFT_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)shift[ \t]*:[ \t]*\{(?P<body>[^}]*)\}(?P<tail>.*)$"
)
"""Строка ``shift:`` в flow-формате ``{ключ: значение, …}`` с хвостовым комментарием."""

_THRESHOLD_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)threshold[ \t]*:[ \t]*\{(?P<body>[^}]*)\}(?P<tail>.*)$"
)
"""Строка ``threshold:`` подсекции ``policy.signal`` в flow-формате."""

_SHIFT_KEY_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_-]*)[ \t]*:")
"""Ключ модели внутри flow-отображения строки ``shift:``/``threshold:``."""


class DeltaFit(NamedTuple):
    """Итог подбора δ семейства: сдвиг, частота при нём и факт достижения цели."""

    delta: float
    fire_rate: float
    target_reached: bool


def fire_fraction(
    probs: Sequence[float] | np.ndarray, delta: float, stairs: Sequence[PolicyStairCfg]
) -> float:
    """Доля вероятностей со сработавшей любой ступенью при сдвиге ``delta``.

    Срабатывание — ``p_eff = clip(p + delta, 0, 1) ≥ min(p_min)`` ступеней (любая
    ступень активна). Чистая функция, монотонная (неубывающая) по ``delta`` —
    основа бинарного поиска :func:`fit_delta`. Частота ступеней выражается
    аналитически по вероятностям — прогон мира на каждый δ не нужен (design
    решение 1).
    """

    if len(probs) == 0:
        raise ValueError("нет вероятностей для расчёта доли срабатываний")
    p_min = min(stair.p_min for stair in stairs)
    p_eff = np.clip(np.asarray(probs, dtype=float) + delta, 0.0, 1.0)
    return float(np.mean(p_eff >= p_min))


def fit_delta(
    probs: Sequence[float] | np.ndarray,
    *,
    target: float,
    stairs: Sequence[PolicyStairCfg],
    tolerance: float = TOLERANCE,
) -> DeltaFit:
    """Подобрать δ бинарным поиском в [−1, 1] до входа в допуск от ``target``.

    Доля срабатываний монотонна по δ, поэтому дихотомия сходится; поиск
    останавливается при первом δ, попавшем в допуск ±``tolerance``. Если ни один
    δ не попадает (дискретный скачок доли — риск §8.4), возвращается наилучший
    найденный δ с ``target_reached=False`` — ограничение фиксируется честно, без
    перекалибровки сетей.
    """

    lo, hi = -_DELTA_BOUND, _DELTA_BOUND
    best_delta, best_gap = lo, abs(fire_fraction(probs, lo, stairs) - target)
    for _ in range(_MAX_ITER):
        mid = (lo + hi) / 2.0
        frac = fire_fraction(probs, mid, stairs)
        gap = abs(frac - target)
        if gap < best_gap:
            best_delta, best_gap = mid, gap
        if gap <= tolerance:
            return DeltaFit(mid, frac, True)
        if frac < target:
            lo = mid
        else:
            hi = mid
    return DeltaFit(best_delta, fire_fraction(probs, best_delta, stairs), False)


class ThresholdFit(NamedTuple):
    """Итог подбора порога семейства: порог, доля при нём и факт достижения цели."""

    threshold: float
    fire_rate: float
    target_reached: bool


def threshold_fire_fraction(probs: Sequence[float] | np.ndarray, threshold: float) -> float:
    """Доля вероятностей не ниже порога — доля срабатываний сигнала.

    Срабатывание сигнала — ``p ≥ threshold`` (порог включается, как и ступень
    политики цены). Чистая функция, невозрастающая по ``threshold`` — основа
    бинарного поиска :func:`fit_threshold`.
    """

    if len(probs) == 0:
        raise ValueError("нет вероятностей для расчёта доли срабатываний сигнала")
    return float(np.mean(np.asarray(probs, dtype=float) >= threshold))


def fit_threshold(
    probs: Sequence[float] | np.ndarray,
    target: float,
    *,
    tolerance: float = TOLERANCE,
) -> ThresholdFit:
    """Подобрать порог сигнала бинарным поиском в [0, 1] до входа в допуск.

    Доля ``p ≥ threshold`` невозрастает по порогу, поэтому дихотомия сходится;
    поиск останавливается при первом пороге, попавшем в допуск ±``tolerance``
    от ``target``. Записываемый порог округляется до 3 знаков (как δ), и
    «частота после» считается уже при округлённом пороге — конфиг, json и
    фактическое поведение согласованы. Недостижимая цель (дискретный скачок
    доли, вырожденное распределение) фиксируется честно: наилучший найденный
    порог и ``target_reached=False``.
    """

    lo, hi = 0.0, _THRESHOLD_BOUND
    best_threshold = lo
    best_gap = abs(threshold_fire_fraction(probs, lo) - target)
    for _ in range(_MAX_ITER):
        mid = (lo + hi) / 2.0
        frac = threshold_fire_fraction(probs, mid)
        gap = abs(frac - target)
        if gap < best_gap:
            best_threshold, best_gap = mid, gap
        if gap <= tolerance:
            threshold = round(mid, 3)
            after = threshold_fire_fraction(probs, threshold)
            return ThresholdFit(threshold, after, abs(after - target) <= tolerance)
        if frac > target:
            lo = mid
        else:
            hi = mid
    threshold = round(best_threshold, 3)
    after = threshold_fire_fraction(probs, threshold)
    return ThresholdFit(threshold, after, abs(after - target) <= tolerance)


class _ValProbs(NamedTuple):
    """Собранные вероятности val-эпохи и контекст записи в json."""

    probs: dict[str, np.ndarray]
    first_day: int  # первый валидационный день (нумерация дней мира с 1)
    last_day: int  # последний валидационный день
    dirs: dict[str, Path]  # каталоги артефактов семейств


def _collect_val_probs(
    cfg: AppConfig, *, db_path: str | Path, artifacts_root: str | Path
) -> _ValProbs:
    """Вероятности ``deficit_prob`` горизонта ``policy.horizon_min`` по val-парам.

    Guard'ы до всякой записи: каталоги обоих семейств должны существовать
    (``FileNotFoundError``), снимки ``ml_cfg`` сетей — совпадать в границах
    ролей и окне (иначе наборы пар разошлись бы), набор val-пар — непуст.
    Примеры собираются один раз сборщиком val-роли (:func:`ml.infer.collect_role_examples`
    со снимком артефакта), инференс обоих семейств — одним батчем
    :func:`ml.infer.predict_windows` на одних окнах.
    """

    dirs = {family: latest_artifact_dir(artifacts_root, family) for family in FAMILIES}
    zones = [zone.id for zone in cfg.city.zones]
    n_zones = max(zones) + 1
    artifacts = {family: load_model_artifact(dirs[family], n_zones=n_zones) for family in FAMILIES}
    snapshots = {
        (
            artifact.ml_cfg.train_days,
            artifact.ml_cfg.val_days,
            artifact.ml_cfg.test_days,
            artifact.ml_cfg.window_hours,
        )
        for artifact in artifacts.values()
    }
    if len(snapshots) > 1:
        raise ValueError(
            f"снимки конфига сетей расходятся в границах ролей или окне ({sorted(snapshots)})"
            " — наборы val-пар разошлись бы; переобучите сети одним запуском train"
        )
    snapshot = artifacts[FAMILIES[0]].ml_cfg

    source = SqliteRowSource(db_path)
    examples = collect_role_examples(
        source, snapshot, zones=zones, start=world_start(cfg), role="val"
    )
    if len(examples.keys) == 0:
        raise ValueError(f"нет val-примеров в {db_path} — проверьте БД эпохи и границы ролей")

    horizon = cfg.policy.horizon_min
    if horizon not in RET_HORIZONS_MIN:
        raise ValueError(
            f"policy.horizon_min={horizon} вне горизонтов контракта {RET_HORIZONS_MIN}"
        )
    horizon_idx = RET_HORIZONS_MIN.index(horizon)

    probs: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        _norm_requests, _norm_free, family_probs = predict_windows(
            artifacts[family], examples.windows, [zone_id for zone_id, _ in examples.keys]
        )
        probs[family] = family_probs[:, horizon_idx].copy()
    return _ValProbs(
        probs=probs,
        first_day=snapshot.train_days - snapshot.val_days + 1,
        last_day=snapshot.train_days,
        dirs=dirs,
    )


def _calibrate_payload(cfg: AppConfig, *, db_path: str | Path, artifacts_root: str | Path) -> dict:
    """Payload json-артефакта: δ, частоты до/после, факт достижения и контекст.

    δ округляется до 3 знаков до расчёта «частоты после» — записанный в json и
    конфиг сдвиг, заявленная частота и фактическое поведение политики согласованы;
    round-trip «загрузка конфига даёт сдвиги, равные json» выполняется точно.
    """

    collected = _collect_val_probs(cfg, db_path=db_path, artifacts_root=artifacts_root)
    stairs = cfg.policy.stairs
    target = cfg.policy.target_fire_rate

    families: dict[str, dict] = {}
    shift: dict[str, float] = {}
    for family in FAMILIES:
        before = fire_fraction(collected.probs[family], 0.0, stairs)
        fit = fit_delta(collected.probs[family], target=target, stairs=stairs)
        delta = round(fit.delta, 3)
        after = fire_fraction(collected.probs[family], delta, stairs)
        families[family] = {
            "delta": delta,
            "fire_rate_before": round(before, 6),
            "fire_rate_after": round(after, 6),
            "target_reached": bool(abs(after - target) <= TOLERANCE),
            "artifact_dir": collected.dirs[family].name,
        }
        shift[family] = delta
    return {
        "shift": shift,
        "families": families,
        "target_fire_rate": target,
        "tolerance": TOLERANCE,
        "window_days": {"first_day": collected.first_day, "last_day": collected.last_day},
        "horizon_min": cfg.policy.horizon_min,
    }


def _write_json(payload: dict, artifacts_root: str | Path) -> Path:
    """Записать json-артефакт калибровки (детерминированный вывод — фиксированный
    порядок ключей и ``ensure_ascii=False``)."""

    json_path = Path(artifacts_root) / JSON_NAME
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return json_path


def _resolve_root(cfg: AppConfig, artifacts_root: str | Path | None) -> Path:
    """Корень артефактов: явный параметр или ``ml.artifacts_root`` конфига.

    ``None`` — ключ конфига (change ``world-final-config``); дефолт
    ``artifacts`` сохраняет прежние пути эталонного мира.
    """

    return Path(cfg.ml.artifacts_root) if artifacts_root is None else Path(artifacts_root)


def calibrate_shifts(
    cfg: AppConfig,
    *,
    db_path: str | Path | None = None,
    artifacts_root: str | Path | None = None,
) -> tuple[dict, Path]:
    """Процедура калибровки (задача 1.3): подбор δ обоих семейств и запись json.

    ``db_path=None`` — БД из ``cfg.ml.db``. Guard'ы отсутствия артефактов и пустой
    val-роли срабатывают до записи. ``artifacts_root=None`` — ``ml.artifacts_root``
    конфига. Возвращает (payload, путь json).
    """

    root = _resolve_root(cfg, artifacts_root)
    payload = _calibrate_payload(
        cfg, db_path=cfg.ml.db if db_path is None else db_path, artifacts_root=root
    )
    return payload, _write_json(payload, root)


# ---------------------------------------------------------------------------
# Калибровка порога сигнала водителям (change policy-driver-signal)
# ---------------------------------------------------------------------------


class _SignalValProbs(NamedTuple):
    """Вероятности val-эпохи сетей планирования и контекст записи в json."""

    probs: dict[str, np.ndarray]
    first_day: int  # первый валидационный день (нумерация дней мира с 1)
    last_day: int  # последний валидационный день
    dirs: dict[str, Path]  # каталоги артефактов семейств планирования


def _signal_val_windows(
    cfg: AppConfig, source: SqliteRowSource, snapshot
) -> tuple[np.ndarray, list[tuple[int, datetime]]]:
    """Окна val-примеров планирования: ``(n, steps, 17)`` и ключи (зона, ts).

    Перебор повторяет сборщик обучающих примеров планирования
    (:func:`ml.planning.collect_planning_examples`): окно истории строится
    без утечки будущего, пара берётся только при полном окне цели — так наборы
    пар совпадают с val-ролью сетей планирования.
    """

    zones = [zone.id for zone in cfg.city.zones]
    step = cfg.engine.agg_step_min
    horizons = list(snapshot.planning.horizons_min)
    steps = int(snapshot.window_hours * 60 // step)
    if steps <= 0:
        raise ValueError(f"ml.window_hours={snapshot.window_hours} даёт пустое окно истории")
    first_ts, last_ts = source.time_range()
    from_idx = bucket_index(first_ts, step) + (steps - 1)
    until_idx = (
        bucket_index(last_ts, step) - (max(horizons) + snapshot.planning.window_min) // step + 1
    )
    start = world_start(cfg)
    windows: list[np.ndarray] = []
    keys: list[tuple[int, datetime]] = []
    for idx in range(from_idx, until_idx + 1):
        ts = from_bucket_index(idx, step)
        role = day_role(
            ts,
            start,
            train_days=snapshot.train_days,
            val_days=snapshot.val_days,
            test_days=snapshot.test_days,
        )
        if role != "val":
            continue
        for zone_id in zones:
            window = build_window(source, zone_id, ts, window_hours=snapshot.window_hours)
            if window is None:
                continue
            targets = planning_targets(
                source,
                zone_id,
                ts,
                horizons_min=horizons,
                window_min=snapshot.planning.window_min,
                agg_step_min=step,
            )
            if targets is None:
                continue
            windows.append(window)
            keys.append((zone_id, ts))
    if not windows:
        return np.empty((0, steps, N_FEATURES), dtype=float), keys
    return np.stack(windows), keys


def _collect_signal_probs(
    cfg: AppConfig, *, db_path: str | Path, planning_artifacts_root: str | Path
) -> _SignalValProbs:
    """Вероятности вреда сетей планирования горизонта ``policy.signal.horizon_min``.

    Guard'ы до всякой записи: каталоги обоих семейств должны существовать
    (``FileNotFoundError``), снимки ``ml_cfg`` — совпадать в границах ролей,
    окне и горизонтах (иначе наборы пар и колонки разошлись бы), горизонт
    сигнала — входить в горизонты снимка (``ValueError``), val-пары — быть
    непустыми. Инференс обоих семейств — одним батчем на одних окнах.
    """

    dirs = {
        family: latest_planning_artifact_dir(planning_artifacts_root, family) for family in FAMILIES
    }
    zones = [zone.id for zone in cfg.city.zones]
    n_zones = max(zones) + 1
    artifacts = {
        family: load_planning_artifact(dirs[family], n_zones=n_zones) for family in FAMILIES
    }
    snapshots = {
        (
            artifact.ml_cfg.train_days,
            artifact.ml_cfg.val_days,
            artifact.ml_cfg.test_days,
            artifact.ml_cfg.window_hours,
            tuple(artifact.ml_cfg.planning.horizons_min),
            artifact.ml_cfg.planning.window_min,
        )
        for artifact in artifacts.values()
    }
    if len(snapshots) > 1:
        raise ValueError(
            f"снимки сетей планирования расходятся в границах ролей, окне или горизонтах"
            f" ({sorted(snapshots)}) — переобучите сети одним запуском plan-train"
        )
    snapshot = artifacts[FAMILIES[0]].ml_cfg
    horizons = list(snapshot.planning.horizons_min)
    horizon = cfg.policy.signal.horizon_min
    if horizon not in horizons:
        raise ValueError(
            f"policy.signal.horizon_min={horizon} вне горизонтов артефакта {horizons}"
            f" ({artifacts[FAMILIES[0]].model_version})"
        )
    horizon_idx = horizons.index(horizon)

    windows, keys = _signal_val_windows(cfg, SqliteRowSource(db_path), snapshot)
    if not keys:
        raise ValueError(
            f"нет val-примеров планирования в {db_path} — проверьте БД эпохи"
            " и границы ролей снимка артефактов"
        )
    probs: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        family_probs = predict_planning_windows(
            artifacts[family], windows, [zone_id for zone_id, _ in keys]
        )
        probs[family] = family_probs[:, horizon_idx].copy()
    return _SignalValProbs(
        probs=probs,
        first_day=snapshot.train_days - snapshot.val_days + 1,
        last_day=snapshot.train_days,
        dirs=dirs,
    )


def _signal_payload(
    cfg: AppConfig, *, db_path: str | Path, planning_artifacts_root: str | Path
) -> dict:
    """Payload json-артефакта: пороги, доли, цель, допуск и контекст (спека)."""

    collected = _collect_signal_probs(
        cfg, db_path=db_path, planning_artifacts_root=planning_artifacts_root
    )
    target = cfg.policy.signal.target_fire_rate
    families: dict[str, dict] = {}
    thresholds: dict[str, float] = {}
    for family in FAMILIES:
        fit = fit_threshold(collected.probs[family], target)
        thresholds[family] = fit.threshold
        families[family] = {
            "threshold": fit.threshold,
            "fire_rate": round(fit.fire_rate, 6),
            "target_reached": bool(fit.target_reached),
            "artifact_dir": collected.dirs[family].name,
        }
    return {
        "threshold": thresholds,
        "families": families,
        "target_fire_rate": target,
        "tolerance": TOLERANCE,
        "window_days": {"first_day": collected.first_day, "last_day": collected.last_day},
        "horizon_min": cfg.policy.signal.horizon_min,
        "mult": cfg.policy.signal.mult,
        "artifacts_root": str(planning_artifacts_root),
    }


def _write_signal_json(payload: dict, artifacts_root: str | Path) -> Path:
    """Записать json-артефакт калибровки сигнала (детерминированный вывод)."""

    json_path = Path(artifacts_root) / SIGNAL_JSON_NAME
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return json_path


def calibrate_signal(
    cfg: AppConfig,
    *,
    db_path: str | Path | None = None,
    planning_artifacts_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
) -> tuple[dict, Path]:
    """Процедура калибровки порога (change ``policy-driver-signal``).

    ``db_path=None`` — БД из ``cfg.ml.db``; ``planning_artifacts_root=None`` —
    ``ml.planning.artifacts_root`` конфига; ``artifacts_root=None`` — корень
    ``ml.artifacts_root`` конфига, куда пишется ``signal-calibration.json``.
    Guard'ы отсутствия артефактов и пустой val-роли срабатывают до записи.
    Возвращает (payload, путь json).
    """

    root = Path(
        cfg.ml.planning.artifacts_root
        if planning_artifacts_root is None
        else planning_artifacts_root
    )
    payload = _signal_payload(
        cfg,
        db_path=cfg.ml.db if db_path is None else db_path,
        planning_artifacts_root=root,
    )
    return payload, _write_signal_json(payload, _resolve_root(cfg, artifacts_root))


def _replace_shift_line(text: str, shift: Mapping[str, float]) -> str:
    """Точечная замена строки ``shift:`` секции ``policy:`` (design решение 2).

    Меняется только значения ключей внутри flow-отображения: отступ, порядок
    ключей строки и хвостовой комментарий сохраняются, остальной файл не
    трогается. Строка не найдена (конфиг heterodox) или в ``shift`` нет значения
    для ключа строки — ``ValueError`` без записи (чинится руками).
    """

    lines = text.splitlines(keepends=True)
    section: str | None = None
    for i, line in enumerate(lines):
        stripped = line.rstrip("\r\n")
        top = _SECTION_RE.match(stripped)
        if top:
            section = top.group(1)
            continue
        if section != "policy":
            continue
        matched = _SHIFT_LINE_RE.match(stripped)
        if matched is None:
            continue
        keys = _SHIFT_KEY_RE.findall(matched.group("body"))
        if not keys:
            raise ValueError("строка 'shift:' секции 'policy:' не содержит ключей моделей")
        missing = [key for key in keys if key not in shift]
        if missing:
            raise ValueError(
                f"нет δ-сдвигов для моделей {missing} — калибровка возвращает {sorted(shift)}"
            )
        new_body = ", ".join(f"{key}: {float(shift[key]):.3f}" for key in keys)
        eol = line[len(stripped) :]
        lines[i] = f"{matched.group('indent')}shift: {{{new_body}}}{matched.group('tail')}{eol}"
        return "".join(lines)
    raise ValueError("строка 'shift:' в секции 'policy:' не найдена — обновите config.yaml вручную")


def update_config_shift(config_path: str | Path, shift: Mapping[str, float]) -> None:
    """Обновить ``policy.shift`` конфига подобранными δ (задача 1.4).

    Точечная правка текста (комментарии эталона сохраняются); при отсутствии
    строки ``shift:`` исключение поднимается до записи — файл не трогается.
    """

    path = Path(config_path)
    new_text = _replace_shift_line(path.read_text(encoding="utf-8"), shift)
    path.write_text(new_text, encoding="utf-8")


def _replace_threshold_line(text: str, thresholds: Mapping[str, float]) -> str:
    """Точечная замена строки ``threshold:`` подсекции ``policy.signal``.

    Меняются только значения ключей внутри flow-отображения: отступ, порядок
    ключей и хвостовой комментарий сохраняются, остальной файл не трогается.
    Строка не найдена или в ``thresholds`` нет значения для ключа строки —
    ``ValueError`` без записи (чинится руками).
    """

    lines = text.splitlines(keepends=True)
    section: str | None = None
    for i, line in enumerate(lines):
        stripped = line.rstrip("\r\n")
        top = _SECTION_RE.match(stripped)
        if top:
            section = top.group(1)
            continue
        if section != "policy":
            continue
        matched = _THRESHOLD_LINE_RE.match(stripped)
        if matched is None:
            continue
        keys = _SHIFT_KEY_RE.findall(matched.group("body"))
        if not keys:
            raise ValueError("строка 'threshold:' подсекции 'policy.signal' не содержит семейств")
        missing = [key for key in keys if key not in thresholds]
        if missing:
            raise ValueError(
                f"нет порогов для семейств {missing} — калибровка возвращает {sorted(thresholds)}"
            )
        new_body = ", ".join(f"{key}: {float(thresholds[key]):.3f}" for key in keys)
        eol = line[len(stripped) :]
        lines[i] = f"{matched.group('indent')}threshold: {{{new_body}}}{matched.group('tail')}{eol}"
        return "".join(lines)
    raise ValueError(
        "строка 'threshold:' в секции 'policy.signal' не найдена — обновите config.yaml вручную"
    )


def update_config_threshold(config_path: str | Path, thresholds: Mapping[str, float]) -> None:
    """Обновить ``policy.signal.threshold`` конфига подобранными порогами.

    Точечная правка текста (комментарии эталона сохраняются); при отсутствии
    строки ``threshold:`` исключение поднимается до записи — файл не трогается.
    """

    path = Path(config_path)
    new_text = _replace_threshold_line(path.read_text(encoding="utf-8"), thresholds)
    path.write_text(new_text, encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Аргументы CLI: конфиг, БД эпохи, корень артефактов и режим ``--signal``."""

    parser = argparse.ArgumentParser(
        prog="simulator.policy_calibration",
        description=(
            "Калибровка честности политики: δ-сдвиги цены (по умолчанию) или порог"
            " сигнала водителям (--signal) на val-днях эпохи → artifacts/*.json +"
            " config.yaml (m6-worlds, policy-driver-signal)."
        ),
    )
    parser.add_argument("--config", default="config.yaml", help="путь к конфигу YAML")
    parser.add_argument("--db", default=None, help="БД эпохи (по умолчанию ml.db из конфига)")
    parser.add_argument(
        "--artifacts-root",
        default=None,
        help=(
            "корень каталогов артефактов моделей и json-результата (по умолчанию"
            " ml.artifacts_root конфига, в дефолте — artifacts)"
        ),
    )
    parser.add_argument(
        "--signal",
        action="store_true",
        help="калибровать порог сигнала водителям (policy.signal.threshold) по сетям планирования",
    )
    parser.add_argument(
        "--planning-artifacts-root",
        default=None,
        help=(
            "корень каталогов артефактов планирования (по умолчанию"
            " ml.planning.artifacts_root конфига); только с --signal"
        ),
    )
    args = parser.parse_args(argv)
    if args.planning_artifacts_root is not None and not args.signal:
        parser.error("--planning-artifacts-root имеет смысл только с --signal")
    return args


def _signal_main(args: argparse.Namespace) -> int:
    """Режим ``--signal``: порог обоих семейств → json → точечная правка конфига.

    Guard'ы (нет каталога семейства, расходящиеся снимки, пустая val-роль,
    горизонт вне снимка, нет строки ``threshold:`` в конфиге) — до всякой
    записи: stderr и exit 2, json не создан, конфиг не изменён. Повторный
    запуск детерминирован.
    """

    try:
        cfg = load_config(args.config)
        payload = _signal_payload(
            cfg,
            db_path=cfg.ml.db if args.db is None else args.db,
            planning_artifacts_root=Path(
                cfg.ml.planning.artifacts_root
                if args.planning_artifacts_root is None
                else args.planning_artifacts_root
            ),
        )
        # Preflight правки конфига до всякой записи: нет строки — ошибка без json.
        config_path = Path(args.config)
        new_text = _replace_threshold_line(
            config_path.read_text(encoding="utf-8"), payload["threshold"]
        )
        json_path = _write_signal_json(payload, _resolve_root(cfg, args.artifacts_root))
        config_path.write_text(new_text, encoding="utf-8")
    except (FileNotFoundError, ValueError) as exc:
        print(f"калибровка сигнала прервана: {exc}", file=sys.stderr)
        return 2
    for family in FAMILIES:
        entry = payload["families"][family]
        print(
            f"signal-calibration: family={family} threshold={entry['threshold']:.3f}"
            f" fire_rate={entry['fire_rate']:.3f}"
            f" target={payload['target_fire_rate']:.3f}"
            f" reached={'yes' if entry['target_reached'] else 'no'}"
            f" artifact={entry['artifact_dir']}"
        )
    print(f"signal-calibration: json={json_path} config={args.config}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI: подбор δ → json → правка конфига → печать.

    ``--signal`` переключает на калибровку порога сигнала водителям (change
    ``policy-driver-signal``). Guard'ы (нет каталога семейства, расходящиеся
    снимки, пустая val-роль, нет строки ``shift:``/``threshold:`` в конфиге) —
    до всякой записи: stderr и exit 2, json не создан, конфиг не изменён.
    Повторный запуск детерминирован.
    """

    args = _parse_args(argv)
    if args.signal:
        return _signal_main(args)
    try:
        cfg = load_config(args.config)
        root = _resolve_root(cfg, args.artifacts_root)
        payload = _calibrate_payload(
            cfg,
            db_path=cfg.ml.db if args.db is None else args.db,
            artifacts_root=root,
        )
        # Preflight правки конфига до всякой записи: нет строки — ошибка без json.
        config_path = Path(args.config)
        new_text = _replace_shift_line(config_path.read_text(encoding="utf-8"), payload["shift"])
        json_path = _write_json(payload, root)
        config_path.write_text(new_text, encoding="utf-8")
    except (FileNotFoundError, ValueError) as exc:
        print(f"калибровка прервана: {exc}", file=sys.stderr)
        return 2
    for family in FAMILIES:
        entry = payload["families"][family]
        print(
            f"calibration: family={family} delta={entry['delta']}"
            f" fire_rate={entry['fire_rate_before']:.3f}->{entry['fire_rate_after']:.3f}"
            f" target={payload['target_fire_rate']:.3f}"
            f" reached={'yes' if entry['target_reached'] else 'no'}"
            f" artifact={entry['artifact_dir']}"
        )
    print(f"calibration: json={json_path} config={args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
