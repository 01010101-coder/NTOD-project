"""Walk-forward инференс вехи M3 (change ``m3-inference``): прогнозы сетей из
артефактов и наивов по тест-эпохе → строки контракта ``forecasts``.

Сбор примеров — тот же :func:`ml.dataset.collect_examples` (решение 1 design.md):
одна семантика границ «есть окно и оба целевых бакета» и одни анти-лик гарантии,
что и в обучении; инференсу нужны только окна test-роли, таргеты собираются
«за компанию» (лишняя работа — секунды, принята сознательно). Сеть
восстанавливается из каталога артефакта версии
(:func:`ml.train.save_model_artifacts`): архитектура и окно — из снимка
``ml_cfg`` в ``weights.pt`` (не из живого конфига, который мог дрейфнуть,
решение 2 design.md), веса — ``state_dict``, нормализация — scaler'ы каталога;
прогноз — батчи без градиентов в ``eval``-режиме, регрессии денормализуются
scaler'ами таргетов (по образцу ``val_mae_requests``), вероятность дефицита —
после сигмоиды, как есть. Наивы считаются по тем же ключам test-роли
(решение 7 design.md) — честное сравнение версий с первого дня тест-эпохи.

Прогон версий (:func:`run_inference`) — guard'ы до всякой записи (решение 6
design.md), миграция PK (:func:`storage.schema.ensure_forecasts_schema`),
затем для каждой версии delete + insert (:func:`storage.repo.delete_forecasts`)
— повтор идемпотентен: сеть детерминирована на фиксированных весах, наивы —
чистые функции, порядок строк фиксирован ключами test-роли. Сводная
(:func:`summarize_run`) считается по строкам из БД (решение 5 design.md) —
чтение заодно валидирует записанное.
"""

import json
import sqlite3
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from common.config import AppConfig, MlCfg
from ml.baselines import NAIVE_PERSIST_V1, NAIVE_SEASON_V1, naive_persist, naive_season
from ml.dataset import (
    RET_HORIZONS_MIN,
    DatasetScalers,
    DayRole,
    Examples,
    RowSource,
    collect_examples,
    world_start,
)
from ml.metrics import build_summary, render_table
from ml.model import Arch, MultiTaskNet, build_model
from ml.train import ARTIFACTS_ROOT, SCALERS_FILE, WEIGHTS_FILE
from storage.repo import delete_forecasts, insert_forecasts, list_forecasts
from storage.schema import ensure_forecasts_schema

INFER_BATCH_SIZE = 4096
"""Батч инференса walk-forward (без градиентов — крупнее обучающего, как в свипе)."""

MODELS_CHOICES = ("all", "naive", "mlp", "lstm")
"""Наборы семейств CLI: ``all`` — наивы и обе сети; остальные — одно семейство."""

_FAMILY_SETS: dict[str, tuple[str, ...]] = {
    "all": ("naive", "mlp", "lstm"),
    "naive": ("naive",),
    "mlp": ("mlp",),
    "lstm": ("lstm",),
}


class ModelArtifact(NamedTuple):
    """Восстановленный артефакт версии модели: сеть в ``eval``-режиме, scaler'ы
    каталога и снимок ``ml_cfg`` обучения (источник границ ролей и окна)."""

    model_version: str
    arch: Arch
    ml_cfg: MlCfg
    model: MultiTaskNet
    scalers: DatasetScalers


class InferredVersion(NamedTuple):
    """Итог записи одной версии: имя ``model_version`` и число строк в ``forecasts``."""

    model_version: str
    rows: int


def load_model_artifact(model_dir: str | Path, *, n_zones: int) -> ModelArtifact:
    """Восстановить сеть из каталога артефакта версии (задача 2.1).

    Сеть строится по снимку ``ml_cfg`` из ``weights.pt`` (не по живому конфигу:
    окно и размеры слоёв соответствуют обучению даже после его дрейфа), веса —
    ``state_dict``, нормализация — :class:`ml.dataset.DatasetScalers` каталога;
    ``model_version`` — имя каталога. ``n_zones`` (ёмкость эмбеддинга зоны) —
    параметр окружения, а не обучения, поэтому берётся вызывающим из
    ``config.yaml`` города (``max(zone.id) + 1``).
    """

    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"каталог модели не найден: {model_dir}")
    blob = torch.load(model_dir / WEIGHTS_FILE, weights_only=True)
    ml_cfg = MlCfg(**blob["ml_cfg"])
    model = build_model(blob["arch"], ml_cfg, n_zones=n_zones)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return ModelArtifact(
        model_version=model_dir.name,
        arch=blob["arch"],
        ml_cfg=ml_cfg,
        model=model,
        scalers=DatasetScalers.load(model_dir / SCALERS_FILE),
    )


def latest_artifact_dir(artifacts_root: str | Path, arch: Arch) -> Path:
    """Самый свежий каталог семейства: лексический максимум ``{arch}-v1-*``.

    Даты ISO в именах сортируются лексически (решение 6 design.md); двух
    каталогов одной даты не бывает — ``train`` отказывается их создавать.
    """

    candidates = sorted(path for path in Path(artifacts_root).glob(f"{arch}-v1-*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"нет каталогов артефактов {arch}-v1-* в {artifacts_root} — прогоните"
            f" python -m ml train --arch {arch}"
        )
    return candidates[-1]


def collect_role_examples(
    source: RowSource,
    ml_cfg: MlCfg,
    *,
    zones: Sequence[int],
    start: datetime,
    role: DayRole = "test",
) -> Examples:
    """Примеры роли ``role`` тем же сборщиком, что и обучение (решение 1 design.md):
    границы «есть окно и оба целевых бакета» встроены, ключи (зона, ts) дают
    привязку строк. Параметры ролей и окна — из снимка ``ml_cfg``. Роль
    ``"test"`` обслуживает walk-forward M3, ``"val"`` — δ-калибровку политики
    (change ``m6-worlds``): одна машина сбора гарантирует одинаковые наборы пар
    у обоих семейств при согласованных снимках."""

    if role not in ("train", "val", "test"):
        raise ValueError(f"неизвестная роль {role!r}: ожидается 'train', 'val' или 'test'")
    return collect_examples(
        source,
        zones=zones,
        start=start,
        train_days=ml_cfg.train_days,
        val_days=ml_cfg.val_days,
        test_days=ml_cfg.test_days,
        window_hours=ml_cfg.window_hours,
    )[role]


def predict_windows(
    artifact: ModelArtifact, windows: np.ndarray, zones: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Батч-прогон сети на сырых окнах — общий путь инференса (change ``m6-policy``,
    design решение 1: walk-forward и контур политики потребляют одну нормализацию
    и один forward). ``windows`` — сырые окна (n, steps, 17), нормализуются
    scaler'ами артефакта; прогноз — батчи без градиентов в ``eval``-режиме сети.
    Возвращает (нормализованный ``pred_requests``, нормализованный
    ``pred_free_drivers``, ``deficit_prob`` после сигмоиды) формами
    (n, 2); денормализация регрессий — забота вызывающего (walk-forward делает
    это scaler'ами таргетов артефакта)."""

    n = windows.shape[0]
    zone_arr = np.asarray(zones, dtype=np.int64)
    norm_requests: list[np.ndarray] = []
    norm_free: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for lo in range(0, n, INFER_BATCH_SIZE):
            batch = slice(lo, lo + INFER_BATCH_SIZE)
            windows_t = torch.from_numpy(
                artifact.scalers.transform_windows(windows[batch]).astype(np.float32)
            )
            zones_t = torch.from_numpy(zone_arr[batch])
            output = artifact.model(windows_t, zones_t)
            norm_requests.append(output.pred_requests.numpy())
            norm_free.append(output.pred_free_drivers.numpy())
            probs.append(output.deficit_prob.numpy())
    if not probs:
        empty = np.empty((0, len(RET_HORIZONS_MIN)))
        return empty, empty.copy(), empty.copy()
    return (
        np.concatenate(norm_requests),
        np.concatenate(norm_free),
        np.concatenate(probs),
    )


def _network_rows(artifact: ModelArtifact, examples: Examples) -> list[dict]:
    """Строки ``forecasts`` сети на готовых примерах (задача 2.2): общий путь
    инференса :func:`predict_windows` → ``inverse_transform`` регрессий (по
    образцу ``val_mae_requests``) → по две строки (горизонты 15/30) на пару,
    ``deficit_prob`` как есть (сигмоида)."""

    n = examples.windows.shape[0]
    if n == 0:
        return []
    norm_requests_np, norm_free_np, deficit_prob = predict_windows(
        artifact, examples.windows, [zone_id for zone_id, _ in examples.keys]
    )

    # Денормализация в физических величинах: scaler'ы таргетов артефакта.
    requests = artifact.scalers.requests.inverse_transform(norm_requests_np.reshape(-1, 1)).reshape(
        n, len(RET_HORIZONS_MIN)
    )
    free_drivers = artifact.scalers.free_drivers.inverse_transform(
        norm_free_np.reshape(-1, 1)
    ).reshape(n, len(RET_HORIZONS_MIN))

    rows: list[dict] = []
    for i, (zone_id, ts) in enumerate(examples.keys):
        for k, horizon in enumerate(RET_HORIZONS_MIN):
            rows.append(
                {
                    "ts": ts,
                    "zone_id": zone_id,
                    "horizon_min": horizon,
                    "pred_requests": float(requests[i, k]),
                    "pred_free_drivers": float(free_drivers[i, k]),
                    "deficit_prob": float(deficit_prob[i, k]),
                    "model_version": artifact.model_version,
                }
            )
    return rows


def walk_forward_model(
    artifact: ModelArtifact,
    source: RowSource,
    *,
    zones: Sequence[int],
    start: datetime,
) -> list[dict]:
    """Walk-forward одной сети по test-роли источника (самодостаточный вызов):
    сбор примеров по снимку артефакта + прогноз. Прогон всех версий на общем
    наборе пар делает :func:`run_inference` — с одной сборкой примеров."""

    examples = collect_role_examples(source, artifact.ml_cfg, zones=zones, start=start)
    return _network_rows(artifact, examples)


def walk_forward_naives(source: RowSource, examples: Examples) -> dict[str, list[dict]]:
    """Наивы тем же проходом (задача 2.4): ``naive_persist``/``naive_season``
    по ключам test-роли тех же примеров — наборы пар совпадают с сетями;
    отсутствие вчерашнего слота у сезонного — молчаливый пропуск (спека
    ``ml-baselines``)."""

    rows: dict[str, list[dict]] = {NAIVE_PERSIST_V1: [], NAIVE_SEASON_V1: []}
    for zone_id, ts in examples.keys:
        rows[NAIVE_PERSIST_V1].extend(naive_persist(source, ts, zone_id))
        rows[NAIVE_SEASON_V1].extend(naive_season(source, ts, zone_id))
    return rows


def run_inference(
    source: RowSource,
    conn: sqlite3.Connection,
    *,
    cfg: AppConfig,
    models: str = "all",
    artifacts_root: str | Path = ARTIFACTS_ROOT,
) -> list[InferredVersion]:
    """Прогон версий по тест-эпохе (задачи 3.1/3.3): набор семейств → список
    записанных версий с числом строк.

    Guard'ы до всякой записи: каталоги артефактов всех запрошенных семейств
    сразу, снимки сетей согласованы (иначе наборы пар разошлись бы и сравнение
    версий стало бы нечестным). Примеры собираются один раз на прогон и
    используются всеми версиями; затем ``ensure_forecasts_schema`` и для
    каждой версии delete + insert — повтор идемпотентен, чужие версии не
    трогаются. Наив-прогон без сетей берёт границы ролей из живого ``ml:``
    конфига (снимков у наивов нет).
    """

    if models not in _FAMILY_SETS:
        raise ValueError(
            f"неизвестный набор моделей {models!r}: ожидается один из {MODELS_CHOICES}"
        )
    families = _FAMILY_SETS[models]
    zones = [zone.id for zone in cfg.city.zones]
    n_zones = max(zones) + 1
    start = world_start(cfg)

    # Guard каталогов всех сетевых семейств сразу — до сбора примеров и записи.
    dirs: dict[str, Path] = {}
    missing: list[str] = []
    for family in families:
        if family == "naive":
            continue
        try:
            dirs[family] = latest_artifact_dir(artifacts_root, family)
        except FileNotFoundError as exc:
            missing.append(str(exc))
    if missing:
        raise FileNotFoundError("\n".join(missing))

    network = {
        family: load_model_artifact(dirs[family], n_zones=n_zones)
        for family in ("mlp", "lstm")
        if family in dirs
    }
    if network:
        snapshots = {
            (
                artifact.ml_cfg.train_days,
                artifact.ml_cfg.val_days,
                artifact.ml_cfg.test_days,
                artifact.ml_cfg.window_hours,
            )
            for artifact in network.values()
        }
        if len(snapshots) > 1:
            raise ValueError(
                "снимки конфига сетей расходятся в границах ролей или окне"
                f" ({sorted(snapshots)}) — наборы пар прогноза разошлись бы;"
                " переобучите сети одним запуском train"
            )
        collection_cfg = next(iter(network.values())).ml_cfg
    else:
        collection_cfg = cfg.ml

    examples = collect_role_examples(source, collection_cfg, zones=zones, start=start)

    ensure_forecasts_schema(conn)
    written: list[InferredVersion] = []
    for family in families:
        if family == "naive":
            versions = walk_forward_naives(source, examples).items()
        else:
            artifact = network[family]
            versions = [(artifact.model_version, _network_rows(artifact, examples))]
        for version, rows in versions:
            delete_forecasts(conn, version)
            insert_forecasts(conn, rows)
            written.append(InferredVersion(version, len(rows)))
    return written


def summarize_run(
    conn: sqlite3.Connection,
    source: RowSource,
    versions: Sequence[InferredVersion],
    *,
    artifacts_root: str | Path = ARTIFACTS_ROOT,
) -> tuple[Path, str]:
    """Сводная прогона (задача 3.2): по строкам из БД (решение 5 design.md —
    чтение валидирует записанное) — ``build_summary`` каждой версии →
    ``artifacts/summary-<локальная дата>.json`` (``{"date", "summaries"}``) и
    печатная таблица :func:`ml.metrics.render_table`."""

    summaries = [
        build_summary(
            version.model_version, list_forecasts(conn, model_version=version.model_version), source
        )
        for version in versions
    ]
    today = date.today().isoformat()
    summary_path = Path(artifacts_root) / f"summary-{today}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({"date": today, "summaries": summaries}, fh, ensure_ascii=False, indent=2)
    return summary_path, render_table(summaries)
