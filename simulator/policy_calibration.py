"""δ-калибровка честности политики (change ``m6-worlds``, спека simulator-policy).

Запуск: ``python -m simulator.policy_calibration [--config config.yaml] [--db …]
[--artifacts-root …]``. На валидационных днях эпохи M3 (хвост train-роли:
дни ``train_days − val_days + 1 .. train_days``, при эталоне — 29–30) обе сети из
свежайших каталогов артефактов семейств ``mlp``/``lstm`` прогоняются walk-forward
инференсом (тот же сборщик val-примеров, что у M3, без ретрейна и перекалибровки —
§2 плана M5+M6) — получается набор вероятностей ``deficit_prob`` горизонта
``policy.horizon_min`` по парам (зона, момент). Для каждого семейства δ-сдвиг
подбирается чистой функцией :func:`fit_delta` (бинарный поиск по монотонной доле
срабатываний) так, чтобы частота «любая ступень активна» ≈ ``policy.target_fire_rate``
в допуске ±1 п.п.; недостижимая цель (дискретный скачок доли, риск §8.4) фиксируется
честно — наилучшие δ и признак недостижения, без Platt/изотонной калибровки.

Модуль в ``simulator/``, а не ``ml/`` (design решение 1): калибровка потребляет
``policy.stairs``/``target_fire_rate`` — предметную область политики, а ``ml`` не
знает про политику (решение 1 design ``m6-policy``). Направление зависимостей —
simulator импортирует ml публично, не наоборот.

Результат — ``artifacts/policy-calibration.json`` (δ, частоты до/после, target,
допуск, факт достижения, имена каталогов артефактов, границы окна, горизонт) и
точечное обновление строки ``shift:`` секции ``policy:`` ``config.yaml`` (design
решение 2: PyYAML не сохраняет комментарии, поэтому правится текст строки, а не
перезаписывается YAML). Guard'ы до всякой записи: отсутствие каталога любого
семейства или строки ``shift:`` в конфиге — ошибка без создания json и без правки
конфига (CLI — exit 2). Процедура детерминирована: повторный запуск даёт тот же
json и те же δ.
"""

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

from common.config import AppConfig, PolicyStairCfg, load_config
from ml.dataset import RET_HORIZONS_MIN, SqliteRowSource, world_start
from ml.infer import (
    collect_role_examples,
    latest_artifact_dir,
    load_model_artifact,
    predict_windows,
)
from ml.train import ARTIFACTS_ROOT

TOLERANCE = 0.01
"""Допуск частоты срабатываний: ±1 п.п. от ``policy.target_fire_rate`` (спека)."""

JSON_NAME = "policy-calibration.json"
"""Имя json-артефакта калибровки в корне каталога артефактов."""

FAMILIES = ("mlp", "lstm")
"""Семейства моделей калибровки — обе ветки C/D вехи M5+M6."""

_DELTA_BOUND = 1.0
"""Граница поиска δ ∈ [−1, 1]: больший сдвиг бессмыслен (p_eff клипится в [0, 1])."""

_MAX_ITER = 100
"""Предел итераций бинарного поиска (запас на дискретный скачок доли, риск §8.4)."""

_SECTION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):")
"""Верхнеуровневая секция YAML-файла (строка без отступа, оканчивающаяся «:»)."""

_SHIFT_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)shift[ \t]*:[ \t]*\{(?P<body>[^}]*)\}(?P<tail>.*)$"
)
"""Строка ``shift:`` в flow-формате ``{ключ: значение, …}`` с хвостовым комментарием."""

_SHIFT_KEY_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_-]*)[ \t]*:")
"""Ключ модели внутри flow-отображения строки ``shift:``."""


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


def calibrate_shifts(
    cfg: AppConfig,
    *,
    db_path: str | Path | None = None,
    artifacts_root: str | Path = ARTIFACTS_ROOT,
) -> tuple[dict, Path]:
    """Процедура калибровки (задача 1.3): подбор δ обоих семейств и запись json.

    ``db_path=None`` — БД из ``cfg.ml.db``. Guard'ы отсутствия артефактов и пустой
    val-роли срабатывают до записи. Возвращает (payload, путь json).
    """

    payload = _calibrate_payload(
        cfg, db_path=cfg.ml.db if db_path is None else db_path, artifacts_root=artifacts_root
    )
    return payload, _write_json(payload, artifacts_root)


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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Аргументы CLI: конфиг, БД эпохи и корень артефактов — все с дефолтами."""

    parser = argparse.ArgumentParser(
        prog="simulator.policy_calibration",
        description=(
            "δ-калибровка честности политики: подбор shift.mlp/shift.lstm на val-днях"
            " эпохи M3 → artifacts/policy-calibration.json + config.yaml (m6-worlds)."
        ),
    )
    parser.add_argument("--config", default="config.yaml", help="путь к конфигу YAML")
    parser.add_argument("--db", default=None, help="БД эпохи (по умолчанию ml.db из конфига)")
    parser.add_argument(
        "--artifacts-root",
        default=str(ARTIFACTS_ROOT),
        help="корень каталогов артефактов моделей (по умолчанию artifacts)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI (задача 1.5): подбор δ → json → правка конфига → печать.

    Guard'ы (нет каталога семейства, расходящиеся снимки, пустая val-роль, нет
    строки ``shift:`` в конфиге) — до всякой записи: stderr и exit 2, json не
    создан, конфиг не изменён. Повторный запуск детерминирован.
    """

    args = _parse_args(argv)
    try:
        cfg = load_config(args.config)
        payload = _calibrate_payload(
            cfg,
            db_path=cfg.ml.db if args.db is None else args.db,
            artifacts_root=args.artifacts_root,
        )
        # Preflight правки конфига до всякой записи: нет строки — ошибка без json.
        config_path = Path(args.config)
        new_text = _replace_shift_line(config_path.read_text(encoding="utf-8"), payload["shift"])
        json_path = _write_json(payload, args.artifacts_root)
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
