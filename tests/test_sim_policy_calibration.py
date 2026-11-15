"""Тесты δ-калибровки честности (change ``m6-worlds``, задачи 1.1–1.5).

Чистые функции (задача 1.1): доля срабатываний ``fire_fraction`` на рукописных
массивах — монотонность по δ, сохранение исходной частоты при δ = 0; подбор
``fit_delta`` — вывод частоты 0.30 к цели 0.10 в допуске ±1 п.п. и честный
признак недостижения на вырожденном распределении (все p равны, риск §8.4).

Сбор val-примеров (задача 1.2): ключи val-роли лежат в валидационных днях
(границы ``train_days``/``val_days`` из снимка артефакта), наборы пар одинаковы
у обоих семейств — одна машина сбора.

Процедура и артефакты (задача 1.3): json создан, частоты обоих семейств после
калибровки в ``target ± 1 п.п.``, повторный вызов даёт побайтово тот же json;
нет каталога семейства — исключение до записи, json не создан.

Правка конфига (задача 1.4): diff ограничен строкой ``shift:``, round-trip
``load_config`` даёт сдвиги из json; строка не найдена — исключение, файл не
тронут.

CLI (задача 1.5): argv-вызов на мини-артефактах — exit 0, json и конфиг
обновлены; отсутствие артефактов — exit 2, stderr непуст, артефакты не созданы.

Мини-эпоха и артефакты — приём ``tests/test_ml_infer.py``: 3 дня × 288 бакетов
(данные в зонах 0/1), гиперпараметры сходятся за секунды; реальная эпоха не
трогается.
"""

import json
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from common.config import AppConfig, MlCfg, PolicyCfg, load_config
from ml.dataset import SqliteRowSource
from ml.infer import collect_role_examples, load_model_artifact
from ml.train import model_version_for, prepare_training_data, save_model_artifacts, train_model
from simulator.policy_calibration import (
    TOLERANCE,
    calibrate_shifts,
    fire_fraction,
    fit_delta,
    main,
    update_config_shift,
)
from storage.db import connect
from storage.repo import insert_zone_metrics
from storage.schema import init_db

SIM_START = datetime(2026, 11, 2)
"""Старт мира мини-БД (дефолт engine.sim_start конфига, понедельник)."""

CITY_ZONES = tuple(range(6))
"""Зоны города конфига по умолчанию; данные паттерна есть только в 0/1."""

STAIRS = PolicyCfg().stairs
"""Ступени прил. А — дефолты модели конфига (эталон = дефолты)."""

REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
"""Эталонный конфиг репозитория — источник копий для тестов правки."""

NEW_SHIFT = {"mlp": 0.123, "lstm": -0.082}
"""Подобранные δ для тестов правки конфига (не нули — видно саму замену)."""


# ---------------------------------------------------------------------------
# Фабрика мини-эпохи и артефактов (приём tests/test_ml_infer.py)
# ---------------------------------------------------------------------------


def _ml_cfg() -> MlCfg:
    """Тестовые гиперпараметры: роли 2/1/1 дня, окно 1 ч, малая сеть."""

    return MlCfg(
        train_days=2,
        val_days=1,
        test_days=1,
        window_hours=1,
        mlp_hidden=[16, 128],
        lstm_hidden=8,
        dropout=0.0,
        lr=0.01,
        batch_size=128,
        max_epochs=4,
        patience=4,
        seed=42,
    )


def _fill_pattern_db(db_path: Path) -> None:
    """Мини-эпоха с закономерностью: 3 дня × 288 бакетов, зоны 0/1."""

    conn = connect(str(db_path))
    init_db(conn)
    rows: list[dict] = []
    t = SIM_START
    end = SIM_START + timedelta(days=3) - timedelta(minutes=5)
    while t <= end:
        j = int((t - SIM_START).total_seconds()) // 300
        for zone_id in (0, 1):
            requests = 3 + int(round(3.0 * np.sin(2.0 * np.pi * (j % 96) / 96.0))) + zone_id
            free_drivers = 4 + (j % 3) + zone_id
            unfulfilled = 1 if requests >= 6 else 0
            rows.append(
                {
                    "ts": t,
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
            )
        t = t + timedelta(minutes=5)
    insert_zone_metrics(conn, rows)
    conn.close()


@pytest.fixture
def pattern_db(tmp_path: Path) -> Path:
    """Мини-эпоха с закономерностью: 3 дня, зоны 0/1."""

    path = tmp_path / "pattern.db"
    _fill_pattern_db(path)
    return path


def _train_and_save(artifacts_root: Path, db_path: Path, arch: str) -> Path:
    """Обучить архитектуру на мини-эпохе и записать каталог артефакта."""

    ml_cfg = _ml_cfg()
    data = prepare_training_data(db_path, ml_cfg, zones=list(CITY_ZONES), start=SIM_START)
    result = train_model(arch, data, ml_cfg)
    version = model_version_for(arch, today=date(2026, 1, 1))
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
    """Корень артефактов с каталогами обоих семейств."""

    root = tmp_path / "artifacts"
    root.mkdir()
    _train_and_save(root, pattern_db, "mlp")
    _train_and_save(root, pattern_db, "lstm")
    return root


# ---------------------------------------------------------------------------
# 1.1 Чистая функция доли и подбор δ
# ---------------------------------------------------------------------------


def test_fire_fraction_delta_zero_preserves_rate() -> None:
    """δ = 0 сохраняет исходную частоту: доля p ≥ min(p_min) считается вручную."""

    probs = np.array([0.10, 0.39, 0.40, 0.55, 0.80, 0.95])
    # Порог включается: 0.40/0.55/0.80/0.95 из шести срабатывают (4/6 ≈ 0.667).
    manual = float(np.mean(probs >= min(stair.p_min for stair in STAIRS)))
    assert manual == pytest.approx(4 / 6)
    assert fire_fraction(probs, 0.0, STAIRS) == pytest.approx(manual)


def test_fire_fraction_monotone_in_delta() -> None:
    """Доля срабатываний не убывает по δ — основание бинарного поиска."""

    probs = np.linspace(0.05, 0.55, 1000)
    fractions = [fire_fraction(probs, delta, STAIRS) for delta in np.linspace(-1.0, 1.0, 41)]
    assert all(b >= a for a, b in zip(fractions, fractions[1:]))
    assert fractions[0] == pytest.approx(0.0)  # δ = −1 гасит все ступени
    assert fractions[-1] == pytest.approx(1.0)  # δ = +1 зажигает все


def test_fit_delta_brings_030_to_010() -> None:
    """Частота 0.30 выводится к цели 0.10: δ < 0, итог в допуске ±1 п.п. и не 0.30.

    Равномерная сетка на [0.05, 0.55]: при δ = 0 ступень ≥ 0.40 активна в ~30 %,
    шаг доли 1/1000 ≪ допуска — цель достижима.
    """

    probs = np.linspace(0.05, 0.55, 1000)
    assert fire_fraction(probs, 0.0, STAIRS) == pytest.approx(0.30, abs=0.01)

    fit = fit_delta(probs, target=0.10, stairs=STAIRS, tolerance=TOLERANCE)

    assert fit.target_reached is True
    assert fit.delta < 0.0
    assert abs(fit.fire_rate - 0.10) <= TOLERANCE
    assert fit.fire_rate != pytest.approx(0.30)


def test_fit_delta_degenerate_reports_not_reached() -> None:
    """Вырожденное распределение (все p равны): признак недостижения и наилучший δ.

    Все p = 0.5: доля при любом δ равна 0 или 1 — допуск ±0.01 от цели 0.10
    недостижим; фиксируется наилучшая достигнутая частота (0.0) без исключений.
    """

    probs = np.full(10, 0.5)

    fit = fit_delta(probs, target=0.10, stairs=STAIRS, tolerance=TOLERANCE)

    assert fit.target_reached is False
    assert fit.fire_rate == 0.0  # наилучшее из достижимых: гасить все ступени
    assert -1.0 <= fit.delta < min(stair.p_min for stair in STAIRS) - 0.5


# ---------------------------------------------------------------------------
# 1.2 Публичный сбор val-примеров
# ---------------------------------------------------------------------------


def test_collect_val_examples_in_val_days(pattern_db: Path, artifacts_root: Path) -> None:
    """Ключи val-роли лежат в валидационных днях — хвосте train-эпохи
    (для мини-снимка 2/1 — день 2; для эталона 30/2 это дни 29–30)."""

    artifact_mlp_dir = next(iter(artifacts_root.glob("mlp-v1-*")))

    artifact = load_model_artifact(artifact_mlp_dir, n_zones=len(CITY_ZONES))
    examples = collect_role_examples(
        SqliteRowSource(pattern_db),
        artifact.ml_cfg,
        zones=list(CITY_ZONES),
        start=SIM_START,
        role="val",
    )

    first_day, last_day = (
        artifact.ml_cfg.train_days - artifact.ml_cfg.val_days,
        (artifact.ml_cfg.train_days),
    )
    lo = SIM_START + timedelta(days=first_day)
    hi = SIM_START + timedelta(days=last_day)
    assert len(examples.keys) > 0
    assert all(lo <= ts < hi for _, ts in examples.keys)


def test_collect_val_examples_same_pairs_both_families(
    pattern_db: Path, artifacts_root: Path
) -> None:
    """Наборы пар val-роли одинаковы у обоих семейств: снимки согласованы, сборщик один."""

    examples: dict[str, set[tuple[int, datetime]]] = {}
    for family in ("mlp", "lstm"):
        artifact_dir = next(iter(artifacts_root.glob(f"{family}-v1-*")))
        artifact = load_model_artifact(artifact_dir, n_zones=len(CITY_ZONES))
        got = collect_role_examples(
            SqliteRowSource(pattern_db),
            artifact.ml_cfg,
            zones=list(CITY_ZONES),
            start=SIM_START,
            role="val",
        )
        examples[family] = set(got.keys)
    assert examples["mlp"] == examples["lstm"]
    assert len(examples["mlp"]) > 0


def test_collect_role_examples_rejects_unknown_role(pattern_db: Path) -> None:
    """Неизвестная роль — ValueError (роли фиксированы сборщиком датасета)."""

    with pytest.raises(ValueError, match="роль"):
        collect_role_examples(
            SqliteRowSource(pattern_db),
            _ml_cfg(),
            zones=list(CITY_ZONES),
            start=SIM_START,
            role="holdout",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 1.3 Процедура калибровки и json-артефакт
# ---------------------------------------------------------------------------


def test_calibrate_shifts_writes_json(pattern_db: Path, artifacts_root: Path) -> None:
    """json создан: δ обоих семейств, частоты после калибровки в target ± 1 п.п.,
    факт достижения, имена каталогов, границы окна (дни 2–2 мини-снимка), горизонт."""

    cfg = AppConfig(ml=_ml_cfg())

    payload, json_path = calibrate_shifts(cfg, db_path=pattern_db, artifacts_root=artifacts_root)

    assert json_path == artifacts_root / "policy-calibration.json"
    assert json_path.is_file()
    stored = json.loads(json_path.read_text(encoding="utf-8"))
    assert stored == payload
    assert payload["target_fire_rate"] == pytest.approx(0.10)
    assert payload["tolerance"] == pytest.approx(TOLERANCE)
    assert payload["window_days"] == {"first_day": 2, "last_day": 2}
    assert payload["horizon_min"] == cfg.policy.horizon_min
    for family in ("mlp", "lstm"):
        entry = payload["families"][family]
        assert entry["target_reached"] is True
        assert abs(entry["fire_rate_after"] - payload["target_fire_rate"]) <= TOLERANCE
        assert entry["artifact_dir"] == f"{family}-v1-2026-01-01"
    assert payload["shift"]["mlp"] == payload["families"]["mlp"]["delta"]
    assert payload["shift"]["lstm"] == payload["families"]["lstm"]["delta"]
    # Частота до калибровки (δ = 0) — исходная, существенно выше цели.
    assert payload["families"]["mlp"]["fire_rate_before"] > payload["target_fire_rate"]
    assert payload["families"]["lstm"]["fire_rate_before"] > payload["target_fire_rate"]


def test_calibrate_shifts_is_deterministic(pattern_db: Path, artifacts_root: Path) -> None:
    """Повторный вызов даёт побайтово тот же json."""

    cfg = AppConfig(ml=_ml_cfg())
    calibrate_shifts(cfg, db_path=pattern_db, artifacts_root=artifacts_root)
    first = (artifacts_root / "policy-calibration.json").read_bytes()

    calibrate_shifts(cfg, db_path=pattern_db, artifacts_root=artifacts_root)
    second = (artifacts_root / "policy-calibration.json").read_bytes()

    assert first == second


def test_calibrate_shifts_missing_artifact_guard(pattern_db: Path, tmp_path: Path) -> None:
    """Нет каталога семейства lstm — FileNotFoundError до записи, json не создан."""

    root = tmp_path / "artifacts"
    root.mkdir()
    _train_and_save(root, pattern_db, "mlp")  # lstm отсутствует

    cfg = AppConfig(ml=_ml_cfg())
    with pytest.raises(FileNotFoundError, match="lstm"):
        calibrate_shifts(cfg, db_path=pattern_db, artifacts_root=root)
    assert not (root / "policy-calibration.json").exists()


# ---------------------------------------------------------------------------
# 1.4 Точечная правка policy.shift в config.yaml
# ---------------------------------------------------------------------------


def test_update_config_shift_touches_only_shift_line(tmp_path: Path) -> None:
    """Правка меняет ровно строку ``shift:`` (комментарий строки и прочий файл
    сохраняются); round-trip: ``load_config`` даёт записанные сдвиги."""

    config_path = tmp_path / "config.yaml"
    shutil.copy(REFERENCE_CONFIG, config_path)
    before = config_path.read_text(encoding="utf-8").splitlines()

    update_config_shift(config_path, NEW_SHIFT)

    after = config_path.read_text(encoding="utf-8").splitlines()
    assert len(after) == len(before)
    changed = [i for i, (old, new) in enumerate(zip(before, after)) if old != new]
    assert changed == [
        before.index(next(line for line in before if line.lstrip().startswith("shift:")))
    ]
    assert "shift: {mlp: 0.123, lstm: -0.082}" in after[changed[0]]
    assert "# δ-калибровка частоты срабатываний" in after[changed[0]]  # комментарий жив

    loaded = load_config(config_path)
    assert loaded.policy.shift["mlp"] == pytest.approx(NEW_SHIFT["mlp"])
    assert loaded.policy.shift["lstm"] == pytest.approx(NEW_SHIFT["lstm"])


def test_update_config_shift_missing_line_no_write(tmp_path: Path) -> None:
    """Конфиг без строки ``shift:`` в секции policy — исключение, файл не тронут."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text("policy:\n  target_fire_rate: 0.10\n", encoding="utf-8")
    original = config_path.read_bytes()

    with pytest.raises(ValueError, match="shift"):
        update_config_shift(config_path, NEW_SHIFT)

    assert config_path.read_bytes() == original


def test_update_config_shift_missing_section_no_write(tmp_path: Path) -> None:
    """Секции policy нет вовсе — тот же отказ без записи."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    original = config_path.read_bytes()

    with pytest.raises(ValueError, match="shift"):
        update_config_shift(config_path, NEW_SHIFT)

    assert config_path.read_bytes() == original


# ---------------------------------------------------------------------------
# 1.5 CLI python -m simulator.policy_calibration
# ---------------------------------------------------------------------------


def _write_cli_config(path: Path, db_path: Path) -> Path:
    """Конфиг CLI: мини-гиперпараметры ml и policy со строкой shift в flow-формате
    (как в эталоне — правка рассчитана на эту строку)."""

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
    lines = [
        "ml:",
        *[f"  {key}: {json.dumps(value)}" for key, value in ml.items()],
        "policy:",
        "  shift: {mlp: 0.0, lstm: 0.0}  # δ-калибровка частоты срабатываний",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_cli_calibrates(pattern_db: Path, artifacts_root: Path, tmp_path: Path, capsys) -> None:
    """argv-вызов на мини-артефактах: exit 0, json и конфиг обновлены, печать δ."""

    config_path = _write_cli_config(tmp_path / "config.yaml", pattern_db)

    rc = main(
        [
            "--config",
            str(config_path),
            "--db",
            str(pattern_db),
            "--artifacts-root",
            str(artifacts_root),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "family=mlp delta=" in captured.out
    assert "family=lstm delta=" in captured.out

    payload = json.loads((artifacts_root / "policy-calibration.json").read_text(encoding="utf-8"))
    loaded = load_config(config_path)
    assert loaded.policy.shift["mlp"] == pytest.approx(payload["shift"]["mlp"])
    assert loaded.policy.shift["lstm"] == pytest.approx(payload["shift"]["lstm"])
    assert "delta=0.000" not in captured.out  # δ подобраны, а не стартовые нули


def test_cli_missing_artifacts_exit_2(pattern_db: Path, tmp_path: Path, capsys) -> None:
    """Нет каталогов артефактов: exit 2, stderr непуст, json и конфиг не созданы/не тронуты."""

    config_path = _write_cli_config(tmp_path / "config.yaml", pattern_db)
    original = config_path.read_bytes()
    empty_root = tmp_path / "empty-artifacts"
    empty_root.mkdir()

    rc = main(
        [
            "--config",
            str(config_path),
            "--db",
            str(pattern_db),
            "--artifacts-root",
            str(empty_root),
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.err.strip() != ""
    assert not (empty_root / "policy-calibration.json").exists()
    assert config_path.read_bytes() == original
