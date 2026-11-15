"""Тесты common/config.py: дефолты секций, эталонный ``config.yaml``, запрет
неизвестных ключей, чтение YAML.

Покрывают сценарии «Все секции скелета присутствуют», «Seed читается», «Конфиг
целиком из дефолтов», «Частичный конфиг дозаполняется дефолтами», «Неизвестный
ключ в корне», «Неизвестный ключ внутри секции», «Эталонный config.yaml валиден»
и «YAML не словарём отклоняется» из specs change'а m1-config (capability
config-loading).
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from common.config import (
    AppCfg,
    AppConfig,
    DriversCfg,
    EngineCfg,
    MlCfg,
    PolicyCfg,
    PolicyStairCfg,
    RepositionCfg,
    StorageCfg,
    load_config,
)

# Эталонный конфиг ищется относительно файла теста, не от cwd (решение
# design.md 7): pytest запускается из корня, но путь не должен на это опираться.
REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"

# Все тринадцать полей корневой модели — секции скелета §5 плана M1, ``app``
# change'а replay-server, ``ml`` change'а m3-dataset и ``policy`` change'а
# m6-policy.
EXPECTED_SECTIONS = {
    "seed",
    "storage",
    "city",
    "population",
    "demand",
    "drivers",
    "weather",
    "events",
    "pricing",
    "engine",
    "app",
    "ml",
    "policy",
}

# Значения YAML, не являющиеся отображением секций (сценарий «YAML не
# словарём отклоняется»): список и скаляр.
NON_DICT_YAML = ["- a\n- b\n", "42\n"]


def test_defaults_full_config() -> None:
    """Конфиг целиком из дефолтов валиден: все секции и стартовые значения §5."""
    cfg = AppConfig()
    assert cfg.seed == 42
    assert cfg.storage.db_path == "data/taxi.db"
    assert set(AppConfig.model_fields) == EXPECTED_SECTIONS
    # Девять секций (кроме скалярного seed) — вложенные модели с дефолтами.
    assert cfg.storage == StorageCfg()
    assert cfg.drivers == DriversCfg()
    assert cfg.engine == EngineCfg()


def test_reference_config_yaml_valid() -> None:
    """Эталонный config.yaml в корне репо валиден и совпадает с дефолтами."""
    cfg = load_config(REFERENCE_CONFIG)
    assert cfg == AppConfig()
    assert cfg.seed == 42


@pytest.mark.parametrize("payload", [{"sied": 1}])
def test_unknown_root_key_rejected(payload: dict) -> None:
    """Неизвестный ключ в корне (``sied``) отклоняется с именем ключа."""
    with pytest.raises(ValidationError, match="sied"):
        AppConfig.model_validate(payload)


def test_unknown_section_key_rejected() -> None:
    """Неизвестный ключ внутри секции отклоняется с путём ``drivers.coun``."""
    with pytest.raises(ValidationError) as excinfo:
        AppConfig.model_validate({"drivers": {"coun": 1}})
    message = str(excinfo.value)
    assert "drivers.coun" in message  # путь секция.ключ — решение design.md 2
    assert "Extra inputs" in message or "extra_forbidden" in message


def test_seed_read_from_file(tmp_path: Path) -> None:
    """``seed`` читается из YAML-файла."""
    path = tmp_path / "config.yaml"
    path.write_text("seed: 7\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.seed == 7


def test_partial_config_filled_with_defaults(tmp_path: Path) -> None:
    """Частичный конфиг (только ``seed``) дозаполняется дефолтами секций."""
    path = tmp_path / "config.yaml"
    path.write_text("seed: 7\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.seed == 7  # заданная секция — из файла
    assert cfg.drivers == DriversCfg()  # незаданные — дефолты
    assert cfg.engine == EngineCfg()


@pytest.mark.parametrize("text", NON_DICT_YAML)
def test_non_dict_yaml_rejected(tmp_path: Path, text: str) -> None:
    """YAML-список и скаляр отклоняются ``ValueError`` «словарём секций».

    Ошибка — именно ``ValueError`` с понятным текстом (решение design.md 5),
    а не внутренний ``TypeError`` и не ``ValidationError`` pydantic: в pydantic
    v2 ``ValidationError`` — подкласс ``ValueError``, поэтому тип проверяется
    явно.
    """
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="словарём секций") as excinfo:
        load_config(path)
    assert type(excinfo.value) is ValueError


def test_app_section_defaults() -> None:
    """Конфиг без секции ``app`` грузится: значения из дефолтов модели (задача 1.2)."""

    cfg = AppConfig()
    assert cfg.app == AppCfg(host="127.0.0.1", port=8000, days=30)


def test_app_section_from_file(tmp_path: Path) -> None:
    """Секция ``app`` читается из YAML-файла, незаданные ключи — дефолты."""

    path = tmp_path / "config.yaml"
    path.write_text("app:\n  host: 0.0.0.0\n  port: 9000\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.app.host == "0.0.0.0"
    assert cfg.app.port == 9000
    assert cfg.app.days == 30


def test_app_unknown_key_rejected() -> None:
    """Неизвестный ключ внутри секции ``app`` отклоняется с путём ``app.hos``."""

    with pytest.raises(ValidationError, match="app.hos"):
        AppConfig.model_validate({"app": {"hos": "x"}})


def test_ml_section_defaults() -> None:
    """Конфиг без секции ``ml`` грузится: значения — приложение А плана M3 (задача 1.1)."""

    cfg = AppConfig()
    assert cfg.ml == MlCfg(
        db="data/m3_epoch.db",
        train_days=30,
        val_days=2,
        test_days=7,
        window_hours=3,  # победитель свипа 3/6/12 (m3-train-cli)
        zone_embed_dim=8,
        mlp_hidden=[256, 128],
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.1,
        lr=0.001,
        batch_size=256,
        max_epochs=100,
        patience=8,
        cls_weight=1.0,
        seed=42,
    )


def test_ml_section_from_file(tmp_path: Path) -> None:
    """Секция ``ml`` читается из YAML-файла, незаданные ключи — дефолты."""

    path = tmp_path / "config.yaml"
    path.write_text("ml:\n  db: data/other.db\n  window_hours: 12\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.ml.db == "data/other.db"
    assert cfg.ml.window_hours == 12
    assert cfg.ml.train_days == 30  # незаданные ключи — дефолты


def test_ml_unknown_key_rejected() -> None:
    """Неизвестный ключ внутри секции ``ml`` отклоняется с путём ``ml.wind``."""

    with pytest.raises(ValidationError, match="ml.wind"):
        AppConfig.model_validate({"ml": {"wind_hours": 6}})


def test_policy_section_defaults() -> None:
    """Конфиг без секции ``policy`` грузится: значения — приложение А плана M5+M6
    (задача 1.1), δ-сдвиги — результат калибровки change ``m6-worlds``."""

    cfg = AppConfig()
    assert cfg.policy == PolicyCfg(
        horizon_min=15,
        stairs=[
            PolicyStairCfg(p_min=0.80, mult=1.50),
            PolicyStairCfg(p_min=0.60, mult=1.30),
            PolicyStairCfg(p_min=0.40, mult=1.15),
        ],
        total_cap=2.0,
        reactive_mult=1.30,
        shift={"mlp": -0.453, "lstm": -0.438},
        target_fire_rate=0.10,
        worlds_dir="data/worlds",
    )


def test_policy_section_from_file(tmp_path: Path) -> None:
    """Секция ``policy`` читается из YAML-файла, незаданные ключи — дефолты.

    Покрывает сценарий «Значения из файла переопределяют дефолты» спеки
    simulator-policy: две заданные ступени и частичный ``shift`` — незаданная
    модель получает δ по умолчанию 0.0.
    """

    path = tmp_path / "config.yaml"
    path.write_text(
        "policy:\n"
        "  stairs:\n"
        "    - {p_min: 0.70, mult: 1.25}\n"
        "    - {p_min: 0.50, mult: 1.10}\n"
        "  shift: {mlp: -0.05}\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.policy.stairs == [
        PolicyStairCfg(p_min=0.70, mult=1.25),
        PolicyStairCfg(p_min=0.50, mult=1.10),
    ]
    assert cfg.policy.shift["mlp"] == -0.05
    assert cfg.policy.shift["lstm"] == -0.438  # незаданная модель — калиброванный дефолт
    assert cfg.policy.horizon_min == 15  # незаданные ключи секции — дефолты


@pytest.mark.parametrize(
    ("payload", "key_path"),
    [
        ({"stairs": [{"p_min": 1.4, "mult": 1.5}]}, "policy.stairs.0.p_min"),
        ({"stairs": [{"p_min": 0.6, "mult": 0.9}]}, "policy.stairs.0.mult"),
        ({"stairs": []}, "policy.stairs"),
    ],
)
def test_policy_invalid_stairs_rejected(payload: dict, key_path: str) -> None:
    """Некорректные ступени отклоняются с путём до неверного ключа.

    Покрывает сценарий «Некорректные ступени отклоняются» спеки
    simulator-policy: ``p_min`` вне (0, 1), ``mult`` < 1 и пустой список.
    """

    with pytest.raises(ValidationError) as excinfo:
        AppConfig.model_validate({"policy": payload})
    message = str(excinfo.value)
    assert key_path in message  # путь до ключа — как у запрета неизвестных ключей


# --- Секция отклика предложения drivers.reposition (change m6-supply-response) ---


def test_reposition_defaults_without_subsection(tmp_path: Path) -> None:
    """Старый конфиг без ``drivers.reposition`` грузится: дефолты, отклик выключен."""

    path = tmp_path / "config.yaml"
    path.write_text("drivers: {count: 380, wait_limit_min: 6}\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.drivers.reposition == RepositionCfg()
    assert cfg.drivers.reposition.enabled is False


def test_reposition_partial_subsection(tmp_path: Path) -> None:
    """Частичная подсекция: заданные поля из файла, остальные — дефолты."""

    path = tmp_path / "config.yaml"
    path.write_text(
        "drivers:\n  reposition:\n    enabled: true\n    max_share: 0.5\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.drivers.reposition.enabled is True
    assert cfg.drivers.reposition.max_share == 0.5
    assert cfg.drivers.reposition.every_min == 5
    assert cfg.drivers.reposition.min_gain == 0.10


def test_reposition_unknown_key_rejected() -> None:
    """Неизвестный ключ внутри ``reposition`` отклоняется с полным путём."""

    with pytest.raises(ValidationError, match="drivers.reposition.max_shar"):
        AppConfig.model_validate({"drivers": {"reposition": {"max_shar": 0.2}}})


@pytest.mark.parametrize(
    "payload",
    [
        {"max_share": 1.5},
        {"every_min": 0},
        {"min_gain": 0.0},
        {"sensitivity": 0.0},
        {"max_reach_min": 0},
    ],
)
def test_reposition_invalid_values_rejected(payload: dict) -> None:
    """Значения вне допустимых диапазонов отклоняются ошибкой валидации."""

    with pytest.raises(ValidationError):
        AppConfig.model_validate({"drivers": {"reposition": payload}})
