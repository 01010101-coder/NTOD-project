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

from common.config import AppConfig, DriversCfg, EngineCfg, StorageCfg, load_config

# Эталонный конфиг ищется относительно файла теста, не от cwd (решение
# design.md 7): pytest запускается из корня, но путь не должен на это опираться.
REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"

# Все десять полей корневой модели — секции скелета §5 плана M1.
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
