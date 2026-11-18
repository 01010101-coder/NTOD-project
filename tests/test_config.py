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
    CalloutCfg,
    DispatchCfg,
    DriversCfg,
    EngineCfg,
    MlCfg,
    OnlineShareCfg,
    PlanningCfg,
    PolicyCfg,
    PolicyStairCfg,
    RepositionCfg,
    ShiftsCfg,
    ShocksCfg,
    SignalCfg,
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


def test_ml_deficit_label_default_legacy(tmp_path: Path) -> None:
    """Конфиг без ключа ``deficit_label`` (дефолты и файл) даёт ``legacy``."""

    assert AppConfig().ml.deficit_label == "legacy"
    path = tmp_path / "config.yaml"
    path.write_text("ml:\n  db: data/other.db\n", encoding="utf-8")
    assert load_config(path).ml.deficit_label == "legacy"


def test_ml_deficit_label_harm_accepted(tmp_path: Path) -> None:
    """Значение ``harm`` принимается и в модели, и из YAML-файла."""

    assert AppConfig.model_validate({"ml": {"deficit_label": "harm"}}).ml.deficit_label == "harm"
    path = tmp_path / "config.yaml"
    path.write_text("ml:\n  deficit_label: harm\n", encoding="utf-8")
    assert load_config(path).ml.deficit_label == "harm"


def test_ml_deficit_label_unknown_rejected() -> None:
    """Значение вне ``{legacy, harm}`` отклоняется с путём ``ml.deficit_label``."""

    with pytest.raises(ValidationError, match="ml.deficit_label"):
        AppConfig.model_validate({"ml": {"deficit_label": "loss"}})


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
        shift={"mlp": -0.719, "lstm": -0.688},
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
    assert cfg.policy.shift["lstm"] == -0.688  # незаданная модель — калиброванный дефолт
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


# --- Подсекция диспатча drivers.dispatch (change world-dispatch-radius) ---


def test_dispatch_defaults_without_subsection(tmp_path: Path) -> None:
    """Старый конфиг без ``drivers.dispatch`` грузится: механизмы выключены."""

    path = tmp_path / "config.yaml"
    path.write_text("drivers: {count: 380, wait_limit_min: 6}\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.drivers.dispatch == DispatchCfg()
    assert cfg.drivers.dispatch.max_pickup_min is None
    assert cfg.drivers.dispatch.cancel_prob_per_min == 0.0


def test_dispatch_partial_subsection(tmp_path: Path) -> None:
    """Частичная подсекция: заданное поле из файла, остальные — дефолты."""

    path = tmp_path / "config.yaml"
    path.write_text(
        "drivers:\n  dispatch:\n    max_pickup_min: 12\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.drivers.dispatch.max_pickup_min == 12
    assert cfg.drivers.dispatch.cancel_grace_min == 5
    assert cfg.drivers.dispatch.cancel_prob_per_min == 0.0
    assert cfg.drivers.dispatch.cancel_max_prob == 0.9


def test_dispatch_unknown_key_rejected() -> None:
    """Неизвестный ключ внутри ``dispatch`` отклоняется с полным путём."""

    with pytest.raises(ValidationError, match="drivers.dispatch.max_pickup"):
        AppConfig.model_validate({"drivers": {"dispatch": {"max_pickup": 10}}})


@pytest.mark.parametrize(
    "payload",
    [
        {"max_pickup_min": 0},
        {"cancel_prob_per_min": -0.1},
        {"cancel_max_prob": 1.5},
    ],
)
def test_dispatch_invalid_values_rejected(payload: dict) -> None:
    """Значения вне допустимых диапазонов отклоняются ошибкой валидации."""

    with pytest.raises(ValidationError):
        AppConfig.model_validate({"drivers": {"dispatch": payload}})


# --- Подсекция смен drivers.shifts (change world-driver-shifts) ---

DEFAULT_WEEKDAY_SHARE = [
    0.40,
    0.35,
    0.33,
    0.33,
    0.33,
    0.40,
    0.55,
    0.70,
    0.80,
    0.80,
    0.75,
    0.70,
    0.70,
    0.70,
    0.70,
    0.75,
    0.80,
    0.85,
    0.85,
    0.80,
    0.72,
    0.65,
    0.55,
    0.45,
]

DEFAULT_WEEKEND_SHARE = [
    0.50,
    0.45,
    0.40,
    0.35,
    0.33,
    0.33,
    0.40,
    0.50,
    0.60,
    0.65,
    0.70,
    0.70,
    0.70,
    0.70,
    0.70,
    0.72,
    0.75,
    0.80,
    0.80,
    0.78,
    0.72,
    0.68,
    0.62,
    0.55,
]


def test_shifts_defaults_without_subsection(tmp_path: Path) -> None:
    """Старый конфиг без ``drivers.shifts`` грузится: дефолты, смены выключены."""

    path = tmp_path / "config.yaml"
    path.write_text("drivers: {count: 380, wait_limit_min: 6}\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.drivers.shifts == ShiftsCfg()
    assert cfg.drivers.shifts.enabled is False
    assert cfg.drivers.shifts.online_share.weekday == DEFAULT_WEEKDAY_SHARE
    assert cfg.drivers.shifts.online_share.weekend == DEFAULT_WEEKEND_SHARE


def test_shifts_partial_subsection(tmp_path: Path) -> None:
    """Сценарий «Частичная подсекция»: задан только ``enabled``, профили — дефолты."""

    path = tmp_path / "config.yaml"
    path.write_text("drivers:\n  shifts:\n    enabled: true\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.drivers.shifts.enabled is True
    assert cfg.drivers.shifts.online_share.weekday == DEFAULT_WEEKDAY_SHARE
    assert cfg.drivers.shifts.online_share.weekend == DEFAULT_WEEKEND_SHARE


def test_shifts_reference_config_equals_defaults() -> None:
    """Эталонный ``config.yaml`` содержит подсекцию, равную дефолтам (задача 1.2)."""

    cfg = load_config(REFERENCE_CONFIG)
    assert cfg.drivers.shifts == ShiftsCfg()


def test_shifts_profile_wrong_length_rejected() -> None:
    """Сценарий «Профиль неверной длины отклоняется»: путь до поля в ошибке."""

    with pytest.raises(ValidationError, match="online_share.weekday"):
        AppConfig.model_validate(
            {"drivers": {"shifts": {"online_share": {"weekday": DEFAULT_WEEKDAY_SHARE[:23]}}}}
        )


@pytest.mark.parametrize("bad_value", [0.0, 1.2])
def test_shifts_profile_value_out_of_range_rejected(bad_value: float) -> None:
    """Сценарий «Доля вне диапазона отклоняется»: 0 и 1.2."""

    profile = list(DEFAULT_WEEKDAY_SHARE)
    profile[3] = bad_value
    with pytest.raises(ValidationError, match="online_share.weekday"):
        AppConfig.model_validate({"drivers": {"shifts": {"online_share": {"weekday": profile}}}})


def test_shifts_unknown_key_rejected() -> None:
    """Сценарий «Неизвестный ключ отклоняется»: путь ``online_share.holidays``."""

    with pytest.raises(ValidationError, match=r"drivers\.shifts\.online_share\.holidays"):
        AppConfig.model_validate(
            {"drivers": {"shifts": {"online_share": {"holidays": [1.0] * 24}}}}
        )


def test_online_share_defaults_are_independent_lists() -> None:
    """Профили по умолчанию — новые списки на каждый экземпляр (нет общей мутации)."""

    first, second = OnlineShareCfg(), OnlineShareCfg()
    first.weekday[0] = 1.0
    assert second.weekday[0] == DEFAULT_WEEKDAY_SHARE[0]


# ---------------------------------------------------------------------------
# ml.planning — секция прогноза планирования (change ml-planning-forecast)
# ---------------------------------------------------------------------------


def test_planning_defaults_without_subsection(tmp_path: Path) -> None:
    """Старый конфиг без ``ml.planning`` грузится: дефолты горизонтов и окна."""

    path = tmp_path / "config.yaml"
    path.write_text("ml:\n  db: data/m3_epoch.db\n  seed: 42\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.ml.planning.horizons_min == [60, 120]
    assert cfg.ml.planning.window_min == 30
    assert cfg.ml.planning == PlanningCfg()


def test_planning_partial_subsection(tmp_path: Path) -> None:
    """Задан только ``window_min`` — остальные поля берутся из дефолтов."""

    path = tmp_path / "config.yaml"
    path.write_text("ml:\n  planning:\n    window_min: 45\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.ml.planning.window_min == 45
    assert cfg.ml.planning.horizons_min == [60, 120]
    assert cfg.ml.planning.artifacts_root == "artifacts/planning"


def test_planning_reference_config_equals_defaults() -> None:
    """Эталонный ``config.yaml`` содержит подсекцию, равную дефолтам (задача 1.1)."""

    cfg = load_config(REFERENCE_CONFIG)
    assert cfg.ml.planning == PlanningCfg()


@pytest.mark.parametrize("bad_horizons", [[120, 60], [], [62]])
def test_planning_bad_horizons_rejected(bad_horizons: list[int]) -> None:
    """Неверные ``horizons_min`` отклоняются с путём до ключа (спека config-loading)."""

    with pytest.raises(ValidationError, match=r"ml\.planning\.horizons_min"):
        AppConfig.model_validate({"ml": {"planning": {"horizons_min": bad_horizons}}})


@pytest.mark.parametrize("bad_window", [0, 7, 125])
def test_planning_bad_window_rejected(bad_window: int) -> None:
    """Окно цели вне [5, 120] или не кратное 5 отклоняется с путём до ключа."""

    with pytest.raises(ValidationError, match=r"ml\.planning\.window_min"):
        AppConfig.model_validate({"ml": {"planning": {"window_min": bad_window}}})


def test_planning_unknown_key_rejected() -> None:
    """Неизвестный ключ подсекции отклоняется с полным путём."""

    with pytest.raises(ValidationError, match=r"ml\.planning\.horizon_min"):
        AppConfig.model_validate({"ml": {"planning": {"horizon_min": 60}}})


# --- Подсекция шоков спроса demand.shocks (change world-demand-shocks) ---


def test_shocks_defaults_without_subsection(tmp_path: Path) -> None:
    """Сценарий «Старый конфиг без подсекции»: шоки выключены, поля — дефолты."""

    path = tmp_path / "config.yaml"
    path.write_text("demand: {base_rate: 2.0}\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.demand.shocks == ShocksCfg()
    assert cfg.demand.shocks.enabled is False
    assert cfg.demand.shocks.city_sigma == 0.12
    assert cfg.demand.shocks.city_tau_min == 240
    assert cfg.demand.shocks.zone_sigma == 0.10
    assert cfg.demand.shocks.zone_tau_min == 120
    assert cfg.demand.shocks.day_sigma == 0.08


def test_shocks_partial_subsection(tmp_path: Path) -> None:
    """Сценарий «Частичная подсекция»: заданные поля из файла, остальные — дефолты."""

    path = tmp_path / "config.yaml"
    path.write_text(
        "demand:\n  shocks:\n    enabled: true\n    city_sigma: 0.2\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.demand.shocks.enabled is True
    assert cfg.demand.shocks.city_sigma == 0.2
    assert cfg.demand.shocks.city_tau_min == 240
    assert cfg.demand.shocks.zone_sigma == 0.10
    assert cfg.demand.shocks.zone_tau_min == 120
    assert cfg.demand.shocks.day_sigma == 0.08


def test_shocks_reference_config_equals_defaults() -> None:
    """Эталонный ``config.yaml`` содержит подсекцию, равную дефолтам (задача 1.2)."""

    cfg = load_config(REFERENCE_CONFIG)
    assert cfg.demand.shocks == ShocksCfg()


# --- Подсекция вызова подработчиков drivers.callout (change world-elastic-supply) ---


def test_callout_defaults_without_subsection(tmp_path: Path) -> None:
    """Сценарий «Старый конфиг без подсекции»: дефолты, вызов выключен."""

    path = tmp_path / "config.yaml"
    path.write_text("drivers: {count: 380, wait_limit_min: 6}\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.drivers.callout == CalloutCfg()
    assert cfg.drivers.callout.enabled is False
    assert cfg.drivers.callout.part == 60
    assert cfg.drivers.callout.lag_min == 20
    assert cfg.drivers.callout.session_min == 120
    assert cfg.drivers.callout.every_min == 5
    assert cfg.drivers.callout.min_gain == 0.10
    assert cfg.drivers.callout.sensitivity == 0.3
    assert cfg.drivers.callout.max_share == 0.3


def test_callout_partial_subsection(tmp_path: Path) -> None:
    """Сценарий «Частичная подсекция»: заданы ``enabled`` и ``lag_min``, остальное — дефолты."""

    path = tmp_path / "config.yaml"
    path.write_text(
        "drivers:\n  callout:\n    enabled: true\n    lag_min: 30\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.drivers.callout.enabled is True
    assert cfg.drivers.callout.lag_min == 30
    assert cfg.drivers.callout.part == 60
    assert cfg.drivers.callout.session_min == 120
    assert cfg.drivers.callout.every_min == 5
    assert cfg.drivers.callout.min_gain == 0.10
    assert cfg.drivers.callout.sensitivity == 0.3
    assert cfg.drivers.callout.max_share == 0.3


def test_callout_reference_config_equals_defaults() -> None:
    """Эталонный ``config.yaml`` содержит подсекцию, равную дефолтам (задача 1.2)."""

    cfg = load_config(REFERENCE_CONFIG)
    assert cfg.drivers.callout == CalloutCfg()
    assert cfg.drivers.callout.enabled is False


@pytest.mark.parametrize(
    "payload",
    [
        {"city_tau_min": 0},
        {"zone_tau_min": 0},
        {"zone_sigma": -0.1},
        {"city_sigma": -0.1},
        {"day_sigma": -0.1},
    ],
)
def test_shocks_invalid_values_rejected(payload: dict) -> None:
    """Сценарий «Недопустимое значение отклоняется»: путь до поля в ошибке."""

    key = next(iter(payload))
    with pytest.raises(ValidationError, match=rf"demand\.shocks\.{key}"):
        AppConfig.model_validate({"demand": {"shocks": payload}})


def test_shocks_unknown_key_rejected() -> None:
    """Сценарий «Неизвестный ключ отклоняется»: путь ``demand.shocks.holiday_mult``."""

    with pytest.raises(ValidationError, match=r"demand\.shocks\.holiday_mult"):
        AppConfig.model_validate({"demand": {"shocks": {"holiday_mult": 1.5}}})


@pytest.mark.parametrize(
    "payload",
    [
        {"lag_min": 0},
        {"session_min": 0},
        {"every_min": 0},
        {"part": -1},
        {"min_gain": -0.1},
        {"sensitivity": -0.1},
        {"max_share": 1.5},
        {"max_share": 0.0},
    ],
)
def test_callout_invalid_values_rejected(payload: dict) -> None:
    """Сценарий «Недопустимое значение отклоняется»: путь до поля в ошибке."""

    field = next(iter(payload))
    with pytest.raises(ValidationError, match=rf"drivers\.callout\.{field}"):
        AppConfig.model_validate({"drivers": {"callout": payload}})


def test_callout_unknown_key_rejected() -> None:
    """Сценарий «Неизвестный ключ отклоняется»: путь ``drivers.callout.bonus_rub``."""

    with pytest.raises(ValidationError, match=r"drivers\.callout\.bonus_rub"):
        AppConfig.model_validate({"drivers": {"callout": {"bonus_rub": 100.0}}})


# --- Канал и сигнал водителям (change policy-driver-signal) ------------------


def test_policy_channel_and_signal_defaults(tmp_path: Path) -> None:
    """Дефолты канала и сигнала: конфиг без ключей грузится (сценарий спеки)."""

    assert AppConfig().policy.channel == "price"
    assert AppConfig().policy.signal == SignalCfg(
        horizon_min=60,
        mult=1.30,
        target_fire_rate=0.15,
        threshold={"mlp": 0.5, "lstm": 0.5},
    )

    path = tmp_path / "config.yaml"
    path.write_text("policy:\n  horizon_min: 15\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.policy.channel == "price"
    assert cfg.policy.signal.horizon_min == 60
    assert cfg.policy.signal.mult == 1.30
    assert cfg.policy.signal.target_fire_rate == 0.15
    assert cfg.policy.signal.threshold == {"mlp": 0.5, "lstm": 0.5}


def test_policy_channel_unknown_rejected() -> None:
    """Неизвестный канал отклоняется с путём ``policy.channel`` (сценарий спеки)."""

    with pytest.raises(ValidationError, match=r"policy\.channel"):
        AppConfig.model_validate({"policy": {"channel": "radio"}})


@pytest.mark.parametrize("channel", ["price", "heatmap", "both"])
def test_policy_channel_accepted(channel: str) -> None:
    """Три канала спеки принимаются; значение доходит до модели."""

    assert AppConfig.model_validate({"policy": {"channel": channel}}).policy.channel == channel


@pytest.mark.parametrize(
    "payload",
    [
        {"horizon_min": 0},
        {"horizon_min": 62},
        {"mult": 1.0},
        {"mult": 0.9},
        {"target_fire_rate": 0.0},
        {"target_fire_rate": 1.0},
        {"threshold": {"mlp": 1.2}},
        {"threshold": {"lstm": -0.1}},
    ],
)
def test_policy_signal_invalid_values_rejected(payload: dict) -> None:
    """Значения вне ограничений спеки отклоняются с путём до ключа."""

    key = next(iter(payload))
    with pytest.raises(ValidationError, match=rf"policy\.signal\.{key}"):
        AppConfig.model_validate({"policy": {"signal": payload}})


def test_policy_signal_unknown_key_rejected() -> None:
    """Неизвестный ключ ``policy.signal`` отклоняется с путём до ключа (спека)."""

    with pytest.raises(ValidationError, match=r"policy\.signal\.clickbait"):
        AppConfig.model_validate({"policy": {"signal": {"clickbait": 0.5}}})


def test_policy_channel_and_signal_reference_config_equals_defaults() -> None:
    """Эталонный ``config.yaml``: ``channel: price`` и сигнал, равный дефолтам."""

    cfg = load_config(REFERENCE_CONFIG)
    assert cfg.policy.channel == "price"
    assert cfg.policy.signal == SignalCfg()


# --- Корень артефактов ml.artifacts_root (change world-final-config) ---------


def test_ml_artifacts_root_default_artifacts(tmp_path: Path) -> None:
    """Дефолт корня артефактов — ``artifacts``: пути прежние (сценарий спеки)."""

    assert AppConfig().ml.artifacts_root == "artifacts"
    path = tmp_path / "config.yaml"
    path.write_text("ml:\n  db: data/other.db\n", encoding="utf-8")
    assert load_config(path).ml.artifacts_root == "artifacts"


def test_ml_artifacts_root_from_file(tmp_path: Path) -> None:
    """Явный корень читается из YAML и не трогает остальные поля ``ml``."""

    path = tmp_path / "config.yaml"
    path.write_text("ml:\n  artifacts_root: artifacts/final\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.ml.artifacts_root == "artifacts/final"
    assert cfg.ml.db == "data/m3_epoch.db"
