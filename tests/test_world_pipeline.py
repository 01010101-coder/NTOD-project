"""Сквозной конвейер `make world` и параметр CONFIG (change world-final-config).

Сценарии спеки ``build-tooling`` проверяются на малом мире (4 + 2 суток,
``max_epochs`` 2, две зоны): временный конфиг финального типа (все механизмы
включены, канал ``both``, метка ``harm``) с путями во временном каталоге.
Конвейер запускается настоящим make из корня проекта — проверяется порядок и
факт создания всех артефактов, а также остановка на первом шаге с ошибкой.
"""

import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAKE = "make"

MICRO_ZONES = [
    {
        "id": 0,
        "name": "a",
        "archetype": "residential",
        "attract_origin": 1.0,
        "attract_dest": 1.0,
        "map": {"x": 0.0, "y": 0.0},
    },
    {
        "id": 1,
        "name": "b",
        "archetype": "business",
        "attract_origin": 1.0,
        "attract_dest": 1.0,
        "map": {"x": 10.0, "y": 0.0},
    },
]


def _write_config(tmp_path: Path, *, ml_db: Path | None = None) -> Path:
    """Временный конфиг малого мира финального типа с путями в ``tmp_path``.

    ``ml_db`` переопределяет путь БД эпохи — для сценария «ошибка шага
    останавливает конвейер» (родитель пути — обычный файл, SQLite не откроет БД).
    """

    root = tmp_path / "world"
    data = {
        "seed": 7,
        "storage": {"db_path": str(root / "taxi.db")},
        "city": {
            "zones": MICRO_ZONES,
            "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 10.0}],
        },
        "population": {"scale": 2.0},
        "drivers": {
            "count": 8,
            "wait_limit_min": 6,
            "reposition": {"enabled": True},
            "dispatch": {
                "max_pickup_min": 12,
                "cancel_grace_min": 5,
                "cancel_prob_per_min": 0.06,
                "cancel_max_prob": 0.9,
            },
            "shifts": {"enabled": True},
            "callout": {
                "enabled": True,
                "part": 6,
                "lag_min": 60,
                "session_min": 120,
            },
        },
        "demand": {"base_rate": 2.0, "shocks": {"enabled": True}},
        "weather": {
            "rain_day_chance": 0.0,
            "episode_start_h": [0, 0],
            "episode_duration_h": [1, 1],
        },
        "pricing": {
            "elasticity_alpha": 0.5,
            "surge_injection": {"day_chance": 0.0},
        },
        "ml": {
            "db": str(root / "epoch.db" if ml_db is None else ml_db),
            "artifacts_root": str(root / "artifacts"),
            "train_days": 4,
            "val_days": 1,
            "test_days": 2,
            "window_hours": 1,
            "zone_embed_dim": 4,
            "mlp_hidden": [16, 128],
            "lstm_hidden": 8,
            "dropout": 0.0,
            "lr": 0.01,
            "batch_size": 128,
            "max_epochs": 2,
            "patience": 2,
            "seed": 42,
            "deficit_label": "harm",
            "planning": {
                "horizons_min": [60, 120],
                "window_min": 30,
                "artifacts_root": str(root / "planning"),
            },
        },
        "policy": {
            "worlds_dir": str(root / "worlds"),
            "channel": "both",
            # Строки flow-формата: калибровки правят их точечно, сохраняя отступы.
            "shift": {"mlp": 0.0, "lstm": 0.0},
            "signal": {"threshold": {"mlp": 0.5, "lstm": 0.5}},
        },
    }
    (tmp_path / "world").mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, default_flow_style=None), encoding="utf-8"
    )
    return path


def _run_make(*args: str) -> subprocess.CompletedProcess:
    """Запустить make из корня проекта (переменные окружения не подменяются)."""

    return subprocess.run(
        [MAKE, *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def test_make_world_small_end_to_end(tmp_path: Path) -> None:
    """`make world CONFIG=<малый мир>`: exit 0 и все артефакты конвейера (3.1)."""

    config = _write_config(tmp_path)
    root = tmp_path / "world"
    result = _run_make("world", f"CONFIG={config}")
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]

    assert (root / "epoch.db").is_file()
    for family in ("mlp", "lstm"):
        model_dirs = list((root / "artifacts").glob(f"{family}-v1-*"))
        assert len(model_dirs) == 1, f"нет каталога M3 {family}"
        assert {p.name for p in model_dirs[0].iterdir()} == {
            "weights.pt",
            "scalers.pkl",
            "loss.json",
            "metrics.json",
        }
        assert len(list((root / "planning").glob(f"{family}-v1-*"))) == 1
    assert (root / "artifacts" / "policy-calibration.json").is_file()
    assert (root / "artifacts" / "signal-calibration.json").is_file()
    for branch in ("a-base", "b-reactive", "c-mlp", "d-lstm"):
        assert (root / "worlds" / f"{branch}.db").is_file()
        assert (root / "worlds" / f"{branch}.meta.json").is_file()
    # Сводка plan-eval — таблица «модель × горизонт» в stdout.
    assert "model" in result.stdout and "horizon" in result.stdout
    assert "mlp" in result.stdout and "календарь" in result.stdout
    # Калибровки точечно обновили именно временный конфиг.
    assert f"config={config}" in result.stdout


def test_make_world_stops_on_first_error(tmp_path: Path) -> None:
    """Ошибка эпохи останавливает конвейер: ненулевой код, артефактов нет (3.1)."""

    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    config = _write_config(tmp_path, ml_db=blocker / "epoch.db")
    root = tmp_path / "world"
    result = _run_make("world", f"CONFIG={config}")

    assert result.returncode != 0
    # Эпоха не создана, дальнейшие шаги не выполнялись.
    assert not (root / "artifacts").exists()
    assert not (root / "planning").exists()
    assert not (root / "worlds").exists()
