"""Тесты change ``m6-worlds``: тень-прогнозы, сетка миров, реестр (задачи 2–3).

Тень (задачи 2.1–2.3, спека simulator-policy): строки обоих семейств и обоих
горизонтов с границы активации — по бакету × зоне × 2 × 2, до границы ни одной;
``ShadowForecaster`` без каталога семейства — исключение до старта; тень в
``generate_branch`` пишет ``forecasts`` БД ветки, guard артефактов действует на
любую ветку реестра, ``shadow=False`` отключает тень и требование артефактов;
чистота наблюдения — мир с тенью и без побайтово равен, повторный прогон с
тенью даёт побитово те же ``forecasts``.

Сетка и реестр (задачи 3.1–3.4, спека simulator-branches): ``generate_worlds``
пишет 4 мира по порядку реестра с диагностикой ``policy_fire_rate`` контурных
веток и итогом; guard срабатывает до первой ветки со списком занятых; CLI
``--all`` отклоняет ``--db`` до старта; ``list_worlds`` — фиксированный состав
в порядке реестра с метками и признаком наличия; прогрев сетки — все 6 пар
веток побитово равны до границы активации, ``forecasts`` прогрева пуста,
после границы b/c/d отличаются от a-base.

Мини-артефакты — фабрика тестов ml (прецедент m6-policy 4.2): обучаемые за
секунды сети обоих семейств на 3-дневной мини-эпохе, разделяемые тестами
модуля (``mini_root``); миры — 2-зонный конфиг m6-policy (``scale=50``).
"""

import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import yaml

import simulator.branches as branches_module
from common.config import AppConfig, MlCfg
from common.timeutil import add_minutes, from_iso
from ml.train import model_version_for, prepare_training_data, save_model_artifacts, train_model
from simulator.__main__ import SqliteTripWriter
from simulator.branches import BRANCHES, generate_branch, generate_worlds, list_worlds
from simulator.engine import Engine
from simulator.policy import ShadowForecaster
from storage.csv_export import export_table
from storage.db import connect
from storage.repo import insert_zone_metrics
from storage.schema import init_db

SEED = 7
SIM_START_DT = from_iso("2026-11-02 00:00:00")
BOUNDARY_ISO = "2026-11-03 00:00:00"
"""Граница окна активации двухдневного мира с ``ml.train_days = 1`` (день 2)."""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Интерпретатор текущего окружения: .venv/bin/python на POSIX, .venv/Scripts/python.exe на Windows.
PYTHON = Path(sys.executable)


def _ml_cfg() -> MlCfg:
    """Тестовые гиперпараметры мини-артефактов (приём test_sim_policy): окно 1 ч."""

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
    """Мини-эпоха с закономерностью: 3 дня × 288 бакетов, зоны 0/1 (приём test_sim_policy)."""

    conn = connect(str(db_path))
    init_db(conn)
    rows: list[dict] = []
    t = SIM_START_DT
    end = add_minutes(SIM_START_DT, 3 * 1440 - 5)
    while t <= end:
        j = int((t - SIM_START_DT).total_seconds()) // 300
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
        t = add_minutes(t, 5)
    insert_zone_metrics(conn, rows)
    conn.close()


def _policy_world_config() -> AppConfig:
    """Мир тестов m6-worlds: 2 зоны, спавн включён, случайных событий нет;
    политика и тень активны со дня ``ml.train_days + 1`` (день 2)."""

    return AppConfig.model_validate(
        {
            "seed": SEED,
            "city": {
                "zones": [
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
                ],
                "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 30.0}],
            },
            "population": {"scale": 50.0},
            "drivers": {"count": 4, "wait_limit_min": 6},
            "weather": {
                "rain_day_chance": 0.0,
                "episode_start_h": [0, 0],
                "episode_duration_h": [1, 1],
            },
            "pricing": {"surge_injection": {"day_chance": 0.0}},
            "ml": {"train_days": 1},
        }
    )


@pytest.fixture(scope="module")
def mini_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Каталог мини-артефактов обоих семейств + мини-эпоха (обучение один раз)."""

    root = tmp_path_factory.mktemp("mini")
    pattern_db = root / "pattern.db"
    _fill_pattern_db(pattern_db)
    ml_cfg = _ml_cfg()
    data = prepare_training_data(pattern_db, ml_cfg, zones=[0, 1], start=SIM_START_DT)
    for arch in ("mlp", "lstm"):
        result = train_model(arch, data, ml_cfg)
        version = model_version_for(arch, today=date(2026, 1, 1))
        save_model_artifacts(
            root / "artifacts" / version,
            model_version=version,
            arch=arch,
            model=result.model,
            data=data,
            result=result,
            ml_cfg=ml_cfg,
        )
    return root


@pytest.fixture
def mini_artifacts(mini_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Подменить корень артефактов веточного генератора на мини-каталог."""

    root = mini_root / "artifacts"
    monkeypatch.setattr(branches_module, "ARTIFACTS_ROOT", root)
    return root


def _firing_config() -> AppConfig:
    """Мир сетки с гарантированными срабатываниями ступеней: δ = 1.0 → p_eff = 1."""

    cfg = _policy_world_config()
    return cfg.model_copy(
        update={"policy": cfg.policy.model_copy(update={"shift": {"mlp": 1.0, "lstm": 1.0}})}
    )


def _forecasts(db: Path) -> list[tuple]:
    """Строки forecasts ветки (ts, zone_id, horizon_min, model_version, deficit_prob)."""

    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(
            "SELECT ts, zone_id, horizon_min, model_version, deficit_prob"
            " FROM forecasts ORDER BY ts, zone_id, horizon_min, model_version"
        ).fetchall()
    finally:
        conn.close()


# --- задача 2.1: ShadowForecaster -------------------------------------------


def test_shadow_forecaster_rows_from_activation(tmp_path: Path, mini_artifacts: Path) -> None:
    """Тень выдаёт строки обоих семейств с границы активации, до границы — ни одной (2.1)."""

    cfg = _policy_world_config()
    rows: list[dict] = []
    shadow = ShadowForecaster(cfg, artifacts_root=mini_artifacts, flush=rows.extend)
    conn = connect(str(tmp_path / "run.db"))
    init_db(conn)
    Engine(cfg, seed=SEED).run(
        2,
        SqliteTripWriter(conn),
        on_minute=shadow.on_minute,
        on_metrics_row=shadow.on_metrics_row,
    )

    boundary = from_iso(BOUNDARY_ISO)
    assert rows, "тень не выдала ни одной строки за тест-день"
    assert all(row["ts"] >= boundary for row in rows)  # до границы — ни одной
    keys = [(row["ts"], row["zone_id"], row["horizon_min"], row["model_version"]) for row in rows]
    assert len(keys) == len(set(keys))  # PK (ts, зона, горизонт, семейство) принимает все
    assert len(rows) == 288 * 2 * 2 * 2  # бакет × зона × горизонт × семейство дня 2
    assert {row["model_version"] for row in rows} == {"mlp", "lstm"}
    assert {row["horizon_min"] for row in rows} == {15, 30}
    assert all(0.0 <= row["deficit_prob"] <= 1.0 for row in rows)


def test_shadow_missing_family_artifact(tmp_path: Path) -> None:
    """Нет каталога семейства — исключение при построении тени, до прогона (2.1)."""

    with pytest.raises(FileNotFoundError, match="mlp"):
        ShadowForecaster(_policy_world_config(), artifacts_root=tmp_path / "empty")


# --- задача 2.2: тень в generate_branch --------------------------------------


def test_generate_branch_shadow_writes_forecasts(tmp_path: Path, mini_artifacts: Path) -> None:
    """a-base с тенью (умолчание): forecasts непуста со дня активации, PK принимает (2.2)."""

    cfg = _policy_world_config()
    db = tmp_path / "a-base.db"
    generate_branch("a-base", cfg, seed=SEED, days=2, db_path=str(db))

    rows = _forecasts(db)
    assert len(rows) == 288 * 2 * 2 * 2
    assert all(row[0] >= BOUNDARY_ISO for row in rows)
    assert {row[3] for row in rows} == {"mlp", "lstm"}


def test_generate_branch_shadow_guard_without_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a-base с тенью без каталогов артефактов: исключение до старта, БД без строк (2.2)."""

    monkeypatch.setattr(branches_module, "ARTIFACTS_ROOT", tmp_path / "empty")
    db = tmp_path / "a-base.db"
    with pytest.raises(FileNotFoundError):
        generate_branch("a-base", _policy_world_config(), seed=SEED, days=2, db_path=str(db))
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0] == 0
    finally:
        conn.close()


def test_generate_branch_no_shadow_without_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """b-reactive с shadow=False без каталогов: генерируется, forecasts пуста (2.2, регресс)."""

    monkeypatch.setattr(branches_module, "ARTIFACTS_ROOT", tmp_path / "empty")
    db = tmp_path / "b-reactive.db"
    summary = generate_branch(
        "b-reactive", _policy_world_config(), seed=SEED, days=2, db_path=str(db), shadow=False
    )
    assert summary is not None
    assert _forecasts(db) == []


# --- задача 2.3: чистота наблюдения и детерминизм тени ------------------------


def test_shadow_purity_and_determinism(tmp_path: Path, mini_artifacts: Path) -> None:
    """Мир с тенью и без побайтово равен; повторная тень — те же forecasts (2.3)."""

    cfg = _policy_world_config()
    with_shadow = tmp_path / "with.db"
    without_shadow = tmp_path / "without.db"
    again = tmp_path / "again.db"
    generate_branch("a-base", cfg, seed=SEED, days=2, db_path=str(with_shadow))
    generate_branch("a-base", cfg, seed=SEED, days=2, db_path=str(without_shadow), shadow=False)
    generate_branch("a-base", cfg, seed=SEED, days=2, db_path=str(again))

    # Чистота: наблюдение не меняет розыгрыши и историю.
    for table in ("trips", "zone_metrics_5min"):
        exports = []
        for db in (with_shadow, without_shadow):
            path = tmp_path / f"{table}_{db.stem}.csv"
            export_table(sqlite3.connect(str(db)), table, str(path))
            exports.append(path.read_bytes())
        assert exports[0] == exports[1]

    # Детерминизм: инференс тени на фиксированных весах побитово повторим.
    path_first = tmp_path / "forecasts_first.csv"
    path_again = tmp_path / "forecasts_again.csv"
    export_table(sqlite3.connect(str(with_shadow)), "forecasts", str(path_first))
    export_table(sqlite3.connect(str(again)), "forecasts", str(path_again))
    assert path_first.read_bytes() == path_again.read_bytes()


# --- задача 3.1: generate_worlds ---------------------------------------------


def test_generate_worlds_short_grid(
    tmp_path: Path, mini_artifacts: Path, capsys: pytest.CaptureFixture
) -> None:
    """Короткая сетка: 4 мира с тенью, fire_rate контурных веток в выводе, итог (3.1)."""

    worlds = tmp_path / "worlds"
    result = generate_worlds(_firing_config(), seed=SEED, days=2, worlds_dir=worlds)
    assert result is not None
    assert list(result) == list(BRANCHES)  # порядок реестра

    for name in BRANCHES:
        db = worlds / f"{name}.db"
        assert db.exists()
        assert _forecasts(db), f"{name}: тень должна писать строки тест-дня"

    out = capsys.readouterr().out
    assert out.count("policy_fire_rate") == 3  # b-reactive, c-mlp, d-lstm
    assert "worlds: branches=4" in out
    for name in BRANCHES:
        assert f"world: branch={name} wall=" in out


def test_generate_worlds_guard_before_any_branch(
    tmp_path: Path, mini_artifacts: Path, capsys: pytest.CaptureFixture
) -> None:
    """Одна целевая БД непуста — ни одна ветка не генерируется, список в stderr (3.1)."""

    worlds = tmp_path / "worlds"
    worlds.mkdir()
    generate_branch(
        "a-base",
        _policy_world_config(),
        seed=SEED,
        days=1,
        db_path=str(worlds / "a-base.db"),
        shadow=False,
    )
    conn = sqlite3.connect(str(worlds / "a-base.db"))
    trips_before = conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    conn.close()

    result = generate_worlds(_policy_world_config(), seed=SEED, days=2, worlds_dir=worlds)

    assert result is None
    assert "занятые ветки: a-base" in capsys.readouterr().err
    for name in ("b-reactive", "c-mlp", "d-lstm"):
        assert not (worlds / f"{name}.db").exists()  # ни одна не создана
    conn = sqlite3.connect(str(worlds / "a-base.db"))
    try:
        assert conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0] == trips_before
    finally:
        conn.close()


# --- задача 3.2: CLI --all ----------------------------------------------------


def _write_micro_config(path: Path, cfg: AppConfig) -> None:
    """Микро-конфиг YAML мира тестов (без комментариев — только для CLI-прогонов)."""

    path.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8")


def _run_branch_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """CLI веток subprocess'ом из заданного cwd (прецедент test_sim_branches)."""

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    return subprocess.run(
        [str(PYTHON), "-m", "simulator.branches", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )


def test_branch_cli_all_short_grid(tmp_path: Path, mini_root: Path) -> None:
    """CLI --all короткой сеткой: 4 БД с тенью, exit 0, итог сетки в выводе (3.2)."""

    shutil.copytree(mini_root / "artifacts", tmp_path / "artifacts")
    cfg = _policy_world_config()
    cfg = cfg.model_copy(update={"ml": cfg.ml.model_copy(update={"train_days": 0})})
    config_path = tmp_path / "config.yaml"
    _write_micro_config(config_path, cfg)

    result = _run_branch_cli(
        [
            "--all",
            "--days",
            "1",
            "--seed",
            str(SEED),
            "--config",
            str(config_path),
            "--worlds-dir",
            str(tmp_path / "worlds"),
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "worlds: branches=4" in result.stdout

    for name in BRANCHES:
        db = tmp_path / "worlds" / f"{name}.db"
        assert db.exists()
        assert _forecasts(db), f"{name}: активация — день 1, тень обязана писать"


def test_branch_cli_all_db_conflict(tmp_path: Path) -> None:
    """--all вместе с --db отклоняется argparse до старта, сетки нет (3.2)."""

    result = _run_branch_cli(
        ["--all", "--db", str(tmp_path / "x.db"), "--worlds-dir", str(tmp_path / "worlds")],
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "--db" in result.stderr
    assert not (tmp_path / "worlds").exists()


# --- задача 3.3: реестр миров -------------------------------------------------


def test_list_worlds_registry(tmp_path: Path, mini_artifacts: Path) -> None:
    """Реестр миров: фиксированный состав в порядке реестра, метки, наличие (3.3)."""

    cfg = _policy_world_config()
    worlds = tmp_path / "w"
    infos = list_worlds(cfg, worlds_dir=worlds)
    assert [info.name for info in infos] == list(BRANCHES)
    assert [info.label for info in infos] == [
        "мир без политики",
        "реактивная политика",
        "ступени по прогнозу MLP",
        "ступени по прогнозу LSTM",
    ]
    assert all(not info.exists for info in infos)  # пустой каталог — не ошибка
    assert all(info.db == worlds / f"{info.name}.db" for info in infos)

    result = generate_worlds(cfg, seed=SEED, days=1, worlds_dir=worlds)
    assert result is not None
    infos = list_worlds(cfg, worlds_dir=worlds)
    assert all(info.exists for info in infos)

    # worlds_dir из конфига (policy.worlds_dir) — тот же перечень.
    cfg_with_dir = cfg.model_copy(
        update={"policy": cfg.policy.model_copy(update={"worlds_dir": str(worlds)})}
    )
    assert list_worlds(cfg_with_dir) == infos


# --- задача 3.4: побитовое совпадение прогрева сетки --------------------------


def test_worlds_warmup_bit_identical(tmp_path: Path, mini_artifacts: Path) -> None:
    """Все 6 пар веток побитово равны до границы; после — b/c/d отличаются (3.4, DoD §9)."""

    worlds = tmp_path / "worlds"
    assert generate_worlds(_firing_config(), seed=SEED, days=2, worlds_dir=worlds) is not None

    conns = {name: sqlite3.connect(str(worlds / f"{name}.db")) for name in BRANCHES}
    try:
        trips = {
            name: [row for row in conn.execute("SELECT * FROM trips ORDER BY trip_id")]
            for name, conn in conns.items()
        }
        metrics = {
            name: [
                row for row in conn.execute("SELECT * FROM zone_metrics_5min ORDER BY ts, zone_id")
            ]
            for name, conn in conns.items()
        }
        prefix_trips = {
            name: [row for row in trips[name] if row[1] < BOUNDARY_ISO] for name in BRANCHES
        }
        prefix_metrics = {
            name: [row for row in metrics[name] if row[0] < BOUNDARY_ISO] for name in BRANCHES
        }
        suffix_metrics = {
            name: [row for row in metrics[name] if row[0] >= BOUNDARY_ISO] for name in BRANCHES
        }
        for left, right in combinations(BRANCHES, 2):
            assert prefix_trips[left] == prefix_trips[right], (left, right)
            assert prefix_metrics[left] == prefix_metrics[right], (left, right)
        for name in BRANCHES:
            assert prefix_trips[name]  # сравнение не вырождено пустым префиксом
            empty = (
                conns[name]
                .execute("SELECT COUNT(*) FROM forecasts WHERE ts < ?", (BOUNDARY_ISO,))
                .fetchone()[0]
            )
            assert empty == 0  # прогнозы прогрева не пишутся ни в одной ветке
        # После границы активная политика меняет историю во всех контурных ветках.
        for name in ("b-reactive", "c-mlp", "d-lstm"):
            assert suffix_metrics[name] != suffix_metrics["a-base"], name
    finally:
        for conn in conns.values():
            conn.close()


def test_branch_uses_ml_artifacts_root(
    tmp_path: Path, mini_artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Контур ветки берёт корень из `ml.artifacts_root`, а не из константы (1.1).

    Сценарий спеки «Отдельный корень изолирует миры»: константа указывает на
    пустой каталог, а конфиг — на подготовленные мини-артефакты. Ветка с
    отдельным корнем строится, ветка с дефолтным конфигом артефактов не находит.
    """

    monkeypatch.setattr(branches_module, "ARTIFACTS_ROOT", tmp_path / "empty")
    base = _policy_world_config()
    isolated = base.model_copy(
        update={"ml": base.ml.model_copy(update={"artifacts_root": str(mini_artifacts)})}
    )
    db = tmp_path / "isolated.db"
    summary = generate_branch("c-mlp", isolated, seed=SEED, days=2, db_path=str(db), shadow=False)
    assert summary is not None

    with pytest.raises(FileNotFoundError, match="mlp"):
        generate_branch(
            "c-mlp",
            base,
            seed=SEED,
            days=2,
            db_path=str(tmp_path / "default.db"),
            shadow=False,
        )
