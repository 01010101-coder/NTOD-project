"""Тесты change ``m6-rng-branches``: разделение rng-потоков и веточный генератор.

Краеугольный тест — независимость экзогенного потока (задача 2.1, design
решение 6): при ``price_fn``, давящем спрос (×3.0 при α=1 → λ/3), поминутные
опросы сред двух движков одного seed (``is_rain``/``price_multiplier`` всех зон)
совпадают, а таблицы ``trips`` закономерно расходятся; до разделения потоков
тест красный — общий генератор сдвигался расходом эндогенных розыгрышей.
Доступ к среде — публичные запросы ``env.is_rain``/``env.price_multiplier``
по всем минутам из шва наблюдения ``on_minute``; приватные списки эпизодов
среды тесты не читают.
"""

import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from common.config import AppConfig, load_config
from simulator.branches import BRANCHES, generate_branch
from simulator.engine import Engine
from storage.csv_export import export_table

REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
SEED = 42
DAYS = 3  # несколько суток: расхождение эндогенного потока успевает разъехаться
POLL_DAYS = 2  # короткий прогон опроса среды на синтетическом мире


def _synthetic_config() -> AppConfig:
    """Синтетический мир: 2 зоны, спавн отключён, дождь и инъекции с p=0.5.

    Спавн не нужен проверкам потоков: розыгрыши среды (дождь, surge-инъекция)
    выполняются в ``advance_day`` независимо от спроса, а ``scale=0`` делает
    прогон мгновенным.
    """

    return AppConfig.model_validate(
        {
            "seed": 7,
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
            "population": {"scale": 0.0},
            "drivers": {"count": 1, "wait_limit_min": 6},
            "weather": {
                "rain_day_chance": 0.5,
                "episode_start_h": [0, 0],
                "episode_duration_h": [1, 1],
            },
            "pricing": {
                "surge_injection": {
                    "day_chance": 0.5,
                    "duration_min": [30, 30],
                    "multiplier": [1.2, 1.6],
                }
            },
        }
    )


class _StubWriter:
    """Writer-заглушка: копит строки обеих таблиц в памяти (шов §10.12)."""

    def __init__(self) -> None:
        self.trips: list[dict] = []
        self.metrics: list[dict] = []

    def write_trips(self, rows: list[dict]) -> None:
        self.trips.extend(rows)

    def write_metrics(self, rows: list[dict]) -> None:
        self.metrics.extend(rows)


def _run_engine(engine: Engine, cfg: AppConfig, days: int):
    """Прогнать ``days`` суток, собрать trips и поминутные опросы среды.

    Опросы — публичные запросы ``is_rain``/``price_multiplier`` среды движка
    в колбэке ``on_minute``: чистые чтения состояния, rng не тратят, поэтому
    наблюдение не смещает ни один из потоков (design, решение 6).
    """

    zone_ids = [zone.id for zone in cfg.city.zones]
    writer = _StubWriter()
    polls: list[tuple[bool, tuple[float, ...]]] = []

    def on_minute(record) -> None:
        env = engine._env  # доступ к среде санкционирован design (решение 6)
        polls.append(
            (
                env.is_rain(record.t),
                tuple(env.price_multiplier(zone_id, record.t) for zone_id in zone_ids),
            )
        )

    engine.run(days, writer, on_minute=on_minute)
    return writer, polls


def test_exogenous_stream_independent_of_price_fn():
    """Тариф ×3.0 (α=1 → λ/3): среды совпадают поминутно, trips расходятся (2.1)."""

    cfg = load_config(REFERENCE_CONFIG)
    writer_base, polls_base = _run_engine(Engine(cfg, seed=SEED), cfg, DAYS)
    writer_priced, polls_priced = _run_engine(
        Engine(cfg, seed=SEED, price_fn=lambda zone_id, t: 3.0), cfg, DAYS
    )

    # Основная фаза даёт ровно минуту на срез, далее — дренаж хвоста (его длина
    # зависит от числа заказов в полёте и потому сравнивается только в основной
    # фазе — одинаковый диапазон минут обеих веток).
    assert len(polls_base) >= DAYS * 1440
    # Экзогенный поток не зависит от эндогенной истории: дождь и тариф среды
    # совпадают в каждой минуте каждой зоны между ветками одного seed.
    assert polls_base[: DAYS * 1440] == polls_priced[: DAYS * 1440]
    # Проверка бессильна без реального расхождения эндогенной истории:
    # λ/3 обязана дать другую таблицу заказов.
    assert writer_base.trips != writer_priced.trips


def test_price_fn_equal_to_env_tariff_identical_to_base():
    """price_fn, повторяющий тариф среды, побитово неотличим от базы (спека simulator-loop).

    Сценарий дельты simulator-loop: политика, не меняющая цену, не меняет и
    мир — подмена равным значением даёт ту же λ и тот же расход эндогенного
    потока, строки обеих таблиц совпадают. В окне 3 суток seed 42 среда
    детерминированно разыгрывает surge-инъекцию (42 минуты ≠ 1.0), поэтому
    mimic действительно следует за тарифом среды, а не за константой 1.0.
    """

    from datetime import datetime

    cfg = load_config(REFERENCE_CONFIG)
    env_box: list = []  # среда mimic-движка, подставляется после сборки

    def env_tariff(zone_id: int, t: datetime) -> float:
        return env_box[0].price_multiplier(zone_id, t)

    mimic = Engine(cfg, seed=SEED, price_fn=env_tariff)
    env_box.append(mimic._env)  # доступ к среде санкционирован design (решение 6)
    writer_base, polls_base = _run_engine(Engine(cfg, seed=SEED), cfg, DAYS)
    writer_mimic, _polls_mimic = _run_engine(mimic, cfg, DAYS)

    # Сценарий не вырождается: в окне разыграна инъекция — тариф среды ≠ 1.0.
    assert any(mult != 1.0 for _rain, mults in polls_base for mult in mults)
    assert writer_base.trips == writer_mimic.trips
    assert writer_base.metrics == writer_mimic.metrics
    assert writer_base.trips


def test_engine_streams_from_config_seed():
    """Без явного seed генераторы — дети ``SeedSequence(cfg.seed).spawn(2)`` (1.2).

    Порядок детей закреплён спекой «Seed по умолчанию берётся из конфига»:
    [эндогенный, экзогенный] — состояния бит-генераторов свежего движка
    совпадают с дочерними ``default_rng`` первого и второго ребёнка.
    """

    cfg = _synthetic_config()
    engine = Engine(cfg)
    endo_seq, exo_seq = np.random.SeedSequence(cfg.seed).spawn(2)
    assert engine._rng.bit_generator.state == np.random.default_rng(endo_seq).bit_generator.state
    assert engine._rng_env.bit_generator.state == np.random.default_rng(exo_seq).bit_generator.state


def test_config_seed_gives_identical_env_polls():
    """Два движка одного ``cfg.seed`` без явного seed — идентичные опросы среды (1.2)."""

    cfg = _synthetic_config()
    _, polls_a = _run_engine(Engine(cfg), cfg, POLL_DAYS)
    _, polls_b = _run_engine(Engine(cfg), cfg, POLL_DAYS)
    assert polls_a == polls_b
    # Сравнение не вырождается: среда реально разыграла и дождь, и инъекцию.
    assert any(is_rain for is_rain, _mults in polls_a)
    assert any(any(mult != 1.0 for mult in mults) for _rain, mults in polls_a)


def test_generate_branch_a_base(tmp_path):
    """Ветка ``a-base`` 2 дня в tmp: схема, метрики 288/сутки·зону, тариф среды (4.1)."""

    cfg = load_config(REFERENCE_CONFIG)
    db = tmp_path / "a-base.db"
    summary = generate_branch("a-base", cfg, seed=SEED, days=2, db_path=str(db), shadow=False)

    assert summary is not None
    assert summary.days == 2 and summary.seed == SEED and summary.trips_total > 0
    conn = sqlite3.connect(db)
    try:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"trips", "zone_metrics_5min"} <= tables  # схема контракта данных
        metrics = conn.execute("SELECT COUNT(*) FROM zone_metrics_5min").fetchone()[0]
        assert metrics == 2 * 288 * len(cfg.city.zones)  # 288 окон на сутки·зону
        mults = [row[0] for row in conn.execute("SELECT DISTINCT price_multiplier FROM trips")]
    finally:
        conn.close()
    # Мир без политики: тариф заказов — из среды (1.0 вне окон инъекций,
    # значения из диапазона конфига внутри).
    low, high = cfg.pricing.surge_injection.multiplier
    assert mults
    assert all(mult == 1.0 or low <= mult <= high for mult in mults)


def test_branch_summary_shows_cancellations(tmp_path, capsys):
    """Сводка ветки печатает ``cancel:`` при включённых отказах (world-dispatch-radius)."""

    cfg = load_config(REFERENCE_CONFIG).model_copy(deep=True)
    cfg.drivers.dispatch.cancel_grace_min = 0
    cfg.drivers.dispatch.cancel_prob_per_min = 0.5
    db = tmp_path / "a-base.db"
    summary = generate_branch("a-base", cfg, seed=SEED, days=1, db_path=str(db), shadow=False)

    assert summary is not None and summary.cancelled > 0
    lines = capsys.readouterr().out.strip().split("\n")
    assert len(lines) == 5
    assert re.fullmatch(r"cancel: cancelled=\d+ cancelled_share=\d+\.\d%", lines[3])
    assert lines[4].startswith("wall: ")
    assert f"total={summary.trips_total}" in lines[1]


def test_generate_branch_unknown_branch_raises():
    """Неизвестное имя ветки: исключение с именем в сообщении, прогона нет (спека)."""

    cfg = load_config(REFERENCE_CONFIG)
    with pytest.raises(ValueError, match="no-such-branch"):
        generate_branch("no-such-branch", cfg, days=2)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Интерпретатор текущего окружения: .venv/bin/python на POSIX, .venv/Scripts/python.exe на Windows.
PYTHON = Path(sys.executable)
PROGRESS_PATTERN = r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] trips=\d+ elapsed=\d+\.\d+s"


def _run_branch_cli(args: list[str]):
    """Запустить CLI веток subprocess'ом из корня проекта (прецедент test_sim_cli)."""

    return subprocess.run(
        [str(PYTHON), "-m", "simulator.branches", *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )


def _table_counts(db: Path) -> tuple[int, int, int]:
    """(строк trips, строк zone_metrics_5min, уникальных ts) в БД."""

    conn = sqlite3.connect(db)
    try:
        trips = conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
        metrics = conn.execute("SELECT COUNT(*) FROM zone_metrics_5min").fetchone()[0]
        stamps = conn.execute("SELECT COUNT(DISTINCT ts) FROM zone_metrics_5min").fetchone()[0]
    finally:
        conn.close()
    return trips, metrics, stamps


def test_branch_cli_smoke(tmp_path):
    """CLI 1 день в tmp: exit 0, сводка §4 из 4 строк, таблицы наполнены (4.2)."""

    db = tmp_path / "a-base.db"
    result = _run_branch_cli(
        ["--branch", "a-base", "--days", "1", "--seed", str(SEED), "--db", str(db)]
    )
    assert result.returncode == 0, result.stderr

    lines = result.stdout.rstrip("\n").split("\n")
    assert len(lines) == 4  # ровно 4 строки сводки формата §4
    assert lines[0] == f"sim: days=1 seed={SEED} db={db}"
    assert re.findall(PROGRESS_PATTERN, result.stderr)  # прогресс раз в сим-час

    trips, metrics, stamps = _table_counts(db)
    assert trips > 0
    assert metrics == 1728  # 1 день: 288 окон × 6 зон
    assert stamps == 288


def test_branch_cli_guard_rejects_nonempty_trips(tmp_path):
    """Повторный запуск в непустую БД: exit 2, guard-сообщение, новых строк нет (4.2)."""

    db = tmp_path / "a-base.db"
    args = ["--branch", "a-base", "--days", "1", "--seed", str(SEED), "--db", str(db)]
    first = _run_branch_cli(args)
    assert first.returncode == 0, first.stderr
    trips_before, _metrics, _stamps = _table_counts(db)

    second = _run_branch_cli(args)
    assert second.returncode == 2
    assert "генерация требует пустой таблицы trips" in second.stderr
    assert _table_counts(db)[0] == trips_before  # прогон не выполнялся


def test_branch_cli_unknown_branch(tmp_path):
    """Неизвестная ветка: exit 2, имя в stderr, БД не создаётся (4.2)."""

    result = _run_branch_cli(["--branch", "no-such-branch", "--db", str(tmp_path / "x.db")])
    assert result.returncode == 2
    assert "no-such-branch" in result.stderr
    assert not (tmp_path / "x.db").exists()


def test_branch_reproducibility(tmp_path):
    """Две генерации a-base побитово идентичны; другой seed различается (5.1)."""

    cfg = load_config(REFERENCE_CONFIG)
    db_a, db_b, db_c = tmp_path / "a.db", tmp_path / "b.db", tmp_path / "c.db"
    generate_branch("a-base", cfg, seed=SEED, days=2, db_path=str(db_a), shadow=False)
    generate_branch("a-base", cfg, seed=SEED, days=2, db_path=str(db_b), shadow=False)
    generate_branch("a-base", cfg, seed=SEED + 1, days=2, db_path=str(db_c), shadow=False)

    # Экспорт таблиц (прецедент epoch_check M3): CSV-файлы сравниваются байтами.
    for table in ("trips", "zone_metrics_5min"):
        for db, tag in ((db_a, "a"), (db_b, "b"), (db_c, "c")):
            export_table(sqlite3.connect(str(db)), table, str(tmp_path / f"{table}_{tag}.csv"))
        same_seed = (tmp_path / f"{table}_a.csv").read_bytes()
        same_seed_again = (tmp_path / f"{table}_b.csv").read_bytes()
        other_seed = (tmp_path / f"{table}_c.csv").read_bytes()
        assert same_seed == same_seed_again  # побитовая повторяемость ветки
        assert same_seed != other_seed  # другой seed — другая история


# --- change m6-policy: контурные ветки реестра b/c/d (задача 5.1) -----------

BOUNDARY_ISO = "2026-11-03 00:00:00"
"""Граница активации политики в двухдневном мире с ml.train_days = 1 (день 2)."""


def _policy_world_config() -> AppConfig:
    """Мир контурных тестов: 2 зоны, спавн включён, случайных событий нет;
    политика активна со дня ``ml.train_days + 1`` (день 2)."""

    return AppConfig.model_validate(
        {
            "seed": 7,
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


def test_registry_contains_policy_branches():
    """Реестр: b-reactive/c-mlp/d-lstm — контурные фабрики, a-base — без политики (5.1)."""

    assert BRANCHES["a-base"].price_fn_factory is None
    assert BRANCHES["a-base"].contour_factory is None
    for name in ("b-reactive", "c-mlp", "d-lstm"):
        assert BRANCHES[name].contour_factory is not None
        assert BRANCHES[name].price_fn_factory is None


def test_generate_branch_b_reactive(tmp_path):
    """Ветка b-reactive: префикс до активации = a-base; множители ≤ total_cap (5.1)."""

    cfg = _policy_world_config()
    db_a, db_b = tmp_path / "a.db", tmp_path / "b.db"
    generate_branch("a-base", cfg, seed=SEED, days=2, db_path=str(db_a), shadow=False)
    summary = generate_branch("b-reactive", cfg, seed=SEED, days=2, db_path=str(db_b), shadow=False)
    assert summary is not None

    metrics = {}
    for db in (db_a, db_b):
        conn = sqlite3.connect(db)
        try:
            metrics[db] = conn.execute(
                "SELECT ts, price_multiplier FROM zone_metrics_5min ORDER BY ts, zone_id"
            ).fetchall()
        finally:
            conn.close()
    prefix_a = [row for row in metrics[db_a] if row[0] < BOUNDARY_ISO]
    prefix_b = [row for row in metrics[db_b] if row[0] < BOUNDARY_ISO]
    assert prefix_a and prefix_a == prefix_b  # день 1 — тариф мира, как у a-base
    suffix_b = [row for row in metrics[db_b] if row[0] >= BOUNDARY_ISO]
    assert suffix_b
    cap = cfg.policy.total_cap
    assert all(mult <= cap for _ts, mult in suffix_b)  # композиция с потолком
    assert any(mult > 1.0 for _ts, mult in suffix_b)  # реактивное правило срабатывало


def test_generate_branch_explicit_price_fn_overrides_contour_factory(tmp_path):
    """Явный price_fn приоритетнее фабрики контура: мир побитово равен a-base (5.1)."""

    cfg = _policy_world_config()
    db_a, db_b = tmp_path / "a.db", tmp_path / "b.db"
    generate_branch("a-base", cfg, seed=SEED, days=1, db_path=str(db_a), shadow=False)
    generate_branch(
        "b-reactive",
        cfg,
        seed=SEED,
        days=1,
        db_path=str(db_b),
        price_fn=lambda zone_id, t: 1.0,  # = тариф среды этого мира (инъекций нет)
        shadow=False,
    )
    # Политика не подключалась: побитово тот же мир, что a-base (сценарий
    # «price_fn, повторяющий тариф среды, неотличим от базы»).
    for table in ("trips", "zone_metrics_5min"):
        exports = []
        for db in (db_a, db_b):
            path = tmp_path / f"{table}_{db.stem}.csv"
            export_table(sqlite3.connect(str(db)), table, str(path))
            exports.append(path.read_bytes())
        assert exports[0] == exports[1]


@pytest.mark.parametrize(("branch", "family"), [("c-mlp", "mlp"), ("d-lstm", "lstm")])
def test_generate_branch_network_branch_missing_artifacts(
    tmp_path, monkeypatch, branch: str, family: str
):
    """Сетевая ветка без каталогов артефактов: исключение до прогона, БД без строк (5.1)."""

    import simulator.branches as branches_module

    cfg = _policy_world_config()
    monkeypatch.setattr(branches_module, "ARTIFACTS_ROOT", tmp_path / "no-artifacts")
    db = tmp_path / f"{branch}.db"
    with pytest.raises(FileNotFoundError, match=family):
        generate_branch(branch, cfg, seed=SEED, days=2, db_path=str(db))
    conn = sqlite3.connect(db)  # БД создана (guard непустой trips), но строк не получила
    try:
        assert conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM zone_metrics_5min").fetchone()[0] == 0
    finally:
        conn.close()


def test_branch_cli_c_mlp_missing_artifacts(tmp_path):
    """CLI c-mlp без артефактов: exit 2, имя семейства в stderr (5.1).

    cwd — временный каталог: корень артефактов ``artifacts`` резолвится от
    cwd, а конфиг передаётся абсолютным путём; таким образом подменять
    артефакты монкипатчем не нужно. ``PYTHONPATH`` даёт subprocess'у пакеты
    проекта (editable-install перечисляет только common/storage/simulator/app).
    """

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    result = subprocess.run(
        [
            str(PYTHON),
            "-m",
            "simulator.branches",
            "--branch",
            "c-mlp",
            "--config",
            str(PROJECT_ROOT / "config.yaml"),
            "--db",
            str(tmp_path / "c.db"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert result.returncode == 2
    assert "mlp" in result.stderr
