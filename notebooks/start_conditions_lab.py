"""Помощники ноутбука ``start_conditions.ipynb``: прогоны мира в памяти и полный ML-конвейер.

Модуль лежит рядом с ноутбуком, потому что ``multiprocessing`` на macOS
запускает дочерние процессы через ``spawn``: функции, объявленные прямо в
ячейке Jupyter, в дочерний процесс не передаются. Здесь нет новой логики мира:
только склейка публичных частей проекта (``Engine``, контуры политик,
обучение и калибровка M3/M6) в функции, которые удобно вызывать из ноутбука.

- :func:`make_cfg` — конфиг проекта с точечными переопределениями ``"a.b.c": v``;
- :func:`run_world` — прогон движка в память (без SQLite), с политикой или без;
- :func:`outcomes` — итоги тест-недели для пассажиров и для таксопарка;
- :func:`full_experiment` — мир → эпоха 37 дней → обучение сетей → калибровка δ
  → ветки A/B/C/D на нескольких seed; эпоха и артефакты кэшируются на диске.
"""

import contextlib
import copy
import hashlib
import math
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import ml.dataset as ml_dataset
from common.config import AppConfig
from common.timeutil import add_minutes
from ml.dataset import SqliteRowSource, world_start
from ml.infer import (
    collect_role_examples,
    latest_artifact_dir,
    load_model_artifact,
    predict_windows,
)
from ml.train import prepare_training_data, save_model_artifacts, train_model
from simulator.city import build_city
from simulator.engine import Engine
from simulator.policy import LiveRowSource, NetworkPredictor, PolicyContour, ReactiveContour
from simulator.policy_calibration import fit_delta
from simulator.writers import SqliteTripWriter
from storage.db import connect
from storage.schema import init_db

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
WORK_DIR = PROJECT_ROOT / "data" / "start_conditions"
"""Кэш экспериментов: эпохи и артефакты сетей (каталог ``data/`` не в git)."""

KEY_MAX_BYTES = 200
"""Предел длины имени каталога кэша мира в байтах (UTF-8)."""

BRANCH_ORDER = ("a-base", "b-reactive", "c-mlp", "d-lstm")
FAMILY_BRANCH = {"mlp": "c-mlp", "lstm": "d-lstm"}


def base_config_dict() -> dict:
    """Текущий ``config.yaml`` как словарь (точка отсчёта всех переопределений)."""

    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def make_cfg(overrides: dict | None = None, **kwargs) -> AppConfig:
    """Конфиг проекта с переопределениями вида ``{"drivers.count": 160}``.

    Значения проходят ту же строгую pydantic-валидацию, что и ``config.yaml``.
    """

    data = copy.deepcopy(base_config_dict())
    for key, value in {**(overrides or {}), **kwargs}.items():
        node = data
        *path, leaf = key.split(".")
        for part in path:
            node = node.setdefault(part, {})
        node[leaf] = value
    return AppConfig.model_validate(data)


class MemWriter:
    """Writer движка в память: строки ``trips`` и ``zone_metrics_5min`` списками."""

    def __init__(self) -> None:
        self.trips: list[dict] = []
        self.metrics: list[dict] = []

    def write_trips(self, rows: list[dict]) -> None:
        self.trips.extend(rows)

    def write_metrics(self, rows: list[dict]) -> None:
        self.metrics.extend(rows)


def run_world(
    cfg: AppConfig, *, contour: PolicyContour | None = None, seed: int | None = None
) -> MemWriter:
    """Прогон ``train_days + test_days`` суток; политика включается с дня ``train_days + 1``."""

    writer = MemWriter()
    engine = Engine(cfg, seed=seed, price_fn=None if contour is None else contour.price)
    if contour is not None:
        contour.bind(engine)
    engine.run(
        cfg.ml.train_days + cfg.ml.test_days,
        writer,
        on_minute=None if contour is None else contour.on_minute,
        on_metrics_row=None if contour is None else contour.on_metrics_row,
    )
    return writer


def outcomes(writer: MemWriter, cfg: AppConfig) -> dict:
    """Итоги тест-недели — то же окно и та же формула выручки, что во вкладке «Сравнение».

    Для пассажиров: ``requests`` (заказы, оставшиеся после ценового отсева),
    ``served`` (перевезено), ``lost`` (заказ истёк без машины), ``cancelled`` (пассажир
    отказался от долгой подачи), ``unserved`` (их сумма), ``wait``/``wait_p90``
    (подача, мин). Для таксопарка: ``revenue`` (млн руб.), ``util`` (средняя
    загрузка парка, %). ``deficit_pct`` — доля дефицитных зона-пятиминуток по
    правилу контракта.
    """

    start = world_start(cfg) + timedelta(days=cfg.ml.train_days)
    end = start + timedelta(days=cfg.ml.test_days)
    trips = pd.DataFrame(writer.trips)
    for col in ("created_at", "pickup_at"):
        trips[col] = pd.to_datetime(trips[col])
    trips = trips[(trips.created_at >= start) & (trips.created_at < end)]
    done = trips[trips.status == "completed"]
    path = build_city(cfg).path_min
    fare = (
        cfg.pricing.base_boarding
        + cfg.pricing.per_min_rub * path[done.origin_zone.to_numpy(), done.dest_zone.to_numpy()]
    )
    wait = (done.pickup_at - done.created_at).dt.total_seconds() / 60

    metrics = pd.DataFrame(writer.metrics)
    metrics["ts"] = pd.to_datetime(metrics["ts"])
    metrics = metrics[(metrics.ts >= start) & (metrics.ts < end)]
    deficit = (metrics.free_drivers / metrics.requests.clip(lower=1) < 0.5) | (
        metrics.unfulfilled > 0
    )
    fleet = metrics.groupby("ts")[["free_drivers", "busy_drivers"]].sum()
    util = fleet.busy_drivers / (fleet.busy_drivers + fleet.free_drivers)
    lost = int((trips.status == "unfulfilled").sum())
    cancelled = int((trips.status == "cancelled").sum())
    return {
        "requests": len(trips),
        "served": len(done),
        "lost": lost,
        "lost_pct": 100.0 * lost / max(len(trips), 1),
        "cancelled": cancelled,
        "unserved": lost + cancelled,
        "unserved_pct": 100.0 * (lost + cancelled) / max(len(trips), 1),
        "wait": float(wait.mean()),
        "wait_p90": float(wait.quantile(0.9)),
        "revenue": float((fare * done.price_multiplier.to_numpy()).sum()) / 1e6,
        "deficit_pct": 100.0 * float(deficit.mean()),
        "util": 100.0 * float(util.mean()),
        "util_p95": 100.0 * float(util.quantile(0.95)),
    }


def base_point(args: tuple[dict, int]) -> dict:
    """Одна точка сетки без политики: ``(переопределения, seed)`` → итоги (для ``Pool.map``)."""

    overrides, seed = args
    cfg = make_cfg(overrides)
    return outcomes(run_world(cfg, seed=seed), cfg) | dict(overrides) | {"seed": seed}


def reactive_point(args: tuple[dict, int]) -> list[dict]:
    """База и реактивная политика (без ML) на одном мире — для дешёвых абляций."""

    overrides, seed = args
    cfg = make_cfg(overrides)
    rows = [outcomes(run_world(cfg, seed=seed), cfg) | {"pol": "a-base"}]
    contour = ReactiveContour(cfg, LiveRowSource(agg_step_min=cfg.engine.agg_step_min))
    rows.append(
        outcomes(run_world(cfg, contour=contour, seed=seed), cfg)
        | {"pol": "b-reactive", "fire": 100.0 * contour.fire_rate}
    )
    return [row | dict(overrides) | {"seed": seed} for row in rows]


# --- полный ML-конвейер ---------------------------------------------------------------


HISTORIES = ("none", "reactive")
"""История эпохи обучения: мир без политики или город, живущий с реактивной ценой."""


def _build_epoch(cfg: AppConfig, db: Path, history: str = "none") -> None:
    """Эпоха мира 37 суток в SQLite (как ``make epoch``); готовая БД не пересоздаётся.

    ``history="reactive"`` — реалистичная история: реактивная политика действует
    с первой минуты (окно активации контура сдвинуто на старт мира), и сети
    учатся на логах, где спрос уже «срезан» надбавками.
    """

    if history not in HISTORIES:
        raise ValueError(f"неизвестная история {history!r}, ожидается одна из {HISTORIES}")
    if db.exists():
        return
    db.parent.mkdir(parents=True, exist_ok=True)
    tmp = db.with_suffix(".tmp.db")
    tmp.unlink(missing_ok=True)
    conn = connect(str(tmp))
    init_db(conn)
    days = cfg.ml.train_days + cfg.ml.test_days
    if history == "none":
        Engine(cfg).run(days, SqliteTripWriter(conn))
    else:
        contour = ReactiveContour(cfg, LiveRowSource(agg_step_min=cfg.engine.agg_step_min))
        contour._activation = world_start(cfg)  # политика с первой минуты истории
        engine = Engine(cfg, price_fn=contour.price)
        contour.bind(engine)
        engine.run(
            days,
            SqliteTripWriter(conn),
            on_minute=contour.on_minute,
            on_metrics_row=contour.on_metrics_row,
        )
    conn.close()
    for suffix in ("-wal", "-shm"):
        Path(str(tmp) + suffix).unlink(missing_ok=True)
    tmp.rename(db)


TARGETS = ("deficit", "harm")
"""Цель обучения головы дефицита: правило контракта или только вред (прототип)."""


@contextlib.contextmanager
def label_target(target: str):
    """Временно подменить метку дефицита при обучении и калибровке (прототип цели «вред»).

    ``deficit`` — правило контракта ``free/requests < 0.5 или unfulfilled > 0``.
    ``harm`` — только ``unfulfilled > 0`` (истёкшие заказы и отказы): порог
    соотношения обнуляется, и условие ``free/max(requests, 1) < 0`` не
    выполняется никогда. :func:`ml.dataset.deficit_label` читает
    ``DEFICIT_RATIO_LIMIT`` при каждом вызове, поэтому подмена действует на
    метки датасета, но не трогает прогоны веток: реактивная политика и
    метрики в других процессах и вне блока видят правило контракта.
    """

    if target not in TARGETS:
        raise ValueError(f"неизвестная цель {target!r}, ожидается одна из {TARGETS}")
    saved = ml_dataset.DEFICIT_RATIO_LIMIT
    if target == "harm":
        ml_dataset.DEFICIT_RATIO_LIMIT = 0.0
    try:
        yield
    finally:
        ml_dataset.DEFICIT_RATIO_LIMIT = saved


def epoch_path(wd: Path, history: str = "none") -> Path:
    """БД эпохи мира: ``epoch.db`` (без политики) или ``epoch-<история>.db``."""

    return wd / ("epoch.db" if history == "none" else f"epoch-{history}.db")


def artifacts_root(wd: Path, target: str = "deficit", history: str = "none") -> Path:
    """Каталог сетей мира: ``artifacts[-<цель>][-h<история>]``; по умолчанию ``artifacts``."""

    name = "artifacts"
    if target != "deficit":
        name += f"-{target}"
    if history != "none":
        name += f"-h{history}"
    return wd / name


def _train(cfg: AppConfig, db: Path, root: Path, family: str, target: str = "deficit") -> dict:
    """Обучить сеть семейства на эпохе (как ``python -m ml train``); кэш по каталогу."""

    out = root / f"{family}-v1-lab"
    if not out.exists():
        zones = [zone.id for zone in cfg.city.zones]
        with label_target(target):
            data = prepare_training_data(db, cfg.ml, zones=zones, start=world_start(cfg))
        result = train_model(family, data, cfg.ml)
        save_model_artifacts(
            out,
            model_version=out.name,
            arch=family,
            model=result.model,
            data=data,
            result=result,
            ml_cfg=cfg.ml,
        )
    return pd.read_json(out / "metrics.json", typ="series").to_dict()


def _calibrate(cfg: AppConfig, db: Path, root: Path, family: str) -> tuple[float, dict]:
    """δ-сдвиг на val-днях (как ``simulator.policy_calibration``) и качество прогноза на val."""

    zones = [zone.id for zone in cfg.city.zones]
    artifact = load_model_artifact(latest_artifact_dir(root, family), n_zones=max(zones) + 1)
    examples = collect_role_examples(
        SqliteRowSource(db), artifact.ml_cfg, zones=zones, start=world_start(cfg), role="val"
    )
    _, _, probs = predict_windows(artifact, examples.windows, [z for z, _ in examples.keys])
    horizon = (15, 30).index(cfg.policy.horizon_min)
    p = probs[:, horizon]
    y = np.asarray(examples.deficit)[:, horizon].astype(bool)
    fit = fit_delta(p, target=cfg.policy.target_fire_rate, stairs=cfg.policy.stairs)
    delta = round(fit.delta, 3)
    fired = np.clip(p + delta, 0, 1) >= min(stair.p_min for stair in cfg.policy.stairs)
    # Точность «срабатываний»: какая доля поднятий цены пришлась на реальный дефицит.
    precision = float(y[fired].mean()) if fired.any() else float("nan")
    return delta, {"val_pos_rate": float(y.mean()), "fire_precision": precision}


def world_key(overrides: dict) -> str:
    """Имя кэша мира: только параметры мира и обучения, без ``policy.*``.

    Политика не меняет эпоху обучения (до дня ``train_days + 1`` ветки побитово
    равны базе), поэтому эксперименты с разными настройками политики на одном
    мире переиспользуют одну эпоху и одни сети. Имя длиннее ``KEY_MAX_BYTES``
    (предел имени каталога в ФС — 255 байт) сокращается до префикса и хэша
    sha1 полного ключа; полный ключ лежит в ``key.txt`` каталога мира.
    """

    items = sorted((k, v) for k, v in overrides.items() if not k.startswith("policy."))
    key = "__".join(f"{k}={v}" for k, v in items) or "default"
    if len(key.encode()) <= KEY_MAX_BYTES:
        return key
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return f"{key[: KEY_MAX_BYTES - 20]}__{digest}"


def _full_key(overrides: dict) -> str:
    items = sorted((k, v) for k, v in overrides.items() if not k.startswith("policy."))
    return "__".join(f"{k}={v}" for k, v in items) or "default"


def prepare_world(
    overrides: dict,
    *,
    families: tuple[str, ...] = ("mlp", "lstm"),
    work_dir: Path = WORK_DIR,
    target: str = "deficit",
    history: str = "none",
) -> Path:
    """Эпоха и обученные сети мира в кэше; повторный вызов ничего не пересчитывает.

    В параллельных прогонах сначала готовьте каждый уникальный мир один раз
    (``Pool.map(prepare_world_job, ...)``), затем запускайте эксперименты над ним.
    """

    torch_threads()
    wd = Path(work_dir) / world_key(overrides)
    cfg = make_cfg(overrides)
    db = epoch_path(wd, history)
    _build_epoch(cfg, db, history)
    (wd / "key.txt").write_text(_full_key(overrides) + "\n", encoding="utf-8")
    for family in families:
        _train(cfg, db, artifacts_root(wd, target, history), family, target)
    return wd


def prepare_world_job(args: tuple) -> str:
    """Обёртка :func:`prepare_world` под ``Pool.map``: ``(overrides, families[, opts])``.

    ``opts`` — строка цели (как раньше) или словарь ``{"target", "history"}``.
    """

    overrides, families, *rest = args
    opts = _opts(rest[0] if rest else None)
    return str(
        prepare_world(overrides, families=families, target=opts["target"], history=opts["history"])
    )


def _opts(raw) -> dict:
    """Варианты прототипов: цель обучения, история эпохи, задержка данных (мин)."""

    opts = {"target": "deficit", "history": "none", "lag": 0, "mask_price": False}
    if isinstance(raw, str):
        opts["target"] = raw
    elif raw:
        opts.update(raw)
    return opts


class PriceMaskedSource:
    """Вид источника для сети без собственного действия: ``price_multiplier`` окна = 1.0.

    Сеть, обученная на мире без политики, видела надбавку только в редких
    инъекциях и читает её как «вреда не будет»; в контуре она видит свою же
    надбавку и снимает её (самоподавление). Остальные методы — делегирование.
    """

    _PRICE = ml_dataset.ROW_MANIFEST.index("price_multiplier")

    def __init__(self, inner: LiveRowSource) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def metrics_window(self, zone_id: int, ts, steps: int):
        window = self._inner.metrics_window(zone_id, ts, steps)
        if window is None:
            return None
        window = window.copy()
        window[:, self._PRICE] = 1.0
        return window


def with_data_lag(contour: PolicyContour, lag_min: int) -> PolicyContour:
    """Задержка данных: решение по бакету, закрытому не позже чем ``lag_min`` минут назад.

    Сдвигается сам момент решения ``ts → ts − lag``: у реактивной — строка
    метрик, у сети — окно признаков вместе с календарём (они согласованы).
    Прогноз сети горизонта H тогда смотрит на момент ``сейчас − lag + H``.
    Одинаково для всех контуров; базовая ветка без контура не затрагивается.
    """

    if lag_min:
        inner = contour._multiplier
        shift = timedelta(minutes=lag_min)
        contour._multiplier = lambda zone_id, ts: inner(zone_id, ts - shift)
    return contour


def full_experiment(
    name: str,
    overrides: dict,
    *,
    families: tuple[str, ...] = ("mlp", "lstm"),
    seeds: tuple[int, ...] = (42,),
    work_dir: Path = WORK_DIR,
) -> pd.DataFrame:
    """Полный цикл проекта на новом мире: эпоха → сети → δ → ветки A/B/C/D.

    Сети учатся на днях 1–30 эпохи ``seed`` конфига (как в проекте), а ветки
    прогоняются на каждом seed из ``seeds``: другой seed — другая реализация того
    же города (другие дожди, другой случайный спрос), проверка, что выигрыш не
    артефакт одной недели.
    """

    torch_threads()
    started = time.time()
    wd = prepare_world(overrides, families=families, work_dir=work_dir)
    cfg = make_cfg(overrides)
    db = wd / "epoch.db"
    shifts, info = {}, {}
    for family in families:
        train_info = _train(cfg, db, wd / "artifacts", family)
        shifts[family], info[family] = _calibrate(cfg, db, wd / "artifacts", family)
        info[family]["best_epoch"] = train_info.get("best_epoch")
    cfg = make_cfg(overrides, **{"policy.shift": shifts})
    step = cfg.engine.agg_step_min

    rows: list[dict] = []
    for seed in seeds:
        rows.append(outcomes(run_world(cfg, seed=seed), cfg) | {"pol": "a-base", "seed": seed})
        contour = ReactiveContour(cfg, LiveRowSource(agg_step_min=step))
        rows.append(
            outcomes(run_world(cfg, contour=contour, seed=seed), cfg)
            | {"pol": "b-reactive", "seed": seed, "fire": 100.0 * contour.fire_rate}
        )
        for family in families:
            source = LiveRowSource(agg_step_min=step)
            predictor = NetworkPredictor(
                source, cfg, family=family, artifacts_root=wd / "artifacts"
            )
            contour = PolicyContour(cfg, predictor, source, model=family)
            rows.append(
                outcomes(run_world(cfg, contour=contour, seed=seed), cfg)
                | {
                    "pol": FAMILY_BRANCH[family],
                    "seed": seed,
                    "fire": 100.0 * contour.fire_rate,
                    "shift": shifts[family],
                    **info[family],
                }
            )
    frame = pd.DataFrame(rows)
    frame["cfg"] = name
    frame["wall_sec"] = time.time() - started
    return frame


def full_experiment_job(args: tuple) -> pd.DataFrame:
    """Обёртка :func:`full_experiment` под ``Pool.map``: ``(name, overrides, families, seeds)``."""

    name, overrides, families, seeds = args
    return full_experiment(name, overrides, families=families, seeds=seeds)


class LostOnlyReactiveContour(ReactiveContour):
    """Контрольная реактивная политика: надбавка только после реальных потерь.

    Правило контракта ``free/requests < 0.5 или unfulfilled > 0`` содержит
    опережающую часть: соотношение «свободные/заказы» падает раньше, чем
    заказы начинают истекать. Настоящая реактивная цена видит только уже
    случившийся вред, поэтому здесь срабатывает лишь ``unfulfilled > 0``
    в последнем закрытом бакете зоны.
    """

    def _multiplier(self, zone_id: int, ts) -> float:
        point = self._source.metrics_at(zone_id, ts)
        if point is None or point.unfulfilled <= 0:
            return 1.0
        return self._reactive_mult


def oracle_harm(overrides: dict, seed: int, work_dir: Path = WORK_DIR) -> dict:
    """Точный вред мира без политики: ``(зона, индекс бакета) → unfulfilled`` (кэш на диске).

    Это ровно то, что учатся предсказывать сети с целью «вред»: та же
    реализация города (seed), без надбавок. Кэш —
    ``<мир>/oracle-harm-<seed>.pkl``; ``policy.*`` в ключ не входит.
    """

    wd = Path(work_dir) / world_key(overrides)
    path = wd / f"oracle-harm-{seed}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    cfg = make_cfg(overrides)
    metrics = pd.DataFrame(run_world(cfg, seed=seed).metrics)
    metrics = metrics[metrics.unfulfilled > 0]
    step = cfg.engine.agg_step_min
    harm = {
        (int(z), ml_dataset.bucket_index(pd.Timestamp(ts).to_pydatetime(), step)): int(u)
        for z, ts, u in zip(metrics.zone_id, metrics.ts, metrics.unfulfilled, strict=True)
    }
    wd.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(harm, path)
    return harm


def oracle_harm_job(args: tuple) -> int:
    """Обёртка :func:`oracle_harm` под ``Pool.map``: ``(overrides, seed)`` → бакетов с вредом."""

    return len(oracle_harm(*args))


class OracleContour(PolicyContour):
    """Оракул: надбавка ``mult`` там, где в мире без политики точно будет вред.

    В момент решения контур видит закрытый бакет ``b − 1`` и ставит цену на
    текущий бакет ``b``. Оракул знает ``unfulfilled`` мира без политики в
    бакетах ``b + offset`` для каждого ``offset`` из ``offsets`` и срабатывает,
    если их сумма не меньше ``min_harm``. Сеть горизонта H предсказывает бакет
    ``b − 1 + H/шаг``: при H=15 это ``offset = 2``. Оракул — потолок любой
    политики «порог по прогнозу вреда»: прогноз идеален, обратной связи нет.
    """

    def __init__(
        self,
        cfg: AppConfig,
        source: LiveRowSource,
        harm: dict,
        *,
        offsets: tuple[int, ...],
        min_harm: int,
        mult: float,
    ) -> None:
        super().__init__(cfg, None, source)
        self._harm = harm
        self._offsets = tuple(offsets)
        self._min_harm = min_harm
        self._mult = mult

    def _multiplier(self, zone_id: int, ts) -> float:
        current = ml_dataset.bucket_index(ts, self._step) + 1
        total = sum(self._harm.get((zone_id, current + o), 0) for o in self._offsets)
        return self._mult if total >= self._min_harm else 1.0


class HybridContour(PolicyContour):
    """Гибрид: обратная связь по факту нехватки + прогноз вреда сети.

    На бакет зоны считаются ``deficit`` — правило контракта по последнему закрытому
    бакету (как у реактивной) и ``p`` — сырая вероятность сети (без δ). Множитель:

    - **ворота**: ``mult``, если ``deficit`` и ``p ≥ gate`` (``gate = 0`` — реактивная);
    - **ранний взвод**: ``early_mult``, если ``p ≥ early`` (без факта нехватки);
    - **удержание**: при ``hold`` надбавка прошлого бакета зоны держится ``mult``,
      пока ``deficit`` сохраняется, даже если ``p`` упал (снимает самоподавление
      сети собственной ценой).

    Итог — максимум сработавших правил, иначе 1.00.
    """

    def __init__(
        self,
        cfg: AppConfig,
        predictor,
        source: LiveRowSource,
        *,
        mult: float,
        gate: float = 0.0,
        early: float = float("inf"),
        early_mult: float = 1.15,
        hold: bool = False,
    ) -> None:
        super().__init__(cfg, predictor, source)
        self._mult = mult
        self._gate = gate
        self._early = early
        self._early_mult = early_mult
        self._hold = hold
        self._last: dict[int, float] = {}

    def _multiplier(self, zone_id: int, ts) -> float:
        point = self._source.metrics_at(zone_id, ts)
        deficit = point is not None and ml_dataset.deficit_label(
            point.requests, point.free_drivers, point.unfulfilled
        )
        p = self._predictor.predict(zone_id, ts)
        p = 0.0 if p is None else p
        mult = 1.0
        if deficit and p >= self._gate:
            mult = self._mult
        if p >= self._early:
            mult = max(mult, self._early_mult)
        if self._hold and deficit and self._last.get(zone_id, 1.0) > 1.0:
            mult = max(mult, self._mult)
        self._last[zone_id] = mult
        return mult


# --- прототип: эластичное предложение (водители выходят на линию по сигналу) -----------


ELASTIC_DEFAULTS = {
    "part": 60,  # подработчики сверх штатного парка: на линии только по вызову
    "lag": 20,  # минут от решения до выхода на линию в зоне сигнала
    "session": 120,  # минут на линии после выхода (затем уходят, занятый — после заказа)
    "sens": 0.3,  # доля свободного резерва на единицу (сигнал − 1) на одно решение
    "channel": "heatmap",  # heatmap — сигнал только водителям; price — ещё и цена пассажиру
}


class ElasticSupplyEngine(Engine):
    """Движок с подработчиками: часть водителей выходит на линию по сигналу политики.

    Штатный парк (``base_count`` первых id) живёт по профилю смен как раньше.
    Подработчики (id ≥ ``base_count``) на линии только по вызову: раз в бакет
    (минута кратна шагу агрегации) по каждой зоне читается сигнал ``callout_fn``
    (множитель ≥ 1); при сигнале ``s > 1`` из свободного резерва вызывается
    ``floor(резерв × sens × (s − 1))`` водителей. Они выходят на линию через ``lag``
    минут в зоне сигнала и работают ``session`` минут. Без сигнала мир побитово
    равен штатному (подработчики уходят offline в первую минуту, rng не тратится).
    ``part_minutes`` — минуты подработчиков на линии с ``count_from`` (цена мер).
    """

    def __init__(self, cfg: AppConfig, *, base_count: int, callout_fn, count_from, **kw) -> None:
        params = ELASTIC_DEFAULTS | {k: v for k, v in kw.items() if k in ELASTIC_DEFAULTS}
        super().__init__(cfg, seed=kw.get("seed"), price_fn=kw.get("price_fn"))
        self._base = base_count
        self._base_drivers = cfg.drivers.model_copy(update={"count": base_count})
        self._callout_fn = callout_fn
        self._lag = int(params["lag"])
        self._session = int(params["session"])
        self._sens = float(params["sens"])
        self._pool = list(range(base_count, cfg.drivers.count))  # свободный резерв
        self._pending: list[tuple] = []  # (момент выхода, зона, id)
        self._sessions: dict[int, object] = {}  # id → момент окончания сессии
        self._count_from = count_from
        self._started = False
        self.part_minutes = 0
        self.callouts = 0

    def _apply_shifts(self, t) -> None:  # noqa: C901 — копия штатной сверки + резерв
        from bisect import insort

        from simulator.drivers import DriverState
        from simulator.shifts import online_target

        fleet, drivers = self._fleet, self._fleet._drivers
        zone_count = len(self._city.zones)
        # 1. Штатный парк — та же сверка, что Engine._apply_shifts, только по id < base.
        base = self._base
        target = online_target(self._base_drivers, t)
        off_part = sum(
            1 for d in range(base, len(drivers)) if drivers[d].state is DriverState.OFFLINE
        )
        leaving_reg = sum(1 for d in fleet._leaving if d < base)
        online = base - (fleet.offline_count() - off_part) - leaving_reg
        if online > target:
            excess = online - target
            for d in range(base - 1, -1, -1):
                if excess == 0:
                    break
                if drivers[d].state is not DriverState.FREE:
                    continue
                zone = drivers[d].zone
                fleet.go_offline(d)
                self._free_ids[zone].remove(d)
                excess -= 1
            for d in range(base - 1, -1, -1):
                if excess == 0:
                    break
                if fleet.is_leaving(d):
                    continue
                if drivers[d].state in (
                    DriverState.TO_PICKUP,
                    DriverState.ON_TRIP,
                    DriverState.RELOCATING,
                ):
                    fleet.mark_leaving(d)
                    excess -= 1
        elif online < target:
            need = target - online
            for d in range(base):
                if need == 0:
                    break
                if fleet.is_leaving(d):
                    fleet.unmark_leaving(d)
                    need -= 1
            for d in range(base):
                if need == 0:
                    break
                if drivers[d].state is not DriverState.OFFLINE:
                    continue
                zone = d % zone_count
                fleet.go_online(d, zone)
                insort(self._free_ids[zone], d)
                need -= 1
        # 2. Подработчики: в первую минуту мира все уходят offline.
        if not self._started:
            self._started = True
            for d in range(base, len(drivers)):
                fleet.go_offline(d)
                self._free_ids[drivers[d].zone].remove(d)
        # Конец сессии: свободный уходит сразу, занятый — после заказа.
        for d, until in list(self._sessions.items()):
            state = drivers[d].state
            if state is DriverState.OFFLINE:
                del self._sessions[d]
                self._pool.append(d)
            elif t >= until:
                if state is DriverState.FREE:
                    fleet.go_offline(d)
                    self._free_ids[drivers[d].zone].remove(d)
                    del self._sessions[d]
                    self._pool.append(d)
                elif not fleet.is_leaving(d):
                    fleet.mark_leaving(d)
        # Выход вызванных на линию.
        still = []
        for when, zone, d in self._pending:
            if t >= when:
                fleet.go_online(d, zone)
                insort(self._free_ids[zone], d)
                self._sessions[d] = add_minutes(t, self._session)
            else:
                still.append((when, zone, d))
        self._pending = still
        # Решения раз в бакет: вызов из резерва по сигналу зоны.
        if self._callout_fn is not None and t.minute % self._cfg.engine.agg_step_min == 0:
            self._pool.sort()
            for zone in range(zone_count):
                s = self._callout_fn(zone, t)
                if s <= 1.0 or not self._pool:
                    continue
                n = math.floor(len(self._pool) * self._sens * (s - 1.0) + 1e-9)
                for d in self._pool[:n]:
                    self._pending.append((add_minutes(t, self._lag), zone, d))
                self._pool = self._pool[n:]
                if t >= self._count_from:
                    self.callouts += n
        if t >= self._count_from:
            self.part_minutes += sum(
                1 for d in self._sessions if drivers[d].state is not DriverState.OFFLINE
            )


def run_world_elastic(
    cfg: AppConfig, *, contour: PolicyContour | None, seed: int, elastic: dict
) -> tuple[MemWriter, ElasticSupplyEngine]:
    """Прогон с подработчиками. ``channel = heatmap``: политика зовёт водителей, цена
    пассажиру — тариф мира; ``price``: тот же множитель ещё и в цене пассажиру."""

    params = ELASTIC_DEFAULTS | elastic
    base = cfg.drivers.count
    big = cfg.model_copy(
        update={"drivers": cfg.drivers.model_copy(update={"count": base + int(params["part"])})}
    )
    activation = add_minutes(world_start(cfg), cfg.ml.train_days * 1440)
    callout = None
    if contour is not None:

        def callout(zone, t):
            return 1.0 if t < activation else contour._cached_multiplier(zone, t)

    engine = ElasticSupplyEngine(
        big,
        base_count=base,
        callout_fn=callout,
        count_from=activation,
        seed=seed,
        price_fn=contour.price if contour is not None and params["channel"] == "price" else None,
        **params,
    )
    if contour is not None:
        contour.bind(engine)
    writer = MemWriter()
    engine.run(
        cfg.ml.train_days + cfg.ml.test_days,
        writer,
        on_minute=None if contour is None else contour.on_minute,
        on_metrics_row=None if contour is None else contour.on_metrics_row,
    )
    return writer, engine


def policy_point(args: tuple) -> dict:
    """Одна ветка на одном мире: ``(overrides, kind, param, seed[, opts])`` → итоги.

    ``opts`` — строка цели или словарь ``{"target", "history", "lag"}`` (см. :func:`_opts`).

    ``kind``: ``a-base``; ``b-reactive`` и ``b-lost-only`` (``param`` —
    множитель надбавки ``policy.reactive_mult``); ``oracle`` (``param`` —
    множитель, ``opts["offsets"]`` и ``opts["min_harm"]`` — см.
    :class:`OracleContour`; вред мира без политики берётся из
    :func:`oracle_harm`, его нужно посчитать заранее); ``mlp`` и ``lstm`` (``param`` —
    δ-сдвиг ступеней, сети берутся из кэша :func:`prepare_world`). Сетка по
    ``param`` даёт кривую «срабатывания → спасённые заказы» для каждой политики.
    """

    torch_threads()
    overrides, kind, param, seed, *rest = args
    opts = _opts(rest[0] if rest else None)
    extra: dict = {}
    if kind in ("b-reactive", "b-lost-only"):
        extra["policy.reactive_mult"] = param
    elif kind in FAMILY_BRANCH:
        extra["policy.shift"] = {kind: param}
    cfg = make_cfg(overrides, **extra)
    step = cfg.engine.agg_step_min
    source = LiveRowSource(agg_step_min=step)
    if kind == "a-base":
        contour = None
    elif kind == "b-reactive":
        contour = ReactiveContour(cfg, source)
    elif kind == "b-lost-only":
        contour = LostOnlyReactiveContour(cfg, source)
    elif kind == "oracle":
        contour = OracleContour(
            cfg,
            source,
            oracle_harm(overrides, seed),
            offsets=tuple(opts["offsets"]),
            min_harm=int(opts["min_harm"]),
            mult=param,
        )
    elif kind == "hybrid":
        wd = Path(WORK_DIR) / world_key(overrides)
        predictor = NetworkPredictor(
            PriceMaskedSource(source) if opts["mask_price"] else source,
            cfg,
            family=opts.get("family", "mlp"),
            artifacts_root=artifacts_root(wd, opts["target"], opts["history"]),
        )
        contour = HybridContour(
            cfg,
            predictor,
            source,
            mult=param,
            gate=opts.get("gate", 0.0),
            early=opts.get("early", float("inf")),
            early_mult=opts.get("early_mult", 1.15),
            hold=opts.get("hold", False),
        )
    else:
        wd = Path(WORK_DIR) / world_key(overrides)
        predictor = NetworkPredictor(
            PriceMaskedSource(source) if opts["mask_price"] else source,
            cfg,
            family=kind,
            artifacts_root=artifacts_root(wd, opts["target"], opts["history"]),
        )
        contour = PolicyContour(cfg, predictor, source, model=kind)
    if contour is not None:
        with_data_lag(contour, opts["lag"])
    extra_row: dict = {}
    if opts.get("elastic") is not None:
        writer, engine = run_world_elastic(cfg, contour=contour, seed=seed, elastic=opts["elastic"])
        row = outcomes(writer, cfg)
        extra_row = {"part_hours": engine.part_minutes / 60.0, "callouts": engine.callouts}
    else:
        row = outcomes(run_world(cfg, contour=contour, seed=seed), cfg)
    fire = float("nan") if contour is None else 100.0 * contour.fire_rate
    return row | {"pol": kind, "param": param, "seed": seed, "fire": fire} | opts | extra_row


def torch_threads(n: int = 1) -> None:
    """Один поток torch на процесс: параллелим прогоны, а не матричные операции."""

    import torch

    torch.set_num_threads(n)
