"""CLI генерации веток будущего и сетки миров: ``python -m simulator.branches``.

Ветки — именованные миры-прогоны одного seed в каталоге миров (конвенция
``<ветка>.db``) — фундамент попарного сравнения политик вехи M5+M6 (§3 плана
``docs/plan-m5-m6-policy.md``): до активации политики ветки побитово
идентичны, при изменении тарифа эндогенная история закономерно расходится, а
экзогенная (дождь, surge-инъекции) остаётся побайтово общей благодаря
разделению rng-потоков движка. Подключение новой ветки — запись в реестре
``BRANCHES`` (фабрика контуров), функция генерации не переписывается
(design решение 4 m6-rng-branches). Источники политики: ``None`` — мир без
политики (ветка ``a-base``), прямая фабрика ``price_fn`` (прецедент тестов),
фабрика контуров политики change ``m6-policy`` (спека simulator-branches):
контур отдаёт движку тариф ``price`` и наблюдателей истории для ``run``.
Контурные ветки реестра: ``b-reactive`` — реактивное правило, ``c-mlp``/
``d-lstm`` — ступени по прогнозу артефактов M3 (их отсутствие — guard до
старта прогона, CLI — exit 2).

Канал политики (change ``policy-driver-signal``, спека simulator-branches):
фабрика ветки возвращает пару «контур цены | None» и «контур сигнала | None»
по ``policy.channel``. ``price`` — цена по M3 (как раньше), сигнал равен
тарифу; ``heatmap`` — цена равна тарифу мира, политика ставит только сигнал
водителям (``driver_signal_fn``): у B — реактивный, у C/D — по сети
планирования (её артефакт семейства обязателен до прогона); ``both`` — оба
контура, каждый со своим источником истории. Явный ``price_fn`` приоритетнее
контура цены. Метаданные ветки — ``<имя>.meta.json`` рядом с БД: канал, доли
срабатываний контуров, вызовы и минуты подработчиков в окне тест-недели.

Тень-прогнозы (change ``m6-worlds``, спека simulator-policy): генерация
ветки реестра по умолчанию пишет в ``forecasts`` БД ветки прогнозы обеих
сетей с границы окна активации (``shadow=False`` — прямые тестовые прогоны
математики политик без артефактов); каталоги обоих семейств обязательны у
любой ветки реестра. Сетка ``generate_worlds``/``--all`` генерирует все ветки
реестра по порядку с guard-пречеком всех целевых БД; реестр миров
``list_worlds`` — производная реестра веток для UI следующих change'ов.
Прогон и запись повторяют CLI генерации истории: guard непустой ``trips``,
прогресс раз в сим-час и сводка §4 — из общего модуля ``simulator.writers``.
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

from common.config import AppConfig, load_config
from common.timeutil import add_minutes
from ml.dataset import world_start
from ml.train import ARTIFACTS_ROOT
from simulator.engine import Engine, Summary
from simulator.live import MinuteRecord
from simulator.policy import (
    LiveRowSource,
    NetworkPredictor,
    PlanningSignalContour,
    PolicyContour,
    ReactiveContour,
    ReactiveSignalContour,
    ShadowForecaster,
)
from simulator.writers import GUARD_MESSAGE, SqliteTripWriter, print_progress, print_summary
from storage.db import connect
from storage.repo import insert_forecasts
from storage.schema import init_db

WORLDS_DIR = Path("data/worlds")

ContourPair = tuple[PolicyContour | None, PolicyContour | None]
"""Пара фабрики ветки: (контур цены | None, контур сигнала | None) по каналу."""


@dataclass(frozen=True)
class BranchSpec:
    """Правило генерации мира ветки (design 4 m6-rng-branches, расширения m6-policy/m6-worlds).

    ``name`` — ключ реестра и имя файла БД; ``label`` — человекочитаемая метка
    политики реестра миров (спека simulator-branches m6-worlds). Источник
    политики — ровно один: ``price_fn_factory`` — прямая фабрика тарифной
    функции ``price_fn(zone_id, t)`` (шов §10.13); ``contour_factory`` —
    фабрика пары контуров политики по ``policy.channel`` (спеки
    ``simulator-branches``/``policy-driver-signal``): контур цены подключается
    как ``price_fn``, контур сигнала — как ``driver_signal_fn``; каждый отдаёт
    наблюдателей истории для ``run``. Оба ``None`` — мир без политики (ветка
    ``a-base``): тариф зон даёт среда.
    """

    name: str
    label: str
    price_fn_factory: Callable[[AppConfig], Callable[[int, datetime], float]] | None = None
    contour_factory: Callable[[AppConfig], ContourPair] | None = None


@dataclass(frozen=True)
class WorldInfo:
    """Запись реестра миров: имя, путь БД, метка политики и наличие файла (m6-worlds)."""

    name: str
    db: Path
    label: str
    exists: bool


def _reactive_contours(cfg: AppConfig) -> ContourPair:
    """Фабрика контуров ветки B по каналу: реактивное правило закрытых бакетов.

    В канале ``price`` — контур цены, в ``heatmap`` — контур сигнала водителям
    с тем же правилом, в ``both`` — оба независимых контура (design решение 3:
    у каждого свой ``LiveRowSource``).
    """

    step = cfg.engine.agg_step_min
    channel = cfg.policy.channel
    price = (
        ReactiveContour(cfg, LiveRowSource(agg_step_min=step))
        if channel in ("price", "both")
        else None
    )
    signal = (
        ReactiveSignalContour(cfg, LiveRowSource(agg_step_min=step))
        if channel in ("heatmap", "both")
        else None
    )
    return price, signal


def _artifacts_root(cfg: AppConfig) -> Path:
    """Корень артефактов M3 ветки: ``ml.artifacts_root`` конфига.

    Дефолт ``artifacts`` совпадает с константой модуля ``ARTIFACTS_ROOT``; сама
    константа остаётся точкой подмены в тестах, поэтому при дефолтном значении
    возвращается она (а не равный ей путь).
    """

    configured = Path(cfg.ml.artifacts_root)
    return ARTIFACTS_ROOT if configured == Path("artifacts") else configured


def _stair_contours_factory(family: str) -> Callable[[AppConfig], ContourPair]:
    """Фабрика контуров веток C/D по каналу для семейства ``family``.

    Контур цены — ступени по прогнозу артефакта M3 семейства; контур сигнала —
    порог по вероятности вреда сети планирования того же семейства. Каталог
    артефакта M3 — самый свежий ``{family}-v1-*`` в корне ``ARTIFACTS_ROOT``
    (читается при вызове фабрики, что позволяет тестам подменять корень);
    отсутствие каталога M3 — ``FileNotFoundError`` при построении контура, до
    старта прогона (guard, спека simulator-branches). Корень артефактов
    планирования — ``ml.planning.artifacts_root`` конфига; его отсутствие —
    тот же guard, если канал требует сигнал.
    """

    def factory(cfg: AppConfig) -> ContourPair:
        step = cfg.engine.agg_step_min
        channel = cfg.policy.channel
        root = _artifacts_root(cfg)
        price: PolicyContour | None = None
        if channel in ("price", "both"):
            source = LiveRowSource(agg_step_min=step)
            predictor = NetworkPredictor(source, cfg, family=family, artifacts_root=root)
            price = PolicyContour(cfg, predictor, source, model=family)
        signal: PolicyContour | None = None
        if channel in ("heatmap", "both"):
            signal = PlanningSignalContour(cfg, LiveRowSource(agg_step_min=step), family=family)
        return price, signal

    return factory


BRANCHES: dict[str, BranchSpec] = {
    "a-base": BranchSpec("a-base", "мир без политики"),
    "b-reactive": BranchSpec(
        "b-reactive", "реактивная политика", contour_factory=_reactive_contours
    ),
    "c-mlp": BranchSpec(
        "c-mlp", "ступени по прогнозу MLP", contour_factory=_stair_contours_factory("mlp")
    ),
    "d-lstm": BranchSpec(
        "d-lstm", "ступени по прогнозу LSTM", contour_factory=_stair_contours_factory("lstm")
    ),
}


def _chain(
    *callbacks: Callable[..., None] | None,
) -> Callable[..., None] | None:
    """Скомпоновать наблюдателей движка в один (``None``-звенья пропускаются)."""

    live = [callback for callback in callbacks if callback is not None]
    if not live:
        return None
    if len(live) == 1:
        return live[0]

    def chained(*args: object) -> None:
        for callback in live:
            callback(*args)

    return chained


def list_worlds(cfg: AppConfig, *, worlds_dir: str | Path | None = None) -> list[WorldInfo]:
    """Реестр миров (задача 3.3): производная ``BRANCHES``, не скан каталога.

    Порядок записей — порядок реестра веток; состав фиксирован (§2.7 плана:
    мусор в каталоге мирами не становится); путь БД — ``<worlds_dir>/<имя>.db``,
    ``worlds_dir`` — параметр вызова или ``policy.worlds_dir`` конфига.
    Отсутствие файла БД — не ошибка: реестр описывает миры генератора,
    наличие файла — отдельный признак (потребитель — UI следующих change'ов).
    """

    root = Path(worlds_dir) if worlds_dir is not None else Path(cfg.policy.worlds_dir)
    entries = []
    for spec in BRANCHES.values():
        db = root / f"{spec.name}.db"
        entries.append(WorldInfo(name=spec.name, db=db, label=spec.label, exists=db.exists()))
    return entries


class _TestWindowCostObserver:
    """Наблюдатель минут: стоимость меры в окне тест-недели (спека simulator-branches).

    Окно — ``[старт мира + ml.train_days суток, + ml.test_days суток)``.
    ``minutes`` — сумма ``MinuteRecord.part_online`` по минутам окна;
    ``callouts`` — число выходов подработчиков на линию: наблюдаемые приросты
    ``part_online``. Выход — следствие вызова с задержкой ``drivers.callout.lag_min``;
    вызов, чей выход попадает за конец окна (последние минуты прогона), в
    наблюдаемых метаданных не отражается. При выключенном вызове наблюдатель
    не подключается и оба счётчика равны 0.
    """

    def __init__(self, cfg: AppConfig) -> None:
        """Запомнить границы окна тест-недели."""

        self._start = add_minutes(world_start(cfg), cfg.ml.train_days * 1440)
        self._end = add_minutes(self._start, cfg.ml.test_days * 1440)
        self._previous: int | None = None
        self.callouts = 0
        self.minutes = 0

    def on_minute(self, record: MinuteRecord) -> None:
        """Наблюдатель минут движка: минуты и выходы подработчиков окна."""

        if self._start <= record.t < self._end:
            if self._previous is not None and record.part_online > self._previous:
                self.callouts += record.part_online - self._previous
            self.minutes += record.part_online
        self._previous = record.part_online


def _meta_path(db: Path) -> Path:
    """Путь метаданных ветки: ``<имя БД без .db>.meta.json`` рядом с БД."""

    return db.parent / f"{db.stem}.meta.json"


def _write_meta(db: Path, payload: dict) -> Path:
    """Записать метаданные ветки (детерминированный json, спека simulator-branches)."""

    path = _meta_path(db)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def generate_branch(
    name: str,
    cfg: AppConfig,
    *,
    seed: int | None = None,
    days: int = 37,
    db_path: str | None = None,
    price_fn: Callable[[int, datetime], float] | None = None,
    shadow: bool = True,
) -> Summary | None:
    """Сгенерировать ветку ``name``: полный прогон движка в SQLite-БД (§3 плана).

    ``seed=None`` — seed из конфига; ``days=37`` — формат ветки «30 суток
    истории + тест-неделя»; ``db_path=None`` → ``WORLDS_DIR/<name>.db``
    (каталог создаётся). Политика ветки — по реестру: фабрика контуров строит
    пару «цена | сигнал» по ``policy.channel`` до прогона (guard отсутствия
    артефактов у ``c-mlp``/``d-lstm`` — ``FileNotFoundError`` здесь; для канала
    ``heatmap``/``both`` — и артефактов планирования), движок получает
    ``contour.price`` контура цены как ``price_fn`` и контура сигнала как
    ``driver_signal_fn``, ``bind`` даёт каждому доступ к тарифу мира (design
    решение 5), наблюдатели обоих контуров компонуются в ``run``; прямая
    фабрика даёт ``price_fn``. Явный аргумент ``price_fn`` имеет приоритет над
    контуром цены (спека: параметр функции генерации); контур сигнала канала
    при этом сохраняется. ``shadow=True`` (умолчание, полный формат ветки) —
    тень-прогнозы обеих сетей в ``forecasts`` БД ветки с границы окна
    активации: каталоги артефактов ОБЕИХ семейств обязательны у любой ветки
    реестра, их отсутствие — ``FileNotFoundError`` до прогона (CLI — exit 2);
    ``shadow=False`` — тени нет, артефакты не нужны (прямые тестовые прогоны
    математики политик, спека simulator-branches). Guard §4 M2 до прогона:
    непустая таблица ``trips`` в целевой БД — ``GUARD_MESSAGE`` в stderr и
    возврат ``None`` (CLI переводит в exit 2); неизвестное имя —
    ``ValueError`` с именем ветки в сообщении, прогон не начинается. Успех —
    ``Summary``; прогресс раз в сим-час и сводка формата §4 печатаются теми
    же функциями ``simulator.writers``, что и у CLI ``simulator`` (при
    включённых отказах пассажиров — с пятой строкой ``cancel:``); после
    контурной ветки печатается диагностика срабатываний (``policy_fire_rate``
    для цены, ``signal_fire_rate`` для сигнала — наблюдаемая база сверки
    честности, спеки m6-worlds/policy-driver-signal). После успешного прогона
    рядом с БД пишется ``<имя>.meta.json`` (канал, доли срабатываний, вызовы и
    минуты подработчиков тест-недели); неуспешный прогон файла не создаёт.
    """

    spec = BRANCHES.get(name)
    if spec is None:
        raise ValueError(f"неизвестная ветка {name!r}; реестр веток: {', '.join(sorted(BRANCHES))}")

    db = WORLDS_DIR / f"{name}.db" if db_path is None else Path(db_path)
    db_dir = os.path.dirname(db)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = connect(str(db))
    init_db(conn)
    existing = conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    if existing > 0:
        print(GUARD_MESSAGE, file=sys.stderr)
        return None

    price_contour: PolicyContour | None = None
    signal_contour: PolicyContour | None = None
    policy = price_fn
    # Контуры строятся до прогона: NetworkPredictor/PlanningPredictor поднимают
    # FileNotFoundError, если каталогов артефактов нет (guard). Явный price_fn
    # отменяет только контур цены; контур сигнала канала нужен и при нём.
    needs_signal = cfg.policy.channel in ("heatmap", "both")
    if spec.contour_factory is not None and (price_fn is None or needs_signal):
        price_contour, signal_contour = spec.contour_factory(cfg)
    if price_fn is not None:
        price_contour = None
    elif price_contour is None and spec.price_fn_factory is not None:
        policy = spec.price_fn_factory(cfg)

    shadow_caster: ShadowForecaster | None = None
    if shadow:
        # Guard обоих семейств — у любой ветки реестра, до старта прогона;
        # flush на закрытый бакет — буфер строк тени в forecasts БД ветки.
        shadow_caster = ShadowForecaster(
            cfg, artifacts_root=_artifacts_root(cfg), flush=partial(insert_forecasts, conn)
        )
    cost = _TestWindowCostObserver(cfg) if cfg.drivers.callout.enabled else None
    engine = Engine(
        cfg,
        seed=seed,
        price_fn=policy if price_contour is None else price_contour.price,
        driver_signal_fn=None if signal_contour is None else signal_contour.price,
    )
    # Поздняя привязка: доступ к тарифу мира у каждого контура (design решение 5).
    for contour in (price_contour, signal_contour):
        if contour is not None:
            contour.bind(engine)
    summary = engine.run(
        days,
        SqliteTripWriter(conn),
        progress=print_progress,
        on_minute=_chain(
            cost.on_minute if cost is not None else None,
            price_contour.on_minute if price_contour is not None else None,
            signal_contour.on_minute if signal_contour is not None else None,
            shadow_caster.on_minute if shadow_caster is not None else None,
        ),
        on_metrics_row=_chain(
            price_contour.on_metrics_row if price_contour is not None else None,
            signal_contour.on_metrics_row if signal_contour is not None else None,
            shadow_caster.on_metrics_row if shadow_caster is not None else None,
        ),
    )
    print_summary(
        summary,
        str(db),
        cancel_enabled=cfg.drivers.dispatch.cancel_prob_per_min > 0.0,
    )
    if price_contour is not None:
        print(
            f"branch: name={name} policy_fire_rate={price_contour.fire_rate:.3f}"
            f" fired={price_contour.fire_count}/{price_contour.fire_slots}"
        )
    if signal_contour is not None:
        print(
            f"branch: name={name} signal_fire_rate={signal_contour.fire_rate:.3f}"
            f" fired={signal_contour.fire_count}/{signal_contour.fire_slots}"
        )
    _write_meta(
        db,
        {
            "branch": name,
            "channel": cfg.policy.channel,
            "seed": cfg.seed if seed is None else seed,
            "days": days,
            "price_fire_rate": None if price_contour is None else price_contour.fire_rate,
            "signal_fire_rate": None if signal_contour is None else signal_contour.fire_rate,
            "callouts_test": 0 if cost is None else cost.callouts,
            "callout_minutes_test": 0 if cost is None else cost.minutes,
        },
    )
    return summary


def generate_worlds(
    cfg: AppConfig,
    *,
    seed: int | None = None,
    days: int = 37,
    worlds_dir: str | Path | None = None,
) -> dict[str, Summary] | None:
    """Сгенерировать сетку миров: все ветки реестра по порядку (задача 3.1).

    Общие seed и число дней, путь каждой БД — ``<worlds_dir>/<имя>.db``
    (``worlds_dir`` — параметр или ``policy.worlds_dir``), тень включена,
    δ-сдвиги — из текущего конфига (результат δ-калибровки). Guard-пречек всех
    целевых БД до первой ветки (fail fast: полусетка после минут прогона хуже
    отказа до старта): непустая ``trips`` хотя бы в одной — ``GUARD_MESSAGE``
    со списком веток в stderr, возврат ``None``, ни одна БД не трогается
    (CLI — exit 2). После каждой ветки — сводка формата CLI ветки и её
    wall-время, после контурных — диагностика ``policy_fire_rate`` (внутри
    :func:`generate_branch`); в конце — итог сетки. Отсутствие каталогов
    артефактов тени — ``FileNotFoundError`` до первой ветки.
    """

    root = Path(worlds_dir) if worlds_dir is not None else Path(cfg.policy.worlds_dir)
    root.mkdir(parents=True, exist_ok=True)
    busy: list[str] = []
    for name in BRANCHES:
        db = root / f"{name}.db"
        if not db.exists():
            continue
        conn = connect(str(db))
        try:
            if conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0] > 0:
                busy.append(name)
        finally:
            conn.close()
    if busy:
        print(f"{GUARD_MESSAGE} занятые ветки: {', '.join(busy)}", file=sys.stderr)
        return None

    started = time.perf_counter()
    summaries: dict[str, Summary] = {}
    for name in BRANCHES:
        branch_started = time.perf_counter()
        summary = generate_branch(name, cfg, seed=seed, days=days, db_path=str(root / f"{name}.db"))
        if summary is None:  # после пре-чека недостижимо; мир не оставляем молчаливым
            print(f"ветка {name}: guard непустой trips вопреки пре-чеку сетки", file=sys.stderr)
            return None
        summaries[name] = summary
        print(f"world: branch={name} wall={time.perf_counter() - branch_started:.1f}s")
    print(f"worlds: branches={len(summaries)} wall={time.perf_counter() - started:.1f}s dir={root}")
    return summaries


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Аргументы CLI: ``--branch`` или ``--all`` — ровно одна из операций (спека)."""

    parser = argparse.ArgumentParser(
        prog="simulator.branches",
        description=("Генерация ветки будущего или сетки миров в data/worlds (§3 плана M5+M6)."),
    )
    parser.add_argument("--branch", default=None, help="имя ветки из реестра BRANCHES")
    parser.add_argument(
        "--all",
        action="store_true",
        help="генерация сетки всех веток реестра в worlds_dir (пути БД задаёт реестр)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=37,
        help="число суток генерации (по умолчанию 37: 30 истории + тест-неделя)",
    )
    parser.add_argument("--config", default="config.yaml", help="путь к конфигу YAML")
    parser.add_argument("--seed", type=int, default=None, help="seed (по умолчанию из конфига)")
    parser.add_argument(
        "--db", default=None, help="путь БД ветки (по умолчанию data/worlds/<ветка>.db)"
    )
    parser.add_argument(
        "--worlds-dir",
        default=None,
        help="каталог миров для --all (по умолчанию policy.worlds_dir из конфига)",
    )
    args = parser.parse_args(argv)
    if bool(args.branch) == bool(args.all):
        parser.error("укажите ровно одну операцию: --branch <имя> или --all")
    if args.all and args.db is not None:
        parser.error("--db несовместим с --all: пути БД определяет реестр миров")
    return args


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI: неизвестная ветка/guard → stderr + exit 2; успех → 0.

    Отсутствие каталогов артефактов (контур ``c-mlp``/``d-lstm`` или тень любой
    ветки) — тот же код: сообщение об ошибке в stderr и exit 2, ни одной строки
    в БД (спека simulator-branches). ``--all`` — сетка миров с guard-пречеком
    всех целевых БД (спека m6-worlds); ``--db`` с ``--all`` отклоняется
    argparse до старта.
    """

    args = _parse_args(argv)
    cfg: AppConfig = load_config(args.config)
    try:
        if args.all:
            result = generate_worlds(
                cfg, seed=args.seed, days=args.days, worlds_dir=args.worlds_dir
            )
            return 0 if result is not None else 2  # None — guard сетки
        if args.branch not in BRANCHES:
            print(
                f"неизвестная ветка {args.branch!r}; реестр веток: {', '.join(sorted(BRANCHES))}",
                file=sys.stderr,
            )
            return 2
        db_path = str(WORLDS_DIR / f"{args.branch}.db") if args.db is None else args.db
        summary = generate_branch(args.branch, cfg, seed=args.seed, days=args.days, db_path=db_path)
        return 0 if summary is not None else 2  # None — guard непустой trips
    except FileNotFoundError as exc:  # нет артефактов модели/тени — guard до прогона
        print(f"нет артефактов модели: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
