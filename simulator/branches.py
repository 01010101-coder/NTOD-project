"""CLI генерации веток будущего и сетки миров: ``python -m simulator.branches``.

Ветки — именованные миры-прогоны одного seed в каталоге миров (конвенция
``<ветка>.db``) — фундамент попарного сравнения политик вехи M5+M6 (§3 плана
``docs/plan-m5-m6-policy.md``): до активации политики ветки побитово
идентичны, при изменении тарифа эндогенная история закономерно расходится, а
экзогенная (дождь, surge-инъекции) остаётся побайтово общей благодаря
разделению rng-потоков движка. Подключение новой ветки — запись в реестре
``BRANCHES`` (фабрика политики), функция генерации не переписывается
(design решение 4 m6-rng-branches). Источники политики: ``None`` — мир без
политики (ветка ``a-base``), прямая фабрика ``price_fn`` (прецедент тестов),
фабрика контура политики change ``m6-policy`` (спека simulator-branches):
контур отдаёт движку тариф ``price`` и наблюдателей истории для ``run``.
Контурные ветки реестра: ``b-reactive`` — реактивное правило, ``c-mlp``/
``d-lstm`` — ступени по прогнозу артефактов M3 (их отсутствие — guard до
старта прогона, CLI — exit 2).

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
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

from common.config import AppConfig, load_config
from ml.train import ARTIFACTS_ROOT
from simulator.engine import Engine, Summary
from simulator.policy import (
    LiveRowSource,
    NetworkPredictor,
    PolicyContour,
    ReactiveContour,
    ShadowForecaster,
)
from simulator.writers import GUARD_MESSAGE, SqliteTripWriter, print_progress, print_summary
from storage.db import connect
from storage.repo import insert_forecasts
from storage.schema import init_db

WORLDS_DIR = Path("data/worlds")


@dataclass(frozen=True)
class BranchSpec:
    """Правило генерации мира ветки (design 4 m6-rng-branches, расширения m6-policy/m6-worlds).

    ``name`` — ключ реестра и имя файла БД; ``label`` — человекочитаемая метка
    политики реестра миров (спека simulator-branches m6-worlds). Источник
    политики — ровно один: ``price_fn_factory`` — прямая фабрика тарифной
    функции ``price_fn(zone_id, t)`` (шов §10.13); ``contour_factory`` —
    фабрика контура политики (спека simulator-branches): контур предоставляет
    движку тариф ``price`` и наблюдателей истории для ``run``. Оба ``None`` —
    мир без политики (ветка ``a-base``): тариф зон даёт среда.
    """

    name: str
    label: str
    price_fn_factory: Callable[[AppConfig], Callable[[int, datetime], float]] | None = None
    contour_factory: Callable[[AppConfig], PolicyContour] | None = None


@dataclass(frozen=True)
class WorldInfo:
    """Запись реестра миров: имя, путь БД, метка политики и наличие файла (m6-worlds)."""

    name: str
    db: Path
    label: str
    exists: bool


def _reactive_contour(cfg: AppConfig) -> PolicyContour:
    """Фабрика контура ветки B: реактивное правило по факту закрытых бакетов."""

    return ReactiveContour(cfg, LiveRowSource(agg_step_min=cfg.engine.agg_step_min))


def _stair_contour_factory(family: str) -> Callable[[AppConfig], PolicyContour]:
    """Фабрика контура ступеней по прогнозу артефактов семейства ``family``.

    Каталог артефакта — самый свежий ``{family}-v1-*`` в корне ``ARTIFACTS_ROOT``
    (читается при вызове фабрики, что позволяет тестам подменять корень);
    отсутствие каталога — ``FileNotFoundError`` при построении контура, до
    старта прогона (guard, спека simulator-branches).
    """

    def factory(cfg: AppConfig) -> PolicyContour:
        source = LiveRowSource(agg_step_min=cfg.engine.agg_step_min)
        predictor = NetworkPredictor(source, cfg, family=family, artifacts_root=ARTIFACTS_ROOT)
        return PolicyContour(cfg, predictor, source, model=family)

    return factory


BRANCHES: dict[str, BranchSpec] = {
    "a-base": BranchSpec("a-base", "мир без политики"),
    "b-reactive": BranchSpec(
        "b-reactive", "реактивная политика", contour_factory=_reactive_contour
    ),
    "c-mlp": BranchSpec(
        "c-mlp", "ступени по прогнозу MLP", contour_factory=_stair_contour_factory("mlp")
    ),
    "d-lstm": BranchSpec(
        "d-lstm", "ступени по прогнозу LSTM", contour_factory=_stair_contour_factory("lstm")
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
    (каталог создаётся). Политика ветки — по реестру: контурная фабрика строит
    контур до прогона (guard отсутствия артефактов у ``c-mlp``/``d-lstm`` —
    ``FileNotFoundError`` здесь), движок получает ``contour.price`` как
    ``price_fn``, ``bind`` даёт контуру доступ к тарифу мира (design решение 5),
    наблюдатели контура передаются в ``run``; прямая фабрика даёт
    ``price_fn``. Явный аргумент ``price_fn`` имеет приоритет над обеими
    фабриками (спека: параметр функции генерации). ``shadow=True`` (умолчание,
    полный формат ветки) — тень-прогнозы обеих сетей в ``forecasts`` БД ветки с
    границы окна активации: каталоги артефактов ОБЕИХ семейств обязательны у
    любой ветки реестра, их отсутствие — ``FileNotFoundError`` до прогона
    (CLI — exit 2); ``shadow=False`` — тени нет, артефакты не нужны (прямые
    тестовые прогоны математики политик, спека simulator-branches). Guard §4
    M2 до прогона: непустая таблица ``trips`` в целевой БД — ``GUARD_MESSAGE``
    в stderr и возврат ``None`` (CLI переводит в exit 2); неизвестное имя —
    ``ValueError`` с именем ветки в сообщении, прогон не начинается. Успех —
    ``Summary``; прогресс раз в сим-час и сводка формата §4 печатаются теми
    же функциями ``simulator.writers``, что и у CLI ``simulator`` (при
    включённых отказах пассажиров — с пятой строкой ``cancel:``); после
    контурной ветки печатается диагностика срабатываний ``policy_fire_rate``
    (наблюдаемая база сверки честности C/D, спека m6-worlds).
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

    policy = price_fn
    contour: PolicyContour | None = None
    if policy is None:
        if spec.contour_factory is not None:
            # Контур строится до прогона: NetworkPredictor поднимает
            # FileNotFoundError, если каталогов артефактов нет (guard).
            contour = spec.contour_factory(cfg)
        elif spec.price_fn_factory is not None:
            policy = spec.price_fn_factory(cfg)
    shadow_caster: ShadowForecaster | None = None
    if shadow:
        # Guard обоих семейств — у любой ветки реестра, до старта прогона;
        # flush на закрытый бакет — буфер строк тени в forecasts БД ветки.
        shadow_caster = ShadowForecaster(
            cfg, artifacts_root=ARTIFACTS_ROOT, flush=partial(insert_forecasts, conn)
        )
    engine = Engine(
        cfg,
        seed=seed,
        price_fn=policy if contour is None else contour.price,
    )
    if contour is not None:
        contour.bind(engine)  # поздняя привязка: доступ к тарифу мира
    summary = engine.run(
        days,
        SqliteTripWriter(conn),
        progress=print_progress,
        on_minute=_chain(
            contour.on_minute if contour is not None else None,
            shadow_caster.on_minute if shadow_caster is not None else None,
        ),
        on_metrics_row=_chain(
            contour.on_metrics_row if contour is not None else None,
            shadow_caster.on_metrics_row if shadow_caster is not None else None,
        ),
    )
    print_summary(
        summary,
        str(db),
        cancel_enabled=cfg.drivers.dispatch.cancel_prob_per_min > 0.0,
    )
    if contour is not None:
        print(
            f"branch: name={name} policy_fire_rate={contour.fire_rate:.3f}"
            f" fired={contour.fire_count}/{contour.fire_slots}"
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
