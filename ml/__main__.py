"""CLI вехи M3 (changes ``m3-train-cli`` и ``m3-inference``):
``python -m ml train|sweep|infer`` (§7 и приложение Б плана M3).

``train`` — обучение ``mlp``/``lstm``/``both`` с записью каталогов
``artifacts/{model_version}/`` (прогресс по эпохам в stderr, итоговые строки
``done: model_version=… artifacts=… val=…`` в stdout). ``sweep`` — свип окна
истории: обучения MLP при нескольких ``window_hours`` без записи каталогов
моделей, критерий — val MAE спроса в физических заказах, результат —
``artifacts/window-sweep.json`` и строка победителя. ``infer`` — walk-forward
по тест-эпохе: строки ``walk-forward: model=… rows=…`` на каждую записанную
версию (наивы и сети из артефактов), затем ``summary: <путь к json>`` и
печатная сводная таблица.

Подкоманды планирования (change ``ml-planning-forecast``): ``plan-train``
обучает одну сеть ``mlp``/``lstm`` и пишет каталог в
``ml.planning.artifacts_root``, ``plan-eval`` печатает таблицу «модель ×
горизонт» (сети и бейзлайны календаря и персистентности). Ожидаемые ошибки
(нет БД эпохи, пустая роль train/val/test, занятый каталог, нет каталога
артефакта, вырожденная доля вреда) — сообщение в stderr и exit 2; успех —
exit 0.
"""

import argparse
import json
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path

from common.config import AppConfig, load_config
from ml.dataset import SqliteRowSource, world_start
from ml.infer import MODELS_CHOICES, run_inference, summarize_run
from ml.planning import (
    evaluate_planning,
    prepare_planning_data,
    render_planning_table,
    save_planning_artifacts,
    train_planning,
)
from ml.train import (
    ARTIFACTS_ROOT,
    model_version_for,
    prepare_training_data,
    save_model_artifacts,
    train_model,
    val_mae_requests,
)
from storage.db import connect

SWEEP_ARCH = "mlp"
"""Архитектура свипа окна: основная (MLP) — буква ТЗ и §5 плана M3."""


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Аргументы CLI: подкоманды train/sweep с дефолтами из конфига."""

    parser = argparse.ArgumentParser(
        prog="ml",
        description="Обучение моделей и свип окна истории (веха M3, change m3-train-cli).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="обучить mlp/lstm/both, записать артефакты")
    train.add_argument("--config", default="config.yaml", help="путь к конфигу YAML")
    train.add_argument(
        "--arch", choices=("mlp", "lstm", "both"), default="both", help="архитектура"
    )
    train.add_argument("--db", default=None, help="путь БД эпохи (по умолчанию из ml: конфига)")

    sweep = subparsers.add_parser("sweep", help="свип окна истории на MLP без каталогов моделей")
    sweep.add_argument("--config", default="config.yaml", help="путь к конфигу YAML")
    sweep.add_argument(
        "--windows",
        default="3,6,12",
        help="окна истории через запятую (часы), по умолчанию 3,6,12",
    )

    infer = subparsers.add_parser(
        "infer", help="walk-forward по тест-эпохе: записи forecasts и сводная метрик"
    )
    infer.add_argument("--config", default="config.yaml", help="путь к конфигу YAML")
    infer.add_argument(
        "--models",
        choices=MODELS_CHOICES,
        default="all",
        help="семейства прогона: наивы и/или сети (по умолчанию все)",
    )
    infer.add_argument("--db", default=None, help="путь БД эпохи (по умолчанию из ml: конфига)")

    plan_train = subparsers.add_parser(
        "plan-train", help="обучить сеть планирования (mlp/lstm) и записать артефакт"
    )
    plan_train.add_argument("--config", default="config.yaml", help="путь к конфигу YAML")
    plan_train.add_argument(
        "--model", choices=("mlp", "lstm"), required=True, help="семейство сети планирования"
    )
    plan_train.add_argument(
        "--db", default=None, help="путь БД эпохи (по умолчанию из ml: конфига)"
    )

    plan_eval = subparsers.add_parser(
        "plan-eval", help="таблица оценки планирования: сети и бейзлайны по горизонтам"
    )
    plan_eval.add_argument("--config", default="config.yaml", help="путь к конфигу YAML")
    plan_eval.add_argument("--db", default=None, help="путь БД эпохи (по умолчанию из ml: конфига)")
    return parser.parse_args(argv)


def _city_zones(cfg: AppConfig) -> list[int]:
    """Идентификаторы районов города из конфига (ёмкость эмбеддинга — max + 1)."""

    return [zone.id for zone in cfg.city.zones]


def _epoch_printer(arch: str, max_epochs: int) -> Callable[[int, float, float, int], None]:
    """Колбэк прогресса обучения: строка формата приложения Б плана M3 в stderr
    после каждой эпохи."""

    def on_epoch(epoch: int, train_total: float, val_total: float, patience_left: int) -> None:
        print(
            f"epoch {epoch}/{max_epochs} arch={arch} loss={train_total:.4f}"
            f" val={val_total:.4f} patience_left={patience_left}",
            file=sys.stderr,
        )

    return on_epoch


def _run_train(args: argparse.Namespace) -> int:
    """Подкоманда train: guard'ы (БД, занятые каталоги — все сразу до обучения)
    → подготовка данных один раз → обучение и запись по архитектурам."""

    cfg = load_config(args.config)
    db_path = args.db if args.db is not None else cfg.ml.db
    if not Path(db_path).is_file():
        print(f"БД эпохи не найдена: {db_path}", file=sys.stderr)
        return 2

    archs: tuple[str, ...] = ("mlp", "lstm") if args.arch == "both" else (args.arch,)
    model_dirs = {arch: ARTIFACTS_ROOT / model_version_for(arch) for arch in archs}
    for model_dir in model_dirs.values():
        if model_dir.exists():
            print(
                f"каталог модели уже существует: {model_dir} — повторное обучение в ту же"
                f" дату требует удалить его",
                file=sys.stderr,
            )
            return 2

    try:
        data = prepare_training_data(
            db_path, cfg.ml, zones=_city_zones(cfg), start=world_start(cfg)
        )
        for arch, model_dir in model_dirs.items():
            result = train_model(
                arch, data, cfg.ml, on_epoch=_epoch_printer(arch, cfg.ml.max_epochs)
            )
            save_model_artifacts(
                model_dir,
                model_version=model_dir.name,
                arch=arch,
                model=result.model,
                data=data,
                result=result,
                ml_cfg=cfg.ml,
            )
            print(
                f"done: model_version={model_dir.name} artifacts={model_dir}"
                f" val={result.best_val['total']:.4f}"
            )
    except (ValueError, FileExistsError, sqlite3.Error) as exc:
        print(f"ошибка обучения: {exc}", file=sys.stderr)
        return 2
    return 0


def _run_sweep(args: argparse.Namespace) -> int:
    """Подкоманда sweep: обучения MLP при каждом окне (подмена ml.window_hours,
    каталоги моделей не пишутся) → artifacts/window-sweep.json + победитель."""

    cfg = load_config(args.config)
    windows = [int(part) for part in args.windows.split(",") if part.strip()]
    if not windows:
        print("пустой список окон: --windows ожидает значения через запятую", file=sys.stderr)
        return 2
    db_path = cfg.ml.db
    if not Path(db_path).is_file():
        print(f"БД эпохи не найдена: {db_path}", file=sys.stderr)
        return 2

    rows: list[dict[str, object]] = []
    try:
        for window in windows:
            ml_cfg = cfg.ml.model_copy(update={"window_hours": window})
            data = prepare_training_data(
                db_path, ml_cfg, zones=_city_zones(cfg), start=world_start(cfg)
            )
            result = train_model(SWEEP_ARCH, data, ml_cfg)
            mae = val_mae_requests(result.model, data)
            rows.append(
                {
                    "window_hours": window,
                    "val_mae_requests": mae,
                    "best_val_total": result.best_val["total"],
                    "best_epoch": result.best_epoch,
                }
            )
            print(
                f"sweep: window_hours={window} val_mae_requests={mae:.4f}"
                f" best_epoch={result.best_epoch}"
            )
    except (ValueError, sqlite3.Error) as exc:
        print(f"ошибка свипа: {exc}", file=sys.stderr)
        return 2

    winner = min(rows, key=lambda row: float(row["val_mae_requests"]))
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_ROOT / "window-sweep.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"windows": rows, "winner": winner["window_hours"]}, fh, ensure_ascii=False, indent=2
        )
    print(
        f"sweep: winner window_hours={winner['window_hours']}"
        f" val_mae_requests={float(winner['val_mae_requests']):.4f}"
    )
    return 0


def _run_infer(args: argparse.Namespace) -> int:
    """Подкоманда infer (change ``m3-inference``): guard БД → прогон версий
    (guard'ы артефактов и запись внутри) → сводная json + печатная таблица.
    Формат вывода — приложение Б плана M3."""

    cfg = load_config(args.config)
    db_path = args.db if args.db is not None else cfg.ml.db
    if not Path(db_path).is_file():
        print(f"БД эпохи не найдена: {db_path}", file=sys.stderr)
        return 2

    try:
        source = SqliteRowSource(db_path)
        conn = connect(str(db_path))
        try:
            versions = run_inference(
                source, conn, cfg=cfg, models=args.models, artifacts_root=ARTIFACTS_ROOT
            )
            summary_path, table = summarize_run(
                conn,
                source,
                versions,
                label_rule=cfg.ml.deficit_label,
                artifacts_root=ARTIFACTS_ROOT,
            )
        finally:
            conn.close()
    except (ValueError, RuntimeError, FileNotFoundError, sqlite3.Error) as exc:
        print(f"ошибка инференса: {exc}", file=sys.stderr)
        return 2

    for version in versions:
        print(f"walk-forward: model={version.model_version} rows={version.rows}")
    print(f"summary: {summary_path}")
    print(table)
    return 0


def _plan_epoch_printer(model: str, max_epochs: int) -> Callable[[int, float, float, int], None]:
    """Колбэк прогресса обучения планирования: строка эпохи в stderr."""

    def on_epoch(epoch: int, train_bce: float, val_bce: float, patience_left: int) -> None:
        print(
            f"epoch {epoch}/{max_epochs} model={model} bce={train_bce:.4f}"
            f" val={val_bce:.4f} patience_left={patience_left}",
            file=sys.stderr,
        )

    return on_epoch


def _run_plan_train(args: argparse.Namespace) -> int:
    """Подкоманда plan-train: guard'ы (БД, занятый каталог) → данные → обучение
    → артефакт ``artifacts_root/{model}-v1-<дата>/``."""

    cfg = load_config(args.config)
    db_path = args.db if args.db is not None else cfg.ml.db
    if not Path(db_path).is_file():
        print(f"БД эпохи не найдена: {db_path}", file=sys.stderr)
        return 2

    model_dir = Path(cfg.ml.planning.artifacts_root) / model_version_for(args.model)
    if model_dir.exists():
        print(
            f"каталог артефакта уже существует: {model_dir} — повторное обучение в ту же"
            f" дату требует удалить его",
            file=sys.stderr,
        )
        return 2

    try:
        data = prepare_planning_data(
            db_path, cfg.ml, zones=_city_zones(cfg), start=world_start(cfg)
        )
        result = train_planning(
            args.model,
            data,
            cfg.ml,
            on_epoch=_plan_epoch_printer(args.model, cfg.ml.max_epochs),
        )
        save_planning_artifacts(
            model_dir,
            model_version=model_dir.name,
            family=args.model,
            model=result.model,
            data=data,
            result=result,
            ml_cfg=cfg.ml,
        )
    except (ValueError, FileExistsError, sqlite3.Error) as exc:
        print(f"ошибка обучения планирования: {exc}", file=sys.stderr)
        return 2
    print(
        f"plan-train: model_version={model_dir.name} artifacts={model_dir}"
        f" val={result.best_val['bce']:.4f}"
    )
    return 0


def _run_plan_eval(args: argparse.Namespace) -> int:
    """Подкоманда plan-eval: таблица «модель × горизонт» по свежайшим артефактам
    обоих семейств; нет эпохи или артефакта — exit 2."""

    cfg = load_config(args.config)
    db_path = args.db if args.db is not None else cfg.ml.db
    if not Path(db_path).is_file():
        print(f"БД эпохи не найдена: {db_path}", file=sys.stderr)
        return 2

    try:
        rows = evaluate_planning(
            SqliteRowSource(db_path),
            cfg.ml,
            zones=_city_zones(cfg),
            start=world_start(cfg),
            artifacts_root=cfg.ml.planning.artifacts_root,
        )
    except (ValueError, FileNotFoundError, sqlite3.Error) as exc:
        print(f"ошибка оценки планирования: {exc}", file=sys.stderr)
        return 2
    print(render_planning_table(rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI: train → _run_train, sweep → _run_sweep, infer → _run_infer,
    plan-train → _run_plan_train, plan-eval → _run_plan_eval; exit 0/2."""

    args = _parse_args(argv)
    if args.command == "train":
        return _run_train(args)
    if args.command == "sweep":
        return _run_sweep(args)
    if args.command == "plan-train":
        return _run_plan_train(args)
    if args.command == "plan-eval":
        return _run_plan_eval(args)
    return _run_infer(args)


if __name__ == "__main__":
    raise SystemExit(main())
