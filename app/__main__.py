"""CLI запуска сервера воспроизведения: ``python -m app`` (change ``replay-server``, задача 4.4).

Аргументы: ``--config`` (по умолчанию ``config.yaml``), ``--seed``/``--days``
(дефолты из конфига — ``seed`` и ``app.days``, флаги переопределяют),
``--host``/``--port`` (дефолты из секции ``app``). Собирает приложение
``app.server.create_app`` (фоновый прогон лога стартует в нём немедленно) и
поднимает uvicorn. Логики мира в модуле нет — только сборка аргументов и запуск.
"""

import argparse

import uvicorn

from app.server import create_app
from common.config import AppConfig, load_config


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Аргументы CLI: всё опционально, дефолты берутся из конфигурации."""

    parser = argparse.ArgumentParser(
        prog="app",
        description="Сервер воспроизведения города такси (change replay-server).",
    )
    parser.add_argument("--config", default="config.yaml", help="путь к конфигу YAML")
    parser.add_argument("--seed", type=int, default=None, help="seed (по умолчанию из конфига)")
    parser.add_argument(
        "--days", type=float, default=None, help="дни прогона лога (по умолчанию app.days)"
    )
    parser.add_argument("--host", default=None, help="адрес (по умолчанию app.host)")
    parser.add_argument("--port", type=int, default=None, help="порт (по умолчанию app.port)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Собрать приложение по флагам/конфигу и запустить uvicorn (блокирующе)."""

    args = _parse_args(argv)
    cfg: AppConfig = load_config(args.config)
    app = create_app(cfg, seed=args.seed, days=args.days)
    host = cfg.app.host if args.host is None else args.host
    port = cfg.app.port if args.port is None else args.port
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
