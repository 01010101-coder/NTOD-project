"""Базовая настройка логирования проекта: единый формат, настраиваемый уровень.

Все входные точки проекта (CLI/демо/симуляция) вызывают :func:`setup_logging`;
формат записей — «время | уровень | имя логгера | сообщение», поток — stderr.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Настроить корневой логгер проекта: единый формат записей, вывод в stderr.

    ``logging.basicConfig(..., force=True)`` снимает прежние обработчики
    корневого логгера, поэтому повторный вызов не плодит дублирующие записи,
    а заменяет предыдущую настройку (решение design.md 6). Вызывается только
    на входных точках проекта, где сторонних обработчиков ещё нет.
    """
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stderr,
        level=level,
        force=True,
    )
