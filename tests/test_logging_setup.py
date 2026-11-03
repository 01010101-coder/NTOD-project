"""Тесты common/logging_setup.py: базовая настройка логирования.

Покрывают сценарии «Настройка и запись» и «Повторная настройка без дублей»
из specs change'а m1-common (capability common-utilities). Проверка через
capsys: setup_logging внутри теста привязывает handler к перехваченному stderr.
"""

import logging

from common.logging_setup import setup_logging


def test_setup_and_record_written_to_stderr(capsys) -> None:
    """Сценарий «Настройка и запись»: одна строка с временем, уровнем, именем и текстом."""
    setup_logging()
    logging.getLogger("demo").info("проверка")
    line = capsys.readouterr().err.strip()
    # Формат «время | уровень | имя логгера | сообщение» — ровно четыре поля.
    fields = line.split(" | ")
    assert len(fields) == 4
    # Время присутствует (первое поле — asctime с датой).
    assert fields[0][:4].isdigit()
    # Уровень, имя логгера и текст сообщения на месте.
    assert fields[1] == "INFO"
    assert fields[2] == "demo"
    assert fields[3] == "проверка"


def test_repeated_setup_does_not_duplicate(capsys) -> None:
    """Сценарий «Повторная настройка без дублей»: запись выводится ровно один раз."""
    setup_logging()
    setup_logging()
    logging.getLogger("demo").info("проверка")
    err = capsys.readouterr().err
    assert err.count("проверка") == 1
    assert len(err.strip().splitlines()) == 1
