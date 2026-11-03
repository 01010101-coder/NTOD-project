"""Smoke-тест вехи M1: пакеты-скелеты импортируются.

Защищает от кода выхода pytest 5 («no tests collected») на пустом скелете
и механически проверяет требование импортируемости пакетов
(change m1-tooling, capability build-tooling).
"""

import common
import simulator
import storage


def test_imports() -> None:
    """Пакеты верхнего уровня common, storage, simulator импортируются без ошибок."""
    assert common is not None
    assert storage is not None
    assert simulator is not None
