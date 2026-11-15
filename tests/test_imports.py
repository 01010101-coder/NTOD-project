"""Smoke-тест вехи M1: пакеты-скелеты импортируются.

Защищает от кода выхода pytest 5 («no tests collected») на пустом скелете
и механически проверяет требование импортируемости пакетов
(change m1-tooling, capability build-tooling).
"""

import app
import common
import ml.model
import simulator
import storage


def test_imports() -> None:
    """Пакеты верхнего уровня app, common, storage, simulator импортируются без ошибок."""
    assert app is not None
    assert common is not None
    assert storage is not None
    assert simulator is not None


def test_ml_model_imports() -> None:
    """Модуль моделей m3-model импортируется без ошибок (torch доступен в .venv)."""
    assert ml.model is not None
