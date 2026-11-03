## 1. pyproject.toml: метаданные и конфигурация инструментов

- [x] 1.1 Создать `pyproject.toml` (PEP 621, бэкенд setuptools): имя `taxi-demand-forecast`, версия `0.1.0`, `requires-python = ">=3.11"`; зависимости `numpy`, `pandas`, `pydantic>=2`, `PyYAML`; extras `[dev]` = `pytest`, `ruff`; явный список пакетов `["common", "storage", "simulator"]` (без автообнаружения flat-layout, `tests/` не пакуется). Проверка: `.venv/bin/python -m pip install -e ".[dev]"` завершается кодом 0; `.venv/bin/python -m pip show taxi-demand-forecast` показывает `0.1.0`.
- [x] 1.2 Добавить в `pyproject.toml` конфигурацию ruff (`line-length = 100`, правила `E,F,I,W,UP`, секция format) и pytest (`testpaths = ["tests"]`). Проверка: `.venv/bin/ruff check .` и `.venv/bin/ruff format --check .` на пустом скелете завершаются кодом 0; `.venv/bin/python -m pytest --collect-only -q` не падает и показывает `tests` как каталог сбора.

## 2. Makefile: цели install / test / lint / fmt

- [x] 2.1 Создать `Makefile` с целями: `install` — при отсутствии `.venv` создать (`python3.11 -m venv .venv`) и выполнить `pip install -e ".[dev]"`, при существующем — только установка; `test` — `.venv/bin/python -m pytest`; `lint` — `.venv/bin/ruff check .` + `.venv/bin/ruff format --check .`; `fmt` — `.venv/bin/ruff format .` + `ruff check --fix`. Проверка: `make install` в существующем `.venv` завершается кодом 0 (идемпотентность); `make -n lint test fmt` показывает ожидаемые команды.

## 3. Пакеты-скелеты и smoke-тест

- [x] 3.1 Создать пакеты верхнего уровня `common/__init__.py`, `storage/__init__.py`, `simulator/__init__.py` с docstring-заглушками на русском; `simulator/` остаётся пустым до M2, файлы `timeutil.py`, `schema.py` и т.п. в этот change не входят (`m1-common`, `m1-storage`). Проверка: `.venv/bin/python -c "import common, storage, simulator"` завершается кодом 0.
- [x] 3.2 Создать `tests/test_imports.py` — smoke-тест, импортирующий `common`, `storage`, `simulator` (защита от кода выхода pytest 5 «no tests collected» на пустом скелете). Проверка: `.venv/bin/python -m pytest -q` — 1 собран, 1 прошёл, код 0.

## 4. Проверка .gitignore

- [x] 4.1 Проверить, что `.gitignore` покрывает `data/`, `reports/`, `artifacts/` и базы SQLite (план M1 §3); при пробелах — дополнить, при полном покрытии не менять. Проверка: `git check-ignore data/taxi.db reports/plot.png artifacts/model.pkl` возвращает все три пути; после создания `data/` команда `git status --porcelain` не содержит `data/`.

## 5. Финальная проверка

- [x] 5.1 Прогнать полный цикл вехи на скелете: `make install`. Проверка: код 0, пакет и зависимости установлены (`pip show` находит `taxi-demand-forecast`).
- [x] 5.2 Прогнать `make lint && make test` на пустых пакетах. Проверка: обе цели код 0; lint без замечаний, pytest — smoke-тест зелёный.
