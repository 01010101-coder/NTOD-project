## Why

Веха M1 «Каркас» начинается с запускаемого скелета: в репозитории `skvoznoi_project` ещё нет `pyproject.toml` и `Makefile` (проверено: файлов нет в корне), а CI `.github/workflows/ci.yml` уже гоняет `make install / lint / test` на Python 3.11. Без единой тулинговой основы последующие changes вехи (`m1-common`, `m1-config`, `m1-storage`) некуда класть код и нечем проверять.

## What Changes

- `pyproject.toml`: пакет `taxi-demand-forecast 0.1.0`; зависимости `numpy`, `pandas`, `pydantic>=2`, `PyYAML`; extras `[dev]`: `pytest`, `ruff`; конфигурация ruff (line-length 100, правила `E,F,I,W,UP`, format как форматтер) и pytest в том же файле.
- `Makefile` с целями `install`, `test`, `lint`, `fmt`; `make lint` = `ruff check` + `ruff format --check`.
- Пакеты-скелеты в корне (без `src/`): `common/`, `storage/`, `simulator/` (по `__init__.py`), `tests/` (с рабочим пустым тестом, чтобы `make test` был зелёным).
- Проверка `.gitignore`: убедиться, что покрывает `data/`, `artifacts/`, `reports/` (файл уже существует — задача верификации, при пробелах — дополнить).

Симулятор, конфиг-модели и слой БД в этом change не создаются: `simulator/` остаётся пустым пакетом (M2), конфиг — в `m1-config`, схема SQLite — в `m1-storage`.

## Capabilities

### New Capabilities

- `build-tooling`: требования к воспроизводимой среде разработки — установка проекта и dev-зависимостей одной командой, цели Makefile (install/test/lint/fmt), импортируемость пакетов-скелетов, lint-чистота пустых пакетов, игнорирование генерируемых данных в git.

### Modified Capabilities

<!-- Нет: openspec/specs пуст, существующих capabilities не меняется. -->

## Impact

- Затронуто: `pyproject.toml` (новый), `Makefile` (новый), `common/`, `storage/`, `simulator/`, `tests/` (новые пакеты), `.gitignore` (только проверка).
- Не затронуто: код `ml/`, `app/` (не создаются на M1), `openspec/config.yaml`.
- Окружение: editable-пакет `taxi-demand-forecast 0.1.0` уже зарегистрирован в `.venv`; deps (numpy 2.4.6, pandas 3.1.0, pydantic 2.14.2, PyYAML 6.0.3, pytest 9.2.0, ruff 0.17.3) уже установлены — `make install` обязан быть идемпотентным.
- CI: цели Makefile должны работать на Python 3.11 без дополнительных системных зависимостей.
