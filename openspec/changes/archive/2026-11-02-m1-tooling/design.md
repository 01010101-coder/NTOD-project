## Context

См. proposal.md — Why. Состояние на момент планирования (проверено в этой сессии): в корне проекта нет `pyproject.toml` и `Makefile`; `.gitignore` уже существует и покрывает `data/`, `*.sqlite`, `reports/`, `artifacts/`; `.venv` содержит numpy 2.4.6, pandas 3.1.0, pydantic 2.14.2, PyYAML 6.0.3, pytest 9.2.0, ruff 0.17.3, editable-пакет `taxi-demand-forecast 0.1.0` зарегистрирован, но `import taxi_demand_forecast` падает (код пакета ещё не написан). CI гоняет `make install / lint / test` на Python 3.11. Пакеты верхнего уровня: `common/`, `storage/`, `simulator/` (план M1, §3).

## Goals / Non-Goals

**Goals:**
- Одинаковый путь запуска локально и в CI: `make install && make lint && make test` зелёные из коробки.
- Скелет пакетов, на который последующие changes вехи кладут код без изменения тулинга.

**Non-Goals:**
- Содержимое `common/timeutil.py`, `common/logging_setup.py` — это `m1-common`.
- pydantic-модели и эталонный `config.yaml` (план §5) — это `m1-config`.
- SQLite-схема (план §6), репозиторий и CSV-экспорт — это `m1-storage`.
- CI-файл `.github/workflows/ci.yml` — он уже существует и вызывает цели Makefile; этот change только обязан им соответствовать.

## Decisions

1. **Раскладка без `src/`, пакеты перечислены явно.** Пакеты на верхнем уровне, дистрибутив `taxi-demand-forecast 0.1.0` (план §7.1; editable уже зарегистрирован в `.venv`). Альтернатива `src/`-layout отклонена — против зафиксированного решения. В `pyproject.toml` список пакетов задаётся явно (`common`, `storage`, `simulator`), потому что автообнаружение setuptools в flat-layout падает при нескольких пакетах верхнего уровня (`Multiple top-level packages discovered`), а `tests/` пакетом дистрибутива не является.
2. **Сборочный бэкенд setuptools, метаданные PEP 621.** Ограничения версий — как в плане §4 (строка 1): `numpy`, `pandas`, `pydantic>=2`, `PyYAML` без верхних границ; extras `[dev]` = `pytest`, `ruff`; `requires-python = ">=3.11"` (CI на 3.11). Конфигурация ruff и pytest живёт в `pyproject.toml`, без отдельных ini-файлов — один источник правды.
3. **`make install` создаёт `.venv`, если его нет.** Цель: при отсутствии `.venv` — `python3.11 -m venv .venv` + `pip install -e ".[dev]"`; при существующем — просто установка, идемпотентно. Это закрывает и CI (чистый раннер), и локальную машину (`.venv` уже есть).
4. **ruff по плану §7.6:** `line-length = 100`, правила `E,F,I,W,UP`, форматтер — `ruff format`; `make lint` = `ruff check .` + `ruff format --check .`; `make fmt` = `ruff format .` (+ безопасный autofix check'а). На пустых пакетах-скелетах lint проходит тривиально.
5. **pytest через `testpaths = ["tests"]`; в `tests/` сразу один smoke-тест импорта пакетов.** Без него `make test` на пустом скелете падает: pytest завершается кодом 5 («no tests collected»). Smoke-тест — это заодно механическая проверка требования импортируемости из specs.
6. **`.gitignore` — только верификация.** Файл уже покрывает `data/`, `*.sqlite`, `reports/`, `artifacts/` (строки 5, 7, 21–22 `.gitignore`); задача в tasks.md — прогнать `git check-ignore`, дописывать ничего не нужно, если проверка зелёная.
7. **Совместимость вперёд.** Скелет `config.yaml` (план §5) и схема SQLite (план §6) в этот change не входят и тулинг для них не требуют: конфиг добавляется в `m1-config`, схема — в `m1-storage`, `simulator/` остаётся пустым до M2. Правило «последняя группа задач — `make lint && make test`» (openspec/config.yaml, rules.tasks) соблюдается в tasks.md.

## Risks / Trade-offs

- [pytest без собранных тестов даёт код 5] → smoke-тест в `tests/` добавляется в том же change (решение 5).
- [`python3.11` может отсутствовать в PATH на локальной машине] → ветка создания venv срабатывает только если `.venv` нет; локально `.venv` уже существует, а CI-раннер имеет Python 3.11.
- [numpy 2.x / pandas 3.x — новые мажоры] → на скелете кода зависимости не используются; поведенческие changes (`m1-common` и далее) ловят несовместимости тестами.
- [различие номеров версий между `pip list` (0.1.0) и `pyproject.toml`] → в `pyproject.toml` фиксируется `version = "0.1.0"` — совпадает с уже зарегистрированным editable.

## Migration Plan

Не применим: новые файлы только добавляются, существующие (`.gitignore`, CI) не меняются. Откат — удаление добавленных файлов.

## Open Questions

Нет.
