## Why

Скелет вехи M1 из `m1-tooling` создал пустой пакет `common/` (проверено в этой сессии: в `common/` только `__init__.py`, в `tests/` только `test_imports.py`), но без него последующие changes вехи не могут стартовать: `m1-config` грузит `config.yaml` (скелет §5 плана M1), `m1-storage` пишет время в SQLite как ISO-текст (схема §6), симулятор M2 агрегирует метрики по 5-минутным интервалам и прибавляет минуты ожидания. По решению §7.4 плана M1 `timeutil` — единственная точка работы с ISO-строками, и её нужно зафиксировать до конфига и хранилища.

## What Changes

- `common/timeutil.py`: работа с наивным `datetime` и минутной точностью (решение §7.2):
  - `to_iso(dt)` / `from_iso(s)` — текст формата «YYYY-MM-DD HH:MM:SS», секунды всегда 00; в БД (`trips.created_at`, `zone_metrics_5min.ts`, `forecasts.ts` схемы §6) время хранится именно в этом формате;
  - `add_minutes(dt, n)` — прибавление минут без выхода за минутную точность (ожидание водителя `wait_limit_min`, горизонты прогноза 15/30 из скелета конфига §5);
  - `bucket5(ts)` → метка начала 5-минутного интервала (шаг агрегации `engine.agg_step_min = 5` из §5; `zone_metrics_5min.ts` — «начало 5-минутного интервала» из §6).
- `common/logging_setup.py`: базовая настройка логирования проекта (единый формат, настраиваемый уровень).
- `tests/test_timeutil.py`: roundtrip ISO; бакеты на границах (…:00–:04 → :00; :05–:09 → :05); переход часа/полуночи; `add_minutes` через границы часа/суток.

Не входит: pydantic-модели и эталонный `config.yaml` (`m1-config`), SQLite-схема и репозиторий (`m1-storage`), какой-либо код в `simulator/` (M2).

## Capabilities

### New Capabilities

- `common-utilities`: требования к общей библиотеке проекта — единственная точка преобразования наивного `datetime` ↔ ISO-текст «YYYY-MM-DD HH:MM:SS» с секундами 00, арифметика минут, привязка к 5-минутным бакетам, базовая настройка логирования; проверяется тестами `tests/test_timeutil.py`.

### Modified Capabilities

<!-- Нет: openspec/specs содержит только build-tooling, его требования этот change не меняет. -->

## Impact

- Затронуто: `common/timeutil.py` (новый), `common/logging_setup.py` (новый), `tests/test_timeutil.py` (новый). `pyproject.toml` и Makefile не меняются — файлы кладутся в существующий пакет `common` из явного списка пакетов `m1-tooling`.
- Не затронуто: `storage/`, `simulator/`, `openspec/config.yaml`.
- Зависимости: новых зависимостей нет — только стандартная библиотека (`datetime`, `logging`).
- Потребители: `m1-config` (скелет конфига §5), `m1-storage` (ISO-текст в схеме §6), симулятор M2 (агрегация и ожидания). DoD вехи §8 («последний прогон — `make lint && make test`») выполняется финальной группой задач tasks.md.
