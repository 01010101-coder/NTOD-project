## Why

Веха M1 «Каркас» требует хранилища как несущей конструкции: SQLite-схема всех трёх таблиц контракта данных (§6 плана M1) создаётся в M1 сразу, чтобы вехи M2 (симулятор), M3 (ML) и M4 (сервер воспроизведения) не трогали схему. Сейчас пакет `storage/` пуст (проверено в этой сессии: `storage/__init__.py` — заглушка «содержимое добавляется в change m1-storage»), поэтому M2 некуда писать trips и zone_metrics_5min, а DoD вехи («БД создаётся, пишет и читает все таблицы, экспортирует CSV», §1 и §8 плана M1) не закрыт.

## What Changes

- `storage/schema.py`: DDL всех трёх таблиц контракта — `trips`, `zone_metrics_5min`, `forecasts` (точная схема §6 плана M1, включая CHECK-ограничения статусов и PRIMARY KEY составных таблиц) + индексы `trips(created_at)`, `trips(origin_zone)`, `forecasts(model_version)`; функция `init_db` идемпотентна — повторный вызов на существующей БД не падает и не портит данные.
- `storage/db.py`: открытие соединения с прагмами `foreign_keys=ON` и `journal_mode=WAL` (§6 плана M1; WAL — чтобы M4 мог читать во время записи) и контекст-менеджер транзакции (commit при успехе, rollback при исключении).
- `storage/repo.py`: слой доступа — пакетная вставка (`executemany`) и выборки по каждой таблице; для `trips` — выборки по зоне происхождения и по диапазону времени, для `zone_metrics_5min` и `forecasts` — по зоне и интервалу времени.
- `storage/csv_export.py`: выгрузка таблиц в CSV (решение §7.5 плана M1 — экспорт живёт в `storage`, не в симуляторе).
- `storage/demo.py`: демо-скрипт `python -m storage.demo` — создаёт временную БД, записывает по строке в каждую таблицу, читает их обратно, выгружает CSV (способ проверки строки №4 таблицы §4 плана M1).
- `tests/test_storage.py`: повторный init без ошибок; roundtrip всех трёх таблиц; время хранится ISO-текстом «YYYY-MM-DD HH:MM:SS»; CSV содержит те же строки; запуск демо-скрипта.

Не входит: наполнение `forecasts` данными (до M3 таблица пустая, но существует), симулятор (M2), миграции сложнее «создать при отсутствии» (§2 плана M1), код в `simulator/`.

## Capabilities

### New Capabilities

- `data-storage`: требования к слою хранения — SQLite-схема трёх таблиц контракта с индексами, идемпотентная инициализация, соединение с прагмами foreign_keys/WAL и транзакциями, репозиторий с батч-вставкой и выборками по таблице/зоне/времени, CSV-экспорт, демо-скрипт; проверяется тестами `tests/test_storage.py` и `python -m storage.demo`.

### Modified Capabilities

<!-- Нет: openspec/specs содержит build-tooling, common-utilities и config-loading, их требования этот change не меняет. -->

## Impact

- Затронуто: `storage/schema.py`, `storage/db.py`, `storage/repo.py`, `storage/csv_export.py`, `storage/demo.py` (все новые, в существующем пакете `storage` из явного списка `pyproject.toml` — `packages = ["common", "storage", "simulator"]`, `pyproject.toml:32`), `tests/test_storage.py` (новый).
- Не затронуто: `common/config.py`, `common/timeutil.py`, `common/logging_setup.py`, `simulator/`, `pyproject.toml`, Makefile, `config.yaml`, `openspec/config.yaml`.
- Зависимости: новых нет — `sqlite3` входит в стандартную библиотеку; потребители `common.timeutil` (`to_iso`/`from_iso`, единая точка работы с ISO-строками — решение §7.4 плана M1) и `common.config.StorageCfg.db_path` уже реализованы changes `m1-common` и `m1-config` (в архиве openspec).
- Потребители: симулятор M2 (пишет `trips`, `zone_metrics_5min`), ML M3 (читает метрики, пишет `forecasts`), сервер M4 (читает всё во время записи — благодаря WAL); DoD §8 плана M1 «`python -m storage.demo` создаёт БД, пишет/читает строки всех трёх таблиц, выгружает CSV» закрывается этим change.
