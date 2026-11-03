## Why

Веха M1 «Каркас» требует конфигурации как несущей конструкции: скелет конфига §5 плана M1 фиксирует секции и ключи, из которых читают `m1-storage` (`storage.db_path`), симулятор M2 (`city`, `demand`, `drivers`, `weather`, `events`, `pricing`, `engine`) и все последующие вехи. Сейчас в репо конфига нет (проверено в этой сессии: `config.yaml` в корне отсутствует, в `common/` только `timeutil.py` и `logging_setup.py` из `m1-common`), поэтому любой код M2+ не сможет получить параметры города, спроса и водителей.

## What Changes

- `common/config.py`: pydantic v2-модели всех секций скелета §5 плана M1:
  - корневая модель `AppConfig` с полями `seed`, `storage`, `city`, `population`, `demand`, `drivers`, `weather`, `events`, `pricing`, `engine`;
  - у каждой секции — значения по умолчанию (правило «новая секция имеет дефолты, старые конфиги грузятся» из соглашений проекта и §5 плана);
  - `extra="forbid"` во всех моделях (решение §7.3 плана M1) — опечатка в ключе даёт понятную ошибку с путём до ключа, а не молчаливое игнорирование;
  - загрузчик `load_config(path)` — чтение YAML-файла и валидация через pydantic.
- `config.yaml` в корне репо — эталонный конфиг со всеми секциями (значения — стартовые для M2, §3 плана M1).
- `tests/test_config.py`: конфиг целиком из дефолтов; эталонный `config.yaml` валиден; неизвестный ключ → понятная ошибка с путём; `seed` читается.

Не входит: SQLite-схема и репозиторий (`m1-storage`, схема §6 плана), наполнение секций смысловыми значениями калибровки (M2, приложение А плана M2), какой-либо код в `simulator/`.

## Capabilities

### New Capabilities

- `config-loading`: требования к загрузке конфигурации проекта — pydantic-модели секций скелета §5 с дефолтами и `extra="forbid"`, загрузка YAML-файла в корневую модель, эталонный `config.yaml` в репо; проверяется тестами `tests/test_config.py`.

### Modified Capabilities

<!-- Нет: openspec/specs содержит build-tooling и common-utilities, их требования этот change не меняет. -->

## Impact

- Затронуто: `common/config.py` (новый), `config.yaml` в корне (новый), `tests/test_config.py` (новый). Файл кладётся в существующий пакет `common` из явного списка `pyproject.toml` (`packages = ["common", "storage", "simulator"]`, `pyproject.toml:28`) — правка упаковки не нужна.
- Не затронуто: `common/timeutil.py`, `common/logging_setup.py`, `storage/`, `simulator/`, Makefile, `openspec/config.yaml`.
- Зависимости: новых нет — `pydantic>=2` и `PyYAML` уже в `dependencies` (`pyproject.toml:14-15`, установлено `m1-tooling`).
- Потребители: `m1-storage` (путь БД `storage.db_path`), симулятор M2 (все секции §5), DoD вехи §8 («конфиг грузится целиком из дефолтов; эталонный `config.yaml` валиден») закрывается тестами этого change.
