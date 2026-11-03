# План M1 «Каркас»

- Статус: черновик для согласования.
- Последовательность вех: **M1 Каркас → M2 Движок → M3 ML → M4 Сервер воспроизведения → M5 Интерфейс → M6 Вмешательство**. Вехи M3+ описываются отдельными планами.
- Источники: `docs/сквозной проект спецификация (1).pdf` (ТЗ), контракт данных и соглашения — `openspec/config.yaml`, решения обсуждения проекта (ноябрь 2026).
- Документ самодостаточен: исполнитель не обязан читать обсуждение, только ТЗ и `openspec/config.yaml`.

## 1. Цель вехи

Запускаемый скелет проекта, на котором стоят все последующие вехи:

- структура пакетов и тулинг (pyproject, Makefile, ruff, pytest, CI зелёный);
- конфигурация: pydantic v2-модели всех секций + эталонный `config.yaml` с дефолтами;
- хранилище: SQLite-схема **всех трёх таблиц контракта**, слой доступа, экспорт CSV;
- общее: типы времени (наивный datetime, минутная точность), логирование.

Логики симуляции в M1 нет. Критерий «веха закрыта»: `make install && make lint && make test` зелёные, библиотека импортируется, БД создаётся, пишет и читает все таблицы, экспортирует CSV.

## 2. Границы

**Входит:** перечисленное в цели.

**Не входит:** симулятор (M2), ML (M3), сервер и UI (M4–M5), какие-либо миграции сложнее «создать при отсутствии».

**Известные расхождения с `openspec/config.yaml`:** там `app/` помечен как «Streamlit-дашборд (позже)», а в стеке упомянуты streamlit/plotly. Это устарело: интерфейсной частью будет FastAPI + WebSocket + canvas-фронтенд без сборки. Текст `openspec/config.yaml` актуализируем в вехе M4; на M1/M2 это не влияет, папка `app/` не создаётся.

**Существующее окружение:** `.venv` в корне уже содержит numpy, pandas, pydantic v2, PyYAML, pytest, ruff, matplotlib (плюс torch/scikit-learn на будущее); зарегистрирован editable-пакет `taxi-demand-forecast 0.1.0`. CI (`.github/workflows/ci.yml`) уже гоняет `make install / lint / test` на Python 3.11 — план обязан это выполнять.

## 3. Структура репозитория после M1

```
skvoznoi_project/
├── pyproject.toml            # пакет taxi-demand-forecast 0.1.0, deps, extras [dev], ruff, pytest
├── Makefile                  # install / test / lint / fmt
├── config.yaml               # эталонный конфиг (все секции, значения — для M2)
├── .gitignore                # уже существует; проверить, что покрывает data/, artifacts/, reports/
├── common/
│   ├── __init__.py
│   ├── timeutil.py           # наивный datetime: ISO-текст, минутная точность, 5-мин бакеты
│   └── logging_setup.py      # базовая настройка логирования
├── storage/
│   ├── __init__.py
│   ├── schema.py             # DDL трёх таблиц + индексы, идемпотентный init
│   ├── db.py                 # соединение, прагмы, контекст транзакции
│   ├── repo.py               # вставка/чтение trips, zone_metrics_5min, forecasts
│   └── csv_export.py         # выгрузка таблиц в CSV
├── simulator/
│   └── __init__.py           # пустой пакет, заполняется в M2
├── tests/
│   ├── test_config.py
│   ├── test_timeutil.py
│   └── test_storage.py
└── data/                     # gitignored: базы и экспорты
```

## 4. Декомпозиция на changes

Формат для конвейера OpenSpec: один change = одна строка таблицы. Правила `openspec/config.yaml` соблюдены: каждая задача имеет способ проверки; последний шаг — `make lint && make test`.

| # | Change | Содержание | Как проверить |
|---|--------|-----------|---------------|
| 1 | `m1-tooling` | pyproject (deps: numpy, pandas, pydantic≥2, PyYAML; extras `[dev]`: pytest, ruff; конфиг ruff и pytest), Makefile, пакеты-скелеты, проверка .gitignore | `make install`; `python -c "import common, storage, simulator"`; `make lint` на пустых пакетах |
| 2 | `m1-common` | `timeutil`: `to_iso`/`from_iso` («YYYY-MM-DD HH:MM:SS», секунд всегда 00), `add_minutes`, `bucket5(ts)` → метка начала 5-минутного интервала; `logging_setup` | тесты: roundtrip ISO; бакеты на границах (…:00–:04 → :00; :05–:09 → :05); переход часа/полуночи |
| 3 | `m1-config` | pydantic-модели секций (скелет §5), у каждой секции дефолты; загрузка `config.yaml`; `extra="forbid"`; эталонный `config.yaml` в репо | тесты: конфиг целиком из дефолтов; эталонный файл валиден; неизвестный ключ → понятная ошибка с путём; `seed` читается |
| 4 | `m1-storage` | схема §6, индексы, идемпотентный init; репозиторий (пакетная вставка, выборки по таблице/зоне/времени); CSV-экспорт | тесты: повторный init без ошибок; roundtrip всех трёх таблиц; время хранится ISO-текстом; CSV содержит те же строки; демо-скрипт `python -m storage.demo` (создать временную БД, записать по строке в каждую таблицу, прочитать, выгрузить CSV) |

Порядок строгий: 1 → 2 → 3 → 4 (3 и 4 зависят только от 1–2, могут идти параллельно).

## 5. Скелет config.yaml (создаётся в `m1-config`)

Секции и ключи фиксируются в M1; смысловые значения (профили, граф, коэффициенты) наполняются и уточняются в M2 (см. `docs/plan-m2-engine.md`, приложение А). Дефолты обязательны для каждой секции — старые конфиги обязаны грузиться после добавления новых секций.

```yaml
seed: 42
storage:
  db_path: data/taxi.db
city:            # зоны (архетип, привлекательность, координаты карты) и дорожный граф
  zones: [...]
  roads: [...]
population:      # масштаб интенсивности спроса
  scale: 1.0
demand:          # суточные профили по архетипам, затухание выбора назначения
  profiles: {...}
  dest_decay_min: 15
drivers:         # размер парка, лимит ожидания назначения
  count: 200
  wait_limit_min: 6
weather:         # шанс дождя в день, параметры эпизода, множители
  {...}
events:          # «пятничный вечер»: дни, часы, архетипы, множитель
  friday_evening: {...}
pricing:         # эластичность и обучающие сурcharge-инъекции
  elasticity_alpha: 1.0
  surge_injection: {...}
engine:          # шаг агрегации
  agg_step_min: 5
```

## 6. Схема SQLite (точная, из контракта `openspec/config.yaml`)

Все три таблицы создаются сразу, чтобы M2+ не трогал схему. `forecasts` до M3 остаётся пустой.

```
trips(
  trip_id INTEGER PRIMARY KEY,      -- сквозной, растёт в порядке создания
  created_at TEXT NOT NULL,          -- ISO "YYYY-MM-DD HH:MM:SS", секунды 00
  origin_zone INTEGER NOT NULL,
  dest_zone INTEGER NOT NULL,
  assigned_at TEXT,                  -- NULL, если заказ не назначен
  pickup_at TEXT,
  dropoff_at TEXT,
  status TEXT NOT NULL CHECK (status IN ('completed','cancelled','unfulfilled')),
  price_multiplier REAL NOT NULL     -- множитель зоны происхождения на момент created_at
)

zone_metrics_5min(
  ts TEXT NOT NULL,                  -- начало 5-минутного интервала
  zone_id INTEGER NOT NULL,
  requests INTEGER NOT NULL,
  completed INTEGER NOT NULL,
  unfulfilled INTEGER NOT NULL,
  free_drivers INTEGER NOT NULL,
  busy_drivers INTEGER NOT NULL,
  avg_wait_min REAL,                 -- NULL, если в интервале никого не назначили
  supply_demand_ratio REAL NOT NULL, -- free_drivers / max(requests, 1)
  is_rain INTEGER NOT NULL,          -- 0/1
  event TEXT NOT NULL,               -- 'friday_evening' | 'surge_injection' | ''
  price_multiplier REAL NOT NULL,    -- среднее по времени значение в интервале
  PRIMARY KEY (ts, zone_id)
)

forecasts(
  ts TEXT NOT NULL,
  zone_id INTEGER NOT NULL,
  horizon_min INTEGER NOT NULL,      -- 15 или 30
  pred_requests REAL NOT NULL,
  pred_free_drivers REAL NOT NULL,
  deficit_prob REAL NOT NULL,
  model_version TEXT NOT NULL,
  PRIMARY KEY (ts, zone_id, horizon_min)
)
```

Индексы: `trips(created_at)`, `trips(origin_zone)`, `forecasts(model_version)`.
Праги: `foreign_keys=ON`, `journal_mode=WAL` (дешёвый побочный выигрыш для M4 — читать во время записи).

## 7. Зафиксированные решения

1. Пакеты на верхнем уровне, без `src/`; имя дистрибутива `taxi-demand-forecast` (editable уже зарегистрирован в `.venv`).
2. Наивный `datetime`, точность до минуты, в БД — ISO-текст. Никаких таймзон.
3. `extra="forbid"` в pydantic-моделях: ловит опечатки в конфиге; правило «новая секция имеет дефолты» сохраняется.
4. Время в `timeutil` — единственная точка работы с ISO-строками; остальной код живёт в `datetime`.
5. CSV-экспорт живёт в `storage`, не в симуляторе: экспорт — свойство данных, а не движка.
6. ruff: line-length 100, правила `E,F,I,W,UP`, `ruff format` как форматтер; `make lint` = check + format-check.
7. `reports/` и `artifacts/` уже в .gitignore — туда же попадут графики калибровки (M2) и артефакты модели (M3).

## 8. Критерии готовности (DoD)

- [ ] `make install`, `make lint`, `make test` зелёные локально и в CI.
- [ ] `python -m storage.demo` создаёт БД, пишет/читает строки всех трёх таблиц, выгружает CSV.
- [ ] Конфиг грузится целиком из дефолтов; эталонный `config.yaml` валиден.
- [ ] Все тесты §4 написаны и зелёные; последний прогон — `make lint && make test`.
