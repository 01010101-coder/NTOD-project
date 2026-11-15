# Proposal: m3-inference

## Why

Веха M3 имеет всё для прогноза, кроме самого прогноза: обученные артефакты
(`m3-train-cli`: каталоги `mlp-v1-*`/`lstm-v1-*` с весами и scaler'ами),
наивы-функции (`m3-baselines`) и метрики (`ml.metrics.build_summary`) — но
таблица `forecasts` пуста, и главный измеримый результат вехи (сводная «наив-1 /
наив-2 / MLP / LSTM», ворота DoD «сети не хуже сезонного наива») не существует.
`m3-inference` — пятый change вехи по плану M3 (§7–§8, `docs/plan-m3-ml.md`):
walk-forward по тест-эпохе заполняет `forecasts` и строит сводную.

## What Changes

- Новый модуль `ml/infer.py`: walk-forward по тест-эпохе (дни после
  `ml.train_days`) — для каждой пары (ts, зона) окно, заканчивающееся на ts
  (сборка `ml-dataset`, только бакеты ≤ ts), прогноз сети, восстановленной из
  артефакта (`weights.pt` несёт arch + state_dict + снимок `ml:`-конфига —
  контракт `m3-train-cli`), денормализация регрессий scaler'ами из каталога
  модели, вероятности дефицита после сигмоиды; две строки `forecasts`
  (горизонты 15/30) на пару.
- Наивы пишутся тем же проходом и тем же набором (ts, зона): честное сравнение
  с первого дня (решение 1 design.md `m3-baselines`).
- **BREAKING (закрыто миграцией)**: первичный ключ `forecasts` расширяется до
  `(ts, zone_id, horizon_min, model_version)` — без этого четыре версии моделей
  не помещаются в таблицу (текущий PK `(ts, zone_id, horizon_min)` из схемы M1
  рассчитан на одну модель). Миграция: новые БД получают новый DDL; для
  существующих — идемпотентный ensure-шаг в `storage/schema.py`, пересоздающий
  таблицу (DROP+CREATE в одной транзакции); таблица старой формы обязана быть
  пустой (инвариант A5 M2: `forecasts = 0` после генерации), непустая — отказ
  с понятной ошибкой.
- `storage/repo.py`: хелпер `delete_forecasts(conn, model_version)` — повторный
  прогон инференса перезаписывает строки своих версий (delete + insert,
  идемпотентность), а не падает на PK и не дублирует.
- CLI `python -m ml infer --config … [--models all|naive|mlp|lstm] [--db …]`
  (приложение Б плана M3): строки `walk-forward: model=… rows=N` на версию,
  затем `summary: artifacts/summary-<дата>.json` и печатная сводная 4×7
  (`ml.metrics.render_table`). Версия сети для семейства — самый свежий каталог
  `artifacts/{arch}-v1-*` (дата в имени сортируется лексически).
- Сводная всего прогона — `artifacts/summary-<дата>.json`: список
  `build_summary`-структур всех записанных версий (полный набор метрик:
  регрессии по таргетам/горизонтам, precision/recall/F1, калибровка) — вход для
  `m3-ml-tab`.
- Тесты `tests/test_ml_infer.py`: анти-лик (мутация будущих строк источника не
  меняет прогноз), границы walk-forward (последние 6 бакетов тест-эпохи без
  пары ts+30 пропускаются), денормализация значений, перезапись повторным
  прогоном, наивы тем же набором пар, CLI smoke; миграция PK — в
  `tests/test_storage.py` (старая форма пустая пересоздаётся, непустая — отказ,
  новая форма — no-op).

## Capabilities

### New Capabilities

- `ml-inference`: walk-forward инференс по тест-эпохе — превращает артефакты
  обучения и наивы в строки контракта `forecasts` (четыре `model_version` на
  одних парах (ts, зона)) и сводную таблицу метрик вехи.

### Modified Capabilities

- `data-storage`: требование «Схема таблиц контракта данных» — PK `forecasts`
  становится `(ts, zone_id, horizon_min, model_version)` (было
  `(ts, zone_id, horizon_min)`), добавляется миграция существующих пустых таблиц
  старой формы.

## Impact

- Код: `ml/infer.py` (новый), `ml/__main__.py` (подкоманда `infer`),
  `storage/schema.py` (DDL PK + ensure-миграция), `storage/repo.py`
  (`delete_forecasts`); тесты `tests/test_ml_infer.py` (новый),
  `tests/test_storage.py` (миграция).
- Не меняются: `ml/dataset.py`, `ml/model.py`, `ml/baselines.py`,
  `ml/metrics.py`, `ml/train.py`, `config.yaml`/`common/config.py` (все
  параметры уже есть), симулятор, веб-сервер.
- Существующие БД (`taxi.db`, `m3_epoch.db`, `accept-*`): таблица `forecasts`
  везде пуста; миграция срабатывает лениво при первом прогоне инференса против
  БД, новые генерации получают новый DDL сразу.
- Новых зависимостей нет; `m3-ml-tab` (следующий change) читает
  `forecasts`/`artifacts/summary-*.json` как есть.
