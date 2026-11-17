# Tasks: world-final-config

Решения — design.md. Реализуется последним, после `ml-harm-deficit-label`, `world-demand-shocks`, `world-elastic-supply`, `ml-planning-forecast`, `policy-driver-signal`. После каждой группы `make lint && make test` должны быть зелёными; тесты на `config.yaml` не правятся.

## 0. Предусловия

- [ ] 0.1 Проверить, что пять change'ей-зависимостей реализованы: ключи `ml.deficit_label`, `demand.shocks`, `drivers.callout`, `ml.planning`, `policy.channel`/`policy.signal` загружаются.
  Verify: короткий скрипт `load_config` с этими ключами; при отсутствии — остановиться и назвать change.

## 1. Корень артефактов

- [ ] 1.1 `common/config.py`: `MlCfg.artifacts_root = "artifacts"`; `ml/train.py`, `ml/infer.py`, `ml/__main__.py`, `simulator/policy.py`, `simulator/branches.py`, `simulator/policy_calibration.py`, `app/server.py` берут корень из `cfg.ml.artifacts_root` (константа остаётся дефолтом).
  Verify: тесты «дефолт сохраняет прежние пути» и «отдельный корень изолирует миры»; существующие тесты без правок.

## 2. Вкладка «Сравнение»

- [ ] 2.1 `app/compare.py`: исходы `cancelled`, `unserved`, `deterred`, `saved`, `price_switches`, `callout_hours`, поле `channel` из `meta.json`; дельты только у абсолютных исходов.
  Verify: тесты в `tests/test_web_compare.py` — «ручной расчёт на мини-БД» (с отказами и `price_switches`), «старая сетка без метаданных», «нет дельт у относительных исходов»; существующие сценарии зелёные.
- [ ] 2.2 `app/static/*`: таблица 9 исходов с каналом, правило подсветки (больше — лучше для поездок, спасённых, выручки), прочерк для `null`.
  Verify: ручная проверка в браузере на сетке; сценарии «открытие вкладки», «пустое состояние», «уход до загрузки».

## 3. Конвейер

- [ ] 3.1 `Makefile`: цель `world` с `CONFIG ?= config.final.yaml` (эпоха → `ml train` mlp/lstm → `ml plan-train` mlp/lstm → `policy_calibration` → `policy_calibration --signal` → `simulator.branches` всех веток → `ml plan-eval`); `serve` с `CONFIG ?= config.yaml`.
  Verify: тест или скрипт «сквозной конвейер на малом мире» (временный конфиг, 4 + 2 дня, `max_epochs` 2) — exit 0 и все артефакты; «ошибка шага останавливает конвейер».

## 4. Подбор и фиксация финального мира

- [ ] 4.1 Собрать `notebooks/start_conditions.ipynb` по плану `notebooks/stop.md` («Мысли и план», §1–§21): диагноз старого мира, механизмы, эксперименты e2–e18 (таблицы из `data/start_conditions/scratch/*.pkl`), подбор значений финального мира по критериям design (решение 3) на трёх seed, абляции каналов `price`/`heatmap`/`both`.
  Verify: `jupyter nbconvert --execute` без ошибок; в ноутбуке таблица критериев 1–3 с фактическими значениями выбранного мира.
- [ ] 4.2 `config.final.yaml`: значения из 4.1, все пять механизмов включены, `deficit_label: harm`, `channel: both`, пути под `data/final/` и `artifacts/final/`, комментарий со ссылкой на раздел ноутбука.
  Verify: тест «финальный конфиг валиден и изолирован»; тест «эталонный конфиг не изменился».

## 5. Сборка финального мира

- [ ] 5.1 `make world CONFIG=config.final.yaml` на полном мире (37 суток).
  Verify: exit 0; в отчёте apply — таблица `plan-eval` и ответ `/api/ml/compare` (9 исходов × 4 ветки).
- [ ] 5.2 `make serve CONFIG=config.final.yaml`: вкладки «Город», «ML», «Сравнение» работают на финальном мире.
  Verify: ручная проверка трёх вкладок; скриншот или описание в отчёте apply.

## 6. Финальные ворота

- [ ] 6.1 `make lint`.
  Verify: exit 0.
- [ ] 6.2 `make test`.
  Verify: exit 0, все тесты зелёные, включая эталонные на `config.yaml`.
