# Tasks: policy-driver-signal

Решения — design.md. Зависит от `world-elastic-supply` и `ml-planning-forecast`. После каждой группы `make lint && make test` должны быть зелёными; при `channel: price` существующие тесты не правятся.

## 0. Предусловия

- [ ] 0.1 Проверить, что в коде есть `Engine(..., driver_signal_fn=...)`, `MinuteRecord.part_online` и `ml.planning.PlanningPredictor`.
  Verify: импорт и сигнатуры в коротком скрипте; при отсутствии — остановиться и сообщить, какой change не реализован.

## 1. Конфиг

- [ ] 1.1 `common/config.py`: `PolicyCfg.channel: Literal["price", "heatmap", "both"] = "price"`, модель `SignalCfg` (`horizon_min = 60`, `mult = 1.30`, `target_fire_rate = 0.15`, `threshold = {mlp: 0.5, lstm: 0.5}` с ограничениями спеки); подсекция в `config.yaml`.
  Verify: тесты в `tests/test_config.py` — дефолты канала, `channel: radio` отклоняется с путём, порог `1.2` отклоняется.

## 2. Контуры сигнала

- [ ] 2.1 `simulator/policy.py`: `ReactiveSignalContour(cfg, source)` (правило контракта, `signal.mult`) и `PlanningSignalContour(cfg, source, *, family, artifacts_root=None)` (порог по `PlanningPredictor` горизонта `signal.horizon_min`; прогрев → 1.0); оба — подклассы `PolicyContour`.
  Verify: тесты в `tests/test_sim_policy.py` — «реактивный сигнал по факту дефицита», «сигнал планирования по порогу» (стаб-прогнозист 0.62/0.40), «до активации сигнал равен тарифу мира»; построение без артефакта — ошибка до прогона.

## 3. Калибровка порога

- [ ] 3.1 `simulator/policy_calibration.py`: `fit_threshold(probs, target)`; режим CLI `--signal` — вероятности сетей планирования на val-днях, пороги обоих семейств, `artifacts/signal-calibration.json`, точечное обновление `policy.signal.threshold` в `config.yaml`.
  Verify: тесты в `tests/test_sim_policy_calibration.py` — «порог выводит долю к цели» (0.30 → 0.15 ± 1 п.п.), детерминизм повторного запуска, «нет артефактов планирования» (exit 2, файлы не тронуты).

## 4. Генерация веток по каналу

- [ ] 4.1 `simulator/branches.py`: фабрики веток возвращают пару (контур цены | None, контур сигнала | None) по `policy.channel`; `generate_branch` подключает `price_fn`/`driver_signal_fn`, привязывает оба контура, компонует наблюдатели; проверка артефактов планирования для C/D в `heatmap`/`both` до прогона.
  Verify: тесты в `tests/test_sim_branches.py` — существующие сценарии канала `price` без правок; «канал heatmap не меняет цену пассажиру»; «канал heatmap требует артефакт планирования» (exit 2, БД пустая).
- [ ] 4.2 Метаданные ветки: наблюдатель минут считает `callouts_test` и `callout_minutes_test` в окне тест-недели (только при включённом вызове); после успешного прогона пишется `<имя>.meta.json`.
  Verify: тесты «метаданные ветки a-base» и «часы подработчиков в окне»; при guard файл не создаётся.

## 5. Сквозная проверка

- [ ] 5.1 Скрипт во временном каталоге: мир e11 + вызов подработчиков (`lag_min = 60`), обучение сетей планирования на его эпохе, калибровка порога, генерация четырёх веток в канале `heatmap` (seed 42, `shadow=False`).
  Verify: в отчёте apply — необслуженные, перевезённые, `signal_fire_rate`, `callout_minutes_test` четырёх веток; `price_multiplier` заказов веток B/C/D равен тарифу мира.

## 6. Финальные ворота

- [ ] 6.1 `make lint`.
  Verify: exit 0.
- [ ] 6.2 `make test`.
  Verify: exit 0, все тесты зелёные.
