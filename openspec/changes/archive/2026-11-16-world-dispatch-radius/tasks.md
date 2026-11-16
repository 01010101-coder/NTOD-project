# Tasks: world-dispatch-radius

Решения — design.md. После каждой группы `make lint && make test` должны быть зелёными.

## 1. Конфиг drivers.dispatch

- [x] 1.1 `common/config.py`: модель `DispatchCfg` (`max_pickup_min: int ≥ 1 | None = None`, `cancel_grace_min: int ≥ 0 = 5`, `cancel_prob_per_min: float ≥ 0 = 0.0`, `cancel_max_prob: float в [0, 1] = 0.9`) и поле `DriversCfg.dispatch` с дефолтом.
  Verify: новые тесты в `tests/test_config.py` — старый конфиг без подсекции, частичная подсекция, неизвестный ключ `drivers.dispatch.max_pickup` с путём, отклонение `max_pickup_min: 0`, `cancel_prob_per_min: -0.1`, `cancel_max_prob: 1.5`.
- [x] 1.2 Подсекция `dispatch:` в `config.yaml` со значениями, равными дефолтам, и комментариями.
  Verify: `load_config('config.yaml').drivers.dispatch == DispatchCfg()`; `test_reference_run_bitwise_compatibility` зелёный.

## 2. Радиус поиска

- [x] 2.1 `simulator/engine.py`: `_nearest_free_id` принимает необязательный предел подачи (с учётом `k_travel`). В шаге 3 заказ вне радиуса остаётся в `still_waiting`, обход очереди продолжается.
  Verify: тесты в `tests/test_sim_loop.py` — «свободная машина за радиусом не назначается» (зона 4 против зоны 5, заказ истекает `unfulfilled`), «дождь сужает радиус» (путь 9, `travel_mult` 1.4, R = 10), «заказ вне радиуса не блокирует очередь».
- [x] 2.2 Тест-инвариант: прогон 1 сутки seed 42 с `max_pickup_min = 10`, у всех `completed` подача ≤ 10 минут.
  Verify: тест зелёный.

## 3. Отказ пассажира

- [x] 3.1 `Engine.__init__`: третий генератор `default_rng(SeedSequence(seed).spawn(3)[2])`; первые два потока создаются как раньше.
  Verify: `test_reference_run_bitwise_compatibility` зелёный без правки хэшей.
- [x] 3.2 Шаг 3: после выбора водителя посчитать `p`, при `p > 0` разыграть отказ. При отказе — строка `cancelled` с NULL-временами, `buffer.add_expiry(origin)`, водитель остаётся в зеркале, счётчик `cancelled` растёт. Назначение, `avg_wait` и `in_flight` не трогаются.
  Verify: тесты — «подача в пределах терпения не разыгрывается», вероятность 0.3 и 0.9 через вынесенную чистую функцию `_cancel_prob`, «отказ освобождает водителя для следующего заказа» (микросценарий с `cancel_max_prob = 1.0`, `cancel_prob_per_min = 1.0`).
- [x] 3.3 Тесты потоков: прогоны одного seed с `cancel_prob_per_min` 0.0 и 0.1 дают одинаковые поминутные `is_rain`, тариф среды и последовательность спавна; два прогона с 0.1 побайтово равны.
  Verify: тесты зелёные.
- [x] 3.4 Агрегат: тест «отказы входят в unfulfilled» — в бакете 1 истечение и 2 отказа дают `unfulfilled = 3` и дефицит.
  Verify: тест зелёный.

## 4. Summary и CLI

- [x] 4.1 `Summary`: поля `cancelled`, `cancelled_share`; `trips_total = completed + unfulfilled + cancelled`.
  Verify: тест сверяет `Summary` со счётом статусов в БД на прогоне с отказами; при дефолтах `cancelled == 0`.
- [x] 4.2 `simulator/__main__.py`: строка `cancel: cancelled=<K> cancelled_share=<Q.Q>%` перед `wall:` только при `cancel_prob_per_min > 0`.
  Verify: `tests/test_sim_cli.py` — при дефолтах 4 строки по прежнему регексу, с отказами 5 строк и формат строки `cancel:`. `make accept-m2` проходит без правки скрипта.

## 5. Поток наблюдения и воспроизведение

- [x] 5.1 `simulator/live.py`: `CancelEvent(internal_id, origin, pickup_min)` и поле `MinuteRecord.cancellations = ()` последним. Движок наполняет его в шаге 3 при заданном `on_minute`.
  Verify: тесты в `tests/test_sim_live.py` — цепочка spawn → cancellation для каждой строки `cancelled`, охват `cancellations == Summary.cancelled`, сумма `free_by_zone` не меняется от отказа.
- [x] 5.2 `app/replay.py` и `app/server.py`: сериализация `cancellations`; `pending_orders` исключает отменённые заказы.
  Verify: тест в `tests/test_web_replay.py` — запись минуты отказа содержит элемент `cancellations`, снимок этой минуты не содержит заказ в `pending_orders`.
- [x] 5.3 `app/static/app.js`: `cancellations` обрабатываются как `expiries` (снять маркер, вспышка в зоне происхождения).
  Verify: вручную — сервер на мире с `cancel_prob_per_min = 0.1`, `max_pickup_min = 10`, маркеры отменённых заказов гаснут со вспышкой и не висят.

## 6. Сквозная проверка

- [x] 6.1 Прогон 7 суток при 160 водителях, `max_pickup_min = 10`, `cancel_prob_per_min = 0.08` в скрипте во временной БД (не в `data/`). Сравнение с прогоном без радиуса: есть бакеты с `unfulfilled > 0` в зоне при `free_drivers > 0` в другой зоне того же бакета.
  Verify: напечатанная доля таких бакетов > 0 и число отказов > 0; результат записать одной строкой в итог apply.
- [x] 6.2 `make sim` укладывается в бюджет A4 (≤ 60 с на 30 дней), `make epoch-check` проходит на существующих БД.
  Verify: вывод команд.

## 7. Финальные ворота

- [x] 7.1 `make lint`.
  Verify: exit 0.
- [x] 7.2 `make test`.
  Verify: exit 0, все тесты зелёные.
