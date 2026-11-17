# Tasks: world-elastic-supply

Решения — design.md. После каждой группы `make lint && make test` должны быть зелёными, `test_reference_run_bitwise_compatibility` — без правки хэшей.

## 1. Конфиг drivers.callout

- [ ] 1.1 `common/config.py`: модель `CalloutCfg` (`enabled = False`, `part = 60` ≥ 0, `lag_min = 20` ≥ 1, `session_min = 120` ≥ 1, `every_min = 5` ≥ 1, `min_gain = 0.10` ≥ 0, `sensitivity = 0.3` ≥ 0, `max_share = 0.3` в (0, 1], `extra="forbid"`); поле `DriversCfg.callout` с дефолтом.
  Verify: тесты в `tests/test_config.py` — старый конфиг без подсекции; `{enabled: true, lag_min: 30}` с дефолтами остальных; `lag_min: 0` и `max_share: 1.5` отклоняются с путём; неизвестный ключ `bonus_rub` отклоняется с путём `drivers.callout.bonus_rub`.
- [ ] 1.2 Подсекция `callout:` в `config.yaml` со значениями, равными дефолтам (`enabled: false`), и комментариями.
  Verify: `load_config('config.yaml').drivers.callout == CalloutCfg()`.

## 2. Правило вызова

- [ ] 2.1 Новый `simulator/callout.py`: `plan_callouts(pool_ids, signal_by_zone, cfg) -> list[tuple[int, int]]` — прирост `signal − 1.0`, порог `min_gain` с допуском, `n = floor(min(max_share, sensitivity × gain) × P + 1e-9)` от размера резерва на входе, зоны по возрастанию, водители из остатка резерва по возрастанию id; вход не меняется.
  Verify: новый `tests/test_sim_callout.py` — сценарии спеки `simulator-callout`: 5 водителей (240–244) в зону 5; малый прирост; потолок 18 из 60; зоны 2 и 4 делят резерв `[250, 251, 252]`; ровный сигнал; повторный вызов равен и вход не изменён.

## 3. Шов сигнала водителям

- [ ] 3.1 `simulator/engine.py`: параметр `Engine(..., driver_signal_fn=None)`; в шаге 1а массив `signal_by_zone` (без шва — тот же объект, что `price_mult_by_zone`); шаг 3а передаёт в `plan_relocations` `signal_by_zone`.
  Verify: тесты в `tests/test_sim_loop.py` — «сигнал по умолчанию — тариф» (два прогона с `driver_signal_fn`, равным тарифу, и без него идентичны); «отклик на сигнал водителям» (`price_fn=None`, сигнал 1.3 в зоне 5: `price_multiplier` строк агрегата равен тарифу среды, среднее `free_drivers` зоны 5 выше, чем без шва); эталонный тест зелёный.

## 4. Подработчики в движке

- [ ] 4.1 Парк размером `count + part` при `callout.enabled`; подработчики уходят в `OFFLINE` до первой минуты и не попадают в `_free_ids`; резерв, очередь вызванных, окончания сессий, `part_offline`/`part_leaving` — приватные поля движка. Шорткат диспатча и размер парка — по `fleet_size` (решение 8).
  Verify: тест «подработчики стартуют вне линии» (count 240, part 60: после первой минуты без сигнала `free + busy` всех зон ≤ 240).
- [ ] 4.2 Шаг смен считает `O` только по штатному парку (решение 4).
  Verify: тест «подработчики не входят в цель смен» — поминутно штатных на смене ≥ `T(t)` при 10+ подработчиках на линии; существующие тесты смен зелёные.
- [ ] 4.3 Шаг 1в по порядку решения 5: возврат ушедших в резерв → закрытие сессий (свободный уходит сразу, занятый получает метку) → выход вызванных в зоне вызова → решения на `k mod every_min == 0` через `plan_callouts`; только в основной фазе.
  Verify: тесты «выход через задержку в зоне сигнала» (5 водителей недоступны на `t … t+19`, свободны в зоне 5 на `t+20`) и «конец сессии» (свободный уходит на `t0+120`, занятый — после высадки, заказ `completed`, водитель снова в резерве).
- [ ] 4.4 `Summary.callouts` и `Summary.callout_minutes` последними полями с дефолтом 0; считаются в основной фазе.
  Verify: тест «вызов детерминирован» (два прогона одного seed и сигнала: таблицы и оба счётчика равны); при выключенном вызове оба счётчика 0.
- [ ] 4.5 Тесты мира: «вызов не тратит rng» (поминутные `is_rain` и тариф среды совпадают); «включённый вызов без сигнала не меняет мир» (со сменами и `price_fn ≡ 1.0` таблицы совпадают с выключенным вызовом, `callouts == 0`).
  Verify: тесты зелёные; `test_reference_run_bitwise_compatibility` зелёный без правки хэшей.

## 5. Поток наблюдения

- [ ] 5.1 `simulator/live.py`: поле `MinuteRecord.part_online` последним с дефолтом 0; движок заполняет `online_drivers = fleet_size − offline_count()` и `part_online`.
  Verify: тесты в `tests/test_sim_live.py` — без смен и вызова `online_drivers == count`, `part_online == 0`; сценарий «подработчики видны в записи минуты» (`part_online` 0 до 20-й минуты, > 0 с 20-й; `online_drivers` = штатные на смене + `part_online`); существующие тесты согласованности свободного парка зелёные.
- [ ] 5.2 `app/server.py`: сериализовать `part_online` в записи минуты (клиент может игнорировать).
  Verify: существующие тесты `tests/test_web_server.py` и `tests/test_web_replay.py` зелёные; ключ есть в сериализованной записи (одна проверка).

## 6. Сквозная проверка

- [ ] 6.1 Скрипт во временном каталоге (не в `data/`): мир e11 (`notebooks/start_conditions_lab.py`, n=240, смены, радиус, reposition, α 0.5) + `callout.enabled`, 37 дней seed 42, три прогона тест-недели — без сигнала (`price_fn ≡ тариф мира`), реактивный контур в режиме «карта спроса» (`driver_signal_fn=contour.price`), реактивный в режиме «цена».
  Verify: в отчёте apply — необслуженные, перевезено, `Summary.callouts`, часы подработчиков в тест-неделе для трёх прогонов; в режиме «карта спроса» `price_multiplier` заказов тест-недели равен тарифу мира.
- [ ] 6.2 `make sim` укладывается в бюджет A4 (≤ 60 с на 30 дней), так же при `callout.enabled = true`; `make epoch-check` проходит на существующих БД.
  Verify: вывод команд.

## 7. Финальные ворота

- [ ] 7.1 `make lint`.
  Verify: exit 0.
- [ ] 7.2 `make test`.
  Verify: exit 0, все тесты зелёные.
