## 1. Типы событийного потока

- [x] 1.1 Создать `simulator/live.py`: frozen dataclass `SpawnEvent` (internal_id, origin, dest, price_multiplier), `AssignEvent` (internal_id, driver_id, driver_zone, origin, dest, pickup_at, dropoff_at), `DropoffEvent` (internal_id, driver_id, dest_zone), `ExpiryEvent` (internal_id, origin) и `MinuteRecord` (t, is_rain, price_mult — кортеж по зонам, free_by_zone — кортеж по зонам, spawns, assignments, dropoffs, expiries — кортежи событий); docstrings на русском, идентификаторы на английском. Проверка: `make lint` чист, unit-тест `tests/test_sim_live.py` конструирует все типы и падает на попытке присвоивания поля frozen dataclass (`dataclasses.FrozenInstanceError`).

## 2. Шов on_minute в движке

- [x] 2.1 В `Engine.run` добавить keyword-only параметр `on_minute: Callable[[MinuteRecord], None] | None = None`: в цикле минуты собирать списки событий (спавн — шаг 2; назначение с `driver_zone` зоны водителя до исключения из зеркала и `pickup_at = t + pickup_min − 1`, `dropoff_at = pickup_at + trip_min` — шаг 3; высадки — шаг 4; истечения — шаг 5) и после шага 6 вызывать колбэк записью `MinuteRecord` с полями среды шага 1а (`is_rain`, вектор `price_mult_by_zone`) и снапшотом `fleet.free_by_zone()`; сбор и построение записи — только при заданном колбэке. Проверка: тест первой записи короткого прогона — `t` равен `sim_start 00:00:00`, длины `price_mult`/`free_by_zone` равны числу зон, стартовые события согласованы со стартом парка `i mod Z` (зоны водителей первых назначений — допустимые id).
- [x] 2.2 Излучать записи дренажа хвоста: минуты после `days × 1440` выдают записи с `dropoffs`/`expiries` оставшихся заказов, пустыми `spawns`/`assignments` и средой, вычисленной чистыми чтениями `env.is_rain(t)`/`_tariff_mult` на `t` записи (rng не тратится); записи идут до терминального состояния всех заказов. Проверка: тест прогона с активным хвостом — записи продолжаются за границей суток, последняя запись после закрытия последнего заказа, в дренаже только dropoffs/expiries.

## 3. Тесты контракта simulator-live

- [x] 3.1 `test_observer_purity`: два прогона одного seed (один с `on_minute`, другой без) пишут в две БД в памяти — строки `trips` и `zone_metrics_5min` совпадают попарно, `Summary` равны; два прогона с `on_minute` дают равные последовательности записей (запись за записью, включая порядок событий). Проверка: pytest зелёный.
- [x] 3.2 `test_events_match_trips`: по короткому прогону с фактической записью в БД — каждой строке `completed` соответствует цепочка spawn → assignment → dropoff с тем же internal_id, где `t` записей равны `created_at`/`assigned_at`/`dropoff_at` строки, зоны согласованы с `origin_zone`/`dest_zone`, а `t` высадки равна `dropoff_at` её назначения; каждой строке `unfulfilled` — spawn → expiry без назначения; суммы по прогону: spawns = `trips_total`, assignments = dropoffs = `completed`, expiries = `unfulfilled`. Проверка: pytest зелёный.
- [x] 3.3 `test_free_park_invariant`: для каждой записи прогона сумма `free_by_zone` равна `drivers.count` минус число активных поездок на конец минуты (назначления без закрывшей их высадки, восстановленные из потока). Проверка: pytest зелёный.
- [x] 3.4 `test_minute_rhythm`: прогон `days` суток с `on_minute` даёт ровно `days × 1440` записей основной фазы (плюс записи дренажа, если хвост активен), `t` растёт ровно на минуту, секунды всегда 00. Проверка: pytest зелёный.

## 4. Приёмка

- [x] 4.1 Прогон `.venv/bin/make lint && .venv/bin/make test` — оба зелёные; существующие тесты движка не менялись (изменения в `tests/` — только новый файл `tests/test_sim_live.py`), `python -m simulator` работает как раньше (smoke 1 день, сводка прежних 4 строк).
