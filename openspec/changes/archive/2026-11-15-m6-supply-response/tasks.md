# Tasks: m6-supply-response

Решения — design.md. Ворота после каждой группы: `make lint && make test`
зелёные.

## 1. Эталон совместимости (до любых правок кода)

- [x] 1.1 Добавить в `tests/test_sim_loop.py` тест: прогон 1 сутки seed 42 на
  эталонном конфиге, sha256 упорядоченных строк `trips` и
  `zone_metrics_5min`; хэши получить на текущем коде и записать литералами
  (design, решение 8).
  Verify: тест зелёный на неизменённом коде.

## 2. Конфиг drivers.reposition

- [x] 2.1 `common/config.py`: модель `RepositionCfg` (`enabled`, `every_min`,
  `min_gain`, `sensitivity`, `max_share`, `max_reach_min`) с дефолтами и
  диапазонами из спеки config-loading; поле `DriversCfg.reposition` с
  дефолтом.
  Verify: новые тесты в `tests/test_config.py` — старый конфиг без
  подсекции, частичная подсекция, неизвестный ключ с путём, `max_share: 1.5`
  и `every_min: 0` отклоняются.
- [x] 2.2 Подсекция `reposition:` в `config.yaml` со значениями, равными
  дефолтам (`enabled: false`), с комментариями.
  Verify: `load_config('config.yaml').drivers.reposition` равен
  `RepositionCfg()`; тест 1.1 зелёный.

## 3. Правило отклика (чистая функция)

- [x] 3.1 Новый `simulator/reposition.py`: `plan_relocations(...)` по спеке
  simulator-supply-response — прирост, допустимость (`min_gain`,
  `max_reach_min`), одна цель на зону с тай-брейками, доля
  `floor(min(max_share, sensitivity × gain) × free)`, наименьшие id, решение по
  снимку, порядок вывода.
  Verify: новый `tests/test_sim_reposition.py` покрывает все сценарии спеки
  (переезд 3 из 10, малый прирост, дальняя зона, тай-брейк путь/id, ровный
  тариф, потолок доли, повторный вызов не меняет вход).

## 4. Парк: состояние RELOCATING

- [x] 4.1 `simulator/drivers.py`: `DriverState.RELOCATING`;
  `Fleet.relocate(driver_id, dest_zone, minutes)` с ошибкой для несвободного;
  тик переезжающих в `tick()`; прибытие → `FREE` в зоне цели, вне списка
  высадок; `arrivals()` — прибытия последнего тика по возрастанию id;
  `busy_by_zone` учитывает переезжающих в исходной зоне.
  Verify: новые тесты в `tests/test_sim_drivers.py` — полный цикл переезда
  3 минуты, `nearest_free` → `None`, ошибка `relocate` для трёх занятых
  состояний, снапшоты вдоль переезда, инвариант суммы на смешанной
  последовательности; старые тесты модуля зелёные.

## 5. Интеграция в движок

- [x] 5.1 `simulator/engine.py`: шаг 3а при `enabled` и `k % every_min == 0` —
  `plan_relocations` по `_free_ids` и `price_mult_by_zone`, длительность
  `travel_minutes(path, travel_k)`, `Fleet.relocate` и удаление из
  `_free_ids`; в шаге 4 после `tick()` — возврат прибывших из
  `Fleet.arrivals()` в `_free_ids` (`insort`); то же в дренаже хвоста.
  Обновить docstring модуля и `Engine` (новые точки смены состояния
  парка).
  Verify: тест 1.1 зелёный при `enabled = false`.
- [x] 5.2 Интеграционные тесты в `tests/test_sim_loop.py`:
  - «свободные подтягиваются к надбавке» — 2 часа, `price_fn` с зоной 5 = 1.3,
    среднее `free_drivers` зоны 5 при `enabled = true` больше, чем при `false`;
  - «отклик не тратит rng» — совпадают `is_rain` и последовательность заказов
    при константном `price_fn`;
  - «решения только на минутах решения» — первый переезд на `k = 5`;
  - инвариант `free + busy = drivers.count` на каждой строке агрегата.
  Verify: тесты зелёные.
- [x] 5.3 Замер wall-time прогона 7 дней с `enabled = true` против `false`
  (скрипт в scratchpad, не в репо).
  Verify: замедление не больше 10 %, цифры записаны в итоговое сообщение
  apply.

## 6. Приёмка

- [x] 6.1 Полный прогон `make lint && make test`.
  Verify: оба зелёные, число тестов выросло относительно 434.
- [x] 6.2 `make epoch-check` на существующих БД без регенерации.
  Verify: проверка проходит, md5 совпадают.
- [x] 6.3 Пробный прогон с откликом на одной ветке во временную БД
  (scratchpad): `enabled = true`, политика C на тест-неделе, сравнение
  `free_drivers` и доли дефицита против существующей `c-mlp.db`.
  Это только наблюдение для следующего change'а, данные в репо не пишутся.
  Verify: цифры записаны в итоговое сообщение apply.
