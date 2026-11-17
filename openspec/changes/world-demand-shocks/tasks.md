# Tasks: world-demand-shocks

Решения — design.md. После каждой группы `make lint && make test` должны быть зелёными, `test_reference_run_bitwise_compatibility` — без правки хэшей.

## 1. Конфиг demand.shocks

- [x] 1.1 `common/config.py`: модель `ShocksCfg` (`enabled = False`, `city_sigma = 0.12` ≥ 0, `city_tau_min = 240` > 0, `zone_sigma = 0.10` ≥ 0, `zone_tau_min = 120` > 0, `day_sigma = 0.08` ≥ 0, `extra="forbid"`); поле `DemandCfg.shocks` с дефолтом.
  Verify: тесты в `tests/test_config.py` — конфиг без подсекции; `{enabled: true, city_sigma: 0.2}`; `city_tau_min: 0` и `zone_sigma: -0.1` отклоняются с путём; неизвестный ключ `holiday_mult` с путём.
- [x] 1.2 Подсекция `shocks:` в `demand` файла `config.yaml` со значениями, равными дефолтам (`enabled: false`), и комментариями.
  Verify: `load_config('config.yaml').demand.shocks == ShocksCfg()`.

## 2. Процесс шоков в среде

- [x] 2.1 `simulator/env.py`: `WorldEnv(cfg, rng, shock_rng=None)`; при `enabled` — стационарный старт `x_c`, `x_z`; `advance_minute(t)` (на 00:00 сначала `d`, затем `x_c`, затем `x_z` по зонам); чистое чтение `demand_factor(zone_id, t)` с поправкой `−Σσ²/2`; при выключенных шоках — 1.0 и генератор не используется.
  Verify: тесты в `tests/test_sim_env.py` — «среднее множителя около 1» (60 суток, [0.95, 1.05]), «персистентность компоненты» (автокорреляция лага 60 в [0.70, 0.85]), «детерминизм шоков по seed», «чтение не тратит rng», «шоки не сдвигают дождь и инъекции» (30 суток).

## 3. Шоки в λ и в движке

- [x] 3.1 `simulator/demand.py`: `lambda_rate(..., shock_mult=1.0)` — множитель перед эластичностью; `spawn_requests(..., shock_mult_by_zone=None)`.
  Verify: тесты в `tests/test_sim_demand.py` — «шок масштабирует λ» (отношение 1.25), существующие сценарии λ без правок.
- [x] 3.2 `simulator/engine.py`: `SeedSequence(seed).spawn(4)`, четвёртый генератор — в `WorldEnv`; в шаге 1 при `enabled` — `env.advance_minute(t)` и массив `demand_factor` по зонам, передаваемый в `spawn_requests`.
  Verify: тесты в `tests/test_sim_loop.py` — «шоки меняют спрос» (7 суток, `city_sigma = 0.3`, суммарное число заказов в пределах 10 %), «путь шоков одинаков во всех ветках seed» (без политики и с ×1.3), «seed по умолчанию»: первые три генератора `spawn(4)` равны `spawn(3)`.
- [x] 3.3 Совместимость: «эталонный прогон без шоков» — `test_reference_run_bitwise_compatibility` зелёный без правки хэшей; `make epoch-check` проходит.
  Verify: вывод тестов и команды.

## 4. Сквозная проверка

- [x] 4.1 Скрипт во временном каталоге (не в `data/`): мир e11 (`notebooks/start_conditions_lab.py`) с шоками по умолчанию, 37 суток seed 42; повторить сравнение e17b (календарь против HGB и sklearn-MLP на H = 15/60/120).
  Verify: в отчёте apply — AP календаря и сетей по горизонтам; ожидание — выигрыш сетей над календарём больше, чем без шоков (§19 `notebooks/stop.md`). Если выигрыша нет, это фиксируется как вход для подбора σ в `world-final-config`, а не как провал change'а.
- [x] 4.2 `make sim` укладывается в бюджет A4 (≤ 60 с на 30 дней), так же при `shocks.enabled = true`.
  Verify: вывод команды.

## 5. Финальные ворота

- [x] 5.1 `make lint`.
  Verify: exit 0.
  Примечание apply: `make lint` красный на неизменённом файле `notebooks/action_model_lab.py` из базового коммита (5 ошибок ruff: I001, E501×2, E702×2; он же не отформатирован). Файлы этого change'а (`common/`, `simulator/`, `tests/`, `config.yaml`) проходят `ruff check` и `ruff format --check` чисто. Правка чужого исследовательского файла/исключений lint — вне scope change'а.
- [x] 5.2 `make test`.
  Verify: exit 0, все тесты зелёные.
