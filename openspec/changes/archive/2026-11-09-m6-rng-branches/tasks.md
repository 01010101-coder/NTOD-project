# Tasks: m6-rng-branches

План вехи — `docs/plan-m5-m6-policy.md` (§3, §7 change 1); решения — design.md.
Ворота после каждой группы: `make lint && make test` зелёные.

## 1. Разделение rng-потоков в движке

- [x] 1.1 В `simulator/engine.py` заменить инициализацию rng: `np.random.SeedSequence(seed).spawn(2)` → два дочерних `default_rng` в порядке `[эндогенный, экзогенный]`; эндогенный остаётся у `spawn_requests`, экзогенный передаётся `WorldEnv`; docstring «один rng на мир (§10.9)» переписан на разделение потоков со ссылкой на §3 плана M5+M6. Verify: `make lint` чист; `make test` зелёный (детерминизм-тесты `test_sim_repro` проходят без правок)
- [x] 1.2 Тест «инициализация из конфига»: `Engine` без явного seed строит генераторы из `SeedSequence(cfg.seed)`; два движка одного seed дают идентичные поминутные опросы среды (`is_rain`/`price_multiplier`). Verify: новый тест в `tests/test_sim_branches.py` (или `test_sim_loop.py` рядом с детерминизмом) зелёный

## 2. Тест независимости экзогенного потока (регрессионная суть change'а)

- [x] 2.1 Тест: два движка один seed — базовый против `price_fn = ×3.0` константой (α=1 → λ/3); после нескольких суток поминутные `is_rain`/`price_multiplier` сред идентичны, а `trips` различаются. Verify: тест зелёный на новой разводке; ручная проверка на старой (до правки 1.1, через stash) — красный, факт зафиксирован в сообщении коммита или отчёте change'а
  - Отчёт: красный прогон на старой разводке зафиксирован (тест написан и запущен ДО правки 1.1): `assert polls_base[: DAYS * 1440] == polls_priced[: DAYS * 1440]` падает на индексе 2220 (сутки 2, ~13:00) — `(False, ...) != (True, ...)`: у базового прогона в этой минуте дождя нет, у прогона с ×3.0 есть (общий rng сдвинулся расходом эндогенных розыгрышей; git stash не потребовался — правка 1.1 ещё не была внесена).
  - Правка по верификации (WARNING «сценарий не прикрыт тестом»): добавлен `test_price_fn_equal_to_env_tariff_identical_to_base` — сценарий дельты simulator-loop «price_fn, повторяющий тариф среды, неотличим от базы»: mimic-движок с `price_fn` = `env.price_multiplier` против базы, один seed; строки обеих таблиц побитово равны; осмысленность обеспечена детерминизмом (в окне 3 суток seed 42 среда разыгрывает инъекцию — 42 минуты ≠ 1.0, проверено пробой). Файл `tests/test_sim_branches.py`: 10 passed, lint чист.

## 3. Общая подложка прогонов writers.py

- [x] 3.1 Перенести `SqliteTripWriter`, `GUARD_MESSAGE` и печать прогресса/сводки из `simulator/__main__.py` в новый `simulator/writers.py` (публичные имена `print_progress`/`print_summary`), `__main__` переиспользует импортом; поведение и формат вывода неизменны. Verify: `make lint && make test` зелёные без правок существующих CLI-тестов (`test_sim_repro`, CLI-смоуки)
  - Отчёт: полный прогон после переноса — 361 passed, lint чист; `test_sim_repro` и CLI-смоуки зелёные без правок (`simulator.__main__.SqliteTripWriter` доступен через реимпорт). Замечание: в двух промежуточных прогонах флейнили 1–2 WS-теста `test_web_server` (гонки сообщений плейбэк-потока под нагрузкой, к change'у отношения не имеют — `app/` не импортирует `simulator.writers`; в финальном прогоне обе зеленые).

## 4. Веточный генератор

- [x] 4.1 `simulator/branches.py`: константа `WORLDS_DIR = Path("data/worlds")`, `BranchSpec(name, price_fn_factory=None)`, реестр `BRANCHES = {"a-base": ...}`, `generate_branch(name, cfg, *, seed=None, days=37, db_path=None)` — полный прогон движка в БД (guard непустой `trips` — сообщение из `writers.GUARD_MESSAGE`, makedirs каталога, сводка формата §4). Verify: unit-тест генерации `a-base` 2 дня в tmp_path (схема БД, строки метрик 288/сутки·зона, `price_multiplier` из среды) зелёный
  - Отчёт: реализовано по design решение 4; в сигнатуру добавлен параметр `price_fn=None` из дельты спеки simulator-branches (явный `price_fn` имеет приоритет над фабрикой ветки). Тесты `test_generate_branch_a_base`/`test_generate_branch_unknown_branch_raises` зелёные.
- [x] 4.2 CLI `main(argv)`: `--branch` (обязателен), `--days` (дефолт 37), `--config`, `--seed`, `--db` (дефолт `data/worlds/<ветка>.db`); неизвестная ветка и guard → stderr + exit 2. Verify: CLI-тесты (успешный прогон 1 день в tmp, guard exit 2, неизвестная ветка exit 2) зелёные
  - Отчёт: `test_branch_cli_smoke`/`test_branch_cli_guard_rejects_nonempty_trips`/`test_branch_cli_unknown_branch` зелёные (неизвестная ветка проверяется до load_config — БД не создаётся).

## 5. Побитовая повторяемость и приёмка

- [x] 5.1 Тест повторяемости: `generate_branch("a-base", days=2)` дважды в разные tmp-БД → экспорты `trips` и `zone_metrics_5min` (`storage.csv_export.export_table`) совпадают; другой seed → различаются. Verify: тест зелёный
  - Отчёт: `test_branch_reproducibility` зелёный — CSV-экспорты обеих таблиц попарно байт-в-байт при одном seed, различаются при seed 43.
- [x] 5.2 Приёмка change'а (ручной прогон, фикс в отчёте): `python -m simulator.branches --branch a-base` — `data/worlds/a-base.db` 37 дней, wall ≈ 25–50 с, `deficit_share` из сводки в коридоре [10, 20] %; `python -m simulator.calibrate --days 30` — `deficit_share` в коридоре [10, 20] %. Verify: цифры записаны; при выходе из коридора — стоп и перес-калибровка (ворота плана §8.1), не архив
  - Отчёт: `python -m simulator.branches --branch a-base` (дефолты: 37 дней, seed 42 из конфига, `data/worlds/a-base.db`) — wall **35.2 с** (коридор 25–50 с ✓), сводка: `trips: total=337454 completed=337454 (100%)`, `wait: avg_wait_min=0.00 deficit_share=18.2%` (коридор [10, 20] ✓). `python -m simulator.calibrate --days 30` — `deficit_share=18.2%` (✓), парк: занятость средняя 18.9%, p95 36.3%, wall 26.3 с. Оба прогона в коридоре — перес-калибровка не требуется. Санкциони­рованное следствие (proposal): побитовая связь с эталонами разорвана — новый мир ≠ `taxi.db`/`m3_epoch.db` (такси.db 30 суток: 272573 поездок, 29 unfulfilled, avg_wait 0.011 мин; a-base 37 суток: 337454 поездок, 0 unfulfilled — характер мира сохранён: парк с запасом, дефицит — зональные окна ratio<0.5).
- [x] 5.3 Финальные ворота: `make lint && make test` зелёные; `openspec validate m6-rng-branches` (или `--all`) без ошибок; эталонные БД `data/taxi.db`, `data/m3_epoch.db` и артефакты `artifacts/` не тронуты (mtime/содержимое). Verify: вывод команд зафиксирован
  - Отчёт: `make lint` — exit 0 («59 files already formatted»); финальный полный прогон `.venv/bin/python -m pytest -q` — **367 passed**, exit 0; `openspec validate --all` — «Totals: 20 passed, 0 failed (20 items)»; эталонные БД и `artifacts/` — снапшот mtime до и после приёмки идентичен (diff пуст, «ЭТАЛОНЫ НЕ ТРОНУТЫ»). Замечание о флейке: `tests/test_web_server.py::test_ws_connect_play_pause` в ~половине прогонов падает с `'log-extended' != 'state'` — гонка сообщения фонового генератора лога с raw `receive_json` теста (`wait_filled(client, 20)` ждёт 20 из 288 записей); на идентичном python-коде тест проходил (361- и 367-прогоны) и падал — к change'у отношения не имеет, `app/` не импортирует `simulator.writers`/`simulator.branches`.
