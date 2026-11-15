# Tasks: m6-compare-tab

План вехи — `docs/plan-m5-m6-policy.md` (строка 5 §7, §6 UI, §9 DoD); решения —
design.md (расчёт в `app/compare.py`, ожидание = `pickup_at − created_at`, доля
дефицита сверх плана, кэш по mtime, точность только по A). Эталоны — только
чтение: `data/worlds/*.db`, `data/m3_epoch.db`, `artifacts/*`. Зависимость:
REST/фронт `m6-worlds-ui` (`/api/worlds`, переключатель) уже в коде.

## 1. Расчёт исходов (`app/compare.py`)

- [x] 1.1 Окно тест-недели и исходы ветки: `test_window(cfg)` (`[world_start +
  train_days, + test_days)`), `branch_outcomes(db, cfg, path_min, window)` —
  `trips`, `unfulfilled`, `avg_wait_min` (`pickup_at − created_at`), `revenue`
  (формула §10.15), `deficit_share` (правило контракта по `zone_metrics_5min`),
  часовой ряд кумулятивной выручки; соединение `mode=ro`. Verify: тесты в новом
  `tests/test_web_compare.py` на мини-БД (`init_db` в tmp) с заказами внутри и
  вне окна — числа равны ручному расчёту; ряд длины `test_days × 24`, не
  убывает, последняя точка = `revenue`; пустая БД — нули/`null` без исключений
- [x] 1.2 Точность против A: `accuracy_vs_base(db_a, cfg, window)` через
  `SqliteRowSource` + `join_facts` + `regression_metrics`/`deficit_metrics` по
  `forecasts` A окна, семейства `mlp`/`lstm` × горизонты 15/30 (MAE спроса,
  precision/recall/F1). Verify: тест с ручными прогнозами и фактом целевых
  бакетов — MAE/F1 совпадают с ручным расчётом; БД без прогнозов — пустой раздел
- [x] 1.3 Сборка ответа `build_compare(cfg, worlds_dir)`: ветки в порядке
  `list_worlds`, `exists`, `null`-исходы у отсутствующих, дельты к A в %
  (`null` при нулевой/отсутствующей базе, у A — `null`), `window`, раздел
  точности с источником `a-base`. Verify: тесты — дельта выручки 1000→1050 =
  5.0; `unfulfilled_A = 0` → `null`; без БД A — дельты и точность пусты;
  изменение `forecasts` в БД C не меняет раздел точности

## 2. REST

- [x] 2.1 `GET /api/ml/compare` в `app/server.py`: вызов `build_compare` с
  `worlds_dir` приложения, кэш в замыкании `create_app` по ключу (путь,
  `st_mtime_ns`, `st_size`) всех БД веток (design решение 7). Verify: тесты
  TestClient — 200 и ключи ответа на tmp-сетке; пустой каталог — 200, все
  `exists=false`; повторный запрос не пересчитывает (счётчик вызовов через
  monkeypatch), появление файла d-lstm — пересчёт и `exists=true`

## 3. Фронт вкладки «Сравнение»

- [x] 3.1 Разметка и стили: кнопка `data-tab="compare"` в `#tab-bar`, секция
  `#compare` (пустое состояние с командой `python -m simulator.branches --all`,
  таблица исходов, канвас кумулятивной выручки с легендой и подписью итоговой
  разницы к A, таблица точности с подписью «против факта ветки A; в B/C/D не
  показывается — политика меняет факт»); CSS видимости `body[data-tab=…]` для
  трёх вкладок. Verify: расширенный тест статики (прецедент
  `test_static_page_lab_tab`) — кнопка, секция и id элементов на месте
- [x] 3.2 Логика `app.js`: `compare = {loaded, loading, data}`,
  `ensureCompareLoaded` при первом открытии, рендер таблицы (дельты под числом,
  класс `best` по направлению столбца, строка «не сгенерирована»), таблица
  точности, `redrawCompare` на каждом показе вкладки и по завершении загрузки,
  если вкладка видна; график — `LabCharts.drawTimeSeries` с цветами веток.
  Verify: `make lint` (ruff не касается JS — проверка `node --check
  app/static/app.js`, если node доступен); поведение — ручная приёмка 4.1

## 4. Приёмка

- [x] 4.1 Ручная приёмка в браузере на реальной сетке (`make serve`): (а) три
  вкладки, переключение без перезагрузки, «Город» продолжает воспроизведение
  после возврата; (б) таблица 4 веток × 5 исходов, дельты к A, подсветка
  лучших; (в) четыре линии кумулятивной выручки, после ухода на «Город» до
  загрузки и возврата график цел; (г) таблица точности против A с подписью.
  Verify: наблюдения по пунктам в отчёте задачи; расхождение со спекой — стоп и
  правка
- [x] 4.2 Сверка с ручным SQL (DoD §9): `sqlite3` по каждой БД `data/worlds/` —
  `COUNT(*)` completed/unfulfilled окна и среднее `pickup_at − created_at`
  совпадают с ответом `/api/ml/compare`; выручка — сверка скриптом в scratchpad
  по формуле. Verify: команды и числа зафиксированы в отчёте задачи

## 5. Финальные ворота

- [ ] 5.1 `make lint` и `make test` зелёные; `openspec validate m6-compare-tab
  --strict` без ошибок; эталонные данные не тронуты (`git status` по `data/`
  без изменений). Verify: вывод команд зафиксирован в отчёте задачи
