# Tasks: city-dispatcher-ui

Решения — design.md.

## 1. Сервер

- [x] 1.1 `GET /api/deficit-timeline`: `start`, `step`, `zones`, `counts`, `requests`, `rain` по закрытым окнам активного мира.
  Verify: `test_api_deficit_timeline_matches_metrics` — совпадение с `/api/metrics`.
- [x] 1.2 `snapshot.env.deficit` — дефицит последнего закрытого окна.
  Verify: `test_ws_seek_snapshot_and_clamp` сверяет с метриками окна [395, 400).
- [x] 1.3 `compare.branches[].tariff_mix`.
  Verify: `tests/test_web_compare.py` — ключ в ответе, пустой список у отсутствующей ветки, тарифы по возрастанию.
- [x] 1.4 WS: ожидание первой записи сравнивает `client_state` с `WebSocketState.DISCONNECTED`.

## 2. Город

- [x] 2.1 Разметка и стили по `design/Main.html`: шапка, сегменты мира/режима/скорости, сводка, легенда, панель района, метрики 24 ч, плеер с таймлайном.
- [x] 2.2 `map.js`: дороги, машины по полосам, ореолы (факт/прогноз), ночь/закат, дождь; `CityMap.layout()`.
- [x] 2.3 `hud.js`: плашки районов, сводка, погода, панель района (статус, правило, график 6 ч, дороги), таймлайн, подсказка тестовой недели с отложенным переходом и скрытием.
- [x] 2.4 `charts.js`: окно 24 ч, offscreen-слой, `selectZone`, `zoneRows`.
  Verify: замер — 165 кадров/с на 600×, худший кадр 12 мс.

## 3. Лаборатория и Сравнение

- [x] 3.1 Заголовки страниц, карточки в стиле Города, кольца F1 (`LabCharts.donut`).
- [x] 3.2 Карточки миров A–D с кольцом распределения тарифа и ключевыми исходами.
- [x] 3.3 «Факт vs прогноз»: факт — площадь, прогноз — линия, выбор дня.
  Verify: `tests/test_web_ml_tab.py`, `tests/test_web_compare.py` — статические проверки id и подсказок.

## 4. Переносимость

- [x] 4.1 Тесты CLI запускают `sys.executable` вместо `.venv/bin/python`.
- [x] 4.2 README: запуск на Windows без `make`.
