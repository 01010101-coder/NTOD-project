## MODIFIED Requirements

### Requirement: Прогнозы, соединённые с фактом

`GET /api/ml/forecasts` SHALL принимать фильтры `zone`, `model`, `horizon`,
`from`, `to` (необязательные; `horizon` — 15 или 30) и возвращать строки
`forecasts`, каждая дополнена фактом бакета `ts + horizon_min` той же БД:
`actual_requests`, `actual_free_drivers`, `actual_deficit`. `actual_deficit`
SHALL вычисляться по правилу метки `ml.deficit_label` конфигурации сервера
(при `legacy` — `free_drivers / max(requests,1) < 0.5` или `unfulfilled > 0`,
при `harm` — `unfulfilled > 0`). Если факт-бакет отсутствует, поля факта SHALL быть
`null`. Фильтры SHALL применяться строго: в ответе только строки, удовлетворяющие
всем заданным фильтрам.

#### Scenario: Фильтр возвращает только запрошенное

- **WHEN** запрошены `zone=5`, `model=<mlp-версия>`, `horizon=15`
- **THEN** каждая строка ответа имеет `zone_id=5`, `model_version=<mlp-версия>`,
  `horizon_min=15`, и строки упорядочены по `ts`

#### Scenario: Факт берётся из бакета ts + горизонт

- **WHEN** строка прогноза имеет `ts=T`, `horizon_min=15`
- **THEN** `actual_requests` строки равен `requests` бакета `zone_metrics_5min`
  с `ts=T+15 мин` той же зоны

#### Scenario: Факт дефицита по вреду

- **WHEN** сервер запущен с `ml.deficit_label: harm`, а факт-бакет имеет
  `free_drivers = 1`, `requests = 10`, `unfulfilled = 0`
- **THEN** `actual_deficit` строки равен `false`

#### Scenario: Несуществующая зона

- **WHEN** запрошена зона, которой нет в данных
- **THEN** ответ содержит пустой список строк с кодом 200
