## ADDED Requirements

### Requirement: Секция вызова подработчиков drivers.callout

Модель секции `drivers` SHALL содержать вложенную подсекцию `callout` с полями:

| поле | тип | дефолт | ограничение |
|---|---|---|---|
| `enabled` | bool | `false` | — |
| `part` | int | `60` | ≥ 0 |
| `lag_min` | int | `20` | ≥ 1 |
| `session_min` | int | `120` | ≥ 1 |
| `every_min` | int | `5` | ≥ 1 |
| `min_gain` | float | `0.10` | ≥ 0 |
| `sensitivity` | float | `0.3` | ≥ 0 |
| `max_share` | float | `0.3` | в (0, 1] |

`part` — число подработчиков сверх штатного парка `drivers.count`. Остальные поля задают правило вызова (capability `simulator-callout`) и его применение в минутном цикле (capability `simulator-loop`).

Конфиг без подсекции `callout` и конфиг с частью её полей SHALL загружаться, а недостающие поля SHALL браться из дефолтов. Неизвестные ключи внутри `callout` SHALL отклоняться с путём до ключа. Значение вне ограничения SHALL отклоняться ошибкой валидации с путём до поля. Эталонный `config.yaml` SHALL содержать подсекцию со значениями, равными дефолтам, в том числе `enabled: false`.

#### Scenario: Старый конфиг без подсекции

- **WHEN** загружается конфиг, где `drivers` задан как `{count: 380, wait_limit_min: 6}`
- **THEN** загрузка успешна, `drivers.callout.enabled` равно `false`, остальные поля равны дефолтам

#### Scenario: Частичная подсекция

- **WHEN** загружается конфиг с `drivers.callout: {enabled: true, lag_min: 30}`
- **THEN** `enabled` равно `true`, `lag_min` равно 30, остальные поля равны дефолтам

#### Scenario: Недопустимое значение отклоняется

- **WHEN** конфиг содержит `drivers.callout.lag_min: 0` или `drivers.callout.max_share: 1.5`
- **THEN** загрузка завершается ошибкой валидации с путём до поля, конфигурация не создана

#### Scenario: Неизвестный ключ отклоняется

- **WHEN** конфиг содержит `drivers.callout.bonus_rub`
- **THEN** загрузка завершается ошибкой с путём `drivers.callout.bonus_rub`
