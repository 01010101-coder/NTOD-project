## ADDED Requirements

### Requirement: Секция отклика предложения drivers.reposition

Модель секции `drivers` SHALL содержать вложенную подсекцию `reposition` с полями и дефолтами:

| поле | тип | дефолт |
|---|---|---|
| `enabled` | bool | `false` |
| `every_min` | int ≥ 1 | `5` |
| `min_gain` | float > 0 | `0.10` |
| `sensitivity` | float > 0 | `1.0` |
| `max_share` | float в (0, 1] | `0.3` |
| `max_reach_min` | int ≥ 1 | `12` |

Конфиг без подсекции `reposition` и конфиг с частью полей SHALL загружаться, недостающие поля берутся из дефолтов. Неизвестные ключи внутри `reposition` SHALL отклоняться с путём до ключа. Значения вне допустимых диапазонов SHALL отклоняться ошибкой валидации. Эталонный `config.yaml` SHALL содержать подсекцию со значениями, равными дефолтам, в том числе `enabled: false`.

#### Scenario: Старый конфиг без подсекции

- **WHEN** загружается конфиг, где `drivers` задан как `{count: 380, wait_limit_min: 6}`
- **THEN** загрузка успешна, `drivers.reposition.enabled` равно `false`, остальные поля равны дефолтам

#### Scenario: Частичная подсекция

- **WHEN** загружается конфиг с `drivers.reposition: {enabled: true, max_share: 0.5}`
- **THEN** `enabled` равно `true`, `max_share` равно `0.5`, `every_min` равно `5`, `min_gain` равно `0.10`

#### Scenario: Неизвестный ключ отклоняется

- **WHEN** конфиг содержит `drivers.reposition.max_shar`
- **THEN** загрузка завершается ошибкой с путём `drivers.reposition.max_shar`

#### Scenario: Недопустимое значение отклоняется

- **WHEN** конфиг содержит `drivers.reposition.max_share: 1.5` или `drivers.reposition.every_min: 0`
- **THEN** загрузка завершается ошибкой валидации, конфигурация не создана
