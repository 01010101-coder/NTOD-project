## ADDED Requirements

### Requirement: Секция диспатча drivers.dispatch

Модель секции `drivers` SHALL содержать вложенную подсекцию `dispatch` с полями и дефолтами:

| поле | тип | дефолт | смысл |
|---|---|---|---|
| `max_pickup_min` | int ≥ 1 или `null` | `null` | радиус поиска: максимальная длительность подачи в минутах; `null` — без ограничения |
| `cancel_grace_min` | int ≥ 0 | `5` | подача, которую пассажир терпит без риска отказа, минут |
| `cancel_prob_per_min` | float ≥ 0 | `0.0` | прирост вероятности отказа за каждую минуту подачи сверх `cancel_grace_min`; `0.0` — отказов нет |
| `cancel_max_prob` | float в [0, 1] | `0.9` | потолок вероятности отказа |

Конфиг без подсекции `dispatch` и конфиг с частью полей SHALL загружаться, недостающие поля берутся из дефолтов. Неизвестные ключи внутри `dispatch` SHALL отклоняться с путём до ключа. Значения вне допустимых диапазонов SHALL отклоняться ошибкой валидации. Эталонный `config.yaml` SHALL содержать подсекцию со значениями, равными дефолтам (`max_pickup_min: null`, `cancel_prob_per_min: 0.0`).

#### Scenario: Старый конфиг без подсекции

- **WHEN** загружается конфиг, где `drivers` задан как `{count: 380, wait_limit_min: 6}`
- **THEN** загрузка успешна, `drivers.dispatch.max_pickup_min` равно `None`, `drivers.dispatch.cancel_prob_per_min` равно `0.0`, остальные поля равны дефолтам

#### Scenario: Частичная подсекция

- **WHEN** загружается конфиг с `drivers.dispatch: {max_pickup_min: 12}`
- **THEN** `max_pickup_min` равно `12`, `cancel_grace_min` равно `5`, `cancel_prob_per_min` равно `0.0`, `cancel_max_prob` равно `0.9`

#### Scenario: Неизвестный ключ отклоняется

- **WHEN** конфиг содержит `drivers.dispatch.max_pickup`
- **THEN** загрузка завершается ошибкой с путём `drivers.dispatch.max_pickup`

#### Scenario: Недопустимое значение отклоняется

- **WHEN** конфиг содержит `drivers.dispatch.max_pickup_min: 0`, или `cancel_prob_per_min: -0.1`, или `cancel_max_prob: 1.5`
- **THEN** загрузка завершается ошибкой валидации, конфигурация не создана
