## ADDED Requirements

### Requirement: Секция шоков спроса demand.shocks

Модель секции `demand` SHALL содержать вложенную подсекцию `shocks` с полями:

| поле | тип | дефолт | ограничение |
|---|---|---|---|
| `enabled` | bool | `false` | — |
| `city_sigma` | float | `0.12` | ≥ 0 |
| `city_tau_min` | float | `240` | > 0 |
| `zone_sigma` | float | `0.10` | ≥ 0 |
| `zone_tau_min` | float | `120` | > 0 |
| `day_sigma` | float | `0.08` | ≥ 0 |

`*_sigma` — стационарное стандартное отклонение компоненты логарифма множителя, `*_tau_min` — время корреляции в минутах. Конфиг без подсекции и конфиг с частью полей SHALL загружаться, недостающие поля SHALL браться из дефолтов. Неизвестные ключи SHALL отклоняться с путём до ключа. Значение вне ограничения SHALL отклоняться ошибкой валидации с путём до поля. Эталонный `config.yaml` SHALL содержать подсекцию со значениями, равными дефолтам, в том числе `enabled: false`.

#### Scenario: Старый конфиг без подсекции

- **WHEN** загружается конфиг, где `demand` задан без `shocks`
- **THEN** загрузка успешна, `demand.shocks.enabled` равно `false`, остальные поля равны дефолтам

#### Scenario: Частичная подсекция

- **WHEN** загружается конфиг с `demand.shocks: {enabled: true, city_sigma: 0.2}`
- **THEN** `enabled` равно `true`, `city_sigma` равно 0.2, остальные поля равны дефолтам

#### Scenario: Недопустимое значение отклоняется

- **WHEN** конфиг содержит `demand.shocks.city_tau_min: 0` или `demand.shocks.zone_sigma: -0.1`
- **THEN** загрузка завершается ошибкой валидации с путём до поля

#### Scenario: Неизвестный ключ отклоняется

- **WHEN** конфиг содержит `demand.shocks.holiday_mult`
- **THEN** загрузка завершается ошибкой с путём `demand.shocks.holiday_mult`
