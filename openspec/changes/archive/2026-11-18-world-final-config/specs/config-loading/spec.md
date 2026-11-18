## ADDED Requirements

### Requirement: Корень артефактов ml.artifacts_root

Секция `ml` SHALL содержать поле `artifacts_root` (строка, дефолт `artifacts`). Все потребители артефактов SHALL брать корень из этого поля конфига, с которым запущены, а не из константы модуля:
- сохранение и поиск артефактов M3 (`{семейство}-v1-*`);
- сводки `summary-*.json` инференса;
- `policy-calibration.json` и `signal-calibration.json`;
- тень-прогнозы и контуры политики;
- сервер воспроизведения.

При дефолте пути и поведение SHALL совпадать с прежними.

#### Scenario: Дефолт сохраняет прежние пути

- **WHEN** обучение и калибровка выполняются с конфигом без `ml.artifacts_root`
- **THEN** артефакты и json калибровки создаются в `artifacts/`, как до change'а

#### Scenario: Отдельный корень изолирует миры

- **WHEN** обучение выполняется с `ml.artifacts_root: artifacts/final` во временном каталоге, а затем контур ветки `c-mlp` строится с конфигом без этого ключа
- **THEN** артефакт создан в `artifacts/final/`, а контур с дефолтным конфигом его не находит и использует `artifacts/`

### Requirement: Финальный конфиг мира config.final.yaml

Репозиторий SHALL содержать `config.final.yaml`, загружаемый `load_config` без ошибок. Эталонный `config.yaml` SHALL оставаться равным дефолтам моделей.

В финальном конфиге SHALL быть включены:
- смены (`drivers.shifts.enabled`);
- радиус подачи и отказы (`drivers.dispatch.max_pickup_min` не `null`, `cancel_prob_per_min > 0`);
- отклик водителей (`drivers.reposition.enabled`);
- вызов подработчиков (`drivers.callout.enabled`);
- шоки спроса (`demand.shocks.enabled`).

Кроме того, SHALL выполняться:
- `ml.deficit_label` равно `harm`, `policy.channel` равно `both`;
- пути `storage.db_path`, `ml.db`, `policy.worlds_dir`, `ml.artifacts_root`, `ml.planning.artifacts_root` лежат под `data/final/` и `artifacts/final/` и не совпадают с путями `config.yaml`.

Комментарий в начале файла SHALL ссылаться на раздел ноутбука `notebooks/start_conditions.ipynb`, где обоснованы значения.

#### Scenario: Финальный конфиг валиден и изолирован

- **WHEN** загружаются `config.yaml` и `config.final.yaml`
- **THEN** оба загружаются; у финального включены все пять механизмов, `deficit_label = harm`, `channel = both`; ни один путь данных или артефактов финального конфига не совпадает с путём эталонного

#### Scenario: Эталонный конфиг не изменился

- **WHEN** выполняются существующие тесты, сравнивающие `config.yaml` с дефолтами, и эталонный тест хэшей прогона
- **THEN** они проходят без правок
