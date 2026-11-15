## Why

Вехе M3 нужна обучаемая модель, закрывающая сразу два требования: тема курса —
MLP, буква ТЗ — LSTM. Решение плана M3 (§5): обучить **обе** архитектуры за
общим контрактом туловища «окно + зона → представление R», с общими
multi-task головками — честное сравнение «наив-1 / наив-2 / MLP / LSTM» в
`m3-inference`. Предпосылки готовы: `m3-dataset` даёт окна `(steps, 17)`,
таргеты `(n, 2)` по горизонтам 15/30 и `DatasetScalers`; `m3-baselines` —
эталон сравнения. Этот change заводит архитектуру и loss; цикл обучения и
артефакты — `m3-train-cli`, запись `forecasts` — `m3-inference`.

## What Changes

- Новый модуль `ml/model.py`:
  - контракт туловища: нормализованное окно `(steps, N_FEATURES)` + `zone_id`
    → представление `R` фиксированной размерности (дефолт 128);
  - `MLPTrunk` (flatten + эмбеддинг зоны → Dense 256 → Dropout → Dense 128,
    основная архитектура) и `LSTMTrunk` (эмбеддинг зоны на каждом шаге →
    LSTM(64) → последний h → Dense 128, сравнение);
  - `MultiTaskNet`: общие головки `Linear(R→2)` ×3 — спрос (+15/+30), сток
    (+15/+30), дефицит-логиты (+15/+30, **одна BCE-головка с двумя выходами**
    под контракт `forecasts`);
  - multi-task loss: MSE(спрос) + MSE(сток) + `ml.cls_weight` × BCE с
    `pos_weight` по train-доле дефицитных бакетов;
  - обучаемый эмбеддинг зоны `nn.Embedding(Z, ml.zone_embed_dim)`;
  - детерминизм: фиксация torch/numpy по `ml.seed` — один seed даёт
    идентичные инициализации и шаги обучения.
- Новые тесты `tests/test_ml_model.py`: forward-формы обеих архитектур,
  smoke-обучение на синтетике (loss убывает), детерминизм двух запусков.
- Конфиг не меняется: все гиперпараметры (`mlp_hidden`, `lstm_hidden`,
  `lstm_layers`, `dropout`, `cls_weight`, `zone_embed_dim`, `seed`) заведены
  `m3-dataset` (приложение А плана M3).
- Не входит: цикл обучения с ранней остановкой и артефакты (`m3-train-cli`),
  walk-forward и запись `forecasts` (`m3-inference`), новых зависимостей нет.

## Capabilities

### New Capabilities

- `ml-model`: архитектуры MLP и LSTM за общим контрактом туловища, multi-task
  головки на три выхода контракта `forecasts`, взвешенный loss и
  seed-детерминизм инициализации и шагов обучения.

### Modified Capabilities

(нет — `ml-dataset` и `ml-baselines` не меняются; их контракты
`FEATURE_MANIFEST`/`Examples`/`DatasetScalers` используются как есть.)

## Impact

- Код: новый `ml/model.py`, новый `tests/test_ml_model.py`; правок
  существующих файлов нет.
- Зависимости: torch (уже в `.venv`), `ml.dataset` (`FEATURE_MANIFEST`,
  `N_FEATURES`, `RET_HORIZONS_MIN`, `Examples`, `DatasetScalers`),
  `common.config.MlCfg`.
- Влияние на последующие changes: `m3-train-cli` строит цикл обучения поверх
  `MultiTaskNet`/loss; `m3-inference` прогоняет `MultiTaskNet` по окнам и
  пишет `forecasts` с `model_version={arch}-v1-<дата>`.
