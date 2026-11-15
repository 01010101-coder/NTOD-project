## 1. Скелет модуля и туловища

- [x] 1.1 Создать `ml/model.py`: константа `TRUNK_OUTPUT_DIM = 128`, функция
  `set_global_seed(seed)` (torch + numpy), фабрика `build_model(arch, cfg,
  n_zones)` с `arch ∈ {"mlp", "lstm"}`; добавить модуль в
  `tests/test_imports.py` — проверка: импорт без ошибок, тест импортов зелёный
- [x] 1.2 Реализовать MLP-туловище (flatten окна + конкат эмбеддинга зоны →
  плотные слои по `ml.mlp_hidden` с ReLU/Dropout → R, решение 2 design.md;
  `mlp_hidden[-1] ≠ TRUNK_OUTPUT_DIM` → `ValueError`) — проверка: тесты
  forward `(n, steps, 17) + зоны (n,) → (n, 128)` на дефолтном конфиге и
  ожидаемый `ValueError` при `mlp_hidden=[256]`
- [x] 1.3 Реализовать LSTM-туловище (эмбеддинг зоны на каждом шаге →
  `nn.LSTM(ml.lstm_hidden, ml.lstm_layers, batch_first=True)` → последнее
  скрытое состояние → `Linear(lstm_hidden, TRUNK_OUTPUT_DIM)`, решение 3
  design.md) — проверка: тест forward `(n, steps, 17) + зоны → (n, 128)`,
  в том числе при `lstm_layers = 2`

## 2. MultiTaskNet и loss

- [x] 2.1 Реализовать `MultiTaskNet` вокруг туловища: общие головки
  `Linear(128, 2)` ×3 и выход `ModelOutput` (`pred_requests`,
  `pred_free_drivers`, `deficit_logits`, `deficit_prob = sigmoid`) —
  проверка: параметризованный тест обеих архитектур (формы `(n, 2)` ×3,
  вероятности в `[0, 1]`; одинаковое окно с разными `zone_id` даёт разные
  выходы; замена туловища не меняет формы — сценарий взаимозаменяемости)
- [x] 2.2 Реализовать `MultitaskLoss` (MSE + MSE + `cls_weight` × BCEWithLogits
  с `pos_weight`) с раздельными компонентами `LossComponents(total,
  mse_requests, mse_free, bce)` и функцию `deficit_pos_weight` (решения 5
  design.md) — проверка: тесты на рукописной синтетике — идеальные прогнозы
  регрессий дают нулевые MSE; `cls_weight = 0` исключает BCE из total;
  `pos_weight = (1 − p) / p` на заданной доле; доли 0 и 1 → `ValueError`

## 3. Smoke-обучение, детерминизм, совместимость с датасетом

- [x] 3.1 Smoke-обучение на синтетике: параметризованный тест обеих
  архитектур — несколько шагов Adam по фиксированному батчу с закономерной
  зависимостью таргетов от входа — проверка: loss после шагов строго меньше
  исходного (сценарий спеки «loss убывает»)
- [x] 3.2 Детерминизм seed: `set_global_seed` перед построением — проверка:
  тесты «один seed → идентичные начальные веса (`state_dict`) и идентичные
  значения loss на каждом шаге», «разные seed → разные веса» (сценарии спеки)
- [x] 3.3 Совместимость с `ml-dataset` без переупаковки: на рукописной
  мини-БД (приём `tests/test_ml_dataset.py`) собрать `Examples`, нормализовать
  `DatasetScalers`, конвертировать numpy → `float32`/`int64` тензоры,
  прогнать `forward` + `MultitaskLoss` — проверка: тест зелёный (сценарий
  «вход совместим с примерами датасета»)

## 4. Ворота

- [x] 4.1 Прогон `make lint && make test` — оба зелёные (последняя группа
  задач правила конвейера)
