# Proposal: m3-train-cli

## Why

Веха M3 имеет датасет (`ml-dataset`), бейзлайны (`ml-baselines`) и архитектуры
с loss (`ml-model`), но обучить сеть пока нечем: нет цикла обучения, CLI и
версионируемых артефактов. `m3-train-cli` — четвёртый change вехи по плану
M3 (§8, `docs/plan-m3-ml.md`): превращает «всё готово для обучения» в
«обученная модель с артефактами», включая экспериментальное подтверждение
дефолта окна истории (свип 3/6/12 на MLP, §5 плана).

## What Changes

- Новый цикл обучения `ml/train.py`: сборка примеров из БД эпохи (роли
  train/val `ml-dataset`), fit scaler'ов на train, обучение выбранной
  архитектуры (`ml-model`: `build_model`/`MultitaskLoss`/`deficit_pos_weight`/
  `set_global_seed`) с ранней остановкой по val-loss (`ml.patience`, кап
  `ml.max_epochs`) — переиспользование существующих API без их доработок.
- Версионированные артефакты на модель в `artifacts/{model_version}/`:
  `weights.pt` (state_dict + архитектура + снимок `ml:`-конфига), `scalers.pkl`
  (fit этой же сборки), `loss.json` (компоненты loss по эпохам train/val),
  `metrics.json` (итог обучения); `model_version = {arch}-v1-{дата}`.
- CLI `python -m ml train --config … [--arch mlp|lstm|both] [--db …]`
  (формат вывода — приложение Б плана M3) и подкоманда свипа окна
  `python -m ml sweep --config …`: три обучения MLP при `window_hours`
  3/6/12, критерий — денормализованный val MAE спроса, результат —
  `artifacts/window-sweep.json` и победитель; фиксация `ml.window_hours`
  в `config.yaml` по победителю — шаг задачи (не кодом).
- Тесты `tests/test_ml_train.py`: smoke-обучение на синтетической мини-БД,
  ранний стоп, детерминизм двух запусков по `ml.seed`, roundtrip артефактов,
  CLI (форматы вывода, exit-коды).

Без изменений поведения существующих модулей: `ml/dataset.py`, `ml/model.py`,
`ml/baselines.py`, `ml/metrics.py`, конфиг (все ключи обучения заведены
`m3-dataset`) и Makefile не правятся.

## Capabilities

### New Capabilities

- `ml-training`: цикл обучения обеих архитектур с ранней остановкой,
  seed-детерминизм, версионированные артефакты моделей, CLI обучения и свипа
  окна истории.

### Modified Capabilities

(нет — требования `ml-dataset`/`ml-model`/`ml-baselines` не меняются)

## Impact

- Новые файлы: `ml/train.py`, `ml/__main__.py`, `tests/test_ml_train.py`.
- Артефакты в `artifacts/` (gitignored): каталоги моделей и
  `window-sweep.json` — появляются при запуске обучения, в git не попадают.
- `config.yaml`: единственная возможная правка по итогам свипа — значение
  `ml.window_hours` (задача с выполнением, отдельный шаг).
- Следующий change `m3-inference` потребляет артефакты этого: каталог
  `artifacts/{model_version}/` — контракт между обучением и инференсом.
- Новых зависимостей нет: torch/numpy/pandas/scikit-learn уже в `.venv`.
