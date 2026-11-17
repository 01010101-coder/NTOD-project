# Tasks: ml-harm-deficit-label

Решения — design.md. После каждой группы `make lint && make test` должны быть зелёными. Существующие тесты при дефолте `legacy` правок не требуют.

## 1. Конфиг ml.deficit_label

- [x] 1.1 `common/config.py`: поле `MlCfg.deficit_label: Literal["legacy", "harm"] = "legacy"`, docstring секции дополнен.
  Verify: тесты в `tests/test_config.py` — конфиг без ключа даёт `legacy`; `harm` принимается; `loss` отклоняется ошибкой валидации с путём `ml.deficit_label`.
- [x] 1.2 `config.yaml`: ключ `deficit_label: legacy` в секции `ml` с комментарием (`legacy` | `harm`, ссылка на change).
  Verify: `load_config('config.yaml').ml.deficit_label == "legacy"`; `test_reference_run_bitwise_compatibility` зелёный.

## 2. Правило метки и таргеты датасета

- [x] 2.1 `ml/dataset.py`: тип `DeficitRule`, параметр `rule="legacy"` у `deficit_label`; `harm` → `unfulfilled > 0`; неизвестное значение → `ValueError`.
  Verify: тесты в `tests/test_ml_dataset.py` — (10, 3, 0): `legacy` True, `harm` False; (4, 6, 1): оба True; `rule="loss"` → `ValueError`; вызов без `rule` совпадает с `legacy`.
- [x] 2.2 `build_targets` и `collect_examples`: именованный параметр `label_rule="legacy"`, проброс в `deficit_label`.
  Verify: тест «метка harm игнорирует соотношение» из спеки (ts+15: free 1, requests 10, unfulfilled 0 → 0; ts+30: unfulfilled 2 → 1); окна, `requests` и `free_drivers` при `harm` и `legacy` совпадают поэлементно.

## 3. Обучение и артефакты

- [x] 3.1 `ml/train.py`: `collect_examples(..., label_rule=ml_cfg.deficit_label)`; поле `deficit_label` в `metrics.json`. Снимок ml-секции в `weights.pt` содержит ключ автоматически.
  Verify: тест в `tests/test_ml_train.py` на малой синтетической БД с `harm`: `metrics.json["deficit_label"] == "harm"`, снимок содержит `deficit_label: harm`, `pos_weight` равен отношению негативов к позитивам по меткам `harm` train-роли.
- [x] 3.2 Совместимость старых артефактов: `MlCfg(**snapshot)` без ключа даёт `legacy`.
  Verify: тест в `tests/test_ml_infer.py` — восстановление сети из снимка, где удалён `deficit_label`, успешно, `artifact.ml_cfg.deficit_label == "legacy"`.

## 4. Наивы, метрики и инференс

- [x] 4.1 `ml/baselines.py`: `naive_persist(..., *, label_rule="legacy")` считает `deficit_prob` по правилу.
  Verify: тест в `tests/test_ml_baselines.py` «Наив по вреду» (10, 3, 0 при `harm` → `deficit_prob = 0.0` на обоих горизонтах); существующий сценарий `legacy` без изменений.
- [x] 4.2 `ml/metrics.py`: `join_facts(..., *, label_rule="legacy")` и `build_summary(..., label_rule=...)`; `fact_deficit` — по правилу.
  Verify: тест «Факт по вреду»: прогноз 0.9 при факте (free/requests < 0.5, unfulfilled 0) и `harm` даёт FP, а при `legacy` — TP; таблица калибровки при `harm` считает долю сбывшихся по вреду.
- [x] 4.3 `ml/infer.py`: `collect_role_examples` передаёт правило снимка артефакта; `walk_forward_naives` и `summarize_run` получают правило живого конфига `cfg.ml.deficit_label`; при расхождении правила снимка артефакта и конфига — предупреждение в stderr с версией и обоими правилами, работа продолжается.
  Verify: тест в `tests/test_ml_infer.py` — артефакт `legacy` при конфиге `harm`: предупреждение в stderr (capsys), сводная построена, F1 считается по `harm`-факту.

## 5. Сервер и политика

- [x] 5.1 `app/server.py`: `actual_deficit` по `cfg.ml.deficit_label`.
  Verify: тест в `tests/test_web_ml_tab.py` «Факт дефицита по вреду» — сервер с `harm`, факт-бакет (free 1, requests 10, unfulfilled 0) → `actual_deficit is False`; при `legacy` → `True`.
- [x] 5.2 `simulator/policy.py`: `has_deficit` вызывает `deficit_label(..., rule="legacy")` явно, docstring объясняет независимость от `ml.deficit_label`.
  Verify: тест в `tests/test_sim_policy.py` «Правило не зависит от метки моделей» — при конфиге `harm` строка (free 1, requests 10, unfulfilled 0) даёт `reactive_mult`.
- [x] 5.3 Показатели мира не зависят от ключа.
  Verify: тест (в `tests/test_sim_loop.py` или `tests/test_ml_dataset.py`) — короткий прогон движка одним seed при `legacy` и `harm`: `trips`, `zone_metrics_5min` и `Summary.deficit_share` совпадают.

## 6. Проверка на реальной эпохе

- [x] 6.1 Скрипт во временном каталоге (не в `data/`): на `epoch.db` мира n=240 из `data/start_conditions/` собрать примеры `collect_examples` при `legacy` и `harm`.
  Verify: доля позитивов `harm` на горизонте 15 ≈ 10 % (доля бакетов с `unfulfilled > 0`), `legacy` ≈ 32 %; число примеров и окна совпадают. Числа записаны в итоговый отчёт apply.

## 7. Финальная проверка

- [x] 7.1 `make lint` проходит без ошибок.
- [x] 7.2 `make test` проходит полностью, включая `test_reference_run_bitwise_compatibility`.
