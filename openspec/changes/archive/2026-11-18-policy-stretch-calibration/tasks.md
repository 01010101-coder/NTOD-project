# Tasks: policy-stretch-calibration

Решения — design.md.

## 1. Формула

- [x] 1.1 `simulator/policy.py`: `effective_prob(p, δ)` — сдвиг при δ ≥ 0, растяжение при δ < 0, граница δ ≤ −1; `stair_multiplier` через неё.
  Verify: `tests/test_sim_policy.py` — прежние сценарии зелёные, новый `test_negative_shift_keeps_upper_stairs_reachable` (p = 1 → ×1.50, ступени монотонны, все три встречаются).
- [x] 1.2 `simulator/policy_calibration.py`: `fire_fraction` считает через `effective_prob`.
  Verify: `tests/test_sim_policy_calibration.py` зелёный.

## 2. Калибровка и миры

- [x] 2.1 `python -m simulator.policy_calibration` → `shift: {mlp: -0.719, lstm: -0.688}` в `config.yaml`; те же значения — `_DEFAULT_SHIFT` в `common/config.py`.
  Verify: `tests/test_config.py`, `tests/test_sim_city.py` (конфиг = дефолты).
- [x] 2.2 Пересобраны `data/worlds/c-mlp.db` и `d-lstm.db`.
  Verify: распределение `price_multiplier` тест-недели — у C встречаются ×1.15, ×1.30, ×1.50; у D — ×1.15, ×1.30.
