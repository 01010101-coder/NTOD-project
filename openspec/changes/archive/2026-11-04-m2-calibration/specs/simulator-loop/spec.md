## MODIFIED Requirements

### Requirement: Заготовка calibrate (строка №5 §7)

Модуль `simulator/calibrate.py` (запуск `python -m simulator.calibrate --days 30 [--seed S]`) SHALL выполнять прогон мира и печатать итог машинной строкой `deficit_share=<X>%` (§9). С change'а `m2-calibration` заготовка докручивается до полного инструмента калибровки: прогон, графики в `reports/` (не меньше трёх PNG), подбор параметров мира и отчёт `docs/m2-calibration.md`; поведенческие требования калибровки определены capability `simulator-calibration`, а оговорка «графики/PNG, подбор параметров и отчёт — scope будущей калибровки» снимается: критерий A13 закрывается `m2-calibration` и больше не является ожидаемым провалом гейта приёмки.

#### Scenario: calibrate печатает deficit_share

- **WHEN** выполняется `python -m simulator.calibrate --days 30 --seed 42`
- **THEN** модуль завершается с exit 0 и печатает строку `deficit_share=<X>%` (доля дефицитных 5-минуток по правилу §3.8 в процентах), согласованную с данными прогона

#### Scenario: Гейт A13 закрыт калибровкой

- **WHEN** на репозитории с реализованным `m2-calibration` выполняется `make accept-m2`
- **THEN** критерий A13 напечатан с `PASS` (`deficit_share` ∈ [10, 20], ≥ 3 PNG в `reports/`, `docs/m2-calibration.md` содержит `deficit_share=`); все A1–A13 — `PASS`, exit-код скрипта 0, веха M2 закрыта (§11.3; остаются только ручные пункты §11.2)
