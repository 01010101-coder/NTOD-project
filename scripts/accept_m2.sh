#!/usr/bin/env bash
# Приёмка вехи M2 «Движок» — дословно из docs/plan-m2-engine.md, приложение В.
# Запуск: make accept-m2  (эквивалент: bash scripts/accept_m2.sh)
# Вывод: PASS/FAIL по критериям A1–A13 (таблица — §11 плана); exit-код = число провалов.

set -u
cd "$(dirname "$0")/.."

PY=.venv/bin/python
DB_A=data/accept-a.db
DB_B=data/accept-b.db
DB_C=data/accept-c.db
fails=0

ok()  { echo "PASS $1 — $2"; }
bad() { echo "FAIL $1 — $2"; fails=$((fails + 1)); }

rm -f "$DB_A" "$DB_A"-* "$DB_B" "$DB_B"-* "$DB_C" "$DB_C"-* data/taxi.db data/taxi.db-*
rm -rf data/accept-a data/accept-b

# --- A1: линт; A2: тесты --------------------------------------------------
if make lint > /dev/null 2>&1; then ok A1 "make lint"; else bad A1 "make lint"; fi
if make test > /dev/null 2>&1; then ok A2 "make test"; else bad A2 "make test"; fi

# --- A3: состав пакета simulator/ ровно из приложения Б --------------------
# Учитываются только *.py: служебные __pycache__/.DS_Store (появляются после make test)
# состав модулей не меняют и проверку ронять не должны.
expected="__init__.py __main__.py aggregate.py calibrate.py city.py demand.py drivers.py engine.py env.py"
actual=$(ls simulator/*.py | sed 's|simulator/||' | sort | tr '\n' ' ' | sed 's/ $//')
if [ "$actual" = "$expected" ]; then
    ok A3 "состав simulator/ = приложение Б"
else
    bad A3 "состав simulator/ = '$actual', ожидалось '$expected'"
fi

# --- A4: генерация 30 дней: exit 0, сводка 4 строки §4, wall <= 60 с ------
"$PY" -m simulator --days 30 --seed 42 --db "$DB_A" --csv data/accept-a \
    > /tmp/m2_summary.txt 2> /tmp/m2_progress.txt
rc=$?
wall=$(sed -n 's/^wall: \([0-9.]*\)s$/\1/p' /tmp/m2_summary.txt)
prog=$(grep -cE '^\[20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}:00\] trips=[0-9]+ elapsed=[0-9.]+s$' /tmp/m2_progress.txt)
if [ "$rc" -eq 0 ] \
   && [ "$(wc -l < /tmp/m2_summary.txt | tr -d ' ')" = "4" ] \
   && head -n 1 /tmp/m2_summary.txt | grep -qE '^sim: days=30 seed=42 db=data/accept-a\.db$' \
   && sed -n 2p /tmp/m2_summary.txt | grep -qE '^trips: total=[0-9]+ completed=[0-9]+ completed_share=[0-9.]+% unfulfilled=[0-9]+ unfulfilled_share=[0-9.]+%$' \
   && sed -n 3p /tmp/m2_summary.txt | grep -qE '^wait: avg_wait_min=[0-9.]+ deficit_share=[0-9.]+%$' \
   && [ -n "$wall" ] && awk -v w="$wall" 'BEGIN { exit !(w <= 60) }' \
   && [ "$prog" -ge 24 ]
then
    ok A4 "генерация 30 дней: сводка §4, wall=${wall}s"
else
    bad A4 "генерация 30 дней (rc=$rc, wall='${wall:-нет}', прогресс=$prog строк)"
fi

# --- A5: наполнение БД ------------------------------------------------------
n_metrics=$(sqlite3 "$DB_A" 'select count(*) from zone_metrics_5min')
n_ts=$(sqlite3 "$DB_A" 'select count(distinct ts) from zone_metrics_5min')
n_bad_zones=$(sqlite3 "$DB_A" 'select count(*) from zone_metrics_5min where zone_id < 0 or zone_id > 5')
n_trips=$(sqlite3 "$DB_A" 'select count(*) from trips')
n_completed=$(sqlite3 "$DB_A" "select count(*) from trips where status = 'completed'")
n_unfulf=$(sqlite3 "$DB_A" "select count(*) from trips where status = 'unfulfilled'")
n_other=$(sqlite3 "$DB_A" "select count(*) from trips where status not in ('completed', 'unfulfilled')")
n_fc=$(sqlite3 "$DB_A" 'select count(*) from forecasts')
if [ "$n_metrics" = "51840" ] && [ "$n_ts" = "8640" ] && [ "$n_bad_zones" = "0" ] \
   && [ "$n_trips" -gt 0 ] && [ "$n_completed" -gt 0 ] && [ "$n_unfulf" -gt 0 ] \
   && [ "$n_other" = "0" ] && [ "$n_fc" = "0" ]
then
    ok A5 "наполнение БД (trips=$n_trips, метрик=$n_metrics)"
else
    bad A5 "наполнение БД (trips=$n_trips, метрик=$n_metrics, ts=$n_ts)"
fi

# --- A6: инварианты данных --------------------------------------------------
n_inv=$(sqlite3 "$DB_A" "
select count(*) from trips where
    (assigned_at is not null and assigned_at < created_at)
 or (pickup_at is not null and pickup_at < assigned_at)
 or (dropoff_at is not null and dropoff_at < pickup_at)
 or (status = 'completed' and (assigned_at is null or pickup_at is null or dropoff_at is null))
 or (status = 'unfulfilled' and assigned_at is not null)
 or created_at < '2026-11-02 00:00:00'
 or created_at >= '2026-12-02 00:00:00'")
n_ratio=$(sqlite3 "$DB_A" '
select count(*) from zone_metrics_5min
 where abs(supply_demand_ratio - free_drivers * 1.0 / max(requests, 1)) > 1e-9')
if [ "$n_inv" = "0" ] && [ "$n_ratio" = "0" ]; then
    ok A6 "инварианты данных (времена, статусы, ratio)"
else
    bad A6 "инварианты (нарушений: $n_inv, ratio: $n_ratio)"
fi

# --- A7: детерминизм seed 42, отличимость seed 43 ---------------------------
"$PY" -m simulator --days 30 --seed 42 --db "$DB_B" > /dev/null 2>&1
rc_b=$?
"$PY" -m simulator --days 3 --seed 43 --db "$DB_C" > /dev/null 2>&1
rc_c=$?
ha_t=$(sqlite3 "$DB_A" 'select * from trips order by trip_id' | shasum | awk '{print $1}')
hb_t=$(sqlite3 "$DB_B" 'select * from trips order by trip_id' | shasum | awk '{print $1}')
ha_m=$(sqlite3 "$DB_A" 'select * from zone_metrics_5min order by ts, zone_id' | shasum | awk '{print $1}')
hb_m=$(sqlite3 "$DB_B" 'select * from zone_metrics_5min order by ts, zone_id' | shasum | awk '{print $1}')
pa=$(sqlite3 "$DB_A" 'select * from trips order by trip_id limit 5000' | shasum | awk '{print $1}')
pc=$(sqlite3 "$DB_C" 'select * from trips order by trip_id limit 5000' | shasum | awk '{print $1}')
if [ "$rc_b" -eq 0 ] && [ "$rc_c" -eq 0 ] \
   && [ "$ha_t" = "$hb_t" ] && [ "$ha_m" = "$hb_m" ] && [ "$pa" != "$pc" ]
then
    ok A7 "детерминизм seed 42; отличимость seed 43"
else
    bad A7 "детерминизм (rc_b=$rc_b, rc_c=$rc_c)"
fi

# --- A8: guard на непустую БД ------------------------------------------------
"$PY" -m simulator --days 1 --db "$DB_A" > /tmp/m2_guard.txt 2>&1
rc_g=$?
if [ "$rc_g" -eq 2 ] && grep -q "trips" /tmp/m2_guard.txt; then
    ok A8 "guard непустой БД (exit=2)"
else
    bad A8 "guard непустой БД (rc=$rc_g)"
fi

# --- A9: CSV = таблицы (+заголовок) ------------------------------------------
lc_t=$(grep -c '' data/accept-a/trips.csv)
lc_m=$(grep -c '' data/accept-a/zone_metrics_5min.csv)
if [ -f data/accept-a/trips.csv ] && [ -f data/accept-a/zone_metrics_5min.csv ] \
   && [ "$lc_t" = "$((n_trips + 1))" ] && [ "$lc_m" = "$((n_metrics + 1))" ]
then
    ok A9 "CSV = таблицы (+заголовок)"
else
    bad A9 "CSV (строк: $lc_t/$lc_m, в БД: $n_trips/$n_metrics)"
fi

# --- A10: make sim -------------------------------------------------------------
make sim > /dev/null 2>&1
rc_s=$?
if [ "$rc_s" -eq 0 ] && [ -f data/taxi.db ]; then ok A10 "make sim"; else bad A10 "make sim (rc=$rc_s)"; fi

# --- A11: config.yaml = канон §6 ------------------------------------------------
if "$PY" - <<'EOF'
from common.config import load_config

cfg = load_config("config.yaml")
assert len(cfg.city.zones) == 6
assert len(cfg.city.roads) == 7
assert cfg.demand.base_rate > 0
assert set(cfg.demand.profiles) == {"residential", "business", "airport", "mixed"}
for prof in cfg.demand.profiles.values():
    assert len(prof["weekday"]) == 24 and len(prof["weekend"]) == 24
assert cfg.engine.sim_start == "2026-11-02"
assert cfg.weather.travel_mult > 1.0 and cfg.weather.demand_mult > 1.0
assert 0.0 < cfg.pricing.surge_injection.day_chance <= 1.0
EOF
then ok A11 "config.yaml = канон §6"; else bad A11 "config.yaml = канон §6"; fi

# --- A12: публичный API = приложение Б -------------------------------------------
if "$PY" - <<'EOF'
from simulator.aggregate import AggregateBuffer
from simulator.city import CitySpec, HourProfile, ZoneInfo, build_city
from simulator.demand import TripRequest, lambda_rate, spawn_requests
from simulator.drivers import Driver, DriverState, Fleet
from simulator.engine import Engine, Summary, TripWriter
from simulator.env import WorldEnv

assert callable(build_city) and callable(lambda_rate) and callable(spawn_requests)
EOF
then ok A12 "публичный API = приложение Б"; else bad A12 "публичный API = приложение Б"; fi

# --- A13: калибровка §9 ------------------------------------------------------------
"$PY" -m simulator.calibrate --days 30 --seed 42 > /tmp/m2_calib.txt 2>&1
rc_k=$?
deficit=$(sed -n 's/.*deficit_share=\([0-9][0-9.]*\)%.*/\1/p' /tmp/m2_calib.txt | tail -n 1)
n_png=$(ls reports/*.png 2> /dev/null | wc -l | tr -d ' ')
if [ "$rc_k" -eq 0 ] && [ -n "$deficit" ] \
   && awk -v d="$deficit" 'BEGIN { exit !(d >= 10 && d <= 20) }' \
   && [ "$n_png" -ge 3 ] \
   && [ -f docs/m2-calibration.md ] && grep -q "deficit_share=" docs/m2-calibration.md
then
    ok A13 "калибровка (deficit_share=${deficit}%)"
else
    bad A13 "калибровка (deficit='${deficit:-нет}', png=$n_png)"
fi

# --- итог ----------------------------------------------------------------------------
echo ""
if [ "$fails" -eq 0 ]; then
    echo "ПРИЁМКА M2 ПРОЙДЕНА: A1–A13 зелёные"
else
    echo "ПРИЁМКА M2 ПРОВАЛЕНА: провалов — $fails"
fi
exit "$fails"
