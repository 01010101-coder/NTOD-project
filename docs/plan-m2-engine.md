# План M2 «Движок» (симулятор города)

- Статус: черновик для согласования; уточнён 2026-11-03 — зафиксированы результат (§1.1), запреты (§2.2), канонический конфиг (§6), структура пакета и API (приложение Б), приёмка руками (приложение В).
- Веха: M2 из последовательности M1 Каркас → **M2 Движок** → M3 ML → M4 Сервер воспроизведения → M5 Интерфейс → M6 Вмешательство.
- Предпосылки: M1 закрыт (`docs/plan-m1-foundation.md`): пакеты, конфиг, хранилище, тесты, CI.
- Документ самодостаточен; контракт данных — `openspec/config.yaml`.
- Как читать исполнителю: механика мира — §3–§5; **приложения А, Б, В обязательны дословно** — значения конфига, состав файлов, имена и сигнатуры API, формат вывода не меняются по усмотрению исполнителя. Расхождение или неясность — вопрос в чейндже, а не самостоятельное решение.

## 1. Цель вехи

Полный **детерминированный** симулятор городской жизни и CLI быстрой генерации истории:

- модель города: зоны-архетипы, дорожный граф, суточные профили спроса;
- сущности: заказы (пуассоновский спрос, эластичность по тарифу), водители (жадный диспатч), поездки с метками времени;
- среда: дождь, «пятничный вечер», тариф с обучающими surge-инъекциями;
- минутный цикл в фиксированном порядке → агрегация в `zone_metrics_5min` каждые 5 минут;
- CLI: сгенерировать N дней истории в SQLite (+CSV) за десятки секунд;
- откалиброванный эталонный конфиг (цель калибровки — §9).

Таблица `forecasts` в M2 не заполняется — её пишет ML (M3).

### 1.1 Картина конца: что конкретно должно получиться

Когда M2 закрыт, воспроизводимо существует следующее — и только оно:

1. Пакет `simulator/` состоит ровно из файлов приложения Б (`city.py`, `env.py`, `demand.py`, `drivers.py`, `aggregate.py`, `engine.py`, `__main__.py`, `calibrate.py` + `__init__.py`). Публичные имена и сигнатуры — в точности как в Б.
2. Команда `.venv/bin/python -m simulator --days 30 --seed 42 --db data/taxi.db` отрабатывает за ≤ 60 с, печатает прогресс раз в сим-час и сводку формата §4. `make sim` делает то же самое.
3. В указанной БД: `trips` заполнена (порядка 0,5–1 млн строк на 30 дней), `zone_metrics_5min` содержит ровно `дней × 288 × 6` строк, `forecasts` пустая. `--csv DIR` кладёт рядом `trips.csv` и `zone_metrics_5min.csv`.
4. Детерминизм проверяется командой из приёмки: два прогона одного seed в разные файлы БД дают одинаковые sha-хэши обеих таблиц (критерий A7, §11; скрипт — приложение В).
5. `config.yaml` в репо — канонический мир приложения А; расхождения со скелетом M1 сняты по §6 и закрыты первым change'ом.
6. Все тесты §8 зелёные, `make lint && make test` проходит.
7. `docs/m2-calibration.md` с графиками в `reports/`: дефицитных 5-минуток 10–20 %, финальные числа внесены в `config.yaml`.

Приёмка вехи — `make accept-m2`: скрипт с критериями A1–A13 (приложение В, таблица — §11) плюс ручные пункты §11.2. Если приёмка не проходит на свежем клоне без правок кода — веха не закрыта.

## 2. Границы

### 2.1 Не входит

(Осознанные упрощения v1; кандидаты в будущие вехи): релокация водителей к спросу, реакция водителей на тариф, пробки/ёмкость рёбер, смены и оффлайн водителей, отмена заказа пассажиром (статус `cancelled` из контракта не используется), сервер и UI.

**Ключевое свойство** — детерминизм: один `seed` → побайтово одинаковые таблицы при повторном прогоне на той же машине. На нём держатся прыжок в произвольную точку времени (M4) и парные A/B-эксперименты (M6).

### 2.2 Чего не делать (запреты для исполнителей)

Эти правила снимают соблазн «улучшить» план своими решениями. Нарушение любого — переделка change'а.

1. **Не изобретать структуру.** Только файлы приложения Б. Никаких абстрактных базовых классов, плагинов, шин событий, «стратегий», фабрик; никаких дополнительных модулей и подпакетов.
2. **Не добавлять зависимости.** В движке — numpy и stdlib. pandas разрешён только в `calibrate.py`; matplotlib — только там же.
3. **Никакого параллелизма и wall-clock времени.** Без asyncio, потоков, `datetime.now()`/`time.*` внутри симуляции. Единственное время — сим-часы (§3), передаваемые явно.
4. **Переиспользовать M1 по именам**, не создавая дублей: `common.config.load_config`, `common.timeutil.{to_iso, from_iso, add_minutes, bucket5}`, `storage.db.{connect, transaction}`, `storage.repo.{insert_trips, insert_zone_metrics}`, `storage.csv_export.export_table`.
5. **Конфиг-модели правятся один раз** и только как описано в §6 (change `m2-city`). Схема SQLite не меняется; `common/` и `storage/` кроме `common/config.py` не трогаются.
6. **Магических чисел в коде нет**: все параметры мира — из конфига; сами значения мира (зоны, дороги, профили) — только в `config.yaml`.
7. **Идентификаторы на английском, комментарии и docstring на русском** (правило `openspec/config.yaml`). Строки пользовательского вывода CLI — на русском или в оговорённом §4 машинном формате.
8. Непонятность в плане — вопрос в чейндже, а не молчаливое решение своим способом.

## 3. Модель мира

**Часы мира.** Симуляция стартует в `engine.sim_start` (эталон: понедельник `2026-11-02 00:00`) и идёт минутными шагами: `t = add_minutes(sim_start, номер_минуты)`. Нумерация дней недели — с понедельника = 0 (та же, что в M1). Сим-время — `datetime` из `common.timeutil`; секунды всегда 00.

### 3.1 Зоны и архетипы (эталон: 6 зон)

| id | имя | архетип | attract_origin | attract_dest | характер |
|----|-----|---------|----------------|--------------|----------|
| 0 | sleep-1 | residential | 1.0 | 0.6 | утренний отток, вечерний приток |
| 1 | sleep-2 | residential | 1.0 | 0.6 | то же |
| 2 | center | business | 1.3 | 1.8 | впитывает утром, выплёвывает вечером, магнит вечером |
| 3 | office | business | 1.2 | 1.5 | рабочий день |
| 4 | airport | airport | 0.6 | 1.2 | ровный фон |
| 5 | lounge | mixed | 1.1 | 1.4 | тусовочный: поздний вечер |

У каждой зоны в конфиге также `map: {x, y}` — координаты для карты (M5); движок их не читает (в `ZoneInfo` не попадают).

### 3.2 Дорожный граф и время поездок

Рёбра двунаправленные с весами-минутами (приложение А), граф связный. При старте считается матрица кратчайших путей (Флойд на 6 узлах) → `path_min[A][B]`.
Время поездки = `ceil(path_min[A][B] × k_дождь)`, минимум 1 минута; время подачи = так же для `path_min[зона водителя][зона посадки]`. Эталонная длительность поездки — 5–15 минут.

### 3.3 Спрос: мультипликативная λ + Пуассон

Каждую сим-минуту для каждой зоны (в порядке возрастания `zone_id`, разыгрывание всегда, даже при λ=0):

```
λ(зона, t) = base_rate
           × профиль_часа(архетип, час, будни/выходные)
           × attract_origin(зоны)
           × population.scale
           × (дождь? × rain.demand_mult)
           × (пятничный вечер для архетипа? × events.friday_evening.demand_mult)
           × m(зоны, t) ^ (−pricing.elasticity_alpha)     ← эластичность
```

`rng.poisson(λ)` даёт число новых пассажиров. Каждый пассажир сразу создаёт заказ и выбирает назначение:

```
P(B | A) ∝ attract_dest(B) × exp(−path_min[A][B] / demand.dest_decay_min),  B ≠ A
```

Эталон: `base_rate` ≈ 2.0 заказов/мин на зону до профилей (город ~10–15/мин днём), калибруется (§9).

### 3.4 Заказы

Заказ рождается с `created_at` и `price_multiplier` = множитель зоны происхождения на момент создания. Ожидает назначения не дольше `drivers.wait_limit_min` (эталон 6 мин); не назначенный за лимит → `status='unfulfilled'`, `assigned_at` NULL. Назначенный проходит `assigned_at → pickup_at → dropoff_at`, `status='completed'`. Выручка не хранится: восстанавливается post-hoc как `базовая_цена(маршрут) × price_multiplier` (формула — §10.15).

### 3.5 Водители и диспатч

Парк: `drivers.count` (эталон 200). Старт — детерминированно без rng: водитель `i` в зоне `i mod Z`. Состояния: `free(зона)`, `to_pickup(заказ, осталось_мин)`, `on_trip(заказ, осталось_мин)`.

Диспатч раз в минуту: ожидающие заказы по возрастанию `(created_at, внутренний_id)`; каждому — свободный водитель с минимальным временем подачи, при равенстве — меньший `driver_id`. После высадки водитель свободен в зоне назначения (отсюда дрейф машин в привлекательные зоны — главный источник локальных дефицитов).

Упрощение: назначение возможно и если подача дольше лимита ожидания — диспатч «спасает» заказ от истечения. Зафиксировано, чтобы не плодить ветвление логики.

### 3.6 Среда

- **Дождь**: раз в день Bernoulli `weather.rain_day_chance` (эталон 0.15). Эпизод: старт U(6:00–22:00), длительность U(1–3) ч. Во время эпизода: спрос × `rain.demand_mult` (1.4), время поездок × `rain.travel_mult` (1.4).
- **«Пятничный вечер»**: календарно, дни из конфига (пт, сб), часы 17–23 включительно, для архетипов business/mixed — спрос × 1.3.
- **Тариф**: всюду 1.0, кроме **обучающих инъекций**: раз в день Bernoulli `surge_injection.day_chance` (0.10) → случайная зона, старт U(сутки), длительность U(30–60) мин, множитель U(1.2–1.6). Это единственный источник вариации `price_multiplier` в эпоху генерации — нужен, чтобы признак не был константой для ML (M3).

### 3.7 Минутный цикл (порядок фиксирован — часть контракта детерминизма)

1. **Среда**: переходы погоды, календарь, состояние тарифа.
2. **Спавн**: пуассоновские заказы по зонам (λ уже содержит эластичность), выбор назначения.
3. **Диспатч**: назначения по §3.5.
4. **Движение**: счётчики `to_pickup`/`on_trip` тикают; происходят посадки и высадки; водители меняют состояния и зону.
5. **Истечения**: не назначенные заказы старше лимита → `unfulfilled`.
6. **Каждая 5-я минута**: агрегат в `zone_metrics_5min` по семантике §3.8.

Вся случайность — через **один** `numpy.random.Generator` (`np.random.default_rng(seed)`), создаётся `Engine` и передаётся модулям; порядок разыгрываний фиксирован (шаги цикла; зоны по возрастанию id). Никаких итераций по множествам/словарям без сортировки. Никакого `datetime.now()` внутри движка — только сим-часы, передаваемые снаружи.

### 3.8 Семантика колонок агрегата

| Колонка | Семантика |
|---|---|
| `ts` | начало 5-минутного интервала |
| `requests` | создано заказов за интервал (счётчик) |
| `completed` | завершено поездок за интервал, `dropoff_at` внутри интервала (счётчик) |
| `unfulfilled` | заказов истекло за интервал (счётчик) |
| `free_drivers`, `busy_drivers` | снапшот на конец интервала (решение: снапшот, не среднее — просто и однозначно) |
| `avg_wait_min` | среднее `assigned_at − created_at` по заказам, **назначенным** в интервале; NULL, если таких нет |
| `supply_demand_ratio` | `free_drivers / max(requests, 1)` (снапшот/счётчик — так в контракте) |
| `is_rain` | 1, если дождь был в любой момент интервала |
| `event` | первое сработавшее: `'friday_evening'` \| `'surge_injection'` \| `''` |
| `price_multiplier` | среднее по времени значение множителя зоны за интервал (по-минутное среднее кусочно-постоянной функции) |

Правило дефицита (по контракту, вычисляется функцией, не хранится): `free_drivers / max(requests,1) < 0.5` **или** `unfulfilled > 0`.

## 4. CLI генерации

```
.venv/bin/python -m simulator --days N [--config config.yaml] [--seed S] [--db data/taxi.db] [--csv DIR]
```

- `--seed` и `--db` по умолчанию из конфига; флаги переопределяют. `--csv DIR` создаёт каталог и выгружает обе таблицы через `storage.csv_export.export_table` (файлы `trips.csv`, `zone_metrics_5min.csv`).
- **Guard**: если таблица `trips` в целевой БД уже содержит строки — сообщение об ошибке и exit code 2 («генерация требует пустой таблицы trips — укажите новый --db»). Причина: `trip_id` назначает SQLite при вставке (§10.11).
- Прогресс — раз в сим-час в stderr, машинный формат: `[2026-11-02 07:00] trips=8214 elapsed=2.1s`.
- Сводка — в stdout в конце, **ровно 4 строки** в формате `ключ=значение` без лишних символов (числа иллюстративны; `deficit_share` — доля дефицитных 5-минуток по правилу §3.8 в процентах, нужна калибровке §9; формат дословно проверяет скрипт приёмки, критерий A4):

```
sim: days=30 seed=42 db=data/taxi.db
trips: total=617204 completed=544870 completed_share=88.2% unfulfilled=72334 unfulfilled_share=11.7%
wait: avg_wait_min=3.42 deficit_share=14.8%
wall: 41.3s
```

- `Makefile` получает цель `sim` (change `m2-loop`): `.venv/bin/python -m simulator --days 30 --seed 42`.

## 5. Производительность

30 дней ≈ 43 200 минут × 6 зон, ~0,5–1 млн заказов. Бюджет: **30 дней ≤ 60 с** на машине разработки. Строки копятся в памяти и сбрасываются writer'у (приложение Б) раз в сим-час — вставка батчами через `storage.repo`, нового SQL в движке нет. Если дольше — векторизовать спавн (пуассоны всех зон одним вызовом numpy), не меняя порядок разыгрываний. Замер ручной, входит в DoD.

## 6. Канонический конфиг: разрешение расхождений со скелетом M1

M1 реализовал скелет конфига, часть ключей которого расходится с миром M2. Расхождение снимается **один раз** в change `m2-city`: правится `common/config.py`, `config.yaml` переписывается целиком под приложение А. Внешних конфигов, кроме эталонного в репо, не существует, обратная совместимость не требуется. После правки дефолты моделей равны каноническим значениям (пустой конфиг грузится и валиден).

| Секция | Скелет M1 (`common/config.py`) | Канон M2 | Почему |
|---|---|---|---|
| `city.zones` | `ZoneCfg(id, archetype, attractiveness, x, y)` | `ZoneCfg(id, name, archetype, attract_origin, attract_dest, map:{x,y})` | нужны отдельные веса источника и цели (§3.1, §3.3); координаты карты (M5) изолированы в `map` и движком не читаются |
| `city.roads` | `RoadCfg(from_zone, to_zone, minutes)` | без изменений | форма M1 сохранена осознанно |
| `demand` | `profiles: dict`, `dest_decay_min` | + `base_rate: float = 2.0`; структура `profiles` жёсткая: архетип → `{weekday: [24 float], weekend: [24 float]}` | `base_rate` — ядро λ (§3.3); свободная структура → самодеятельность |
| `weather` | `rain_chance_per_day, episode_duration_min, request_multiplier, driver_multiplier` | `rain_day_chance: 0.15, episode_start_h: [6,22], episode_duration_h: [1,3], demand_mult: 1.4, travel_mult: 1.4` | старт и длительность эпизода — случайные интервалы (§3.6); `driver_multiplier` удалён: реакция водителей на дождь не моделируется (§2.1) |
| `events.friday_evening` | `enabled, weekdays, hours: список, demand_multiplier` | `days: [4,5], hours: [17,23] (диапазон включительно), archetypes: [business, mixed], demand_mult: 1.3` | пт+сб вечер 17–23 (§3.6); календарное событие флага не требует |
| `pricing.surge_injection` | `enabled, days, hours, multiplier` | `day_chance: 0.10, duration_min: [30,60], multiplier: [1.2,1.6]` | инъекции случайные (§3.6), а не календарные: только так `price_multiplier` варьируется для ML (M3) |
| `pricing` | `elasticity_alpha` | + `base_boarding: 100.0, per_min_rub: 12.0` | формула выручки фиксируется сейчас (§10.15), в БД не пишется |
| `engine` | `agg_step_min` | + `sim_start: "2026-11-02"` | точка отсчёта сим-часов — понедельник (§3) |

Правило на будущее: новые ключи добавляются только с дефолтами; переименование/смена семантики существующих — только через явный параграф в этом плане.

## 7. Декомпозиция на changes

Формат для конвейера OpenSpec: один change = одна строка. Порядок: 1 → (2, 3, 4 можно параллельно после 1) → 5 → 6. Каждый change заканчивается прогоном `make lint && make test` (правило `openspec/config.yaml`).

Гейты приёмки по ходу вехи: перед сдачей change'а агент прогоняет `make accept-m2` и прикладывает вывод; провалены могут быть только критерии будущих changes. После `m2-city` зелёные A1, A2, A11; после `m2-loop` — A1–A12; после `m2-calibration` — все A1–A13.

| # | Change | Файлы | Содержание | Как проверить |
|---|--------|-------|-----------|---------------|
| 1 | `m2-city` | `common/config.py`, `config.yaml`, `simulator/city.py`, `tests/test_sim_city.py` | конфиг-модели под канон §6; `config.yaml` = приложение А; `build_city` + Флойд → `path_min` | эталонный конфиг валиден; пустой конфиг из дефолтов валиден; граф связный; `path_min` симметричен, нули на диагонали, положителен вне; профили длины 24, неотрицательные, на 4 архетипа × будни/выходные; `zone_id` уникальны |
| 2 | `m2-demand` | `simulator/demand.py`, `tests/test_sim_demand.py` | λ-формула, пуассоновский спавн, выбор назначения, эластичность, лимит ожидания, объект заказа | λ без факторов = `base_rate`; эластичность m=2 при α=1 режет поток ~вдвое (фиксированный rng); назначение тяготеет к `attract_dest` с затуханием по `dest_decay_min`; нулевой парк → все заказы `unfulfilled` ровно через лимит |
| 3 | `m2-drivers` | `simulator/drivers.py`, `tests/test_sim_drivers.py` | состояния водителей, диспатч (ближайший, тай-брейк по id), движение, высадка в dest | ближайший водитель получает заказ; при равенстве — меньший id; после dropoff водитель free в dest; длительность = `ceil(путь × k)`; старт парка: водитель i в зоне i mod Z |
| 4 | `m2-env` | `simulator/env.py`, `tests/test_sim_env.py` | дождь, пятница, инъекции, множители зон | форс-дождь (шанс 1.0) → спрос выше и поездки дольше; форс-инъекция → `price_multiplier` в агрегате > 1 в заданном окне; пятница даёт горб в заданных архетипах |
| 5 | `m2-loop` | `simulator/aggregate.py`, `simulator/engine.py`, `simulator/__main__.py`, `simulator/calibrate.py` (заготовка: прогон + печать `deficit_share`), `scripts/accept_m2.sh` (дословно из приложения В), `Makefile` (цели `sim`, `accept-m2`), `tests/test_sim_loop.py`, `tests/test_sim_repro.py`, `tests/test_sim_cli.py` | минутный цикл, агрегация §3.8, writer, CLI, CSV, сводка §4, скрипт приёмки | `test_reproducibility` (два прогона seed 42 → идентичные trips и агрегаты); `test_aggregation_math` (микросценарий, все 11 колонок сверены вручную); CLI smoke 1 день (N>0 строк в обеих таблицах, лимит времени, guard на непустую БД) |
| 6 | `m2-calibration` | `simulator/calibrate.py` (докручивается графиками), `docs/m2-calibration.md`, `reports/*.png`, `config.yaml` | калибровка эталона по §9 | `deficit_share` 10–20 %; графики: суточный профиль по зонам, ряд с дождём, распределение `free_drivers`; числа внесены в `config.yaml` |

## 8. Тесты (сводный список)

`test_reproducibility` — краеугольный: два полных прогона одного seed, сравнение всех строк обеих таблиц. Один файл на механику:

- `tests/test_sim_city.py` — конфиг-канон, связность, `path_min`, профили;
- `tests/test_sim_demand.py` — λ, эластичность, спавн, выбор назначения, лимит ожидания;
- `tests/test_sim_drivers.py` — диспатч, тай-брейки, движение, высадка;
- `tests/test_sim_env.py` — дождь, пятница, инъекции;
- `tests/test_sim_loop.py` — арифметика агрегации по микросценарию (все 11 колонок);
- `tests/test_sim_repro.py` — детерминизм;
- `tests/test_sim_cli.py` — CLI smoke, guard, CSV, сводка.

## 9. Калибровка эталона

Цель — мир, в котором ML-задача (M3) корректно поставлена:

- дефицитные 5-минутки занимают **10–20 %** времени (не «всё зелёное» — предсказывать нечего; не «всё красное» — класс вырождается во «всегда дефицит»);
- суточная волна читается глазами; дождь и пятница дают различимые горбы;
- средняя занятость парка ~60–70 %, пиковая ~90 %+ (водители есть, но впритык).

Инструменты: `.venv/bin/python -m simulator.calibrate --days 30 [--seed S]` — прогоняет мир, кладёт matplotlib-графики в `reports/` (суточный профиль по зонам, ряд с дождём, гистограмма `free_drivers` — не меньше трёх PNG) и печатает итог строкой машинного формата `deficit_share=14.8%`; эту строку проверяет скрипт приёмки (критерий A13), то же число приводится в отчёте. Подбор: `base_rate`, профили, `drivers.count`, `dest_decay_min`. Итог — обновлённые значения приложения А + отчёт `docs/m2-calibration.md` (какие числа, почему, графики). Стартовая точка: `base_rate` 2.0, парк 200, лимит ожидания 6.

## 10. Зафиксированные решения

1. Порядок шагов цикла и правила тай-брейков — часть контракта детерминизма; их изменение меняет мир (другая история при том же seed).
2. Снапшот драйверов на конец интервала (не среднее); `avg_wait_min` NULL при пустом интервале.
3. Диспатч «спасает» от истечения даже при долгой подаче; истечение — только для неназначенных.
4. Дождь действует одинаково на спрос (×1.4) и на время поездок (×1.4); значения калибруемы.
5. Длительности: `ceil`, минимум 1 минута; подача считается так же, как поездка.
6. Статус `cancelled` не используется (зарезервирован контрактом на будущее).
7. Эластичность внутри λ (модель ×m^(−α)), а не отдельной фильтрацией пассажиров.
8. Выручка не хранится — вычислима post-hoc из графа и `price_multiplier` заказа.
9. Один `numpy.random.Generator`; разыгрывания всегда для всех зон (λ=0 → Poisson(0)) — устойчивость к рефакторингу.
10. Сим-часы стартуют в понедельник `engine.sim_start` (эталон 2026-11-02); нумерация дней недели Mon=0; wall-clock внутри движка запрещён.
11. Внутренние id заказов — счётчик движка с 1 в порядке создания (нужен диспатчу для тай-брейков); в БД `trip_id` назначает SQLite при вставке батчей в порядке создания — поэтому генерация требует пустой таблицы `trips` (guard §4).
12. Движок не знает про SQLite: отдаёт строки интерфейсу `TripWriter` (приложение Б); sqlite-реализация writer'а — в `__main__.py` поверх `storage.repo`; батчи сбрасываются раз в сим-час.
13. Шов ценовой политики для M6: `Engine(cfg, price_fn=None)`. При `price_fn=None` тариф даёт `WorldEnv` (инъекции §3.6). Если `price_fn` задан — он подменяет тариф зон, но **розыгрыши инъекций всё равно происходят**: поток rng не сдвигается, ветки A/B на одном seed сравнимы попарно.
14. Старт парка детерминирован без rng: водитель `i` стартует в зоне `i mod Z`.
15. Базовая цена маршрута: `pricing.base_boarding + pricing.per_min_rub × path_min[A][B]` (руб.). В M2 нигде не вычисляется и не пишется; фиксируется сейчас, чтобы M6 не изобретал формулу.

## 11. Критерии приёмки

### 11.1 Автоматическая приёмка: `make accept-m2`

Единый определитель готовности для обеих сторон. Команда `make accept-m2` запускает `scripts/accept_m2.sh` (ставится change'ом `m2-loop` **дословно** из приложения В; правка скрипта допустима только вместе с правкой плана). Скрипт печатает `PASS`/`FAIL` по каждому критерию и возвращает exit-код, равный числу провалов. Веха принята ⇔ exit 0.

| # | Что проверяет | Ожидание |
|---|---------------|----------|
| A1 | `make lint` | exit 0 |
| A2 | `make test` | exit 0, все тесты §8 зелёные |
| A3 | состав `simulator/` | ровно 9 файлов приложения Б (`__init__, city, env, demand, drivers, aggregate, engine, __main__, calibrate`), ничего лишнего |
| A4 | генерация 30 дней, `--seed 42`, `--csv` | exit 0; сводка — ровно 4 строки формата §4; `wall` ≤ 60 с; в stderr ≥ 24 строк прогресса машинного формата |
| A5 | наполнение БД | `zone_metrics_5min` = 51 840 строк и 8 640 уникальных `ts`; `zone_id` в 0–5; `trips` > 0; есть и `completed`, и `unfulfilled`; других статусов нет; `forecasts` = 0 |
| A6 | инварианты данных | 0 строк с нарушением порядка `created_at ≤ assigned_at ≤ pickup_at ≤ dropoff_at`, доменов статусов или окна `created_at` ⊂ [2026-11-02, 2026-12-02); во всех строках `supply_demand_ratio = free_drivers / max(requests, 1)` |
| A7 | детерминизм | sha-хэши `trips` и `zone_metrics_5min` двух прогонов seed 42 совпадают; префикс `trips` у seed 43 отличается |
| A8 | guard | повторная генерация в непустую БД: exit 2, сообщение упоминает `trips` |
| A9 | CSV-экспорт | `trips.csv` и `zone_metrics_5min.csv` существуют; строк в каждом = строк в таблице + 1 (заголовок) |
| A10 | `make sim` | exit 0, БД по умолчанию из конфига создана |
| A11 | `config.yaml` = канон §6 | 6 зон, 7 дорог, `base_rate > 0`, 4 архетипа с профилями 24×2, `sim_start = "2026-11-02"`, множители дождя > 1, `day_chance ∈ (0, 1]` |
| A12 | публичный API = приложение Б | импорт всех перечисленных в Б имён проходит (`build_city`, `WorldEnv`, `lambda_rate`, `spawn_requests`, `Fleet`, `AggregateBuffer`, `Engine`, `Summary`, `TripWriter`, …) |
| A13 | калибровка §9 | `simulator.calibrate` exit 0 и печатает `deficit_share=<X>%` с X ∈ [10, 20]; в `reports/` ≥ 3 PNG; `docs/m2-calibration.md` содержит строку `deficit_share=` |

### 11.2 Ручные проверки (заказчик; скриптом не заменяются)

1. **Ревью кода против §2.2 и приложения Б**: нет лишних абстракций, классов и модулей; комментарии/docstring на русском; переиспользованы M1-API.
2. **Отчёт `docs/m2-calibration.md` осмыслен**: объясняет выбор чисел; на графиках глазами видны суточная волна, горбы дождя и «пятницы».
3. **Живой взгляд в данные** (по желанию, для демонстрации): выборка метрик утренних часов residential-зон и вечерних business/mixed показывает ожидаемые горбы.

### 11.3 Правило двусторонности

Агент прогоняет `make accept-m2` перед сдачей каждого change (гейты §7) и прикладывает вывод; заказчик при приёмке повторяет ту же команду — вердикт обязан совпасть, потому что его печатает один и тот же скрипт. Спор «готово/не готово» решается не обсуждением, а выводом скрипта: FAIL с номером критерия — это список работ, PASS по всем — веха закрыта (остаются только пункты §11.2).

## Приложение А. Эталонный config.yaml (мир)

Копируется в репо целиком. Числа — стартовая точка; финальные фиксируются калибровкой (`m2-calibration`) и обновляются здесь же и в `config.yaml`.

```yaml
seed: 42

storage:
  db_path: data/taxi.db

city:
  zones:
    - {id: 0, name: sleep-1, archetype: residential, attract_origin: 1.0, attract_dest: 0.6, map: {x: 180, y: 180}}
    - {id: 1, name: sleep-2, archetype: residential, attract_origin: 1.0, attract_dest: 0.6, map: {x: 820, y: 180}}
    - {id: 2, name: center,   archetype: business,   attract_origin: 1.3, attract_dest: 1.8, map: {x: 500, y: 400}}
    - {id: 3, name: office,   archetype: business,   attract_origin: 1.2, attract_dest: 1.5, map: {x: 500, y: 620}}
    - {id: 4, name: airport,  archetype: airport,    attract_origin: 0.6, attract_dest: 1.2, map: {x: 900, y: 620}}
    - {id: 5, name: lounge,   archetype: mixed,      attract_origin: 1.1, attract_dest: 1.4, map: {x: 180, y: 620}}
  roads:   # двунаправленные рёбра, минуты
    - {from_zone: 0, to_zone: 2, minutes: 9}
    - {from_zone: 1, to_zone: 2, minutes: 9}
    - {from_zone: 2, to_zone: 3, minutes: 6}
    - {from_zone: 3, to_zone: 4, minutes: 10}
    - {from_zone: 2, to_zone: 5, minutes: 8}
    - {from_zone: 0, to_zone: 5, minutes: 5}
    - {from_zone: 1, to_zone: 4, minutes: 14}
  # матрица path_min считается при старте (Флойд), не хранится

population: {scale: 1.0}

demand:
  base_rate: 2.0        # заказов/мин на зону до всех множителей
  dest_decay_min: 15
  profiles:             # 24 множителя на час 00–23; будни и выходные отдельно, на архетип
    residential:
      weekday: [0.2, 0.15, 0.1, 0.1, 0.1, 0.2, 0.6, 1.2, 1.6, 1.3, 0.9, 0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 1.2, 1.4, 1.3, 1.1, 0.9, 0.6, 0.35]
      weekend:  [0.3, 0.25, 0.2, 0.15, 0.1, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.2, 1.3, 1.3, 1.2, 1.0, 0.8, 0.5]
    business:
      weekday: [0.15, 0.1, 0.1, 0.1, 0.15, 0.3, 0.7, 1.3, 1.7, 1.5, 1.2, 1.1, 1.1, 1.1, 1.1, 1.2, 1.3, 1.6, 1.7, 1.4, 1.0, 0.7, 0.4, 0.25]
      weekend:  [0.2, 0.15, 0.1, 0.1, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.2, 1.1, 0.9, 0.6, 0.4]
    airport:
      weekday: [0.7, 0.7, 0.7, 0.8, 0.8, 0.9, 1.0, 1.1, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1, 1.0, 0.9, 0.8]
      weekend:  [0.8, 0.75, 0.7, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.1, 1.1, 1.1, 1.0, 0.95, 0.85]
    mixed:
      weekday: [0.5, 0.4, 0.3, 0.2, 0.15, 0.15, 0.25, 0.4, 0.6, 0.8, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.5, 1.3, 0.9]
      weekend:  [0.8, 0.7, 0.55, 0.4, 0.25, 0.2, 0.25, 0.35, 0.5, 0.7, 0.85, 0.95, 1.0, 1.0, 1.05, 1.1, 1.15, 1.25, 1.35, 1.45, 1.6, 1.6, 1.4, 1.1]

drivers: {count: 200, wait_limit_min: 6}

weather:
  rain_day_chance: 0.15
  episode_start_h: [6, 22]      # U(6:00–22:00)
  episode_duration_h: [1, 3]
  demand_mult: 1.4
  travel_mult: 1.4

events:
  friday_evening:
    days: [4, 5]                # пт, сб (нумерация с понедельника = 0)
    hours: [17, 23]             # диапазон включительно
    archetypes: [business, mixed]
    demand_mult: 1.3

pricing:
  elasticity_alpha: 1.0
  base_boarding: 100.0          # руб., формула выручки §10.15
  per_min_rub: 12.0
  surge_injection:
    day_chance: 0.10
    duration_min: [30, 60]
    multiplier: [1.2, 1.6]

engine:
  agg_step_min: 5
  sim_start: "2026-11-02"       # понедельник, старт сим-часов 00:00
```

## Приложение Б. Структура пакета simulator/ и публичный API

Состав файлов фиксирован. Публичные имена/сигнатуры ниже — контракт; внутренние детали реализации (приватные функции, поля) — на усмотрение исполнителя. Типы `AppConfig`, `datetime` — из `common.config`, `common.timeutil`.

```
simulator/
├── __init__.py       # пустой (или re-export Engine); пакет импортируется как `simulator`
├── city.py           # CitySpec, ZoneInfo, HourProfile, build_city
├── env.py            # WorldEnv: дождь/пятница/инъекции/тариф
├── demand.py         # lambda_rate, spawn_requests, TripRequest
├── drivers.py        # DriverState, Driver, Fleet
├── aggregate.py      # AggregateBuffer: 5-минутные окна §3.8
├── engine.py         # Engine, Summary, TripWriter — минутный цикл §3.7
├── __main__.py       # CLI §4: argparse, sqlite-writer, сводка, CSV
└── calibrate.py      # python -m simulator.calibrate — прогон + deficit_share + графики (§9)
```

```python
# city.py
@dataclass(frozen=True)
class ZoneInfo:
    id: int
    name: str
    archetype: str
    attract_origin: float
    attract_dest: float

@dataclass(frozen=True)
class HourProfile:
    weekday: np.ndarray   # длины 24
    weekend: np.ndarray   # длины 24

@dataclass(frozen=True)
class CitySpec:
    zones: tuple[ZoneInfo, ...]        # сортированы по id, id = 0..Z-1 без пропусков
    path_min: np.ndarray               # (Z, Z), int, кратчайшие пути в минутах
    profiles: dict[str, HourProfile]   # по архетипам из зон

def build_city(cfg: AppConfig) -> CitySpec: ...

# env.py
class WorldEnv:
    def __init__(self, cfg: AppConfig, rng: np.random.Generator) -> None: ...
    def advance_day(self, day_index: int) -> None: ...          # розыгрыши дождя/инъекции на сутки (00:00)
    def is_rain(self, t: datetime) -> bool: ...
    def event_label(self, zone: ZoneInfo, t: datetime) -> str: ...   # 'friday_evening' | 'surge_injection' | ''
    def price_multiplier(self, zone_id: int, t: datetime) -> float: ...

# demand.py
@dataclass(frozen=True)
class TripRequest:
    created_at: datetime
    origin: int
    dest: int
    price_multiplier: float

def lambda_rate(city: CitySpec, zone: ZoneInfo, t: datetime, *,
                is_rain: bool, event_mult: float, price_mult: float,
                cfg: AppConfig) -> float: ...

def spawn_requests(rng: np.random.Generator, city: CitySpec, t: datetime, *,
                   is_rain: bool, event_mult: float,
                   price_mult_by_zone: np.ndarray, cfg: AppConfig) -> list[TripRequest]: ...

# drivers.py
class DriverState(enum.Enum):
    FREE = "free"
    TO_PICKUP = "to_pickup"
    ON_TRIP = "on_trip"

@dataclass
class Driver:
    id: int
    zone: int
    state: DriverState
    trip_id: int | None      # внутренний id заказа
    remaining_min: int       # до посадки (TO_PICKUP) или высадки (ON_TRIP)

class Fleet:
    def __init__(self, count: int, zone_count: int) -> None: ...   # водитель i → зона i mod Z
    def nearest_free(self, zone: int, path_row: np.ndarray) -> int | None: ...  # тай-брейк: меньший id
    def assign(self, driver_id: int, trip_id: int, pickup_min: int) -> None: ...
    def tick(self) -> list[tuple[int, int]]: ...   # высадки этой минуты: (driver_id, зона_высадки)
    def free_by_zone(self) -> np.ndarray: ...      # counts
    def busy_by_zone(self) -> np.ndarray: ...      # counts

# aggregate.py
class AggregateBuffer:
    def __init__(self, zone_count: int) -> None: ...
    def add_request(self, zone: int) -> None: ...
    def add_assignment(self, zone: int, wait_min: int) -> None: ...
    def add_completion(self, zone: int) -> None: ...
    def add_expiry(self, zone: int) -> None: ...
    def snapshot_drivers(self, free: np.ndarray, busy: np.ndarray) -> None: ...
    def set_env(self, zone: int, is_rain: bool, event: str, minute_price: float) -> None: ...  # поминутно
    def flush(self, ts: datetime) -> list[dict]: ...  # строки zone_metrics_5min (дикты под storage.repo)

# engine.py
class TripWriter(Protocol):
    def write_trips(self, rows: list[dict]) -> None: ...      # дикты под storage.repo.insert_trips
    def write_metrics(self, rows: list[dict]) -> None: ...    # дикты под storage.repo.insert_zone_metrics

@dataclass(frozen=True)
class Summary:
    days: int
    seed: int
    trips_total: int
    completed: int
    unfulfilled: int
    unfulfilled_share: float
    avg_wait_min: float
    deficit_share: float
    wall_sec: float

class Engine:
    def __init__(self, cfg: AppConfig, *,
                 seed: int | None = None,                      # None → cfg.seed
                 price_fn: Callable[[int, datetime], float] | None = None) -> None: ...
    def run(self, days: int, writer: TripWriter, *,
            progress: Callable[[datetime, int, float], None] | None = None) -> Summary: ...
    # progress(t, trips_so_far, elapsed_sec) вызывается раз в сим-час
```

Правила: `__main__.py` содержит sqlite-реализацию `TripWriter` (обёртка над `storage.repo.insert_trips` / `insert_zone_metrics` поверх `storage.db.connect`) и не содержит логики мира; `calibrate.py` не участвует в `make test` обязательными тестами (только smoke, если удобно).

## Приложение В. Скрипт приёмки `scripts/accept_m2.sh` (дословный текст)

Change `m2-loop` кладёт этот текст в `scripts/accept_m2.sh` без изменений и добавляет в `Makefile` цель `accept-m2` (`bash scripts/accept_m2.sh`). Скрипт самодостаточен: чистит артефакты прошлых прогонов, создаёт свои БД в `data/`, ничего не требует кроме `.venv`, `sqlite3` и `shasum`. Полный прогон занимает ~2–4 минуты (пять симуляций, включая калибровку). Правка скрипта допустима только вместе с правкой этого приложения и таблицы §11.1.

```bash
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
expected="__init__.py __main__.py aggregate.py calibrate.py city.py demand.py drivers.py engine.py env.py"
actual=$(ls simulator | sort | tr '\n' ' ' | sed 's/ $//')
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
```
