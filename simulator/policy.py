"""Политики цены вехи M5+M6 (change ``m6-policy``): математика и живой источник.

Ступенчатая политика по прогнозной вероятности дефицита (ветки C/D) и
реактивная по факту последнего закрытого бакета (ветка B, анти-пример ТЗ):
``p_eff = effective_prob(deficit_prob, shift[model])`` → ступень с наибольшим
``p_min ≤ p_eff``; итоговый тариф зоны компонуется с тарифом мира умножением
с потолком ``total_cap`` (решение 6 плана M5+M6), при множителе 1.00 тариф
мира возвращается без изменений — политика, не вмешивающаяся в данный момент,
не меняет цену и мир. Числа ступеней и порогов живут в секции ``policy:``
конфига (прил. А плана ``docs/plan-m5-m6-policy.md``), стартовые значения
калибруются в change'е ``m6-worlds``. Правило дефицита — единственное правило
контракта (:func:`ml.dataset.deficit_label`); реактивная политика не
сглаживает и не предсказывает — только факт. Направление зависимости
simulator → ml — решение 1 design.md: ml-слой M3 не знает про движок,
политика потребляет его публичные контракты.

:class:`LiveRowSource` — реализация протокола :class:`ml.dataset.RowSource`
поверх тикающего движка (решение 2 design.md): строки метрик закрытых бакетов
питаются швом ``Engine.run(on_metrics_row=...)``, расписание высадок —
событиями назначений шва ``on_minute``; семантика запросов идентична
``SqliteRowSource`` поверх БД того же прогона.

:class:`PolicyContour` — контур применения политики в живом прогоне
(решения 5–8 design.md): тарифная функция ``price`` подключается движку как
``price_fn``, наблюдатели истории передаются в ``run``, привязка к движку
(``bind``) — поздняя; множитель кэшируется на бакет, до окна активации
(день ``ml.train_days + 1``) тариф мира возвращается без изменений,
диагностика срабатываний копится в памяти. Сетевой прогнозист
:class:`NetworkPredictor` восстанавливает сеть из артефакта M3 без ретрейна.
"""

from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from common.config import AppConfig, PolicyStairCfg
from common.timeutil import add_minutes
from ml.dataset import (
    RET_HORIZONS_MIN,
    ROW_MANIFEST,
    MetricPoint,
    bucket_index,
    build_window,
    deficit_label,
    from_bucket_index,
    world_start,
)
from ml.infer import latest_artifact_dir, load_model_artifact, predict_windows
from ml.train import ARTIFACTS_ROOT
from simulator.live import MinuteRecord

if TYPE_CHECKING:
    from simulator.engine import Engine

__all__ = [
    "LiveRowSource",
    "NetworkPredictor",
    "PolicyContour",
    "Predictor",
    "ReactiveContour",
    "ShadowForecaster",
    "compose_tariff",
    "has_deficit",
    "effective_prob",
    "model_shift",
    "reactive_multiplier",
    "stair_multiplier",
]

# Индексы нужных колонок вектора строки источника — производные публичного
# манифеста :data:`ml.dataset.ROW_MANIFEST` (те же значения, что у приватных
# констант ml.dataset; дублирование сознательное — импорт приватных имён
# другого пакета нарушал бы решение 1 design.md об импорте ml публично).
_REQUESTS_COL = ROW_MANIFEST.index("requests")
_UNFULFILLED_COL = ROW_MANIFEST.index("unfulfilled")
_FREE_DRIVERS_COL = ROW_MANIFEST.index("free_drivers")


def _minute_index(ts: datetime) -> int:
    """Минутный индекс момента — семантика ``ml.dataset._minute_index``.

    Выражается публичным ``bucket_index(ts, 1)``: минуты от общего начала
    отсчёта ml.dataset; сравнения «назначено ≤ cutoff» через ``bisect_right``.
    """

    return bucket_index(ts, 1)


def model_shift(model: str, shift: Mapping[str, float]) -> float:
    """δ-сдвиг ступеней для семейства моделей; без записи в ``shift`` — 0.0."""

    return shift.get(model, 0.0)


def effective_prob(deficit_prob, shift: float):
    """Эффективная вероятность ступенчатой политики (скаляр или массив numpy).

    ``shift ≥ 0`` — сдвиг вверх: ``clip(p + shift, 0, 1)``. ``shift < 0`` —
    растяжение отрезка ``[−shift, 1]`` на ``[0, 1]``: ``clip((p + shift) /
    (1 + shift), 0, 1)``. Простой сдвиг вниз срезал верх шкалы: при
    калиброванном δ ≈ −0.44 вероятность 1.0 давала ``p_eff`` 0.56, и ступени
    с порогами 0.6 и 0.8 не срабатывали никогда. Растяжение сохраняет порог
    первой ступени (та же частота вмешательств) и оставляет ``p = 1 → 1``,
    поэтому уверенный прогноз доходит до верхних ступеней. Обе ветви
    монотонны по ``shift`` — на этом держится бинарный поиск калибровки.
    """

    p = np.asarray(deficit_prob, dtype=float)
    if shift >= 0:
        p_eff = p + shift
    elif shift <= -1.0:
        p_eff = np.where(p >= 1.0, 1.0, 0.0)
    else:
        p_eff = (p + shift) / (1.0 + shift)
    p_eff = np.clip(p_eff, 0.0, 1.0)
    return float(p_eff) if p_eff.ndim == 0 else p_eff


def stair_multiplier(deficit_prob: float, shift: float, stairs: Sequence[PolicyStairCfg]) -> float:
    """Множитель ступенчатой политики: верхняя ступень с ``p_min ≤ p_eff``.

    ``p_eff = effective_prob(deficit_prob, shift)``; порог ступени включается
    (``p_eff`` ровно ``p_min`` уже активирует ступень), порядок списка
    ``stairs`` на результат не влияет — ищется наибольший ``p_min`` среди
    наступивших; ни одна ступень не наступила — множитель 1.00. Чистая
    функция без скрытого состояния.
    """

    p_eff = effective_prob(deficit_prob, shift)
    active = [stair for stair in stairs if stair.p_min <= p_eff]
    if not active:
        return 1.0
    return max(active, key=lambda stair: stair.p_min).mult


def compose_tariff(world_tariff: float, multiplier: float, total_cap: float) -> float:
    """Итоговый тариф зоны: умножение на тариф мира с потолком ``total_cap``.

    При множителе 1.00 тариф мира возвращается как есть — без клиппирования,
    даже если сам тариф мира выше потолка (безусловная инертность, решение 4
    design.md); при множителе выше 1.00 произведение ограничивается потолком.
    """

    if multiplier <= 1.0:
        return world_tariff
    return min(world_tariff * multiplier, total_cap)


def has_deficit(row: Mapping[str, Any]) -> bool:
    """Дефицит строки метрик закрытого бакета по правилу контракта.

    Единственное правило (:func:`ml.dataset.deficit_label`):
    ``free_drivers / max(requests, 1) < 0.5`` **или** ``unfulfilled > 0``.
    """

    return deficit_label(row["requests"], row["free_drivers"], row["unfulfilled"])


def reactive_multiplier(row: Mapping[str, Any] | None, reactive_mult: float) -> float:
    """Множитель реактивной политики (ветка B): факт последнего закрытого бакета.

    Дефицит в последней закрытой строке метрик зоны — ``reactive_mult``,
    иначе 1.00; нет ни одного закрытого бакета (прогрев) — 1.00. Без
    прогноза, упреждения, сглаживания и гистерезиса — мигание у границы
    является частью контрастируемой патологии ветки.
    """

    if row is None:
        return 1.0
    return reactive_mult if has_deficit(row) else 1.0


class LiveRowSource:
    """:class:`ml.dataset.RowSource` поверх живого прогона движка (спека simulator-policy).

    Питание — два чистых наблюдателя (design решение 2): строки метрик закрытых
    бакетов приходят швом ``Engine.run(on_metrics_row=...)`` (те же дикты, что
    уходят writer-батчу), расписание высадок — из событий назначений наблюдения
    минут ``on_minute`` (``AssignEvent``: зона высадки, ``dropoff_at``, минута
    назначения = минута записи). Семантика запросов идентична
    :class:`ml.dataset.SqliteRowSource` поверх БД того же прогона: окно требует
    непрерывной истории всех ``steps`` бакетов, иначе ``None``;
    ``dropoff_counts`` — высадки в бакетах ``(after, until]`` с назначением не
    позднее ``assigned_until``; ``time_range`` растёт по мере закрытия бакетов.
    Источник зависит только от поданных записей наблюдения и не от порядка
    последующих запросов. Назначенный в движке заказ всегда завершается
    высадкой (дренаж хвоста прогоняет мир до терминальных состояний), поэтому
    индекс назначений полон против запроса ``status='completed'`` Sqlite.
    """

    def __init__(self, *, agg_step_min: int = 5) -> None:
        """Создать пустой источник; шаг агрегации — как у движка (``engine.agg_step_min``)."""

        self._step = agg_step_min
        self._metrics: dict[tuple[int, int], np.ndarray] = {}
        self._dropoffs: dict[tuple[int, int], list[int]] = {}
        self._first_idx: int | None = None
        self._last_idx: int | None = None

    def on_metrics_row(self, row: Mapping[str, Any]) -> None:
        """Наблюдатель закрытых бакетов (шов ``Engine.run(on_metrics_row=...)``).

        Строка-дикт ``AggregateBuffer`` переводится в 10-колоночный вектор
        :data:`ml.dataset.ROW_MANIFEST` с той же конвертацией, что у
        ``SqliteRowSource`` при загрузке БД (пустое ожидание → 0.0, метка
        события → флаг 0/1); время строки — начало бакета.
        """

        idx = bucket_index(row["ts"], self._step)
        wait = row["avg_wait_min"]
        vector = np.array(
            [
                row["requests"],
                row["completed"],
                row["unfulfilled"],
                row["free_drivers"],
                row["busy_drivers"],
                0.0 if wait is None else float(wait),
                row["supply_demand_ratio"],
                float(row["is_rain"]),
                1.0 if row["event"] else 0.0,
                row["price_multiplier"],
            ],
            dtype=float,
        )
        self._metrics[(int(row["zone_id"]), idx)] = vector
        if self._first_idx is None or idx < self._first_idx:
            self._first_idx = idx
        if self._last_idx is None or idx > self._last_idx:
            self._last_idx = idx

    def on_minute(self, record: MinuteRecord) -> None:
        """Наблюдатель минут (шов ``Engine.run(on_minute=...)``): расписание высадок.

        Каждое назначение минуты индексируется парой (зона высадки, бакет
        ``dropoff_at``) с минутой назначения ``record.t`` — семантика
        ret-признаков: «назначены не позднее ts, высадка в будущем легальна —
        известное расписание». Повторные назначения разных заказов в одну
        минуту дают повторные записи списка — счёт ``dropoff_counts`` верен.
        """

        minute = _minute_index(record.t)
        for event in record.assignments:
            key = (event.dest, bucket_index(event.dropoff_at, self._step))
            self._dropoffs.setdefault(key, []).append(minute)

    def metrics_window(self, zone_id: int, ts: datetime, steps: int) -> np.ndarray | None:
        """Окно метрик: строки ровно запрошенных бакетов, ``None`` при дыре (как в M3)."""

        last = bucket_index(ts, self._step)
        window = np.empty((steps, len(ROW_MANIFEST)), dtype=float)
        for k in range(steps):
            row = self._metrics.get((zone_id, last - (steps - 1 - k)))
            if row is None:
                return None
            window[k] = row
        return window

    def metrics_at(self, zone_id: int, ts: datetime) -> MetricPoint | None:
        """Точечные метрики бакета ``ts`` или ``None``, если строки нет."""

        row = self._metrics.get((zone_id, bucket_index(ts, self._step)))
        if row is None:
            return None
        return MetricPoint(
            float(row[_REQUESTS_COL]),
            float(row[_FREE_DRIVERS_COL]),
            float(row[_UNFULFILLED_COL]),
        )

    def dropoff_counts(
        self, zone_id: int, after: datetime, until: datetime, assigned_until: datetime
    ) -> int:
        """Число завершённых высадок в бакетах ``(after, until]`` с назначением ≤ cutoff."""

        limit = _minute_index(assigned_until)
        total = 0
        for idx in range(bucket_index(after, self._step) + 1, bucket_index(until, self._step) + 1):
            assigned = self._dropoffs.get((zone_id, idx))
            if assigned:
                total += bisect_right(assigned, limit)
        return total

    def time_range(self) -> tuple[datetime, datetime]:
        """Первый и последний бакет поданных метрик; пустой источник — ``ValueError``."""

        if self._first_idx is None or self._last_idx is None:
            raise ValueError("LiveRowSource пуст: не подано ни одной строки метрик")
        return (
            from_bucket_index(self._first_idx, self._step),
            from_bucket_index(self._last_idx, self._step),
        )


class Predictor(Protocol):
    """Прогнозист контура (design решение 8): вероятность дефицита зоны.

    ``ts`` — момент прогноза (последний закрытый бакет); ``None`` — полного
    окна нет (прогрев), контур отвечает множителем 1.00. Стаб-реализации живут
    в тестах, сетевая — :class:`NetworkPredictor`.
    """

    def predict(self, zone_id: int, ts: datetime) -> float | None:
        """Вероятность дефицита зоны ``zone_id`` на момент ``ts`` или ``None``."""
        ...


class NetworkPredictor:
    """Сетевой прогнозист: артефакт семейства ``mlp``/``lstm`` + окно над источником.

    Каталог — самый свежий ``{family}-v1-*`` в корне артефактов (прецедент
    ml-inference); его отсутствие — ``FileNotFoundError`` при построении
    контура (guard до старта прогона — спека simulator-policy). Архитектура
    и окно признаков — из снимка ``ml_cfg`` артефакта (не из живого конфига,
    защита от дрейфа), вероятность — горизонта ``policy.horizon_min`` (индекс
    головы в :data:`ml.dataset.RET_HORIZONS_MIN`). Окно строится сборщиком M3
    :func:`ml.dataset.build_window` над источником истории контура — анти-лик
    гарантии M3 наследуются; неполное окно (прогрев) — ``None``.
    """

    def __init__(
        self,
        source: LiveRowSource,
        cfg: AppConfig,
        *,
        family: str,
        artifacts_root: str | Path = ARTIFACTS_ROOT,
    ) -> None:
        """Восстановить сеть из свежайшего каталога семейства ``family``.

        ``n_zones`` (ёмкость эмбеддинга) — параметр окружения, берётся из
        города конфига, как в ml-inference.
        """

        if cfg.policy.horizon_min not in RET_HORIZONS_MIN:
            raise ValueError(
                f"policy.horizon_min={cfg.policy.horizon_min} вне горизонтов"
                f" контракта {RET_HORIZONS_MIN}"
            )
        model_dir = latest_artifact_dir(artifacts_root, family)
        n_zones = max(zone.id for zone in cfg.city.zones) + 1
        self._artifact = load_model_artifact(model_dir, n_zones=n_zones)
        self._source = source
        self._horizon_idx = RET_HORIZONS_MIN.index(cfg.policy.horizon_min)

    def predict(self, zone_id: int, ts: datetime) -> float | None:
        """Окно признаков над источником → инференс → ``deficit_prob`` горизонта."""

        window = build_window(
            self._source, zone_id, ts, window_hours=self._artifact.ml_cfg.window_hours
        )
        if window is None:
            return None
        _norm_requests, _norm_free, probs = predict_windows(
            self._artifact, window[np.newaxis], [zone_id]
        )
        return float(probs[0, self._horizon_idx])


class PolicyContour:
    """Контур политики цены: тариф ``price`` + наблюдатели истории + диагностика.

    Позднее связывание с движком (design решение 5): ``price`` подключается
    как ``price_fn`` при построении движка, ``bind(engine)`` вызывается сразу
    после и запоминает ``engine.world_tariff``; ``price`` вызывается только
    внутри ``run`` — к первому вызову привязка выполнена. Каденс — закрытие
    бакета (решение 6): множитель зоны пересчитывается один раз на
    5-минутный бакет по первому запросу тарифа в нём, по последнему закрытому
    бакету; результат зависит только от закрытых бакетов, порядок вызовов не
    важен. Окно активации (решение 7): до первого бакета дня
    ``ml.train_days + 1`` от старта мира тариф мира возвращается без изменений
    (префикс ветки побитово равен ``a-base``). Диагностика срабатываний
    (решение 10): число бакетов-зон с множителем > 1.00 и доля от вычисленных
    бакетов-зон окна активации — в памяти контура, в БД не пишется.
    """

    def __init__(
        self,
        cfg: AppConfig,
        predictor: Predictor | None,
        source: LiveRowSource,
        *,
        model: str = "",
    ) -> None:
        """Собрать контур: прогнозист, источник истории, δ-сдвиг семейства ``model``.

        ``model`` — имя семейства для lookup в ``policy.shift`` (``mlp``/``lstm``);
        для стаб-моделей и реактивной ветки сдвиг пуст — 0.0. ``None``-прогнозист
        допустим только контурам, переопределяющим :meth:`_multiplier`
        (:class:`ReactiveContour`) — базовый расчёт ступеней без него не имеет
        смысла.
        """

        self._predictor = predictor
        self._source = source
        self._stairs = cfg.policy.stairs
        self._total_cap = cfg.policy.total_cap
        self._shift = model_shift(model, cfg.policy.shift)
        self._step = cfg.engine.agg_step_min
        self._activation = add_minutes(world_start(cfg), cfg.ml.train_days * 1440)
        self._world_tariff: Any = None  # bind() до первого вызова price
        self._cache: dict[tuple[int, int], float] = {}  # (зона, бакет) → множитель
        self._fire_count = 0
        self._fire_slots = 0

    def bind(self, engine: "Engine") -> None:
        """Поздняя привязка: запомнить публичное чтение тарифа мира движка."""

        self._world_tariff = engine.world_tariff

    def on_minute(self, record: MinuteRecord) -> None:
        """Наблюдатель минут движка — питание источника истории (делегирование)."""

        self._source.on_minute(record)

    def on_metrics_row(self, row: Mapping[str, Any]) -> None:
        """Наблюдатель закрытых бакетов движка — питание источника (делегирование)."""

        self._source.on_metrics_row(row)

    def price(self, zone_id: int, t: datetime) -> float:
        """Тариф зоны политики: композиция тарифа мира с множителем ступеней.

        До границы активации — тариф мира как есть (без клиппа, решение 4);
        после — умножение с потолком ``total_cap``.
        """

        world = self._world_tariff(zone_id, t)
        if t < self._activation:
            return world
        return compose_tariff(world, self._cached_multiplier(zone_id, t), self._total_cap)

    def _cached_multiplier(self, zone_id: int, t: datetime) -> float:
        """Множитель зоны на бакет момента ``t`` — лениво, один расчёт на бакет.

        Момент прогноза — последний закрытый бакет (начало текущего минус шаг):
        его строки метрик уже поданы наблюдателем, назначения ≤ него тоже.
        Счётчики диагностики растут ровно один раз на вычисленный бакет-зону.
        """

        bucket = bucket_index(t, self._step)
        key = (zone_id, bucket)
        cached = self._cache.get(key)
        if cached is None:
            closed_ts = from_bucket_index(bucket - 1, self._step)
            cached = self._multiplier(zone_id, closed_ts)
            self._cache[key] = cached
            self._fire_slots += 1
            if cached > 1.0:
                self._fire_count += 1
        return cached

    def _multiplier(self, zone_id: int, ts: datetime) -> float:
        """Множитель ступеней по прогнозу; нет полного окна — 1.00 (design решение 8)."""

        if self._predictor is None:  # защитный путь: базовый контур без прогнозиста
            return 1.0
        p = self._predictor.predict(zone_id, ts)
        if p is None:
            return 1.0
        return stair_multiplier(p, self._shift, self._stairs)

    @property
    def fire_count(self) -> int:
        """Число бакетов-зон окна активации с множителем > 1.00 (решение 10)."""

        return self._fire_count

    @property
    def fire_slots(self) -> int:
        """Число вычисленных бакетов-зон окна активации (знаменатель доли)."""

        return self._fire_slots

    @property
    def fire_rate(self) -> float:
        """Доля срабатываний ступеней; без вычислений — 0.0."""

        return self._fire_count / self._fire_slots if self._fire_slots else 0.0


class ReactiveContour(PolicyContour):
    """Контур реактивной политики (ветка ``b-reactive`` — анти-пример ТЗ).

    Множитель задаёт не ступени по прогнозу, а факт дефицита последнего
    закрытого бакета: ``policy.reactive_mult`` при дефиците по правилу
    контракта (:func:`reactive_multiplier`), иначе 1.00 — без прогноза,
    упреждения, сглаживания и гистерезиса (мигание у границы — часть
    контрастируемой патологии ветки). Кэш на бакет, окно активации,
    композиция с тарифом мира и диагностика срабатываний — общие с
    :class:`PolicyContour`; «срабатыванием» считается множитель > 1.00.
    """

    def __init__(self, cfg: AppConfig, source: LiveRowSource) -> None:
        """Собрать контур без прогнозиста: правило читает метрики источника."""

        super().__init__(cfg, None, source)
        self._reactive_mult = cfg.policy.reactive_mult

    def _multiplier(self, zone_id: int, ts: datetime) -> float:
        """Множитель по метрикам последнего закрытого бакета зоны."""

        point = self._source.metrics_at(zone_id, ts)
        row = (
            None
            if point is None
            else {
                "requests": point.requests,
                "free_drivers": point.free_drivers,
                "unfulfilled": point.unfulfilled,
            }
        )
        return reactive_multiplier(row, self._reactive_mult)


class ShadowForecaster:
    """Тень-прогнозист обеих сетей (design решение 3, спека simulator-policy).

    С границы окна активации (первый бакет дня ``ml.train_days + 1`` — та же
    граница, что у политики) на закрытии каждого укомплектованного бакета
    (счётчик зон, не порядок строк) обе сети семейств ``mlp``/``lstm`` из
    свежайших каталогов артефактов независимо от политики ветки вычисляют для
    каждой зоны прогнозы контракта ``forecasts`` обоих горизонтов (15/30 мин);
    ``model_version`` строки — имя семейства, не каталога (§3 плана M5+M6).
    Момент прогноза — последний закрытый бакет; признаки строятся только из
    закрытой истории (:func:`ml.dataset.build_window`, анти-лик гарантий M3),
    неполное окно строк не даёт. До границы активации строки не вычисляются и
    не пишутся. Тень — чистое наблюдение: чтения не меняют розыгрыши и
    эндогенную историю движка. Запись — буфер строк закрытого бакета одним
    ``flush(rows)`` (в генерации ветки — :func:`storage.repo.insert_forecasts`);
    ``flush=None`` — вычисление без записи (тесты). Отсутствие каталога
    семейства — ``FileNotFoundError`` при построении (guard до старта прогона).
    """

    FAMILIES = ("mlp", "lstm")
    """Семейства тени — обе сети сетки миров (решение 7 плана M5+M6)."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        artifacts_root: str | Path = ARTIFACTS_ROOT,
        flush: Callable[[list[dict]], None] | None = None,
    ) -> None:
        """Поднять обе сети и собрать границы; каталогов нет — исключение до прогона."""

        zones = [zone.id for zone in cfg.city.zones]
        n_zones = max(zones) + 1
        self._artifacts = {
            family: load_model_artifact(
                latest_artifact_dir(artifacts_root, family), n_zones=n_zones
            )
            for family in self.FAMILIES
        }
        self._source = LiveRowSource(agg_step_min=cfg.engine.agg_step_min)
        self._zones = zones
        self._step = cfg.engine.agg_step_min
        self._activation = add_minutes(world_start(cfg), cfg.ml.train_days * 1440)
        self._flush = flush
        self._seen: dict[int, set[int]] = {}

    def on_minute(self, record: MinuteRecord) -> None:
        """Наблюдатель минут движка — питание источника истории (делегирование)."""

        self._source.on_minute(record)

    def on_metrics_row(self, row: Mapping[str, Any]) -> None:
        """Наблюдатель закрытых бакетов: укомплектован бакет — один прогноз тени.

        Бакет укомплектован, когда строки всех зон города собраны (счётчик зон,
        порядок прихода строк не важен); прогноз — по последнему закрытому
        бакету, строки обоих семейств и горизонтов уходят буфером в ``flush``.
        """

        self._source.on_metrics_row(row)
        idx = bucket_index(row["ts"], self._step)
        seen = self._seen.setdefault(idx, set())
        seen.add(int(row["zone_id"]))
        if len(seen) < len(self._zones):
            return
        del self._seen[idx]
        ts = from_bucket_index(idx, self._step)
        if ts < self._activation:
            return
        rows = self._forecast_rows(ts)
        if rows and self._flush is not None:
            self._flush(rows)

    def _forecast_rows(self, ts: datetime) -> list[dict]:
        """Строки контракта обоих семейств на момент ``ts``; неполное окно — ни одной."""

        rows: list[dict] = []
        for family, artifact in self._artifacts.items():
            windows = [
                build_window(self._source, zone_id, ts, window_hours=artifact.ml_cfg.window_hours)
                for zone_id in self._zones
            ]
            if any(window is None for window in windows):
                continue
            norm_requests, norm_free, probs = predict_windows(
                artifact, np.stack(windows), self._zones
            )
            requests = artifact.scalers.requests.inverse_transform(
                norm_requests.reshape(-1, 1)
            ).reshape(len(self._zones), len(RET_HORIZONS_MIN))
            free_drivers = artifact.scalers.free_drivers.inverse_transform(
                norm_free.reshape(-1, 1)
            ).reshape(len(self._zones), len(RET_HORIZONS_MIN))
            for i, zone_id in enumerate(self._zones):
                for k, horizon in enumerate(RET_HORIZONS_MIN):
                    rows.append(
                        {
                            "ts": ts,
                            "zone_id": zone_id,
                            "horizon_min": horizon,
                            "pred_requests": float(requests[i, k]),
                            "pred_free_drivers": float(free_drivers[i, k]),
                            "deficit_prob": float(probs[i, k]),
                            "model_version": family,
                        }
                    )
        return rows
