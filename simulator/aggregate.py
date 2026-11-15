"""Агрегация состояния города в 5-минутные окна ``zone_metrics_5min`` (§3.8 плана M2).

Публичный контракт — приложение Б плана M2 (docs/plan-m2-engine.md): класс
``AggregateBuffer`` копит за текущее окно счётчики спроса и обслуживания,
поминутные значения среды и снапшот парка, а ``flush`` закрывает окно строками-
диктами, готовыми для ``storage.repo.insert_zone_metrics``, и обнуляет накопители.
Правило дефицита контракта («``free_drivers / max(requests, 1) < 0.5`` или
``unfulfilled > 0``») вычисляется функцией и колонкой не хранится (§3.8);
``_is_deficit`` — единый источник истины для ``Summary.deficit_share`` движка и
заготовки ``simulator.calibrate``.
"""

from datetime import datetime

import numpy as np

# Порог доли свободных водителей в правиле дефицита — константа контракта данных
# (§3.8 плана M2, раздел «Правило дефицита»); ключа в config.yaml нет, поэтому
# именованная константа магическим числом не считается (§2.2 п. 6).
_DEFICIT_RATIO_LIMIT = 0.5


def _is_deficit(row: dict) -> bool:
    """Правило дефицита §3.8 для строки метрик окна.

    Строка ``row`` — дикт колонок ``zone_metrics_5min`` (как возвращает
    ``AggregateBuffer.flush``). Дефицит — низкое обеспечение относительно спроса
    **или** наличие истёкших заказов в интервале.
    """

    return (
        row["free_drivers"] / max(row["requests"], 1) < _DEFICIT_RATIO_LIMIT
        or row["unfulfilled"] > 0
    )


def _new_zone_window() -> dict:
    """Пустой накопитель текущего окна одной зоны.

    Ключи: ``requests``/``completed``/``unfulfilled`` — счётчики событий цикла;
    ``waits`` — список ожиданий назначенных заказов; ``price_sum``/``minutes`` —
    сумма поминутных множителей тарифа и число накопленных минут; ``is_rain`` —
    ИЛИ по минутам; ``event`` — первая непустая метка события.
    """

    return {
        "requests": 0,
        "completed": 0,
        "unfulfilled": 0,
        "waits": [],
        "price_sum": 0.0,
        "minutes": 0,
        "is_rain": False,
        "event": "",
    }


class AggregateBuffer:
    """Буфер текущего 5-минутного окна агрегации по каждой зоне (§3.8).

    Жизненный цикл окна: движок каждую сим-минуту вызывает ``set_env`` для каждой
    зоны и счётчики ``add_*`` по событиям шагов цикла; в минуту закрытия окна —
    ``snapshot_drivers`` (снапшот на конец интервала, §10.2) и ``flush`` с меткой
    начала интервала. Семантика колонок — §3.8 дословно.
    """

    def __init__(self, zone_count: int) -> None:
        """Создать пустое окно для зон ``0 .. zone_count - 1``."""

        self._zone_count = zone_count
        # Накопитель текущего окна: по одному dict'у на зону (детали внутреннего
        # состояния — на усмотрение исполнителя, приложение Б).
        self._pending: list[dict] = [_new_zone_window() for _ in range(zone_count)]
        self._free = np.zeros(zone_count, dtype=int)
        self._busy = np.zeros(zone_count, dtype=int)

    def add_request(self, zone: int) -> None:
        """Счётчик созданных заказов интервала (шаг 2 цикла §3.7)."""

        self._pending[zone]["requests"] += 1

    def add_assignment(self, zone: int, wait_min: int) -> None:
        """Запомнить ожидание назначенного заказа: ``wait_min = t − created_at``.

        Среднее по списку даёт колонку ``avg_wait_min``; пустой список — NULL
        (§10.2).
        """

        self._pending[zone]["waits"].append(wait_min)

    def add_completion(self, zone: int) -> None:
        """Счётчик завершённых поездок с ``dropoff_at`` внутри интервала."""

        self._pending[zone]["completed"] += 1

    def add_expiry(self, zone: int) -> None:
        """Счётчик истекших неназначенных заказов интервала."""

        self._pending[zone]["unfulfilled"] += 1

    def snapshot_drivers(self, free: np.ndarray, busy: np.ndarray) -> None:
        """Перезаписать снапшот парка — значение на конец интервала (§10.2).

        Снапшот, а не среднее; копия входных массивов защищает окно от внешних
        мутаций буферов ``Fleet``.
        """

        self._free = np.array(free, dtype=int)
        self._busy = np.array(busy, dtype=int)

    def set_env(self, zone: int, is_rain: bool, event: str, minute_price: float) -> None:
        """Накопить минутное значение среды зоны (вызов на каждую минуту окна).

        ``is_rain`` — ИЛИ по минутам интервала; ``event`` — первая непустая метка
        (``env.event_label`` уже отдаёт метку с приоритетом
        ``friday_evening`` > ``surge_injection``, поэтому «первая непустая»
        эквивалентна приоритету §3.8); ``minute_price`` — поминутный множитель
        тарифа, колонка ``price_multiplier`` — среднее по минутам интервала.
        """

        window = self._pending[zone]
        window["is_rain"] = window["is_rain"] or is_rain
        if event and not window["event"]:
            window["event"] = event
        window["price_sum"] += minute_price
        window["minutes"] += 1

    def flush(self, ts: datetime) -> list[dict]:
        """Закрыть окно: строка-дикт на зону в порядке возрастания id (§3.8).

        ``price_multiplier`` — сумма поминутных значений, делённая на число
        накопленных минут окна; в рабочем цикле движка ``set_env`` вызывается
        для зоны каждую минуту, поэтому знаменатель равен ``engine.agg_step_min``.
        После возврата строк окно обнуляется, включая снапшот парка. Ключи диктов
        — под ``storage.repo.insert_zone_metrics``: ``ts`` — наивный ``datetime``
        (конвертацию в ISO делает repo), остальные — значения колонок.
        """

        rows: list[dict] = []
        for zone_id in range(self._zone_count):
            window = self._pending[zone_id]
            free = int(self._free[zone_id])
            waits = window["waits"]
            avg_wait = sum(waits) / len(waits) if waits else None
            price = window["price_sum"] / window["minutes"] if window["minutes"] else 0.0
            rows.append(
                {
                    "ts": ts,
                    "zone_id": zone_id,
                    "requests": window["requests"],
                    "completed": window["completed"],
                    "unfulfilled": window["unfulfilled"],
                    "free_drivers": free,
                    "busy_drivers": int(self._busy[zone_id]),
                    "avg_wait_min": avg_wait,
                    "supply_demand_ratio": free / max(window["requests"], 1),
                    "is_rain": 1 if window["is_rain"] else 0,
                    "event": window["event"],
                    "price_multiplier": price,
                }
            )
        self._pending = [_new_zone_window() for _ in range(self._zone_count)]
        self._free = np.zeros(self._zone_count, dtype=int)
        self._busy = np.zeros(self._zone_count, dtype=int)
        return rows
