"""Тесты чистого правила вызова подработчиков (change ``world-elastic-supply``).

Покрывают сценарии спеки ``simulator-callout``: вызов пяти водителей из
резерва 60 в зону с надбавкой 1.3, малый прирост ниже порога, потолок доли,
деление резерва между зонами по возрастанию ``zone_id``, ровный сигнал без
вызова, детерминизм и неизменность входа. Правило не использует rng; вход —
рукописный резерв, сигналы зон и параметры ``drivers.callout``.
"""

from common.config import CalloutCfg
from simulator.callout import plan_callouts

ZONE_COUNT = 6


def _config(**overrides: float) -> CalloutCfg:
    """Параметры правила из спеки; ``overrides`` меняют отдельные поля."""

    params = {
        "min_gain": 0.10,
        "sensitivity": 0.3,
        "max_share": 0.3,
    }
    params.update(overrides)
    return CalloutCfg(**params)


def _signals(overrides: dict[int, float]) -> list[float]:
    """Сигнал всех зон 1.0, кроме переопределённых."""

    signals = [1.0] * ZONE_COUNT
    for zone_id, value in overrides.items():
        signals[zone_id] = value
    return signals


def test_five_drivers_to_surcharge_zone() -> None:
    """Сценарий «Вызов в зону с надбавкой»: 5 из 60 в зону 5, id 240–244."""

    pool = list(range(240, 300))
    assert plan_callouts(pool, _signals({5: 1.3}), _config()) == [
        (240, 5),
        (241, 5),
        (242, 5),
        (243, 5),
        (244, 5),
    ]


def test_small_gain_does_not_call() -> None:
    """Сценарий «Малый прирост не вызывает»: 1.05 при ``min_gain = 0.10``."""

    pool = list(range(240, 300))
    assert plan_callouts(pool, _signals({1: 1.05}), _config()) == []


def test_share_is_capped_by_max_share() -> None:
    """Сценарий «Доля ограничена сверху»: сигнал 3.0 при потолке 0.3 — 18 из 60."""

    pool = list(range(240, 300))
    calls = plan_callouts(pool, _signals({3: 3.0}), _config())
    assert len(calls) == 18
    assert [driver_id for driver_id, _zone in calls] == list(range(240, 258))
    assert {zone for _driver_id, zone in calls} == {3}


def test_zones_share_pool_in_zone_order() -> None:
    """Сценарий «Несколько зон делят резерв по порядку»: зона 2 забирает всё."""

    pool = [250, 251, 252]
    calls = plan_callouts(pool, _signals({2: 2.0, 4: 2.0}), _config(sensitivity=1.0, max_share=1.0))
    assert calls == [(250, 2), (251, 2), (252, 2)]


def test_flat_signal_returns_empty() -> None:
    """Сценарий «Ровный сигнал — нет вызова»: все зоны 1.0."""

    pool = list(range(240, 300))
    assert plan_callouts(pool, [1.0] * ZONE_COUNT, _config()) == []


def test_empty_pool_returns_empty() -> None:
    """Пустой резерв — вызова нет, даже при высоком сигнале."""

    assert plan_callouts([], _signals({5: 2.0}), _config()) == []


def test_deterministic_and_input_unchanged() -> None:
    """Сценарий «Повторный вызов правила»: списки равны, вход не изменён."""

    pool = list(range(240, 300))
    signals = _signals({2: 1.6, 4: 1.2})
    first = plan_callouts(pool, signals, _config())
    second = plan_callouts(pool, signals, _config())
    assert first == second
    assert pool == list(range(240, 300))
    assert signals == _signals({2: 1.6, 4: 1.2})
    # Порядок выхода: зоны по возрастанию, внутри зоны — id по возрастанию.
    assert [zone for _driver_id, zone in first] == sorted(zone for _driver_id, zone in first)


def test_split_pool_between_zones() -> None:
    """Две зоны делят резерв по порядку: первая забирает свою долю, вторая — остаток."""

    pool = list(range(280, 300))
    # 20 водителей, sensitivity 1.0: зона 1 при 1.1 просит 2, зона 5 при 2.0 — 20,
    # но резерва осталось 18 — забирает остаток.
    calls = plan_callouts(pool, _signals({1: 1.1, 5: 2.0}), _config(sensitivity=1.0, max_share=1.0))
    assert calls[:2] == [(280, 1), (281, 1)]
    assert [driver_id for driver_id, _zone in calls[2:]] == list(range(282, 300))
    assert {zone for _driver_id, zone in calls[2:]} == {5}
