"""Калибровка эталона §9 (change ``m2-calibration``).

Запуск: ``python -m simulator.calibrate --days 30 [--seed S]`` (дефолты: 30 суток,
seed из конфига). Выполняет честный прогон мира с внутренним writer'ом, копящим
только строки метрик; по строкам этого единственного прогона строятся
matplotlib-графики в ``reports/`` — суточный профиль по зонам, ряд с дождём и
гистограмма ``free_drivers`` (§9, не меньше трёх PNG), — и печатается итог одной
машинной строкой ``deficit_share=<X>%`` — доля дефицитных 5-минуток по правилу
§3.8, вычисленная той же приватной функцией ``simulator.aggregate._is_deficit``,
что и ``Summary.deficit_share``. Эту строку, PNG и отчёт проверяет критерий
приёмки A13. pandas и matplotlib разрешены только в этом модуле (§2.2 п. 2);
параметры мира берутся из конфига — магических чисел мира в коде нет (§2.2 п. 6).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-бэкенд до импорта pyplot: команда работает без дисплея
import matplotlib.pyplot as plt  # noqa: E402 — импорт строго после выбора бэкенда
import pandas as pd  # noqa: E402 — своды метрик прогона разрешены только здесь

from common.config import AppConfig, load_config
from simulator.aggregate import _is_deficit
from simulator.engine import Engine

# Каталог графиков зафиксирован §9 (аргумент «--out» планом не назван).
_REPORTS_DIR = Path("reports")


class _MetricsWriter:
    """Внутренний writer калибровки: копит только строки метрик прогона.

    Строки trips калибровке не нужны — метод-заглушка, движок вызывает шов
    ``write_trips`` на каждой часовой границе независимо от интереса потребителя.
    """

    def __init__(self) -> None:
        self.metrics: list[dict] = []

    def write_trips(self, rows: list[dict]) -> None:
        """Пустая операция: строки trips калибровка не собирает."""

    def write_metrics(self, rows: list[dict]) -> None:
        """Накопить батч закрытых 5-минутных окон."""

        self.metrics.extend(rows)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Аргументы инструмента §9: ``--days`` (по умолчанию 30), ``--seed``."""

    parser = argparse.ArgumentParser(
        prog="simulator.calibrate",
        description="Прогон мира, графики в reports/ и печать deficit_share (§9).",
    )
    parser.add_argument("--days", type=int, default=30, help="число суток прогона")
    parser.add_argument("--seed", type=int, default=None, help="seed (по умолчанию из конфига)")
    return parser.parse_args(argv)


def _zone_labels(cfg: AppConfig) -> list[str]:
    """Подписи линий зон: ``"<id> <имя>"`` в порядке возрастания id."""

    return [f"{zone.id} {zone.name}" for zone in cfg.city.zones]


def _plot_daily_profile(df: pd.DataFrame, cfg: AppConfig, labels: list[str]) -> None:
    """График «суточный профиль по зонам»: средние почасовые ``requests`` (§11.2 п. 2).

    Три панели — будни пн–чт, пятница (событие вечернего горба) и выходные:
    волна §3.1 (утренний отток residential, вечерний магнит business/mixed) и
    «пятничный вечер» из ``events.friday_evening`` читаются глазами. Дни групп —
    усредняются, внутри дня суммируются 12 окон часа.
    """

    frame = df.assign(day=df["ts"].dt.date, hour=df["ts"].dt.hour, wd=df["ts"].dt.day_of_week)
    sums = frame.groupby(["day", "hour", "zone_id"], as_index=False)["requests"].sum()
    wd_by_day = frame.groupby("day")["wd"].first()
    sums = sums.assign(wd=sums["day"].map(wd_by_day))
    event = cfg.events.friday_evening
    panels = [
        ("Будни пн–чт", sums["wd"] <= 3),
        (f"Пятница (событие {event.hours[0]}–{event.hours[1]})", sums["wd"] == 4),
        ("Выходные сб–вс", sums["wd"] >= 5),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (title, mask) in zip(axes, panels):
        part = sums[mask]
        if part.empty:
            ax.text(0.5, 0.5, "нет данных", ha="center", va="center", transform=ax.transAxes)
        else:
            avg = part.groupby(["hour", "zone_id"])["requests"].mean().unstack("zone_id")
            for zone_id, label in enumerate(labels):
                series = avg[zone_id] if zone_id in avg.columns else None
                if series is not None:
                    ax.plot(series.index, series.values, marker="o", markersize=3, label=label)
        ax.set_title(title)
        ax.set_xlabel("час суток")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("заказов/час (среднее по дням)")
            ax.legend(fontsize=8)
    if event.hours[0] <= event.hours[1]:
        axes[1].axvspan(
            event.hours[0], event.hours[1] + 1, color="tab:orange", alpha=0.15, zorder=0
        )
    fig.suptitle("Суточный профиль спроса по зонам (из строк метрик прогона)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(_REPORTS_DIR / "daily_profile.png", dpi=150)
    plt.close(fig)


def _plot_rain_series(df: pd.DataFrame, cfg: AppConfig) -> None:
    """График «ряд с дождём»: почасовые заказы города с выделением ``is_rain`` (§3.6).

    Эпизоды дождя (множители спроса и времени из ``weather``) затеняются на оси
    времени, чтобы горбы/провалы читались относительно сухих участков.
    """

    hourly = df.groupby(df["ts"].dt.floor("h"))["requests"].sum()
    rain = df.groupby(df["ts"].dt.floor("h"))["is_rain"].max()
    step = pd.Timedelta(hours=1)
    # Непрерывные часы с дождём объединяются в эпизоды (спаны) для заливки.
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    run_start: pd.Timestamp | None = None
    prev: pd.Timestamp | None = None
    for ts in (ts for ts, flag in rain.items() if flag):
        if run_start is None:
            run_start = ts
        elif ts - prev != step:
            spans.append((run_start, prev + step))
            run_start = ts
        prev = ts
    if run_start is not None:
        spans.append((run_start, prev + step))

    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.plot(hourly.index, hourly.values, color="tab:blue", linewidth=1.0, label="заказы города")
    for start, end in spans:
        ax.axvspan(start, end, color="tab:cyan", alpha=0.3, zorder=0)
    ax.set_title("Почасовой ряд спроса города и эпизоды дождя")
    ax.set_xlabel("дата")
    ax.set_ylabel("заказов/час (город)")
    ax.grid(alpha=0.3)
    handles, names = ax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color="tab:cyan", alpha=0.3))
    names.append(f"дождь (спрос ×{cfg.weather.demand_mult}, время ×{cfg.weather.travel_mult})")
    ax.legend(handles, names, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(_REPORTS_DIR / "rain_series.png", dpi=150)
    plt.close(fig)


def _plot_free_drivers_hist(df: pd.DataFrame) -> None:
    """График «гистограмма free_drivers»: снапшоты окон по всем зонам (§9).

    Распределение показывает, что парк «есть, но впритык»: хвост малых значений
    соотносится с долей дефицитных окон (10–20 % после подбора).
    """

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(df["free_drivers"], bins=30, color="tab:green", edgecolor="black", alpha=0.8)
    ax.set_title("Распределение снапшотов free_drivers (все окна и зоны прогона)")
    ax.set_xlabel("free_drivers (снапшот на конец 5-минутного окна)")
    ax.set_ylabel("число окон-зон")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_REPORTS_DIR / "free_drivers_hist.png", dpi=150)
    plt.close(fig)


def _park_occupancy(df: pd.DataFrame) -> tuple[float, float]:
    """Занятость парка по снапшотам окон (§9, решение 3 design.md): средняя и p95.

    Формула — ``busy / (busy + free)``; снапшоты по всем зонам окна суммируются
    в городское значение (по окнам города), средняя и верхняя перцентиль —
    ориентиры подбора §9 (avg ~60–70 %, пик ~90 %+), фиксируются в отчёте.
    """

    by_window = df.groupby("ts")[["free_drivers", "busy_drivers"]].sum()
    total = by_window.sum(axis=1)
    occupancy = by_window["busy_drivers"] / total.where(total > 0)
    return float(occupancy.mean()), float(occupancy.quantile(0.95))


def main(argv: list[str] | None = None) -> int:
    """Прогон мира, три PNG в ``reports/`` и машинная строка ``deficit_share=<X>%``."""

    args = _parse_args(argv)
    cfg: AppConfig = load_config("config.yaml")
    writer = _MetricsWriter()
    summary = Engine(cfg, seed=args.seed).run(args.days, writer)
    df = pd.DataFrame(writer.metrics)
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        _plot_daily_profile(df, cfg, _zone_labels(cfg))
        _plot_rain_series(df, cfg)
        _plot_free_drivers_hist(df)
        occ_mean, occ_p95 = _park_occupancy(df)
        # Человеческая сводка итераций подбора (§9): занятость и wall; замер
        # wall ведёт движок по периметру ``run`` (§2.2 п. 3 — только отчётность).
        print(
            f"парк: занятость средняя {100 * occ_mean:.1f}%, p95 {100 * occ_p95:.1f}%; "
            f"wall {summary.wall_sec:.1f} с"
        )
    deficit_rows = sum(1 for row in writer.metrics if _is_deficit(row))
    total = len(writer.metrics)
    share = 100.0 * deficit_rows / total if total else 0.0
    print(f"deficit_share={share:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
