"""Прототип 3а: нейросеть, учитывающая действие (вред и спрос при тарифе m), и контур на ней.

1. Эпоха исследования: мир с рандомизированной надбавкой в каждой зона-пятиминутке
   (как A/B-эксперименты платформ с ценой) — эффект цены выучивается без смещения.
2. Сеть: признаки окна 3 ч + тариф m бакета решения → пуассоновские интенсивности
   потерь (``unfulfilled``) и заказов (``requests``) в бакетах решения b..b+2 (15 мин).
3. Контур: для m ∈ {1, 1.15, 1.3, 1.5} — J(m) = вред(m) + c · (заказы(1) − заказы(m));
   выбирается argmin. Цена c за отпугнутого пассажира — ручка кривой.

Признаки при обучении и в контуре считает одна функция :func:`window_features`.
"""

import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import start_conditions_lab as L
from ml.dataset import bucket_index
from simulator.engine import Engine
from simulator.policy import LiveRowSource, PolicyContour, compose_tariff

STEPS = 36  # окно 3 ч
AHEAD = 3  # бакетов цели: b..b+2
MULTS = (1.0, 1.15, 1.30, 1.50)
EXPLORE_P = (0.55, 0.15, 0.15, 0.15)
N_ZONES = 6


def window_features(win, hour, dow, zone, ret15, ret30) -> np.ndarray:
    """Признаки решения: ``win`` (N, 36, 10) — окно ROW_MANIFEST, последний шаг — закрытый
    бакет; ``hour``/``dow`` — момент решения; ``ret*`` — уже назначенные высадки в зону."""

    win = np.asarray(win, dtype=float)
    last = win[:, -1, :]
    s12 = win[:, -12:, :3].sum(axis=1)
    s36 = win[:, :, :3].sum(axis=1)
    prev12 = win[:, -24:-12, 0].sum(axis=1)
    cols = [
        last,
        s12,
        s36,
        win[:, -3:, 3].mean(axis=1, keepdims=True),
        win[:, -2, 9:10],
        win[:, -3, 9:10],
        win[:, -12:, 7].mean(axis=1, keepdims=True),
        (s12[:, 0] - prev12)[:, None],
        np.sin(2 * np.pi * np.asarray(hour) / 24)[:, None],
        np.cos(2 * np.pi * np.asarray(hour) / 24)[:, None],
        np.eye(7)[np.asarray(dow, dtype=int)],
        np.eye(N_ZONES)[np.asarray(zone, dtype=int)],
        np.asarray(ret15, dtype=float)[:, None],
        np.asarray(ret30, dtype=float)[:, None],
    ]
    return np.concatenate(cols, axis=1)


def action_inputs(m) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    return np.stack([np.log(m), m - 1.0], axis=1)


# --- эпоха исследования -----------------------------------------------------------------


class ExplorationPricer:
    """Случайная надбавка на каждую зона-пятиминутку; детерминирована seed'ом."""

    def __init__(self, cfg, seed: int) -> None:
        self._cap = cfg.policy.total_cap
        self._step = cfg.engine.agg_step_min
        self._seed = seed
        self._cache: dict = {}
        self.world = None

    def price(self, zone_id: int, t) -> float:
        key = (zone_id, bucket_index(t, self._step))
        mult = self._cache.get(key)
        if mult is None:
            rng = np.random.default_rng([self._seed, 7919, zone_id, key[1]])
            mult = float(rng.choice(MULTS, p=EXPLORE_P))
            self._cache[key] = mult
        return compose_tariff(self.world(zone_id, t), mult, self._cap)


def exploration_run(overrides: dict, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Эпоха исследования (кэш в каталоге мира): метрики и завершённые поездки."""

    wd = L.WORK_DIR / L.world_key(overrides)
    pm, pt = wd / f"explore-metrics-{seed}.pkl", wd / f"explore-trips-{seed}.pkl"
    if pm.exists() and pt.exists():
        return pd.read_pickle(pm), pd.read_pickle(pt)
    cfg = L.make_cfg(overrides)
    pricer = ExplorationPricer(cfg, seed)
    engine = Engine(cfg, seed=seed, price_fn=pricer.price)
    pricer.world = engine.world_tariff
    writer = L.MemWriter()
    engine.run(cfg.ml.train_days + cfg.ml.test_days, writer)
    metrics = pd.DataFrame(writer.metrics)
    trips = pd.DataFrame(writer.trips)
    trips = trips[trips.status == "completed"][["dest_zone", "assigned_at", "dropoff_at"]]
    metrics.to_pickle(pm)
    trips.to_pickle(pt)
    return metrics, trips


def build_dataset(metrics: pd.DataFrame, trips: pd.DataFrame, step: int = 5) -> dict:
    """Примеры (зона, бакет решения b): признаки окна ≤ b−1, тариф бакета b, цели b..b+2."""

    metrics = metrics.copy()
    metrics["ts"] = pd.to_datetime(metrics.ts)
    metrics["b"] = [bucket_index(t, step) for t in metrics.ts.dt.to_pydatetime()]
    cols = [
        "requests",
        "completed",
        "unfulfilled",
        "free_drivers",
        "busy_drivers",
        "avg_wait_min",
        "supply_demand_ratio",
        "is_rain",
        "is_event",
        "price_multiplier",
    ]
    metrics["avg_wait_min"] = metrics.avg_wait_min.fillna(0.0)
    metrics["is_event"] = (metrics.event.fillna("") != "").astype(float)
    metrics["is_rain"] = metrics.is_rain.astype(float)
    # ret: высадки в зону в (c, c+K], назначенные не позднее начала закрытого бакета c
    trips = trips.copy()
    a_min = np.array([bucket_index(t, 1) for t in pd.to_datetime(trips.assigned_at).dt.to_pydatetime()])
    d_b = np.array([bucket_index(t, step) for t in pd.to_datetime(trips.dropoff_at).dt.to_pydatetime()])
    b_lo, b_hi = metrics.b.min(), metrics.b.max()
    out = {k: [] for k in ("X", "m", "harm", "req", "b", "zone")}
    for zone, g in metrics.groupby("zone_id"):
        g = g.sort_values("b")
        assert (np.diff(g.b) == 1).all()
        arr = g[cols].to_numpy(dtype=float)
        bs = g.b.to_numpy()
        n = len(bs)
        ret = {}
        sel = trips.dest_zone.to_numpy() == zone
        for k_steps in (3, 6):
            diff = np.zeros(b_hi - b_lo + 3)
            lo = np.maximum(-(-a_min[sel] // step), d_b[sel] - k_steps) - b_lo
            hi = d_b[sel] - 1 - b_lo
            ok = hi >= lo
            np.add.at(diff, np.clip(lo[ok], 0, len(diff) - 1), 1)
            np.add.at(diff, np.clip(hi[ok] + 1, 0, len(diff) - 1), -1)
            ret[k_steps] = np.cumsum(diff)[: n]
        idx = np.arange(STEPS, n - AHEAD + 1)  # позиция бакета решения b
        win = np.lib.stride_tricks.sliding_window_view(arr, (STEPS, arr.shape[1]))[:, 0]
        win = win[idx - STEPS]  # окно — бакеты b−36..b−1
        dec_ts = g.ts.to_numpy()[idx]
        hours = pd.DatetimeIndex(dec_ts).hour + pd.DatetimeIndex(dec_ts).minute / 60
        out["X"].append(
            window_features(win, hours, pd.DatetimeIndex(dec_ts).weekday, np.full(len(idx), zone),
                            ret[3][idx - 1], ret[6][idx - 1])
        )
        out["m"].append(arr[idx, 9])
        out["harm"].append(np.stack([arr[idx + j, 2] for j in range(AHEAD)]).sum(axis=0))
        out["req"].append(np.stack([arr[idx + j, 0] for j in range(AHEAD)]).sum(axis=0))
        out["b"].append(bs[idx])
        out["zone"].append(np.full(len(idx), zone))
    return {k: np.concatenate(v) for k, v in out.items()}


# --- сеть -------------------------------------------------------------------------------


def _net(n_in: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 2),
    )


def train_action_model(data: dict, train_mask, val_mask, out_dir: Path, seed: int = 0) -> dict:
    import torch

    torch.manual_seed(seed)
    X = data["X"]
    mu, sd = X[train_mask].mean(0), X[train_mask].std(0) + 1e-6
    def tensors(mask):
        xin = np.concatenate([(X[mask] - mu) / sd, action_inputs(data["m"][mask])], axis=1)
        y = np.stack([data["harm"][mask], data["req"][mask]], axis=1)
        return torch.tensor(xin, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    xt, yt = tensors(train_mask)
    xv, yv = tensors(val_mask)
    net = _net(xt.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.PoissonNLLLoss(log_input=True, full=False)
    best, best_state, bad = math.inf, None, 0
    for epoch in range(60):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 512):
            j = perm[i : i + 512]
            out = net(xt[j])
            loss = loss_fn(out[:, 0], yt[j, 0]) + loss_fn(out[:, 1], yt[j, 1])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            out = net(xv)
            vl = float(loss_fn(out[:, 0], yv[:, 0]) + loss_fn(out[:, 1], yv[:, 1]))
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "mu": mu, "sd": sd, "n_in": xt.shape[1]}, out_dir / "model.pt")
    return {"val_loss": best, "epochs": epoch + 1}


class ActionModel:
    def __init__(self, path: Path) -> None:
        import torch

        blob = torch.load(path, weights_only=False)
        self._net = _net(blob["n_in"])
        self._net.load_state_dict(blob["state"])
        self._net.eval()
        self._mu, self._sd = blob["mu"], blob["sd"]

    def predict(self, X: np.ndarray, m: np.ndarray) -> np.ndarray:
        """(N, 2): ожидаемые потери и заказы в бакетах b..b+2 при тарифе m."""

        import torch

        xin = np.concatenate([(X - self._mu) / self._sd, action_inputs(m)], axis=1)
        with torch.no_grad():
            return np.exp(self._net(torch.tensor(xin, dtype=torch.float32)).numpy())


def model_path(overrides: dict) -> Path:
    return L.WORK_DIR / L.world_key(overrides) / "action-model" / "model.pt"


# --- контур -----------------------------------------------------------------------------


class ActionContour(PolicyContour):
    """Выбор тарифа по сети, учитывающей действие: argmin вред(m) + c·отпугнутые(m)."""

    def __init__(self, cfg, source: LiveRowSource, model: ActionModel, cost: float) -> None:
        super().__init__(cfg, None, source)
        self._model = model
        self._cost = cost
        self.mults_chosen: list[float] = []

    def _multiplier(self, zone_id: int, ts) -> float:
        win = self._source.metrics_window(zone_id, ts, STEPS)
        if win is None:
            return 1.0
        dec = ts + timedelta(minutes=self._step)
        r15 = self._source.dropoff_counts(zone_id, ts, ts + timedelta(minutes=15), ts)
        r30 = self._source.dropoff_counts(zone_id, ts, ts + timedelta(minutes=30), ts)
        X = window_features(win[None], [dec.hour + dec.minute / 60], [dec.weekday()], [zone_id],
                            [r15], [r30])
        world = self._world_tariff(zone_id, dec)
        acts = np.array([compose_tariff(world, m, self._total_cap) for m in MULTS])
        pred = self._model.predict(np.repeat(X, len(MULTS), axis=0), acts)
        J = pred[:, 0] + self._cost * (pred[0, 1] - pred[:, 1])
        mult = MULTS[int(np.argmin(J))]
        self.mults_chosen.append(mult)
        return mult


def action_point(args: tuple) -> dict:
    """``(overrides, cost, seed)`` → итоги тест-недели ветки с контуром на сети действия."""

    L.torch_threads()
    overrides, cost, seed = args
    cfg = L.make_cfg(overrides)
    source = LiveRowSource(agg_step_min=cfg.engine.agg_step_min)
    contour = ActionContour(cfg, source, ActionModel(model_path(overrides)), cost)
    row = L.outcomes(L.run_world(cfg, contour=contour, seed=seed), cfg)
    chosen = np.array(contour.mults_chosen) if contour.mults_chosen else np.array([1.0])
    return row | {"pol": "action", "param": cost, "seed": seed, "fire": 100.0 * contour.fire_rate,
                  "mean_mult": float(chosen.mean()),
                  **{f"share_{m}": float((chosen == m).mean()) for m in MULTS}}
