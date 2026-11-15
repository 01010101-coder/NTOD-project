"""Тесты сервера воспроизведения change ``replay-server`` (``app/server.py`` и
``app/__main__.py``, задачи 4.1–4.4, 5.1–5.4).

REST: ``test_api_city`` — структура ответа города (зоны с координатами,
дороги, квадратная матрица маршрутов с диагональю ``[zone]``, задача 4.1);
``test_api_metrics_filters`` — фильтры ``zone``/``from``/``to`` сужают выдачу,
порядок ``(ts, zone_id)`` (задача 4.2). Статика: ``test_static_page`` —
``GET /`` отдаёт страницу city-canvas, скрипты и стиль отдаются статикой
(replay-server 4.3 → city-canvas 2.3). CLI: ``test_cli_
smoke`` — ``python -m app --days 1`` в subprocess, ``GET /`` и ``/api/city``
200, WS-подключение получает ``state``+``snapshot``, процесс останавливается
(задача 4.4). WebSocket (TestClient, тик вручную через
``app.state.replay.tick``): connect получает ``state``+``snapshot``, play/pause
меняют продвижение (5.1); seek переставляет снимок и зажимается за краем (5.2);
скорость зажимается в ``[0.25, 600]`` (5.3); у края лога поток ждёт пополнения
и продолжается (5.4). Мир — синтетический (2 зоны), файлы тестов самостоятельны.
"""

import json
import socket
import sqlite3
import subprocess
import time
import urllib.request
from pathlib import Path

from fastapi.testclient import TestClient

from app.server import EPOCH_LABEL, create_app
from common.config import AppConfig
from common.timeutil import add_minutes, from_iso, to_iso
from simulator.branches import BRANCHES
from storage.repo import insert_forecasts
from storage.schema import init_db

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
SIM_START_DT = from_iso("2026-11-02 00:00:00")

MICRO_ZONES = [
    {
        "id": 0,
        "name": "a",
        "archetype": "residential",
        "attract_origin": 1.0,
        "attract_dest": 1.0,
        "map": {"x": 0.0, "y": 0.0},
    },
    {
        "id": 1,
        "name": "b",
        "archetype": "business",
        "attract_origin": 1.0,
        "attract_dest": 1.0,
        "map": {"x": 10.0, "y": 0.0},
    },
]


def micro_config(**overrides) -> AppConfig:
    """Синтетический мир: 2 зоны, дорога 30 мин, без случайных событий."""

    base = {
        "seed": 7,
        "city": {
            "zones": MICRO_ZONES,
            "roads": [{"from_zone": 0, "to_zone": 1, "minutes": 30.0}],
        },
        "population": {"scale": 0.0},
        "drivers": {"count": 3, "wait_limit_min": 6},
        "weather": {
            "rain_day_chance": 0.0,
            "episode_start_h": [0, 0],
            "episode_duration_h": [1, 1],
        },
        "pricing": {"surge_injection": {"day_chance": 0.0}},
    }
    base.update(overrides)
    return AppConfig.model_validate(base)


def make_client(cfg: AppConfig, days: float) -> TestClient:
    """TestClient без фонового тика: playhead двигают тесты вручную."""

    app = create_app(cfg, days=days, tick_interval=None)
    return TestClient(app)


def wait_filled(client: TestClient, minimum: int, timeout: float = 60.0) -> None:
    """Дождаться заполненности лога ``minimum`` записей (фоновый прогон)."""

    controller = client.app.state.replay
    deadline = time.monotonic() + timeout
    while controller.log.filled < minimum and time.monotonic() < deadline:
        time.sleep(0.02)
    assert controller.log.filled >= minimum, "лог не заполнился за отведённое время"


def active_manual(client: TestClient, minute: int) -> list[dict]:
    """Активные поездки минуты, восстановленные вручную из лога (независимая свёртка)."""

    log = client.app.state.replay.log
    active: dict[int, dict] = {}
    for record in log.read(0, minute + 1):
        for event in record.assignments:
            active[event.internal_id] = {
                "internal_id": event.internal_id,
                "driver_id": event.driver_id,
                "driver_zone": event.driver_zone,
                "origin": event.origin,
                "dest": event.dest,
                "assigned_at": to_iso(record.t),
                "pickup_at": to_iso(event.pickup_at),
                "dropoff_at": to_iso(event.dropoff_at),
            }
        for event in record.dropoffs:
            active.pop(event.internal_id, None)
    return [active[key] for key in sorted(active)]


def pending_manual(client: TestClient, minute: int) -> list[dict]:
    """Неназначенные заказы минуты: спавны ≤ t минус назначенные и истёкшие."""

    log = client.app.state.replay.log
    pending: dict[int, dict] = {}
    for record in log.read(0, minute + 1):
        for event in record.spawns:
            pending[event.internal_id] = {
                "internal_id": event.internal_id,
                "origin": event.origin,
                "created_at": to_iso(record.t),
            }
        for event in record.assignments:
            pending.pop(event.internal_id, None)
        for event in record.expiries:
            pending.pop(event.internal_id, None)
    return [pending[key] for key in sorted(pending)]


def receive_until(ws, type_: str, limit: int = 50) -> dict:
    """Читать сообщения до нужного типа (пропуская log-extended между ними)."""

    for _ in range(limit):
        message = ws.receive_json()
        if message["type"] == type_:
            return message
    raise AssertionError(f"сообщение {type_} не получено за {limit} чтений")


def test_api_city():
    """Ответ /api/city: все зоны с x/y, все дороги, квадратная routes с диагональю (4.1)."""

    cfg = micro_config()
    with make_client(cfg, days=0.2) as client:
        data = client.get("/api/city").json()
    assert [zone["id"] for zone in data["zones"]] == [0, 1]
    for zone, source in zip(data["zones"], cfg.city.zones):
        assert zone["name"] == source.name
        assert zone["archetype"] == source.archetype
        assert zone["x"] == source.map.x and zone["y"] == source.map.y
    assert data["roads"] == [{"from": 0, "to": 1, "minutes": 30.0}]
    routes = data["routes"]
    assert len(routes) == 2 and all(len(row) == 2 for row in routes)
    assert routes[0][0] == [0] and routes[1][1] == [1]
    assert routes[0][1] == [0, 1] and routes[1][0] == [1, 0]


def test_api_metrics_filters():
    """/api/metrics: без фильтров всё доступное, фильтры сужают, порядок (ts, zone_id) (4.2)."""

    cfg = micro_config(population={"scale": 50.0})
    with make_client(cfg, days=0.2) as client:
        wait_filled(client, 288)  # 0.2 суток = 288 минут = 57 полных окон + хвост

        everything = client.get("/api/metrics").json()["metrics"]
        assert everything, "лог метрик пуст"
        stamps = [row["ts"] for row in everything]
        assert stamps == sorted(stamps)
        keyed = [(row["ts"], row["zone_id"]) for row in everything]
        assert keyed == sorted(keyed)
        assert all(row["zone_id"] in (0, 1) for row in everything)

        zone_only = client.get("/api/metrics", params={"zone": 1}).json()["metrics"]
        assert zone_only
        assert all(row["zone_id"] == 1 for row in zone_only)
        assert len(zone_only) < len(everything)

        mid = everything[len(everything) // 2]["ts"]
        window = client.get("/api/metrics", params={"zone": 1, "from": mid, "to": mid}).json()[
            "metrics"
        ]
        assert window and all(row["ts"] == mid for row in window)
        # Inclusivity границ: строка ровно на границе попадает в оба запроса.
        assert client.get("/api/metrics", params={"from": mid}).json()["metrics"][0]["ts"] == mid


def test_static_page():
    """GET / отдаёт страницу city-canvas; скрипты и стиль отдаются статикой (city-canvas 2.3)."""

    with make_client(micro_config(), days=0.2) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert 'data-app="city-canvas"' in response.text
        # m6-worlds-ui: селектор миров, метка активного мира и тумблер каймы.
        assert 'id="world-select"' in response.text
        assert 'id="world-label"' in response.text
        assert 'id="paint-toggle"' in response.text
        scripts = {
            name: client.get(f"/static/{name}") for name in ("app.js", "map.js", "charts.js")
        }
        for name, script in scripts.items():
            assert script.status_code == 200, f"статика {name} недоступна"
        assert "WebSocket" in scripts["app.js"].text
        # Связка селектора с state.world и REST переключения (спека web-city-ui).
        assert "setActiveWorld(message.world)" in scripts["app.js"].text
        assert "/api/worlds" in scripts["app.js"].text
        assert "/switch" in scripts["app.js"].text
        # Окраска прогнозом — карта читает режим и кайму из мира.
        assert "forecastDeficit" in scripts["map.js"].text
        assert "paintMode" in scripts["map.js"].text
        style = client.get("/static/style.css")
        assert style.status_code == 200
        assert "--bg" in style.text


def test_lab_chart_click_seek_switch_order():
    """Seek-связка: клик лаб-графика из мира ветки — switch на Эпоху, затем seek (5)."""

    source = (PROJECT_ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert "labEls['forecast-chart'].addEventListener('click', onLabChartClick)" in source
    start = source.index("const onLabChartClick")
    block = source[start : start + 1200]
    assert "world.activeWorld.name !== EPOCH_WORLD" in block
    assert "switchWorld(EPOCH_WORLD)" in block
    assert block.index("switchWorld(EPOCH_WORLD)") < block.index("seekMinute(target)")
    assert "labMinuteAtClick" in block and "LAB_PAD" in source


def _free_port() -> int:
    """Свободный TCP-порт для smoke-сервера."""

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_cli_smoke():
    """``python -m app --days 1``: HTTP готов, REST отвечает, WS присылает state+snapshot (4.4)."""

    port = _free_port()
    proc = subprocess.Popen(
        [str(PYTHON), "-m", "app", "--days", "1", "--host", "127.0.0.1", "--port", str(port)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 60.0
        ready = False
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{base}/", timeout=2) as response:
                    ready = response.status == 200
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.2)
        assert ready, "сервер не поднялся за 60 с"

        with urllib.request.urlopen(f"{base}/api/city", timeout=5) as response:
            city = json.loads(response.read())
        assert len(city["zones"]) == 6  # эталонный config.yaml репо

        from websockets.sync.client import connect as ws_connect

        with ws_connect(f"ws://127.0.0.1:{port}/ws/replay", open_timeout=10) as sock:
            state = json.loads(sock.recv(timeout=30))
            snapshot = json.loads(sock.recv(timeout=30))
            assert state["type"] == "state"
            assert snapshot["type"] == "snapshot"
            assert state["world"]["seed"] == 42 and state["world"]["zones"] == 6
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def test_ws_connect_play_pause():
    """Connect: state+snapshot; play гонит записи по минутам, pause замирает (5.1)."""

    cfg = micro_config(population={"scale": 50.0})
    with make_client(cfg, days=0.2) as client:
        controller = client.app.state.replay
        wait_filled(client, 20)

        with client.websocket_connect("/ws/replay") as ws:
            state = ws.receive_json()
            snapshot = ws.receive_json()
            assert state["type"] == "state" and snapshot["type"] == "snapshot"
            assert state["index"] == 0
            assert state["playing"] is False and state["speed"] == 1.0
            assert state["world"] == {
                "name": "epoch",
                "label": EPOCH_LABEL,
                "seed": 7,
                "days": 0.2,
                "zones": 2,
                "sim_start": "2026-11-02",
            }
            assert snapshot["env"]["price_mult"] == list(controller.log.record(0).price_mult)
            assert snapshot["env"]["free_by_zone"] == list(controller.log.record(0).free_by_zone)

            ws.send_json({"cmd": "play"})
            assert receive_until(ws, "state")["type"] == "state"

            controller.tick(dt=1.0)  # скорость 1 сим-мин/сек → playhead 1
            batch = receive_until(ws, "records")
            assert [record["t"] for record in batch["records"]] == ["2026-11-02 00:01:00"]
            assert batch["from_index"] == 1

            ws.send_json({"cmd": "pause"})
            assert receive_until(ws, "state")["type"] == "state"
            frozen = controller.playhead
            controller.tick(dt=5.0)  # пауза: позиция не двигается
            assert controller.playhead == frozen

            ws.send_json({"cmd": "play"})
            assert receive_until(ws, "state")["type"] == "state"
            controller.tick(dt=1.0)
            batch = receive_until(ws, "records")
            assert [record["t"] for record in batch["records"]] == ["2026-11-02 00:02:00"]


def test_ws_quiet_start_snapshot_empty():
    """Тихий мир (scale=0): на стартовой минуте нет ни поездок, ни ожидания (city-canvas 1.2)."""

    with make_client(micro_config(), days=0.1) as client:
        wait_filled(client, 5)
        with client.websocket_connect("/ws/replay") as ws:
            state = ws.receive_json()
            snapshot = ws.receive_json()
            assert state["type"] == "state" and state["index"] == 0
            assert snapshot["type"] == "snapshot"
            assert snapshot["active_trips"] == []
            assert snapshot["pending_orders"] == []


def test_ws_seek_snapshot_and_clamp():
    """Seek на известную минуту переставляет снимок; за край зажимается (5.2)."""

    cfg = micro_config(population={"scale": 50.0})
    with make_client(cfg, days=0.3) as client:
        controller = client.app.state.replay
        wait_filled(client, 420)

        with client.websocket_connect("/ws/replay") as ws:
            ws.receive_json(), ws.receive_json()  # state, snapshot минуты 0

            minute = 400
            ws.send_json({"cmd": "seek", "t": "2026-11-02 06:40:00"})  # минута 400
            state = ws.receive_json()
            snapshot = ws.receive_json()
            assert state["type"] == "state" and state["index"] == minute
            assert snapshot["index"] == minute
            record = controller.log.record(minute)
            assert snapshot["t"] == "2026-11-02 06:40:00"
            assert snapshot["env"]["is_rain"] == record.is_rain
            assert snapshot["env"]["price_mult"] == list(record.price_mult)
            assert snapshot["env"]["free_by_zone"] == list(record.free_by_zone)
            expected = active_manual(client, minute)
            assert snapshot["active_trips"] == expected
            assert snapshot["active_trips"], "сценарий без активных поездок не проверен"
            expected_pending = pending_manual(client, minute)
            assert snapshot["pending_orders"] == expected_pending
            assert snapshot["pending_orders"], "сценарий без ожидания не проверен"
            # Оба множества упорядочены по internal_id.
            trip_ids = [trip["internal_id"] for trip in snapshot["active_trips"]]
            order_ids = [order["internal_id"] for order in snapshot["pending_orders"]]
            assert trip_ids == sorted(trip_ids)
            assert order_ids == sorted(order_ids)

            # Seek за заполненный край зажимается к краю (filled - 1).
            edge = controller.log.filled - 1
            beyond = add_minutes(SIM_START_DT, 10_000)
            ws.send_json({"cmd": "seek", "t": beyond.strftime("%Y-%m-%d %H:%M:%S")})
            state = receive_until(ws, "state")
            assert receive_until(ws, "snapshot")["type"] == "snapshot"
            assert state["index"] == edge

            # Seek до старта зажимается в 0.
            ws.send_json({"cmd": "seek", "t": "1999-01-01 00:00:00"})
            state = receive_until(ws, "state")
            assert receive_until(ws, "snapshot")["type"] == "snapshot"
            assert state["index"] == 0


def test_ws_speed_clamp():
    """Скорость зажимается в [0.25, 600], state несёт фактическое значение (5.3)."""

    with make_client(micro_config(population={"scale": 50.0}), days=0.2) as client:
        wait_filled(client, 5)
        with client.websocket_connect("/ws/replay") as ws:
            ws.receive_json(), ws.receive_json()

            ws.send_json({"cmd": "speed", "value": 0.01})
            state = receive_until(ws, "state")
            assert state["type"] == "state" and state["speed"] == 0.25

            ws.send_json({"cmd": "speed", "value": 10000})
            state = receive_until(ws, "state")
            assert state["speed"] == 600.0

            ws.send_json({"cmd": "speed", "value": 3.5})
            state = receive_until(ws, "state")
            assert state["speed"] == 3.5


def test_ws_edge_wait_and_continue():
    """У края незаполненного лога поток ждёт; после дозаполнения продолжается (5.4).

    Прогон длинный (2 суток ≈ несколько секунд фона), поэтому лог ещё растёт,
    когда играющее воспроизведение на скорости 600 упирается в край. Инвариант
    «playhead не опережает край» проверяется после каждого тика — чтение
    заполненности ПОСЛЕ тика даёт верхнюю границу, поэтому гонки с фоновым
    аппендом не мешают: filled монотонно растёт.
    """

    cfg = micro_config(population={"scale": 50.0})
    with make_client(cfg, days=2.0) as client:
        controller = client.app.state.replay
        wait_filled(client, 300)

        with client.websocket_connect("/ws/replay") as ws:
            ws.receive_json(), ws.receive_json()

            ws.send_json({"cmd": "speed", "value": 600})
            receive_until(ws, "state")  # state скорости
            ws.send_json({"cmd": "play"})
            receive_until(ws, "state")  # state

            controller.tick(dt=100.0)  # 600 сим-мин/сек × 100 сек ≫ заполненного края
            edge = int(controller.playhead)
            assert edge >= 300, "воспроизведение не пересекло ожидаемую часть"
            assert edge <= controller.log.filled - 1, "playhead опередил заполненный край"

            # Всё пересечённое пришло пачками с лимитом 200 и без пропусков —
            # ни одной записи за край (поток на краю приостановлен).
            collected: list[int] = []
            while len(collected) < edge:
                message = receive_until(ws, "records")
                assert 0 < len(message["records"]) <= 200
                expected_start = collected[-1] + 1 if collected else 1
                assert message["from_index"] == expected_start
                collected.extend(
                    range(message["from_index"], message["from_index"] + len(message["records"]))
                )
            assert collected == list(range(1, edge + 1))

            # Дозаполнение фоном → следующий тик продолжает ровно с новой записи.
            deadline = time.monotonic() + 60.0
            while controller.log.filled <= edge and time.monotonic() < deadline:
                time.sleep(0.02)
            assert controller.log.filled > edge
            controller.tick(dt=100.0)
            assert int(controller.playhead) > edge, "поток не продолжился после дозаполнения"
            assert int(controller.playhead) <= controller.log.filled - 1
            message = receive_until(ws, "records")
            assert message["from_index"] == edge + 1
            assert message["records"][0]["t"] == to_iso(add_minutes(SIM_START_DT, edge + 1))


# ── Реестр миров и прогнозы мира (change m6-worlds-ui, задачи 1–3) ──


def _write_forecast_db(path: Path, rows: list[dict]) -> None:
    """Мини-БД контракта с заданными строками ``forecasts`` (tmp, не эталон)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        init_db(conn)
        insert_forecasts(conn, rows)
    finally:
        conn.close()


def _wait_log(log, minimum: int, timeout: float = 60.0) -> None:
    """Дождаться заполненности произвольного лога ``minimum`` записей."""

    deadline = time.monotonic() + timeout
    while log.filled < minimum and time.monotonic() < deadline:
        time.sleep(0.02)
    assert log.filled >= minimum, "лог мира не заполнился за отведённое время"


def test_api_worlds_registry(tmp_path):
    """GET /api/worlds: Эпоха первой + 4 ветки реестра с метками и exists (1)."""

    worlds = tmp_path / "worlds"
    worlds.mkdir()
    with TestClient(
        create_app(micro_config(), days=0.05, tick_interval=None, worlds_dir=worlds)
    ) as client:
        wait_filled(client, 1)
        entries = client.get("/api/worlds").json()["worlds"]

    assert [entry["name"] for entry in entries] == [
        "epoch",
        "a-base",
        "b-reactive",
        "c-mlp",
        "d-lstm",
    ]
    assert entries[0]["label"] == EPOCH_LABEL
    assert "rng-разделения" in entries[0]["label"]
    assert entries[0]["exists"] is True and entries[0]["active"] is True
    # Пустой worlds_dir: ветки есть в реестре, но файлов БД нет — не ошибка.
    assert [entry["exists"] for entry in entries[1:]] == [False] * len(BRANCHES)
    assert [entry["label"] for entry in entries[1:]] == [spec.label for spec in BRANCHES.values()]
    assert not any(entry["active"] for entry in entries[1:])


def test_world_switch(tmp_path):
    """POST switch: ленивый мир ветки, позиция на старт, 409/404, WS-рассылка (2)."""

    worlds = tmp_path / "worlds"
    worlds.mkdir()
    (worlds / "a-base.db").write_bytes(b"")  # лог строится прогоном, БД не читается
    with TestClient(
        create_app(micro_config(), days=0.05, tick_interval=None, worlds_dir=worlds)
    ) as client:
        wait_filled(client, 3)
        epoch_controller = client.app.state.replay
        epoch_controller.playhead = 2.0  # повторный выбор мира должен вернуть на старт

        assert client.post("/api/worlds/nope/switch").status_code == 404
        missing = client.post("/api/worlds/c-mlp/switch")
        assert missing.status_code == 409
        assert client.app.state.replay is epoch_controller  # мир не сменился

        response = client.post("/api/worlds/a-base/switch")
        assert response.status_code == 200
        state = response.json()
        assert state["world"]["name"] == "a-base"
        assert state["world"]["label"] == BRANCHES["a-base"].label
        assert state["position"] == 0.0 and state["index"] == 0

        branch_controller = client.app.state.replay
        assert branch_controller is not epoch_controller
        assert branch_controller.meta["name"] == "a-base"
        _wait_log(branch_controller.log, 3)  # лог ветки растёт фоном

        # Воспроизведение (tick/seek) идёт на новом контроллере.
        branch_controller.playing = True
        branch_controller.tick(dt=1.0)
        assert branch_controller.playhead > 0.0
        branch_controller.playing = False
        assert client.get("/api/worlds").json()["worlds"][1]["active"] is True

        with client.websocket_connect("/ws/replay") as ws:
            ws_state = receive_until(ws, "state")
            assert ws_state["world"]["name"] == "a-base"
            receive_until(ws, "snapshot")
            # Переключение обратно рассылает state+snapshot живой WS-сессии.
            assert client.post("/api/worlds/epoch/switch").status_code == 200
            switched = receive_until(ws, "state")
            assert switched["world"]["name"] == "epoch"
            assert switched["world"]["label"] == EPOCH_LABEL
            assert switched["position"] == 0.0  # мир всегда начинается с начала
            snapshot = receive_until(ws, "snapshot")
            assert snapshot["type"] == "snapshot" and snapshot["index"] == 0


def _forecast_row(ts: str, version: str, zone: int, horizon: int) -> dict:
    """Строка forecasts контракта для тестов (значения не важны)."""

    return {
        "ts": from_iso(ts),
        "zone_id": zone,
        "horizon_min": horizon,
        "pred_requests": 10.0,
        "pred_free_drivers": 2.0,
        "deficit_prob": 0.7 if zone == 0 else 0.2,
        "model_version": version,
    }


def test_world_forecasts(tmp_path):
    """GET прогнозов мира: обе семьи с границы, Эпоха на ml.db, 404/пусто (3)."""

    worlds = tmp_path / "worlds"
    worlds.mkdir()
    boundary = "2026-12-02 00:00:00"  # день 31: окно активации тени
    branch_rows = [
        _forecast_row(ts, version, zone, horizon)
        for ts in (boundary, "2026-12-02 00:05:00")
        for version in ("mlp", "lstm")
        for zone in (0, 1)
        for horizon in (15, 30)
    ]
    _write_forecast_db(worlds / "c-mlp.db", branch_rows)

    ml_db = tmp_path / "ml.db"
    epoch_rows = [
        _forecast_row(boundary, version, zone, horizon)
        for version in ("mlp-v1-2026-11-07", "lstm-v1-2026-11-07")
        for zone in (0, 1)
        for horizon in (15, 30)
    ]
    _write_forecast_db(ml_db, epoch_rows)
    cfg = micro_config()
    cfg = cfg.model_copy(update={"ml": cfg.ml.model_copy(update={"db": str(ml_db)})})

    with TestClient(create_app(cfg, days=0.05, tick_interval=None, worlds_dir=worlds)) as client:
        empty = client.get(
            "/api/worlds/c-mlp/forecasts",
            params={"from": "2026-12-01 23:55:00", "to": "2026-12-01 23:55:00"},
        )
        assert empty.status_code == 200 and empty.json()["rows"] == []

        data = client.get("/api/worlds/c-mlp/forecasts", params={"zone": 0}).json()
        assert {row["model_version"] for row in data["rows"]} == {"mlp", "lstm"}
        assert all(row["horizon_min"] == 15 for row in data["rows"])  # policy.horizon_min
        assert {row["ts"] for row in data["rows"]} == {boundary, "2026-12-02 00:05:00"}
        assert all(row["zone_id"] == 0 for row in data["rows"])
        assert set(data["rows"][0]) == {
            "ts",
            "zone_id",
            "horizon_min",
            "deficit_prob",
            "model_version",
        }

        wide = client.get("/api/worlds/c-mlp/forecasts", params={"zone": 1, "horizon": 30})
        assert wide.status_code == 200
        assert {row["horizon_min"] for row in wide.json()["rows"]} == {30}

        epoch = client.get("/api/worlds/epoch/forecasts").json()["rows"]
        assert {row["model_version"] for row in epoch} == {
            "mlp-v1-2026-11-07",
            "lstm-v1-2026-11-07",
        }
        assert all("-v1-" in row["model_version"] for row in epoch)

        # Неизвестное имя — 404; мир без файла БД — пустой список без ошибки.
        assert client.get("/api/worlds/nope/forecasts").status_code == 404
        assert client.get("/api/worlds/b-reactive/forecasts").json()["rows"] == []
