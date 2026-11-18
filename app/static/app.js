// Живой город (change city-canvas, задача 2.2): шина данных страницы.
// WS-клиент протокола воспроизведения (state/snapshot/records/log-extended,
// команды play/pause/speed/seek), хранилище мира, локальные сим-часы с
// серверной сверкой, идемпотентное применение записей по from_index и
// rAF-цикл кадра. Отрисовка — CityMap (map.js) и CityCharts (charts.js):
// оба читают World и зовут App.seekMinute, своей логики связи не имеют.
// Клиент — чистая функция сим-времени (design.md, решения 1–4): механика
// движка не дублируется, мир после seek восстанавливается снимком.

(() => {
  'use strict';

  const AGG_STEP = 5; // шаг агрегации метрик, минут (контракт данных §2)
  const DEFICIT_RATIO_LIMIT = 0.5; // правило дефицита §3.8: free/max(req,1) <
  const FLASH_TTL = 5; // TTL красной вспышки истечения, сим-минут
  const CLOCK_SYNC_LIMIT = 1; // расхождение часов с сервером, больше — жёсткий синкинг
  const EPOCH_WORLD = 'epoch'; // имя мира «Эпоха» в реестре (m6-worlds-ui)
  const DEFICIT_PROB_LIMIT = 0.5; // кайма прогноза: deficit_prob ≥ порога (спека)
  const FORECAST_BLOCK_MIN = 360; // кэш прогнозов блоками по 6 сим-часов
  const LAB_PAD = { left: 48, right: 14, top: 26, bottom: 22 }; // геометрия лаб-графика

  const WEEKDAYS = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье'];
  const WEEKDAYS_SHORT = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'];
  const dayPeriod = (hour) =>
    hour < 5 ? 'ночь' : hour < 8 ? 'рассвет' : hour < 12 ? 'утро' : hour < 17 ? 'день' : hour < 21 ? 'сумерки' : 'ночь';

  // ── Хранилище мира: единственный источник состояния для отрисовки ──
  const world = {
    cityReady: false, // /api/city загружен
    zones: [], // {id, name, archetype, x, y} из /api/city
    roads: [], // {from, to, minutes}
    routes: [], // routes[i][j] = последовательность зон
    simStartUTC: null, // epoch ms полуночи sim_start (арифметика UTC — наивные времена)
    playing: false,
    speed: 1,
    position: 0, // серверная позиция playhead, дробные сим-минуты
    clock: 0, // локальные сим-часы, минут от старта мира
    filled: { count: 0, first: null, last: null },
    freeByZone: [],
    priceMult: [],
    isRain: false,
    event: '', // метка события текущего окна (заполняет charts.js из метрик)
    activeTrips: new Map(), // internal_id → {driverZone, origin, dest, assignedAt, pickupAt, dropoffAt}
    pending: new Map(), // internal_id → {origin, createdAt}
    winRequests: [], // счётчики спавнов текущего окна по зонам
    winExpiries: [], // счётчики истечений текущего окна по зонам
    deficit: [], // кайма дефицита по зонам (после закрытия окна)
    forecastDeficit: {}, // zone_id → true: прогнозная кайма ближайшего закрытого бакета
    flashes: [], // [{zone, born}] — вспышки истечений, born в сим-минутах
    appliedUpTo: -1, // индекс последней применённой записи
    // Реестр миров (change m6-worlds-ui): перечень, активный мир и режим каймы.
    availableWorlds: [], // ответ GET /api/worlds
    activeWorld: null, // {name, label} из state.world
    awaitingSnapshot: false, // после switch записи старого мира не применяются
    paintMode: 'fact', // 'fact' | 'forecast' — тумблер окраски зон
    forecastBucket: null, // минута последнего закрытого бакета, для которого красили
    forecastFetched: new Set(), // ключи мир|версия|блок уже запрошенных прогнозов
    forecastBuckets: new Map(), // минута бакета → строки прогноза выбранной модели
  };

  // ── Время: наивные ISO-минуты ↔ индекс сим-минуты через UTC-арифметику ──

  const setSimStart = (dateStr) => {
    const [y, m, d] = dateStr.split('-').map(Number);
    world.simStartUTC = Date.UTC(y, m - 1, d);
  };

  const isoToMinute = (iso) => {
    const [date, time] = iso.split(' ');
    const [y, m, d] = date.split('-').map(Number);
    const [h, min] = time.split(':').map(Number);
    return (Date.UTC(y, m - 1, d, h, min) - world.simStartUTC) / 60000;
  };

  const minuteToIso = (minute) => {
    const dt = new Date(world.simStartUTC + Math.round(minute) * 60000);
    const pad = (n) => String(n).padStart(2, '0');
    return (
      `${dt.getUTCFullYear()}-${pad(dt.getUTCMonth() + 1)}-${pad(dt.getUTCDate())} ` +
      `${pad(dt.getUTCHours())}:${pad(dt.getUTCMinutes())}:00`
    );
  };

  const clockParts = (minute) => {
    const dt = new Date(world.simStartUTC + minute * 60000);
    const pad = (n) => String(n).padStart(2, '0');
    return {
      time: `${pad(dt.getUTCHours())}:${pad(dt.getUTCMinutes())}`,
      date: `${pad(dt.getUTCDate())}.${pad(dt.getUTCMonth() + 1)}.${dt.getUTCFullYear()}`,
      weekday: WEEKDAYS[(dt.getUTCDay() + 6) % 7],
      // «Пт · день 12 из 37 · сумерки» — строка шапки по макету.
      long: `${WEEKDAYS_SHORT[(dt.getUTCDay() + 6) % 7]} · день ${Math.floor(minute / 1440) + 1} из ${
        world.activeWorld ? Math.ceil(world.totalDays || 37) : 37
      } · ${dayPeriod(dt.getUTCHours())}`,
    };
  };

  // ── WS-клиент ──

  let socket = null;
  let reconnectTimer = null;

  const send = (message) => {
    if (socket && socket.readyState === 1) socket.send(JSON.stringify(message));
  };

  const status = (text, isError = false) => {
    const el = document.getElementById('status');
    el.textContent = text;
    el.classList.toggle('error', isError);
  };

  const seekMinute = (minute) => {
    if (!world.simStartUTC || world.filled.count === 0) return;
    const clamped = Math.max(0, Math.min(Math.round(minute), world.filled.count - 1));
    send({ cmd: 'seek', t: minuteToIso(clamped) });
  };

  const setSpeed = (value) => send({ cmd: 'speed', value });

  const setPlaying = (play) => send({ cmd: play ? 'play' : 'pause' });

  // ── Миры и тумблер каймы (change m6-worlds-ui, спеки web-*-ui) ──

  const resetForecastState = () => {
    world.forecastFetched.clear();
    world.forecastBuckets.clear();
    world.forecastBucket = null;
    world.forecastDeficit = {};
  };

  // Версия модели прогноза для окраски: Эпоха — выбранная в «Лаборатории»
  // версия с артефактами, ветка — её семейство (arch версии, design решение 4).
  const forecastVersion = () => {
    if (!lab.loaded) return null;
    if ((world.activeWorld?.name ?? EPOCH_WORLD) === EPOCH_WORLD) return lab.sel.fcModel;
    const model = lab.models.find((item) => item.model_version === lab.sel.fcModel);
    return model?.arch ?? null;
  };

  const forecastBlockStart = (minute) => Math.floor(minute / FORECAST_BLOCK_MIN) * FORECAST_BLOCK_MIN;

  const loadForecastBlock = async (block) => {
    const name = world.activeWorld?.name;
    const version = forecastVersion();
    if (!name || !version) return;
    const key = `${name}|${version}|${block}`;
    if (world.forecastFetched.has(key)) return;
    world.forecastFetched.add(key);
    const params = new URLSearchParams({
      from: minuteToIso(block),
      to: minuteToIso(block + FORECAST_BLOCK_MIN),
    });
    try {
      const response = await fetch(
        `/api/worlds/${encodeURIComponent(name)}/forecasts?${params}`
      );
      if (!response.ok) throw new Error(String(response.status));
      const rows = (await response.json()).rows || [];
      // Мир/модель могли смениться, пока ответ летел: кэш нового мира не трогаем.
      if (name !== world.activeWorld?.name || version !== forecastVersion()) return;
      for (const row of rows) {
        if (row.model_version !== version) continue;
        const minute = isoToMinute(row.ts);
        if (!world.forecastBuckets.has(minute)) world.forecastBuckets.set(minute, []);
        world.forecastBuckets.get(minute).push(row);
      }
      world.forecastBucket = null; // перекрасить текущий закрытый бакет из кэша
      updateForecastColoring();
    } catch {
      world.forecastFetched.delete(key); // повтор на следующем бакете
    }
  };

  // Окраска — по ближайшему ЗАКРЫТОМУ бакету (анти-лик m6-worlds): бакет
  // [b, b+5) закрыт к минуте clock, когда b + 5 ≤ clock. Нет строки (прогрев,
  // дыра) — кайма нейтральная (design решение 5).
  const updateForecastColoring = () => {
    if (world.paintMode !== 'forecast') return;
    const closed = Math.floor((world.clock - AGG_STEP) / AGG_STEP) * AGG_STEP;
    if (closed === world.forecastBucket) return;
    world.forecastBucket = closed;
    if (closed < 0 || !forecastVersion()) {
      world.forecastDeficit = {};
      world.forecastProb = {};
      return;
    }
    void loadForecastBlock(forecastBlockStart(closed));
    const rows = world.forecastBuckets.get(closed);
    const next = {};
    const prob = {};
    if (rows) {
      for (const row of rows) {
        next[row.zone_id] = row.deficit_prob >= DEFICIT_PROB_LIMIT;
        prob[row.zone_id] = row.deficit_prob;
      }
    }
    world.forecastDeficit = next;
    world.forecastProb = prob; // вероятности для плашек и ореолов карты
  };

  const setPaintMode = async (mode) => {
    world.paintMode = mode === 'forecast' ? 'forecast' : 'fact';
    // Сегмент «Факт / Прогноз» и класс body (легенда меняет шкалу).
    for (const button of els.paintToggle?.querySelectorAll('button') ?? []) {
      button.setAttribute('aria-pressed', String(button.dataset.mode === world.paintMode));
    }
    document.body.classList.toggle('mode-forecast', world.paintMode === 'forecast');
    resetForecastState();
    if (world.paintMode === 'forecast') {
      if (!lab.loaded) await ensureLabLoaded(); // модель — селектор «Лаборатории»
      updateForecastColoring();
    }
  };

  const setActiveWorld = (meta) => {
    if (!meta?.name) return;
    const changed = world.activeWorld?.name !== meta.name;
    world.activeWorld = { name: meta.name, label: meta.label || meta.name };
    if (els.worldSelect && [...els.worldSelect.options].some((o) => o.value === meta.name)) {
      els.worldSelect.value = meta.name;
    }
    if (els.worldLabel) {
      els.worldLabel.textContent = world.activeWorld.label;
      els.worldLabel.title = world.activeWorld.label;
    }
    syncWorldSeg();
    if (changed) {
      // Мир сменился: записи старого мира не применяем до снимка нового,
      // кэш прогнозов и метрики городских графиков сбрасываются.
      world.awaitingSnapshot = true;
      resetForecastState();
      if (window.CityCharts) window.CityCharts.reset();
      setTimeout(() => {
        world.awaitingSnapshot = false; // WS мог переподключиться — не глушим навсегда
      }, 5000);
    }
  };

  // Сегмент миров в шапке: «Эпоха · A · B · C · D», подсказка — метка политики.
  const WORLD_SHORT = { 'a-base': 'A', 'b-reactive': 'B', 'c-mlp': 'C', 'd-lstm': 'D' };
  // Подсказки к кнопкам миров: зачем каждый нужен (а не технические метки веток).
  const WORLD_HINTS = {
    epoch:
      'Эпоха — данные для нейросетей: 37 дней без политики. На днях 1–30 модели обучались, ' +
      'на днях 31–37 проверялись; режим «Прогноз» показывает именно эти прогнозы.',
    'a-base': 'A — контроль: тот же город без ценовой политики, база для сравнения миров',
    'b-reactive': 'B — реактивная политика: ×1.30 там, где дефицит уже наступил (дни 31–37)',
    'c-mlp': 'C — тариф ступенями ×1.15 / ×1.30 / ×1.50 по прогнозу MLP (дни 31–37)',
    'd-lstm': 'D — тариф ступенями ×1.15 / ×1.30 / ×1.50 по прогнозу LSTM (дни 31–37)',
  };
  const renderWorldSeg = () => {
    const seg = document.getElementById('world-seg');
    if (!seg) return;
    seg.innerHTML = '';
    for (const entry of world.availableWorlds) {
      const button = document.createElement('button');
      button.textContent = entry.name === EPOCH_WORLD ? 'Эпоха' : WORLD_SHORT[entry.name] ?? entry.name;
      const hint = WORLD_HINTS[entry.name] ?? entry.label;
      button.title = entry.exists ? hint : `${hint} (БД не сгенерирована)`;
      button.disabled = !entry.exists;
      button.dataset.world = entry.name;
      button.setAttribute('aria-pressed', String(entry.name === world.activeWorld?.name));
      button.onclick = () => switchWorld(entry.name);
      seg.appendChild(button);
    }
  };

  const syncWorldSeg = () => {
    for (const button of document.querySelectorAll('#world-seg button')) {
      button.setAttribute('aria-pressed', String(button.dataset.world === world.activeWorld?.name));
    }
  };

  const loadWorlds = async () => {
    try {
      const response = await fetch('/api/worlds');
      const data = await response.json();
      world.availableWorlds = data.worlds || [];
    } catch {
      return; // реестр недоступен: оставляем текущий мир, повтор не критичен
    }
    if (!els.worldSelect) return;
    els.worldSelect.innerHTML = '';
    for (const entry of world.availableWorlds) {
      const option = document.createElement('option');
      option.value = entry.name;
      // Метки политик веток видны в перечне (спека web-city-ui); у Эпохи
      // полная честная метка — в чипе панели, в списке — короткое имя.
      const title =
        entry.name === EPOCH_WORLD ? 'Эпоха' : `${entry.name} — ${entry.label}`;
      option.textContent = entry.exists
        ? title
        : `${title} (БД не сгенерирована)`;
      option.title = entry.label;
      option.disabled = !entry.exists;
      els.worldSelect.appendChild(option);
    }
    renderWorldSeg();
    const active = world.activeWorld?.name;
    if (active && [...els.worldSelect.options].some((o) => o.value === active)) {
      els.worldSelect.value = active;
    }
  };

  // Переключение мира — одна REST-операция (design решение 3): позиция на
  // старт мира, WS присылает state+snapshot нового мира.
  const switchWorld = async (name) => {
    if (!name || name === world.activeWorld?.name) return true;
    status(`переключение мира ${name}…`);
    world.awaitingSnapshot = true;
    try {
      const response = await fetch(`/api/worlds/${encodeURIComponent(name)}/switch`, {
        method: 'POST',
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${response.status}`);
      }
      const state = await response.json();
      setActiveWorld(state.world);
      world.filled = state.filled;
      if (window.CityCharts) window.CityCharts.syncControls(world);
      status('подключено');
      return true;
    } catch (error) {
      world.awaitingSnapshot = false;
      if (els.worldSelect && world.activeWorld) els.worldSelect.value = world.activeWorld.name;
      status(`мир не переключён: ${error.message}`, true);
      return false;
    }
  };

  // Seek-связка «Лаборатория → Город» (design решение 7): иной мир сначала
  // переключается на «Эпоху», затем карта встаёт на момент (одна операция).
  const seekWorldMinute = async (minute) => {
    if (!Number.isFinite(minute)) return;
    if (world.activeWorld && world.activeWorld.name !== EPOCH_WORLD) {
      const switched = await switchWorld(EPOCH_WORLD);
      if (!switched) return;
    }
    seekMinute(Math.round(minute));
  };

  // ── Применение записей: идемпотентно по from_index, разрыв — пересинк ──

  const applyRecord = (index, rec) => {
    world.freeByZone = rec.free_by_zone;
    world.priceMult = rec.price_mult;
    world.isRain = rec.is_rain;
    for (const spawn of rec.spawns) {
      world.pending.set(spawn.internal_id, { origin: spawn.origin, createdAt: index });
      world.winRequests[spawn.origin] += 1;
    }
    for (const event of rec.assignments) {
      world.pending.delete(event.internal_id);
      world.activeTrips.set(event.internal_id, {
        driverZone: event.driver_zone,
        origin: event.origin,
        dest: event.dest,
        assignedAt: index, // минута записи назначения — та же свёртка, что в снимке
        pickupAt: isoToMinute(event.pickup_at),
        dropoffAt: isoToMinute(event.dropoff_at),
      });
    }
    for (const event of rec.dropoffs) world.activeTrips.delete(event.internal_id);
    for (const event of rec.expiries) {
      if (world.pending.delete(event.internal_id)) {
        world.winExpiries[event.origin] += 1;
        world.flashes.push({ zone: event.origin, born: world.clock });
      }
    }
    // Отказ пассажира от долгой подачи (change world-dispatch-radius): маркер
    // ожидания гаснет, в зоне происхождения — та же вспышка, что у истечения.
    for (const event of rec.cancellations) {
      if (world.pending.delete(event.internal_id)) {
        world.winExpiries[event.origin] += 1;
        world.flashes.push({ zone: event.origin, born: world.clock });
      }
    }
    // Окно [index-4, index] закрылось: снапшот парка этой записи — его
    // последняя минута, тот же момент, что snapshot_drivers движка.
    if (index % AGG_STEP === AGG_STEP - 1) closeWindows();
  };

  const closeWindows = () => {
    for (let z = 0; z < world.winRequests.length; z += 1) {
      const requests = world.winRequests[z];
      const free = world.freeByZone[z] ?? 0;
      world.deficit[z] = free / Math.max(requests, 1) < DEFICIT_RATIO_LIMIT || world.winExpiries[z] > 0;
      world.winRequests[z] = 0;
      world.winExpiries[z] = 0;
    }
  };

  const applyRecords = (message) => {
    const from = message.from_index;
    const records = message.records;
    if (from > world.appliedUpTo + 1) {
      // Пропуск пачки (после лага сети): мир недодополнен — просим пересынк.
      status('разрыв потока записей — пересинхронизация…');
      seekMinute(world.clock);
      return;
    }
    const skip = world.appliedUpTo - from + 1; // дубли: пачка в пути пересекла seek
    if (skip >= records.length) return;
    for (let i = Math.max(skip, 0); i < records.length; i += 1) {
      applyRecord(from + i, records[i]);
    }
    world.appliedUpTo = from + records.length - 1;
    // Часы, отставшие от потока (троттлящийся rAF скрытой вкладки), догоняют
    // записи: поток — факт сервера, рисовать мир позади применённого нельзя.
    if (world.playing && world.clock < world.appliedUpTo - CLOCK_SYNC_LIMIT) {
      world.clock = world.appliedUpTo;
    }
  };

  const rebuildFromSnapshot = (message) => {
    world.activeTrips = new Map();
    for (const trip of message.active_trips) {
      world.activeTrips.set(trip.internal_id, {
        driverZone: trip.driver_zone,
        origin: trip.origin,
        dest: trip.dest,
        assignedAt: isoToMinute(trip.assigned_at),
        pickupAt: isoToMinute(trip.pickup_at),
        dropoffAt: isoToMinute(trip.dropoff_at),
      });
    }
    world.pending = new Map();
    for (const order of message.pending_orders) {
      world.pending.set(order.internal_id, {
        origin: order.origin,
        createdAt: isoToMinute(order.created_at),
      });
    }
    world.isRain = message.env.is_rain;
    world.priceMult = message.env.price_mult;
    world.freeByZone = message.env.free_by_zone;
    world.appliedUpTo = message.index;
    world.winRequests = new Array(world.zones.length).fill(0);
    world.winExpiries = new Array(world.zones.length).fill(0);
    // Кайма последнего закрытого окна приходит в снимке — после seek карта
    // сразу показывает дефицит, а не ждёт закрытия следующего окна.
    world.deficit = world.zones.map((zone) => Boolean(message.env.deficit?.[zone.id]));
    world.flashes = [];
    world.clock = message.index;
    world.position = message.index;
  };

  const syncClock = (message) => {
    // Пауза и большие расхождения — сервер авторитетен; при нормальном
    // воспроизведении лёгкий подтяг без видимого скачка назад (design.md).
    const pos = message.position;
    if (!message.playing || Math.abs(world.clock - pos) > CLOCK_SYNC_LIMIT) {
      world.clock = pos;
    } else {
      world.clock += (pos - world.clock) * 0.5;
    }
    world.position = pos;
  };

  const route = (message) => {
    switch (message.type) {
      case 'state':
        world.playing = message.playing;
        world.speed = message.speed;
        world.filled = message.filled;
        if (message.world) setActiveWorld(message.world);
        if (message.world?.days) world.totalDays = message.world.days;
        if (!world.simStartUTC && message.world?.sim_start) setSimStart(message.world.sim_start);
        syncClock(message);
        status('подключено');
        if (window.CityCharts) window.CityCharts.syncControls(world);
        break;
      case 'snapshot':
        world.awaitingSnapshot = false; // новый мир получен — поток записей можно применять
        rebuildFromSnapshot(message);
        status('подключено');
        break;
      case 'records':
        if (world.awaitingSnapshot) break; // записи прежнего мира (switch в полёте)
        if (world.appliedUpTo >= 0) applyRecords(message);
        break;
      case 'log-extended':
        world.filled = {
          count: message.filled,
          first: world.filled.first,
          last: message.last,
        };
        if (window.CityCharts) {
          window.CityCharts.onLogExtended();
          window.CityCharts.syncControls(world);
        }
        break;
      case 'error':
        status(`сервер: ${message.message}`, true);
        break;
    }
    // Панель — сразу по сообщениям, а не только в кадре: при троттлящемся
    // rAF (скрытая вкладка) seek/пауза всё равно видны мгновенно.
    renderPanel();
  };

  const connect = () => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${proto}://${location.host}/ws/replay`);
    socket.onmessage = (event) => route(JSON.parse(event.data));
    socket.onopen = () => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      status('подключено');
    };
    socket.onclose = () => {
      status('соединение потеряно — переподключение…');
      reconnectTimer = setTimeout(connect, 1000);
    };
    socket.onerror = () => status('ошибка соединения', true);
  };

  // ── Панель: DOM обновляется в кадре (значения кэшируются) ──

  const els = {};
  const cache = {};

  const setText = (key, el, value) => {
    if (cache[key] !== value) {
      cache[key] = value;
      el.textContent = value;
    }
  };

  const renderPanel = () => {
    if (!world.simStartUTC) return;
    const parts = clockParts(world.clock);
    setText('time', els.time, parts.time);
    setText('date', els.date, parts.long);
    // Иконка play/pause рисуется CSS по классу кнопки.
    els.play.classList.toggle('playing', world.playing);
    // Селектор скорости — по серверному значению (скорость общая для всех вкладок).
    const speedValue = String(world.speed);
    if (cache.speed !== speedValue && els.speed) {
      cache.speed = speedValue;
      for (const button of els.speed.querySelectorAll('button')) {
        button.setAttribute('aria-pressed', String(Number(button.dataset.speed) === world.speed));
      }
    }
    // Метка события окна — из метрик выбранной зоны (charts.js), вне кадра тоже.
    if (window.CityCharts) world.event = window.CityCharts.eventAt(world.clock);
    const eventLabel = { friday_evening: 'пятничный вечер', surge_injection: 'всплеск тарифа' }[world.event];
    els.event.classList.toggle('hidden', !eventLabel);
    if (eventLabel) setText('event', els.event, eventLabel);
  };

  // ── Вкладка «Лаборатория» (change m3-ml-tab, задачи 4.2, 5.1–5.2, 6.2–6.3) ──
  // Данные — только из /api/ml/* (спека экрана); загрузка ленивая — при первом
  // открытии вкладки, дальше — перезагрузка зависимых блоков по смене
  // селекторов. Городская шина (WS, карта, графики города) вкладку не читает:
  // скрытые канвасы продолжают рисоваться (design.md m3-ml-tab, решение 6).

  const BUCKET_MS = 5 * 60000; // ширина 5-минутного бакета для полосы дефицита
  const DEFICIT_BAND_PROB = 0.5; // полоса там, где deficit_prob больше порога

  const lab = {
    loaded: false,
    loading: false,
    models: [], // ответ /api/ml/models: наивы + сети с loss
    summary: null, // содержимое summary-*.json или null (404)
    forecasts: [], // текущий срез /api/ml/forecasts
    errors: null, // текущий ответ /api/ml/errors
    sel: {
      panelModel: null, // модель панели (loss + калибровка) — по умолчанию последняя сеть
      zone: null, // зона графика факт-vs-прогноз — первая из /api/city
      horizon: 15,
      fcModel: null, // модель графика факт-vs-прогноз — последняя сеть
      target: 'requests', // спрос | сток
      day: null, // день графика факт-vs-прогноз: 'all' или ключ даты; null — первый день
      calHorizon: 15, // горизонт калибровки (свой селектор панели)
    },
  };

  const labEls = {};

  const cacheLabEls = () => {
    for (const id of [
      'lab-empty',
      'lab-grid',
      'ml-model-select',
      'model-meta',
      'loss-chart',
      'ml-cal-horizon-select',
      'calibration-chart',
      'summary-table',
      'ml-zone-select',
      'ml-horizon-select',
      'ml-fc-model-select',
      'ml-target-select',
      'forecast-chart',
      'residual-chart',
      'histogram-chart',
      'by-hour-chart',
    ]) {
      labEls[id] = document.getElementById(id);
    }
  };

  // Наивные времена эпохи → UTC-ms: арифметика без локального сдвига,
  // подписи осей хелпера возвращают настенные времена БД.
  const isoToMs = (iso) => {
    const [date, time] = iso.split(' ');
    const [y, m, d] = date.split('-').map(Number);
    const [h, min] = time.split(':').map(Number);
    return Date.UTC(y, m - 1, d, h, min);
  };

  // Точка лаб-графика под курсором → сим-минута мира «Города». Обе серии
  // (прогноз и факт) используют ts строки прогноза, геометрия — как в
  // LabCharts.drawTimeSeries (те же поля канваса и оси x).
  const labMinuteAtClick = (event) => {
    const canvas = labEls['forecast-chart'];
    const shown = visibleForecasts();
    if (!canvas || !shown.length) return null;
    const rect = canvas.getBoundingClientRect();
    const plotW = rect.width - LAB_PAD.left - LAB_PAD.right;
    const x = event.clientX - rect.left;
    if (plotW <= 0 || x < LAB_PAD.left || x > LAB_PAD.left + plotW) return null;
    const xs = shown.map((row) => isoToMs(row.ts));
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const ms = xMax === xMin ? xMin : xMin + ((x - LAB_PAD.left) / plotW) * (xMax - xMin);
    let nearest = shown[0];
    for (const row of shown) {
      if (Math.abs(isoToMs(row.ts) - ms) < Math.abs(isoToMs(nearest.ts) - ms)) nearest = row;
    }
    return isoToMinute(nearest.ts);
  };

  const onLabChartClick = async (event) => {
    const minute = labMinuteAtClick(event);
    if (minute == null) return;
    const target = Math.round(minute);
    if (world.activeWorld && world.activeWorld.name !== EPOCH_WORLD) {
      const switched = await switchWorld(EPOCH_WORLD); // одна операция: switch → seek
      if (!switched) return;
    }
    seekMinute(target);
  };

  const showLabEmpty = (empty) => {
    labEls['lab-empty'].classList.toggle('hidden', !empty);
    labEls['lab-grid'].classList.toggle('hidden', empty);
  };

  const fillModelSelect = (select) => {
    select.innerHTML = '';
    const groups = [
      ['Наивы', lab.models.filter((m) => m.kind === 'naive')],
      ['Нейросети', lab.models.filter((m) => m.kind === 'nn')],
    ];
    for (const [label, models] of groups) {
      if (!models.length) continue;
      const optgroup = document.createElement('optgroup');
      optgroup.label = label;
      for (const model of models) {
        const option = document.createElement('option');
        option.value = model.model_version;
        option.textContent = model.model_version;
        optgroup.appendChild(option);
      }
      select.appendChild(optgroup);
    }
  };

  const buildLabSelectors = () => {
    // Зоны графика — имена из /api/city (world.zones уже загружен городом).
    const zoneSelect = labEls['ml-zone-select'];
    zoneSelect.innerHTML = '';
    for (const zone of world.zones) {
      const option = document.createElement('option');
      option.value = String(zone.id);
      option.textContent = `${zone.id} · ${zone.name}`;
      zoneSelect.appendChild(option);
    }
    lab.sel.zone = world.zones.length ? world.zones[0].id : null;
    if (lab.sel.zone != null) zoneSelect.value = String(lab.sel.zone);

    fillModelSelect(labEls['ml-model-select']);
    fillModelSelect(labEls['ml-fc-model-select']);
    const networks = lab.models.filter((m) => m.kind === 'nn');
    const fallback = lab.models.length ? lab.models[0].model_version : null;
    const lastNetwork = networks.length ? networks[networks.length - 1].model_version : fallback;
    lab.sel.panelModel = lastNetwork;
    lab.sel.fcModel = lastNetwork;
    if (lastNetwork != null) {
      labEls['ml-model-select'].value = lastNetwork;
      labEls['ml-fc-model-select'].value = lastNetwork;
    }

    labEls['ml-zone-select'].onchange = () => {
      lab.sel.zone = Number(labEls['ml-zone-select'].value);
      loadForecastData();
    };
    labEls['ml-horizon-select'].onchange = () => {
      lab.sel.horizon = Number(labEls['ml-horizon-select'].value);
      loadForecastData();
    };
    labEls['ml-fc-model-select'].onchange = () => {
      lab.sel.fcModel = labEls['ml-fc-model-select'].value;
      loadForecastData();
      if (world.paintMode === 'forecast') {
        resetForecastState(); // модель окраски — селектор «Лаборатории»
        updateForecastColoring();
      }
    };
    labEls['ml-target-select'].onchange = () => {
      lab.sel.target = labEls['ml-target-select'].value;
      drawForecastChart();
    };
    labEls['ml-model-select'].onchange = () => {
      lab.sel.panelModel = labEls['ml-model-select'].value;
      renderModelPanel();
      renderCalibration();
    };
    labEls['ml-cal-horizon-select'].onchange = () => {
      lab.sel.calHorizon = Number(labEls['ml-cal-horizon-select'].value);
      renderCalibration();
    };
  };

  const labChip = (text) => {
    const el = document.createElement('span');
    el.className = 'chip';
    el.textContent = text;
    return el;
  };

  const renderModelPanel = () => {
    const model = lab.models.find((m) => m.model_version === lab.sel.panelModel);
    const meta = labEls['model-meta'];
    meta.innerHTML = '';
    if (!model) return;
    meta.appendChild(labChip(model.kind === 'nn' ? 'нейросеть' : 'наив'));
    if (model.kind !== 'nn') {
      meta.appendChild(labChip('без артефактов — кривой loss нет'));
      window.LabCharts.drawTimeSeries(labEls['loss-chart'], { series: [] });
      return;
    }
    meta.appendChild(labChip(`arch: ${model.arch}`));
    meta.appendChild(labChip(`окно: ${model.window_hours} ч`));
    meta.appendChild(labChip(`лучшая эпоха: ${model.best_epoch}`));
    window.LabCharts.drawTimeSeries(labEls['loss-chart'], {
      series: [
        {
          name: 'train',
          color: '#5aa9e6',
          points: model.loss.map((e) => ({ x: e.epoch, y: e.train.total })),
        },
        {
          name: 'val',
          color: '#a48fd6',
          points: model.loss.map((e) => ({ x: e.epoch, y: e.val.total })),
        },
      ],
      xIsTime: false,
    });
  };

  const num3 = (value) => (typeof value === 'number' ? value.toFixed(3) : '—');

  const regCell = (summary, target, horizon, metric) =>
    num3(summary.regression?.[target]?.[String(horizon)]?.[metric]);
  const defCell = (summary, horizon, metric) => num3(summary.deficit?.[String(horizon)]?.[metric]);

  const renderSummaryTable = () => {
    const wrap = labEls['summary-table'];
    wrap.innerHTML = '';
    if (!lab.summary || !lab.summary.summaries?.length) {
      const note = document.createElement('p');
      note.className = 'lab-note';
      note.textContent =
        'Сводки нет: artifacts/summary-*.json не найден — прогоните python -m ml infer.';
      wrap.appendChild(note);
      return;
    }
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    const nameTh = document.createElement('th');
    nameTh.rowSpan = 2;
    nameTh.textContent = 'модель';
    headRow.appendChild(nameTh);
    const groups = [
      ['MAE спрос', 2],
      ['MAE сток', 2],
      ['RMSE спрос', 2],
      ['RMSE сток', 2],
      ['дефицит 15', 3],
      ['дефицит 30', 3],
    ];
    for (const [title, span] of groups) {
      const th = document.createElement('th');
      th.colSpan = span;
      th.textContent = title;
      headRow.appendChild(th);
    }
    const subRow = document.createElement('tr');
    for (const label of ['15м', '30м', '15м', '30м', '15м', '30м', '15м', '30м', 'P', 'R', 'F1', 'P', 'R', 'F1']) {
      const th = document.createElement('th');
      th.textContent = label;
      subRow.appendChild(th);
    }
    thead.append(headRow, subRow);
    const tbody = document.createElement('tbody');
    for (const summary of lab.summary.summaries) {
      const tr = document.createElement('tr');
      const name = document.createElement('td');
      name.textContent = summary.model_version;
      tr.appendChild(name);
      for (const metric of ['mae', 'rmse']) {
        for (const target of ['pred_requests', 'pred_free_drivers']) {
          for (const horizon of [15, 30]) {
            const td = document.createElement('td');
            td.textContent = regCell(summary, target, horizon, metric);
            tr.appendChild(td);
          }
        }
      }
      for (const horizon of [15, 30]) {
        for (const metric of ['precision', 'recall', 'f1']) {
          const td = document.createElement('td');
          td.textContent = defCell(summary, horizon, metric);
          tr.appendChild(td);
        }
      }
      tbody.appendChild(tr);
    }
    // Лучшее значение колонки — зелёным: MAE/RMSE (первые 8) меньше лучше,
    // precision/recall/F1 — больше лучше.
    const rows = [...tbody.rows];
    for (let col = 1; rows.length && col < rows[0].cells.length; col += 1) {
      const cells = rows.map((row) => row.cells[col]);
      const values = cells.map((cell) => Number(cell.textContent)).filter(Number.isFinite);
      if (!values.length) continue;
      const best = col <= 8 ? Math.min(...values) : Math.max(...values);
      for (const cell of cells) {
        if (Number(cell.textContent) === best) cell.classList.add('best');
      }
    }
    table.append(thead, tbody);
    wrap.appendChild(table);
  };

  // Кольца F1 детекции дефицита по моделям (горизонт 15 мин) — верх «Лаборатории».
  const MODEL_TITLES = [
    [/^naive-persist/, 'Как сейчас'],
    [/^naive-season/, 'Сутки назад'],
    [/^mlp/, 'MLP'],
    [/^lstm/, 'LSTM'],
  ];
  const modelTitle = (version) => MODEL_TITLES.find(([re]) => re.test(version))?.[1] ?? version;

  const renderLabKpis = () => {
    const host = document.getElementById('lab-kpis');
    if (!host) return;
    const summaries = lab.summary?.summaries ?? [];
    if (!summaries.length) {
      host.innerHTML = '<p class="lab-note">Сводки нет — прогоните python -m ml infer.</p>';
      return;
    }
    const naive = summaries.find((s) => /^naive-persist/.test(s.model_version));
    const naiveMae = naive?.regression?.pred_requests?.['15']?.mae;
    const bestF1 = Math.max(...summaries.map((s) => s.deficit?.['15']?.f1 ?? 0));
    host.innerHTML = '';
    for (const summary of summaries) {
      const d = summary.deficit?.['15'] ?? {};
      const f1 = d.f1 ?? 0;
      const nn = !/^naive/.test(summary.model_version);
      const color = f1 === bestF1 ? '#f5c518' : nn ? '#c9a227' : '#4a505a';
      const mae = summary.regression?.pred_requests?.['15']?.mae;
      const gain = nn && naiveMae && mae ? Math.round((1 - mae / naiveMae) * 100) : null;
      const item = document.createElement('div');
      item.className = 'ring-item';
      item.innerHTML =
        window.LabCharts.donut({
          segments: [
            { value: f1, color },
            { value: 1 - f1, color: 'transparent' },
          ],
          center: f1.toFixed(2),
          sub: 'F1',
        }) +
        `<div class="ring-title">${modelTitle(summary.model_version)}</div>` +
        `<div class="ring-meta mono">P ${(d.precision ?? 0).toFixed(2)} · R ${(d.recall ?? 0).toFixed(2)}</div>` +
        `<div class="ring-meta">${gain != null ? `ошибка спроса <b class="good">−${gain} %</b> к наиву` : 'наивный прогноз'}</div>`;
      host.appendChild(item);
    }
  };

  const renderCalibration = () => {
    const model = lab.models.find((m) => m.model_version === lab.sel.panelModel);
    const bins = model
      ? lab.summary?.summaries?.find((s) => s.model_version === model.model_version)
          ?.calibration?.[String(lab.sel.calHorizon)]
      : null;
    if (!bins?.length) {
      window.LabCharts.drawTimeSeries(labEls['calibration-chart'], { series: [] });
      return;
    }
    const points = bins
      .map((bin) => ({ x: bin.mean_prob, y: bin.deficit_share }))
      .sort((a, b) => a.x - b.x);
    window.LabCharts.drawTimeSeries(labEls['calibration-chart'], {
      series: [
        {
          name: 'идеал',
          color: 'rgba(138, 144, 156, 0.7)',
          dash: [4, 4],
          points: [
            { x: 0, y: 0 },
            { x: 1, y: 1 },
          ],
        },
        { name: model.model_version, color: '#5aa9e6', points },
      ],
      xIsTime: false,
    });
  };

  // Строки графика «Факт vs прогноз» с учётом выбранного дня: неделя целиком
  // (~2 тыс. точек) сливается в полосу, один день читается линиями.
  const dayKey = (ts) => ts.slice(0, 10);
  const visibleForecasts = () =>
    lab.sel.day && lab.sel.day !== 'all' ? lab.forecasts.filter((row) => dayKey(row.ts) === lab.sel.day) : lab.forecasts;

  const fillDaySelect = () => {
    const select = document.getElementById('ml-day-select');
    if (!select) return;
    const days = [...new Set(lab.forecasts.map((row) => dayKey(row.ts)))];
    const short = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'];
    select.innerHTML = '<option value="all">вся неделя</option>';
    for (const day of days) {
      const [y, m, d] = day.split('-').map(Number);
      const minute = (Date.UTC(y, m - 1, d) - world.simStartUTC) / 60000;
      const option = document.createElement('option');
      option.value = day;
      option.textContent = `день ${Math.floor(minute / 1440) + 1} · ${short[(new Date(Date.UTC(y, m - 1, d)).getUTCDay() + 6) % 7]}`;
      select.appendChild(option);
    }
    if (lab.sel.day == null || (lab.sel.day !== 'all' && !days.includes(lab.sel.day))) {
      lab.sel.day = days[0] ?? 'all';
    }
    select.value = lab.sel.day;
    select.onchange = () => {
      lab.sel.day = select.value;
      drawForecastChart();
    };
  };

  const drawForecastChart = () => {
    const rowsShown = visibleForecasts();
    const target = lab.sel.target;
    const predKey = target === 'requests' ? 'pred_requests' : 'pred_free_drivers';
    const actKey = target === 'requests' ? 'actual_requests' : 'actual_free_drivers';
    // Полоса дефицита — по deficit_prob независимо от таргета; смежные бакеты
    // склеиваются в интервалы шириной окна (5 мин).
    const bands = [];
    let run = null;
    for (const row of rowsShown) {
      if (row.deficit_prob > DEFICIT_BAND_PROB) {
        const ms = isoToMs(row.ts);
        if (run) run.x1 = ms + BUCKET_MS;
        else {
          run = { x0: ms, x1: ms + BUCKET_MS };
          bands.push(run);
        }
      } else {
        run = null;
      }
    }
    const withFact = rowsShown.filter((row) => row[actKey] != null);
    // Факт — серая площадь на заднем плане, прогноз — яркая линия поверх.
    window.LabCharts.drawTimeSeries(labEls['forecast-chart'], {
      series: [
        {
          name: 'факт',
          color: 'rgba(201, 204, 209, 0.55)',
          area: 'rgba(201, 204, 209, 0.16)',
          width: 1,
          points: withFact.map((row) => ({ x: isoToMs(row.ts), y: row[actKey] })),
        },
        {
          name: 'прогноз',
          color: '#f5c518',
          width: 2,
          points: rowsShown.map((row) => ({ x: isoToMs(row.ts), y: row[predKey] })),
        },
      ],
      bands,
      legend: false,
    });
  };

  const drawErrors = () => {
    const errors = lab.errors;
    if (!errors) return;
    window.LabCharts.drawTimeSeries(labEls['residual-chart'], {
      series: [
        {
          name: 'остаток',
          color: '#a48fd6',
          points: (errors.residuals || []).map((r) => ({ x: isoToMs(r.ts), y: r.residual })),
        },
      ],
      includeZero: true,
    });
    const hist = errors.histogram;
    const histValues = hist?.counts || [];
    if (histValues.length) {
      const labels = hist.bin_edges.slice(0, -1).map(
        (lo, i) => ((lo + hist.bin_edges[i + 1]) / 2).toFixed(1)
      );
      window.LabCharts.drawValueBars(labEls['histogram-chart'], { labels, values: histValues });
    } else {
      window.LabCharts.drawValueBars(labEls['histogram-chart'], { labels: [], values: [] });
    }
    const byHour = new Array(24).fill(null);
    for (const cell of errors.by_hour || []) byHour[cell.hour] = cell.mae;
    window.LabCharts.drawValueBars(labEls['by-hour-chart'], {
      labels: byHour.map((_, hour) => String(hour).padStart(2, '0')),
      values: byHour,
      color: '#3fb57a',
    });
  };

  // Перерисовка лаборатории из кэшированных данных: селекторы, сменившиеся
  // прямо перед уходом с вкладки, завершают fetch уже на скрытых канвасах —
  // рисование в нулевой layout пропущено (guard в charts.js), поэтому при
  // возврате на «Лабораторию» блоки строятся заново.
  let labNeedsRedraw = false;

  const redrawLabCharts = () => {
    renderModelPanel();
    renderCalibration();
    drawForecastChart();
    drawErrors();
  };

  const loadForecastData = async () => {
    if (lab.sel.zone == null || lab.sel.fcModel == null) return;
    const params = new URLSearchParams({
      zone: String(lab.sel.zone),
      model: lab.sel.fcModel,
      horizon: String(lab.sel.horizon),
    });
    try {
      const [forecastRes, errorsRes] = await Promise.all([
        fetch(`/api/ml/forecasts?${params}`),
        fetch(`/api/ml/errors?${params}`),
      ]);
      lab.forecasts = (await forecastRes.json()).forecasts || [];
      fillDaySelect();
      lab.errors = await errorsRes.json();
      drawForecastChart();
      drawErrors();
    } catch {
      // сеть моргнула: данные статичны — перезагрузка случится по смене селектора
    }
  };

  const ensureLabLoaded = async () => {
    if (lab.loaded || lab.loading) return;
    lab.loading = true;
    try {
      const [modelsRes, summaryRes] = await Promise.all([
        fetch('/api/ml/models'),
        fetch('/api/ml/summary'),
      ]);
      lab.models = (await modelsRes.json()).models || [];
      lab.summary = summaryRes.ok ? await summaryRes.json() : null;
      lab.loaded = true;
      if (!lab.models.length) {
        showLabEmpty(true);
        return;
      }
      showLabEmpty(false);
      buildLabSelectors();
      renderModelPanel();
      renderSummaryTable();
      renderLabKpis();
      renderCalibration();
      await loadForecastData();
      if (labNeedsRedraw) {
        labNeedsRedraw = false;
        redrawLabCharts();
      }
    } catch {
      // сеть недоступна: lab.loaded остался false — следующее открытие повторит загрузку
    } finally {
      lab.loading = false;
    }
  };

  // ── Вкладка «Сравнение» (change m6-compare-tab, задачи 3.1–3.2) ──
  // Данные — только из /api/ml/compare; загрузка ленивая при первом открытии.
  // Таблицы и график по прецеденту m3-ml-tab: рисование в скрытый канвас
  // пропускает charts.js (нулевой layout), поэтому каждый показ вкладки
  // перерисовывает график из уже загруженных данных.

  const BRANCH_COLORS = {
    'a-base': '#7d838c', // A — серый: база сравнения
    'b-reactive': '#d99a3a', // B — янтарный
    'c-mlp': '#6f9bd1', // C — голубой
    'd-lstm': '#a48fd6', // D — фиолетовый
  };

  const OUTCOME_COLUMNS = [
    { key: 'trips', title: 'поездки', digits: 0, better: 'more' },
    { key: 'unfulfilled', title: 'невыполненные', digits: 0, better: 'less' },
    { key: 'avg_wait_min', title: 'ожидание, мин', digits: 2, better: 'less' },
    { key: 'revenue', title: 'выручка, млн ₽', digits: 3, better: 'more', scale: 1e-6 },
    {
      key: 'deficit_share',
      title: 'доля дефицита',
      digits: 1,
      better: 'less',
      scale: 100,
      suffix: ' %',
    },
  ];

  const COMPARE_HOUR_MS = 60 * 60 * 1000;

  const compare = {
    loaded: false,
    loading: false,
    data: null,
  };

  const compareEls = {};

  const cacheCompareEls = () => {
    for (const id of [
      'compare',
      'compare-empty',
      'compare-grid',
      'compare-window',
      'compare-table',
      'compare-legend',
      'revenue-chart',
      'compare-revenue-note',
      'compare-accuracy',
    ]) {
      compareEls[id] = document.getElementById(id);
    }
  };

  const branchColor = (name) => BRANCH_COLORS[name] || '#8a909c';

  const formatCompareValue = (column, value) => {
    if (value == null || !Number.isFinite(value)) return '—';
    const scaled = value * (column.scale ?? 1);
    if (column.digits === 0) return Math.round(scaled).toLocaleString('ru-RU');
    return `${scaled.toFixed(column.digits)}${column.suffix ?? ''}`;
  };

  const formatDelta = (delta) => {
    if (delta == null || !Number.isFinite(delta)) return '';
    return `${delta >= 0 ? '+' : ''}${delta.toFixed(1)} %`;
  };

  const bestOfColumn = (branches, column) => {
    const values = branches
      .filter((branch) => branch.exists && branch.outcomes?.[column.key] != null)
      .map((branch) => branch.outcomes[column.key]);
    if (!values.length) return null;
    return column.better === 'more' ? Math.max(...values) : Math.min(...values);
  };

  const renderCompareOutcomes = (data) => {
    const wrap = compareEls['compare-table'];
    wrap.innerHTML = '';
    const branches = data.branches || [];
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    for (const title of ['ветка', 'политика', ...OUTCOME_COLUMNS.map((c) => c.title)]) {
      const th = document.createElement('th');
      th.textContent = title;
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    const tbody = document.createElement('tbody');
    const bestByColumn = new Map(
      OUTCOME_COLUMNS.map((column) => [column.key, bestOfColumn(branches, column)])
    );

    for (const branch of branches) {
      const tr = document.createElement('tr');
      const nameCell = document.createElement('td');
      nameCell.textContent = branch.name;
      const labelCell = document.createElement('td');
      labelCell.textContent = branch.label;
      tr.append(nameCell, labelCell);
      if (!branch.exists) {
        const missing = document.createElement('td');
        missing.className = 'missing';
        missing.colSpan = OUTCOME_COLUMNS.length;
        missing.textContent = 'не сгенерирована';
        tr.appendChild(missing);
      } else {
        for (const column of OUTCOME_COLUMNS) {
          const td = document.createElement('td');
          const value = branch.outcomes?.[column.key];
          td.textContent = formatCompareValue(column, value);
          if (value != null && value === bestByColumn.get(column.key)) td.classList.add('best');
          const delta = branch.delta?.[column.key];
          const deltaText = formatDelta(delta);
          if (deltaText) {
            const span = document.createElement('span');
            span.className = 'delta';
            span.textContent = `к A: ${deltaText}`;
            td.appendChild(span);
          }
          tr.appendChild(td);
        }
      }
      tbody.appendChild(tr);
    }
    table.append(thead, tbody);
    wrap.appendChild(table);
    compareEls['compare-window'].textContent =
      `окно: ${data.window?.start ?? '—'} — ${data.window?.end ?? '—'}`;
  };

  const renderCompareLegend = (data) => {
    const legend = compareEls['compare-legend'];
    legend.innerHTML = '';
    for (const branch of (data.branches || []).filter((item) => item.exists)) {
      const item = document.createElement('span');
      const swatch = document.createElement('i');
      swatch.className = 'swatch';
      swatch.style.background = branchColor(branch.name);
      item.append(swatch, document.createTextNode(branch.name));
      legend.appendChild(item);
    }
  };

  const drawCompareChart = (data) => {
    if (!data) return;
    const startMs = isoToMs(data.window.start);
    const series = [];
    for (const branch of data.branches || []) {
      if (!branch.exists || !branch.revenue_series) continue;
      series.push({
        name: branch.name,
        color: branchColor(branch.name),
        points: branch.revenue_series.map((value, index) => ({
          // Точка i — конец часа i окна; текст оси — настенные времена БД.
          x: startMs + (index + 1) * COMPARE_HOUR_MS,
          y: value / 1e6,
        })),
      });
    }
    // Ось Y от нуля не начинается: линии к концу недели различаются на ~0.5 %
    // (design решение 9), при нуле их не различить.
    window.LabCharts.drawTimeSeries(compareEls['revenue-chart'], {
      series,
      includeZero: false,
      legend: false,
    });
  };

  const renderCompareNote = (data) => {
    const note = compareEls['compare-revenue-note'];
    const base = (data.branches || []).find((branch) => branch.name === 'a-base');
    if (!base?.exists || base.outcomes?.revenue == null) {
      note.textContent = 'Итоговая разница к A недоступна: ветка A не сгенерирована.';
      return;
    }
    const parts = (data.branches || [])
      .filter((branch) => branch.exists && branch.name !== base.name)
      .map((branch) => {
        const delta = branch.delta?.revenue;
        return delta == null
          ? `${branch.name}: —`
          : `${branch.name}: ${delta >= 0 ? '+' : ''}${delta.toFixed(2)} %`;
      });
    note.textContent = `Итог тест-недели к A (выручка): ${parts.length ? parts.join(', ') : '—'}.`;
  };

  const renderCompareAccuracy = (accuracy) => {
    const wrap = compareEls['compare-accuracy'];
    wrap.innerHTML = '';
    const rows = accuracy?.rows || [];
    if (!rows.length) {
      const note = document.createElement('p');
      note.className = 'lab-note';
      note.textContent =
        'Точности нет: у ветки A нет прогнозов тень-моделей в тест-неделе.';
      wrap.appendChild(note);
      return;
    }
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    for (const title of ['модель', 'горизонт', 'MAE спроса', 'F1 дефицита']) {
      const th = document.createElement('th');
      th.textContent = title;
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    const tbody = document.createElement('tbody');
    for (const row of rows) {
      const tr = document.createElement('tr');
      for (const value of [
        row.family,
        `${row.horizon_min} мин`,
        row.mae_requests.toFixed(3),
        row.f1.toFixed(3),
      ]) {
        const td = document.createElement('td');
        td.textContent = value;
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.append(thead, tbody);
    wrap.appendChild(table);
  };

  const showCompareEmpty = (empty) => {
    compareEls['compare-empty'].classList.toggle('hidden', !empty);
    compareEls['compare-grid'].classList.toggle('hidden', empty);
  };

  const redrawCompare = () => {
    if (!compare.data) return;
    drawCompareChart(compare.data);
    renderCompareNote(compare.data);
  };

  // Карточки миров: кольцо распределения тарифа (как часто и какой ступенью
  // политика поднимала цену) и ключевые исходы недели.
  const TARIFF_COLORS = [
    [1.0, '#2a2f37', '×1.00'],
    [1.15, '#7a6312', '×1.15'],
    [1.3, '#c9a227', '×1.30'],
    [1.5, '#f5c518', '×1.50'],
  ];
  const BRANCH_LETTERS = { 'a-base': 'A', 'b-reactive': 'B', 'c-mlp': 'C', 'd-lstm': 'D' };

  const renderCompareCards = (data) => {
    const host = document.getElementById('compare-cards');
    if (!host) return;
    host.innerHTML = '';
    for (const branch of data.branches || []) {
      const card = document.createElement('article');
      card.className = 'world-card';
      card.style.setProperty('--wc', branchColor(branch.name));
      const mix = branch.tariff_mix || [];
      const total = mix.reduce((sum, item) => sum + item.count, 0);
      const raised = mix.filter((item) => Math.abs(item.mult - 1) > 1e-9).reduce((sum, item) => sum + item.count, 0);
      const segments = mix.map((item) => ({
        value: item.count,
        color: TARIFF_COLORS.find(([m]) => Math.abs(m - item.mult) < 1e-6)?.[1] ?? '#a3a9b2',
      }));
      const used = TARIFF_COLORS.filter(([m]) => m !== 1.0 && mix.some((item) => Math.abs(item.mult - m) < 1e-6));
      const o = branch.outcomes || {};
      const delta = branch.delta || {};
      const fmtDelta = (v) => (v == null ? '' : `<small class="${v <= 0 ? 'good' : 'bad'}">${v > 0 ? '+' : ''}${v.toFixed(1)} %</small>`);
      const fmtRev = (v) => (v == null ? '' : `<small class="${v >= 0 ? 'good' : 'bad'}">${v > 0 ? '+' : ''}${v.toFixed(2)} %</small>`);
      card.innerHTML = branch.exists
        ? `<div class="wc-head"><span class="wc-letter">${BRANCH_LETTERS[branch.name] ?? '?'}</span><div><b>${branch.label}</b><span class="mono">${branch.name}</span></div></div>` +
          `<div class="wc-ring">${window.LabCharts.donut({ segments, size: 128, thickness: 14, center: total ? `${((raised / total) * 100).toFixed(1)}%` : '—', sub: 'тариф ↑' })}` +
          `<ul class="wc-mix">${
            used.length
              ? used.map(([m, color, label]) => {
                  const count = mix.find((item) => Math.abs(item.mult - m) < 1e-6)?.count ?? 0;
                  return `<li><i style="background:${color}"></i>${label}<span class="mono">${count}</span></li>`;
                }).join('')
              : '<li class="muted">тариф не менялся</li>'
          }</ul></div>` +
          `<dl class="wc-kpi">` +
          `<div><dt>Доля дефицита</dt><dd class="mono">${o.deficit_share != null ? (o.deficit_share * 100).toFixed(1) + ' %' : '—'} ${fmtDelta(delta.deficit_share)}</dd></div>` +
          `<div><dt>Ожидание, мин</dt><dd class="mono">${o.avg_wait_min != null ? o.avg_wait_min.toFixed(2) : '—'} ${fmtDelta(delta.avg_wait_min)}</dd></div>` +
          `<div><dt>Поездки</dt><dd class="mono">${o.trips != null ? Math.round(o.trips).toLocaleString('ru-RU') : '—'}</dd></div>` +
          `<div><dt>Выручка к A</dt><dd class="mono">${delta.revenue != null ? fmtRev(delta.revenue) : 'база'}</dd></div>` +
          `</dl>`
        : `<div class="wc-head"><span class="wc-letter">${BRANCH_LETTERS[branch.name] ?? '?'}</span><div><b>${branch.label}</b><span class="mono">${branch.name}</span></div></div><p class="lab-note">Мир не сгенерирован.</p>`;
      host.appendChild(card);
    }
  };

  const renderCompare = () => {
    if (!compare.data) return;
    const branches = compare.data.branches || [];
    const hasAny = branches.some((branch) => branch.exists);
    showCompareEmpty(!hasAny);
    if (!hasAny) return;
    renderCompareCards(compare.data);
    renderCompareOutcomes(compare.data);
    renderCompareLegend(compare.data);
    renderCompareAccuracy(compare.data.accuracy);
    redrawCompare();
  };

  const ensureCompareLoaded = async () => {
    if (compare.loaded || compare.loading) return;
    compare.loading = true;
    try {
      const response = await fetch('/api/ml/compare');
      if (!response.ok) throw new Error(String(response.status));
      compare.data = await response.json();
      compare.loaded = true;
      if (document.body.dataset.tab === 'compare') renderCompare();
    } catch {
      // сеть моргнула: compare.loaded остался false — следующее открытие повторит
    } finally {
      compare.loading = false;
    }
  };

  const initTabs = () => {
    const buttons = document.querySelectorAll('#tab-bar .tab');
    for (const button of buttons) {
      button.addEventListener('click', () => {
        if (document.body.dataset.tab === button.dataset.tab) return;
        document.body.dataset.tab = button.dataset.tab;
        // Вкладка — в хэше адреса: ссылка /#lab открывает сразу «Лабораторию».
        try {
          history.replaceState(null, '', button.dataset.tab === 'city' ? location.pathname : `#${button.dataset.tab}`);
        } catch {
          // file:// или песочница без History API — без хэша
        }
        for (const other of buttons) other.classList.toggle('active', other === button);
        if (button.dataset.tab === 'lab') {
          if (lab.loaded) redrawLabCharts();
          else if (!lab.loading) ensureLabLoaded();
          else labNeedsRedraw = true; // загрузка в полёте — достроим по её завершении
        } else if (button.dataset.tab === 'compare') {
          // Каждый показ вкладки перерисовывает график: в скрытый канвас
          // рисование пропускается (guard charts.js), а загрузка в полёте
          // достроит вкладку по своему завершении.
          if (compare.loaded) renderCompare();
          else ensureCompareLoaded();
        }
      });
    }
  };

  // ── Цикл кадра: часы → панель → карта → графики ──

  const frame = (nowMs) => {
    if (world.playing && world.lastFrame != null) {
      const dt = Math.min((nowMs - world.lastFrame) / 1000, 5); // подвисание вкладки не перематывает часы
      world.clock += world.speed * dt;
      // Живой край: часы не опережают заполненный лог (сервер ждёт пополнения).
      if (world.filled.count) world.clock = Math.min(world.clock, world.filled.count - 1);
    }
    world.lastFrame = nowMs;
    updateForecastColoring();
    renderPanel();
    // Карта и график рисуются только на видимой вкладке «Город»: на других
    // вкладках кадр ограничивается часами и панелью (экономия CPU).
    if (document.body.dataset.tab === 'city') {
      if (window.CityMap && world.cityReady) window.CityMap.draw(world, nowMs / 1000);
      if (window.CityCharts) window.CityCharts.draw(world);
    }
    requestAnimationFrame(frame);
  };

  // ── Инициализация: город → карта/графики → WS → кадры ──

  const init = async () => {
    els.time = document.getElementById('clock-time');
    els.date = document.getElementById('clock-date');
    els.weather = document.getElementById('weather');
    els.event = document.getElementById('event-badge');
    els.cars = document.getElementById('cars-count');
    els.play = document.getElementById('play-pause');
    els.worldSelect = document.getElementById('world-select');
    els.worldLabel = document.getElementById('world-label');
    els.paintToggle = document.getElementById('paint-toggle');

    const response = await fetch('/api/city');
    const city = await response.json();
    world.zones = city.zones;
    world.roads = city.roads;
    world.routes = city.routes;
    world.cityReady = true;

    window.CityMap.init(document.getElementById('map'), city);
    window.CityCharts.init(document.getElementById('chart'), city, {
      seekMinute,
      isoToMinute,
      minuteToIso,
      clockMinute: () => world.clock,
      simStarted: () => world.simStartUTC != null,
    });

    document.getElementById('play-pause').onclick = () => setPlaying(!world.playing);
    els.speed = document.getElementById('speed-seg');
    for (const button of els.speed.querySelectorAll('button')) {
      button.onclick = () => setSpeed(Number(button.dataset.speed));
    }
    // Пробел — play/pause на вкладке «Город» (кроме фокуса в полях ввода).
    document.addEventListener('keydown', (event) => {
      if (event.code !== 'Space' || document.body.dataset.tab !== 'city') return;
      if (event.target.closest('input, select, textarea, button')) return;
      event.preventDefault();
      setPlaying(!world.playing);
    });

    // Реестр миров (change m6-worlds-ui): перечень из GET /api/worlds,
    // активный мир приходит в state.world, выбор — REST switch.
    await loadWorlds();
    els.worldSelect.onchange = (event) => switchWorld(event.target.value);
    els.paintToggle.onclick = (event) => {
      const mode = event.target.closest('button')?.dataset.mode;
      if (mode && mode !== world.paintMode) setPaintMode(mode);
    };

    // Вкладка «Лаборатория»: кэш DOM и обработчики таб-бара; данные грузятся
    // лениво при первом открытии (design.md m3-ml-tab, решение 6).
    cacheLabEls();
    cacheCompareEls();
    initTabs();
    // Клик по точке лаб-графика — seek-связка с «Городом» (m6-worlds-ui).
    labEls['forecast-chart'].addEventListener('click', onLabChartClick);

    connect();
    requestAnimationFrame(frame);

    // Стартовая вкладка из хэша адреса (#lab, #compare).
    const startTab = location.hash.slice(1);
    if (startTab === 'lab' || startTab === 'compare') {
      document.querySelector(`#tab-bar .tab[data-tab="${startTab}"]`)?.click();
    }
  };

  window.App = {
    world,
    seekMinute,
    setSpeed,
    setPlaying,
    status,
    switchWorld,
    seekWorldMinute,
    setPaintMode,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
