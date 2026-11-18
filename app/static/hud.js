// Интерфейс «Города» поверх карты (макет design/Main.html): плашки районов,
// сводка по городу, панель выбранного района, погода в шапке, таймлайн с
// полосами «дефицит» и «дождь», сворачиваемый график района. Модуль читает
// App.world, CityMap и CityCharts, зовёт App.seekMinute; с сервером общается
// только через компактную ленту /api/deficit-timeline.

(() => {
  'use strict';

  const AGG_STEP = 5; // шаг агрегации метрик, минут
  const SPARK_HOURS = 6; // окно мини-графика панели района
  const UI_REFRESH_MS = 250; // цифры обновляются ~4 Гц, не каждый кадр
  const STRIP_REFRESH_MS = 3000; // дозагрузка ленты, пока лог растёт
  const LABEL_W = 60; // колонка подписей таймлайна, px (= отступ ползунка в CSS)

  const ARCH = {
    residential: 'жилой район',
    business: 'деловой район',
    airport: 'аэропорт',
    mixed: 'смешанная зона',
  };

  const $ = (id) => document.getElementById(id);
  let world = null;
  const els = {};
  const cards = new Map(); // zone_id → {root, tag, free, k, bar}
  let layoutVersion = -1;

  // ── Время ──

  const isoToMinute = (iso) => {
    const [date, time] = iso.split(' ');
    const [y, m, d] = date.split('-').map(Number);
    const [h, min] = time.split(':').map(Number);
    return (Date.UTC(y, m - 1, d, h, min) - world.simStartUTC) / 60000;
  };

  const hhmm = (minute) => {
    const total = ((Math.floor(minute) % 1440) + 1440) % 1440;
    return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
  };

  // ── Состояние района: факт или прогноз ──

  const forecastMode = () => world.paintMode === 'forecast';
  const probOf = (id) => world.forecastProb?.[id];
  const level = (p) => (p >= 0.6 ? 'bad' : p >= 0.3 ? 'mid' : '');
  const isDeficit = (id) =>
    forecastMode() ? (probOf(id) ?? 0) >= 0.5 || world.forecastDeficit?.[id] === true : Boolean(world.deficit[id]);

  // ── Плашки районов ──

  const buildCards = () => {
    const host = els.districts;
    host.innerHTML = '';
    for (const zone of world.zones) {
      const root = document.createElement('button');
      root.className = 'district';
      root.setAttribute('aria-label', `Район ${zone.name}`);
      root.innerHTML =
        '<div class="d-top"><span class="d-name"></span><span class="tag"></span></div>' +
        '<div class="d-mid"><span class="d-free"></span><span class="d-free-l">свободно</span><span class="d-k"></span></div>' +
        '<div class="d-bar"><i></i></div>';
      root.querySelector('.d-name').textContent = zone.name;
      root.onclick = () => selectZone(zone.id);
      host.appendChild(root);
      cards.set(zone.id, {
        root,
        tag: root.querySelector('.tag'),
        free: root.querySelector('.d-free'),
        k: root.querySelector('.d-k'),
        bar: root.querySelector('.d-bar'),
        fill: root.querySelector('.d-bar i'),
      });
    }
  };

  const placeCards = () => {
    const view = window.CityMap?.layout();
    if (!view || view.version === layoutVersion) return;
    layoutVersion = view.version;
    for (const [id, card] of cards) {
      const c = view.centers.get(id);
      card.root.style.left = `${c.x}px`;
      card.root.style.top = `${c.y}px`;
    }
  };

  const renderCards = () => {
    const fc = forecastMode();
    for (const [id, card] of cards) {
      const p = probOf(id);
      const bad = isDeficit(id);
      card.root.classList.toggle('deficit', !fc && bad);
      card.root.classList.toggle('selected', id === world.selectedZone);
      if (fc) {
        card.tag.textContent = p == null ? 'p —' : `p ${p.toFixed(2)}`;
        card.tag.className = `tag ${p == null ? '' : level(p)}`;
        card.bar.style.display = '';
        card.fill.style.width = `${Math.round((p ?? 0) * 100)}%`;
        card.fill.style.background = p >= 0.6 ? 'var(--red)' : p >= 0.3 ? 'var(--amber)' : '#3a3f48';
      } else {
        card.tag.textContent = bad ? 'дефицит' : 'норма';
        card.tag.className = `tag ${bad ? 'bad' : ''}`;
        card.bar.style.display = 'none';
      }
      card.free.textContent = String(world.freeByZone[id] ?? 0);
      const mult = world.priceMult[id] ?? 1;
      card.k.textContent = `×${mult.toFixed(2)}`;
      card.k.classList.toggle('up', Math.abs(mult - 1) > 1e-9);
    }
  };

  // ── Сводка по городу ──

  // Подсказка о тестовой неделе (дни 31–37): только там есть прогнозы
  // моделей и только там миры B/C/D меняют тариф. Без неё пустой режим
  // «Прогноз» и тариф ×1.00 выглядят как поломка.
  const POLICY_WORLDS = { 'b-reactive': 'B', 'c-mlp': 'C', 'd-lstm': 'D' };
  const TEST_WEEK_MIN = 30 * 1440;
  const JUMP_MIN = TEST_WEEK_MIN + 18 * 60; // день 31, 18:00 — вечерний пик
  let noteText = '';
  let pendingJump = false; // переход ждёт, пока фоновый прогон мира дойдёт до дня 31
  const dismissed = new Set(); // скрытые подсказки: ключ «мир|вид»

  const renderForecastNote = () => {
    const name = world.activeWorld?.name;
    const beforeTest = world.clock < TEST_WEEK_MIN;
    let kind = '';
    let text = '';
    if (pendingJump) {
      if (world.filled.count > JUMP_MIN) {
        pendingJump = false;
        window.App.seekMinute(JUMP_MIN);
      } else {
        kind = 'wait';
        const pct = Math.min(99, Math.floor((world.filled.count / JUMP_MIN) * 100));
        text = `Мир ещё просчитывается (${pct} %) — перейду к дню 31, как только он будет готов.`;
      }
    }
    if (!kind && forecastMode() && !Object.keys(world.forecastProb ?? {}).length) {
      kind = 'forecast';
      text = 'Прогноз моделей есть только для тестовой недели — дни 31–37.';
    } else if (!kind && POLICY_WORLDS[name] && beforeTest) {
      kind = 'policy';
      text = `Политика мира ${POLICY_WORLDS[name]} меняет тариф только на тестовой неделе — дни 31–37.`;
    }
    if (kind && kind !== 'wait' && dismissed.has(`${name}|${kind}`)) text = '';
    if (text !== noteText) {
      noteText = text;
      els.fcNote.dataset.kind = kind;
      els.fcNoteText.textContent = text;
      els.fcNote.classList.toggle('hidden', !text);
      els.fcJump.classList.toggle('hidden', kind === 'wait' || !beforeTest);
    }
  };

  const jumpToTestWeek = () => {
    if (world.filled.count > JUMP_MIN) window.App.seekMinute(JUMP_MIN);
    else pendingJump = true;
    noteText = '';
    renderForecastNote();
  };

  const dismissNote = () => {
    const kind = els.fcNote.dataset.kind;
    if (kind === 'wait') pendingJump = false;
    else if (kind) dismissed.add(`${world.activeWorld?.name}|${kind}`);
    noteText = '';
    els.fcNote.classList.add('hidden');
  };

  const renderSummary = () => {
    let onTrip = 0;
    let toPickup = 0;
    for (const trip of world.activeTrips.values()) {
      if (world.clock >= trip.dropoffAt) continue;
      if (world.clock < trip.pickupAt) toPickup += 1;
      else onTrip += 1;
    }
    els.cars.textContent = String(onTrip);
    els.pickup.textContent = String(toPickup);
    let bad = 0;
    for (const zone of world.zones) if (isDeficit(zone.id)) bad += 1;
    els.deficit.textContent = String(bad);
    els.deficit.parentElement.classList.toggle('bad', bad > 0);
    // Заказов в час — сумма 12 последних закрытых окон ленты.
    const i = closedWindowIndex();
    if (i >= 11 && strip.requests.length > i) {
      let sum = 0;
      for (let j = i - 11; j <= i; j += 1) sum += strip.requests[j];
      els.ordersHour.textContent = sum.toLocaleString('ru-RU');
    } else {
      els.ordersHour.textContent = '—';
    }
  };

  // ── Погода в шапке: «Дождь с 17:55» по ленте дождя ──

  const renderWeather = () => {
    els.weather.classList.toggle('rain', world.isRain);
    if (!world.isRain) {
      els.weather.textContent = 'без осадков';
      return;
    }
    let i = closedWindowIndex() + 1;
    if (i >= strip.rain.length) i = strip.rain.length - 1;
    while (i > 0 && strip.rain[i - 1]) i -= 1;
    const start = strip.start ? isoToMinute(strip.start) + Math.max(i, 0) * AGG_STEP : null;
    els.weather.textContent = start != null && strip.rain.length ? `Дождь с ${hhmm(start)}` : 'Дождь';
  };

  // ── Панель района ──

  const selectZone = (id) => {
    world.selectedZone = id;
    els.card.classList.remove('hidden');
    const zone = world.zones.find((z) => z.id === id);
    els.name.textContent = zone.name;
    els.type.textContent = ARCH[zone.archetype] ?? zone.archetype;
    window.CityCharts?.selectZone(id);
    renderRoads();
    renderPanel();
    renderCards();
  };

  const closePanel = () => {
    els.card.classList.add('hidden');
  };

  const lastClosedRow = (rows) => {
    let last = null;
    for (const row of rows) {
      if (row._min === undefined) row._min = isoToMinute(row.ts);
      if (row._min + AGG_STEP > world.clock) break;
      last = row;
    }
    return last;
  };

  const renderPanel = () => {
    const id = world.selectedZone;
    if (id == null || els.card.classList.contains('hidden')) return;
    const fc = forecastMode();
    const p = probOf(id);
    const bad = isDeficit(id);
    const rows = window.CityCharts?.zoneId() === id ? window.CityCharts.zoneRows() : [];
    const last = lastClosedRow(rows);
    if (fc) {
      els.status.textContent = p == null ? 'Нет прогноза' : bad ? 'Ожидается дефицит' : p >= 0.3 ? 'Риск' : 'Норма';
      els.status.className = `tag ${p == null ? '' : bad ? 'bad' : p >= 0.3 ? 'mid' : 'ok'}`;
    } else {
      els.status.textContent = bad ? 'Дефицит' : 'Норма';
      els.status.className = `tag ${bad ? 'bad' : 'ok'}`;
    }
    els.rule.textContent = last
      ? `${last.free_drivers} свободных ${last.free_drivers < last.requests / 2 ? '<' : '≥'} ${last.requests} / 2 заказов`
      : '';
    els.free.textContent = String(world.freeByZone[id] ?? 0);
    const mult = world.priceMult[id] ?? 1;
    els.price.textContent = `×${mult.toFixed(2)}`;
    els.price.style.color = Math.abs(mult - 1) > 1e-9 ? 'var(--accent)' : '';
    els.req.textContent = last ? String(last.requests) : '—';
    els.prob.textContent = p == null ? '—' : p.toFixed(2);
    els.prob.style.color = p == null ? 'var(--dim)' : p >= 0.6 ? 'var(--red)' : p >= 0.3 ? 'var(--amber)' : '';
    els.model.textContent = fc ? 'прогноз модели' : '';
    els.prob.title = fc ? '' : 'Включите режим «Прогноз» в шапке';
    drawSpark(rows);
    renderRoadCounts();
  };

  const drawSpark = (rows) => {
    const canvas = els.spark;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    if (canvas.width !== Math.round(w * dpr)) canvas.width = Math.round(w * dpr);
    if (canvas.height !== Math.round(h * dpr)) canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const to = world.clock;
    const from = to - SPARK_HOURS * 60;
    const recent = rows.filter((row) => row._min !== undefined && row._min >= from && row._min + AGG_STEP <= to);
    const ticks = [0, 1, 2, 3].map((i) => hhmm(from + (i * SPARK_HOURS * 60) / 3));
    if (els.axis.dataset.key !== ticks.join()) {
      els.axis.dataset.key = ticks.join();
      els.axis.innerHTML = ticks.map((t) => `<span>${t}</span>`).join('');
    }
    ctx.strokeStyle = '#262b33';
    ctx.beginPath();
    ctx.moveTo(0, h - 0.5);
    ctx.lineTo(w, h - 0.5);
    ctx.stroke();
    if (!recent.length) return;
    let max = 4;
    for (const row of recent) max = Math.max(max, row.free_drivers, row.requests / 2);
    const x = (minute) => ((minute - from) / (to - from)) * w;
    const y = (value) => h - 2 - (value / max) * (h - 6);
    ctx.fillStyle = 'rgba(240, 86, 91, 0.16)';
    for (const row of recent) {
      if (row.deficit) ctx.fillRect(x(row._min), 0, Math.max(x(row._min + AGG_STEP) - x(row._min), 1), h);
    }
    const line = (value, color, width, dash) => {
      ctx.beginPath();
      recent.forEach((row, i) => {
        const px = x(row._min + AGG_STEP / 2);
        const py = y(value(row));
        if (i) ctx.lineTo(px, py);
        else ctx.moveTo(px, py);
      });
      ctx.setLineDash(dash);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.stroke();
      ctx.setLineDash([]);
    };
    line((row) => row.requests / 2, '#a3a9b2', 1.5, [4, 3]);
    line((row) => row.free_drivers, '#f5c518', 2, []);
  };

  // Дороги из района: сосед, время в пути и машины, уехавшие по ней.
  const roadItems = [];
  const renderRoads = () => {
    const id = world.selectedZone;
    els.roads.innerHTML = '';
    roadItems.length = 0;
    for (const road of world.roads) {
      if (road.from !== id && road.to !== id) continue;
      const other = road.from === id ? road.to : road.from;
      const li = document.createElement('li');
      const left = document.createElement('span');
      left.textContent = `→ ${world.zones.find((z) => z.id === other).name}`;
      const right = document.createElement('span');
      li.append(left, right);
      els.roads.appendChild(li);
      roadItems.push({ other, minutes: road.minutes, right });
    }
    roadItems.sort((a, b) => a.minutes - b.minutes);
    for (const item of roadItems) els.roads.appendChild(item.right.parentElement);
  };

  const renderRoadCounts = () => {
    const id = world.selectedZone;
    const counts = new Map();
    for (const trip of world.activeTrips.values()) {
      if (trip.origin !== id || world.clock < trip.pickupAt || world.clock >= trip.dropoffAt) continue;
      const hop = world.routes[trip.origin]?.[trip.dest]?.[1];
      if (hop != null) counts.set(hop, (counts.get(hop) ?? 0) + 1);
    }
    for (const item of roadItems) {
      item.right.textContent = `${item.minutes} мин · ${counts.get(item.other) ?? 0} в пути`;
    }
  };

  // ── Лента таймлайна: дефицит, дождь, заказы по окнам ──

  const strip = { start: null, counts: [], requests: [], rain: [], zones: 6, world: undefined, filled: -1, loading: false, lastFetch: -Infinity };

  const closedWindowIndex = () => {
    if (!strip.start) return -1;
    return Math.floor((world.clock - isoToMinute(strip.start)) / AGG_STEP) - 1;
  };

  const loadStrip = async () => {
    if (strip.loading) return;
    strip.loading = true;
    strip.lastFetch = performance.now();
    const worldName = world.activeWorld?.name ?? null;
    const filled = world.filled.count;
    try {
      const response = await fetch('/api/deficit-timeline');
      if (!response.ok) throw new Error(String(response.status));
      const data = await response.json();
      Object.assign(strip, {
        start: data.start,
        counts: data.counts || [],
        requests: data.requests || [],
        rain: data.rain || [],
        zones: data.zones || 6,
        world: worldName,
        filled,
      });
      timelineKey = '';
    } catch {
      // лента — дополнение таймлайна: при ошибке повторим позже
    } finally {
      strip.loading = false;
    }
  };

  let timelineKey = '';
  const drawTimeline = () => {
    const canvas = els.timeline;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h || !world.simStartUTC) return;
    const total = Math.max(world.filled.count - 1, 1);
    const key = `${w}x${h}|${strip.counts.length}|${total}|${Math.round((world.clock / total) * w * 2)}`;
    if (key === timelineKey) return;
    timelineKey = key;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const x0 = LABEL_W;
    const tw = w - x0;
    const xOf = (minute) => x0 + (minute / total) * tw;
    const rowY = { ticks: 10, deficit: 22, rain: 36, track: 54 };

    ctx.font = '11px "IBM Plex Mono", ui-monospace, monospace';
    ctx.fillStyle = '#7d838c';
    ctx.textBaseline = 'middle';
    const days = Math.ceil(total / 1440);
    for (let d = 0; d < days; d += 7) ctx.fillText(`день ${d + 1}`, xOf(d * 1440), rowY.ticks);
    ctx.font = '11px "IBM Plex Sans", system-ui, sans-serif';
    ctx.fillStyle = '#a3a9b2';
    ctx.fillText('дефицит', 0, rowY.deficit + 3);
    ctx.fillText('дождь', 0, rowY.rain + 3);

    ctx.fillStyle = '#16191e';
    ctx.fillRect(x0, rowY.deficit, tw, 6);
    ctx.fillRect(x0, rowY.rain, tw, 6);
    if (strip.start) {
      const start = isoToMinute(strip.start);
      const columns = Math.max(Math.floor(tw), 1);
      const peak = new Float32Array(columns);
      const wet = new Uint8Array(columns);
      strip.counts.forEach((count, i) => {
        const col = Math.floor(((start + i * AGG_STEP) / total) * columns);
        if (col < 0 || col >= columns) return;
        if (count > peak[col]) peak[col] = count;
        if (strip.rain[i]) wet[col] = 1;
      });
      for (let col = 0; col < columns; col += 1) {
        if (peak[col]) {
          ctx.fillStyle = `rgba(240, 86, 91, ${(0.35 + 0.65 * (peak[col] / strip.zones)).toFixed(3)})`;
          ctx.fillRect(x0 + col, rowY.deficit, 1, 6);
        }
        if (wet[col]) {
          ctx.fillStyle = '#6f9bd1';
          ctx.fillRect(x0 + col, rowY.rain, 1, 6);
        }
      }
    }
    const head = xOf(Math.min(Math.max(world.clock, 0), total));
    ctx.fillStyle = '#262b33';
    ctx.fillRect(x0, rowY.track, tw, 2);
    ctx.fillStyle = '#8a7425';
    ctx.fillRect(x0, rowY.track, head - x0, 2);
    ctx.fillStyle = '#f5c518';
    ctx.fillRect(head - 1, rowY.deficit - 4, 2, rowY.track - rowY.deficit + 4);
    ctx.beginPath();
    ctx.arc(head, rowY.track + 1, 6, 0, Math.PI * 2);
    ctx.fill();
  };

  const renderPosition = () => {
    const day = Math.floor(world.clock / 1440) + 1;
    els.now.textContent = `день ${day} · ${hhmm(world.clock)}`;
    els.of.textContent = `из ${Math.ceil(world.totalDays || 37)} дней`;
  };

  // ── Цикл обновления ──

  let lastUi = 0;
  const frame = (now) => {
    if (world.simStartUTC && document.body.dataset.tab === 'city') {
      placeCards();
      drawTimeline();
      if (now - lastUi >= UI_REFRESH_MS) {
        lastUi = now;
        renderCards();
        renderForecastNote();
        renderSummary();
        renderWeather();
        renderPanel();
        renderPosition();
        const worldName = world.activeWorld?.name ?? null;
        const grown = world.filled.count !== strip.filled;
        if (worldName !== strip.world || (grown && now - strip.lastFetch > STRIP_REFRESH_MS)) loadStrip();
      }
    }
    requestAnimationFrame(frame);
  };

  const toggleCharts = () => {
    const collapsed = els.charts.classList.toggle('collapsed');
    els.chartsToggle.setAttribute('aria-expanded', String(!collapsed));
    els.chartsToggle.firstChild.textContent = collapsed ? 'Развернуть ' : 'Свернуть ';
    try {
      localStorage.setItem('city.chartsOpen', collapsed ? '0' : '1');
    } catch {
      // приватный режим — без запоминания
    }
  };

  const waitZones = () => {
    if (!world.zones.length) {
      setTimeout(waitZones, 100);
      return;
    }
    buildCards();
    const preferred = world.zones.find((z) => z.name === 'center') ?? world.zones[0];
    selectZone(preferred.id);
  };

  const init = () => {
    if (!window.App) return;
    world = window.App.world;
    world.selectedZone = null;
    Object.assign(els, {
      districts: $('districts'),
      cars: $('cars-count'),
      pickup: $('pickup-count'),
      deficit: $('deficit-count'),
      ordersHour: $('orders-hour'),
      weather: $('weather'),
      card: $('zone-card'),
      name: $('zc-name'),
      type: $('zc-type'),
      status: $('zc-status'),
      rule: $('zc-rule'),
      free: $('zc-free'),
      price: $('zc-price'),
      req: $('zc-req'),
      prob: $('zc-prob'),
      model: $('zc-model'),
      spark: $('zc-spark'),
      axis: $('zc-axis'),
      roads: $('zc-roads'),
      timeline: $('deficit-strip'),
      now: $('tl-now'),
      of: $('tl-of'),
      charts: $('charts'),
      chartsToggle: $('charts-toggle'),
      fcNote: $('fc-note'),
      fcNoteText: $('fc-note-text'),
      fcJump: $('fc-jump'),
    });
    // Тестовая неделя начинается после 30 дней обучающей эпохи.
    $('fc-jump').addEventListener('click', jumpToTestWeek);
    $('fc-close').addEventListener('click', dismissNote);
    $('zc-close').addEventListener('click', closePanel);
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') closePanel();
    });
    els.chartsToggle.addEventListener('click', toggleCharts);
    try {
      if (localStorage.getItem('city.chartsOpen') === '0') toggleCharts();
    } catch {
      // без сохранённого состояния — развёрнуто, как в макете
    }
    window.addEventListener('resize', () => {
      timelineKey = '';
    });
    waitZones();
    requestAnimationFrame(frame);
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
