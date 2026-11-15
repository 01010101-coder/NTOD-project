// Слайдер времени и графики метрик (change city-canvas, задачи 4.1–4.2).
// Слайдер покрывает заполненный диапазон лога (растёт по log-extended):
// драг шлёт seek с троттлом ~200 мс, клик — одиночный seek, движок следует
// локальным часам, пока пользователь не держит его (design.md, решение 6).
// Графики — самописный canvas-рендер (решение 7): пять линий по выбранной
// зоне из /api/metrics, штриховка дефицитных окон по правилу контракта
// (то же, что кайма на карте), вертикальный курсор playhead по локальным
// часам, клик по точке → seek к началу 5-минутного окна; дозагрузка
// from=<последний ts> с троттлом ~1 Гц, полная перезагрузка при смене зоны.

(() => {
  'use strict';

  const AGG_STEP = 5; // шаг агрегации метрик, минут (контракт данных §2)
  const DEFICIT_RATIO_LIMIT = 0.5; // правило дефицита §3.8
  const SEEK_THROTTLE_MS = 200; // троттл seek при драге слайдера
  const FETCH_THROTTLE_MS = 1000; // троттл дозагрузки метрик (~1 Гц)
  const REDRAW_INTERVAL_MS = 120; // перерисовка графиков ~8 Гц (слайдер — каждый кадр)
  const PAD = { left: 48, right: 48, top: 14, bottom: 24 };

  const SERIES = [
    { key: 'requests', color: '#8ecae6', axis: 'count' },
    { key: 'completed', color: '#7bd88f', axis: 'count' },
    { key: 'unfulfilled', color: '#ff5d5d', axis: 'count' },
    { key: 'free_drivers', color: '#ffd166', axis: 'count' },
    { key: 'price_multiplier', color: '#c792ea', axis: 'price' },
  ];

  let canvas = null;
  let ctx = null;
  let city = null;
  let api = null; // {seekMinute, isoToMinute, minuteToIso} из app.js
  let zoneId = null;
  let rows = []; // строки /api/metrics выбранной зоны, порядок ts; ts как в ответе
  const seenTs = new Set();
  let lastTs = null; // ISO последней загруженной строки (для from=)
  let fetchTimer = null;
  let lastFetchWall = -Infinity;
  let lastDrawWall = 0;
  let hatch = null; // паттерн штриховки дефицита

  // ── Данные метрик ──

  const isDeficit = (row) =>
    row.free_drivers / Math.max(row.requests, 1) < DEFICIT_RATIO_LIMIT || row.unfulfilled > 0;

  const fetchRows = async (zone, fromIso) => {
    const params = new URLSearchParams({ zone: String(zone) });
    if (fromIso) params.set('from', fromIso);
    try {
      const response = await fetch(`/api/metrics?${params}`);
      const data = await response.json();
      if (zone !== zoneId) return; // зону сменили, пока ответ летел
      for (const row of data.metrics) {
        if (seenTs.has(row.ts)) continue;
        seenTs.add(row.ts);
        row.deficit = isDeficit(row);
        rows.push(row);
      }
      rows.sort((a, b) => (a.ts < b.ts ? -1 : 1));
      if (rows.length) lastTs = rows[rows.length - 1].ts;
    } catch {
      // сеть моргнула — дозагрузка случится по следующему log-extended
    }
  };

  const onLogExtended = () => {
    if (zoneId == null) return;
    if (fetchTimer) return; // дозагрузка уже запланирована
    const wait = Math.max(0, FETCH_THROTTLE_MS - (performance.now() - lastFetchWall));
    fetchTimer = setTimeout(() => {
      fetchTimer = null;
      lastFetchWall = performance.now();
      fetchRows(zoneId, lastTs);
    }, wait);
  };

  const reload = (zone) => {
    zoneId = zone;
    rows = [];
    seenTs.clear();
    lastTs = null;
    fetchRows(zone, null);
  };

  // ── Слайдер времени ──

  let slider = null;
  let rangeFirst = null;
  let rangeLast = null;
  let dragging = false;
  let pendingSeek = null;
  let lastSeekSent = 0;

  const onSliderInput = () => {
    const minute = Number(slider.value);
    if (!dragging) {
      api.seekMinute(minute); // клик — одиночный seek
      return;
    }
    const now = performance.now();
    if (now - lastSeekSent >= SEEK_THROTTLE_MS) {
      lastSeekSent = now;
      api.seekMinute(minute);
    } else {
      pendingSeek = minute; // финальный seek уйдёт на отпускании
    }
  };

  const onSliderChange = () => {
    dragging = false;
    if (pendingSeek != null) {
      api.seekMinute(pendingSeek);
      pendingSeek = null;
    }
  };

  // Диапазон и края слайдера — по сообщениям state/log-extended, не только
  // в кадре: управление должно работать и при троттлящемся rAF.
  const syncControls = (world) => {
    const max = Math.max(world.filled.count - 1, 0);
    if (Number(slider.max) !== max) slider.max = String(max);
    rangeFirst.textContent = shortStamp(world.filled.first);
    rangeLast.textContent = shortStamp(world.filled.last);
  };

  const drawSlider = (world) => {
    syncControls(world);
    if (!dragging) {
      const max = Number(slider.max);
      slider.value = String(Math.round(Math.min(Math.max(world.clock, 0), max)));
    }
  };

  const shortStamp = (iso) => {
    if (!iso) return '—';
    return `${iso.slice(8, 10)}.${iso.slice(5, 7)} ${iso.slice(11, 16)}`;
  };

  // ── Графики ──

  const makeHatch = () => {
    const tile = document.createElement('canvas');
    tile.width = 8;
    tile.height = 8;
    const tctx = tile.getContext('2d');
    tctx.strokeStyle = 'rgba(255, 93, 93, 0.22)';
    tctx.lineWidth = 2;
    tctx.beginPath();
    tctx.moveTo(-2, 10);
    tctx.lineTo(10, -2);
    tctx.moveTo(-2, 2);
    tctx.lineTo(2, -2);
    tctx.moveTo(6, 10);
    tctx.lineTo(10, 6);
    tctx.stroke();
    return ctx.createPattern(tile, 'repeat');
  };

  const niceMax = (value, floor) => Math.max(floor, Math.ceil(value * 1.15));

  const drawChart = (world) => {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      hatch = null;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!hatch) hatch = makeHatch();

    const plotW = w - PAD.left - PAD.right;
    const plotH = h - PAD.top - PAD.bottom;
    const xMax = Math.max(world.filled.count - 1, AGG_STEP);
    const xOf = (minute) => PAD.left + (minute / xMax) * plotW;

    // Дефицитная штриховка фона: полоса окна [ts, ts+5] тем же правилом,
    // что и кайма зоны на карте.
    if (world.simStartUTC) {
      for (const row of rows) {
        if (!row.deficit) continue;
        const start = api.isoToMinute(row.ts);
        ctx.fillStyle = hatch;
        ctx.fillRect(xOf(start), PAD.top, Math.max(xOf(start + AGG_STEP) - xOf(start), 2), plotH);
      }
    }

    // Сетка и оси.
    let maxCount = 0;
    let maxPrice = 0;
    for (const row of rows) {
      maxCount = Math.max(maxCount, row.requests, row.completed, row.unfulfilled, row.free_drivers);
      maxPrice = Math.max(maxPrice, row.price_multiplier);
    }
    const countMax = niceMax(maxCount, 10);
    const priceMax = niceMax(maxPrice, 2);
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillStyle = '#8b98a9';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i += 1) {
      const y = PAD.top + (plotH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(PAD.left + plotW, y);
      ctx.strokeStyle = 'rgba(38, 46, 59, 0.7)';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillText(String(Math.round((countMax * (4 - i)) / 4)), PAD.left - 6, y);
    }
    ctx.textAlign = 'left';
    ctx.fillText(`×${(priceMax / 4).toFixed(1)}`, PAD.left + plotW + 6, PAD.top + plotH * 0.75);
    ctx.fillText(`×${priceMax.toFixed(1)}`, PAD.left + plotW + 6, PAD.top + plotH * 0.25);

    // Вертикали времени: шаг — первый из ряда, дающий ≤ 10 подписей.
    const step = [60, 120, 180, 360, 720, 1440, 2880].find((s) => xMax / s <= 10) ?? 2880;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    const first = world.filled.count ? api.isoToMinute(world.filled.first) : 0;
    for (let m = Math.ceil(first / step) * step; m <= xMax; m += step) {
      const x = xOf(m);
      ctx.beginPath();
      ctx.moveTo(x, PAD.top);
      ctx.lineTo(x, PAD.top + plotH);
      ctx.strokeStyle = 'rgba(38, 46, 59, 0.4)';
      ctx.stroke();
      ctx.fillText(shortStamp(api.minuteToIso(m)), x, h - 8);
    }

    // Линии серий: count-ряды по левой оси, тариф по правой.
    for (const series of SERIES) {
      if (!rows.length) break;
      ctx.beginPath();
      let started = false;
      for (const row of rows) {
        if (!world.simStartUTC) break;
        const x = xOf(api.isoToMinute(row.ts));
        const value = row[series.key];
        const y =
          series.axis === 'count'
            ? PAD.top + plotH * (1 - value / countMax)
            : PAD.top + plotH * (1 - value / priceMax);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.strokeStyle = series.color;
      ctx.lineWidth = 1.6;
      ctx.stroke();
    }

    if (!rows.length) {
      ctx.fillStyle = '#8b98a9';
      ctx.font = '13px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('метрики появятся после закрытия первого 5-минутного окна', w / 2, h / 2);
    }

    // Курсор playhead по локальным часам.
    const cursorX = xOf(Math.min(Math.max(world.clock, 0), xMax));
    ctx.beginPath();
    ctx.moveTo(cursorX, PAD.top - 4);
    ctx.lineTo(cursorX, PAD.top + plotH);
    ctx.strokeStyle = 'rgba(230, 237, 243, 0.85)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cursorX - 5, PAD.top - 4);
    ctx.lineTo(cursorX + 5, PAD.top - 4);
    ctx.lineTo(cursorX, PAD.top + 3);
    ctx.closePath();
    ctx.fillStyle = '#e6edf3';
    ctx.fill();
  };

  const onChartClick = (event) => {
    // Геометрия считается на месте, а не по последнему кадру: клик должен
    // работать и когда перерисовка ещё не случилась (скрытая вкладка).
    const w = window.App?.world;
    if (!w || !w.simStartUTC || !w.filled.count) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const plotW = rect.width - PAD.left - PAD.right;
    if (x < PAD.left || x > PAD.left + plotW) return;
    const xMax = Math.max(w.filled.count - 1, AGG_STEP);
    const minute = ((x - PAD.left) / plotW) * xMax;
    const windowStart = Math.floor(minute / AGG_STEP) * AGG_STEP;
    api.seekMinute(windowStart); // seek к началу 5-минутного окна (зажим в App)
  };

  // Метка события окна, содержащего минуту (панель спрашивает вне кадра).
  const eventAt = (minute) => {
    if (!api.simStarted()) return '';
    for (const row of rows) {
      if (row._min === undefined) row._min = api.isoToMinute(row.ts);
      if (row._min > minute) break; // rows отсортированы по ts
      if (minute < row._min + AGG_STEP) return row.event;
    }
    return '';
  };

  // ── Инициализация ──

  const init = (canvasEl, cityData, apiObj) => {
    canvas = canvasEl;
    ctx = canvas.getContext('2d');
    city = cityData;
    api = apiObj;

    slider = document.getElementById('time-slider');
    rangeFirst = document.getElementById('range-first');
    rangeLast = document.getElementById('range-last');
    slider.addEventListener('pointerdown', () => {
      dragging = true;
    });
    slider.addEventListener('input', onSliderInput);
    slider.addEventListener('change', onSliderChange);

    canvas.addEventListener('click', onChartClick);

    const select = document.getElementById('zone-select');
    for (const zone of city.zones) {
      const option = document.createElement('option');
      option.value = String(zone.id);
      option.textContent = `${zone.id} · ${zone.name}`;
      select.appendChild(option);
    }
    // Дефолт — центр: по имени, затем по архетипу, затем первая зона.
    const preferred =
      city.zones.find((zone) => zone.name === 'center') ??
      city.zones.find((zone) => zone.archetype === 'business') ??
      city.zones[0];
    select.value = String(preferred.id);
    select.addEventListener('change', () => reload(Number(select.value)));
    reload(preferred.id);
  };

  const draw = (world) => {
    // Слайдер следует за часами в каждом кадре; сам график перерисовывается
    // с троттлом по внутреннему performance.now (единицы не зависят от вызова).
    drawSlider(world);
    if (!api.simStarted()) return;
    const now = performance.now();
    if (now - lastDrawWall < REDRAW_INTERVAL_MS) return;
    lastDrawWall = now;
    drawChart(world);
  };

  window.CityCharts = { init, draw, onLogExtended, eventAt, syncControls, reset: () => zoneId != null && reload(zoneId) };

  // ── Хелперы лаборатории (change m3-ml-tab, задача 6.1) ──
  // Переиспользуемые рендеры для вкладки «Лаборатория»: ось времени (или
  // числовая ось — кривая loss, калибровка), серии [{name, color, points,
  // dash?}], полосы-заливки интервалов (deficit_prob > 0.5) и столбиковая
  // диаграмма (гистограмма остатков, MAE по часу). График города не
  // мигрирует (Non-Goal design.md m3-ml-tab) — хелперы живут рядом.

  const LAB_PAD = { left: 48, right: 14, top: 26, bottom: 22 };

  const sizeLabCanvas = (canvas) => {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    // Скрытый канвас (уходим на «Город» до завершения fetch лаборатории) имеет
    // нулевой layout: рисование затёрло бы содержимое 0-размерным буфером.
    // Пропуск — перерисовка по показу вкладки (redrawLabCharts в app.js).
    if (!w || !h) return null;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const context = canvas.getContext('2d');
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: context, w, h };
  };

  const fmtAxisNumber = (value) => {
    const abs = Math.abs(value);
    if (abs >= 100) return String(Math.round(value));
    if (abs >= 10) return value.toFixed(1);
    return value.toFixed(2);
  };

  const labStamp = (ms) => {
    // Наивные времена эпохи конвертируются клиентом в UTC-ms — подписи по
    // UTC-геттерам повторяют настенные времена БД, без локального сдвига.
    const dt = new Date(ms);
    const pad = (n) => String(n).padStart(2, '0');
    return `${pad(dt.getUTCDate())}.${pad(dt.getUTCMonth() + 1)} ${pad(dt.getUTCHours())}:${pad(dt.getUTCMinutes())}`;
  };

  const labEmptyText = (ctx, w, h) => {
    ctx.fillStyle = '#8b98a9';
    ctx.font = '13px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('нет данных', w / 2, h / 2);
  };

  const drawTimeSeries = (
    canvas,
    { series, bands = [], includeZero = false, xIsTime = true, legend = true }
  ) => {
    const sized = sizeLabCanvas(canvas);
    if (!sized) return;
    const { ctx, w, h } = sized;
    ctx.clearRect(0, 0, w, h);
    const points = [];
    for (const s of series) for (const p of s.points) points.push(p);
    if (!points.length) {
      labEmptyText(ctx, w, h);
      return;
    }
    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const p of points) {
      xMin = Math.min(xMin, p.x);
      xMax = Math.max(xMax, p.x);
      yMin = Math.min(yMin, p.y);
      yMax = Math.max(yMax, p.y);
    }
    if (includeZero) {
      yMin = Math.min(yMin, 0);
      yMax = Math.max(yMax, 0);
    }
    if (xMax === xMin) xMax = xMin + 1;
    if (yMax === yMin) {
      yMax += 1;
      yMin -= 1;
    }
    const yPad = (yMax - yMin) * 0.08;
    yMin -= yPad;
    yMax += yPad;

    const plotW = w - LAB_PAD.left - LAB_PAD.right;
    const plotH = h - LAB_PAD.top - LAB_PAD.bottom;
    const xOf = (x) => LAB_PAD.left + ((x - xMin) / (xMax - xMin)) * plotW;
    const yOf = (y) => LAB_PAD.top + plotH * (1 - (y - yMin) / (yMax - yMin));
    ctx.font = '11px system-ui, sans-serif';

    // Полосы-заливки (дефицит) — под линиями.
    for (const band of bands) {
      const x0 = xOf(Math.max(band.x0, xMin));
      const x1 = xOf(Math.min(band.x1, xMax));
      if (x1 <= x0) continue;
      ctx.fillStyle = 'rgba(255, 93, 93, 0.16)';
      ctx.fillRect(x0, LAB_PAD.top, x1 - x0, plotH);
    }

    // Горизонтальная сетка с подписями значений.
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i += 1) {
      const y = LAB_PAD.top + (plotH * i) / 4;
      ctx.strokeStyle = 'rgba(38, 46, 59, 0.7)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(LAB_PAD.left, y);
      ctx.lineTo(LAB_PAD.left + plotW, y);
      ctx.stroke();
      ctx.fillStyle = '#8b98a9';
      ctx.fillText(fmtAxisNumber(yMax - ((yMax - yMin) * i) / 4), LAB_PAD.left - 6, y);
    }

    // 5 вертикалей оси x: штампы времени или числа (loss, калибровка).
    ctx.textBaseline = 'alphabetic';
    for (let i = 0; i <= 4; i += 1) {
      const x = LAB_PAD.left + (plotW * i) / 4;
      ctx.strokeStyle = 'rgba(38, 46, 59, 0.4)';
      ctx.beginPath();
      ctx.moveTo(x, LAB_PAD.top);
      ctx.lineTo(x, LAB_PAD.top + plotH);
      ctx.stroke();
      ctx.fillStyle = '#8b98a9';
      ctx.textAlign = i === 0 ? 'left' : i === 4 ? 'right' : 'center';
      const xv = xMin + ((xMax - xMin) * i) / 4;
      ctx.fillText(xIsTime ? labStamp(xv) : fmtAxisNumber(xv), x, h - 6);
    }

    // Линии серий с поддержкой штриховки (диагональ калибровки).
    for (const s of series) {
      if (!s.points.length) continue;
      ctx.beginPath();
      s.points.forEach((p, i) => {
        if (i === 0) ctx.moveTo(xOf(p.x), yOf(p.y));
        else ctx.lineTo(xOf(p.x), yOf(p.y));
      });
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.6;
      ctx.setLineDash(s.dash || []);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Легенда серий в шапке канваса (у «Сравнения» она вынесена в HTML —
    // подписи веток и цвета рисует вкладка, дубль в канвасе не нужен).
    if (!legend) return;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    let lx = LAB_PAD.left + 2;
    for (const s of series) {
      if (!s.points.length) continue;
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, 12, 12, 3);
      lx += 16;
      ctx.fillStyle = '#c7d0da';
      ctx.fillText(s.name, lx, 13);
      lx += ctx.measureText(s.name).width + 14;
    }
  };

  const drawValueBars = (canvas, { labels, values, color = '#ffd166' }) => {
    const sized = sizeLabCanvas(canvas);
    if (!sized) return;
    const { ctx, w, h } = sized;
    ctx.clearRect(0, 0, w, h);
    const present = values.filter((v) => v != null);
    if (!present.length) {
      labEmptyText(ctx, w, h);
      return;
    }
    const pad = { left: 42, right: 10, top: 10, bottom: 20 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;
    const maxV = Math.max(...present, 1e-9) * 1.15;
    const slot = plotW / values.length;
    const barW = Math.max(slot * 0.72, 1);
    ctx.font = '11px system-ui, sans-serif';

    // Верхняя метка шкалы и базовая линия.
    ctx.fillStyle = '#8b98a9';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(fmtAxisNumber(maxV), pad.left - 5, pad.top);
    ctx.strokeStyle = 'rgba(38, 46, 59, 0.7)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    ctx.fillStyle = color;
    values.forEach((v, i) => {
      if (v == null) return;
      const barH = (v / maxV) * plotH;
      ctx.fillRect(pad.left + slot * i + (slot - barW) / 2, pad.top + plotH - barH, barW, barH);
    });

    // Подписи по оси x: шаг — дающий ≤ 12 меток.
    const step = Math.ceil(values.length / 12);
    ctx.fillStyle = '#8b98a9';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    values.forEach((_, i) => {
      if (i % step) return;
      const x = pad.left + slot * i + slot / 2;
      ctx.fillText(String(labels[i]), x, h - 4);
    });
  };

  window.LabCharts = { drawTimeSeries, drawValueBars };
})();
