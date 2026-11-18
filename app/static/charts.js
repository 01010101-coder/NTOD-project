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
  const STATIC_REDRAW_MS = 500; // пересборка статичного слоя графика не чаще 2 Гц
  const PAD = { left: 32, right: 8, top: 6, bottom: 18 };

  const SERIES = [
    { key: 'requests', color: '#5aa9e6', axis: 'count' },
    { key: 'completed', color: '#3fb57a', axis: 'count' },
    { key: 'unfulfilled', color: '#e5484d', axis: 'count' },
    { key: 'free_drivers', color: '#f5c518', axis: 'count' },
    { key: 'price_multiplier', color: '#a48fd6', axis: 'price' },
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
  // Статичный слой графика (сетка, штриховка, линии серий) рисуется в
  // offscreen-канвас только при смене данных или размера; каждый кадр —
  // копия слоя и курсор. Полная перерисовка ~10 тыс. точек на серию стоит
  // десятки мс и давала рывки анимации карты.
  let layer = null;
  let layerKey = '';
  let rowsVersion = 0;

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
      rowsVersion += 1;
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
    const label = document.getElementById('chart-zone');
    const found = city?.zones.find((z) => z.id === zone);
    if (label && found) label.textContent = found.name;
    rows = [];
    rowsVersion += 1;
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

  const niceMax = (value, floor) => Math.max(floor, Math.ceil(value * 1.15));

  // График района — окно последних 24 часов до playhead (макет design/Main.html):
  // свободные машины, заказы за 5 минут, полосы дефицитных окон. Слой
  // пересобирается не чаще STATIC_REDRAW_MS: в окне ~288 точек на серию.
  const WINDOW_MIN = 24 * 60;
  const DAYS_SHORT = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'];

  const windowOf = (world) => {
    const to = Math.max(world.clock, WINDOW_MIN);
    return { from: to - WINDOW_MIN, to };
  };

  const stampOf = (minute) => {
    const iso = api.minuteToIso(minute);
    const [y, m, d] = iso.slice(0, 10).split('-').map(Number);
    const wd = (new Date(Date.UTC(y, m - 1, d)).getUTCDay() + 6) % 7;
    return `${DAYS_SHORT[wd]} ${iso.slice(11, 16)}`;
  };

  const renderStatic = (world, ctx, w, h) => {
    ctx.clearRect(0, 0, w, h);
    const { from, to } = windowOf(world);
    const plotW = w - PAD.left - PAD.right;
    const plotH = h - PAD.top - PAD.bottom;
    const xOf = (minute) => PAD.left + ((minute - from) / (to - from)) * plotW;
    const visible = [];
    for (const row of rows) {
      if (row._min === undefined) row._min = api.isoToMinute(row.ts);
      if (row._min + AGG_STEP > to) break;
      if (row._min >= from) visible.push(row);
    }
    let max = 10;
    for (const row of visible) max = Math.max(max, row.requests, row.free_drivers);
    max = niceMax(max, 10);
    const yOf = (value) => PAD.top + plotH * (1 - value / max);

    ctx.fillStyle = 'rgba(240, 86, 91, 0.12)';
    for (const row of visible) {
      if (row.deficit) ctx.fillRect(xOf(row._min), PAD.top, Math.max(xOf(row._min + AGG_STEP) - xOf(row._min), 1), plotH);
    }
    ctx.strokeStyle = '#262b33';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top + plotH + 0.5);
    ctx.lineTo(PAD.left + plotW, PAD.top + plotH + 0.5);
    ctx.stroke();
    ctx.strokeStyle = '#1d2127';
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top + plotH / 2);
    ctx.lineTo(PAD.left + plotW, PAD.top + plotH / 2);
    ctx.stroke();

    const line = (key, color, width) => {
      ctx.beginPath();
      visible.forEach((row, i) => {
        const x = xOf(row._min + AGG_STEP / 2);
        const y = yOf(row[key]);
        if (i) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.stroke();
    };
    line('requests', '#c9ccd1', 1.5);
    line('free_drivers', '#f5c518', 2);

    ctx.font = '11px "IBM Plex Mono", ui-monospace, monospace';
    ctx.fillStyle = '#7d838c';
    ctx.textBaseline = 'alphabetic';
    for (let i = 0; i <= 6; i += 1) {
      const minute = from + (WINDOW_MIN * i) / 6;
      ctx.textAlign = i === 0 ? 'left' : i === 6 ? 'right' : 'center';
      ctx.fillText(stampOf(minute), xOf(minute), h - 4);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(max), PAD.left - 8, PAD.top + 4);
    ctx.fillText('0', PAD.left - 8, PAD.top + plotH);
    if (!visible.length) {
      ctx.textAlign = 'center';
      ctx.fillText('метрики появятся после закрытия первого 5-минутного окна', w / 2, h / 2);
    }
  };

  const drawChart = (world) => {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return; // свёрнутая панель — рисовать некуда
    const bw = Math.round(w * dpr);
    const bh = Math.round(h * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    if (!layer) layer = document.createElement('canvas');
    const key = `${bw}x${bh}|${rowsVersion}|${Math.floor(world.clock)}`;
    const now = performance.now();
    const sizeChanged = layer.width !== bw || layer.height !== bh;
    if (key !== layerKey && (sizeChanged || now - lastDrawWall >= STATIC_REDRAW_MS)) {
      if (sizeChanged) {
        layer.width = bw;
        layer.height = bh;
      }
      const lctx = layer.getContext('2d');
      lctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      renderStatic(world, lctx, w, h);
      layerKey = key;
      lastDrawWall = now;
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, bw, bh);
    ctx.drawImage(layer, 0, 0);
  };

  const onChartClick = (event) => {
    // Клик — seek к началу 5-минутного окна под курсором в пределах 24 ч.
    const w = window.App?.world;
    if (!w || !w.simStartUTC || !w.filled.count) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const plotW = rect.width - PAD.left - PAD.right;
    if (x < PAD.left || x > PAD.left + plotW) return;
    const { from, to } = windowOf(w);
    const minute = from + ((x - PAD.left) / plotW) * (to - from);
    api.seekMinute(Math.floor(minute / AGG_STEP) * AGG_STEP);
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
    drawChart(world);
  };

  // Выбор зоны снаружи (клик по району на карте): селектор и данные графика.
  const selectZone = (id) => {
    if (id === zoneId) return;
    const select = document.getElementById('zone-select');
    if (select) select.value = String(id);
    reload(id);
  };

  window.CityCharts = {
    init,
    draw,
    onLogExtended,
    eventAt,
    syncControls,
    selectZone,
    zoneRows: () => rows, // метрики выбранной зоны (карточка района)
    zoneId: () => zoneId,
    reset: () => zoneId != null && reload(zoneId),
  };

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
    ctx.fillStyle = '#8a909c';
    ctx.font = '13px Onest, system-ui, sans-serif';
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
    ctx.font = '11px Onest, system-ui, sans-serif';

    // Полосы-заливки (дефицит) — под линиями.
    for (const band of bands) {
      const x0 = xOf(Math.max(band.x0, xMin));
      const x1 = xOf(Math.min(band.x1, xMax));
      if (x1 <= x0) continue;
      ctx.fillStyle = 'rgba(229, 72, 77, 0.16)';
      ctx.fillRect(x0, LAB_PAD.top, x1 - x0, plotH);
    }

    // Горизонтальная сетка с подписями значений.
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i += 1) {
      const y = LAB_PAD.top + (plotH * i) / 4;
      ctx.strokeStyle = 'rgba(48, 54, 66, 0.7)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(LAB_PAD.left, y);
      ctx.lineTo(LAB_PAD.left + plotW, y);
      ctx.stroke();
      ctx.fillStyle = '#8a909c';
      ctx.fillText(fmtAxisNumber(yMax - ((yMax - yMin) * i) / 4), LAB_PAD.left - 6, y);
    }

    // 5 вертикалей оси x: штампы времени или числа (loss, калибровка).
    ctx.textBaseline = 'alphabetic';
    for (let i = 0; i <= 4; i += 1) {
      const x = LAB_PAD.left + (plotW * i) / 4;
      ctx.strokeStyle = 'rgba(48, 54, 66, 0.4)';
      ctx.beginPath();
      ctx.moveTo(x, LAB_PAD.top);
      ctx.lineTo(x, LAB_PAD.top + plotH);
      ctx.stroke();
      ctx.fillStyle = '#8a909c';
      ctx.textAlign = i === 0 ? 'left' : i === 4 ? 'right' : 'center';
      const xv = xMin + ((xMax - xMin) * i) / 4;
      ctx.fillText(xIsTime ? labStamp(xv) : fmtAxisNumber(xv), x, h - 6);
    }

    // Линии серий с поддержкой штриховки (диагональ калибровки).
    for (const s of series) {
      if (!s.points.length) continue;
      // Серия-площадь (факт): заливка до нижней оси под линией.
      if (s.area) {
        ctx.beginPath();
        s.points.forEach((p, i) => {
          if (i === 0) ctx.moveTo(xOf(p.x), yOf(p.y));
          else ctx.lineTo(xOf(p.x), yOf(p.y));
        });
        const last = s.points[s.points.length - 1];
        ctx.lineTo(xOf(last.x), LAB_PAD.top + plotH);
        ctx.lineTo(xOf(s.points[0].x), LAB_PAD.top + plotH);
        ctx.closePath();
        ctx.fillStyle = s.area;
        ctx.fill();
      }
      ctx.beginPath();
      s.points.forEach((p, i) => {
        if (i === 0) ctx.moveTo(xOf(p.x), yOf(p.y));
        else ctx.lineTo(xOf(p.x), yOf(p.y));
      });
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width ?? 1.6;
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

  const drawValueBars = (canvas, { labels, values, color = '#f5c518' }) => {
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
    ctx.font = '11px Onest, system-ui, sans-serif';

    // Верхняя метка шкалы и базовая линия.
    ctx.fillStyle = '#8a909c';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(fmtAxisNumber(maxV), pad.left - 5, pad.top);
    ctx.strokeStyle = 'rgba(48, 54, 66, 0.7)';
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
    ctx.fillStyle = '#8a909c';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    values.forEach((_, i) => {
      if (i % step) return;
      const x = pad.left + slot * i + slot / 2;
      ctx.fillText(String(labels[i]), x, h - 4);
    });
  };

  // Кольцевая диаграмма (SVG-строка): сегменты [{value, color}], подпись в центре.
  const donut = ({ segments, size = 120, thickness = 12, center = '', sub = '', track = '#22262d' }) => {
    const r = (size - thickness) / 2;
    const c = size / 2;
    const len = 2 * Math.PI * r;
    const total = segments.reduce((sum, seg) => sum + Math.max(seg.value, 0), 0);
    let offset = 0;
    const arcs = segments
      .filter((seg) => seg.value > 0 && total > 0)
      .map((seg) => {
        const part = (seg.value / total) * len;
        const arc = `<circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="${seg.color}" stroke-width="${thickness}" stroke-dasharray="${part.toFixed(2)} ${(len - part).toFixed(2)}" stroke-dashoffset="${(-offset).toFixed(2)}"/>`;
        offset += part;
        return arc;
      })
      .join('');
    const esc = (text) => String(text).replace(/[&<>]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[ch]);
    return (
      `<svg class="donut" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" role="img">` +
      `<circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="${track}" stroke-width="${thickness}"/>` +
      `<g transform="rotate(-90 ${c} ${c})">${arcs}</g>` +
      `<text x="${c}" y="${c + (sub ? 2 : 7)}" text-anchor="middle" class="donut-v">${esc(center)}</text>` +
      (sub ? `<text x="${c}" y="${c + 20}" text-anchor="middle" class="donut-s">${esc(sub)}</text>` : '') +
      '</svg>'
    );
  };

  window.LabCharts = {
    donut, drawTimeSeries, drawValueBars };
})();
