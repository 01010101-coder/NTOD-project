// Карта живого города (change city-canvas, задачи 3.1–3.4): canvas-рендер
// без библиотек. Зоны по map-координатам /api/city (скейл к канвасу), рёбра
// дорог, декоративные парки в пустотах графа (игровая стилистика, без OSM).
// Активные машины — интерполяция равномерно по длине полилинии маршрута
// (design.md, решение 5): фаза «к пассажиру» [assigned_at, pickup_at] по
// routes[driver_zone][origin], «в поездке» [pickup_at, dropoff_at] по
// routes[origin][dest]; состав ведёт App (снимок + поток записей). Бейджи:
// счётчик свободных из free_by_zone, ×N при тарифе ≠ 1; кайма дефицита —
// закрытое окно по правилу контракта (считает App из потока). Ожидание —
// пульсирующие маркеры, истечение — красная вспышка с клиентским TTL.

(() => {
  'use strict';

  const ZONE_HALF_W = 96; // полуширина блока зоны, px
  const ZONE_HALF_H = 62; // полувысота блока зоны, px
  const PAD = 90; // поля карты, px
  const MULT_EPS = 1e-9; // тариф сравнивается с 1.0 с допуском float
  const MAX_MARKERS = 12; // видимых маркеров ожидания на зоне, дальше «+N»

  const ICONS = {
    residential: '🏠',
    business: '🏢',
    airport: '✈️',
    mixed: '🏬',
  };
  const FILLS = {
    residential: '#2c4a6e',
    business: '#6b5b2e',
    airport: '#4a3566',
    mixed: '#2e6b5e',
  };
  const STROKES = {
    residential: '#4a6f9e',
    business: '#9e8b4a',
    airport: '#7357a1',
    mixed: '#4a9e8b',
  };

  let canvas = null;
  let ctx = null;
  let city = null;
  let view = null; // {centers: [{x,y}], parks: [{x,y}], w, h}

  const roundRect = (x, y, w, h, r) => {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  };

  const chip = (x, y, text, bg, fg) => {
    ctx.font = '600 12px system-ui, sans-serif';
    const w = ctx.measureText(text).width + 14;
    const h = 20;
    roundRect(x - w / 2, y - h / 2, w, h, 6);
    ctx.fillStyle = bg;
    ctx.fill();
    ctx.fillStyle = fg;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x, y + 0.5);
  };

  // ── Геометрия: пересчёт при изменении размера канваса ──

  const resize = () => {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (view && canvas.width === Math.round(w * dpr) && canvas.height === Math.round(h * dpr)) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const xs = city.zones.map((zone) => zone.x);
    const ys = city.zones.map((zone) => zone.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const scale = Math.min((w - PAD * 2) / Math.max(maxX - minX, 1), (h - PAD * 2) / Math.max(maxY - minY, 1));
    const offX = (w - (maxX - minX) * scale) / 2;
    const offY = (h - (maxY - minY) * scale) / 2;
    const toScreen = (zone) => ({
      x: (zone.x - minX) * scale + offX,
      y: (zone.y - minY) * scale + offY,
    });
    const centers = new Map(city.zones.map((zone) => [zone.id, toScreen(zone)]));
    view = { centers, w, h, minX, minY, scale, offX, offY };

    // Декор: парки на серединах пар зон без прямого ребра дороги —
    // заполняют пустоты графа, не мешая чтению зон и дорог.
    const linked = new Set(city.roads.map((road) => key(road.from, road.to)));
    view.parks = [];
    for (let i = 0; i < city.zones.length; i += 1) {
      for (let j = i + 1; j < city.zones.length; j += 1) {
        const a = city.zones[i].id;
        const b = city.zones[j].id;
        if (linked.has(key(a, b))) continue;
        const pa = centers.get(a);
        const pb = centers.get(b);
        view.parks.push({
          x: (pa.x + pb.x) / 2,
          y: (pa.y + pb.y) / 2,
          r: 16 + ((i * 7 + j * 13) % 3) * 5, // детерминированный «случайный» размер
        });
      }
    }
  };

  const key = (a, b) => `${Math.min(a, b)}-${Math.max(a, b)}`;

  // ── Машины: позиция фазы вдоль полилинии маршрута ──

  const tripPosition = (world, trip) => {
    const t = world.clock;
    if (t >= trip.dropoffAt) return null; // поездка закончилась — запись высадки ещё не дошла
    let routeIds;
    let start;
    let end;
    let phase;
    if (t < trip.pickupAt) {
      routeIds = world.routes[trip.driverZone][trip.origin];
      start = trip.assignedAt;
      end = trip.pickupAt;
      phase = 'pickup';
    } else {
      routeIds = world.routes[trip.origin][trip.dest];
      start = trip.pickupAt;
      end = trip.dropoffAt;
      phase = 'trip';
    }
    const pts = routeIds.map((id) => view.centers.get(id));
    if (pts.length < 2) return { x: pts[0].x, y: pts[0].y, angle: -Math.PI / 2, phase };
    const progress = end > start ? Math.min(Math.max((t - start) / (end - start), 0), 1) : 1;
    // Равномерно по длине полилинии: доля времени → доля пройденного пути.
    let total = 0;
    const segs = [];
    for (let i = 1; i < pts.length; i += 1) {
      const len = Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
      segs.push(len);
      total += len;
    }
    let rest = progress * total;
    for (let i = 0; i < segs.length; i += 1) {
      if (rest > segs[i] && i < segs.length - 1) {
        rest -= segs[i];
        continue;
      }
      const frac = segs[i] > 0 ? rest / segs[i] : 1;
      return {
        x: pts[i].x + (pts[i + 1].x - pts[i].x) * frac,
        y: pts[i].y + (pts[i + 1].y - pts[i].y) * frac,
        angle: Math.atan2(pts[i + 1].y - pts[i].y, pts[i + 1].x - pts[i].x),
        phase,
      };
    }
    return { x: pts[0].x, y: pts[0].y, angle: 0, phase };
  };

  const drawCar = (pos) => {
    ctx.save();
    ctx.translate(pos.x, pos.y);
    ctx.rotate(pos.angle);
    roundRect(-8, -4, 16, 8, 3);
    if (pos.phase === 'trip') {
      ctx.fillStyle = '#4cc9f0';
      ctx.fill();
    } else {
      ctx.fillStyle = 'rgba(255, 209, 102, 0.3)';
      ctx.fill();
      ctx.strokeStyle = '#ffd166';
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }
    ctx.fillStyle = 'rgba(255, 255, 255, 0.75)'; // лобовое стекло — курс движения
    ctx.fillRect(3, -3, 2.5, 6);
    ctx.restore();
  };

  // ── Слои карты ──

  const drawDecor = () => {
    for (const park of view.parks) {
      ctx.beginPath();
      ctx.ellipse(park.x, park.y, park.r * 1.4, park.r, 0, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(46, 84, 48, 0.4)';
      ctx.fill();
      for (let k = 0; k < 3; k += 1) {
        const ang = (k / 3) * Math.PI * 2 + park.r;
        ctx.beginPath();
        ctx.arc(park.x + Math.cos(ang) * park.r * 0.5, park.y + Math.sin(ang) * park.r * 0.4, 4, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(74, 124, 78, 0.55)';
        ctx.fill();
      }
    }
  };

  const drawRoads = () => {
    ctx.lineCap = 'round';
    for (const road of city.roads) {
      const a = view.centers.get(road.from);
      const b = view.centers.get(road.to);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = '#232a36';
      ctx.lineWidth = 16;
      ctx.stroke();
      ctx.beginPath();
      ctx.setLineDash([10, 9]);
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = 'rgba(139, 152, 169, 0.5)';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);
    }
  };

  const drawZones = (world, now) => {
    ctx.textAlign = 'center';
    const forecastMode = world.paintMode === 'forecast';
    for (const zone of city.zones) {
      const c = view.centers.get(zone.id);
      const x = c.x - ZONE_HALF_W;
      const y = c.y - ZONE_HALF_H;
      roundRect(x, y, ZONE_HALF_W * 2, ZONE_HALF_H * 2, 14);
      ctx.fillStyle = FILLS[zone.archetype] ?? '#3a4356';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = STROKES[zone.archetype] ?? '#5a6376';
      ctx.stroke();

      // Тумблер m6-worlds-ui: «факт» — дефицит закрытого окна (как раньше),
      // «прогноз» — deficit_prob ≥ 0.5 ближайшего закрытого бакета выбранной
      // модели; нет строки (прогрев/дыра) — каймы нет.
      const deficit = forecastMode
        ? world.forecastDeficit?.[zone.id] === true
        : world.deficit[zone.id];
      if (deficit) {
        // Кайма дефицита — пульсирующее красное свечение.
        const pulse = 0.55 + 0.35 * Math.sin(now * 3);
        ctx.save();
        ctx.shadowColor = 'rgba(255, 93, 93, 0.9)';
        ctx.shadowBlur = 14;
        roundRect(x - 3, y - 3, ZONE_HALF_W * 2 + 6, ZONE_HALF_H * 2 + 6, 16);
        ctx.strokeStyle = `rgba(255, 93, 93, ${pulse.toFixed(3)})`;
        ctx.lineWidth = 3.5;
        ctx.stroke();
        ctx.restore();
      }

      const icon = ICONS[zone.archetype] ?? '🏙';
      ctx.font = '700 15px system-ui, sans-serif';
      ctx.fillStyle = '#e6edf3';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(`${icon} ${zone.name}`, c.x, c.y - ZONE_HALF_H + 26);

      // Счётчик свободных машин — из free_by_zone последней применённой минуты.
      const free = world.freeByZone[zone.id] ?? 0;
      chip(x + 24, y + ZONE_HALF_H * 2 - 16, `🚗 ${free}`, 'rgba(11, 14, 20, 0.8)', '#e6edf3');

      const mult = world.priceMult[zone.id] ?? 1;
      if (Math.abs(mult - 1) > MULT_EPS) {
        chip(c.x + ZONE_HALF_W - 26, y + 16, `×${Math.round(mult * 100) / 100}`, 'rgba(255, 209, 102, 0.92)', '#4a3608');
      }
    }
  };

  const drawPending = (world, now) => {
    const byZone = new Map();
    for (const order of world.pending.values()) {
      if (!byZone.has(order.origin)) byZone.set(order.origin, []);
      byZone.get(order.origin).push(order);
    }
    for (const [zoneId, orders] of byZone) {
      const c = view.centers.get(zoneId);
      const shown = Math.min(orders.length, MAX_MARKERS);
      for (let i = 0; i < shown; i += 1) {
        const x = c.x + (i - (shown - 1) / 2) * 17;
        const y = c.y + 22;
        const pulse = 4.5 + 1.6 * Math.sin(now * 4 + i * 1.7);
        ctx.beginPath();
        ctx.arc(x, y, pulse + 3.5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 159, 67, 0.25)';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x, y, pulse, 0, Math.PI * 2);
        ctx.fillStyle = '#ff9f43';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      if (orders.length > MAX_MARKERS) {
        chip(c.x + ZONE_HALF_W - 24, c.y + 22, `+${orders.length - MAX_MARKERS}`, 'rgba(11, 14, 20, 0.8)', '#ff9f43');
      }
    }
  };

  const drawFlashes = (world) => {
    world.flashes = world.flashes.filter((flash) => world.clock - flash.born < 5);
    for (const flash of world.flashes) {
      const age = world.clock - flash.born; // сим-минут с истечения
      const c = view.centers.get(flash.zone);
      const frac = age / 5;
      ctx.beginPath();
      ctx.arc(c.x, c.y, 10 + frac * 38, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255, 93, 93, ${(0.8 * (1 - frac)).toFixed(3)})`;
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(c.x, c.y, 7, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 93, 93, ${(0.35 * (1 - frac)).toFixed(3)})`;
      ctx.fill();
    }
  };

  const drawCars = (world) => {
    for (const trip of world.activeTrips.values()) {
      const pos = tripPosition(world, trip);
      if (pos) drawCar(pos);
    }
  };

  const draw = (world, now) => {
    if (!canvas) return;
    resize();
    ctx.clearRect(0, 0, view.w, view.h);
    drawDecor();
    drawRoads();
    drawZones(world, now);
    drawPending(world, now);
    drawFlashes(world);
    drawCars(world);
  };

  window.CityMap = {
    init(canvasEl, cityData) {
      canvas = canvasEl;
      ctx = canvas.getContext('2d');
      city = cityData;
      view = null;
    },
    draw,
  };
})();
