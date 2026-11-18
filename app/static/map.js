// Карта живого города (по макету design/Main.html): canvas рисует дороги,
// машины, ореолы риска вокруг районов, время суток и дождь. Плашки районов —
// DOM поверх канваса (hud.js), позиции берут из CityMap.layout().
// Машины — интерполяция по длине маршрута (design.md city-canvas, решение 5):
// фаза «к пассажиру» [assigned_at, pickup_at] по routes[driver_zone][origin],
// «в поездке» [pickup_at, dropoff_at] по routes[origin][dest]; состав ведёт
// App (снимок + поток записей).

(() => {
  'use strict';

  const LANE = 3; // смещение полосы от оси дороги, px: встречные потоки не сливаются
  const HALO_R = 100; // радиус ореола риска вокруг района, px
  const C = {
    road: '#1d2127',
    label: '#a3a9b2',
    labelBg: '#0d0f12',
    labelBorder: '#2a2f37',
    trip: '#f5c518',
    pickup: '#ebe8e2',
    rain: 'rgba(150, 190, 230, ',
  };
  const MONO = '"IBM Plex Mono", ui-monospace, monospace';

  let canvas = null;
  let ctx = null;
  let city = null;
  let view = null; // {centers: Map(id → {x, y}), w, h, version}
  let version = 0;
  let rainLevel = 0;
  let lastNow = null;
  const drops = Array.from({ length: 160 }, (_, i) => ({
    x: ((i * 97) % 1000) / 1000,
    y: ((i * 61) % 1000) / 1000,
    v: 0.55 + ((i * 37) % 100) / 220,
  }));

  // ── Геометрия: районы в прямоугольнике карты с полями под сводку и легенду ──

  const resize = () => {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (view && view.w === w && view.h === h && canvas.width === Math.round(w * dpr)) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const xs = city.zones.map((zone) => zone.x);
    const ys = city.zones.map((zone) => zone.y);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const spanX = Math.max(Math.max(...xs) - minX, 1);
    const spanY = Math.max(Math.max(...ys) - minY, 1);
    // Поля: по бокам — полплашки, сверху — сводка, снизу — легенда.
    const left = 110;
    const right = 110;
    const top = 150;
    const bottom = 100;
    const centers = new Map(
      city.zones.map((zone) => [
        zone.id,
        {
          x: left + ((zone.x - minX) / spanX) * Math.max(w - left - right, 1),
          y: top + ((zone.y - minY) / spanY) * Math.max(h - top - bottom, 1),
        },
      ])
    );
    version += 1;
    view = { centers, w, h, version };
  };

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
    if (pts.length < 2) return { x: pts[0].x, y: pts[0].y, angle: 0, phase };
    const progress = end > start ? Math.min(Math.max((t - start) / (end - start), 0), 1) : 1;
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
      const dx = pts[i + 1].x - pts[i].x;
      const dy = pts[i + 1].y - pts[i].y;
      const len = segs[i] || 1;
      // Правосторонняя полоса: сдвиг по нормали к направлению движения.
      return {
        x: pts[i].x + dx * frac - (dy / len) * LANE,
        y: pts[i].y + dy * frac + (dx / len) * LANE,
        angle: Math.atan2(dy, dx),
        phase,
      };
    }
    return { x: pts[0].x, y: pts[0].y, angle: 0, phase };
  };

  // ── Слои ──

  const hourOf = (world) => (((world.clock / 60) % 24) + 24) % 24;

  // Ночь — затемнение сетки, рассвет и закат — тёплое зарево снизу.
  const drawSky = (world) => {
    const hour = hourOf(world);
    const day = 0.5 - 0.5 * Math.cos(((hour - 3) / 24) * Math.PI * 2);
    const night = 1 - day * day * (3 - 2 * day);
    if (night > 0.02) {
      ctx.fillStyle = `rgba(4, 6, 10, ${(0.45 * night).toFixed(3)})`;
      ctx.fillRect(0, 0, view.w, view.h);
    }
    const dusk = Math.max(
      Math.exp(-(((hour - 6.5) / 1.3) ** 2)),
      Math.exp(-(((hour - 19) / 1.5) ** 2))
    );
    if (dusk > 0.02) {
      const glow = ctx.createRadialGradient(view.w / 2, view.h * 1.2, 0, view.w / 2, view.h * 1.2, view.h * 1.2);
      glow.addColorStop(0, `rgba(230, 140, 60, ${(0.18 * dusk).toFixed(3)})`);
      glow.addColorStop(1, 'rgba(230, 140, 60, 0)');
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, view.w, view.h);
    }
  };

  // Ореол: факт — дефицит района (тумблер paintMode), прогноз — вероятность
  // p ≥ 0.3; если вероятность не пришла, но бакет помечен — как дефицит.
  const drawHalos = (world) => {
    const forecast = world.paintMode === 'forecast';
    for (const zone of city.zones) {
      let alpha = 0;
      if (forecast) {
        const p = world.forecastProb?.[zone.id];
        if (p != null && p >= 0.3) alpha = p * 0.22;
        else if (p == null && world.forecastDeficit?.[zone.id]) alpha = 0.16;
      } else if (world.deficit[zone.id]) {
        alpha = 0.16;
      }
      if (!alpha) continue;
      const c = view.centers.get(zone.id);
      ctx.beginPath();
      ctx.arc(c.x, c.y, HALO_R, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(240, 86, 91, ${alpha.toFixed(3)})`;
      ctx.fill();
    }
  };

  const drawRoads = () => {
    ctx.lineCap = 'round';
    ctx.lineWidth = 14;
    ctx.strokeStyle = C.road;
    for (const road of city.roads) {
      const a = view.centers.get(road.from);
      const b = view.centers.get(road.to);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
  };

  const drawRoadLabels = () => {
    ctx.font = `11px ${MONO}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const road of city.roads) {
      const a = view.centers.get(road.from);
      const b = view.centers.get(road.to);
      const text = `${road.minutes} мин`;
      const x = (a.x + b.x) / 2;
      const y = (a.y + b.y) / 2;
      const w = ctx.measureText(text).width + 12;
      ctx.fillStyle = C.labelBg;
      ctx.strokeStyle = C.labelBorder;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(x - w / 2, y - 9, w, 18, 4);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = C.label;
      ctx.fillText(text, x, y + 0.5);
    }
  };

  const drawCars = (world) => {
    const pickups = [];
    ctx.fillStyle = C.trip;
    for (const trip of world.activeTrips.values()) {
      const pos = tripPosition(world, trip);
      if (!pos) continue;
      if (pos.phase === 'pickup') {
        pickups.push(pos);
        continue;
      }
      ctx.save();
      ctx.translate(pos.x, pos.y);
      ctx.rotate(pos.angle);
      ctx.beginPath();
      ctx.roundRect(-4, -1.5, 8, 3, 1.5);
      ctx.fill();
      ctx.restore();
    }
    ctx.fillStyle = C.pickup;
    for (const pos of pickups) {
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const drawRain = (world, now) => {
    const dt = lastNow == null ? 0 : Math.min(now - lastNow, 0.1);
    lastNow = now;
    rainLevel += ((world.isRain ? 1 : 0) - rainLevel) * Math.min(dt * 1.5, 1);
    if (rainLevel < 0.02) return;
    ctx.strokeStyle = `${C.rain}${(0.25 * rainLevel).toFixed(3)})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const count = Math.round(drops.length * rainLevel);
    for (let i = 0; i < count; i += 1) {
      const drop = drops[i];
      drop.y += drop.v * dt;
      drop.x += drop.v * dt * 0.18;
      if (drop.y > 1) {
        drop.y -= 1;
        drop.x = (drop.x + 0.37) % 1;
      }
      if (drop.x > 1) drop.x -= 1;
      const x = drop.x * view.w;
      const y = drop.y * view.h;
      ctx.moveTo(x, y);
      ctx.lineTo(x - 3, y - 14);
    }
    ctx.stroke();
  };

  const draw = (world, now) => {
    if (!canvas || !canvas.clientWidth) return;
    resize();
    ctx.clearRect(0, 0, view.w, view.h);
    drawSky(world);
    drawHalos(world);
    drawRoads();
    drawCars(world);
    drawRoadLabels();
    drawRain(world, now);
  };

  window.CityMap = {
    init(canvasEl, cityData) {
      canvas = canvasEl;
      ctx = canvas.getContext('2d');
      city = cityData;
      view = null;
    },
    draw,
    layout: () => view, // центры районов для DOM-плашек (hud.js)
  };
})();
