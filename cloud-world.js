// The cloud as a real 3D world: every piece the model knows is a point at its spot in a 3-D t-SNE of the
// model's own embedding rows (prep/cloud_layout3d.py, the cloud3d-*.js files), drawn with WebGL under a true
// perspective camera. Drag to turn, scroll or pinch to fly in (and on through the cloud), right-drag or two
// fingers to slide, W A S D / Q E to fly, arrow keys to turn. It turns slowly on its own until you touch it.
// The page hands it the same things it gives the flat map: the chances (lit points), the labels and the path.
(function (root) {
  'use strict';
  const FILES = { 'tiny-stories': 'cloud3d-tiny.js', 'pythia-14m': 'cloud3d-pythia.js', distilgpt2: 'cloud3d-gpt2.js', 'smollm2-instruct': 'cloud3d-smollm2.js' };
  const loading = {};
  function load(key) {
    if (root.CLOUDS3D && root.CLOUDS3D[key]) return Promise.resolve(true);
    if (!FILES[key]) return Promise.resolve(false);
    return loading[key] || (loading[key] = new Promise(ok => { const s = document.createElement('script'); s.src = FILES[key]; s.onload = () => ok(!!(root.CLOUDS3D && root.CLOUDS3D[key])); s.onerror = () => ok(false); document.head.appendChild(s); }));
  }
  const pointsCache = {};
  function points(key, onMap, colorOf) { // positions centred on 0 (the cube is 1 wide); pieces not on the map are parked far away
    if (pointsCache[key]) return pointsCache[key];
    const unb64 = t => Uint8Array.from(atob(t || ''), ch => ch.charCodeAt(0));
    const raw = new Uint16Array(unb64(root.CLOUDS3D[key].xyz).buffer), n = raw.length / 3, pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const off = raw[3 * i] === 65535 || !onMap(i);
      for (let k = 0; k < 3; k++) pos[3 * i + k] = off ? 99 : raw[3 * i + k] / 65534 - 0.5;
      const c = off ? [0, 0, 0] : colorOf(i); col[3 * i] = c[0]; col[3 * i + 1] = c[1]; col[3 * i + 2] = c[2];
    }
    return (pointsCache[key] = { n, pos, col, meta: root.CLOUDS3D[key].meta });
  }

  // ---- small matrix kit (column-major 4x4, as WebGL wants) ----
  function perspective(fovy, aspect, near, far) { const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far); return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]); }
  function lookAt(e, t, up) {
    let zx = e[0] - t[0], zy = e[1] - t[1], zz = e[2] - t[2], l = Math.hypot(zx, zy, zz) || 1; zx /= l; zy /= l; zz /= l;
    let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx; l = Math.hypot(xx, xy, xz) || 1; xx /= l; xy /= l; xz /= l;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0, -(xx * e[0] + xy * e[1] + xz * e[2]), -(yx * e[0] + yy * e[1] + yz * e[2]), -(zx * e[0] + zy * e[1] + zz * e[2]), 1]);
  }
  function mul(a, b) { const o = new Float32Array(16); for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s; } return o; }

  const VS = `attribute vec3 p; attribute vec3 col; attribute float lit; uniform mat4 VP; uniform float px, dpr; varying vec4 vc;
    void main() {
      vec4 c = VP * vec4(p, 1.0); gl_Position = c; float w = max(c.w, 1e-4);
      gl_PointSize = clamp((0.0032 + 0.014 * lit * lit) * px / w, 1.1 * dpr, 48.0 * dpr);
      float fog = clamp(1.25 - w * 0.42, 0.22, 1.0), nearFade = smoothstep(0.004, 0.06, w);
      float a = (0.42 + 0.58 * lit) * fog * nearFade;
      vc = vec4(mix(col, vec3(1.0, 0.95, 0.75), lit * 0.7) * a, a);
      if (p.x > 50.0 || c.w <= 0.0) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    }`;
  const FS = `precision mediump float; varying vec4 vc; void main() { vec2 d = gl_PointCoord - 0.5; float r = dot(d, d) * 4.0; if (r > 1.0) discard; gl_FragColor = vc * (1.0 - r * 0.65); }`;

  const START = { yaw: 0.6, pitch: 0.32, dist: 1.55 };
  const reduce = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

  // host: a positioned element that holds the flat map's canvas. opts: key(), onMap(id), colorOf(id) -> [r,g,b] 0..1,
  // piece(id), neighbors(id, k) -> [{id, sim}], chanceOf(q) -> 0..1, tip(id, clientX, clientY, chance) (id -1 hides it)
  function create(host, opts) {
    const gcv = document.createElement('canvas'); gcv.className = 'cw-gl'; gcv.setAttribute('aria-label', 'The cloud in 3D: drag to turn, scroll or pinch to fly in, right-drag or two fingers to slide');
    const ov = document.createElement('canvas'); ov.className = 'cw-ov'; ov.setAttribute('aria-hidden', 'true');
    const hint = document.createElement('div'); hint.className = 'cw-hint';
    hint.textContent = typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches ? 'Drag to turn · pinch to fly in · two fingers to slide · tap a point'
      : 'Drag to turn · scroll to fly in · right-drag to slide · W A S D to fly · click a point';
    const home = document.createElement('button'); home.type = 'button'; home.className = 'cw-home'; home.textContent = '⟲'; home.title = 'Back to the start, turning on its own'; home.setAttribute('aria-label', home.title);
    const W = { on: false, key: null, gl: null, prog: null, u: null, pts: null, litBuf: null, lit: null, state: { watch: null, labels: null, path: null, sel: -1 }, litFor: undefined,
      yaw: START.yaw, pitch: START.pitch, dist: START.dist, t: [0, 0, 0], spin: !reduce, keys: new Set(), raf: 0, dirty: true, VP: null, last: 0 };

    function size() { const r = gcv.getBoundingClientRect(), dpr = Math.min(2, root.devicePixelRatio || 1), w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr)); if (gcv.width !== w || gcv.height !== h) { gcv.width = ov.width = w; gcv.height = ov.height = h; } return dpr; }
    function eye() { const cp = Math.cos(W.pitch); return [W.t[0] + W.dist * cp * Math.sin(W.yaw), W.t[1] + W.dist * Math.sin(W.pitch), W.t[2] + W.dist * cp * Math.cos(W.yaw)]; }
    function basis() { // forward (into the screen), right and up, for flying and sliding
      const e = eye(), f = [W.t[0] - e[0], W.t[1] - e[1], W.t[2] - e[2]], l = Math.hypot(...f) || 1; f[0] /= l; f[1] /= l; f[2] /= l;
      const r = [-f[2], 0, f[0]], rl = Math.hypot(...r) || 1; r[0] /= rl; r[2] /= rl;
      const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]; return { f, r, u };
    }
    const FOV = 1.05;
    function initGL() {
      const gl = gcv.getContext('webgl', { alpha: true, antialias: false, premultipliedAlpha: true }); if (!gl) return false; W.gl = gl;
      const sh = (t, src) => { const o = gl.createShader(t); gl.shaderSource(o, src); gl.compileShader(o); if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(o)); return o; };
      const prog = gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog); gl.useProgram(prog); W.prog = prog;
      W.u = { VP: gl.getUniformLocation(prog, 'VP'), px: gl.getUniformLocation(prog, 'px'), dpr: gl.getUniformLocation(prog, 'dpr') };
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); return true;
    }
    function upload() {
      const gl = W.gl, P = W.pts;
      const buf = (name, arr, n) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW); const loc = gl.getAttribLocation(W.prog, name); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, n, gl.FLOAT, false, 0, 0); return b; };
      buf('p', P.pos, 3); buf('col', P.col, 3); W.lit = new Float32Array(P.n); W.litBuf = buf('lit', W.lit, 1); W.litFor = undefined;
    }
    function project(i) { // a point's place on the overlay, or null when it is behind us or off screen
      const P = W.pts, x = P.pos[3 * i], y = P.pos[3 * i + 1], z = P.pos[3 * i + 2]; if (x > 50) return null; const m = W.VP;
      const cx = m[0] * x + m[4] * y + m[8] * z + m[12], cy = m[1] * x + m[5] * y + m[9] * z + m[13], cw = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (cw <= 0.01) return null; return [(cx / cw * 0.5 + 0.5) * ov.width, (0.5 - cy / cw * 0.5) * ov.height, cw];
    }
    function frame(now) {
      W.raf = 0; if (!W.on || !W.pts) return;
      const dt = Math.min(0.05, W.last ? (now - W.last) / 1000 : 0.016); W.last = now;
      let moving = false;
      if (W.spin) { W.yaw += dt * 0.12; moving = true; }
      if (W.keys.size) { const { f, r, u } = basis(), sp = dt * Math.max(0.12, Math.min(0.8, W.dist * 0.9)); const k = W.keys; const mv = (v, s) => { W.t[0] += v[0] * s; W.t[1] += v[1] * s; W.t[2] += v[2] * s; };
        if (k.has('w')) mv(f, sp); if (k.has('s')) mv(f, -sp); if (k.has('d')) mv(r, sp); if (k.has('a')) mv(r, -sp); if (k.has('e')) mv(u, sp); if (k.has('q')) mv(u, -sp);
        if (k.has('arrowleft')) W.yaw -= dt * 1.2; if (k.has('arrowright')) W.yaw += dt * 1.2; if (k.has('arrowup')) W.pitch = Math.min(1.5, W.pitch + dt); if (k.has('arrowdown')) W.pitch = Math.max(-1.5, W.pitch - dt);
        moving = true; }
      draw(); if (moving) W.raf = requestAnimationFrame(frame); else W.last = 0;
    }
    function kick() { if (!W.raf && W.on) W.raf = requestAnimationFrame(frame); }
    function draw() {
      const gl = W.gl, dpr = size(), P = W.pts, s = W.state;
      if (s.watch !== W.litFor) { const lit = W.lit; if (s.watch) for (let i = 0; i < P.n; i++) { const q = s.watch[i] || 0; lit[i] = q >= 60 ? Math.min(1, Math.sqrt((q - 60) / 195)) : 0; } else lit.fill(0); gl.bindBuffer(gl.ARRAY_BUFFER, W.litBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, lit); W.litFor = s.watch; }
      W.VP = mul(perspective(FOV, gcv.width / gcv.height, 0.003, 30), lookAt(eye(), W.t, [0, 1, 0]));
      gl.viewport(0, 0, gcv.width, gcv.height); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.useProgram(W.prog);
      gl.uniformMatrix4fv(W.u.VP, false, W.VP); gl.uniform1f(W.u.px, gcv.height / 2 / Math.tan(FOV / 2)); gl.uniform1f(W.u.dpr, dpr);
      gl.drawArrays(gl.POINTS, 0, P.n);
      overlay(dpr);
    }
    function overlay(dpr) { // the path of best guesses, the tapped point with its neighbours, and the labels, over the same camera
      const c = ov.getContext('2d'), s = W.state; c.clearRect(0, 0, ov.width, ov.height); c.textBaseline = 'middle';
      if (s.path && s.path.length > 1) { c.strokeStyle = 'rgba(255,233,184,.75)'; c.lineWidth = 1.5 * dpr; c.beginPath(); let started = false;
        for (const id of s.path) { const p = id < W.pts.n ? project(id) : null; if (!p) { started = false; continue; } started ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1]); started = true; } c.stroke(); }
      const items = (s.labels || []).map(([id, txt], i) => ({ id, txt, first: i === 0 }));
      if (s.sel >= 0 && s.sel < W.pts.n) { const sp = project(s.sel);
        if (sp) {
          for (const nb of opts.neighbors(s.sel, 4)) { const q = project(nb.id); if (!q) continue; c.strokeStyle = 'rgba(255,233,184,.5)'; c.lineWidth = dpr; c.setLineDash([3 * dpr, 4 * dpr]); c.beginPath(); c.moveTo(sp[0], sp[1]); c.lineTo(q[0], q[1]); c.stroke(); c.setLineDash([]); items.push({ id: nb.id, txt: `${opts.piece(nb.id)} ${nb.sim.toFixed(2)}`, nb: true }); }
          c.strokeStyle = '#ffe9b8'; c.lineWidth = 2 * dpr; c.beginPath(); c.arc(sp[0], sp[1], 8 * dpr, 0, 7); c.stroke(); items.unshift({ id: s.sel, txt: opts.piece(s.sel), first: true }); } }
      c.font = `600 ${12 * dpr}px "IBM Plex Mono", monospace`; const placed = [];
      items.forEach(({ id, txt, first, nb }, i) => { // labels never overlap: each moves down until it has room, with a thin line back to its dot
        const p = id < W.pts.n ? project(id) : null; if (!p || p[0] < 0 || p[1] < 0 || p[0] > ov.width || p[1] > ov.height) return;
        c.fillStyle = nb ? '#ffd58a' : first ? '#ffe9b8' : 'rgba(255,255,255,.9)'; c.beginPath(); c.arc(p[0], p[1], (first ? 4.5 : 3) * dpr, 0, 7); c.fill();
        const tw = c.measureText(txt).width, h = 18 * dpr, tx = Math.min(ov.width - tw - 6 * dpr, p[0] + 10 * dpr); let ty = p[1];
        while (placed.some(r => tx < r.x + r.w && tx + tw + 6 * dpr > r.x && Math.abs(ty - r.y) < h + 2 * dpr)) ty += h + 2 * dpr;
        placed.push({ x: tx - 3 * dpr, y: ty, w: tw + 6 * dpr });
        if (ty !== p[1]) { c.strokeStyle = 'rgba(255,255,255,.45)'; c.lineWidth = dpr; c.beginPath(); c.moveTo(p[0], p[1]); c.lineTo(tx - 3 * dpr, ty); c.stroke(); }
        c.fillStyle = 'rgba(7,8,13,.8)'; c.fillRect(tx - 3 * dpr, ty - h / 2, tw + 6 * dpr, h); c.fillStyle = nb ? '#ffd58a' : first ? '#ffe9b8' : '#e7e9f3'; c.fillText(txt, tx, ty);
      });
    }
    function nearest(px, py, maxPx) { let best = -1, bd = maxPx * maxPx, bw = Infinity; for (let i = 0; i < W.pts.n; i++) { const p = project(i); if (!p) continue; const d = (p[0] - px) ** 2 + (p[1] - py) ** 2; if (d < bd || (d === bd && p[2] < bw)) { bd = d; best = i; bw = p[2]; } } return best; }

    // ---- moving around ----
    const PT = new Map(); let drag = null, pinch0 = 0, dist0 = 0;
    const local = e => { const r = gcv.getBoundingClientRect(); return [(e.clientX - r.left) * gcv.width / r.width, (e.clientY - r.top) * gcv.height / r.height]; };
    const touched = () => { W.spin = false; home.hidden = false; };
    function zoomBy(f) { // in: the camera closes on its target; once close, it keeps flying forward through the cloud
      if (f < 1 && W.dist * f < 0.06) { const { f: fw } = basis(), step = W.dist * (1 - f) * 1.4 + 0.004; W.t[0] += fw[0] * step; W.t[1] += fw[1] * step; W.t[2] += fw[2] * step; }
      else W.dist = Math.max(0.06, Math.min(4, W.dist * f));
    }
    function slide(dx, dy) { const { r, u } = basis(), k = W.dist * Math.tan(FOV / 2) * 2 / gcv.height; W.t[0] += (-dx * r[0] + dy * u[0]) * k; W.t[1] += (-dx * r[1] + dy * u[1]) * k; W.t[2] += (-dx * r[2] + dy * u[2]) * k; }
    gcv.addEventListener('contextmenu', e => e.preventDefault());
    gcv.addEventListener('pointerdown', e => { gcv.setPointerCapture(e.pointerId); const [x, y] = local(e); PT.set(e.pointerId, [x, y]);
      drag = { x, y, yaw: W.yaw, pitch: W.pitch, t: W.t.slice(), slide: e.button === 2 || e.shiftKey, moved: false, lx: x, ly: y };
      if (PT.size === 2) { const [a, b] = [...PT.values()]; pinch0 = Math.hypot(a[0] - b[0], a[1] - b[1]); dist0 = W.dist; drag.mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]; drag.moved = true; } });
    gcv.addEventListener('pointermove', e => { if (!drag || !PT.has(e.pointerId)) return; const [x, y] = local(e); PT.set(e.pointerId, [x, y]);
      if (PT.size === 2) { const [a, b] = [...PT.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]), mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        if (pinch0) { const want = dist0 * pinch0 / Math.max(1, d); zoomBy(want / W.dist); dist0 = W.dist; pinch0 = d; }
        slide(mid[0] - drag.mid[0], mid[1] - drag.mid[1]); drag.mid = mid; touched(); kick(); if (!W.spin) draw(); return; }
      const dx = x - drag.x, dy = y - drag.y; if (!drag.moved && Math.hypot(dx, dy) < 6 * (gcv.width / gcv.clientWidth)) return; drag.moved = true; touched();
      if (drag.slide) { slide(x - drag.lx, y - drag.ly); drag.lx = x; drag.ly = y; }
      else { const k = 3.2 / gcv.height; W.yaw = drag.yaw - dx * k; W.pitch = Math.max(-1.5, Math.min(1.5, drag.pitch + dy * k)); }
      draw(); });
    const up = e => { PT.delete(e.pointerId); if (!drag) return; const tapped = !drag.moved; if (!PT.size) drag = null; if (PT.size || !tapped) return;
      const [x, y] = local(e), id = nearest(x, y, 16 * (gcv.width / gcv.clientWidth)); W.state.sel = id; draw();
      opts.tip(id, e.clientX, e.clientY, id >= 0 && W.state.watch && W.state.watch[id] ? opts.chanceOf(W.state.watch[id]) : null); };
    gcv.addEventListener('pointerup', up); gcv.addEventListener('pointercancel', e => { PT.delete(e.pointerId); if (!PT.size) drag = null; });
    gcv.addEventListener('wheel', e => { e.preventDefault(); touched(); zoomBy(Math.pow(1.0012, e.deltaY)); draw(); }, { passive: false });
    gcv.addEventListener('dblclick', e => { e.preventDefault(); reset(); });
    gcv.tabIndex = 0;
    const KEYS = new Set(['w', 'a', 's', 'd', 'q', 'e', 'arrowleft', 'arrowright', 'arrowup', 'arrowdown']);
    gcv.addEventListener('keydown', e => { const k = e.key.toLowerCase(); if (!KEYS.has(k)) return; e.preventDefault(); W.keys.add(k); touched(); kick(); });
    gcv.addEventListener('keyup', e => W.keys.delete(e.key.toLowerCase()));
    gcv.addEventListener('blur', () => W.keys.clear());
    function reset() { Object.assign(W, { yaw: START.yaw, pitch: START.pitch, dist: START.dist, spin: !reduce }); W.t = [0, 0, 0]; W.state.sel = -1; home.hidden = true; opts.tip(-1); kick(); draw(); }
    home.addEventListener('click', reset); home.hidden = true;
    const ro = typeof ResizeObserver === 'function' ? new ResizeObserver(() => { if (W.on && W.pts) draw(); }) : null;

    async function show(on) {
      W.on = on; host.classList.toggle('w3d', on);
      if (!on) { gcv.remove(); ov.remove(); hint.remove(); home.remove(); if (ro) ro.disconnect(); cancelAnimationFrame(W.raf); W.raf = 0; W.keys.clear(); opts.tip(-1); return true; }
      host.append(gcv, ov, home, hint); if (ro) ro.observe(gcv);
      const key = opts.key(); if (!(await load(key)) || !W.on) { if (W.on) { show(false); return false; } return false; }
      if (!W.gl && !initGL()) { show(false); return false; }
      if (W.key !== key) { W.pts = points(key, opts.onMap, opts.colorOf); W.key = key; upload(); }
      W.last = 0; draw(); kick(); return true;
    }
    function setState(watch, labels, path, sel) { W.state.watch = watch; W.state.labels = labels; W.state.path = path; if (sel !== undefined) W.state.sel = sel; if (W.on && W.pts) { if (W.key !== opts.key()) { show(true); return; } draw(); } }
    return { show, setState, get on() { return W.on; }, reset, focus: () => gcv.focus({ preventScroll: true }) };
  }
  root.CloudWorld = { create, load };
})(typeof window !== 'undefined' ? window : globalThis);
