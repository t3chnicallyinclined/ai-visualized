/* A real Linux machine inside the page, for the agent to work on.
   v86 (https://github.com/copy/v86, BSD-2-Clause) is a PC emulator compiled to WebAssembly: it boots a small Buildroot
   Linux (a kernel and BusyBox) in this tab. The agent's commands are typed into its serial console exactly as written,
   and what Linux prints comes back. It has no network, and it can only reach its own memory: a reset puts it back.
   Usage: const pc = await MACHINE.boot({ onText, onStatus, files }); const out = await pc.run('ls'); await pc.reset();
   onText gets the terminal's characters ('\f' means: clear the terminal). Works in the browser (window.MACHINE). */
(function (root) {
  'use strict';
  const V86 = '0.5.471', GH = '94faf3d6b19ecabc53a4010aaa24252fa6a162e2'; // npm version and the matching repository commit (2026-10-09)
  const URLS = {
    lib: `https://cdn.jsdelivr.net/npm/v86@${V86}/build/libv86.js`,
    wasm: `https://cdn.jsdelivr.net/npm/v86@${V86}/build/v86.wasm`,
    bios: `https://cdn.jsdelivr.net/gh/copy/v86@${GH}/bios/seabios.bin`, // SeaBIOS (LGPL), from the v86 repository
    vga: `https://cdn.jsdelivr.net/gh/copy/v86@${GH}/bios/vgabios.bin`,
    linux: 'linux/bzimage.bin', // the Buildroot kernel and BusyBox (see linux/README.md)
  };
  const PROMPT = '~% '; // BusyBox's shell prompt in this image
  const loadScript = src => new Promise((ok, no) => { if (root.V86) return ok(); const s = document.createElement('script'); s.src = src; s.onload = ok; s.onerror = () => no(new Error('could not load ' + src)); document.head.appendChild(s); });
  const sh = t => "'" + String(t).replace(/'/g, "'\\''") + "'"; // a shell-quoted string
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  async function boot({ onText = () => {}, onStatus = () => {}, files = {}, urls = {} } = {}) { // files: { name: text } for the home folder
    const U = Object.assign({}, URLS, urls);
    onStatus('Downloading the emulator…');
    await loadScript(U.lib);
    let buf = '', waiters = [], quiet = false; // quiet: the page's own housekeeping stays out of the terminal
    const emu = new root.V86({
      wasm_path: U.wasm, memory_size: 64 << 20, vga_memory_size: 2 << 20,
      bios: { url: U.bios }, vga_bios: { url: U.vga }, bzimage: { url: U.linux, async: false }, filesystem: {},
      cmdline: 'tsc=reliable mitigations=off random.trust_cpu=on', autostart: true,
      disable_keyboard: true, disable_mouse: true, disable_speaker: true,
    });
    emu.add_listener('download-progress', e => { if (e.file_name && /bzimage/.test(e.file_name) && e.total) onStatus(`Downloading Linux: ${Math.round(e.loaded / 1e6)} of ${Math.round(e.total / 1e6)} MB`); });
    emu.add_listener('emulator-started', () => onStatus('Linux is booting…'));
    emu.add_listener('serial0-output-byte', b => {
      const ch = String.fromCharCode(b); if (ch === '\r') return;
      buf += ch; if (!quiet) onText(ch);
      waiters = waiters.filter(w => { if (buf.endsWith(w.until)) { w.ok(buf); return false; } return true; });
    });
    const waitFor = (until, ms) => new Promise((ok, no) => { const w = { until, ok }; waiters.push(w); if (ms) setTimeout(() => { if (waiters.includes(w)) { waiters = waiters.filter(x => x !== w); no(new Error('timeout')); } }, ms); });
    await waitFor(PROMPT, 120000);
    // set up the home folder: the example files, no colours; then a clean terminal
    const setup = 'rm -rf ~/*; ' + Object.entries(files).map(([f, t]) => `printf '%s' ${sh(t)} > ${f}`).join('; ') + '; alias ls="ls --color=never"\n';
    buf = ''; quiet = true; emu.serial0_send(setup); await waitFor(PROMPT, 20000); quiet = false;
    onText('\f' + PROMPT); onStatus('ready');
    const fresh = await emu.save_state(); // the machine just after setup: a reset puts this back (instant; nothing survives)
    let queue = Promise.resolve();
    const pc = {
      emu, files: Object.keys(files),
      // run one command exactly as written; resolves with what it printed (no echo, no prompt).
      // typeMs: type it one character at a time, as a person would; quiet: keep it out of the terminal
      run(cmd, { typeMs = 0, quiet: q = false } = {}) {
        const job = queue.then(async () => {
          const line = String(cmd).replace(/\n/g, ' ');
          buf = ''; quiet = q;
          if (typeMs) { for (const ch of line) { emu.serial0_send(ch); await sleep(typeMs); } emu.serial0_send('\n'); } else emu.serial0_send(line + '\n');
          let out;
          try { out = await waitFor(PROMPT, 15000); }
          catch (e) { emu.serial0_send('\x03'); out = (await waitFor(PROMPT, 5000).catch(() => buf)) + '\n(stopped after 15 seconds)'; }
          quiet = false;
          const lines = out.split('\n'); lines.shift(); // the echoed command
          return lines.join('\n').replace(/~% $/, '').replace(/\x1b\[[0-9;]*[A-Za-z]/g, '').replace(/\n$/, '');
        });
        queue = job.catch(() => {}); return job;
      },
      async ls() { const o = await pc.run('ls -1', { quiet: true }); return o ? o.split('\n') : []; },
      async reset() {
        onStatus('Restarting…'); await queue; await emu.restore_state(fresh);
        buf = ''; quiet = true; emu.serial0_send('\n'); await waitFor(PROMPT, 10000); quiet = false;
        onText('\f' + PROMPT); onStatus('ready');
      },
    };
    return pc;
  }
  root.MACHINE = { boot, URLS };
})(typeof window !== 'undefined' ? window : globalThis);
