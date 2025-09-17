"use strict";
// Worksheet 2 – Part 3
// Two drawing modes:
//  - POINTS mode: clicking adds a 20×20 square (two triangles) at the click.
//  - TRIANGLE mode: first two clicks show points; third click replaces those two
//    points with ONE triangle whose three vertices (positions & colors) are the
//    three clicks; then the temporary record is cleared.
// All geometry is rendered as triangle-list with per-vertex colors (float3).

// NOTE: If you click many times in Points mode the buffers grow. We now track
// buffer capacities and resize explicitly; relying on `GPUBuffer.size` is not
// supported in all runtimes and caused writes to overflow, blanking the canvas.

window.onload = () => { main().catch(err => showError(String(err))); };

async function main() {
  if (!('gpu' in navigator) || !navigator.gpu) {
    showError(
      "WebGPU is not available (navigator.gpu is undefined).\n" +
      "Use a recent Chrome/Edge on http://localhost and not a file:// URL."
    );
    return;
  }

  const canvas = document.getElementById('my-canvas');
  if (!canvas) { showError("No <canvas id='my-canvas'> found."); return; }

  // Controls (reuse Part 2 bar + add mode buttons)
  ensureControls(canvas);
  const clearBtn = document.getElementById('btn-clear');
  const bgPicker = document.getElementById('bg-color');
  const ptPicker = document.getElementById('pt-color');
  const btnPoints  = document.getElementById('btn-mode-points');
  const btnTri     = document.getElementById('btn-mode-triangle');
  const modeLabel  = document.getElementById('mode-label');

  // Canvas sizing (CSS → device px)
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth || 512;
  const cssH = canvas.clientHeight || 512;
  const W = Math.floor(cssW * dpr);
  const H = Math.floor(cssH * dpr);
  canvas.width = W; canvas.height = H;

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { showError("Failed to acquire a GPU adapter."); return; }
  const device = await adapter.requestDevice();

  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  // Shaders: position + color
  const shaderModule = device.createShaderModule({
    code: /* wgsl */`
      struct VSOut { @builtin(position) pos: vec4f, @location(0) color: vec3f };
      @vertex fn vs_main(@location(0) p: vec2f, @location(1) c: vec3f) -> VSOut {
        return VSOut(vec4f(p,0,1), c);
      }
      @fragment fn fs_main(@location(0) c: vec3f) -> @location(0) vec4f { return vec4f(c,1); }
    `
  });

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
      buffers: [
        { arrayStride: 8,  attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] },
        { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] },
      ]
    },
    fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' }
  });

  // Geometry state
  const sideCSS = 20;                 // point size in CSS px
  const SIDE = sideCSS * dpr;         // device px

  // Final shapes stored permanently
  const points = [];   // each: {cx, cy, rgb:[r,g,b]}
  const tris   = [];   // each: {pos:[[x,y],[x,y],[x,y]], col:[[r,g,b],[r,g,b],[r,g,b]]} in device px & [0..1]

  // Temporary record used only in TRIANGLE mode (first two clicks)
  const tmpClicks = []; // [{x,y,rgb}]

  // GPU buffers reused (track capacities manually; GPUBuffer.size is not reliable)
  let posBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  let colBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  let posCapacity = 4; // bytes
  let colCapacity = 4; // bytes

  function rebuildBuffers() {
    const pos = [];
    const col = [];

    // Permanent points (as small quads)
    for (const p of points) addColoredSquare(pos, col, p.cx, p.cy, SIDE, W, H, p.rgb);

    // Permanent triangles
    for (const t of tris) addColoredTriangle(pos, col, t.pos, t.col, W, H);

    // Preview points (only in TRIANGLE mode for first 1-2 clicks)
    for (const p of tmpClicks) addColoredSquare(pos, col, p.x, p.y, SIDE, W, H, p.rgb);

    const posData = new Float32Array(pos);
    const colData = new Float32Array(col);

    // (Re)create buffers if needed
    if (posCapacity < posData.byteLength) {
      posBuf.destroy?.();
      posCapacity = Math.max(posData.byteLength, posCapacity * 2);
      posBuf = device.createBuffer({ size: posCapacity, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    }
    if (colCapacity < colData.byteLength) {
      colBuf.destroy?.();
      colCapacity = Math.max(colData.byteLength, colCapacity * 2);
      colBuf = device.createBuffer({ size: colCapacity, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    }

    // Write data (only the used portion)
    device.queue.writeBuffer(posBuf, 0, posData.buffer, posData.byteOffset, posData.byteLength);
    device.queue.writeBuffer(colBuf, 0, colData.buffer, colData.byteOffset, colData.byteLength);
    return posData.length / 2; // vertex count
  }

  function parseHexColor(hex) {
    const v = hex.startsWith('#') ? hex.slice(1) : hex;
    return [parseInt(v.slice(0,2),16)/255, parseInt(v.slice(2,4),16)/255, parseInt(v.slice(4,6),16)/255];
}

  function render() {
    const [br, bg, bb] = parseHexColor(bgPicker.value);

    // Build CPU-side arrays and write GPU buffers before we begin the pass.
    const vtxCount = rebuildBuffers();

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
    colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        clearValue: { r: br, g: bg, b: bb, a: 1 },
        storeOp: 'store'
    }]
    });

    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, posBuf);
    pass.setVertexBuffer(1, colBuf);
    if (vtxCount > 0) pass.draw(vtxCount);
    pass.end();
    device.queue.submit([encoder.finish()]);
   }

  // Initial draw (empty scene)
  render();

  // Modes
  const MODE = { POINTS: 'POINTS', TRIANGLE: 'TRIANGLE' };
  let mode = MODE.POINTS;
  updateModeUI();

  btnPoints.onclick = () => { mode = MODE.POINTS; tmpClicks.length = 0; updateModeUI(); render(); };
  btnTri.onclick    = () => { mode = MODE.TRIANGLE; tmpClicks.length = 0; updateModeUI(); render(); };

  // Mouse clicks
  canvas.addEventListener('click', (ev) => {
    const rect = canvas.getBoundingClientRect();
    const cssX = ev.clientX - rect.left, cssY = ev.clientY - rect.top;
    const x = cssX * dpr, y = cssY * dpr; // device px
    const rgb = parseHexColor(ptPicker.value);

    if (mode === MODE.POINTS) {
      points.push({ cx: x, cy: y, rgb });
      render();
      return;
    }

    // TRIANGLE mode
    tmpClicks.push({ x, y, rgb });
    if (tmpClicks.length < 3) { render(); return; }

    // Build a triangle from the three clicks and clear the temporary points
    const triPos = tmpClicks.map(p => [p.x, p.y]);
    const triCol = tmpClicks.map(p => p.rgb);
    tris.push({ pos: triPos, col: triCol });
    tmpClicks.length = 0;
    render();
  });

  // Clear button clears all shapes (points & triangles)
  clearBtn.addEventListener('click', () => {
    points.length = 0; tris.length = 0; tmpClicks.length = 0; render();
  });

  function updateModeUI() {
    modeLabel.textContent = mode === MODE.POINTS ? 'Mode: Points' : 'Mode: Triangle';
    btnPoints.disabled  = mode === MODE.POINTS;
    btnTri.disabled     = mode === MODE.TRIANGLE;
  }
}

// --- Geometry helpers -------------------------------------------------------
function addColoredSquare(posOut, colOut, cx, cy, size, W, H, rgb) {
  const h = size / 2;
  const x0 = cx - h, x1 = cx + h;
  const y0 = cy - h, y1 = cy + h;
  const toNDC = (x, y) => [ (x / W) * 2 - 1, 1 - (y / H) * 2 ];
  const [x0n, y0n] = toNDC(x0, y0);
  const [x1n, y1n] = toNDC(x1, y1);
  const quad = [ x0n,y0n,  x1n,y0n,  x1n,y1n,   x0n,y0n,  x1n,y1n,  x0n,y1n ];
  for (let i = 0; i < quad.length; i += 2) { posOut.push(quad[i], quad[i+1]); colOut.push(rgb[0], rgb[1], rgb[2]); }
}

function addColoredTriangle(posOut, colOut, threePosDevPx, threeCol, W, H) {
  const toNDC = (x, y) => [ (x / W) * 2 - 1, 1 - (y / H) * 2 ];
  for (let i = 0; i < 3; i++) {
    const [x, y] = threePosDevPx[i];
    const [r, g, b] = threeCol[i];
    const [xn, yn] = toNDC(x, y);
    posOut.push(xn, yn);
    colOut.push(r, g, b);
  }
}

function ensureControls(canvas) {
  // If controls already exist, just make sure the new elements are present
  let bar = document.getElementById('controls-bar');
  if (!bar) {
    bar = document.createElement('div');
    bar.id = 'controls-bar';
    bar.style.margin = '8px auto';
    bar.style.maxWidth = canvas.clientWidth ? canvas.clientWidth + 'px' : '512px';
    bar.style.display = 'flex';
    bar.style.flexWrap = 'wrap';
    bar.style.gap = '12px';
    bar.style.alignItems = 'center';
    canvas.parentNode.insertBefore(bar, canvas);
  }

  const ensure = (id, factory) => { let el = document.getElementById(id); if (!el) { el = factory(); bar.appendChild(el); } return el; };

  ensure('btn-clear', () => { const b = document.createElement('button'); b.id='btn-clear'; b.textContent='Clear'; return b; });
  ensure('bg-color',  () => { const i = document.createElement('input'); i.type='color'; i.id='bg-color'; i.value='#6495ed'; return i; });
  ensure('pt-color',  () => { const i = document.createElement('input'); i.type='color'; i.id='pt-color'; i.value='#000000'; return i; });
  ensure('btn-mode-points', () => { const b=document.createElement('button'); b.id='btn-mode-points'; b.textContent='Points Mode'; return b; });
  ensure('btn-mode-triangle', () => { const b=document.createElement('button'); b.id='btn-mode-triangle'; b.textContent='Triangle Mode'; return b; });
  ensure('mode-label', () => { const s=document.createElement('span'); s.id='mode-label'; s.textContent='Mode: Points'; s.style.marginLeft='8px'; return s; });
}

function showError(msg) {
  const pre = document.createElement('pre');
  pre.textContent = msg;
  pre.style.position = 'fixed'; pre.style.top = '8px'; pre.style.left = '8px';
  pre.style.padding = '8px'; pre.style.background = 'rgba(0,0,0,0.75)'; pre.style.color = '#fff';
  pre.style.zIndex = 9999; pre.style.whiteSpace = 'pre-wrap';
  document.body.appendChild(pre);
  console.error(msg);
}
