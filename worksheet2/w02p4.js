"use strict";
// Worksheet 2 – Part 4
// Two drawing modes:
//  - POINTS mode: clicking adds a 20×20 square (two triangles) at the click.
//  - TRIANGLE mode: first two clicks show points; third click replaces those two
//    points with ONE triangle whose three vertices (positions & colors) are the
//    three clicks; then the temporary record is cleared.
//  - CIRCLE mode: first click shows a point; second click defines the radius;
//    third click replaces the point with a circle of the defined radius.
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
  const btnCircle  = document.getElementById('btn-mode-circle');
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

  // Final shapes stored permanently in draw order
  const drawList = []; // items: { kind: 'square'|'triangle'|'circle', ... }

  // Temporary record used only in TRIANGLE mode (first two clicks)
  const tmpClicks = []; // [{x,y,rgb}]
  let circleTmp = null;

  // GPU buffers reused (track capacities manually; GPUBuffer.size is not reliable)
  let posBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  let colBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  let posCapacity = 4; // bytes
  let colCapacity = 4; // bytes

  function rebuildBuffers() {
    const pos = [];
    const col = [];

    // Permanent geometry in insertion order
    for (const item of drawList) {
      if (item.kind === 'square') {
        addColoredSquare(pos, col, item.cx, item.cy, SIDE, W, H, item.rgb);
      } else if (item.kind === 'triangle') {
        addColoredTriangle(pos, col, item.pos, item.col, W, H);
      } else if (item.kind === 'circle') {
        addColoredCircle(pos, col, item.cx, item.cy, item.radius, item.center, item.rim, 64, W, H);
      }
    }

    // Preview points (only in TRIANGLE mode for first 1-2 clicks)
    for (const p of tmpClicks) addColoredSquare(pos, col, p.x, p.y, SIDE, W, H, p.rgb);

    // Preview circle point (only one small square)
    if (circleTmp) addColoredSquare(pos, col, circleTmp.x, circleTmp.y, SIDE, W, H, circleTmp.rgb);

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
  const MODE = { POINTS: 'POINTS', TRIANGLE: 'TRIANGLE', CIRCLE: 'CIRCLE' };
  let mode = MODE.POINTS;
  updateModeUI();

  btnPoints.onclick = () => { mode = MODE.POINTS; tmpClicks.length = 0; circleTmp = null; updateModeUI(); render(); };
  btnTri.onclick    = () => { mode = MODE.TRIANGLE; tmpClicks.length = 0; circleTmp = null; updateModeUI(); render(); };
  btnCircle.onclick = () => { mode = MODE.CIRCLE; tmpClicks.length = 0; circleTmp = null; updateModeUI(); render(); };

  // Mouse clicks
  canvas.addEventListener('click', (ev) => {
    const rect = canvas.getBoundingClientRect();
    const cssX = ev.clientX - rect.left, cssY = ev.clientY - rect.top;
    const x = cssX * dpr, y = cssY * dpr; // device px
    const rgb = parseHexColor(ptPicker.value);

    if (mode === MODE.POINTS) {
      drawList.push({ kind: 'square', cx: x, cy: y, rgb });
      render();
      return;
    }

    if (mode === MODE.TRIANGLE) {
      tmpClicks.push({ x, y, rgb });
      if (tmpClicks.length < 3) { render(); return; }

      // Build a triangle from the three clicks and clear the temporary points
      const triPos = tmpClicks.map(p => [p.x, p.y]);
      const triCol = tmpClicks.map(p => p.rgb);
      drawList.push({ kind: 'triangle', pos: triPos, col: triCol });
      tmpClicks.length = 0;
      render();
      return;
    }

    if (mode === MODE.CIRCLE) {
      if (!circleTmp) {
        // First click: store center position AND its color
        circleTmp = { x, y, rgb }; // rgb is center color
        render();
        return;
      }
      // Second click: compute radius and use the *current* color as rim color
      const dx = x - circleTmp.x;
      const dy = y - circleTmp.y;
      const radius = Math.sqrt(dx*dx + dy*dy);
      const centerRGB = circleTmp.rgb; // from first click
      const rimRGB = rgb;              // from second click
      drawList.push({ kind: 'circle', cx: circleTmp.x, cy: circleTmp.y, radius, center: centerRGB, rim: rimRGB });
      circleTmp = null;
      render();
      return;
    }
  });

  // Clear button clears all shapes (points & triangles & circles)
  clearBtn.addEventListener('click', () => {
    drawList.length = 0; tmpClicks.length = 0; circleTmp = null; render();
  });

  function updateModeUI() {
    if (mode === MODE.POINTS) {
      modeLabel.textContent = 'Mode: Points';
      btnPoints.disabled = true;
      btnTri.disabled = false;
      btnCircle.disabled = false;
    } else if (mode === MODE.TRIANGLE) {
      modeLabel.textContent = 'Mode: Triangle';
      btnPoints.disabled = false;
      btnTri.disabled = true;
      btnCircle.disabled = false;
    } else if (mode === MODE.CIRCLE) {
      modeLabel.textContent = 'Mode: Circle';
      btnPoints.disabled = false;
      btnTri.disabled = false;
      btnCircle.disabled = true;
    }
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

function addColoredCircle(posOut, colOut, cx, cy, radius, centerRGB, rimRGB, slices, W, H) {
  // Build as triangle-list: (center, rim[i], rim[i+1]) for i in [0..slices-1]
  const toNDC = (x, y) => [ (x / W) * 2 - 1, 1 - (y / H) * 2 ];
  const [cxn, cyn] = toNDC(cx, cy);

  // Precompute rim points in NDC (closed loop)
  const rim = [];
  for (let i = 0; i <= slices; i++) {
    const a = (i / slices) * Math.PI * 2;
    const x = cx + radius * Math.cos(a);
    const y = cy + radius * Math.sin(a);
    rim.push(toNDC(x, y));
  }

  for (let i = 0; i < slices; i++) {
    // center (red)
    posOut.push(cxn, cyn);
    colOut.push(centerRGB[0], centerRGB[1], centerRGB[2]);
    // rim[i] (point color)
    posOut.push(rim[i][0], rim[i][1]);
    colOut.push(rimRGB[0], rimRGB[1], rimRGB[2]);
    // rim[i+1] (point color)
    posOut.push(rim[i+1][0], rim[i+1][1]);
    colOut.push(rimRGB[0], rimRGB[1], rimRGB[2]);
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
  ensure('btn-mode-circle', () => { const b=document.createElement('button'); b.id='btn-mode-circle'; b.textContent='Circle Mode'; return b; });
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
