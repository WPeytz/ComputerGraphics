"use strict";
// Worksheet 2 – Part 2
// Add UI: a Clear button, a background color picker, and a point color picker.
// Clicking on the canvas adds a 20×20 square (two triangles) in the chosen point color.
// Clear uses the selected background color. Shaders are updated to accept colors
// (like Worksheet 1 Part 3), using a second vertex buffer for float3 colors.

window.onload = () => { main().catch(err => showError(String(err))); };

async function main() {
  // 1) WebGPU feature detection
  if (!('gpu' in navigator) || !navigator.gpu) {
    showError(
      "WebGPU is not available (navigator.gpu is undefined).\n" +
      "Use a recent Chrome/Edge on http://localhost and not a file:// URL."
    );
    return;
  }

  const canvas = document.getElementById('my-canvas');
  if (!canvas) { showError("No <canvas id='my-canvas'> found."); return; }

  // Inject minimal UI (button + two color inputs) just above the canvas
  ensureControls(canvas);
  const clearBtn = document.getElementById('btn-clear');
  const bgPicker = document.getElementById('bg-color');
  const ptPicker = document.getElementById('pt-color');

  // Canvas sizing: respect CSS size but render HiDPI crisp
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth || 512;
  const cssH = canvas.clientHeight || 512;
  const W = Math.floor(cssW * dpr);
  const H = Math.floor(cssH * dpr);
  canvas.width = W;
  canvas.height = H;

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { showError("Failed to acquire a GPU adapter."); return; }
  const device = await adapter.requestDevice();

  // Configure swap chain
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  // --- WGSL shaders with color attribute ---
  const shaderModule = device.createShaderModule({
    label: 'Part2 shaders',
    code: /* wgsl */`
      struct VSOut {
        @builtin(position) pos : vec4f,
        @location(0) color : vec3f,
      };
      @vertex
      fn vs_main(@location(0) inPos : vec2f, @location(1) inCol : vec3f) -> VSOut {
        var out : VSOut;
        out.pos = vec4f(inPos, 0.0, 1.0);
        out.color = inCol;
        return out;
      }
      @fragment
      fn fs_main(@location(0) color : vec3f) -> @location(0) vec4f {
        return vec4f(color, 1.0);
      }
    `
  });

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
      buffers: [
        { // positions: float2
          arrayStride: 8,
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }]
        },
        { // colors: float3
          arrayStride: 12,
          attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }]
        }
      ]
    },
    fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' }
  });

  // --- Geometry state: up to 3 points (20×20 pixel squares) ---
  const sideCSS = 20;             // requested size in CSS pixels
  const S = sideCSS * dpr;        // convert to device pixels for math below
  const maxPoints = 3;            // keep behavior from Part 1
  let centers = [];// list of [x,y] in device px
  let colors = [];// list of [r,g,b] for each vertex (expanded later)
  let idx = 0;     // rotating index to replace oldest

  // GPU buffers reused across redraws
  let posBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  let colBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });

  function rebuildBuffers() {
    const pos = [];
    const col = [];
    for (let i = 0; i < centers.length; i++) {
      const [cx, cy] = centers[i];
      const rgb = colors[i] || [0,0,0];
      addColoredSquare(pos, col, cx, cy, S, W, H, rgb);
    }
    const posData = new Float32Array(pos);
    const colData = new Float32Array(col);
    if (posBuf.size < posData.byteLength) {
      posBuf.destroy?.();
      posBuf = device.createBuffer({ size: posData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    }
    if (colBuf.size < colData.byteLength) {
      colBuf.destroy?.();
      colBuf = device.createBuffer({ size: colData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    }
    device.queue.writeBuffer(posBuf, 0, posData.buffer, posData.byteOffset, posData.byteLength);
    device.queue.writeBuffer(colBuf, 0, colData.buffer, colData.byteOffset, colData.byteLength);
    return posData.length / 2; // vertex count
  }

  let vertexCount = 0;

  function parseHexColor(hex) {
    // hex like #RRGGBB
    const v = hex.startsWith('#') ? hex.slice(1) : hex;
    const r = parseInt(v.slice(0,2), 16) / 255;
    const g = parseInt(v.slice(2,4), 16) / 255;
    const b = parseInt(v.slice(4,6), 16) / 255;
    return [r,g,b];
  }

  function render() {
    const [br, bg, bb] = parseHexColor(bgPicker.value);
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
    pass.draw(vertexCount);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  // Initial state: draw three default points like before
  (function seedInitialPoints(){
    const defaults = [
      [cssW - sideCSS / 2, sideCSS / 2],
      [cssW - sideCSS / 2, cssH / 2],
      [cssW / 2,           cssH / 2]
    ];
    centers = defaults.map(([x,y]) => [x*dpr, y*dpr]);
    colors  = [ [0,0,0], [0,0,0], [0,0,0] ]; // black
    vertexCount = rebuildBuffers();
    render();
  })();

  // --- Mouse click handler ---
  canvas.addEventListener('click', (ev) => {
    const rect = canvas.getBoundingClientRect();
    const cssX = ev.clientX - rect.left; // CSS pixels relative to canvas
    const cssY = ev.clientY - rect.top;
    const devX = cssX * dpr;
    const devY = cssY * dpr;

    const [r,g,b] = parseHexColor(ptPicker.value);

    if (centers.length < maxPoints) {
      centers.push([devX, devY]);
      colors.push([r,g,b]);
    } else {
      const i = idx % maxPoints;
      centers[i] = [devX, devY];
      colors[i] = [r,g,b];
      idx++;
    }
    vertexCount = rebuildBuffers();
    render();
  });

  // --- Clear button: empties points and redraws background with chosen color ---
  clearBtn.addEventListener('click', () => {
    centers = [];
    colors = [];
    vertexCount = rebuildBuffers();
    render();
  });
}

// Helpers
function addColoredSquare(posOut, colOut, cx, cy, size, W, H, rgb) {
  const h = size / 2;
  const x0 = cx - h, x1 = cx + h;
  const y0 = cy - h, y1 = cy + h;
  const toNDC = (x, y) => [ (x / W) * 2 - 1, 1 - (y / H) * 2 ];
  const [x0n, y0n] = toNDC(x0, y0);
  const [x1n, y1n] = toNDC(x1, y1);
  // two triangles (6 vertices) with same color per vertex
  const quad = [
    x0n, y0n,  x1n, y0n,  x1n, y1n,
    x0n, y0n,  x1n, y1n,  x0n, y1n
  ];
  for (let i = 0; i < quad.length; i += 2) {
    posOut.push(quad[i], quad[i+1]);
    colOut.push(rgb[0], rgb[1], rgb[2]);
  }
}

function ensureControls(canvas) {
  if (document.getElementById('controls-bar')) return;
  const bar = document.createElement('div');
  bar.id = 'controls-bar';
  bar.style.margin = '8px auto';
  bar.style.maxWidth = canvas.clientWidth ? canvas.clientWidth + 'px' : '512px';
  bar.style.display = 'flex';
  bar.style.gap = '12px';
  bar.style.alignItems = 'center';

  const clearBtn = document.createElement('button');
  clearBtn.id = 'btn-clear';
  clearBtn.textContent = 'Clear';

  const bgLabel = document.createElement('label');
  bgLabel.textContent = 'Background:';
  const bgColor = document.createElement('input');
  bgColor.type = 'color';
  bgColor.id = 'bg-color';
  bgColor.value = '#6495ed'; // cornflower-ish

  const ptLabel = document.createElement('label');
  ptLabel.textContent = ' Point color:';
  const ptColor = document.createElement('input');
  ptColor.type = 'color';
  ptColor.id = 'pt-color';
  ptColor.value = '#000000';

  bar.appendChild(clearBtn);
  bar.appendChild(bgLabel);
  bar.appendChild(bgColor);
  bar.appendChild(ptLabel);
  bar.appendChild(ptColor);

  canvas.parentNode.insertBefore(bar, canvas);
}

function showError(msg) {
  const pre = document.createElement('pre');
  pre.textContent = msg;
  pre.style.position = 'fixed';
  pre.style.top = '8px';
  pre.style.left = '8px';
  pre.style.padding = '8px';
  pre.style.background = 'rgba(0,0,0,0.75)';
  pre.style.color = '#fff';
  pre.style.zIndex = 9999;
  pre.style.whiteSpace = 'pre-wrap';
  document.body.appendChild(pre);
  console.error(msg);
}
