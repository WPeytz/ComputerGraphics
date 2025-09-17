"use strict";
// Worksheet 2 – Part 1
// Start from W01P2 solution: clear the canvas and draw three points (20×20 squares).
// Add a mouse click handler that draws points where the user clicks. Use
// getBoundingClientRect() so the mouse coordinates are corrected by the
// canvas' client-area offset. Handle HiDPI via devicePixelRatio.

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

  // --- WGSL shaders ---
  const shaderModule = device.createShaderModule({
    label: 'Part1 shaders',
    code: /* wgsl */`
      @vertex
      fn vs_main(@location(0) inPos : vec2f) -> @builtin(position) vec4f {
        return vec4f(inPos, 0.0, 1.0);
      }

      @fragment
      fn fs_main() -> @location(0) vec4f {
        return vec4f(0.0, 0.0, 0.0, 1.0); // constant black
      }
    `
  });

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
      buffers: [{
        arrayStride: 8, // 2 floats per vertex
        attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }]
      }]
    },
    fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' }
  });

  // --- Geometry state: three 20×20 pixel squares ---
  const sideCSS = 20;             // requested size in CSS pixels
  const S = sideCSS * dpr;        // convert to device pixels for math below

  // Initial centers (top-right, mid-right, center)
  const centersCSS = [
    [cssW - sideCSS / 2, sideCSS / 2],
    [cssW - sideCSS / 2, cssH / 2],
    [cssW / 2,           cssH / 2],
  ];

  let centers = centersCSS.map(([x, y]) => [x * dpr, y * dpr]); // in device px

  // GPU buffer reused across redraws
  let vbo = device.createBuffer({ size: 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });

  function rebuildVertexBuffer() {
    const verts = [];
    for (const [cx, cy] of centers) addSquareTriangles(verts, cx, cy, S, W, H);
    const vertexData = new Float32Array(verts);
    // Recreate/resize buffer if needed
    if (vbo.size < vertexData.byteLength) {
      vbo.destroy?.();
      vbo = device.createBuffer({ size: vertexData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    }
    device.queue.writeBuffer(vbo, 0, vertexData.buffer, vertexData.byteOffset, vertexData.byteLength);
    return vertexData.length / 2; // number of vertices
  }

  let vertexCount = rebuildVertexBuffer();

  function render() {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        clearValue: { r: 0.3921, g: 0.5843, b: 0.9294, a: 1 }, // blue background
        storeOp: 'store'
      }]
    });
    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, vbo);
    pass.draw(vertexCount);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  // Initial draw
  render();

  // --- Mouse click handler ---
  // Replace the oldest point so that up to 3 points are drawn. Correct for canvas offset.
  let idx = 0;
  canvas.addEventListener('click', (ev) => {
    const rect = canvas.getBoundingClientRect();
    const cssX = ev.clientX - rect.left; // CSS pixels relative to canvas
    const cssY = ev.clientY - rect.top;
    const devX = cssX * dpr;
    const devY = cssY * dpr;

    centers[idx % 3] = [devX, devY];
    idx++;

    vertexCount = rebuildVertexBuffer();
    render();
  });
}

// Push two triangles (6 vertices) for a square centered at (cx, cy) with side `size`
function addSquareTriangles(out, cx, cy, size, W, H) {
  const h = size / 2;
  const x0 = cx - h, x1 = cx + h;
  const y0 = cy - h, y1 = cy + h;

  // Convert from device pixels to NDC [-1,1] (origin top-left => center)
  const toNDC = (x, y) => [ (x / W) * 2 - 1, 1 - (y / H) * 2 ];
  const [x0n, y0n] = toNDC(x0, y0);
  const [x1n, y1n] = toNDC(x1, y1);

  // Two triangles: (x0,y0)-(x1,y0)-(x1,y1) and (x0,y0)-(x1,y1)-(x0,y1)
  out.push(
    x0n, y0n,  x1n, y0n,  x1n, y1n,
    x0n, y0n,  x1n, y1n,  x0n, y1n
  );
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
