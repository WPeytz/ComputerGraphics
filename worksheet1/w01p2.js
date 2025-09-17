"use strict";
// Part 2: Shaders and buffers — draw three 20×20 px black squares
// using a vertex buffer (two triangles per square).

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
    label: 'Part2 shaders',
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

  // --- Geometry: three 20×20 pixel squares ---
  const sideCSS = 20;             // requested size in CSS pixels
  const S = sideCSS * dpr;        // convert to device pixels for math below

  // Choose centers (cx, cy) in CSS pixels to center the middle square exactly and
  // account for the 20px size so edge squares are flush with the borders
  const centersCSS = [
    [cssW - sideCSS / 2, sideCSS / 2], // top-right corner (flush to top & right)
    [cssW - sideCSS / 2, cssH / 2],    // exact middle-right
    [cssW / 2,           cssH / 2],    // exact center of canvas
  ];

  // Convert to device pixels for consistent mapping to NDC with W×H buffer
  const centers = centersCSS.map(([x, y]) => [x * dpr, y * dpr]);

  const verts = [];
  for (const [cx, cy] of centers) addSquareTriangles(verts, cx, cy, S, W, H);

  const vertexData = new Float32Array(verts);
  const vbo = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false
  });
  device.queue.writeBuffer(vbo, 0, vertexData.buffer, vertexData.byteOffset, vertexData.byteLength);

  // --- Render ---
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: 'clear',
      clearValue: { r: 0.3921, g: 0.5843, b: 0.9294, a: 1 }, // blue-ish background
      storeOp: 'store'
    }]
  });

  pass.setPipeline(pipeline);
  pass.setVertexBuffer(0, vbo);
  pass.draw(vertexData.length / 2); // 2 floats per vertex
  pass.end();

  device.queue.submit([encoder.finish()]);
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
