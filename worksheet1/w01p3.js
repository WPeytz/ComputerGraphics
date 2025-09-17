"use strict";
// Part 3: Draw a single triangle with per-vertex color interpolation
// Two vertex buffers: positions (float2) and colors (float3)

window.onload = () => { main().catch(err => showError(String(err))); };

async function main() {
  if (!('gpu' in navigator) || !navigator.gpu) {
    showError("WebGPU not available. Use a recent Chrome/Edge on http://localhost.");
    return;
  }

  const canvas = document.getElementById('my-canvas');
  if (!canvas) { showError("Missing <canvas id='my-canvas'>."); return; }

  // Canvas sizing
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth || 512;
  const cssH = canvas.clientHeight || 512;
  const W = Math.floor(cssW * dpr);
  const H = Math.floor(cssH * dpr);
  canvas.width = W;
  canvas.height = H;

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { showError("Failed to get GPU adapter"); return; }
  const device = await adapter.requestDevice();

  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  // --- WGSL shaders ---
  const shader = device.createShaderModule({
    code: /* wgsl */`
      struct VSOut {
        @builtin(position) pos : vec4f,
        @location(0) color : vec3f,
      };

      @vertex
      fn vs_main(@location(0) inPos : vec2f, @location(1) inColor : vec3f) -> VSOut {
        var out : VSOut;
        out.pos = vec4f(inPos, 0.0, 1.0);
        out.color = inColor;
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
      module: shader,
      entryPoint: 'vs_main',
      buffers: [
        { // position buffer: float2
          arrayStride: 8,
          attributes: [{ shaderLocation: 0, format: 'float32x2', offset: 0 }]
        },
        { // color buffer: float3
          arrayStride: 12,
          attributes: [{ shaderLocation: 1, format: 'float32x3', offset: 0 }]
        }
      ]
    },
    fragment: { module: shader, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' }
  });

  // --- Geometry (in CSS pixels). Right-angled triangle similar to the worksheet image.
  // Vertices in CSS pixels (then converted to NDC via W,H from device pixels)
  const pCss = [
    [cssW/2, cssH/2], // left vertex (red)
    [cssW , cssH/2], // right-bottom (green)
    [cssW , 0]  // right-top (blue)
  ];

  // Convert from pixel coords to NDC [-1,1]
  const toNDC = (x, y) => [ (x * dpr / W) * 2 - 1, 1 - (y * dpr / H) * 2 ];
  const pos = new Float32Array(pCss.flatMap(([x, y]) => toNDC(x, y)));

  // Colors per vertex: R, G, B
  const col = new Float32Array([
    1, 0, 0,  // red
    0, 1, 0,  // green
    0, 0, 1   // blue
  ]);

  const posBuf = device.createBuffer({ size: pos.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  const colBuf = device.createBuffer({ size: col.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(posBuf, 0, pos.buffer, pos.byteOffset, pos.byteLength);
  device.queue.writeBuffer(colBuf, 0, col.buffer, col.byteOffset, col.byteLength);

  // --- Render ---
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: 'clear',
      clearValue: { r: 0.3921, g: 0.5843, b: 0.9294, a: 1 },
      storeOp: 'store'
    }]
  });

  pass.setPipeline(pipeline);
  pass.setVertexBuffer(0, posBuf);
  pass.setVertexBuffer(1, colBuf);
  pass.draw(3, 1, 0, 0);
  pass.end();

  device.queue.submit([encoder.finish()]);
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
