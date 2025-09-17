"use strict";
// Part 5: Bouncing circle using TRIANGLE_STRIP
// We build a vertex buffer that alternates points on the upper and lower
// semicircles, producing a filled disk when drawn as a triangle strip.
// The circle bounces up and down via a uniform translation matrix.

window.onload = () => { main().catch(err => showError(String(err))); };

async function main() {
  if (!('gpu' in navigator) || !navigator.gpu) {
    showError("WebGPU not available. Use a recent Chrome/Edge on http://localhost.");
    return;
  }

  const canvas = document.getElementById('my-canvas');
  if (!canvas) { showError("Missing <canvas id='my-canvas'>."); return; }

  // Canvas sizing (respect CSS, render crisp on HiDPI)
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

  // ---- Shaders (WGSL) ----
  const shader = device.createShaderModule({
    code: /* wgsl */`
      struct Uniforms { mvp : mat4x4f };
      @group(0) @binding(0) var<uniform> u : Uniforms;

      @vertex
      fn vs_main(@location(0) inPos : vec2f) -> @builtin(position) vec4f {
        return u.mvp * vec4f(inPos, 0.0, 1.0);
      }

      @fragment
      fn fs_main() -> @location(0) vec4f {
        return vec4f(1.0, 1.0, 1.0, 1.0); // white fill
      }
    `
  });

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: shader,
      entryPoint: 'vs_main',
      buffers: [{
        arrayStride: 8, // float2 position
        attributes: [{ shaderLocation: 0, format: 'float32x2', offset: 0 }]
      }]
    },
    fragment: { module: shader, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-strip' }
  });

  // --- Geometry: circle centered at origin (NDC), radius r
  const r = 0.35;              // circle radius in NDC
  const slices = 128;          // must be even (we use upper/lower pairs)
  const verts = [];
  for (let i = 0; i <= slices; ++i) {
    const t = (i / slices) * Math.PI;          // 0 .. π
    const x = r * Math.cos(t);
    const y = r * Math.sin(t);
    // Alternate top and bottom points at the same x to form a strip
    verts.push(x,  y);  // upper semicircle point
    verts.push(x, -y);  // mirrored lower point
  }
  const vertexData = new Float32Array(verts);
  const vbo = device.createBuffer({ size: vertexData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vbo, 0, vertexData.buffer, vertexData.byteOffset, vertexData.byteLength);

  // Uniform buffer for translation matrix (bounce in Y)
  const ubo = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: ubo } }]
  });

  function render(tsMs) {
    const t = tsMs * 0.001; // seconds
    // Bounce between [-amp, +amp] in Y using a sine wave. Keep within view: amp < 1 - r
    const amp = 1.0 - r;
    const ty = Math.sin(t * 2.0) * amp; // speed factor 2.0

    // Column-major 4x4 translation matrix (no rotation or scale)
    const m = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      ty * 0 + 0, ty, 0, 1, // WGSL uses column-major; translation in last column
    ]);
    // The above sets m[12]=0, m[13]=ty (translation x/y). Written explicitly for clarity:
    m[12] = 0; // translate X
    m[13] = ty; // translate Y

    device.queue.writeBuffer(ubo, 0, m.buffer, m.byteOffset, m.byteLength);

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
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vbo);
    pass.draw(vertexData.length / 2); // 2 floats per vertex
    pass.end();
    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
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
