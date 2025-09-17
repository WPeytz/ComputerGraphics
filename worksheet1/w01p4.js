"use strict";
// Part 4: Rotating square (two triangles), centered, width/height = 1 in NDC
// Uses a uniform mat4 rotation updated every frame via requestAnimationFrame.

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
        return vec4f(1.0, 1.0, 1.0, 1.0); // white quad
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
    primitive: { topology: 'triangle-list' }
  });

  // Quad centered at origin, side length = 1 (so width=1, height=1 in NDC)
  const verts = new Float32Array([
    // Triangle 1
    -0.5, -0.5,
     0.5, -0.5,
     0.5,  0.5,
    // Triangle 2
    -0.5, -0.5,
     0.5,  0.5,
    -0.5,  0.5,
  ]);
  const vbo = device.createBuffer({ size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vbo, 0, verts.buffer, verts.byteOffset, verts.byteLength);

  // Uniform buffer for rotation matrix
  const ubo = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: ubo } }]
  });

  function render(tsMs) {
    const t = tsMs * 0.001; // seconds
    const c = Math.cos(t);
    const s = Math.sin(t);
    // Column-major mat4 rotation around Z
    const m = new Float32Array([
       c,  s, 0, 0,
      -s,  c, 0, 0,
       0,  0, 1, 0,
       0,  0, 0, 1,
    ]);
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
    pass.draw(6);
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
