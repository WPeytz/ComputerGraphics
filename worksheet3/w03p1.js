"use strict";
// Worksheet 3 – Part 1
// Draw a wireframe unit cube in isometric view using ORTHOGRAPHIC projection.
// World space: cube corners at (0/1)^3 (diagonal from (0,0,0) to (1,1,1)).
// Render as line-list (12 edges).

window.onload = () => { main().catch(err => showError(String(err))); };

async function main() {
  // --- Feature detect ---
  if (!('gpu' in navigator) || !navigator.gpu) {
    showError("WebGPU not available. Use a recent Chrome/Edge on http://localhost.");
    return;
  }

    const canvas = document.getElementById('my-canvas');
  if (!canvas) { showError("Missing <canvas id='my-canvas'> in HTML."); return; }

  // Size the drawing buffer (keep CSS size; render crisp on HiDPI)
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth || 512;
  const cssH = canvas.clientHeight || 512;
  canvas.width  = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { showError("No GPU adapter available."); return; }
  const device = await adapter.requestDevice();

  device.onuncapturederror = (e) => console.error('WebGPU error:', e.error || e);

  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  // --- Shaders (WGSL) ---
  const shader = device.createShaderModule({
    label: 'W03P1 shaders',
    code: /* wgsl */`
      struct Uniforms { mvp : mat4x4f, };
      @group(0) @binding(0) var<uniform> U : Uniforms;

      struct VSOut { @builtin(position) pos : vec4f, };

      @vertex
      fn vs_main(@location(0) inPos : vec3f) -> VSOut {
        var out : VSOut;
        out.pos = U.mvp * vec4f(inPos, 1.0);
        return out;
      }

      @fragment
      fn fs_main() -> @location(0) vec4f {
        return vec4f(0.0, 0.0, 0.0, 1.0);
      }
    `
  });

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: shader,
      entryPoint: 'vs_main',
      buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }]
    },
    fragment: { module: shader, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'line-list' }
  });

  // --- Geometry ---
  // Cube vertices at the 8 corners of [0,1]^3 (world space)
  const V = new Float32Array([
    // order: (x,y,z)
    0,0,0,  1,0,0,  1,1,0,  0,1,0, // z=0 face
    0,0,1,  1,0,1,  1,1,1,  0,1,1  // z=1 face
  ]);
  const vertexBuffer = device.createBuffer({
    size: V.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(vertexBuffer, 0, V);

  // 12 edges as pairs of vertex indices (line-list)
  const I = new Uint16Array([
    // bottom square (z=0)
    0,1,  1,2,  2,3,  3,0,
    // top square (z=1)
    4,5,  5,6,  6,7,  7,4,
    // verticals
    0,4,  1,5,  2,6,  3,7
  ]);
  const indexBuffer = device.createBuffer({
    size: I.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(indexBuffer, 0, I);

  // --- Uniforms: build MVP for an ISOMETRIC view with ORTHO projection ---
  const ubo = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: ubo } }]
  });

  // Ortho projection with z in [0, 1] (DirectX/WebGPU convention)
  const ortho = mat4_ortho(-1, 1, -1, 1, 0.1, 10.0);

  // Model/View: center the cube at origin, rotate to isometric, then push it forward in +Z
  const Tcenter = mat4_translate(-0.5, -0.5, -0.5);    // move cube center to origin
  const Ry = mat4_rotate_y(Math.PI / 4);                // 45° around Y
  const Rx = mat4_rotate_x(35.264389682754654 * Math.PI / 180); // ~35.264° around X
  const Tz = mat4_translate(0, 0, 2.5);                 // move cube into the near/far range

  // MVP = ortho * Tz * Rx * Ry * Tcenter
  const MVP = mat4_mul(mat4_mul(mat4_mul(mat4_mul(ortho, Tz), Rx), Ry), Tcenter);
  device.queue.writeBuffer(ubo, 0, MVP);

  // --- Draw once ---
  draw();

  function draw() {
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
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setIndexBuffer(indexBuffer, 'uint16');
    pass.drawIndexed(I.length);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }
}

// ----------------- Math helpers (column-major) -----------------------------
function mat4_identity() { return new Float32Array([1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1]); }
function mat4_mul(a, b) {
  // c = a * b (column-major)
  const c = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    const bi = col*4;
    for (let row = 0; row < 4; row++) {
      c[bi+row] = a[row] * b[bi] + a[4+row] * b[bi+1] + a[8+row] * b[bi+2] + a[12+row] * b[bi+3];
    }
  }
  return c;
}
function mat4_translate(x, y, z) {
  const m = mat4_identity();
  m[12] = x; m[13] = y; m[14] = z; // last column
  return m;
}
function mat4_rotate_x(rad) {
  const c = Math.cos(rad), s = Math.sin(rad);
  return new Float32Array([
    1, 0, 0, 0,
    0, c, s, 0,
    0,-s, c, 0,
    0, 0, 0, 1,
  ]);
}
function mat4_rotate_y(rad) {
  const c = Math.cos(rad), s = Math.sin(rad);
  return new Float32Array([
     c, 0,-s, 0,
     0, 1, 0, 0,
     s, 0, c, 0,
     0, 0, 0, 1,
  ]);
}
function mat4_ortho(l, r, b, t, n, f) {
  // DirectX-style depth [0,1]
  const m = new Float32Array(16);
  m[0] = 2/(r-l); m[5] = 2/(t-b); m[10] = 1/(f-n); m[15] = 1;
  m[12] = -(r+l)/(r-l);
  m[13] = -(t+b)/(t-b);
  m[14] = -n/(f-n);
  return m;
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
