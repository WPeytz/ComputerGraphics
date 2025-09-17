"use strict";
// Worksheet 3 – Part 2
// Draw the unit cube (wireframe) in THREE classical perspective views in one render:
//  • One-point (front) perspective
//  • Two-point (X) perspective
//  • Three-point perspective
// Use a pinhole camera with 45° vertical FOV (perspective projection). Draw as line-list.

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
    label: 'W03P2 shaders',
    code: /* wgsl */`
      struct Uniforms { mvp : mat4x4f, color : vec3f, };
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
        return vec4f(U.color, 1.0);
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

  // --- Geometry (8 vertices, 12 edges) ---
  const V = new Float32Array([
    0,0,0,  1,0,0,  1,1,0,  0,1,0, // z=0 face
    0,0,1,  1,0,1,  1,1,1,  0,1,1  // z=1 face
  ]);
  const vertexBuffer = device.createBuffer({ size: V.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vertexBuffer, 0, V);

  const I = new Uint16Array([
    0,1,  1,2,  2,3,  3,0, // bottom
    4,5,  5,6,  6,7,  7,4, // top
    0,4,  1,5,  2,6,  3,7  // verticals
  ]);
  const indexBuffer = device.createBuffer({ size: I.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(indexBuffer, 0, I);

  const uniformByteSize = 96; // mat4 (64) + padded vec3 (16)
  const colorScratch = new Float32Array(4);

  // Camera: perspective 45° vertical FOV
  const fovy = 45 * Math.PI/180; 
  const aspect = canvas.width / canvas.height;
  const persp = mat4_perspective_fovY(fovy, aspect, 0.1, 100.0);
  const view  = mat4_translate(0, 0, 5.0); // view moves scene so camera sits at origin

  // Common: move cube center (0.5,0.5,0.5) to origin
  const Tcenter = mat4_translate(-0.5, -0.5, -0.5);

  const deg = Math.PI / 180;
  // Three classical perspective setups, laid out left→right in view space
  const cfgs = [
    { label: '1pt-front',  rotate: [0,          0,            0],            translate: [-1.3, 0, 0], color: [1.0, 0.25, 0.25] },
    { label: '2pt-x',      rotate: [0,         30 * deg,      0],            translate: [ 0.0, 0, 0], color: [0.3, 0.9, 0.3] },
    { label: '3pt-xyz',    rotate: [20 * deg,  30 * deg, -15 * deg],         translate: [ 1.3, 0, 0], color: [0.25, 0.45, 1.0] },
  ];

  for (const cfg of cfgs) {
    cfg.ubo = device.createBuffer({ size: uniformByteSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    cfg.bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: cfg.ubo } }] });
  }

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
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setIndexBuffer(indexBuffer, 'uint16');

    for (const c of cfgs) {
      const [rx, ry, rz] = c.rotate;
      const Rx = mat4_rotate_x(rx);
      const Ry = mat4_rotate_y(ry);
      const Rz = mat4_rotate_z(rz);
      // Apply layout in VIEW space so cubes are side-by-side regardless of rotation
      const Tview = mat4_translate(c.translate[0], c.translate[1], c.translate[2]);
      const Vlayout = mat4_mul(Tview, view);

      let mvp = mat4_mul(persp, Vlayout);
      mvp = mat4_mul(mvp, Rz);
      mvp = mat4_mul(mvp, Ry);
      mvp = mat4_mul(mvp, Rx);
      mvp = mat4_mul(mvp, Tcenter);
      device.queue.writeBuffer(c.ubo, 0, mvp);

      colorScratch[0] = c.color[0];
      colorScratch[1] = c.color[1];
      colorScratch[2] = c.color[2];
      colorScratch[3] = 0;
      device.queue.writeBuffer(c.ubo, 64, colorScratch);
      pass.setBindGroup(0, c.bindGroup);
      pass.drawIndexed(I.length);
    }

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
  m[12] = x; m[13] = y; m[14] = z; // last column (column-major)
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
function mat4_rotate_z(rad) {
  const c = Math.cos(rad), s = Math.sin(rad);
  return new Float32Array([
     c, s, 0, 0,
    -s, c, 0, 0,
     0, 0, 1, 0,
     0, 0, 0, 1,
  ]);
}
function mat4_perspective_fovY(fovy, aspect, near, far) {
  // DirectX-style depth [0,1], right-handed clip; column-major
  const f = 1 / Math.tan(fovy / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect; // x
  m[5] = f;          // y
  m[10] = far / (far - near); // z
  m[11] = 1;                  // w term
  m[14] = (-near * far) / (far - near);
  // others default 0; m[15] is 0 for perspective
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
