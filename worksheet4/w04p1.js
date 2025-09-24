"use strict";
// Worksheet 4 – Part 1
// Draw a sphere in perspective view.
// - Start from W3P2 but simplify to ONE object centered in view.
// - Render TRIANGLES (filled), not wireframe.
// - Build a unit sphere by subdividing a tetrahedron; after each subdivision
//   project vertices to the unit sphere (geodesic sphere). Two buttons (+ / -)
//   change the subdivision level.

window.onload = () => { main().catch(err => showError(String(err))); };

async function main() {
  if (!('gpu' in navigator) || !navigator.gpu) {
    showError("WebGPU not available. Use a recent Chrome/Edge on http://localhost.");
    return;
  }

  const canvas = document.getElementById('my-canvas');
  if (!canvas) { showError("Missing <canvas id='my-canvas'> in HTML."); return; }

  // Controls (− / +)
  ensureControls();

  // HiDPI sizing
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth || 640;
  const cssH = canvas.clientHeight || 480;
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
    label: 'W04P1 shaders',
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
        // Solid light gray; lighting will be added in later parts
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
    primitive: { topology: 'triangle-list', cullMode: 'back' }
  });

  // GPU buffers (will be resized when subdivision changes)
  let vertexBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  let indexBuffer  = device.createBuffer({ size: 4, usage: GPUBufferUsage.INDEX  | GPUBufferUsage.COPY_DST });

  // Uniforms
  const ubo = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ubo } }] });

  // Camera (perspective)
  const fovy = 45 * Math.PI/180;
  const aspect = canvas.width / canvas.height;
  const P = mat4_perspective_fovY(fovy, aspect, 0.1, 100.0);
  const V = mat4_translate(0, 0, 3.0); // push scene forward

  // Model: center-to-origin is identity because sphere already centered
  const MVP = mat4_mul(P, V);
  device.queue.writeBuffer(ubo, 0, MVP);
  device.queue.writeBuffer(ubo, 64, new Float32Array([0.92, 0.92, 0.95, 0]));

  // Geometry generation (subdividing tetrahedron → unit sphere)
  let level = 2; // default
  rebuild(level);

  // Hook up buttons
  document.getElementById('btnSubDec').onclick = () => { level = Math.max(0, level-1); rebuild(level); };
  document.getElementById('btnSubInc').onclick = () => { level = Math.min(8, level+1); rebuild(level); };

  draw();

  function rebuild(n) {
    const { positions, indices } = buildTetraSphere(n);

    // (Re)allocate buffers if needed
    const vbBytes = positions.byteLength;
    const ibBytes = indices.byteLength;
    if (vertexBuffer.size < vbBytes) {
      vertexBuffer.destroy();
      vertexBuffer = device.createBuffer({ size: nextPow2(vbBytes), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    }
    if (indexBuffer.size < ibBytes) {
      indexBuffer.destroy();
      indexBuffer = device.createBuffer({ size: nextPow2(ibBytes), usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    }
    device.queue.writeBuffer(vertexBuffer, 0, positions);
    device.queue.writeBuffer(indexBuffer, 0, indices);

    // Store counts for draw
    rebuild._indexCount = indices.length;
    draw();
  }

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
    pass.setIndexBuffer(indexBuffer, 'uint32');
    pass.drawIndexed(rebuild._indexCount || 0);

    pass.end();
    device.queue.submit([encoder.finish()]);
  }
}

// ----------------- Sphere generation -----------------------------
// Start with a regular tetrahedron then repeatedly split each triangle into
// 4 by edge midpoints. Project every vertex to the unit sphere after each
// subdivision. This yields a geodesic sphere that is perfectly adequate for
// this part (and matches the spirit of Loop-refinement toward a smooth limit).
function buildTetraSphere(level) {
  // Regular tetrahedron centered at origin, unit radius
  const t = Math.sqrt(2);
  const V0 = [ 1,  1,  1];
  const V1 = [-1, -1,  1];
  const V2 = [-1,  1, -1];
  const V3 = [ 1, -1, -1];
  let verts = [V0, V1, V2, V3].map(nrm);
  let faces = [
    [0,1,2],
    [0,3,1],
    [0,2,3],
    [1,3,2],
  ];

  for (let s = 0; s < level; s++) {
    const edgeMid = new Map();
    const newFaces = [];
    function key(a,b){ return a<b ? (a+","+b) : (b+","+a); }
    function midpoint(a,b){
      const k = key(a,b);
      if (edgeMid.has(k)) return edgeMid.get(k);
      const va = verts[a], vb = verts[b];
      const m = nrm([ (va[0]+vb[0])*0.5, (va[1]+vb[1])*0.5, (va[2]+vb[2])*0.5 ]);
      const idx = verts.push(m) - 1;
      edgeMid.set(k, idx);
      return idx;
    }
    for (const f of faces) {
      const [a,b,c] = f;
      const ab = midpoint(a,b);
      const bc = midpoint(b,c);
      const ca = midpoint(c,a);
      newFaces.push([a,ab,ca]);
      newFaces.push([ab,b,bc]);
      newFaces.push([ca,bc,c]);
      newFaces.push([ab,bc,ca]);
    }
    faces = newFaces;
  }

  // Flatten to buffers
  const positions = new Float32Array(verts.length * 3);
  for (let i=0;i<verts.length;i++){ positions.set(verts[i], i*3); }
  const indices = new Uint32Array(faces.length * 3);
  let k=0; for (const f of faces) { indices[k++]=f[0]; indices[k++]=f[1]; indices[k++]=f[2]; }
  return { positions, indices };
}

function nrm(v){
  const l = Math.hypot(v[0],v[1],v[2]) || 1;
  return [v[0]/l, v[1]/l, v[2]/l];
}

function nextPow2(n){ let p=1; while(p<n) p<<=1; return p; }

// ----------------- Math helpers (column-major) -----------------------------
function mat4_identity() { return new Float32Array([1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1]); }
function mat4_mul(a, b) {
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
  m[12] = x; m[13] = y; m[14] = z;
  return m;
}
function mat4_perspective_fovY(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect; m[5] = f; m[10] = far / (far - near); m[11] = 1; m[14] = (-near * far) / (far - near);
  return m;
}

function ensureControls(){
  if (document.getElementById('btnSubInc')) return;
  const wrap = document.getElementById('controls') || (function(){
    const d = document.createElement('div');
    d.id = 'controls';
    d.style.margin = '8px 0';
    d.style.display = 'flex';
    d.style.gap = '8px';
    document.body.insertBefore(d, document.body.firstChild);
    return d;
  })();
  const dec = document.createElement('button'); dec.id='btnSubDec'; dec.textContent='− subdiv';
  const inc = document.createElement('button'); inc.id='btnSubInc'; inc.textContent='+ subdiv';
  wrap.append(dec, inc);
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
