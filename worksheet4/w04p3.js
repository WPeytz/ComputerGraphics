"use strict";
// Worksheet 4 – Part 3
// Gouraud-shaded sphere (Lambert diffuse) with a distant light; orbiting camera.

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

  let depthTex = device.createTexture({
    size: { width: canvas.width, height: canvas.height },
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // --- Shaders (WGSL) ---
  const shader = device.createShaderModule({
    label: 'W04P1 shaders',
    code: /* wgsl */`
      struct Uniforms { mvp : mat4x4f, color : vec3f, light : vec3f, };
      @group(0) @binding(0) var<uniform> U : Uniforms; // color=k_d, light=world/model dir

      struct VSOut {
        @builtin(position) pos : vec4f,
        @location(0) vColor : vec3f,
      };

      @vertex
      fn vs_main(@location(0) inPos : vec3f) -> VSOut {
        // Unit sphere: normal is normalized position in model space
        let n = normalize(inPos);
        // Use rotating light direction from uniform (already normalized on CPU)
        let l = U.light;
        let ndotl = max(dot(n, l), 0.0);
        let Le = vec3f(1.0, 1.0, 1.0);
        let kd = U.color;
        var out : VSOut;
        out.pos = U.mvp * vec4f(inPos, 1.0);
        out.vColor = kd * Le * ndotl;
        return out;
      }

      @fragment
      fn fs_main(inFrag : VSOut) -> @location(0) vec4f {
        return vec4f(inFrag.vColor, 1.0);
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
    primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' }, // Keep back-face culling; if triangles disappear, try frontFace: 'cw'
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' }
  });

  // GPU buffers (will be resized when subdivision changes)
  let vertexBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  let indexBuffer  = device.createBuffer({ size: 4, usage: GPUBufferUsage.INDEX  | GPUBufferUsage.COPY_DST });

  // Uniforms
  const ubo = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ubo } }] });

  // Camera (perspective)
  const fovy = 45 * Math.PI/180;
  const aspect = canvas.width / canvas.height;
  const P = mat4_perspective_fovY(fovy, aspect, 0.1, 100.0);

  // Write initial kd; MVP will be updated every frame as the camera orbits
  device.queue.writeBuffer(ubo, 64, new Float32Array([1.0, 1.0, 1.0, 0])); // k_d (white for clarity)
  device.queue.writeBuffer(ubo, 80, new Float32Array([0.0, 0.0, -1.0, 0])); // light dir fixed to (0,0,-1)
  let angle = 0; // radians
  let lastTs = null; // for delta-time based animation

  // Model: center-to-origin is identity because sphere already centered
  // Geometry generation (subdividing tetrahedron → unit sphere)
  let level = 2; // default
  rebuild(level);

  // Hook up buttons
  document.getElementById('btnSubDec').onclick = () => { level = Math.max(0, level-1); rebuild(level); };
  document.getElementById('btnSubInc').onclick = () => { level = Math.min(8, level+1); rebuild(level); };

  requestAnimationFrame(tick);

  function tick(ts) {
    // Drive animation straight from timestamp so it always moves,
    // even if the tab throttles rAF (very obvious rotation).
    const speed = 0.001; // slower rotation speed
    angle = (ts * speed) % (Math.PI * 2);

    // Removed rotating light update to keep fixed light direction

    const r = 2.2;
    const eyex = r * Math.sin(angle);
    const eyez = r * Math.cos(angle);
    const eyeY = 1.0; // fixed height, no bobbing
    const eye  = [eyex, eyeY, eyez];
    const target = [0,0,0];
    const up = [0,1,0];
    const V = mat4_lookAt(eye, target, up);
    const M = mat4_identity(); // No model spin
    const MVP = mat4_mul(P, mat4_mul(V, M)); // MVP = P * V * M

    device.queue.writeBuffer(ubo, 0, MVP);

    // Removed on-screen status update and background blink

    drawFrame();
    requestAnimationFrame(tick);
  }

  function drawFrame() {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        clearValue: { r: 0.3921, g: 0.5843, b: 0.9294, a: 1 },
        storeOp: 'store'
      }],
      depthStencilAttachment: {
        view: depthTex.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store'
      }
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setIndexBuffer(indexBuffer, 'uint32');
    pass.drawIndexed(rebuild._indexCount || 0);

    pass.end();
    device.queue.submit([encoder.finish()]);
  }

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
function mat4_rotate_y(rad) {
  const c = Math.cos(rad), s = Math.sin(rad);
  return new Float32Array([
     c, 0,-s, 0,
     0, 1, 0, 0,
     s, 0, c, 0,
     0, 0, 0, 1,
  ]);
}
function mat4_lookAt(eye, target, up){
  // Right-handed lookAt (DirectX-style). Column-major.
  // zaxis points from eye to target.
  let zx = target[0]-eye[0], zy = target[1]-eye[1], zz = target[2]-eye[2];
  const zlen = Math.hypot(zx,zy,zz) || 1; zx/=zlen; zy/=zlen; zz/=zlen;
  // xaxis = normalize(cross(up, zaxis))
  let xx = up[1]*zz - up[2]*zy;
  let xy = up[2]*zx - up[0]*zz;
  let xz = up[0]*zy - up[1]*zx;
  const xlen = Math.hypot(xx,xy,xz) || 1; xx/=xlen; xy/=xlen; xz/=xlen;
  // yaxis = cross(zaxis, xaxis)
  const yx = zy*xz - zz*xy;
  const yy = zz*xx - zx*xz;
  const yz = zx*xy - zy*xx;
  // Translation components: -dot(axis, eye)
  const tx = -(xx*eye[0] + xy*eye[1] + xz*eye[2]);
  const ty = -(yx*eye[0] + yy*eye[1] + yz*eye[2]);
  const tz = -(zx*eye[0] + zy*eye[1] + zz*eye[2]);
  return new Float32Array([
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    tx, ty, tz, 1,
  ]);
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
