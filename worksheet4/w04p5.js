"use strict";
// Worksheet 4 – Part 5
// Full Phong reflection (ambient + diffuse + specular) in the FRAGMENT shader (Phong).
// Sliders for kd, ks, shininess s, and light Le/La (white light); camera orbits.

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
  const USE_SMOKE_TEST = false; // set to false after we see something on screen

  let depthTex = device.createTexture({
    size: { width: canvas.width, height: canvas.height },
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // --- Shaders (WGSL) ---

  let smokeVbo = null;

  let pipeline;
  let pipelineReady = false;
  if (USE_SMOKE_TEST) {
    const smoke = device.createShaderModule({
      label: 'smoke-shader',
      code: /* wgsl */`
        @vertex
        fn vs(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> {
          return vec4<f32>(p, 1.0);
        }
        @fragment
        fn fs() -> @location(0) vec4<f32> {
          return vec4<f32>(0.95, 0.2, 0.2, 1.0);
        }
      `
    });
    // Compilation diagnostics for smoke shader
    const smokeInfo = await smoke.getCompilationInfo();
    if (smokeInfo.messages?.length) {
      for (const m of smokeInfo.messages) {
        const where = (m.lineNum !== undefined) ? `:${m.lineNum}:${m.linePos || 0}` : '';
        console[m.type === 'error' ? 'error' : (m.type === 'warning' ? 'warn' : 'log')](
          `SMOKE WGSL ${m.type}${where}: ${m.message}`
        );
      }
    }
    device.pushErrorScope('validation');
    try {
      pipeline = await device.createRenderPipelineAsync({
        label: 'smoke-pipeline',
        layout: 'auto',
        vertex: { module: smoke, entryPoint: 'vs', buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
        fragment: { module: smoke, entryPoint: 'fs', targets: [{ format }] },
        primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' }
      });
      pipelineReady = true;
    } catch (err) {
      console.error('Smoke pipeline async error:', err);
      showError('Smoke pipeline async error: ' + (err && err.message ? err.message : String(err)));
    }
    const perr = await device.popErrorScope();
    if (perr) {
      console.error('Smoke pipeline validation error:', perr.message || perr);
      showError('Smoke pipeline error: ' + (perr.message || perr));
    }
    // Minimal dedicated vertex buffer for smoke test (NDC triangle)
    const smokeVerts = new Float32Array([
      -1.0, -1.0, 0.0,
       3.0, -1.0, 0.0,
      -1.0,  3.0, 0.0,
    ]);
    smokeVbo = device.createBuffer({
      size: smokeVerts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(smokeVbo, 0, smokeVerts);
  } else {
    // --- DEBUG: log WGSL with line numbers as the browser receives it ---
    const WGSL_SRC = `
struct Uniforms {
  mvp   : mat4x4<f32>,
  kd    : vec4<f32>,   // diffuse color (rgb), a unused
  ks    : vec4<f32>,   // specular color (rgb), a unused
  light : vec4<f32>,   // light direction (xyz), w unused
  eye   : vec4<f32>,   // camera position (xyz), w unused
  params: vec4<f32>,   // x: shininess s, y: Le scale, z: La scale
};
@group(0) @binding(0) var<uniform> U : Uniforms;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) vNormal   : vec3<f32>,
  @location(1) vPos      : vec3<f32>,
};

@vertex
fn vs_main(@location(0) inPos: vec3<f32>) -> VSOut {
  var out : VSOut;
  // Model is identity; normal = position on unit sphere
  out.vNormal = normalize(inPos);
  out.vPos    = inPos;
  out.pos     = U.mvp * vec4<f32>(inPos, 1.0);
  return out;
}

@fragment
fn fs_main(@location(0) vNormal: vec3<f32>, @location(1) vPos: vec3<f32>) -> @location(0) vec4<f32> {
  // Re-normalize interpolated vectors
  let n = normalize(vNormal);
  let l = normalize(U.light.xyz);
  let v = normalize(U.eye.xyz - vPos);
  let r = reflect(-l, n);

  let ndotl = max(dot(n, l), 0.0);
  let rv    = max(dot(r, v), 0.0);

  let s  = U.params.x;
  let Le = U.params.y;
  let La = U.params.z;

  let ambient  = La * U.kd.xyz;
  let diffuse  = Le * U.kd.xyz * ndotl;
  let specular = Le * U.ks.xyz * pow(rv, s);
  let color = ambient + diffuse + specular;
  return vec4<f32>(color, 1.0);
}
`;
    console.log('\n----- WGSL (w04p5) -----');
    WGSL_SRC.split('\n').forEach((ln, i)=>console.log(String(i+1).padStart(3,' ')+': '+ln));
    // Createhopw the main shader module and show diagnostics only in non-smoke path
    device.pushErrorScope('validation');
    const shader = device.createShaderModule({
      label: 'W04P5 shaders',
      code: WGSL_SRC
    });
    const compInfo = await shader.getCompilationInfo();
    if (compInfo.messages?.length) {
      for (const m of compInfo.messages) {
        const where = (m.lineNum !== undefined) ? `:${m.lineNum}:${m.linePos || 0}` : '';
        console[m.type === 'error' ? 'error' : (m.type === 'warning' ? 'warn' : 'log')](
          `WGSL ${m.type}${where}: ${m.message}`
        );
        if (m.type === 'error') showError(`WGSL error${where}: ${m.message}`);
      }
    }
    await device.popErrorScope().then(err => {
      if (err) {
        console.error('Shader validation error:', err.message || err);
        showError('Shader error: ' + (err.message || err));
      }
    });
    // Catch pipeline validation errors with a readable message
    device.pushErrorScope('validation');
    try {
      pipeline = await device.createRenderPipelineAsync({
        layout: 'auto',
        vertex: {
          module: shader,
          entryPoint: 'vs_main',
          buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }]
        },
        fragment: { module: shader, entryPoint: 'fs_main', targets: [{ format }] },
        depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
        primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' }
      });
      pipelineReady = true;
    } catch (err) {
      console.error('Pipeline async error:', err);
      showError('Pipeline async error: ' + (err && err.message ? err.message : String(err)));
    }
    const perr = await device.popErrorScope();
    if (perr) {
      console.error('Pipeline validation error:', perr.message || perr);
      showError('Pipeline error: ' + (perr.message || perr));
    }
    // If the pipeline failed to compile, stop before trying to use it
    if (!pipeline || !pipelineReady) {
      showError('Pipeline failed to compile — see WGSL errors above.');
      return;
    }
  }

  // GPU buffers (will be resized when subdivision changes)
  let vertexBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  let indexBuffer  = device.createBuffer({ size: 4, usage: GPUBufferUsage.INDEX  | GPUBufferUsage.COPY_DST });

  let ubo, bindGroup, sliders;
  let kdScale, ksScale, shin, LeScale, LaScale;
  const fovy = 45 * Math.PI/180;
  const aspect = canvas.width / canvas.height;
  const P = mat4_perspective_fovY(fovy, aspect, 0.1, 100.0);
  const kdBase = new Float32Array([1.0, 0.35, 0.2, 0]);   // warm diffuse
  const ksBase = new Float32Array([1.0, 1.0, 1.0, 0]);    // white specular

  if (!USE_SMOKE_TEST) {
    ubo = device.createBuffer({ size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ubo } }] });
    sliders = addPhongSliders();
    // initial scales
    kdScale = +sliders.kd.value;    // 0..1
    ksScale = +sliders.ks.value;    // 0..1
    shin    = +sliders.s.value;     // 1..128
    LeScale = +sliders.Le.value;    // 0..3
    LaScale = +sliders.La.value;    // 0..1

    function writeMaterial() {
      const kd = new Float32Array([ kdBase[0]*kdScale, kdBase[1]*kdScale, kdBase[2]*kdScale, 0 ]);
      const ks = new Float32Array([ ksBase[0]*ksScale, ksBase[1]*ksScale, ksBase[2]*ksScale, 0 ]);
      device.queue.writeBuffer(ubo, 64, kd);
      device.queue.writeBuffer(ubo, 80, ks);
      device.queue.writeBuffer(ubo, 128, new Float32Array([shin, LeScale, LaScale, 0]));
    }
    // set once
    writeMaterial();

    // React to slider changes immediately
    for (const key of ['kd','ks','s','Le','La']) {
      sliders[key].addEventListener('input', () => {
        kdScale = +sliders.kd.value;
        ksScale = +sliders.ks.value;
        shin    = +sliders.s.value;
        LeScale = +sliders.Le.value;
        LaScale = +sliders.La.value;
        writeMaterial();
      });
    }

    // Write initial light direction only (kd/ks handled by writeMaterial)
    device.queue.writeBuffer(ubo, 96, new Float32Array([0.0, 0.0, 1.0, 0])); // light dir (offset 96)
  }
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
    // Delta-time (seconds) to make speed consistent across machines
    if (lastTs === null) lastTs = ts;
    const dt = Math.max(0, (ts - lastTs) / 1000);
    lastTs = ts;
    
    // Make orbit obvious: faster turn + gentle vertical bobbing
    const omega = 1.2; // rad/s (~69°/s)
    angle = (angle + omega * dt) % (Math.PI * 2);

    if (!USE_SMOKE_TEST) {
      // Rotate light around Y so highlight sweeps across the sphere
      const lx = Math.sin(angle);
      const lz = Math.cos(angle);
      device.queue.writeBuffer(ubo, 96, new Float32Array([lx, 0.0, lz, 0]));

      const r = 3.0;
      const eyex = r * Math.sin(angle);
      const eyez = r * Math.cos(angle);
      const eyeY = 1.0 + 0.4 * Math.sin(angle * 0.5); // bob up/down
      const eye  = [eyex, eyeY, eyez];
      const target = [0,0,0];
      const up = [0,1,0];
      const V = mat4_lookAt(eye, target, up);
      const MVP = mat4_mul(P, V);

      device.queue.writeBuffer(ubo, 0, MVP);
      device.queue.writeBuffer(ubo, 112, new Float32Array([eye[0], eye[1], eye[2], 0]));
    }

    drawFrame();
    requestAnimationFrame(tick);
  }

  function drawFrame() {
    if (!pipelineReady) return; // wait until pipeline finishes compiling/validating
    device.pushErrorScope('validation');
    const encoder = device.createCommandEncoder();
    const passDesc = {
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        clearValue: { r: 0.3921, g: 0.5843, b: 0.9294, a: 1 },
        storeOp: 'store'
      }]
    };
    if (!USE_SMOKE_TEST) {
      passDesc.depthStencilAttachment = {
        view: depthTex.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store'
      };
    }
    const pass = encoder.beginRenderPass(passDesc);

    pass.setPipeline(pipeline);
    if (!USE_SMOKE_TEST) pass.setBindGroup(0, bindGroup);
    if (USE_SMOKE_TEST) {
      pass.setVertexBuffer(0, smokeVbo);
      pass.draw(3);
    } else {
      pass.setVertexBuffer(0, vertexBuffer);
      pass.setIndexBuffer(indexBuffer, 'uint32');
      const count = rebuild._indexCount | 0;
      if (count > 0) pass.drawIndexed(count);
    }

    pass.end();
    device.queue.submit([encoder.finish()]);
    device.popErrorScope().then(err => {
      if (err) {
        console.error('Submit validation error:', err.message || err);
        showError('Submit error: ' + (err.message || err));
      }
    });
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
  if (document.getElementById('btnSubInc')) return; // we'll also add sliders if missing below
  const wrap = document.getElementById('controls') || (function(){
    const d = document.createElement('div');
    d.id = 'controls';
    d.style.margin = '8px 0';
    d.style.display = 'flex';
    d.style.flexWrap = 'wrap';
    d.style.gap = '12px 16px';
    d.style.alignItems = 'center';
    document.body.insertBefore(d, document.body.firstChild);
    return d;
  })();
  const dec = document.createElement('button'); dec.id='btnSubDec'; dec.textContent='− subdiv';
  const inc = document.createElement('button'); inc.id='btnSubInc'; inc.textContent='+ subdiv';
  wrap.append(dec, inc);
}

function addPhongSliders(){
  const wrap = document.getElementById('controls');
  if (!wrap) { throw new Error('controls wrapper missing'); }
  function slider(id, label, min, max, step, val){
    const group = document.createElement('label');
    group.style.display='flex'; group.style.alignItems='center'; group.style.gap='6px';
    group.style.fontFamily='system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
    const span = document.createElement('span'); span.textContent = label; span.style.minWidth='2.5rem';
    const input = document.createElement('input'); input.type='range'; input.id=id; input.min=min; input.max=max; input.step=step; input.value=String(val);
    group.append(span, input); wrap.append(group);
    return input;
  }
  const kd = slider('sKd','kd',  '0','1','0.01', 0.8);
  const ks = slider('sKs','ks',  '0','1','0.01', 0.5);
  const s  = slider('sSh','s',   '1','128','1',  32);
  const Le = slider('sLe','Le',  '0','3','0.01', 1.0);
  const La = slider('sLa','La',  '0','1','0.01', 0.15);
  return { kd, ks, s, Le, La };
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
