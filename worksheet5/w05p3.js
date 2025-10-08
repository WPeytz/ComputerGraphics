// worksheet5/w05p3.js
// 3D Models - Part 3
// Load and render a 3D model from an OBJ file with basic lighting
// NOTE: Include OBJParser.js BEFORE this file as a classic <script> (no type="module").

let device, context, pipeline;
let positionBuffer, normalBuffer, indexBuffer;
let indexCount = 0;

let uniformBuffer, bindGroup, depthTex;


// ---------- tiny mat4 helpers (column-major) ----------
const m4 = {
  identity() {
    return new Float32Array([
      1,0,0,0,
      0,1,0,0,
      0,0,1,0,
      0,0,0,1
    ]);
  },
  multiply(a,b){ // a*b
    const out = new Float32Array(16);
    for(let r=0;r<4;r++){
      for(let c=0;c<4;c++){
        out[c*4+r] =
          a[0*4+r]*b[c*4+0] +
          a[1*4+r]*b[c*4+1] +
          a[2*4+r]*b[c*4+2] +
          a[3*4+r]*b[c*4+3];
      }
    }
    return out;
  },
  translate(x,y,z){
    const m = m4.identity();
    m[12]=x; m[13]=y; m[14]=z;
    return m;
  },
  scale(x,y,z){
    const m = m4.identity();
    m[0]=x; m[5]=y; m[10]=z;
    return m;
  },
  rotateY(rad){
    const c=Math.cos(rad), s=Math.sin(rad);
    return new Float32Array([
       c,0,-s,0,
       0,1, 0,0,
       s,0, c,0,
       0,0, 0,1
    ]);
  },
  lookAt(eye, target, up){
    const ez = normalize(sub(eye, target));
    const ex = normalize(cross(up, ez));
    const ey = cross(ez, ex);
    const m = new Float32Array([
      ex[0], ey[0], ez[0], 0,
      ex[1], ey[1], ez[1], 0,
      ex[2], ey[2], ez[2], 0,
      0,     0,     0,     1,
    ]);
    const t = m4.translate(-eye[0], -eye[1], -eye[2]);
    return m4.multiply(m, t);
  },
  perspective(fovy, aspect, near, far){
    const f = 1/Math.tan(fovy/2);
    const nf = 1/(near-far);
    return new Float32Array([
      f/aspect, 0, 0,                          0,
      0,        f, 0,                          0,
      0,        0, (far+near)*nf,             -1,
      0,        0, (2*far*near)*nf,            0
    ]);
  },
};
function sub(a,b){ return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function cross(a,b){ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function normalize(v){ const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; }

// Matrices (declared after m4 so m4 is initialized)
let modelBase = m4.identity();
let angle = 0;

async function initWebGPU() {
  const canvas = document.getElementById('gfx');
  if (!canvas) throw new Error("Canvas element with id 'gfx' not found");

  // Resize canvas to device pixels for crisp rendering
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter not available');
  device = await adapter.requestDevice();

  context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  // depth
  depthTex = device.createTexture({
    size: { width: canvas.width, height: canvas.height },
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Load model + create GPU buffers
  await loadModel();

  // Uniforms buffer (mvp + model) -> 2 * 64 bytes
  uniformBuffer = device.createBuffer({
    size: 128,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Create render pipeline
  pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: device.createShaderModule({
        code: /* wgsl */`
struct Uniforms {
  mvp   : mat4x4f,
  model : mat4x4f,
};
@group(0) @binding(0) var<uniform> U : Uniforms;

struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0) normal : vec3f,
};

@vertex
fn vs_main(
  @location(0) position : vec3f,
  @location(1) normal   : vec3f
) -> VSOut {
  var out : VSOut;
  out.pos = U.mvp * vec4f(position, 1.0);
  // transform normal by model (uniform scale/rotation assumed)
  out.normal = (U.model * vec4f(normal, 0.0)).xyz;
  return out;
}

@fragment
fn fs_main(@location(0) normal : vec3f) -> @location(0) vec4f {
  let n = normalize(normal);
  let lightDir = normalize(vec3f(0.5, 0.7, -1.0));
  let diff = max(dot(n, lightDir), 0.0);
  return vec4f(vec3f(0.2) + diff * vec3f(0.8), 1.0);
}
        `,
      }),
      entryPoint: 'vs_main',
      buffers: [
        // Positions come from vec4 arrays (stride 16). We read only xyz.
        { arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
        // Normals also as vec4 arrays (stride 16). We read only xyz.
        { arrayStride: 16, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] },
      ],
    },
    fragment: {
      module: device.createShaderModule({
        code: /* wgsl */`
@fragment
fn fs_main(@location(0) normal : vec3f) -> @location(0) vec4f {
  let n = normalize(normal);
  let lightDir = normalize(vec3f(0.5, 0.7, -1.0));
  let diff = max(dot(n, lightDir), 0.0);
  return vec4f(vec3f(0.2) + diff * vec3f(0.8), 1.0);
}
        `,
      }),
      entryPoint: 'fs_main',
      targets: [{ format }],
    },
    primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less',
    },
  });

  // Bind group
  bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  requestAnimationFrame(drawFrame);
}

async function loadModel() {
  // Ensure parser exists (provided by classic script OBJParser.js)
  const readOBJFile = globalThis.readOBJFile;
  if (typeof readOBJFile !== 'function') {
    throw new Error('OBJ parser not found. Make sure OBJParser.js is included BEFORE w05p3.js');
  }

  // Load + parse OBJ. The course helper returns an object with
  // vertices, normals, colors (all Float32Array, 4 comps/vertex) and indices (Uint32Array).
  const info = await readOBJFile('./model.obj', 1.0, false);
  if (!info) throw new Error('Failed to load ./model.obj');

  let positions = info.vertices;
  let normals   = info.normals;
  const indices = info.indices;

  if (!(positions instanceof Float32Array) || positions.length % 4 !== 0) {
    throw new Error('Unexpected vertex format from OBJ parser');
  }
  if (!(indices instanceof Uint32Array)) {
    throw new Error('Unexpected index format from OBJ parser');
  }

  // If normals are missing or wrong size, compute per-vertex normals
  if (!(normals instanceof Float32Array) || normals.length !== positions.length) {
    normals = computeVertexNormalsFromVec4(positions, indices);
  }

  // Fit to view: center and scale to unit radius
  const { center, radius } = computeBoundingSphereFromVec4(positions);
  const s = 1.0 / radius;
  // Model base = S * T(-center)
  modelBase = m4.multiply(m4.scale(s, s, s), m4.translate(-center[0], -center[1], -center[2]));

  // Create GPU buffers (note the 16-byte stride expectation in the pipeline)
  positionBuffer = device.createBuffer({
    size: positions.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(positionBuffer.getMappedRange()).set(positions);
  positionBuffer.unmap();

  normalBuffer = device.createBuffer({
    size: normals.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(normalBuffer.getMappedRange()).set(normals);
  normalBuffer.unmap();

  indexBuffer = device.createBuffer({
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(indexBuffer.getMappedRange()).set(indices);
  indexBuffer.unmap();

  indexCount = indices.length;
}

// Compute vertex normals when positions are vec4 (stride 4 floats) and indices are Uint32.
function computeVertexNormalsFromVec4(positions, indices) {
  const vCount = positions.length / 4;
  const normals = new Float32Array(vCount * 4);

  // accumulate face normals
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i + 0] * 4;
    const b = indices[i + 1] * 4;
    const c = indices[i + 2] * 4;

    const ax = positions[a + 0], ay = positions[a + 1], az = positions[a + 2];
    const bx = positions[b + 0], by = positions[b + 1], bz = positions[b + 2];
    const cx = positions[c + 0], cy = positions[c + 1], cz = positions[c + 2];

    const abx = bx - ax, aby = by - ay, abz = bz - az;
    const acx = cx - ax, acy = cy - ay, acz = cz - az;

    // face normal = normalize(cross(ab, ac))
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;

    normals[a + 0] += nx; normals[a + 1] += ny; normals[a + 2] += nz;
    normals[b + 0] += nx; normals[b + 1] += ny; normals[b + 2] += nz;
    normals[c + 0] += nx; normals[c + 1] += ny; normals[c + 2] += nz;
  }

  // normalize and set w component to 0
  for (let v = 0; v < vCount; v++) {
    const i = v * 4;
    const nx = normals[i + 0], ny = normals[i + 1], nz = normals[i + 2];
    const len = Math.max(1e-8, Math.hypot(nx, ny, nz));
    normals[i + 0] = nx / len;
    normals[i + 1] = ny / len;
    normals[i + 2] = nz / len;
    normals[i + 3] = 0.0;
  }

  return normals;
}

function computeBoundingSphereFromVec4(positions){
  let minX= Infinity, minY= Infinity, minZ= Infinity;
  let maxX=-Infinity, maxY=-Infinity, maxZ=-Infinity;
  for(let i=0;i<positions.length;i+=4){
    const x=positions[i], y=positions[i+1], z=positions[i+2];
    if(x<minX)minX=x; if(y<minY)minY=y; if(z<minZ)minZ=z;
    if(x>maxX)maxX=x; if(y>maxY)maxY=y; if(z>maxZ)maxZ=z;
  }
  const cx = 0.5*(minX+maxX), cy = 0.5*(minY+maxY), cz = 0.5*(minZ+maxZ);
  let r = 0;
  for(let i=0;i<positions.length;i+=4){
    const dx=positions[i]-cx, dy=positions[i+1]-cy, dz=positions[i+2]-cz;
    r = Math.max(r, Math.hypot(dx,dy,dz));
  }
  return { center:[cx,cy,cz], radius:r || 1 };
}

function drawFrame() {
  const canvas = context.canvas;
  // Camera
  angle += 0.4 * (1/60); // slow spin
  const model = m4.multiply(m4.rotateY(angle), modelBase);
  const eye = [0, 0, 3.0];
  const target = [0, 0, 0];
  const up = [0, 1, 0];
  const view = m4.lookAt(eye, target, up);
  const proj = m4.perspective((45*Math.PI)/180, canvas.width/canvas.height, 0.01, 100.0);
  const mvp = m4.multiply(m4.multiply(proj, view), model);

  // write uniforms
  const uData = new Float32Array(32); // 2 * 16
  uData.set(mvp, 0);
  uData.set(model, 16);
  device.queue.writeBuffer(uniformBuffer, 0, uData.buffer);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: 'clear',
      clearValue: { r: 0.92, g: 0.96, b: 1.0, a: 1.0 },
      storeOp: 'store',
    }],
    depthStencilAttachment: {
      view: depthTex.createView(),
      depthLoadOp: 'clear',
      depthClearValue: 1.0,
      depthStoreOp: 'store',
    },
  });

  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.setVertexBuffer(0, positionBuffer);
  pass.setVertexBuffer(1, normalBuffer);
  pass.setIndexBuffer(indexBuffer, 'uint32');
  pass.drawIndexed(indexCount);
  pass.end();

  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(drawFrame);
}

initWebGPU();