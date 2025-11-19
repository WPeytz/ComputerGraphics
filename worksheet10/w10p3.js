// Worksheet 10 Part 3: Dolly and Panning
// Implements orbiting, dollying, and panning with mouse buttons

let device, context;
let teapotPipeline, groundPipeline;
let teapotPositionBuffer, teapotNormalBuffer, teapotIndexBuffer;
let groundVertexBuffer;
let teapotIndexCount = 0;
let teapotUniformBuffer, teapotBindGroup;
let groundUniformBuffer, groundBindGroup;
let depthTex;
let marbleTexture, sampler;

// Camera parameters
let orbitRadius = 6;           // Distance from eye to look-at point
let panOffset = [0, 0, 0];     // XY displacement of look-at point in world space

// Base scene center (before panning)
const baseSceneCenter = [0, -0.5, -3];

// Quaternion for accumulated rotation
let rotationQuat = new Quaternion();

// Interaction mode: 'orbit', 'dolly', or 'pan'
let interactionMode = 'orbit';

// Mouse tracking
let isDragging = false;
let lastMouseX = 0;
let lastMouseY = 0;
let activeButton = 0; // Which mouse button is pressed

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
  multiply(a,b){
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
function vec3(x, y, z) { return [x, y, z]; }
function add(a,b){ return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function scale(v, s){ return [v[0]*s, v[1]*s, v[2]*s]; }

// Teapot model base transformation
let teapotModelBase = m4.identity();

// Project mouse coordinates to trackball surface
function projectToTrackball(x, y, canvas) {
  const nx = (2.0 * x / canvas.width) - 1.0;
  const ny = 1.0 - (2.0 * y / canvas.height);

  const r = 0.8;
  const d = Math.sqrt(nx * nx + ny * ny);

  let z;
  if (d < r / Math.sqrt(2)) {
    z = Math.sqrt(r * r - d * d);
  } else {
    z = (r * r / 2) / d;
  }

  return normalize([nx, ny, z]);
}

// Get current scene center (base + pan offset)
function getSceneCenter() {
  return add(baseSceneCenter, panOffset);
}

// Calculate eye position from rotation quaternion
function getEyePosition() {
  const initialPos = [0, 0, orbitRadius];
  const rotatedPos = rotationQuat.apply(initialPos);
  const center = getSceneCenter();

  return [
    center[0] + rotatedPos[0],
    center[1] + rotatedPos[1],
    center[2] + rotatedPos[2]
  ];
}

// Get up vector from rotation quaternion
function getUpVector() {
  const initialUp = [0, 1, 0];
  return rotationQuat.apply(initialUp);
}

// Get right vector from rotation quaternion
function getRightVector() {
  const initialRight = [1, 0, 0];
  return rotationQuat.apply(initialRight);
}

async function initWebGPU() {
  const canvas = document.getElementById('my-canvas');
  if (!canvas) throw new Error("Canvas element with id 'my-canvas' not found");

  canvas.width = 800;
  canvas.height = 600;

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter not available');
  device = await adapter.requestDevice();

  context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  depthTex = device.createTexture({
    size: { width: canvas.width, height: canvas.height },
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  await loadTeapot();
  createGroundQuad();
  marbleTexture = await loadTexture(device, 'xamp23.png');

  sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  teapotUniformBuffer = device.createBuffer({
    size: 128,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  groundUniformBuffer = device.createBuffer({
    size: 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Create teapot pipeline
  teapotPipeline = device.createRenderPipeline({
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
  out.normal = (U.model * vec4f(normal, 0.0)).xyz;
  return out;
}
        `,
      }),
      entryPoint: 'vs_main',
      buffers: [
        { arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
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

  // Create ground pipeline
  const groundShaderCode = `
    struct Uniforms {
      projection: mat4x4<f32>,
      view: mat4x4<f32>,
      model: mat4x4<f32>,
      visibility: f32,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var mySampler: sampler;
    @group(0) @binding(2) var myTexture: texture_2d<f32>;

    struct VertexInput {
      @location(0) position: vec3<f32>,
      @location(1) texCoord: vec2<f32>,
    }

    struct VertexOutput {
      @builtin(position) position: vec4<f32>,
      @location(0) texCoord: vec2<f32>,
    }

    @vertex
    fn vs_main(input: VertexInput) -> VertexOutput {
      var output: VertexOutput;
      output.position = uniforms.projection * uniforms.view * uniforms.model * vec4<f32>(input.position, 1.0);
      output.texCoord = input.texCoord;
      return output;
    }

    @fragment
    fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
      let texColor = textureSample(myTexture, mySampler, input.texCoord);
      return vec4<f32>(texColor.rgb * uniforms.visibility, texColor.a);
    }
  `;

  const groundShaderModule = device.createShaderModule({ code: groundShaderCode });

  const groundBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }
    ]
  });

  groundPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [groundBindGroupLayout] }),
    vertex: {
      module: groundShaderModule,
      entryPoint: 'vs_main',
      buffers: [{
        arrayStride: 20,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x2' }
        ]
      }]
    },
    fragment: {
      module: groundShaderModule,
      entryPoint: 'fs_main',
      targets: [{ format: format }]
    },
    primitive: { topology: 'triangle-list' },
    depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus' }
  });

  teapotBindGroup = device.createBindGroup({
    layout: teapotPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: teapotUniformBuffer } }],
  });

  groundBindGroup = device.createBindGroup({
    layout: groundBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: groundUniformBuffer } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: marbleTexture.createView() }
    ]
  });

  setupMouseEvents(canvas);
  setupModeButtons();

  requestAnimationFrame(drawFrame);
}

function setupModeButtons() {
  const btnOrbit = document.getElementById('btn-orbit');
  const btnDolly = document.getElementById('btn-dolly');
  const btnPan = document.getElementById('btn-pan');

  function setMode(mode) {
    interactionMode = mode;
    btnOrbit.classList.toggle('active', mode === 'orbit');
    btnDolly.classList.toggle('active', mode === 'dolly');
    btnPan.classList.toggle('active', mode === 'pan');
  }

  btnOrbit.addEventListener('click', () => setMode('orbit'));
  btnDolly.addEventListener('click', () => setMode('dolly'));
  btnPan.addEventListener('click', () => setMode('pan'));
}

function setupMouseEvents(canvas) {
  let lastTrackballPos = null;

  // Prevent context menu on right click
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  canvas.addEventListener('mousedown', (e) => {
    isDragging = true;
    activeButton = e.button;
    lastMouseX = e.offsetX;
    lastMouseY = e.offsetY;

    // For orbit mode, project to trackball
    if (getEffectiveMode(e.button) === 'orbit') {
      lastTrackballPos = projectToTrackball(e.offsetX, e.offsetY, canvas);
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isDragging) return;

    const deltaX = e.offsetX - lastMouseX;
    const deltaY = e.offsetY - lastMouseY;

    const mode = getEffectiveMode(activeButton);

    if (mode === 'orbit') {
      // Quaternion-based orbit
      const currentTrackballPos = projectToTrackball(e.offsetX, e.offsetY, canvas);

      if (lastTrackballPos) {
        const rotQuat = new Quaternion();
        rotQuat.make_rot_vec2vec(lastTrackballPos, currentTrackballPos);
        rotationQuat = rotQuat.multiply(rotationQuat);
      }

      lastTrackballPos = currentTrackballPos;
    } else if (mode === 'dolly') {
      // Dolly: move camera closer/further based on Y movement
      const dollySpeed = 0.02;
      orbitRadius += deltaY * dollySpeed;
      orbitRadius = Math.max(1, Math.min(50, orbitRadius)); // Clamp
    } else if (mode === 'pan') {
      // Pan: move look-at point along image plane
      const panSpeed = 0.005;

      // Get camera right and up vectors in world space
      const right = getRightVector();
      const up = getUpVector();

      // Calculate pan displacement
      const panX = scale(right, -deltaX * panSpeed);
      const panY = scale(up, deltaY * panSpeed);

      panOffset = add(panOffset, panX);
      panOffset = add(panOffset, panY);
    }

    lastMouseX = e.offsetX;
    lastMouseY = e.offsetY;
  });

  canvas.addEventListener('mouseup', () => {
    isDragging = false;
    lastTrackballPos = null;
  });

  canvas.addEventListener('mouseleave', () => {
    isDragging = false;
    lastTrackballPos = null;
  });

  // Mouse wheel for dolly
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const dollySpeed = 0.01;
    orbitRadius += e.deltaY * dollySpeed;
    orbitRadius = Math.max(1, Math.min(50, orbitRadius));
  });
}

// Determine effective mode based on mouse button
// Button 0 = left (uses selected mode), 1 = middle (pan), 2 = right (dolly)
function getEffectiveMode(button) {
  if (button === 1) return 'pan';
  if (button === 2) return 'dolly';
  return interactionMode; // Left button uses selected mode
}

async function loadTeapot() {
  const readOBJFile = globalThis.readOBJFile;
  if (typeof readOBJFile !== 'function') {
    throw new Error('OBJ parser not found');
  }

  const info = await readOBJFile('./teapot/teapot.obj', 1.0, false);
  if (!info) throw new Error('Failed to load teapot.obj');

  let positions = info.vertices;
  let normals = info.normals;
  const indices = info.indices;

  if (!(normals instanceof Float32Array) || normals.length !== positions.length) {
    normals = computeVertexNormalsFromVec4(positions, indices);
  }

  const s = 0.25;
  const translation = m4.translate(0, -1, -3);
  const scaling = m4.scale(s, s, s);
  teapotModelBase = m4.multiply(translation, scaling);

  teapotPositionBuffer = device.createBuffer({
    size: positions.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(teapotPositionBuffer.getMappedRange()).set(positions);
  teapotPositionBuffer.unmap();

  teapotNormalBuffer = device.createBuffer({
    size: normals.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(teapotNormalBuffer.getMappedRange()).set(normals);
  teapotNormalBuffer.unmap();

  teapotIndexBuffer = device.createBuffer({
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(teapotIndexBuffer.getMappedRange()).set(indices);
  teapotIndexBuffer.unmap();

  teapotIndexCount = indices.length;
}

function createGroundQuad() {
  const groundVertices = new Float32Array([
    -2.0, -1.0, -1.0,  0.0, 0.0,
     2.0, -1.0, -1.0,  1.0, 0.0,
     2.0, -1.0, -5.0,  1.0, 1.0,
    -2.0, -1.0, -1.0,  0.0, 0.0,
     2.0, -1.0, -5.0,  1.0, 1.0,
    -2.0, -1.0, -5.0,  0.0, 1.0,
  ]);

  groundVertexBuffer = device.createBuffer({
    size: groundVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(groundVertexBuffer, 0, groundVertices);
}

function computeVertexNormalsFromVec4(positions, indices) {
  const vCount = positions.length / 4;
  const normals = new Float32Array(vCount * 4);

  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i + 0] * 4;
    const b = indices[i + 1] * 4;
    const c = indices[i + 2] * 4;

    const ax = positions[a], ay = positions[a+1], az = positions[a+2];
    const bx = positions[b], by = positions[b+1], bz = positions[b+2];
    const cx = positions[c], cy = positions[c+1], cz = positions[c+2];

    const abx = bx-ax, aby = by-ay, abz = bz-az;
    const acx = cx-ax, acy = cy-ay, acz = cz-az;

    const nx = aby*acz - abz*acy;
    const ny = abz*acx - abx*acz;
    const nz = abx*acy - aby*acx;

    normals[a] += nx; normals[a+1] += ny; normals[a+2] += nz;
    normals[b] += nx; normals[b+1] += ny; normals[b+2] += nz;
    normals[c] += nx; normals[c+1] += ny; normals[c+2] += nz;
  }

  for (let v = 0; v < vCount; v++) {
    const i = v * 4;
    const len = Math.max(1e-8, Math.hypot(normals[i], normals[i+1], normals[i+2]));
    normals[i] /= len; normals[i+1] /= len; normals[i+2] /= len;
  }

  return normals;
}

async function loadTexture(device, url) {
  const response = await fetch(url);
  const blob = await response.blob();
  const imageBitmap = await createImageBitmap(blob);

  const texture = device.createTexture({
    size: [imageBitmap.width, imageBitmap.height, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });

  device.queue.copyExternalImageToTexture(
    { source: imageBitmap },
    { texture: texture },
    [imageBitmap.width, imageBitmap.height, 1]
  );

  return texture;
}

function drawFrame() {
  const canvas = context.canvas;

  const eye = getEyePosition();
  const target = getSceneCenter();
  const up = getUpVector();

  const view = m4.lookAt(eye, target, up);
  const proj = m4.perspective((45*Math.PI)/180, canvas.width/canvas.height, 0.01, 100.0);

  const teapotMVP = m4.multiply(m4.multiply(proj, view), teapotModelBase);

  const teapotUniforms = new Float32Array(32);
  teapotUniforms.set(teapotMVP, 0);
  teapotUniforms.set(teapotModelBase, 16);
  device.queue.writeBuffer(teapotUniformBuffer, 0, teapotUniforms.buffer);

  const identityMatrix = m4.identity();
  const groundUniforms = new Float32Array([
    ...proj, ...view, ...identityMatrix, 1.0, 0, 0, 0
  ]);
  device.queue.writeBuffer(groundUniformBuffer, 0, groundUniforms);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: 'clear',
      clearValue: { r: 0.1, g: 0.1, b: 0.2, a: 1.0 },
      storeOp: 'store',
    }],
    depthStencilAttachment: {
      view: depthTex.createView(),
      depthLoadOp: 'clear',
      depthClearValue: 1.0,
      depthStoreOp: 'store',
    },
  });

  pass.setPipeline(groundPipeline);
  pass.setBindGroup(0, groundBindGroup);
  pass.setVertexBuffer(0, groundVertexBuffer);
  pass.draw(6);

  pass.setPipeline(teapotPipeline);
  pass.setBindGroup(0, teapotBindGroup);
  pass.setVertexBuffer(0, teapotPositionBuffer);
  pass.setVertexBuffer(1, teapotNormalBuffer);
  pass.setIndexBuffer(teapotIndexBuffer, 'uint32');
  pass.drawIndexed(teapotIndexCount);

  pass.end();

  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(drawFrame);
}

initWebGPU();
