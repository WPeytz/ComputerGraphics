// Worksheet 9 Part 1: Scene with teapot jumping and textured ground with animated light
// Combines teapot from Worksheet 5 Part 3 and ground quad from Worksheet 8 Part 3

let device, context;
let teapotPipeline, groundPipeline;
let teapotPositionBuffer, teapotNormalBuffer, teapotIndexBuffer;
let groundVertexBuffer;
let teapotIndexCount = 0;
let teapotUniformBuffer, teapotBindGroup;
let groundUniformBuffer, groundBindGroup;
let depthTex;
let marbleTexture, sampler;

// Animation state
let animationEnabled = true;
let startTime = Date.now();

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

// Teapot model base transformation
let teapotModelBase = m4.identity();

async function initWebGPU() {
  const canvas = document.getElementById('my-canvas');
  if (!canvas) throw new Error("Canvas element with id 'my-canvas' not found");

  canvas.width = 1024;
  canvas.height = 512;

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter not available');
  device = await adapter.requestDevice();

  context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  // Create depth texture
  depthTex = device.createTexture({
    size: { width: canvas.width, height: canvas.height },
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Load teapot model + create GPU buffers
  await loadTeapot();

  // Create ground quad vertex buffer
  createGroundQuad();

  // Load marble texture
  marbleTexture = await loadTexture(device, 'xamp23.png');

  // Create sampler
  sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  // Create uniform buffers
  teapotUniformBuffer = device.createBuffer({
    size: 128, // 2 * mat4x4f (mvp + model)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  groundUniformBuffer = device.createBuffer({
    size: 256, // projection + view + model + visibility
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Create teapot render pipeline (for 3D model with lighting)
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

  // Create ground pipeline (for textured quad)
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
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: 'filtering' }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'float' }
      }
    ]
  });

  groundPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [groundBindGroupLayout]
    }),
    vertex: {
      module: groundShaderModule,
      entryPoint: 'vs_main',
      buffers: [{
        arrayStride: 20,
        attributes: [
          {
            shaderLocation: 0,
            offset: 0,
            format: 'float32x3'
          },
          {
            shaderLocation: 1,
            offset: 12,
            format: 'float32x2'
          }
        ]
      }]
    },
    fragment: {
      module: groundShaderModule,
      entryPoint: 'fs_main',
      targets: [{
        format: format
      }]
    },
    primitive: {
      topology: 'triangle-list',
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus',
    }
  });

  // Create bind groups
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

  // Set up button
  const toggleButton = document.getElementById('toggle-animation');
  if (toggleButton) {
    toggleButton.addEventListener('click', () => {
      animationEnabled = !animationEnabled;
      toggleButton.textContent = animationEnabled ? 'Pause Animation' : 'Resume Animation';
    });
  }

  requestAnimationFrame(drawFrame);
}

async function loadTeapot() {
  const readOBJFile = globalThis.readOBJFile;
  if (typeof readOBJFile !== 'function') {
    throw new Error('OBJ parser not found. Make sure OBJParser.js is included BEFORE w09p1.js');
  }

  // Load teapot.obj from teapot subdirectory
  const info = await readOBJFile('./teapot/teapot.obj', 1.0, false);
  if (!info) throw new Error('Failed to load ./teapot/teapot.obj');

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

  // Scale to quarter size and translate by (0, -1, -3)
  // The assignment says: scale to 1/4 and translate by (0, -1, -3)
  const s = 0.25;
  const translation = m4.translate(0, -1, -3);
  const scaling = m4.scale(s, s, s);
  teapotModelBase = m4.multiply(translation, scaling);

  // Create GPU buffers
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
  // Ground quad: y = -1, x in [-2, 2], z in [-1, -5]
  const groundVertices = new Float32Array([
    -2.0, -1.0, -1.0,        0.0, 0.0,
     2.0, -1.0, -1.0,        1.0, 0.0,
     2.0, -1.0, -5.0,        1.0, 1.0,
    -2.0, -1.0, -1.0,        0.0, 0.0,
     2.0, -1.0, -5.0,        1.0, 1.0,
    -2.0, -1.0, -5.0,        0.0, 1.0,
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

    const ax = positions[a + 0], ay = positions[a + 1], az = positions[a + 2];
    const bx = positions[b + 0], by = positions[b + 1], bz = positions[b + 2];
    const cx = positions[c + 0], cy = positions[c + 1], cz = positions[c + 2];

    const abx = bx - ax, aby = by - ay, abz = bz - az;
    const acx = cx - ax, acy = cy - ay, acz = cz - az;

    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;

    normals[a + 0] += nx; normals[a + 1] += ny; normals[a + 2] += nz;
    normals[b + 0] += nx; normals[b + 1] += ny; normals[b + 2] += nz;
    normals[c + 0] += nx; normals[c + 1] += ny; normals[c + 2] += nz;
  }

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

  // Calculate teapot position with animation (y from -1 to 0.5)
  let yOffset = 0;
  if (animationEnabled) {
    const currentTime = (Date.now() - startTime) / 1000;
    // Oscillate between -1 and 0.5 (range of 1.5)
    yOffset = -1 + 1.5 * (0.5 + 0.5 * Math.sin(currentTime * Math.PI * 2 / 2)); // 2 second period
  }

  // Teapot model matrix: add yOffset translation
  const teapotTranslation = m4.translate(0, yOffset, 0);
  const teapotModel = m4.multiply(teapotTranslation, teapotModelBase);

  // Camera setup
  const eye = [0, 0, 3.0];
  const target = [0, 0, -3];
  const up = [0, 1, 0];
  const view = m4.lookAt(eye, target, up);
  const proj = m4.perspective((45*Math.PI)/180, canvas.width/canvas.height, 0.01, 100.0);

  // Teapot MVP
  const teapotMVP = m4.multiply(m4.multiply(proj, view), teapotModel);

  // Update teapot uniforms
  const teapotUniforms = new Float32Array(32);
  teapotUniforms.set(teapotMVP, 0);
  teapotUniforms.set(teapotModel, 16);
  device.queue.writeBuffer(teapotUniformBuffer, 0, teapotUniforms.buffer);

  // Ground uniforms (identity model matrix)
  const identityMatrix = m4.identity();
  const groundUniforms = new Float32Array([
    ...proj,
    ...view,
    ...identityMatrix,
    1.0, 0, 0, 0
  ]);
  device.queue.writeBuffer(groundUniformBuffer, 0, groundUniforms);

  // Render
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: 'clear',
      clearValue: { r: 0.0, g: 0.0, b: 1.0, a: 1.0 },
      storeOp: 'store',
    }],
    depthStencilAttachment: {
      view: depthTex.createView(),
      depthLoadOp: 'clear',
      depthClearValue: 1.0,
      depthStoreOp: 'store',
    },
  });

  // Draw ground quad
  pass.setPipeline(groundPipeline);
  pass.setBindGroup(0, groundBindGroup);
  pass.setVertexBuffer(0, groundVertexBuffer);
  pass.draw(6);

  // Draw teapot
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
