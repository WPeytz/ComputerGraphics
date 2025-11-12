// Worksheet 9 Part 3-4: Shadow Mapping
// Replaces projection shadows from Part 2 with proper shadow mapping

let device, context;
let teapotPipeline, groundPipeline, shadowMapPipeline;
let teapotPositionBuffer, teapotNormalBuffer, teapotIndexBuffer;
let groundVertexBuffer;
let teapotIndexCount = 0;
let teapotUniformBuffer, teapotBindGroup;
let groundUniformBuffer, groundBindGroup;
let shadowMapTeapotUniformBuffer, shadowMapTeapotBindGroup;
let shadowMapGroundUniformBuffer, shadowMapGroundBindGroup;
let depthTex, shadowDepthTexture;
let marbleTexture, sampler, shadowSampler;

// Animation state
let animationEnabled = true;
let lightAnimationEnabled = true;
let startTime = Date.now();

// Shadow map size
const SHADOW_MAP_SIZE = 2048;

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
  ortho(left, right, bottom, top, near, far){
    const lr = 1 / (left - right);
    const bt = 1 / (bottom - top);
    const nf = 1 / (near - far);
    return new Float32Array([
      -2 * lr, 0, 0, 0,
      0, -2 * bt, 0, 0,
      0, 0, 2 * nf, 0,
      (left + right) * lr, (top + bottom) * bt, (far + near) * nf, 1
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

  // Create depth texture for main rendering
  depthTex = device.createTexture({
    size: { width: canvas.width, height: canvas.height },
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Create depth texture for shadow map
  shadowDepthTexture = device.createTexture({
    size: { width: SHADOW_MAP_SIZE, height: SHADOW_MAP_SIZE },
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  // Load teapot model + create GPU buffers
  await loadTeapot();

  // Create ground quad vertex buffer
  createGroundQuad();

  // Load marble texture
  marbleTexture = await loadTexture(device, 'xamp23.png');

  // Create sampler for textures
  sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  // Create comparison sampler for shadow mapping
  shadowSampler = device.createSampler({
    compare: 'less',
    magFilter: 'linear',
    minFilter: 'linear',
  });

  // Create uniform buffers
  teapotUniformBuffer = device.createBuffer({
    size: 256, // mvp, model, lightMVP, lightPos
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  groundUniformBuffer = device.createBuffer({
    size: 320,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  shadowMapTeapotUniformBuffer = device.createBuffer({
    size: 64, // lightMVP
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  shadowMapGroundUniformBuffer = device.createBuffer({
    size: 64, // lightMVP
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Shadow map pipeline (depth-only rendering from light's POV)
  const shadowMapShaderCode = `
    struct Uniforms {
      lightMVP: mat4x4<f32>,
    }
    @group(0) @binding(0) var<uniform> uniforms: Uniforms;

    @vertex
    fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
      return uniforms.lightMVP * vec4<f32>(position, 1.0);
    }
  `;

  const shadowMapShaderModule = device.createShaderModule({ code: shadowMapShaderCode });

  const shadowMapBindGroupLayout = device.createBindGroupLayout({
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: 'uniform' }
    }]
  });

  shadowMapPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [shadowMapBindGroupLayout]
    }),
    vertex: {
      module: shadowMapShaderModule,
      entryPoint: 'vs_main',
      buffers: [{
        arrayStride: 16, // vec4 for teapot
        attributes: [{
          shaderLocation: 0,
          offset: 0,
          format: 'float32x3'
        }]
      }]
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'back',
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth32float',
    }
  });

  // Create teapot render pipeline with shadow mapping
  const teapotShaderCode = `
diagnostic(off, derivative_uniformity);

struct Uniforms {
  mvp      : mat4x4f,
  model    : mat4x4f,
  lightMVP : mat4x4f,
  lightPos : vec3f,
};
@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var shadowMap: texture_depth_2d;
@group(0) @binding(2) var shadowSampler: sampler_comparison;

struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0) normal : vec3f,
  @location(1) worldPos : vec3f,
  @location(2) shadowPos : vec3f,
};

@vertex
fn vs_main(
  @location(0) position : vec3f,
  @location(1) normal   : vec3f
) -> VSOut {
  var out : VSOut;
  let worldPos = U.model * vec4f(position, 1.0);
  out.pos = U.mvp * vec4f(position, 1.0);
  out.normal = (U.model * vec4f(normal, 0.0)).xyz;
  out.worldPos = worldPos.xyz;

  // Transform to light space
  let lightSpacePos = U.lightMVP * worldPos;
  out.shadowPos = lightSpacePos.xyz / lightSpacePos.w;

  return out;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4f {
  let n = normalize(input.normal);
  let lightDir = normalize(U.lightPos - input.worldPos);
  let diff = max(dot(n, lightDir), 0.0);

  // Shadow mapping
  let shadowCoord = vec3f(
    input.shadowPos.x * 0.5 + 0.5,
    -input.shadowPos.y * 0.5 + 0.5,
    input.shadowPos.z
  );

  var shadow = 0.0;
  if (shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 &&
      shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0) {
    shadow = textureSampleCompare(shadowMap, shadowSampler, shadowCoord.xy, shadowCoord.z);
  } else {
    shadow = 1.0;
  }

  // Ambient + diffuse with shadow
  let ambient = 0.3;
  let lighting = ambient + (1.0 - ambient) * diff * shadow;

  return vec4f(vec3f(0.7, 0.7, 0.7) * lighting, 1.0);
}
  `;

  const teapotShaderModule = device.createShaderModule({ code: teapotShaderCode });

  teapotPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: teapotShaderModule,
      entryPoint: 'vs_main',
      buffers: [
        { arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
        { arrayStride: 16, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] },
      ],
    },
    fragment: {
      module: teapotShaderModule,
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

  // Create ground pipeline with shadow mapping
  const groundShaderCode = `
    diagnostic(off, derivative_uniformity);

    struct Uniforms {
      projection: mat4x4<f32>,
      view: mat4x4<f32>,
      model: mat4x4<f32>,
      lightMVP: mat4x4<f32>,
      lightPos: vec3<f32>,
    }

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var mySampler: sampler;
    @group(0) @binding(2) var myTexture: texture_2d<f32>;
    @group(0) @binding(3) var shadowMap: texture_depth_2d;
    @group(0) @binding(4) var shadowSampler: sampler_comparison;

    struct VertexInput {
      @location(0) position: vec3<f32>,
      @location(1) texCoord: vec2<f32>,
    }

    struct VertexOutput {
      @builtin(position) position: vec4<f32>,
      @location(0) texCoord: vec2<f32>,
      @location(1) worldPos: vec3<f32>,
      @location(2) shadowPos: vec3<f32>,
    }

    @vertex
    fn vs_main(input: VertexInput) -> VertexOutput {
      var output: VertexOutput;
      let worldPos = uniforms.model * vec4<f32>(input.position, 1.0);
      output.position = uniforms.projection * uniforms.view * worldPos;
      output.texCoord = input.texCoord;
      output.worldPos = worldPos.xyz;

      // Transform to light space
      let lightSpacePos = uniforms.lightMVP * worldPos;
      output.shadowPos = lightSpacePos.xyz / lightSpacePos.w;

      return output;
    }

    @fragment
    fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
      let texColor = textureSample(myTexture, mySampler, input.texCoord);

      // Shadow mapping
      let shadowCoord = vec3<f32>(
        input.shadowPos.x * 0.5 + 0.5,
        -input.shadowPos.y * 0.5 + 0.5,
        input.shadowPos.z
      );

      var shadow = 0.0;
      if (shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 &&
          shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0) {
        shadow = textureSampleCompare(shadowMap, shadowSampler, shadowCoord.xy, shadowCoord.z);
      } else {
        shadow = 1.0;
      }

      // Ambient + shadow
      let ambient = 0.3;
      let lighting = ambient + (1.0 - ambient) * shadow;

      return vec4<f32>(texColor.rgb * lighting, texColor.a);
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
      },
      {
        binding: 3,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'depth' }
      },
      {
        binding: 4,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: 'comparison' }
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
    entries: [
      { binding: 0, resource: { buffer: teapotUniformBuffer } },
      { binding: 1, resource: shadowDepthTexture.createView() },
      { binding: 2, resource: shadowSampler }
    ],
  });

  groundBindGroup = device.createBindGroup({
    layout: groundBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: groundUniformBuffer } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: marbleTexture.createView() },
      { binding: 3, resource: shadowDepthTexture.createView() },
      { binding: 4, resource: shadowSampler }
    ]
  });

  shadowMapTeapotBindGroup = device.createBindGroup({
    layout: shadowMapBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: shadowMapTeapotUniformBuffer } }
    ]
  });

  shadowMapGroundBindGroup = device.createBindGroup({
    layout: shadowMapBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: shadowMapGroundUniformBuffer } }
    ]
  });

  // Set up buttons
  const toggleButton = document.getElementById('toggle-animation');
  if (toggleButton) {
    toggleButton.addEventListener('click', () => {
      animationEnabled = !animationEnabled;
      toggleButton.textContent = animationEnabled ? 'Pause Animation' : 'Resume Animation';
    });
  }

  const toggleLightButton = document.getElementById('toggle-light');
  if (toggleLightButton) {
    toggleLightButton.addEventListener('click', () => {
      lightAnimationEnabled = !lightAnimationEnabled;
      toggleLightButton.textContent = lightAnimationEnabled ? 'Pause Light' : 'Resume Light';
    });
  }

  requestAnimationFrame(drawFrame);
}

async function loadTeapot() {
  const readOBJFile = globalThis.readOBJFile;
  if (typeof readOBJFile !== 'function') {
    throw new Error('OBJ parser not found. Make sure OBJParser.js is included BEFORE w09p3.js');
  }

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
    yOffset = -1 + 1.5 * (0.5 + 0.5 * Math.sin(currentTime * Math.PI * 2 / 2));
  }

  // Calculate animated light position (circling)
  let lightX = 0, lightY = 2, lightZ = -2;
  if (lightAnimationEnabled) {
    const currentTime = (Date.now() - startTime) / 1000;
    lightX = 0 + 2 * Math.cos(currentTime);
    lightY = 2;
    lightZ = -2 + 2 * Math.sin(currentTime);
  }

  // Teapot model matrix
  const teapotTranslation = m4.translate(0, yOffset, 0);
  const teapotModel = m4.multiply(teapotTranslation, teapotModelBase);

  // Camera setup
  const eye = [0, 0, 3.0];
  const target = [0, 0, -3];
  const up = [0, 1, 0];
  const view = m4.lookAt(eye, target, up);
  const proj = m4.perspective((45*Math.PI)/180, canvas.width/canvas.height, 0.01, 100.0);

  // Light view and projection matrices
  const lightView = m4.lookAt([lightX, lightY, lightZ], [0, -1, -3], [0, 1, 0]);
  const lightProj = m4.ortho(-4, 4, -4, 4, 0.1, 10.0);

  // MVP matrices
  const teapotMVP = m4.multiply(m4.multiply(proj, view), teapotModel);
  const teapotLightMVP = m4.multiply(m4.multiply(lightProj, lightView), teapotModel);

  const identityMatrix = m4.identity();
  const groundLightMVP = m4.multiply(m4.multiply(lightProj, lightView), identityMatrix);

  // Update teapot uniforms
  const teapotUniforms = new Float32Array(52); // 16+16+16+3 = 51, rounded to 52
  teapotUniforms.set(teapotMVP, 0);
  teapotUniforms.set(teapotModel, 16);
  teapotUniforms.set(teapotLightMVP, 32);
  teapotUniforms[48] = lightX;
  teapotUniforms[49] = lightY;
  teapotUniforms[50] = lightZ;
  device.queue.writeBuffer(teapotUniformBuffer, 0, teapotUniforms.buffer);

  // Update ground uniforms
  const groundUniforms = new Float32Array(67); // 16+16+16+16+3 = 67
  groundUniforms.set(proj, 0);
  groundUniforms.set(view, 16);
  groundUniforms.set(identityMatrix, 32);
  groundUniforms.set(groundLightMVP, 48);
  groundUniforms[64] = lightX;
  groundUniforms[65] = lightY;
  groundUniforms[66] = lightZ;
  device.queue.writeBuffer(groundUniformBuffer, 0, groundUniforms.buffer);

  // Update shadow map uniforms
  device.queue.writeBuffer(shadowMapTeapotUniformBuffer, 0, teapotLightMVP.buffer);
  device.queue.writeBuffer(shadowMapGroundUniformBuffer, 0, groundLightMVP.buffer);

  const encoder = device.createCommandEncoder();

  // Pass 1: Render shadow map from light's POV
  const shadowPass = encoder.beginRenderPass({
    colorAttachments: [],
    depthStencilAttachment: {
      view: shadowDepthTexture.createView(),
      depthLoadOp: 'clear',
      depthClearValue: 1.0,
      depthStoreOp: 'store',
    },
  });

  shadowPass.setPipeline(shadowMapPipeline);

  // Draw teapot to shadow map
  shadowPass.setBindGroup(0, shadowMapTeapotBindGroup);
  shadowPass.setVertexBuffer(0, teapotPositionBuffer);
  shadowPass.setIndexBuffer(teapotIndexBuffer, 'uint32');
  shadowPass.drawIndexed(teapotIndexCount);

  // Draw ground to shadow map (need to create a pipeline for ground in shadow pass)
  // For simplicity, we skip ground in shadow map as it's the receiver

  shadowPass.end();

  // Pass 2: Main render pass
  const mainPass = encoder.beginRenderPass({
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

  // Draw ground
  mainPass.setPipeline(groundPipeline);
  mainPass.setBindGroup(0, groundBindGroup);
  mainPass.setVertexBuffer(0, groundVertexBuffer);
  mainPass.draw(6);

  // Draw teapot
  mainPass.setPipeline(teapotPipeline);
  mainPass.setBindGroup(0, teapotBindGroup);
  mainPass.setVertexBuffer(0, teapotPositionBuffer);
  mainPass.setVertexBuffer(1, teapotNormalBuffer);
  mainPass.setIndexBuffer(teapotIndexBuffer, 'uint32');
  mainPass.drawIndexed(teapotIndexCount);

  mainPass.end();

  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(drawFrame);
}

initWebGPU();
