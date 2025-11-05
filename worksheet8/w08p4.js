// WebGPU with semi-transparent shadows using alpha blending
async function init() {
  const canvas = document.getElementById('my-canvas');
  canvas.width = 1024;
  canvas.height = 512;

  if (!navigator.gpu) {
    console.error('WebGPU not supported');
    throw new Error('WebGPU not supported');
  }

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device: device,
    format: format,
    alphaMode: 'opaque',
  });

  // Shader code (WGSL) with visibility uniform
  const shaderCode = `
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

  // Ground quad: y = -1, x  [-2, 2], z  [-1, -5]
  const groundVertices = new Float32Array([
    -2.0, -1.0, -1.0,        0.0, 0.0,
     2.0, -1.0, -1.0,        1.0, 0.0,
     2.0, -1.0, -5.0,        1.0, 1.0,
    -2.0, -1.0, -1.0,        0.0, 0.0,
     2.0, -1.0, -5.0,        1.0, 1.0,
    -2.0, -1.0, -5.0,        0.0, 1.0,
  ]);

  // Small quad 1 (parallel to ground): y = -0.5, x  [0.25, 0.75], z  [-1.25, -1.75]
  const smallQuad1Vertices = new Float32Array([
     0.25, -0.5, -1.25,      0.0, 0.0,
     0.75, -0.5, -1.25,      1.0, 0.0,
     0.75, -0.5, -1.75,      1.0, 1.0,
     0.25, -0.5, -1.25,      0.0, 0.0,
     0.75, -0.5, -1.75,      1.0, 1.0,
     0.25, -0.5, -1.75,      0.0, 1.0,
  ]);

  // Small quad 2 (perpendicular to ground): x = -1, y  [-1, 0], z  [-2.5, -3]
  const smallQuad2Vertices = new Float32Array([
    -1.0, -1.0, -2.5,        0.0, 0.0,
    -1.0, -1.0, -3.0,        1.0, 0.0,
    -1.0,  0.0, -3.0,        1.0, 1.0,
    -1.0, -1.0, -2.5,        0.0, 0.0,
    -1.0,  0.0, -3.0,        1.0, 1.0,
    -1.0,  0.0, -2.5,        0.0, 1.0,
  ]);

  // Combine small quads
  const smallQuadsVertices = new Float32Array([
    ...smallQuad1Vertices,
    ...smallQuad2Vertices,
  ]);

  // Create vertex buffers
  const groundVertexBuffer = device.createBuffer({
    size: groundVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(groundVertexBuffer, 0, groundVertices);

  const smallQuadsVertexBuffer = device.createBuffer({
    size: smallQuadsVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(smallQuadsVertexBuffer, 0, smallQuadsVertices);

  // Create 1x1 red texture
  const redTextureData = new Uint8Array([255, 0, 0, 255]);
  const redTexture = device.createTexture({
    size: [1, 1, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: redTexture },
    redTextureData,
    { bytesPerRow: 4 },
    [1, 1, 1]
  );

  // Create 1x1 semi-transparent black texture for shadows (alpha = 0.6)
  const shadowAlpha = Math.floor(0.6 * 255); // 0.6 * 255 = 153
  const shadowTextureData = new Uint8Array([0, 0, 0, shadowAlpha]);
  const shadowTexture = device.createTexture({
    size: [1, 1, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: shadowTexture },
    shadowTextureData,
    { bytesPerRow: 4 },
    [1, 1, 1]
  );

  // Load marble texture
  const marbleTexture = await loadTexture(device, 'xamp23.png');

  // Create sampler
  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  // Create depth texture
  const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Create uniform buffers
  const uniformBufferSize = 256;

  const groundUniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const quadsUniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const shadowUniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Set up projection matrix
  const aspect = canvas.width / canvas.height;
  const fov = 90 * Math.PI / 180;
  const projectionMatrix = perspective(fov, aspect, 0.1, 100);

  // Identity view matrix
  const viewMatrix = new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  ]);

  // Identity model matrix
  const identityMatrix = new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  ]);

  // Create bind group layout
  const bindGroupLayout = device.createBindGroupLayout({
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

  // Create bind groups
  const marbleBindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: groundUniformBuffer } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: marbleTexture.createView() }
    ]
  });

  const redBindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: quadsUniformBuffer } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: redTexture.createView() }
    ]
  });

  const shadowBindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: shadowUniformBuffer } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: shadowTexture.createView() }
    ]
  });

  // Create shader module
  const shaderModule = device.createShaderModule({
    code: shaderCode
  });

  // Create pipeline layout
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout]
  });

  // Create render pipeline with "less" depth test (for normal objects)
  const pipelineNormal = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
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
      module: shaderModule,
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

  // Create render pipeline with "greater" depth test and alpha blending (for shadows)
  const pipelineShadow = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
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
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [{
        format: format,
        blend: {
          color: {
            srcFactor: 'src-alpha',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add'
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add'
          }
        }
      }]
    },
    primitive: {
      topology: 'triangle-list',
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'greater',
      format: 'depth24plus',
    }
  });

  // Animation loop
  let startTime = Date.now();

  function render() {
    const currentTime = (Date.now() - startTime) / 1000;

    // Animated light position
    const lightX = 0 + 2 * Math.cos(currentTime);
    const lightY = 2;
    const lightZ = -2 + 2 * Math.sin(currentTime);

    // Create shadow projection matrix with small offset
    const epsilon = 0.01;
    const shadowProjectionMatrix = createShadowProjectionMatrix(lightX, lightY, lightZ, epsilon);

    // Update uniform buffers
    const groundUniforms = new Float32Array([
      ...projectionMatrix,
      ...viewMatrix,
      ...identityMatrix,
      1.0, 0, 0, 0
    ]);
    device.queue.writeBuffer(groundUniformBuffer, 0, groundUniforms);

    const quadsUniforms = new Float32Array([
      ...projectionMatrix,
      ...viewMatrix,
      ...identityMatrix,
      1.0, 0, 0, 0
    ]);
    device.queue.writeBuffer(quadsUniformBuffer, 0, quadsUniforms);

    const shadowUniforms = new Float32Array([
      ...projectionMatrix,
      ...viewMatrix,
      ...shadowProjectionMatrix,
      0.0, 0, 0, 0
    ]);
    device.queue.writeBuffer(shadowUniformBuffer, 0, shadowUniforms);

    // Render
    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0.0, g: 0.0, b: 1.0, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store'
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      }
    });

    // Drawing order with proper pipeline switching

    // 1. Draw ground quad with normal pipeline
    renderPass.setPipeline(pipelineNormal);
    renderPass.setBindGroup(0, marbleBindGroup);
    renderPass.setVertexBuffer(0, groundVertexBuffer);
    renderPass.draw(6, 1, 0, 0);

    // 2. Draw shadow polygons with shadow pipeline (greater depth test + alpha blending)
    renderPass.setPipeline(pipelineShadow);
    renderPass.setBindGroup(0, shadowBindGroup);
    renderPass.setVertexBuffer(0, smallQuadsVertexBuffer);
    renderPass.draw(12, 1, 0, 0);

    // 3. Draw small quads with normal pipeline
    renderPass.setPipeline(pipelineNormal);
    renderPass.setBindGroup(0, redBindGroup);
    renderPass.setVertexBuffer(0, smallQuadsVertexBuffer);
    renderPass.draw(12, 1, 0, 0);

    renderPass.end();
    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(render);
  }

  render();
}

// Create shadow projection matrix with offset to avoid z-fighting
function createShadowProjectionMatrix(Lx, Ly, Lz, epsilon = 0) {
  // Plane with offset: y + 1 - epsilon = 0
  const n = [0, 1, 0, 1 - epsilon];
  const L = [Lx, Ly, Lz, 1];

  const dot = n[0] * L[0] + n[1] * L[1] + n[2] * L[2] + n[3] * L[3];

  return new Float32Array([
    dot - n[0] * L[0], -n[1] * L[0],       -n[2] * L[0],       -n[3] * L[0],
    -n[0] * L[1],      dot - n[1] * L[1],  -n[2] * L[1],       -n[3] * L[1],
    -n[0] * L[2],      -n[1] * L[2],       dot - n[2] * L[2],  -n[3] * L[2],
    -n[0] * L[3],      -n[1] * L[3],       -n[2] * L[3],       dot - n[3] * L[3]
  ]);
}

// Load texture from image file
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

// Create perspective projection matrix
function perspective(fov, aspect, near, far) {
  const f = 1.0 / Math.tan(fov / 2);
  const nf = 1 / (near - far);

  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0
  ]);
}

// Start the application
init().catch(console.error);
