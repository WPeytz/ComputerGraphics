// WebGPU initialization
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

  // Shader code (WGSL)
  const shaderCode = `
    struct Uniforms {
      projection: mat4x4<f32>,
      view: mat4x4<f32>,
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
      output.position = uniforms.projection * uniforms.view * vec4<f32>(input.position, 1.0);
      output.texCoord = input.texCoord;
      return output;
    }

    @fragment
    fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
      return textureSample(myTexture, mySampler, input.texCoord);
    }
  `;

  // Ground quad: y = -1, x  [-2, 2], z  [-1, -5]
  // Two triangles forming the quad
  const groundVertices = new Float32Array([
    // Position (x, y, z)    // Texture coords (u, v)
    -2.0, -1.0, -1.0,        0.0, 0.0,
     2.0, -1.0, -1.0,        1.0, 0.0,
     2.0, -1.0, -5.0,        1.0, 1.0,
    -2.0, -1.0, -1.0,        0.0, 0.0,
     2.0, -1.0, -5.0,        1.0, 1.0,
    -2.0, -1.0, -5.0,        0.0, 1.0,
  ]);

  // Small quad 1 (parallel to ground, above it): y = -0.5, x  [0.25, 0.75], z  [-1.25, -1.75]
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

  // Combine small quads into one buffer
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

  // Load marble texture from xamp23.png
  const marbleTexture = await loadTexture(device, 'xamp23.png');

  // Create sampler
  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  // Create uniform buffer for projection and view matrices
  const uniformBuffer = device.createBuffer({
    size: 128, // Two 4x4 matrices (64 bytes each)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Set up projection matrix (perspective)
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

  // Write matrices to uniform buffer
  const matrices = new Float32Array([...projectionMatrix, ...viewMatrix]);
  device.queue.writeBuffer(uniformBuffer, 0, matrices);

  // Create bind group layout
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
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

  // Create bind group for marble texture (ground)
  const marbleBindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: marbleTexture.createView() }
    ]
  });

  // Create bind group for red texture (small quads)
  const redBindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: redTexture.createView() }
    ]
  });

  // Create shader module
  const shaderModule = device.createShaderModule({
    code: shaderCode
  });

  // Create render pipeline
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    }),
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
      buffers: [{
        arrayStride: 20, // 3 floats for position + 2 floats for texCoord = 5 floats * 4 bytes
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
    }
  });

  // Render
  const commandEncoder = device.createCommandEncoder();
  const textureView = context.getCurrentTexture().createView();

  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [{
      view: textureView,
      clearValue: { r: 0.0, g: 0.0, b: 1.0, a: 1.0 }, // Blue background
      loadOp: 'clear',
      storeOp: 'store'
    }]
  });

  renderPass.setPipeline(pipeline);

  // Draw ground quad with marble texture
  renderPass.setBindGroup(0, marbleBindGroup);
  renderPass.setVertexBuffer(0, groundVertexBuffer);
  renderPass.draw(6, 1, 0, 0); // 6 vertices for ground quad

  // Draw small quads with red texture
  renderPass.setBindGroup(0, redBindGroup);
  renderPass.setVertexBuffer(0, smallQuadsVertexBuffer);
  renderPass.draw(12, 1, 0, 0); // 12 vertices for both small quads

  renderPass.end();
  device.queue.submit([commandEncoder.finish()]);
}

// Helper function to load texture from image file
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
