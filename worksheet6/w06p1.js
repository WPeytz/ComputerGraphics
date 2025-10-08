const canvas = document.getElementById('my-canvas');
const gl = canvas.getContext('webgl2');

if (!gl) {
  console.error('WebGL2 not supported');
  throw new Error('WebGL2 not supported');
}

canvas.width = 1024;
canvas.height = 512;

// Vertex shader
const vertexShaderSource = `#version 300 es
in vec3 a_position;
in vec2 a_texCoord;

out vec2 v_texCoord;

uniform mat4 u_projection;
uniform mat4 u_view;

void main() {
  gl_Position = u_projection * u_view * vec4(a_position, 1.0);
  v_texCoord = a_texCoord;
}
`;

// Fragment shader
const fragmentShaderSource = `#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_texture;

void main() {
  fragColor = texture(u_texture, v_texCoord);
}
`;

// Compile shader
function compileShader(gl, source, type) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Shader compile error:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

// Create program
function createProgram(gl, vertexSource, fragmentSource) {
  const vertexShader = compileShader(gl, vertexSource, gl.VERTEX_SHADER);
  const fragmentShader = compileShader(gl, fragmentSource, gl.FRAGMENT_SHADER);

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(program));
    return null;
  }
  return program;
}

const program = createProgram(gl, vertexShaderSource, fragmentShaderSource);

// Rectangle vertices (x, y, z)
const vertices = new Float32Array([
  -4, -1, -1,   // bottom-left
   4, -1, -1,   // bottom-right
   4, -1, -21,  // top-right
  -4, -1, -21   // top-left
]);

// Texture coordinates
const texCoords = new Float32Array([
  -1.5,  0.0,   // bottom-left
   2.5,  0.0,   // bottom-right
   2.5, 10.0,   // top-right
  -1.5, 10.0    // top-left
]);

// Indices for two triangles
const indices = new Uint16Array([
  0, 1, 2,
  0, 2, 3
]);

// Create and bind vertex buffer
const vertexBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

// Create and bind texture coordinate buffer
const texCoordBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);

// Create and bind index buffer
const indexBuffer = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

// Generate 64x64 checkerboard texture (8x8 pattern)
function generateCheckerboard() {
  const size = 64;
  const data = new Uint8Array(size * size * 4);
  const squareSize = size / 8; // 8x8 checkerboard

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const squareX = Math.floor(x / squareSize);
      const squareY = Math.floor(y / squareSize);
      const isBlack = (squareX + squareY) % 2 === 0;

      const index = (y * size + x) * 4;
      const color = isBlack ? 0 : 255;
      data[index] = color;     // R
      data[index + 1] = color; // G
      data[index + 2] = color; // B
      data[index + 3] = 255;   // A
    }
  }
  return data;
}

// Create and configure texture
const texture = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, texture);

const checkerboardData = generateCheckerboard();
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 64, 64, 0, gl.RGBA, gl.UNSIGNED_BYTE, checkerboardData);

// Set texture parameters (repeat addressing and nearest filtering)
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

// Set up perspective projection (90 degrees FOV)
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

// Default view matrix (identity)
const viewMatrix = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1
]);

const aspect = canvas.width / canvas.height;
const fov = 90 * Math.PI / 180; // 90 degrees in radians
const projectionMatrix = perspective(fov, aspect, 0.1, 100);

// Use program and set up uniforms
gl.useProgram(program);

const u_projection = gl.getUniformLocation(program, 'u_projection');
const u_view = gl.getUniformLocation(program, 'u_view');
const u_texture = gl.getUniformLocation(program, 'u_texture');

gl.uniformMatrix4fv(u_projection, false, projectionMatrix);
gl.uniformMatrix4fv(u_view, false, viewMatrix);
gl.uniform1i(u_texture, 0);

// Set up vertex attributes
const a_position = gl.getAttribLocation(program, 'a_position');
const a_texCoord = gl.getAttribLocation(program, 'a_texCoord');

// Create VAO
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);

// Set up position attribute
gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
gl.enableVertexAttribArray(a_position);
gl.vertexAttribPointer(a_position, 3, gl.FLOAT, false, 0, 0);

// Set up texture coordinate attribute
gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
gl.enableVertexAttribArray(a_texCoord);
gl.vertexAttribPointer(a_texCoord, 2, gl.FLOAT, false, 0, 0);

// Bind index buffer
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);

// Render
gl.viewport(0, 0, canvas.width, canvas.height);
gl.clearColor(0.0, 0.0, 1.0, 1.0); // Blue background
gl.clear(gl.COLOR_BUFFER_BIT);

gl.bindVertexArray(vao);
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, texture);
gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
