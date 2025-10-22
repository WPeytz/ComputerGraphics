const canvas = document.getElementById('my-canvas');
const gl = canvas.getContext('webgl2');

if (!gl) {
  console.error('WebGL2 not supported');
  throw new Error('WebGL2 not supported');
}

canvas.width = 1024;
canvas.height = 512;

// Vertex shader - passes normal to fragment shader
const vertexShaderSource = `#version 300 es
in vec3 a_position;

out vec3 v_normal;

uniform mat4 u_mvp;

void main() {
  gl_Position = u_mvp * vec4(a_position, 1.0);
  // For unit sphere, the normal is just the normalized position
  v_normal = normalize(a_position);
}
`;

// Fragment shader - uses world space normal to look up cube map
const fragmentShaderSource = `#version 300 es
precision highp float;

in vec3 v_normal;
out vec4 fragColor;

uniform samplerCube u_texture;

void main() {
  // Use world space normal directly as texture coordinates for cube map
  // No normalization needed, cube map lookup handles it
  vec3 texColor = texture(u_texture, v_normal).rgb;

  // No shading, just return the texture color
  fragColor = vec4(texColor, 1.0);
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

// Sphere generation (subdivided tetrahedron)
function buildTetraSphere(level) {
  // Regular tetrahedron centered at origin
  const V0 = [ 1,  1,  1];
  const V1 = [-1, -1,  1];
  const V2 = [-1,  1, -1];
  const V3 = [ 1, -1, -1];

  function normalize(v) {
    const len = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0]/len, v[1]/len, v[2]/len];
  }

  let verts = [V0, V1, V2, V3].map(normalize);
  let faces = [
    [0,1,2],
    [0,3,1],
    [0,2,3],
    [1,3,2],
  ];

  // Subdivide
  for (let s = 0; s < level; s++) {
    const edgeMid = new Map();
    const newFaces = [];

    function key(a,b) { return a < b ? `${a},${b}` : `${b},${a}`; }

    function midpoint(a, b) {
      const k = key(a, b);
      if (edgeMid.has(k)) return edgeMid.get(k);

      const va = verts[a], vb = verts[b];
      const m = normalize([
        (va[0] + vb[0]) * 0.5,
        (va[1] + vb[1]) * 0.5,
        (va[2] + vb[2]) * 0.5
      ]);
      const idx = verts.push(m) - 1;
      edgeMid.set(k, idx);
      return idx;
    }

    for (const f of faces) {
      const [a, b, c] = f;
      const ab = midpoint(a, b);
      const bc = midpoint(b, c);
      const ca = midpoint(c, a);
      newFaces.push([a, ab, ca]);
      newFaces.push([ab, b, bc]);
      newFaces.push([ca, bc, c]);
      newFaces.push([ab, bc, ca]);
    }
    faces = newFaces;
  }

  // Flatten to buffers
  const positions = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) {
    positions.set(verts[i], i * 3);
  }

  const indices = new Uint32Array(faces.length * 3);
  let k = 0;
  for (const f of faces) {
    indices[k++] = f[0];
    indices[k++] = f[1];
    indices[k++] = f[2];
  }

  return { positions, indices };
}

// Generate sphere geometry
const sphereLevel = 5; // Good detail for Earth texture
const { positions, indices } = buildTetraSphere(sphereLevel);

// Create vertex buffer
const vertexBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

// Create index buffer
const indexBuffer = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

// Load cube map texture
const texture = gl.createTexture();
gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture);

// Cube map file names and their orientation
var cubemap = ['textures/cm_left.png',    // POSITIVE_X
               'textures/cm_right.png',   // NEGATIVE_X
               'textures/cm_top.png',     // POSITIVE_Y
               'textures/cm_bottom.png',  // NEGATIVE_Y
               'textures/cm_back.png',    // POSITIVE_Z
               'textures/cm_front.png'];  // NEGATIVE_Z

// Load images into an array
const imgs = [];
let loadedCount = 0;

// Load all images
for (let i = 0; i < 6; i++) {
  imgs[i] = new Image();
  imgs[i].onload = () => {
    loadedCount++;
    if (loadedCount === 6) {
      // All images loaded, create cube map texture
      gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture);

      // Copy each image to the right layer of the cube map
      const faces = [
        gl.TEXTURE_CUBE_MAP_POSITIVE_X,
        gl.TEXTURE_CUBE_MAP_NEGATIVE_X,
        gl.TEXTURE_CUBE_MAP_POSITIVE_Y,
        gl.TEXTURE_CUBE_MAP_NEGATIVE_Y,
        gl.TEXTURE_CUBE_MAP_POSITIVE_Z,
        gl.TEXTURE_CUBE_MAP_NEGATIVE_Z
      ];

      for (let j = 0; j < 6; j++) {
        gl.texImage2D(faces[j], 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imgs[j]);
      }

      // Set texture parameters
      gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

      console.log('Cube map loaded successfully');
    }
  };
  imgs[i].onerror = () => {
    console.error('Failed to load cube map texture:', cubemap[i]);
  };
  imgs[i].src = cubemap[i];
}

// Matrix math helpers (column-major)
function mat4_identity() {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  ]);
}

function mat4_multiply(a, b) {
  const result = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      result[col * 4 + row] =
        a[0 * 4 + row] * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return result;
}

function mat4_perspective(fovy, aspect, near, far) {
  const f = 1.0 / Math.tan(fovy / 2);
  const rangeInv = 1.0 / (near - far);

  // Standard OpenGL/WebGL perspective matrix (column-major)
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (near + far) * rangeInv, -1,
    0, 0, near * far * rangeInv * 2, 0
  ]);
}

function mat4_lookAt(eye, target, up) {
  let zx = target[0] - eye[0];
  let zy = target[1] - eye[1];
  let zz = target[2] - eye[2];
  const zlen = Math.hypot(zx, zy, zz) || 1;
  zx /= zlen; zy /= zlen; zz /= zlen;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  const xlen = Math.hypot(xx, xy, xz) || 1;
  xx /= xlen; xy /= xlen; xz /= xlen;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  const tx = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  const ty = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  const tz = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);

  return new Float32Array([
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    tx, ty, tz, 1
  ]);
}

// Set up camera
const aspect = canvas.width / canvas.height;
const fov = 45 * Math.PI / 180;
const projectionMatrix = mat4_perspective(fov, aspect, 0.1, 100.0);

// Use program and get uniform locations
gl.useProgram(program);

const u_mvp = gl.getUniformLocation(program, 'u_mvp');
const u_texture = gl.getUniformLocation(program, 'u_texture');

gl.uniform1i(u_texture, 0);

// Set up vertex attributes
const a_position = gl.getAttribLocation(program, 'a_position');

// Create VAO
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);

gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
gl.enableVertexAttribArray(a_position);
gl.vertexAttribPointer(a_position, 3, gl.FLOAT, false, 0, 0);

gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);

// Enable depth testing
gl.enable(gl.DEPTH_TEST);
gl.depthFunc(gl.LESS);

// Animation
let angle = 0;

function render(timestamp) {
  angle = (timestamp * 0.0005) % (Math.PI * 2);

  // Orbiting camera using rotation + translation
  const radius = 3.0;

  // Rotation around Y axis
  const c = Math.cos(-angle);
  const s = Math.sin(-angle);

  const viewMatrix = new Float32Array([
    c, 0, s, 0,
    0, 1, 0, 0,
    -s, 0, c, 0,
    0, 0, -radius, 1
  ]);

  const mvpMatrix = mat4_multiply(projectionMatrix, viewMatrix);

  gl.uniformMatrix4fv(u_mvp, false, mvpMatrix);

  // Clear to black background
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.0, 0.0, 0.0, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // Draw sphere
  gl.bindVertexArray(vao);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture);
  gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_INT, 0);

  requestAnimationFrame(render);
}

// Start animation
requestAnimationFrame(render);

