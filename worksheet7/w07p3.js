const canvas = document.getElementById('my-canvas');
const gl = canvas.getContext('webgl2');

if (!gl) {
  console.error('WebGL2 not supported');
  throw new Error('WebGL2 not supported');
}

canvas.width = 1024;
canvas.height = 512;

// Vertex shader - outputs both world position and normal for reflection calculation
//
// PART 3 CHANGES:
// - Added v_worldPos output to pass world space position to fragment shader
// - This allows the fragment shader to compute the view direction (eye - worldPos)
//
const vertexShaderSource = `#version 300 es
in vec3 a_position;

out vec3 v_normal;
out vec3 v_worldPos;

uniform mat4 u_mvp;
uniform mat4 u_mtex;

void main() {
  gl_Position = u_mvp * vec4(a_position, 1.0);

  // Transform position to texture coordinates using M_tex
  // For sphere: M_tex is identity, so this gives normalized position (= normal)
  // For background quad: M_tex transforms from clip space to world space directions
  vec4 texCoord = u_mtex * vec4(a_position, 1.0);
  v_normal = texCoord.xyz;

  // Pass world position for reflection calculation
  // For sphere: position is already in world space (identity transform)
  // For background: we still need the direction vector
  v_worldPos = texCoord.xyz;
}
`;

// Fragment shader - implements reflection for mirror ball effect
//
// REFLECTION EXPLAINED:
// Instead of using the surface normal directly, we compute the reflection of the
// view direction (incident ray) across the surface normal. This creates a mirror effect.
//
// The reflection formula: r = i - 2 * dot(n, i) * n
// Where:
//   i = incident direction (from eye to surface, normalized)
//   n = surface normal (normalized)
//   r = reflected direction (what we see in the mirror)
//
// GLSL provides: reflect(incident, normal) which does this calculation
//
const fragmentShaderSource = `#version 300 es
precision highp float;

in vec3 v_normal;
in vec3 v_worldPos;
out vec4 fragColor;

uniform samplerCube u_texture;
uniform bool u_reflective;      // true for mirror ball, false for background
uniform vec3 u_eyePosition;     // camera position in world space

void main() {
  vec3 lookupDir;

  if (u_reflective) {
    // REFLECTIVE SPHERE (Mirror Ball):
    // Compute reflection direction for realistic mirror effect

    // 1. Compute incident direction (from eye to surface point)
    vec3 incident = normalize(v_worldPos - u_eyePosition);

    // 2. Get surface normal (for sphere, this is just normalized position)
    vec3 normal = normalize(v_normal);

    // 3. Compute reflection direction
    //    The reflect() function computes: i - 2 * dot(n, i) * n
    //    This gives us the direction the light would bounce off the mirror
    vec3 reflectionDir = reflect(incident, normal);

    lookupDir = reflectionDir;
  } else {
    // NON-REFLECTIVE (Background):
    // Use the direction vector directly (no reflection)
    lookupDir = v_normal;
  }

  // Sample cube map in the computed direction
  vec3 texColor = texture(u_texture, lookupDir).rgb;
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
const sphereLevel = 5;
const { positions: spherePositions, indices: sphereIndices } = buildTetraSphere(sphereLevel);

// Create vertex buffer for sphere
const sphereVertexBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, sphereVertexBuffer);
gl.bufferData(gl.ARRAY_BUFFER, spherePositions, gl.STATIC_DRAW);

// Create index buffer for sphere
const sphereIndexBuffer = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, sphereIndexBuffer);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, sphereIndices, gl.STATIC_DRAW);

// Create background quad geometry (in clip space, close to far plane)
const quadPositions = new Float32Array([
  -1.0, -1.0, 0.999,  // bottom-left
   1.0, -1.0, 0.999,  // bottom-right
   1.0,  1.0, 0.999,  // top-right
  -1.0,  1.0, 0.999   // top-left
]);

const quadIndices = new Uint32Array([
  0, 1, 2,  // first triangle
  0, 2, 3   // second triangle
]);

// Create vertex buffer for quad
const quadVertexBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadVertexBuffer);
gl.bufferData(gl.ARRAY_BUFFER, quadPositions, gl.STATIC_DRAW);

// Create index buffer for quad
const quadIndexBuffer = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, quadIndexBuffer);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, quadIndices, gl.STATIC_DRAW);

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

// Matrix inversion using Gauss-Jordan elimination
function mat4_invert(m) {
  const inv = new Float32Array(16);

  inv[0] = m[5]  * m[10] * m[15] - m[5]  * m[11] * m[14] - m[9]  * m[6]  * m[15] +
           m[9]  * m[7]  * m[14] + m[13] * m[6]  * m[11] - m[13] * m[7]  * m[10];
  inv[4] = -m[4]  * m[10] * m[15] + m[4]  * m[11] * m[14] + m[8]  * m[6]  * m[15] -
            m[8]  * m[7]  * m[14] - m[12] * m[6]  * m[11] + m[12] * m[7]  * m[10];
  inv[8] = m[4]  * m[9] * m[15] - m[4]  * m[11] * m[13] - m[8]  * m[5] * m[15] +
           m[8]  * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4]  * m[9] * m[14] + m[4]  * m[10] * m[13] + m[8]  * m[5] * m[14] -
             m[8]  * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
  inv[1] = -m[1]  * m[10] * m[15] + m[1]  * m[11] * m[14] + m[9]  * m[2] * m[15] -
            m[9]  * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0]  * m[10] * m[15] - m[0]  * m[11] * m[14] - m[8]  * m[2] * m[15] +
           m[8]  * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0]  * m[9] * m[15] + m[0]  * m[11] * m[13] + m[8]  * m[1] * m[15] -
            m[8]  * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0]  * m[9] * m[14] - m[0]  * m[10] * m[13] - m[8]  * m[1] * m[14] +
            m[8]  * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
  inv[2] = m[1]  * m[6] * m[15] - m[1]  * m[7] * m[14] - m[5]  * m[2] * m[15] +
           m[5]  * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0]  * m[6] * m[15] + m[0]  * m[7] * m[14] + m[4]  * m[2] * m[15] -
            m[4]  * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0]  * m[5] * m[15] - m[0]  * m[7] * m[13] - m[4]  * m[1] * m[15] +
            m[4]  * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0]  * m[5] * m[14] + m[0]  * m[6] * m[13] + m[4]  * m[1] * m[14] -
             m[4]  * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
            m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
           m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
             m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
            m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

  let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

  if (det === 0) {
    console.error('Matrix is not invertible');
    return mat4_identity();
  }

  det = 1.0 / det;
  for (let i = 0; i < 16; i++) {
    inv[i] = inv[i] * det;
  }

  return inv;
}

// Extract rotational part of view matrix (3x3 upper-left, extended to 4x4)
function mat4_extractRotation(m) {
  return new Float32Array([
    m[0], m[1], m[2], 0,
    m[4], m[5], m[6], 0,
    m[8], m[9], m[10], 0,
    0, 0, 0, 1
  ]);
}

// Set up camera
const aspect = canvas.width / canvas.height;
const fov = 45 * Math.PI / 180;
const projectionMatrix = mat4_perspective(fov, aspect, 0.1, 100.0);

// Precompute inverse of projection matrix (constant)
const invProjectionMatrix = mat4_invert(projectionMatrix);

// Use program and get uniform locations
gl.useProgram(program);

const u_mvp = gl.getUniformLocation(program, 'u_mvp');
const u_mtex = gl.getUniformLocation(program, 'u_mtex');
const u_texture = gl.getUniformLocation(program, 'u_texture');
const u_reflective = gl.getUniformLocation(program, 'u_reflective');
const u_eyePosition = gl.getUniformLocation(program, 'u_eyePosition');

gl.uniform1i(u_texture, 0);

// Set up vertex attributes
const a_position = gl.getAttribLocation(program, 'a_position');

// Create VAO for sphere
const sphereVao = gl.createVertexArray();
gl.bindVertexArray(sphereVao);

gl.bindBuffer(gl.ARRAY_BUFFER, sphereVertexBuffer);
gl.enableVertexAttribArray(a_position);
gl.vertexAttribPointer(a_position, 3, gl.FLOAT, false, 0, 0);

gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, sphereIndexBuffer);

// Create VAO for background quad
const quadVao = gl.createVertexArray();
gl.bindVertexArray(quadVao);

gl.bindBuffer(gl.ARRAY_BUFFER, quadVertexBuffer);
gl.enableVertexAttribArray(a_position);
gl.vertexAttribPointer(a_position, 3, gl.FLOAT, false, 0, 0);

gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, quadIndexBuffer);

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

  // Compute eye position in world space
  // The camera is at (0, 0, -radius) in camera space
  // After the rotation, it's at (radius * sin(angle), 0, radius * cos(angle))
  const eyeX = radius * Math.sin(-angle);
  const eyeY = 0.0;
  const eyeZ = radius * Math.cos(-angle);

  // Extract rotational part of view matrix (no translation)
  const viewRotation = mat4_extractRotation(viewMatrix);

  // Compute inverse of rotational part
  const invViewRotation = mat4_invert(viewRotation);

  // Clear
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.0, 0.0, 0.0, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture);

  // ============================================================================
  // DRAW BACKGROUND QUAD (Non-Reflective Environment)
  // ============================================================================
  const quadMvp = mat4_identity();
  const quadMtex = mat4_multiply(invViewRotation, invProjectionMatrix);

  gl.bindVertexArray(quadVao);
  gl.uniformMatrix4fv(u_mvp, false, quadMvp);
  gl.uniformMatrix4fv(u_mtex, false, quadMtex);
  gl.uniform1i(u_reflective, 0);  // false - not reflective
  gl.uniform3f(u_eyePosition, eyeX, eyeY, eyeZ);
  gl.drawElements(gl.TRIANGLES, quadIndices.length, gl.UNSIGNED_INT, 0);

  // ============================================================================
  // DRAW SPHERE (Reflective Mirror Ball)
  // ============================================================================
  //
  // REFLECTION CHANGES:
  // - Set u_reflective = true to enable reflection calculation in fragment shader
  // - Pass eye position so fragment shader can compute incident direction
  // - Fragment shader will use reflect() to compute proper reflection direction
  //
  const sphereMvp = mat4_multiply(projectionMatrix, viewMatrix);
  const sphereMtex = mat4_identity();

  gl.bindVertexArray(sphereVao);
  gl.uniformMatrix4fv(u_mvp, false, sphereMvp);
  gl.uniformMatrix4fv(u_mtex, false, sphereMtex);
  gl.uniform1i(u_reflective, 1);  // true - use reflection
  gl.uniform3f(u_eyePosition, eyeX, eyeY, eyeZ);
  gl.drawElements(gl.TRIANGLES, sphereIndices.length, gl.UNSIGNED_INT, 0);

  requestAnimationFrame(render);
}

// Start animation
requestAnimationFrame(render);
