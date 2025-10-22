const canvas = document.getElementById('my-canvas');
const gl = canvas.getContext('webgl2');

if (!gl) {
  console.error('WebGL2 not supported');
  throw new Error('WebGL2 not supported');
}

canvas.width = 1024;
canvas.height = 512;

// Vertex shader - outputs world position, normal, and texture coordinates for normal mapping
//
// PART 4 CHANGES:
// - Added v_texCoord output for sampling the normal map
// - Texture coordinates computed using spherical mapping (same as Worksheet 6 Part 3)
//
const vertexShaderSource = `#version 300 es
in vec3 a_position;

out vec3 v_normal;
out vec3 v_worldPos;
out vec2 v_texCoord;

uniform mat4 u_mvp;
uniform mat4 u_mtex;

void main() {
  gl_Position = u_mvp * vec4(a_position, 1.0);

  // Transform position to texture coordinates using M_tex
  vec4 texCoord = u_mtex * vec4(a_position, 1.0);
  v_normal = texCoord.xyz;
  v_worldPos = texCoord.xyz;

  // Compute spherical texture coordinates for normal map
  // Same technique as Worksheet 6 Part 3
  vec3 n = normalize(a_position);
  float u = 0.5 + atan(n.x, n.z) / (2.0 * 3.14159265359);
  float v = 0.5 - asin(n.y) / 3.14159265359;
  v_texCoord = vec2(u, v);
}
`;

// Fragment shader - implements bump mapping with normal map
//
// BUMP MAPPING EXPLAINED:
// Normal mapping (bump mapping) creates the illusion of surface detail by perturbing
// the surface normal before lighting/reflection calculations. The normal map stores
// normal vectors in tangent space, which must be transformed to world space.
//
// Key steps:
// 1. Sample normal map texture
// 2. Transform from [0,1] color range to [-1,1] normal range
// 3. Transform from tangent space to world space using rotate_to_normal()
// 4. Use perturbed normal for reflection calculation
//
const fragmentShaderSource = `#version 300 es
precision highp float;

in vec3 v_normal;
in vec3 v_worldPos;
in vec2 v_texCoord;
out vec4 fragColor;

uniform samplerCube u_texture;
uniform sampler2D u_normalMap;
uniform bool u_reflective;
uniform vec3 u_eyePosition;

// Transform tangent space normal to world space
// n: surface normal in world space
// v: tangent space vector (from normal map)
// Returns: world space normal
//
// This function builds a tangent-to-world transformation matrix on the fly
// using the surface normal. It's more efficient than storing tangent/bitangent
// vectors per vertex.
vec3 rotate_to_normal(vec3 n, vec3 v) {
  float sgn_nz = sign(n.z + 1.0e-16);
  float a = -1.0/(1.0 + abs(n.z));
  float b = n.x*n.y*a;
  return vec3(1.0 + n.x*n.x*a, b, -sgn_nz*n.x)*v.x
       + vec3(sgn_nz*b, sgn_nz*(1.0 + n.y*n.y*a), -n.y)*v.y
       + n*v.z;
}

void main() {
  vec3 lookupDir;

  if (u_reflective) {
    // REFLECTIVE SPHERE WITH BUMP MAPPING:

    // 1. Get base surface normal (sphere normal)
    vec3 surfaceNormal = normalize(v_normal);

    // 2. Sample normal map
    //    The normal map stores normals in tangent space as RGB colors [0,1]³
    vec3 tangentNormal = texture(u_normalMap, v_texCoord).rgb;

    // 3. Transform from [0,1] color range to [-1,1] normal range
    //    Color 0.5 represents 0, color 0.0 represents -1, color 1.0 represents +1
    tangentNormal = tangentNormal * 2.0 - 1.0;

    // 4. Transform tangent space normal to world space
    //    The tangent space normal is relative to the surface
    //    We need to rotate it to world space using the surface normal
    vec3 perturbedNormal = rotate_to_normal(surfaceNormal, tangentNormal);
    perturbedNormal = normalize(perturbedNormal);

    // 5. Compute incident direction (from eye to surface)
    vec3 incident = normalize(v_worldPos - u_eyePosition);

    // 6. Compute reflection using the PERTURBED normal instead of surface normal
    //    This is what creates the bumpy appearance!
    vec3 reflectionDir = reflect(incident, perturbedNormal);

    lookupDir = reflectionDir;
  } else {
    // NON-REFLECTIVE (Background): use direction vector directly
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
  -1.0, -1.0, 0.999,
   1.0, -1.0, 0.999,
   1.0,  1.0, 0.999,
  -1.0,  1.0, 0.999
]);

const quadIndices = new Uint32Array([
  0, 1, 2,
  0, 2, 3
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
const cubeTexture = gl.createTexture();
gl.bindTexture(gl.TEXTURE_CUBE_MAP, cubeTexture);

var cubemap = ['textures/cm_left.png',
               'textures/cm_right.png',
               'textures/cm_top.png',
               'textures/cm_bottom.png',
               'textures/cm_back.png',
               'textures/cm_front.png'];

const imgs = [];
let loadedCount = 0;

for (let i = 0; i < 6; i++) {
  imgs[i] = new Image();
  imgs[i].onload = () => {
    loadedCount++;
    if (loadedCount === 6) {
      gl.bindTexture(gl.TEXTURE_CUBE_MAP, cubeTexture);

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

// Load normal map texture
const normalMapTexture = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, normalMapTexture);

// Placeholder while loading
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
              new Uint8Array([128, 128, 255, 255])); // Neutral normal (0, 0, 1)

const normalMapImage = new Image();
normalMapImage.onload = () => {
  gl.bindTexture(gl.TEXTURE_2D, normalMapTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, normalMapImage);

  // Generate mipmaps for better quality
  gl.generateMipmap(gl.TEXTURE_2D);

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);

  console.log('Normal map loaded successfully');
};
normalMapImage.onerror = () => {
  console.error('Failed to load normal map texture');
};
normalMapImage.src = 'textures/normalmap.png';

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

  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (near + far) * rangeInv, -1,
    0, 0, near * far * rangeInv * 2, 0
  ]);
}

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
const invProjectionMatrix = mat4_invert(projectionMatrix);

// Use program and get uniform locations
gl.useProgram(program);

const u_mvp = gl.getUniformLocation(program, 'u_mvp');
const u_mtex = gl.getUniformLocation(program, 'u_mtex');
const u_texture = gl.getUniformLocation(program, 'u_texture');
const u_normalMap = gl.getUniformLocation(program, 'u_normalMap');
const u_reflective = gl.getUniformLocation(program, 'u_reflective');
const u_eyePosition = gl.getUniformLocation(program, 'u_eyePosition');

// Bind textures to texture units
gl.uniform1i(u_texture, 0);    // Cube map on texture unit 0
gl.uniform1i(u_normalMap, 1);  // Normal map on texture unit 1

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

  const radius = 3.0;
  const c = Math.cos(-angle);
  const s = Math.sin(-angle);

  const viewMatrix = new Float32Array([
    c, 0, s, 0,
    0, 1, 0, 0,
    -s, 0, c, 0,
    0, 0, -radius, 1
  ]);

  const eyeX = radius * Math.sin(-angle);
  const eyeY = 0.0;
  const eyeZ = radius * Math.cos(-angle);

  const viewRotation = mat4_extractRotation(viewMatrix);
  const invViewRotation = mat4_invert(viewRotation);

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.0, 0.0, 0.0, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // Bind both textures
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, cubeTexture);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, normalMapTexture);

  // Draw background quad
  const quadMvp = mat4_identity();
  const quadMtex = mat4_multiply(invViewRotation, invProjectionMatrix);

  gl.bindVertexArray(quadVao);
  gl.uniformMatrix4fv(u_mvp, false, quadMvp);
  gl.uniformMatrix4fv(u_mtex, false, quadMtex);
  gl.uniform1i(u_reflective, 0);
  gl.uniform3f(u_eyePosition, eyeX, eyeY, eyeZ);
  gl.drawElements(gl.TRIANGLES, quadIndices.length, gl.UNSIGNED_INT, 0);

  // ============================================================================
  // DRAW SPHERE WITH BUMP MAPPING
  // ============================================================================
  //
  // The sphere now uses both:
  // - Cube map (texture unit 0) for environment reflections
  // - Normal map (texture unit 1) for bump detail
  //
  // The fragment shader samples the normal map, transforms it from tangent space
  // to world space, and uses the perturbed normal for reflection calculation.
  //
  const sphereMvp = mat4_multiply(projectionMatrix, viewMatrix);
  const sphereMtex = mat4_identity();

  gl.bindVertexArray(sphereVao);
  gl.uniformMatrix4fv(u_mvp, false, sphereMvp);
  gl.uniformMatrix4fv(u_mtex, false, sphereMtex);
  gl.uniform1i(u_reflective, 1);
  gl.uniform3f(u_eyePosition, eyeX, eyeY, eyeZ);
  gl.drawElements(gl.TRIANGLES, sphereIndices.length, gl.UNSIGNED_INT, 0);

  requestAnimationFrame(render);
}

requestAnimationFrame(render);
