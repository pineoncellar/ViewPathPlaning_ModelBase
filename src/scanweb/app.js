"use strict";

const $ = (id) => document.getElementById(id);
const FOV = 45 * Math.PI / 180;
const SIDE_VECTORS = {
  "+X": [1, 0, 0], "-X": [-1, 0, 0],
  "+Y": [0, 1, 0], "-Y": [0, -1, 0],
  "+Z": [0, 0, 1], "-Z": [0, 0, -1]
};
const AXIS_LABELS = { "+X": "车尾 +X", "-X": "车头 −X", "+Y": "右侧 +Y", "-Y": "左侧 −Y", "+Z": "上方 +Z", "-Z": "下方 −Z" };
const MODEL_LABELS = { part: "零件", support: "夹具", base: "底座" };
const MAX_TRIANGLES = 500000;

const state = {
  models: {},
  picked: [],
  pointCount: 3,
  measured: Array.from({ length: 4 }, () => ["", "", ""]),
  autoFace: false,
  registration: null,
  solving: false,
  result: null,
  resultSource: null,
  showPath: true,
  relocate: { busy: false, token: 0 }
};

let registrationTimer = 0;
let registrationRequest = 0;
let toastTimer = 0;

const glState = {
  gl: null,
  meshProgram: null,
  flatProgram: null,
  meshLoc: {},
  flatLoc: {},
  meshes: {},
  center: [0, 0, 0],
  radius: 1000,
  distance: 2600,
  yaw: -Math.PI / 3,
  pitch: Math.PI / 6,
  home: null,
  pathLine: null,
  pathLineCount: 0,
  pathPoint: null,
  pathPointCount: 0,
  raf: 0,
  drag: null,
  lastView: null,
  lastProjection: null,
  lastWidth: 1,
  lastHeight: 1
};

/* ---------- small vector / matrix helpers ---------- */

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function mul(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function unit(a) { const n = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / n, a[1] / n, a[2] / n]; }
function clamp(value, low, high) { return Math.min(Math.max(value, low), high); }

function transform4(m, p) {
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15]
  ];
}

function rotate3(m, v) {
  return [
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2]
  ];
}

function camera() {
  const cp = Math.cos(glState.pitch);
  const eye = [
    glState.center[0] + glState.distance * cp * Math.cos(glState.yaw),
    glState.center[1] + glState.distance * cp * Math.sin(glState.yaw),
    glState.center[2] + glState.distance * Math.sin(glState.pitch)
  ];
  const forward = unit(sub(glState.center, eye));
  const right = unit(cross(forward, [0, 0, 1]));
  const up = cross(right, forward);
  return { eye, forward, right, up };
}

function lookAt(eye, target, up) {
  const z = unit(sub(eye, target));
  const x = unit(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
  ]);
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0
  ]);
}

function projectPoint(view, projection, width, height, point) {
  const v = transform4(view, point);
  const c = transform4(projection, v);
  if (c[3] <= 1e-6) return null;
  return {
    x: (c[0] / c[3] * 0.5 + 0.5) * width,
    y: (1 - (c[1] / c[3] * 0.5 + 0.5)) * height,
    depth: v[2]
  };
}

/* ---------- WebGL renderer ---------- */

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || "着色器编译失败");
  }
  return shader;
}

function link(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) || "着色器链接失败");
  }
  return program;
}

function initRenderer() {
  const canvas = $("modelCanvas");
  const gl = canvas.getContext("webgl", { antialias: true, alpha: false, preserveDrawingBuffer: true });
  if (!gl) {
    setRegistrationStatus("当前浏览器无法初始化 WebGL，3D 视图不可用，但配准与规划仍可使用。", "bad");
    return;
  }
  glState.gl = gl;
  try {
    glState.meshProgram = link(gl,
      [
        "attribute vec3 aPos;", "attribute vec3 aNrm;",
        "uniform mat4 uProj;", "uniform mat4 uView;", "uniform mat3 uNormal;",
        "varying vec3 vNrm;",
        "void main() {",
        "  vNrm = uNormal * aNrm;",
        "  gl_Position = uProj * (uView * vec4(aPos, 1.0));",
        "}"
      ].join("\n"),
      [
        "precision mediump float;",
        "varying vec3 vNrm;",
        "uniform vec3 uColor;",
        "uniform vec3 uScanAxis;",
        "uniform float uHighlight;",
        "void main() {",
        "  vec3 n = normalize(vNrm);",
        "  if (!gl_FrontFacing) n = -n;",
        "  vec3 c = uColor;",
        "  if (uHighlight > 0.5) {",
        "    float a = dot(n, uScanAxis);",
        "    if (a > 0.45) c = vec3(0.83, 0.39, 0.20);",
        "    else if (a < -0.45) c = vec3(0.40, 0.52, 0.60);",
        "    else c = vec3(0.62, 0.67, 0.71);",
        "  }",
        "  vec3 light = normalize(vec3(0.35, 0.55, 1.0));",
        "  float diffuse = max(dot(n, light), 0.0);",
        "  gl_FragColor = vec4(c * (0.38 + 0.62 * diffuse), 1.0);",
        "}"
      ].join("\n")
    );
    glState.flatProgram = link(gl,
      [
        "attribute vec3 aPos;",
        "uniform mat4 uProj;", "uniform mat4 uView;",
        "uniform float uSize;", "uniform float uShift;",
        "void main() {",
        "  vec4 view = uView * vec4(aPos, 1.0);",
        "  view.z += uShift;",
        "  gl_Position = uProj * view;",
        "  gl_PointSize = uSize;",
        "}"
      ].join("\n"),
      [
        "precision mediump float;",
        "uniform vec3 uColor;", "uniform float uAlpha;",
        "void main() { gl_FragColor = vec4(uColor, uAlpha); }"
      ].join("\n")
    );
  } catch (error) {
    setRegistrationStatus("3D 渲染器初始化失败：" + error.message, "bad");
    glState.gl = null;
    return;
  }
  glState.meshLoc = {
    pos: gl.getAttribLocation(glState.meshProgram, "aPos"),
    nrm: gl.getAttribLocation(glState.meshProgram, "aNrm"),
    proj: gl.getUniformLocation(glState.meshProgram, "uProj"),
    view: gl.getUniformLocation(glState.meshProgram, "uView"),
    normal: gl.getUniformLocation(glState.meshProgram, "uNormal"),
    color: gl.getUniformLocation(glState.meshProgram, "uColor"),
    axis: gl.getUniformLocation(glState.meshProgram, "uScanAxis"),
    highlight: gl.getUniformLocation(glState.meshProgram, "uHighlight")
  };
  glState.flatLoc = {
    pos: gl.getAttribLocation(glState.flatProgram, "aPos"),
    proj: gl.getUniformLocation(glState.flatProgram, "uProj"),
    view: gl.getUniformLocation(glState.flatProgram, "uView"),
    size: gl.getUniformLocation(glState.flatProgram, "uSize"),
    shift: gl.getUniformLocation(glState.flatProgram, "uShift"),
    color: gl.getUniformLocation(glState.flatProgram, "uColor"),
    alpha: gl.getUniformLocation(glState.flatProgram, "uAlpha")
  };
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  gl.disable(gl.CULL_FACE);
}

function computeBounds(triangles) {
  const low = [Infinity, Infinity, Infinity];
  const high = [-Infinity, -Infinity, -Infinity];
  for (const triangle of triangles) {
    for (const point of triangle) {
      for (let axis = 0; axis < 3; axis += 1) {
        if (point[axis] < low[axis]) low[axis] = point[axis];
        if (point[axis] > high[axis]) high[axis] = point[axis];
      }
    }
  }
  return { low, high };
}

function buildMeshBuffers(name, triangles) {
  const gl = glState.gl;
  if (!gl) return;
  const positions = new Float32Array(triangles.length * 9);
  const normals = new Float32Array(triangles.length * 9);
  for (let index = 0; index < triangles.length; index += 1) {
    const tri = triangles[index];
    const ab = sub(tri[1], tri[0]);
    const ac = sub(tri[2], tri[0]);
    const normal = unit(cross(ab, ac));
    for (let vertex = 0; vertex < 3; vertex += 1) {
      const offset = index * 9 + vertex * 3;
      positions[offset] = tri[vertex][0];
      positions[offset + 1] = tri[vertex][1];
      positions[offset + 2] = tri[vertex][2];
      normals[offset] = normal[0];
      normals[offset + 1] = normal[1];
      normals[offset + 2] = normal[2];
    }
  }
  const previous = glState.meshes[name];
  if (previous) {
    gl.deleteBuffer(previous.posBuffer);
    gl.deleteBuffer(previous.nrmBuffer);
  }
  const posBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  const nrmBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
  glState.meshes[name] = { posBuffer, nrmBuffer, count: triangles.length * 3 };
}

function render() {
  glState.raf = 0;
  const gl = glState.gl;
  const canvas = $("modelCanvas");
  const overlay = $("overlayCanvas");
  if (!gl || !canvas) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssWidth = Math.max(canvas.clientWidth, 1);
  const cssHeight = Math.max(canvas.clientHeight, 1);
  const width = Math.max(1, Math.round(cssWidth * dpr));
  const height = Math.max(1, Math.round(cssHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }
  glState.lastWidth = cssWidth;
  glState.lastHeight = cssHeight;

  const cam = camera();
  const view = lookAt(cam.eye, glState.center, [0, 0, 1]);
  const near = Math.max(glState.distance * 0.002, 0.05);
  const far = Math.max(glState.distance + glState.radius * 8, glState.radius * 12);
  const projection = perspective(FOV, width / Math.max(height, 1), near, far);
  glState.lastView = view;
  glState.lastProjection = projection;

  gl.viewport(0, 0, width, height);
  gl.clearColor(0.039, 0.067, 0.102, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.disable(gl.BLEND);

  const normalMatrix = new Float32Array([
    view[0], view[1], view[2],
    view[4], view[5], view[6],
    view[8], view[9], view[10]
  ]);
  const scanAxisView = rotate3(view, SIDE_VECTORS[currentAxis()] || [0, 0, 1]);
  const styles = {
    part: { color: [0.62, 0.67, 0.71], highlight: 1 },
    support: { color: [0.49, 0.56, 0.61], highlight: 0 },
    base: { color: [0.36, 0.43, 0.49], highlight: 0 }
  };
  gl.useProgram(glState.meshProgram);
  gl.uniformMatrix4fv(glState.meshLoc.proj, false, projection);
  gl.uniformMatrix4fv(glState.meshLoc.view, false, view);
  gl.uniformMatrix3fv(glState.meshLoc.normal, false, normalMatrix);
  gl.uniform3fv(glState.meshLoc.axis, scanAxisView);
  gl.enableVertexAttribArray(glState.meshLoc.pos);
  gl.enableVertexAttribArray(glState.meshLoc.nrm);
  for (const name of ["base", "support", "part"]) {
    if (name === "support" && !$("showFixture").checked) continue;
    if (name === "base" && !$("showBase").checked) continue;
    const mesh = glState.meshes[name];
    if (!mesh) continue;
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.posBuffer);
    gl.vertexAttribPointer(glState.meshLoc.pos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.nrmBuffer);
    gl.vertexAttribPointer(glState.meshLoc.nrm, 3, gl.FLOAT, false, 0, 0);
    gl.uniform3fv(glState.meshLoc.color, styles[name].color);
    gl.uniform1f(glState.meshLoc.highlight, styles[name].highlight);
    gl.drawArrays(gl.TRIANGLES, 0, mesh.count);
  }

  gl.useProgram(glState.flatProgram);
  gl.uniformMatrix4fv(glState.flatLoc.proj, false, projection);
  gl.uniformMatrix4fv(glState.flatLoc.view, false, view);
  gl.enableVertexAttribArray(glState.flatLoc.pos);
  if (glState.meshLoc.nrm >= 0) gl.disableVertexAttribArray(glState.meshLoc.nrm);

  const shift = glState.radius * 0.004;
  if (state.result && state.showPath && $("showPath").checked && glState.pathLine) {
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.lineWidth(Math.max(1, Math.round(2 * dpr)));
    drawFlat(glState.pathLine, gl.LINES, glState.pathLineCount, [0.06, 0.72, 0.88], 1, 0, 0.95);
    drawFlat(glState.pathPoint, gl.POINTS, glState.pathPointCount, [1.0, 0.58, 0.12], Math.max(6, 8 * dpr), 0, 1.0);
    gl.disable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST);
  }

  drawOverlay(view, projection, cssWidth, cssHeight, dpr);
}

function drawFlat(buffer, mode, count, color, size, shift, alpha) {
  const gl = glState.gl;
  if (!buffer || count <= 0) return;
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.vertexAttribPointer(glState.flatLoc.pos, 3, gl.FLOAT, false, 0, 0);
  gl.uniform3fv(glState.flatLoc.color, color);
  gl.uniform1f(glState.flatLoc.size, size);
  gl.uniform1f(glState.flatLoc.shift, shift);
  gl.uniform1f(glState.flatLoc.alpha, alpha);
  gl.drawArrays(mode, 0, count);
}

function drawOverlay(view, projection, width, height, dpr) {
  const canvas = $("overlayCanvas");
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const points = state.picked.slice(0, state.pointCount);
  points.forEach((point, index) => {
    const projected = projectPoint(view, projection, width, height, point);
    if (!projected) return;
    ctx.beginPath();
    ctx.arc(projected.x, projected.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#1b6fb5";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#eaf5ff";
    ctx.stroke();
    const label = "P" + (index + 1);
    ctx.font = "bold 12px Segoe UI, Microsoft YaHei, sans-serif";
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(8, 22, 35, .82)";
    ctx.fillRect(projected.x + 10, projected.y - 19, textWidth + 10, 17);
    ctx.fillStyle = "#cde8ff";
    ctx.fillText(label, projected.x + 15, projected.y - 6);
  });
}

function requestRender() {
  if (glState.raf) return;
  glState.raf = requestAnimationFrame(render);
}

/* ---------- scene camera and picking ---------- */

function refreshBounds(forceFit) {
  const low = [Infinity, Infinity, Infinity];
  const high = [-Infinity, -Infinity, -Infinity];
  let any = false;
  for (const name of Object.keys(state.models)) {
    const bounds = state.models[name].bounds;
    any = true;
    for (let axis = 0; axis < 3; axis += 1) {
      low[axis] = Math.min(low[axis], bounds.low[axis]);
      high[axis] = Math.max(high[axis], bounds.high[axis]);
    }
  }
  if (!any) return;
  glState.center = [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, (low[2] + high[2]) / 2];
  glState.radius = Math.max(1, 0.5 * Math.hypot(high[0] - low[0], high[1] - low[1], high[2] - low[2]));
  glState.home = { center: glState.center.slice(), radius: glState.radius };
  if (forceFit || !glState.fitted) {
    fitView();
    glState.fitted = true;
  }
  $("stageEmpty").style.display = "none";
  requestRender();
}

function fitView() {
  if (!glState.home) return;
  glState.center = glState.home.center.slice();
  glState.radius = glState.home.radius;
  glState.distance = Math.max(glState.radius * 2.6, 10);
  glState.yaw = -Math.PI / 3;
  glState.pitch = Math.PI / 6;
  requestRender();
}

function lookAlongAxis() {
  if (!glState.home) return;
  glState.center = glState.home.center.slice();
  glState.distance = Math.max(glState.radius * 2.3, 10);
  const axis = currentAxis();
  const angles = {
    "+X": [0, 0], "-X": [Math.PI, 0],
    "+Y": [Math.PI / 2, 0], "-Y": [-Math.PI / 2, 0],
    "+Z": [-Math.PI / 2, Math.PI / 2 - 0.001], "-Z": [-Math.PI / 2, -Math.PI / 2 + 0.001]
  };
  glState.yaw = angles[axis][0];
  glState.pitch = angles[axis][1];
  requestRender();
}

function resetPicks() {
  state.picked = [];
  state.registration = null;
  registrationRequest += 1;
  window.clearTimeout(registrationTimer);
  renderDatumRows();
  updateSolveAvailability();
  requestRender();
  setRegistrationStatus("选够定位点并填写现场坐标后，会自动计算整车 CAD 坐标到现场测量坐标系的安装变换。");
}

function undoPoint() {
  if (!state.picked.length) {
    toast("当前没有可撤销的定位点");
    return;
  }
  const index = state.picked.length;
  state.picked.pop();
  state.registration = null;
  registrationRequest += 1;
  window.clearTimeout(registrationTimer);
  renderDatumRows();
  updateSolveAvailability();
  requestRender();
  setRegistrationStatus("已撤销 P" + index + "，下一次点击将重新选择该点。");
}

function pickScene(clientX, clientY) {
  const canvas = $("modelCanvas");
  const rect = canvas.getBoundingClientRect();
  if (!glState.lastView || !glState.lastProjection) return null;
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  const aspect = rect.width / Math.max(rect.height, 1);
  const ndcX = (x / rect.width) * 2 - 1;
  const ndcY = 1 - (y / rect.height) * 2;
  const cam = camera();
  const tan = Math.tan(FOV / 2);
  const direction = unit(add(cam.forward, add(mul(cam.right, ndcX * tan * aspect), mul(cam.up, ndcY * tan))));
  let best = null;
  for (const name of ["part", "support", "base"]) {
    if (name === "support" && !$("showFixture").checked) continue;
    if (name === "base" && !$("showBase").checked) continue;
    const model = state.models[name];
    if (!model) continue;
    const triangles = model.triangles;
    for (let index = 0; index < triangles.length; index += 1) {
      const tri = triangles[index];
      const edge1 = sub(tri[1], tri[0]);
      const edge2 = sub(tri[2], tri[0]);
      const pvec = cross(direction, edge2);
      const det = dot(edge1, pvec);
      if (Math.abs(det) < 1e-9) continue;
      const inv = 1 / det;
      const tvec = sub(cam.eye, tri[0]);
      const u = dot(tvec, pvec) * inv;
      if (u < -1e-6 || u > 1 + 1e-6) continue;
      const qvec = cross(tvec, edge1);
      const v = dot(direction, qvec) * inv;
      if (v < -1e-6 || u + v > 1 + 1e-6) continue;
      const t = dot(edge2, qvec) * inv;
      if (t <= 1e-6) continue;
      if (!best || t < best.t) {
        best = {
          t,
          model: name,
          point: add(cam.eye, mul(direction, t)),
          normal: unit(cross(edge1, edge2))
        };
      }
    }
  }
  return best;
}

function handleCanvasClick(clientX, clientY) {
  const hit = pickScene(clientX, clientY);
  if (!hit) {
    toast("没有点中模型表面，请对准零件点击");
    return;
  }
  if (hit.model !== "part") {
    toast("请点击零件表面选择定位点，不要点到夹具或底座");
    return;
  }
  if (state.autoFace) {
    const normal = hit.normal;
    const component = [Math.abs(normal[0]), Math.abs(normal[1]), Math.abs(normal[2])];
    let axisIndex = 0;
    if (component[1] > component[axisIndex]) axisIndex = 1;
    if (component[2] > component[axisIndex]) axisIndex = 2;
    const sign = normal[axisIndex] >= 0 ? "+" : "-";
    const axis = sign + "XYZ"[axisIndex];
    $("scanAxis").value = axis;
    state.autoFace = false;
    $("pickAxis").classList.remove("active");
    updateAxisBadge("已按点击表面设为 " + axis);
    renderDatumRows();
    requestRender();
    toast("已根据点击表面选择扫描面 " + axis);
    return;
  }
  if (state.picked.length >= state.pointCount) {
    toast("已选满 " + state.pointCount + " 个定位点；右键撤销后再选，或切换为 4 点");
    return;
  }
  state.picked.push([round(hit.point[0]), round(hit.point[1]), round(hit.point[2])]);
  state.registration = null;
  registrationRequest += 1;
  renderDatumRows();
  updateSolveAvailability();
  requestRender();
  setRegistrationStatus("已选 P" + state.picked.length + "，请继续选择并填写现场坐标。");
  scheduleRegistrationCheck();
}

function round(value) { return Math.round(value * 1000) / 1000; }

/* ---------- UI: models and datum rows ---------- */

function parseStl(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength >= 84) {
    const count = view.getUint32(80, true);
    if (count > 0 && 84 + count * 50 <= buffer.byteLength) {
      const triangles = new Array(count);
      let offset = 84;
      for (let index = 0; index < count; index += 1) {
        offset += 12;
        const triangle = [];
        for (let vertex = 0; vertex < 3; vertex += 1) {
          triangle.push([
            view.getFloat32(offset, true),
            view.getFloat32(offset + 4, true),
            view.getFloat32(offset + 8, true)
          ]);
          offset += 12;
        }
        offset += 2;
        triangles[index] = triangle;
      }
      return triangles;
    }
  }
  const text = new TextDecoder().decode(buffer);
  const values = [...text.matchAll(/vertex\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)/gi)];
  if (!values.length || values.length % 3) {
    throw new Error("无法识别 STL：文件需为有效的 ASCII 或 Binary STL。");
  }
  const triangles = [];
  for (let index = 0; index < values.length; index += 3) {
    triangles.push(values.slice(index, index + 3).map((match) => [
      Number(match[1]), Number(match[2]), Number(match[3])
    ]));
  }
  return triangles;
}

function fileBase64(file) {
  return file.arrayBuffer().then((buffer) => {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    for (let start = 0; start < bytes.length; start += 0x8000) {
      binary += String.fromCharCode(...bytes.subarray(start, Math.min(start + 0x8000, bytes.length)));
    }
    return btoa(binary);
  });
}

function base64ToBuffer(encoded) {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes.buffer;
}

async function convertToStl(file) {
  const data = await fileBase64(file);
  const response = await fetch("/api/convert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: file.name, data_base64: data })
  });
  let body;
  try {
    body = await response.json();
  } catch (_) {
    throw new Error("模型转换服务返回了无法解析的响应（HTTP " + response.status + "）");
  }
  if (!response.ok || !body.ok) throw new Error(body.error || "模型转换失败");
  return { buffer: base64ToBuffer(body.stl_base64), faces: Number(body.faces || 0) };
}

async function loadModel(name, file) {
  if (!file) return;
  let buffer = await file.arrayBuffer();
  let stlFile = file;
  if (!/\.stl$/i.test(file.name || "")) {
    $("formFeedback").textContent = "正在将 " + file.name + " 转换为三角网格…";
    const converted = await convertToStl(file);
    buffer = converted.buffer;
    stlFile = new File(
      [buffer],
      (file.name || "model").replace(/\.[^.]+$/, "") + ".stl",
      { type: "model/stl" }
    );
    $("formFeedback").textContent = "";
  }
  const triangles = parseStl(buffer);
  if (triangles.length > MAX_TRIANGLES) {
    throw new Error(MODEL_LABELS[name] + " 三角面超过 " + MAX_TRIANGLES.toLocaleString() + "，请先简化网格");
  }
  state.models[name] = {
    file: stlFile,
    sourceName: file.name,
    triangles,
    bounds: computeBounds(triangles),
    base64: null
  };
  buildMeshBuffers(name, triangles);
  const card = document.querySelector('label[for="' + name + 'File"]');
  if (card) card.classList.add("has-file");
  const target = $(name + "Name");
  if (target) target.textContent = (file.name || "模型") + " · " + triangles.length.toLocaleString() + " 面";
  if (name === "part") {
    state.picked = [];
    state.registration = null;
    registrationRequest += 1;
    renderDatumRows();
    recommendAxis(false);
  }
  refreshBounds(name === "part" || Object.keys(state.models).length === 1);
  updateSolveAvailability();
  toast(MODEL_LABELS[name] + " 已导入");
}

function renderDatumRows() {
  const body = $("datumRows");
  body.replaceChildren();
  for (let index = 0; index < state.pointCount; index += 1) {
    const row = document.createElement("tr");
    const label = document.createElement("td");
    label.className = "point";
    label.textContent = "P" + (index + 1);
    const cad = document.createElement("td");
    if (state.picked[index]) {
      cad.className = "cad";
      cad.textContent = state.picked[index].map((value) => value.toFixed(2)).join(", ");
    } else {
      cad.className = "cad empty";
      cad.textContent = "在 3D 模型上点击";
    }
    row.append(label, cad);
    for (let axis = 0; axis < 3; axis += 1) {
      const cell = document.createElement("td");
      const input = document.createElement("input");
      input.type = "number";
      input.step = "0.001";
      input.dataset.measure = String(index);
      input.dataset.axis = String(axis);
      input.value = state.measured[index][axis] || "";
      input.placeholder = "mm";
      input.addEventListener("input", () => {
        state.measured[index][axis] = input.value;
        mirrorDatumInput("relocateRows", index, axis, input.value);
        state.registration = null;
        updateSolveAvailability();
        scheduleRegistrationCheck();
      });
      cell.append(input);
      row.append(cell);
    }
    body.append(row);
  }
  $("pointCounter").textContent = "已选 " + state.picked.length + " / " + state.pointCount;
  $("canvasCounter").textContent = state.picked.length + " / " + state.pointCount;
  $("pickInstruction").textContent = state.autoFace
    ? "点击零件主扫描面，自动推断整车坐标轴方向"
    : "按顺序点击零件表面的 P1…P" + state.pointCount + "；右键撤销最后一点";
}

function measuredPoints() {
  const rows = [];
  for (let index = 0; index < state.pointCount; index += 1) {
    const row = state.measured[index].map((value) => {
      const number = Number(value);
      if (value === "" || !Number.isFinite(number)) {
        throw new Error("请填写现场 P" + (index + 1) + " 的 X/Y/Z 坐标");
      }
      return number;
    });
    rows.push(row);
  }
  return rows;
}

function hasMeasuredPoints() {
  for (let index = 0; index < state.pointCount; index += 1) {
    if (state.measured[index].some((value) => value === "" || !Number.isFinite(Number(value)))) {
      return false;
    }
  }
  return true;
}

function setPointCount(count) {
  state.pointCount = count;
  if (state.picked.length > count) state.picked = state.picked.slice(0, count);
  state.registration = null;
  registrationRequest += 1;
  window.clearTimeout(registrationTimer);
  document.querySelectorAll("[data-point-count]").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.pointCount) === count);
  });
  renderDatumRows();
  updateSolveAvailability();
  requestRender();
  scheduleRegistrationCheck();
}

function currentAxis() { return $("scanAxis").value; }

function updateAxisBadge(text) {
  $("axisBadge").textContent = text;
}

function recommendAxis(showToast) {
  const model = state.models.part;
  if (!model) {
    updateAxisBadge("尚未导入零件");
    if (showToast) toast("请先导入零件 STL");
    return;
  }
  const areas = { "+X": 0, "-X": 0, "+Y": 0, "-Y": 0, "+Z": 0, "-Z": 0 };
  for (const triangle of model.triangles) {
    const edge1 = sub(triangle[1], triangle[0]);
    const edge2 = sub(triangle[2], triangle[0]);
    const crossProduct = cross(edge1, edge2);
    const normal = unit(crossProduct);
    const area = 0.5 * Math.hypot(crossProduct[0], crossProduct[1], crossProduct[2]);
    for (const axis of Object.keys(SIDE_VECTORS)) {
      if (dot(normal, SIDE_VECTORS[axis]) > 0.75) areas[axis] += area;
    }
  }
  let bestAxis = "+Z";
  let bestArea = -1;
  for (const axis of Object.keys(areas)) {
    if (areas[axis] > bestArea) {
      bestArea = areas[axis];
      bestAxis = axis;
    }
  }
  if (bestArea <= 0) {
    updateAxisBadge("模型没有明显平整面，请手动选择扫描面");
    if (showToast) toast("未能从模型统计出主平整面，请手动选择");
    return;
  }
  $("scanAxis").value = bestAxis;
  updateAxisBadge("模型推荐：" + AXIS_LABELS[bestAxis] + " · 主平整面约 " + (bestArea / 1e6).toFixed(2) + " m²");
  requestRender();
  if (showToast) toast("已推荐扫描面 " + bestAxis);
}

function updateSolveAvailability() {
  const tolerance = Number($("errorLimit").value);
  const ready = Boolean(state.models.part && state.models.support)
    && state.picked.length === state.pointCount
    && hasMeasuredPoints()
    && Boolean(state.registration)
    && Number.isFinite(tolerance) && tolerance > 0
    && state.registration.max_error_mm <= tolerance
    && !state.solving;
  $("solveButton").disabled = !ready;
  updateWorkflow();
  updateRelocateAvailability();
}

function updateWorkflow() {
  const items = document.querySelectorAll("#workflowList li");
  const stage = !state.models.part || !state.models.support ? 0
    : state.picked.length < state.pointCount || !state.registration ? 1 : 2;
  items.forEach((item, index) => {
    item.classList.toggle("current", index === stage);
    item.classList.toggle("done", index < stage);
  });
}

/* ---------- registration and planning ---------- */

function payload(includeMeshes) {
  if (state.picked.length !== state.pointCount) {
    throw new Error("请在 3D 模型上依次选取 " + state.pointCount + " 个定位点");
  }
  const frame = Number($("measureFrame").value);
  const program = $("progName").value.trim().toUpperCase();
  if (!/^[A-Z][A-Z0-9_]{0,15}$/.test(program)) {
    throw new Error("LS 程序名必须以字母开头，仅含 A–Z、0–9 和下划线，最多 16 字符");
  }
  const runName = $("runName").value.trim();
  const data = {
    cad_points_vehicle_mm: state.picked.slice(0, state.pointCount),
    measured_points_mm: measuredPoints(),
    measured_frame: frame === 0 ? "robot_base" : "robot_uframe_1",
    uframe: frame,
    scan_face_axis: SIDE_VECTORS[currentAxis()],
    max_registration_error_mm: Number($("errorLimit").value),
    prog_name: program,
    delivery_mode: $("scanMode").value,
    fast: $("fastMode").value === "fast",
    fast_no_occlusion: $("fastMode").value === "fast" && $("fastNoOcclusion").checked,
    run_name: runName || undefined,
    run_id: /^[A-Za-z0-9_-]{1,64}$/.test(runName) ? runName : undefined
  };
  if (includeMeshes) {
    data.models = {};
    for (const name of ["part", "support", "base"]) {
      if (state.models[name]) data.models[name] = { stl_base64: state.models[name].base64 };
    }
  }
  return data;
}

async function prepareMeshPayload() {
  for (const name of ["part", "support", "base"]) {
    if (state.models[name] && !state.models[name].base64) {
      state.models[name].base64 = await fileBase64(state.models[name].file);
    }
  }
}

function setRegistrationStatus(message, kind = "") {
  const node = $("registrationStatus");
  node.className = "reg-status" + (kind ? " " + kind : "");
  node.textContent = message;
}

function scheduleRegistrationCheck() {
  window.clearTimeout(registrationTimer);
  registrationRequest += 1;
  if (state.picked.length !== state.pointCount || !hasMeasuredPoints()) return;
  setRegistrationStatus("正在检查配准…", "busy");
  registrationTimer = window.setTimeout(checkRegistration, 220);
}

async function checkRegistration() {
  const request = ++registrationRequest;
  try {
    const body = await fetchJson("/api/register", payload(false));
    if (request !== registrationRequest) return;
    state.registration = body.result;
    const errors = body.result.point_errors_mm.map((value, index) => "P" + (index + 1) + " " + value.toFixed(3) + " mm").join(" · ");
    const translation = body.result.translation_vehicle_origin_in_measured_mm.map((value) => value.toFixed(3)).join(", ");
    const tolerance = Number($("errorLimit").value);
    const ok = body.result.max_error_mm <= tolerance;
    setRegistrationStatus(
      "安装变换：整车 CAD → " + body.result.measured_frame
      + "\n平移 XYZ：" + translation + " mm"
      + "\n配准 RMS " + body.result.rms_error_mm.toFixed(3) + " mm；最大误差 " + body.result.max_error_mm.toFixed(3) + " mm（" + errors + "）",
      ok ? "ok" : "bad"
    );
    updateSolveAvailability();
    updateRelocateResiduals();
    if (state.result) {
      setRelocateStatus(
        ok
          ? "新坐标已校核：最大残差 " + body.result.max_error_mm.toFixed(3) + " mm，可点「按新坐标重新定位路径」变换路径。"
          : "新坐标超出限值（最大残差 " + body.result.max_error_mm.toFixed(3) + " mm），请核对点位或放宽限值。",
        ok ? "ok" : "bad"
      );
    }
  } catch (error) {
    if (request !== registrationRequest) return;
    state.registration = null;
    updateSolveAvailability();
    updateRelocateResiduals();
    setRegistrationStatus(error.message, "bad");
  }
}

async function fetchJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  let data;
  try {
    data = await response.json();
  } catch (_) {
    throw new Error("服务返回了无法解析的响应（HTTP " + response.status + "）");
  }
  if (!response.ok || !data.ok) throw new Error(data.error || "请求失败");
  return data;
}

// 规划请求体可能带有几十 MB 的 base64 网格，用 XHR 才能拿到上传字节进度。
function postJsonWithUpload(url, body, handlers = {}) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", url);
    request.setRequestHeader("Content-Type", "application/json");
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable && handlers.onUpload) {
        handlers.onUpload(event.loaded / event.total);
      }
    });
    request.upload.addEventListener("load", () => {
      if (handlers.onUploaded) handlers.onUploaded();
    });
    request.addEventListener("load", () => {
      let data;
      try {
        data = JSON.parse(request.responseText);
      } catch (_) {
        reject(new Error("服务返回了无法解析的响应（HTTP " + request.status + "）"));
        return;
      }
      if (request.status < 200 || request.status >= 300 || !data.ok) {
        reject(new Error((data && data.error) || "请求失败"));
        return;
      }
      resolve(data);
    });
    request.addEventListener("error", () => reject(new Error("无法连接本机规划服务，请确认命令行仍在运行")));
    request.addEventListener("abort", () => reject(new Error("规划请求已取消")));
    request.send(JSON.stringify(body));
  });
}

/* ---------- planning progress bar ---------- */

function newProgressToken() {
  const raw = window.crypto && window.crypto.randomUUID
    ? window.crypto.randomUUID()
    : String(Date.now()) + Math.random().toString(16).slice(2);
  return raw.replace(/[^A-Za-z0-9_-]/g, "").slice(0, 40) || "job";
}

// 规划是一个同步请求，服务端把阶段写进 /api/progress，页面在这里轮询它的快照。
// 返回停止函数；轮询本身是可选的，失败不影响任何结果。
function pollProgress(token, onUpdate) {
  let stopped = false;
  let timer = 0;
  const tick = async () => {
    if (stopped) return;
    try {
      const response = await fetch("/api/progress?id=" + encodeURIComponent(token),
        { cache: "no-store" });
      if (response.ok) {
        const data = await response.json();
        if (!stopped && data && data.ok && data.progress) onUpdate(data.progress);
      }
    } catch (_error) {
      /* 进度只是展示，断一次不影响服务端 */
    }
    if (!stopped) timer = window.setTimeout(tick, 400);
  };
  tick();
  return () => {
    stopped = true;
    window.clearTimeout(timer);
  };
}

// 阶段顺序与服务端上报的 stage 键一一对应：进度条走到哪一步由真实的阶段决定，
// 不再用时间曲线去猜。上传阶段的百分比仍来自真实上传字节。
const PLAN_STAGES = [
  { key: "upload", label: "上传模型" },
  { key: "registration", label: "坐标配准" },
  { key: "segmentation", label: "目标面分割" },
  { key: "candidates", label: "候选视点" },
  { key: "occlusion", label: "遮挡计算" },
  { key: "strokes", label: "扫描带规划" },
  { key: "report", label: "报告生成" },
  { key: "export", label: "交付文件导出" }
];
// 上传占 0–15%，服务端自己的 0–100% 线性映射到 15–99%。
const UPLOAD_CEILING = 15;
const SOLVE_FLOOR = 15;
const SOLVE_CEILING = 99;
// 服务端超过这么久没有新阶段，说明可能是旧版本或某个阶段确实很慢：只允许缓慢爬升到 90%。
const PROGRESS_STALE_MS = 8000;

const planProgress = {
  timer: 0,
  value: 0,
  stage: 0,
  uploadRatio: 0,
  solveStart: 0,
  serverAt: 0,
  serverDetail: "",
  stopPolling: null,

  start() {
    this.stop();
    this.value = 0;
    this.stage = -1;
    this.uploadRatio = 0;
    this.solveStart = 0;
    this.serverAt = 0;
    this.serverDetail = "";
    const node = $("planProgress");
    if (!node) return;
    node.classList.remove("is-done", "is-failed");
    document.querySelectorAll("#progressSteps li").forEach((item) => {
      item.classList.remove("done", "active");
    });
    this.caption("", 0);
    this.markStage(0);
    this.render();
    this.timer = window.setInterval(() => this.tick(), 140);
  },

  stop() {
    if (this.timer) window.clearInterval(this.timer);
    this.timer = 0;
    if (this.stopPolling) {
      this.stopPolling();
      this.stopPolling = null;
    }
  },

  caption(detail, elapsedSeconds) {
    const node = $("progressCaption");
    if (!node) return;
    const text = detail ? String(detail).toUpperCase() : "GEOMETRY ANALYSIS IN PROGRESS";
    node.textContent = text + (elapsedSeconds > 0
      ? " · " + Math.round(elapsedSeconds) + " S"
      : "");
  },

  // 服务端阶段快照：百分比直接采用，只有阶段键不在表里（done/failed）时才保持原位。
  applyServer(progress) {
    if (!progress) return;
    this.serverAt = Date.now();
    this.serverDetail = progress.detail || "";
    const pct = Number(progress.pct);
    if (Number.isFinite(pct)) {
      const mapped = SOLVE_FLOOR
        + (SOLVE_CEILING - SOLVE_FLOOR) * Math.max(0, Math.min(100, pct)) / 100;
      this.value = Math.max(this.value, Math.min(SOLVE_CEILING, mapped));
    }
    const index = PLAN_STAGES.findIndex((stage) => stage.key === progress.stage);
    if (index >= 0) this.markStage(index);
    else this.markStage(PLAN_STAGES.length - 1);
    this.caption(this.serverDetail, Number(progress.elapsed_s) || 0);
    this.render();
  },

  markStage(index) {
    const stage = PLAN_STAGES[index];
    if (!stage) return;
    if (index === this.stage && $("progressStage").textContent === stage.label) return;
    this.stage = index;
    $("progressStage").textContent = stage.label;
    document.querySelectorAll("#progressSteps li").forEach((item, position) => {
      item.classList.toggle("done", position < index);
      item.classList.toggle("active", position === index);
    });
  },

  render() {
    const value = Math.max(0, Math.min(100, this.value));
    const fill = $("progressFill");
    if (!fill) return;
    fill.style.width = value.toFixed(1) + "%";
    $("progressPercent").innerHTML = Math.round(value) + "<small>%</small>";
    $("planProgress").setAttribute("aria-valuenow", String(Math.round(value)));
  },

  tick() {
    if (this.solveStart) {
      const seconds = (Date.now() - this.solveStart) / 1000;
      const fresh = this.serverAt && Date.now() - this.serverAt < PROGRESS_STALE_MS;
      if (!fresh) {
        // 没有阶段回报时不假装知道进度：只缓慢爬升到 90%，绝不提前冲到 99%。
        const estimate = SOLVE_FLOOR + 75 * (1 - Math.exp(-seconds / 120));
        this.value = Math.max(this.value, Math.min(90, estimate));
      }
      this.caption(this.serverDetail, seconds);
    } else {
      // 上传阶段：真实上传字节比为主，读文件 / 转 base64 时缓慢爬升。
      const estimate = Math.min(UPLOAD_CEILING, 1 + this.uploadRatio * (UPLOAD_CEILING - 1));
      this.value = Math.max(this.value + 0.1, estimate);
    }
    this.render();
  },

  setUploadRatio(ratio) {
    if (!Number.isFinite(ratio)) return;
    this.uploadRatio = Math.max(this.uploadRatio, Math.min(1, ratio));
    if (!this.solveStart) {
      this.value = Math.max(this.value,
        Math.min(UPLOAD_CEILING, 1 + this.uploadRatio * (UPLOAD_CEILING - 1)));
      this.render();
    }
  },

  beginSolve(token) {
    this.solveStart = this.solveStart || Date.now();
    this.value = Math.max(this.value, SOLVE_FLOOR);
    this.render();
    if (token && !this.stopPolling) {
      this.stopPolling = pollProgress(token, (progress) => this.applyServer(progress));
    }
  },

  complete() {
    this.stop();
    this.value = 100;
    this.markStage(PLAN_STAGES.length - 1);
    document.querySelectorAll("#progressSteps li").forEach((item) => {
      item.classList.remove("active");
      item.classList.add("done");
    });
    $("progressStage").textContent = "规划完成";
    $("planProgress").classList.remove("is-failed");
    $("planProgress").classList.add("is-done");
    $("progressCaption").textContent = "PLANNING COMPLETE";
    this.render();
  },

  fail() {
    this.stop();
    $("planProgress").classList.remove("is-done");
    $("planProgress").classList.add("is-failed");
    $("progressStage").textContent = "规划中断";
    $("progressCaption").textContent = "PLANNING FAILED";
    document.querySelectorAll("#progressSteps li.active").forEach((item) => {
      item.classList.remove("active");
    });
    this.render();
  }
};

async function solve() {
  if (state.solving) return;
  try {
    payload(false);
  } catch (error) {
    $("formFeedback").textContent = error.message;
    toast(error.message);
    return;
  }
  state.solving = true;
  state.result = null;
  renderRelocatePanel(null);
  $("formFeedback").textContent = "";
  $("showPath").checked = false;
  $("showPath").disabled = true;
  $("openViewer").disabled = true;
  $("downloadReport").disabled = true;
  updateSolveAvailability();
  showResultState("progress");
  scrollToId("resultPanel");
  $("progressTitle").textContent = "正在上传模型并规划扫描路径";
  $("progressMessage").textContent = "正在执行配准、目标面分割、候选视点与遮挡计算。零件复杂时可能需要几分钟，请勿关闭页面。";
  try {
    await prepareMeshPayload();
    const token = newProgressToken();
    const requestBody = payload(true);
    requestBody.progress_id = token;
    const body = await postJsonWithUpload("/api/solve", requestBody, {
      onUpload: (ratio) => planProgress.setUploadRatio(ratio),
      onUploaded: () => planProgress.beginSolve(token)
    });
    planProgress.complete();
    state.result = body.result;
    state.resultSource = "live";
    state.showPath = true;
    $("showPath").checked = true;
    $("showPath").disabled = false;
    buildPathBuffers();
    syncSetupFromResult(body.result);
    renderResult(body.result);
    await delay(520);
    showResultState("result");
    requestRender();
    scrollToId("resultPanel");
    toast("扫描路径规划完成");
  } catch (error) {
    planProgress.fail();
    $("progressTitle").textContent = "规划未能完成";
    $("progressMessage").textContent = error.message;
    $("progressState").querySelector(".spinner").style.display = "none";
    toast(error.message);
  } finally {
    state.solving = false;
    updateSolveAvailability();
  }
}

function buildPathBuffers() {
  const gl = glState.gl;
  if (!gl || !state.result || !Array.isArray(state.result.viewpoints)) return;
  const points = state.result.viewpoints
    .slice()
    .sort((a, b) => a.order - b.order)
    .map((viewpoint) => viewpoint.position);
  const line = [];
  for (let index = 0; index < points.length - 1; index += 1) {
    line.push(...points[index], ...points[index + 1]);
  }
  const lineData = new Float32Array(line);
  const pointData = new Float32Array(points.flat());
  if (!glState.pathLine) glState.pathLine = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, glState.pathLine);
  gl.bufferData(gl.ARRAY_BUFFER, lineData, gl.STATIC_DRAW);
  glState.pathLineCount = line.length / 3;
  if (!glState.pathPoint) glState.pathPoint = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, glState.pathPoint);
  gl.bufferData(gl.ARRAY_BUFFER, pointData, gl.STATIC_DRAW);
  glState.pathPointCount = points.length;
}

function formatElapsed(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value <= 0) return "—";
  // 先整体取整再拆分/秒，否则 239.6 s 会先算成 3 分再四舍五入成 60 秒。
  const rounded = Math.round(value * 10) / 10;
  if (rounded < 60) return rounded.toFixed(1) + "<small>秒</small>";
  const total = Math.round(rounded);
  const minutes = Math.floor(total / 60);
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60);
    const restMinutes = String(minutes - hours * 60).padStart(2, "0");
    return hours + "<small>时</small>" + restMinutes + "<small>分</small>";
  }
  const rest = String(total - minutes * 60).padStart(2, "0");
  return minutes + "<small>分</small>" + rest + "<small>秒</small>";
}

function renderResult(result) {
  $("resultName").textContent = result.program_name + " · 扫描路径规划";
  $("resultSubtitle").textContent = "目标面 " + (result.face || "—") + " · " + result.datum_count + " 点配准 · " + result.measured_frame;
  $("metricCoverage").innerHTML = Number(result.coverage_pct).toFixed(2) + "<small>%</small>";
  $("metricViewpoints").textContent = result.point_count;
  $("metricPoints").textContent = Number(result.target_points).toLocaleString();
  $("metricUncovered").textContent = Number(result.uncovered_points).toLocaleString();
  $("metricElapsed").innerHTML = formatElapsed(result.elapsed_seconds);
  const reasons = Object.entries(result.uncovered_by_reason || {})
    .map(([key, value]) => key + " " + value).join(" · ") || "无";
  const details = [
    ["运行编号", result.run_id],
    ["LS 程序名", result.program_name],
    ["交付模式", result.delivery_mode === "stroke"
      ? "扫掠模式 · 连续扫描带"
      : "纯位姿模式（旧版）"],
    ["规划速度", result.fast && result.fast.enabled
      ? "快速 · " + Number(result.fast.sample_grid_mm).toFixed(0) + " mm 采样 · 遮挡"
        + (result.fast.occlusion_modelled ? "已建模" : "未建模")
        + "（数字只描述路径形状，不能当验收依据）"
      : "精确" + (result.sample_grid_mm
        ? " · " + Number(result.sample_grid_mm).toFixed(0) + " mm 采样" : "")],
    ["扫描面外法向（整车）", "[" + result.scan_face_axis_vehicle.map((value) => Number(value).toFixed(3)).join(", ") + "]"],
    ["目标面面积", (Number(result.target_face_mm2) / 1e6).toFixed(4) + " m²"],
    ["配准 RMS / 最大误差", result.registration_rms_error_mm.toFixed(3) + " / " + result.registration_max_error_mm.toFixed(3) + " mm"],
    ...(result.relocalised
      ? [["定位方式", "自 " + result.relocated_from + " 变换 "
          + Number(result.shift_mm || 0).toFixed(2) + " mm（未重新规划）"]]
      : []),
    ["底座", result.base_supplied ? "已导入" : "未导入 · 已自动生成参考板"],
    ["未覆盖原因", reasons],
    ["输出目录", result.output_dir]
  ];
  const container = $("resultDetails");
  container.replaceChildren();
  for (const [label, value] of details) {
    const item = document.createElement("div");
    item.className = "detail-item"
      + (label === "未覆盖原因" || label === "输出目录" || label === "规划速度"
        ? " wide" : "");
    const title = document.createElement("span");
    title.textContent = label;
    const content = document.createElement("b");
    content.textContent = value;
    item.append(title, content);
    container.append(item);
  }
  const list = $("downloadList");
  list.replaceChildren();
  for (const file of result.downloads) {
    const link = document.createElement("a");
    link.className = "download-link";
    link.href = "/api/file?run_id=" + encodeURIComponent(result.run_id) + "&name=" + encodeURIComponent(file.name);
    link.download = file.name;
    const title = document.createElement("span");
    title.textContent = file.label;
    const action = document.createElement("span");
    action.textContent = "DOWNLOAD ↓";
    link.append(title, action);
    list.append(link);
  }
  $("openViewer").dataset.url = result.viewer_url;
  $("openViewer").disabled = false;
  $("downloadReport").dataset.url = "/api/file?run_id=" + encodeURIComponent(result.run_id) + "&name=planning_report.html";
  $("downloadReport").disabled = false;
  renderRelocatePanel(result);
}

/* ---------- re-localisation after a plan ---------- */

// The page keeps one datum state: the step-02 table and the result panel's review
// table read and write the same rows, so editing either one cannot leave the other
// showing coordinates the next run will not use.
function mirrorDatumInput(containerId, index, axis, value) {
  const target = document.querySelector(
    "#" + containerId + ' input[data-measure="' + index + '"][data-axis="' + axis + '"]'
  );
  if (target && target.value !== value) target.value = value;
}

function axisFromVector(vector) {
  const direction = unit(vector.map(Number));
  let best = null;
  let score = -Infinity;
  for (const axis of Object.keys(SIDE_VECTORS)) {
    const value = dot(direction, SIDE_VECTORS[axis]);
    if (value > score) {
      score = value;
      best = axis;
    }
  }
  return best;
}

// Restore what a completed run actually used, so its datum points can be corrected
// afterwards instead of starting over from an empty table.
function syncSetupFromResult(result) {
  const cad = result && result.cad_points_vehicle_mm;
  const measured = result && result.measured_points_mm;
  if (!Array.isArray(cad) || !cad.length
    || !Array.isArray(measured) || measured.length !== cad.length) {
    return false;
  }
  state.pointCount = Math.min(cad.length, 4);
  state.picked = cad.map((point) => point.map((value) => round(Number(value))));
  const sameAsTyped = measured.every((point, index) => point.every((value, axis) => {
    const typed = state.measured[index] && state.measured[index][axis];
    return typed !== "" && Number.isFinite(Number(typed))
      && Math.abs(Number(typed) - Number(value)) < 1e-6;
  }));
  if (!sameAsTyped) {
    state.measured = Array.from({ length: 4 }, () => ["", "", ""]);
    measured.forEach((point, index) => {
      state.measured[index] = point.map((value) => String(round(Number(value))));
    });
  }
  state.registration = {
    measured_frame: result.measured_frame,
    rms_error_mm: Number(result.registration_rms_error_mm) || 0,
    max_error_mm: Number(result.registration_max_error_mm) || 0,
    point_errors_mm: Array.isArray(result.point_errors_mm) ? result.point_errors_mm : [],
    translation_vehicle_origin_in_measured_mm: []
  };
  if (result.program_name) $("progName").value = result.program_name;
  if (result.delivery_mode) $("scanMode").value = result.delivery_mode;
  const wasFast = !!(result.fast && result.fast.enabled);
  $("fastMode").value = wasFast ? "fast" : "accurate";
  $("fastNoOcclusion").checked = wasFast && !result.fast.occlusion_modelled;
  $("fastNoOcclusion").disabled = !wasFast;
  if (Number.isFinite(Number(result.uframe))) {
    $("measureFrame").value = String(Number(result.uframe));
  }
  if (Number.isFinite(Number(result.error_limit_mm))) {
    $("errorLimit").value = String(Number(result.error_limit_mm));
    $("relocateLimit").value = String(Number(result.error_limit_mm));
  }
  if (Array.isArray(result.scan_face_axis_vehicle)
    && result.scan_face_axis_vehicle.length === 3) {
    const axis = axisFromVector(result.scan_face_axis_vehicle);
    if (axis) {
      $("scanAxis").value = axis;
      updateAxisBadge("本次运行：" + AXIS_LABELS[axis]);
    }
  }
  document.querySelectorAll("[data-point-count]").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.pointCount) === state.pointCount);
  });
  renderDatumRows();
  updateSolveAvailability();
  requestRender();
  return true;
}

function setRelocateStatus(message, kind = "") {
  const node = $("relocateStatus");
  if (!node) return;
  node.className = "relocate-status" + (kind ? " " + kind : "");
  node.textContent = message;
}

function renderRelocatePanel(result) {
  const panel = $("relocatePanel");
  if (!panel) return;
  const cad = result && result.cad_points_vehicle_mm;
  if (!Array.isArray(cad) || !cad.length) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  const body = $("relocateRows");
  body.replaceChildren();
  for (let index = 0; index < cad.length; index += 1) {
    const row = document.createElement("tr");
    const label = document.createElement("td");
    label.className = "point";
    label.textContent = "P" + (index + 1);
    const cadCell = document.createElement("td");
    cadCell.className = "cad";
    cadCell.textContent = cad[index].map((value) => Number(value).toFixed(2)).join(", ");
    row.append(label, cadCell);
    for (let axis = 0; axis < 3; axis += 1) {
      const cell = document.createElement("td");
      const input = document.createElement("input");
      input.type = "number";
      input.step = "0.001";
      input.dataset.measure = String(index);
      input.dataset.axis = String(axis);
      input.value = (state.measured[index] && state.measured[index][axis]) || "";
      input.placeholder = "mm";
      input.addEventListener("input", () => {
        state.measured[index][axis] = input.value;
        mirrorDatumInput("datumRows", index, axis, input.value);
        state.registration = null;
        updateSolveAvailability();
        scheduleRegistrationCheck();
      });
      cell.append(input);
      row.append(cell);
    }
    const residual = document.createElement("td");
    residual.className = "residual";
    residual.textContent = "—";
    row.append(residual);
    body.append(row);
  }
  updateRelocateResiduals();
  updateRelocateAvailability();
  if (!state.relocate.busy) {
    setRelocateStatus(
      "改动现场坐标后会自动重算配准残差；确认无误后点「按新坐标重新定位路径」，"
      + "把已规划的路径整体变换到新坐标系，不重新规划。"
    );
  }
}

function updateRelocateResiduals() {
  const cells = document.querySelectorAll("#relocateRows td.residual");
  if (!cells.length) return;
  const errors = state.registration && Array.isArray(state.registration.point_errors_mm)
    ? state.registration.point_errors_mm
    : [];
  const tolerance = Number($("relocateLimit").value);
  cells.forEach((cell, index) => {
    const value = Number(errors[index]);
    if (!Number.isFinite(value)) {
      cell.textContent = "—";
      cell.classList.remove("bad");
      return;
    }
    cell.textContent = value.toFixed(3);
    cell.classList.toggle("bad", Number.isFinite(tolerance) && value > tolerance);
  });
}

function updateRelocateAvailability() {
  const button = $("relocateButton");
  if (!button) return;
  const tolerance = Number($("relocateLimit").value);
  const ready = Boolean(state.result && state.result.run_id)
    && Boolean(state.registration)
    && state.picked.length === state.pointCount
    && hasMeasuredPoints()
    && Number.isFinite(tolerance) && tolerance > 0
    && Number(state.registration.max_error_mm) <= tolerance
    && !state.relocate.busy
    && !state.solving;
  button.disabled = !ready;
}

async function relocate() {
  if (state.relocate.busy || state.solving) return;
  const result = state.result;
  if (!result || !result.run_id) {
    toast("请先完成一次规划，再修改定位点重新定位");
    return;
  }
  let body;
  try {
    const tolerance = Number($("relocateLimit").value);
    if (!Number.isFinite(tolerance) || tolerance <= 0) {
      throw new Error("配准最大误差限值必须是正数");
    }
    if (state.picked.length !== state.pointCount) {
      throw new Error("定位点数量已改变，请先在 3D 模型上选满 " + state.pointCount + " 个点");
    }
    body = {
      run_id: result.run_id,
      cad_points_vehicle_mm: state.picked.slice(0, state.pointCount),
      measured_points_mm: measuredPoints(),
      max_registration_error_mm: tolerance,
      prog_name: $("progName").value.trim().toUpperCase() || result.program_name,
      delivery_mode: $("scanMode").value || result.delivery_mode,
      scan_face_axis: SIDE_VECTORS[currentAxis()],
      uframe: Number($("measureFrame").value),
      run_name: $("runName").value.trim() || undefined,
      progress_id: newProgressToken()
    };
  } catch (error) {
    setRelocateStatus(error.message, "bad");
    toast(error.message);
    return;
  }
  state.relocate.busy = true;
  const token = state.relocate.token;
  updateRelocateAvailability();
  const started = Date.now();
  setRelocateStatus("正在把扫描路径变换到新坐标系…", "busy");
  let lastReport = 0;
  const stopPolling = pollProgress(body.progress_id, (progress) => {
    lastReport = Date.now();
    const stage = PLAN_STAGES.find((entry) => entry.key === progress.stage);
    setRelocateStatus(
      "正在把扫描路径变换到新坐标系 · " + (stage ? stage.label : "收尾")
      + " " + Number(progress.pct).toFixed(0) + "% · 已用时 "
      + Math.round(Number(progress.elapsed_s) || 0) + " 秒",
      "busy"
    );
  });
  const ticker = window.setInterval(() => {
    if (Date.now() - lastReport < 1500) return;
    const seconds = Math.round((Date.now() - started) / 1000);
    setRelocateStatus("正在把扫描路径变换到新坐标系… 已用时 " + seconds + " 秒", "busy");
  }, 1000);
  try {
    const response = await fetchJson("/api/relocate", body);
    if (token !== state.relocate.token) return;
    state.result = response.result;
    state.resultSource = "live";
    state.showPath = true;
    $("showPath").checked = true;
    $("showPath").disabled = false;
    buildPathBuffers();
    syncSetupFromResult(response.result);
    renderResult(response.result);
    requestRender();
    toast("已把扫描路径变换到新的定位点坐标系");
  } catch (error) {
    if (token !== state.relocate.token) return;
    setRelocateStatus("重新定位失败：" + error.message, "bad");
    toast(error.message);
  } finally {
    window.clearInterval(ticker);
    stopPolling();
    state.relocate.busy = false;
    updateRelocateAvailability();
  }
}

function showResultState(name) {
  $("emptyState").classList.toggle("hidden", name !== "empty");
  $("progressState").classList.toggle("hidden", name !== "progress");
  $("resultState").classList.toggle("hidden", name !== "result");
  if (name === "progress") {
    $("progressState").querySelector(".spinner").style.display = "";
    planProgress.start();
  } else {
    planProgress.stop();
  }
}

function openViewer() {
  if (!state.result) {
    toast("请先完成一次路径规划");
    return;
  }
  window.open(state.result.viewer_url, "_blank", "noopener");
}

function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => node.classList.remove("show"), 2800);
}

function delay(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function scrollToId(id) {
  const node = $(id);
  if (node) node.scrollIntoView({ behavior: "auto", block: "start" });
  window.setTimeout(requestRender, 60);
}

/* ---------- event wiring ---------- */

function bindEvents() {
  for (const name of ["part", "support", "base"]) {
    $(name + "File").addEventListener("change", async (event) => {
      const file = event.target.files[0];
      try {
        await loadModel(name, file);
      } catch (error) {
        $("formFeedback").textContent = error.message;
        toast(error.message);
      }
    });
  }

  $("scanAxis").addEventListener("change", () => {
    updateAxisBadge("手动选择：" + AXIS_LABELS[currentAxis()]);
    requestRender();
  });
  $("fastMode").addEventListener("change", () => {
    const fast = $("fastMode").value === "fast";
    $("fastNoOcclusion").disabled = !fast;
    if (!fast) $("fastNoOcclusion").checked = false;
  });
  $("measureFrame").addEventListener("change", () => {
    state.registration = null;
    updateSolveAvailability();
    scheduleRegistrationCheck();
  });
  $("errorLimit").addEventListener("input", () => {
    state.registration = null;
    $("relocateLimit").value = $("errorLimit").value;
    updateSolveAvailability();
    scheduleRegistrationCheck();
  });
  $("recommendAxis").addEventListener("click", () => recommendAxis(true));
  $("pickAxis").addEventListener("click", () => {
    if (!state.models.part) {
      toast("请先导入零件 STL");
      return;
    }
    state.autoFace = !state.autoFace;
    $("pickAxis").classList.toggle("active", state.autoFace);
    renderDatumRows();
    toast(state.autoFace ? "请点击零件主扫描面，自动判断整车坐标轴" : "已取消在模型上点选扫描面");
  });
  document.querySelectorAll("[data-point-count]").forEach((button) => {
    button.addEventListener("click", () => setPointCount(Number(button.dataset.pointCount)));
  });
  $("clearPoints").addEventListener("click", resetPicks);
  $("undoPoint").addEventListener("click", undoPoint);
  $("fitView").addEventListener("click", fitView);
  $("lookAlongAxis").addEventListener("click", lookAlongAxis);
  $("showFixture").addEventListener("change", requestRender);
  $("showBase").addEventListener("change", requestRender);
  $("showPath").addEventListener("change", (event) => {
    state.showPath = event.target.checked;
    requestRender();
  });
  $("solveButton").addEventListener("click", solve);
  $("relocateButton").addEventListener("click", relocate);
  $("relocateLimit").addEventListener("input", () => {
    $("errorLimit").value = $("relocateLimit").value;
    state.registration = null;
    updateSolveAvailability();
    scheduleRegistrationCheck();
  });
  $("openViewer").addEventListener("click", openViewer);
  $("downloadReport").addEventListener("click", () => {
    if (!state.result) return;
    window.location.href = "/api/file?run_id=" + encodeURIComponent(state.result.run_id) + "&name=planning_report.html";
  });
  $("openViewerNav").addEventListener("click", () => {
    if (!state.result) {
      scrollToId("resultPanel");
      toast("完成一次规划后即可打开 viewer.html");
      return;
    }
    openViewer();
  });
  $("goSetup").addEventListener("click", () => scrollToId("projectPanel"));
  $("showPathOnModel").addEventListener("click", () => {
    if (!state.result) return;
    state.showPath = true;
    $("showPath").checked = true;
    scrollToId("modelStage");
    requestRender();
    toast("已在零件上显示扫描路径");
  });
  document.querySelectorAll(".nav-btn[data-view]").forEach((button) => {
    button.addEventListener("click", () => {
      const name = button.dataset.view;
      if (name === "workspace" && state.resultSource === "history") {
        resetWorkspace();
        toast("已新建项目");
      }
      showView(name);
    });
  });
  $("historyRefresh").addEventListener("click", () => loadHistory());
  $("historySearch").addEventListener("input", renderHistory);
  $("historyGoSetup").addEventListener("click", () => showView("workspace"));

  const canvas = $("modelCanvas");
  canvas.addEventListener("pointerdown", (event) => {
    canvas.setPointerCapture(event.pointerId);
    glState.drag = {
      x: event.clientX,
      y: event.clientY,
      moved: 0,
      button: event.button,
      mode: event.button === 0 && !event.shiftKey ? "rotate" : "pan"
    };
  });
  canvas.addEventListener("pointermove", (event) => {
    const drag = glState.drag;
    if (!drag) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    drag.moved += Math.abs(dx) + Math.abs(dy);
    if (drag.mode === "rotate") {
      glState.yaw -= dx * 0.008;
      glState.pitch = clamp(glState.pitch + dy * 0.008, -Math.PI / 2 + 0.01, Math.PI / 2 - 0.01);
    } else {
      const cam = camera();
      const scale = glState.distance * Math.tan(FOV / 2) * 2 / Math.max($("modelCanvas").clientHeight, 1);
      glState.center = add(glState.center, add(mul(cam.right, -dx * scale), mul(cam.up, dy * scale)));
    }
    drag.x = event.clientX;
    drag.y = event.clientY;
    requestRender();
  });
  canvas.addEventListener("pointerup", (event) => {
    const drag = glState.drag;
    glState.drag = null;
    if (!drag) return;
    if (drag.moved < 5 && drag.button === 0) handleCanvasClick(event.clientX, event.clientY);
  });
  canvas.addEventListener("pointercancel", () => { glState.drag = null; });
  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    if (glState.drag && glState.drag.moved > 5) return;
    undoPoint();
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    glState.distance = clamp(glState.distance * Math.exp(event.deltaY * 0.0012), glState.radius * 0.25, glState.radius * 20);
    requestRender();
  }, { passive: false });
  window.addEventListener("resize", requestRender);
}

/* ---------- views and history ---------- */

let historyRuns = [];

function showView(name) {
  document.querySelectorAll(".page-view").forEach((view) => {
    view.classList.toggle("active", view.id === name + "-view");
  });
  document.querySelectorAll(".nav-btn[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === name);
  });
  const active = $(name + "-view");
  $("page-title").textContent = active ? active.dataset.title : "工位与定位";
  window.scrollTo(0, 0);
  requestRender();
  if (name === "history") loadHistory();
}

async function loadHistory() {
  $("historyHint").textContent = "正在读取数据库…";
  try {
    const response = await fetch("/api/history");
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "历史记录读取失败");
    historyRuns = Array.isArray(data.runs) ? data.runs : [];
    renderHistory();
    $("historyHint").textContent = "记录保存在 output/web_runs/history.sqlite3，按时间倒序显示最近 200 条。";
  } catch (error) {
    $("historyHint").textContent = error.message;
    toast(error.message);
  }
}

function historyTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value || "—";
  return date.toLocaleString("zh-CN", {
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit"
  });
}

function renderHistory() {
  const query = $("historySearch").value.trim().toLowerCase();
  const runs = historyRuns.filter((run) => {
    if (!query) return true;
    return [run.display_name, run.prog_name, run.run_id]
      .join(" ").toLowerCase().includes(query);
  });
  $("historyCount").textContent = historyRuns.length;
  $("historyEmpty").classList.toggle("hidden", historyRuns.length > 0);
  const list = $("historyList");
  list.replaceChildren();
  list.classList.toggle("hidden", runs.length === 0);
  if (historyRuns.length > 0 && runs.length === 0) {
    $("historyHint").textContent = "没有匹配「" + query + "」的记录。";
  }
  for (const run of runs) list.append(historyItem(run));
}

function historyItem(run) {
  const item = document.createElement("article");
  item.className = "history-item";
  item.dataset.runId = run.run_id;

  const main = document.createElement("div");
  main.className = "history-main";
  const title = document.createElement("div");
  title.className = "history-title";
  const name = document.createElement("b");
  name.textContent = run.display_name;
  const pill = document.createElement("span");
  pill.className = "completion-pill";
  pill.textContent = run.status === "completed" ? "已完成" : "未完成";
  title.append(name, pill);

  const meta = document.createElement("div");
  meta.className = "history-meta";
  for (const text of [
    run.prog_name,
    historyTime(run.created_at),
    "覆盖率 " + Number(run.coverage_pct || 0).toFixed(2) + "%",
    Number(run.point_count || 0) + " 视点",
    Number(run.target_points || 0).toLocaleString() + " 采样点"
  ]) {
    const span = document.createElement("span");
    span.textContent = text;
    meta.append(span);
  }

  const identifier = document.createElement("div");
  identifier.className = "history-meta history-id";
  const idText = document.createElement("span");
  idText.textContent = run.run_id;
  identifier.append(idText);
  main.append(title, meta, identifier);

  const actions = document.createElement("div");
  actions.className = "history-actions";
  const open = document.createElement("button");
  open.className = "secondary-button";
  open.type = "button";
  open.textContent = "查看结果";
  open.addEventListener("click", () => openHistoryRun(run.run_id));
  const program = document.createElement("a");
  program.className = "secondary-button";
  program.textContent = "下载 LS";
  program.href = "/api/file?run_id=" + encodeURIComponent(run.run_id)
    + "&name=" + encodeURIComponent(run.prog_name + ".LS");
  program.download = run.prog_name + ".LS";
  const report = document.createElement("a");
  report.className = "secondary-button";
  report.textContent = "下载报告";
  report.href = "/api/file?run_id=" + encodeURIComponent(run.run_id)
    + "&name=" + encodeURIComponent("planning_report.html");
  report.download = "planning_report.html";
  const remove = document.createElement("button");
  remove.className = "secondary-button danger";
  remove.type = "button";
  remove.textContent = "删除";
  remove.addEventListener("click", () => deleteHistoryRun(run));
  actions.append(open, program, report, remove);

  item.append(main, actions);
  return item;
}

async function loadHistoryModels(runId) {
  for (const name of ["part", "support", "base"]) {
    // `base.stl` in a run folder is already installed in the measured frame; the
    // vehicle-frame copy is the one the planner expects when the run is repeated.
    const candidates = name === "base" ? ["base_vehicle", "base"] : [name];
    for (const candidate of candidates) {
      const response = await fetch(
        "/api/history/model?run_id=" + encodeURIComponent(runId)
        + "&name=" + encodeURIComponent(candidate)
      );
      if (!response.ok) continue;
      const buffer = await response.arrayBuffer();
      const file = new File([buffer], name + ".stl", { type: "model/stl" });
      await loadModel(name, file);
      break;
    }
  }
}

async function openHistoryRun(runId) {
  try {
    const response = await fetch(
      "/api/history/detail?run_id=" + encodeURIComponent(runId)
    );
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "该项目无法读取");
    state.result = data.result;
    state.resultSource = "history";
    state.showPath = true;
    $("showPath").checked = true;
    toast("正在载入历史项目的零件与夹具模型…");
    await loadHistoryModels(runId);
    syncSetupFromResult(data.result);
    renderResult(data.result);
    showResultState("result");
    buildPathBuffers();
    showView("workspace");
    scrollToId("resultPanel");
    requestRender();
    toast("已载入历史项目 " + data.result.program_name);
  } catch (error) {
    toast(error.message);
  }
}

function resetWorkspace() {
  state.models = {};
  state.picked = [];
  state.pointCount = 3;
  state.measured = Array.from({ length: 4 }, () => ["", "", ""]);
  state.autoFace = false;
  state.registration = null;
  state.result = null;
  state.resultSource = null;
  state.showPath = false;
  registrationRequest += 1;
  window.clearTimeout(registrationTimer);

  glState.meshes = {};
  glState.pathLine = null;
  glState.pathLineCount = 0;
  glState.pathPoint = null;
  glState.pathPointCount = 0;
  glState.center = [0, 0, 0];
  glState.radius = 1000;
  glState.distance = 2600;
  glState.home = null;
  glState.fitted = false;

  for (const name of ["part", "support", "base"]) {
    const input = $(name + "File");
    if (input) input.value = "";
    const card = document.querySelector('label[for="' + name + 'File"]');
    if (card) card.classList.remove("has-file");
  }
  $("partName").textContent = "选择零件模型";
  $("supportName").textContent = "选择夹具模型";
  $("baseName").textContent = "未导入时自动生成参考板";

  document.querySelectorAll("[data-point-count]").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.pointCount) === 3);
  });
  $("pickAxis").classList.remove("active");
  $("showFixture").checked = true;
  $("showBase").checked = true;
  $("showPath").checked = false;
  $("showPath").disabled = true;
  $("openViewer").disabled = true;
  $("downloadReport").disabled = true;
  $("downloadList").replaceChildren();
  $("resultDetails").replaceChildren();
  renderRelocatePanel(null);
  state.relocate.busy = false;
  state.relocate.token += 1;
  $("resultName").textContent = "项目";
  $("resultSubtitle").textContent = "规划完成";
  $("stageEmpty").style.display = "";
  $("formFeedback").textContent = "";
  $("scanAxis").value = "+Z";
  setRegistrationStatus(
    "选够定位点并填写现场坐标后，会自动计算整车 CAD 坐标到现场测量坐标系的安装变换。"
  );

  renderDatumRows();
  updateAxisBadge("尚未导入零件");
  updateSolveAvailability();
  showResultState("empty");
  requestRender();
}

async function deleteHistoryRun(run) {
  const confirmed = window.confirm(
    "删除「" + run.display_name + "」？\n"
    + "将同时移除 output/web_runs 下该项目的输出文件，且不可恢复。"
  );
  if (!confirmed) return;
  try {
    const response = await fetch("/api/history/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: run.run_id })
    });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "删除失败");
    historyRuns = historyRuns.filter((entry) => entry.run_id !== run.run_id);
    if (state.result && state.result.run_id === run.run_id) {
      state.result = null;
      $("openViewer").disabled = true;
      $("downloadReport").disabled = true;
      renderRelocatePanel(null);
      showResultState("empty");
    }
    renderHistory();
    toast(data.deleted ? "已删除该项目" : "记录已不存在");
  } catch (error) {
    toast(error.message);
  }
}

// Defaults come from the configuration the server will actually run, so the screen and
// the plan cannot disagree about which deliverable is being asked for. The page keeps
// working if the endpoint is unavailable: its own default stays in place.
async function applyServerDefaults() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) return;
    const config = await response.json();
    const mode = $("scanMode");
    if (mode && config.delivery_mode) mode.value = config.delivery_mode;
    // The labels carry the grid each choice will actually run, so they cannot drift from
    // the configuration the server holds.
    const accurate = document.querySelector('#fastMode option[value="accurate"]');
    if (accurate && config.sample_grid_mm) {
      accurate.textContent = "精确 · " + Number(config.sample_grid_mm).toFixed(0) + " mm 采样";
    }
    const fast = document.querySelector('#fastMode option[value="fast"]');
    if (fast && config.fast_grid_mm) {
      fast.textContent = "快速 · " + Number(config.fast_grid_mm).toFixed(0) + " mm 采样（约 2 分钟）";
    }
  } catch (_error) {
    /* keep the built-in default */
  }
}

function init() {
  initRenderer();
  bindEvents();
  renderDatumRows();
  updateAxisBadge("尚未导入零件");
  updateSolveAvailability();
  showResultState("empty");
  render();
}

window.__scanplan = { state, glState, camera, projectPoint, pickScene };
init();
applyServerDefaults();
