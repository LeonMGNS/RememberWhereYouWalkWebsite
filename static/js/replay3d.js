// Interactive 3D replay of the real-world run: the recorded Go2 motion in the maze, the LSMM matrix
// memory as a cell overlay (off / on the floor / floating above the maze) and the walls the policy
// predicts from its memory.
//
// Data (static/replay/, written by scripts/web/ in the RememberWhereYouWalk repo):
//   scene.glb          maze, ground texture and the Go2 with its recorded motion baked in (25 fps)
//   replay.json        trajectory, wall slots and the predicted wall probability per frame
//   memory_atlas.png   the 50x50 matrix memory of every frame (alpha 0 = cell not written yet)
//
// Coordinates: the data uses the figure frame of the paper (x right, y up, z height);
// three.js is y-up, so a figure point (x, y, z) is (x, z, -y) here.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

const BASE = 'static/replay/';
const DRACO_PATH = 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/libs/draco/gltf/';
const LSMM_BLUE = 0x0173b2;
const SKY = 0xe9eef2;
const WALL_HEIGHT = 0.5;
const MEMORY_FLOOR_Y = 0.015;
const MEMORY_FLOAT_Y = 1.05;
// memory overlay look, as in the poster figure
const WRITTEN_ALPHA = 0.6;
const UNWRITTEN_RGBA = [230, 230, 230, 0.3];

const fig = (x, y, z = 0) => new THREE.Vector3(x, z, -y);

function copper(s) { // matplotlib "copper", the trajectory colour map of the paper figures
  return new THREE.Color().setRGB(Math.min(1, 1.25 * s), 0.7812 * s, 0.4975 * s, THREE.SRGBColorSpace);
}

function webglAvailable() {
  try {
    const c = document.createElement('canvas');
    return !!(window.WebGL2RenderingContext && c.getContext('webgl2'));
  } catch (e) {
    return false;
  }
}

async function loadAtlas(url) {
  const img = new Image();
  img.src = url;
  await img.decode();
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  return ctx.getImageData(0, 0, img.width, img.height);
}

class Replay {
  constructor(root) {
    this.root = root;
    this.stage = root.querySelector('.replay3d-stage');
    this.status = root.querySelector('.replay3d-status');
    this.ui = {
      play: root.querySelector('[data-action="play"]'),
      slider: root.querySelector('[data-control="time"]'),
      time: root.querySelector('[data-label="time"]'),
      speed: root.querySelector('[data-control="speed"]'),
    };
    this.playing = true;
    this.speed = 1;
    this.frame = 0; // playhead in frames (float)
    this.visible = false;
    this.cameraMode = 'free';
    this.memoryMode = 'floor';
  }

  setStatus(text) {
    this.status.textContent = text;
    this.status.hidden = !text;
  }

  async init() {
    this.setStatus('Loading 3D replay…');
    const draco = new DRACOLoader().setDecoderPath(DRACO_PATH);
    const gltfLoader = new GLTFLoader().setDRACOLoader(draco);
    const [gltf, data] = await Promise.all([
      gltfLoader.loadAsync(BASE + 'scene.glb', (e) => {
        if (e.total) this.setStatus(`Loading 3D replay… ${Math.round((100 * e.loaded) / e.total)} %`);
      }),
      fetch(BASE + 'replay.json').then((r) => r.json()),
    ]);
    this.data = data;
    this.atlas = await loadAtlas(BASE + data.atlas.file);
    this.nFrames = data.frames;
    this.fps = data.fps;

    this.buildRenderer();
    this.buildScene(gltf);
    this.buildMemory();
    this.buildPredictedWalls();
    this.buildTrajectory();
    this.bindUI();
    this.setCamera('free');
    this.setMemoryMode(this.memoryMode);
    this.setStatus('');
    this.showFrame(0);

    this.clock = new THREE.Clock();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  buildRenderer() {
    const r = new THREE.WebGLRenderer({ antialias: true });
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.NoToneMapping;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    this.stage.prepend(r.domElement);
    this.renderer = r;

    this.camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.05, 200);
    this.controls = new OrbitControls(this.camera, r.domElement);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.minDistance = 0.4;
    this.controls.maxDistance = 25;
    // dragging in top-down view switches to free orbit
    this.controls.addEventListener('start', () => {
      if (this.cameraMode === 'top') this.setCamera('free', false);
    });

    const resize = () => {
      const w = this.stage.clientWidth;
      const h = this.stage.clientHeight;
      r.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      if (this.trajectory) this.trajectory.material.resolution.set(w, h);
    };
    new ResizeObserver(resize).observe(this.stage);
    resize();
  }

  buildScene(gltf) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(SKY);
    scene.fog = new THREE.Fog(SKY, 10, 35);
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8a7a66, 0.95));
    const sun = new THREE.DirectionalLight(0xfff4e5, 1.1);
    sun.position.copy(fig(-3, -4, 8));
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: 1, far: 20 });
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.02;
    scene.add(sun);

    const model = gltf.scene;
    this.innerWalls = [];
    let groundTexture = null;
    model.traverse((o) => {
      if (!o.isMesh) return;
      const name = o.material.name;
      if (name === 'Web_Ground') {
        // the texture is a render with lighting already in it: show it as is, but let it take shadows
        groundTexture = o.material.map;
        o.material = new THREE.MeshLambertMaterial({ map: groundTexture });
        o.receiveShadow = true;
      } else if (name === 'Wall_WhiteTarp') {
        o.material = new THREE.MeshStandardMaterial({ color: 0xc9cfd4, roughness: 0.95, side: THREE.DoubleSide });
        o.castShadow = o.receiveShadow = true;
      } else if (name === 'Stake_Wood') {
        o.material = new THREE.MeshStandardMaterial({ color: 0x7b5a3a, roughness: 0.85 });
        o.castShadow = true;
      } else {
        o.castShadow = o.receiveShadow = true;
      }
    });
    // inner walls and their stakes can be hidden; the outer boundary always stays
    model.traverse((o) => {
      if (!o.isMesh || !/^(Wall_|Stake)/.test(o.name)) return;
      const box = new THREE.Box3().setFromObject(o);
      const c = box.getCenter(new THREE.Vector3());
      const s = box.getSize(new THREE.Vector3());
      const onBoundary = (s.x > 4.5 || s.z > 4.5) || Math.max(Math.abs(c.x), Math.abs(c.z)) > 2.4;
      if (!onBoundary) this.innerWalls.push(o);
    });
    scene.add(model);
    this.robot = model.getObjectByName('go2_base');

    // ground outside the maze: a pale tint of the soil colour that fades into the sky
    const outside = new THREE.Mesh(
      new THREE.CircleGeometry(60, 64).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: this.averageColor(groundTexture).lerp(new THREE.Color(SKY), 0.6) }),
    );
    outside.position.y = -0.003;
    scene.add(outside);

    this.mixer = new THREE.AnimationMixer(model);
    this.clipEnd = Math.min(...gltf.animations.map((c) => c.duration)) - 1e-4;
    for (const clip of gltf.animations) this.mixer.clipAction(clip).play();
    this.scene = scene;
  }

  averageColor(texture) {
    const fallback = new THREE.Color(0x8f8068);
    const img = texture && texture.image;
    if (!img) return fallback;
    try {
      const c = document.createElement('canvas');
      c.width = c.height = 1;
      const ctx = c.getContext('2d');
      ctx.drawImage(img, 0, 0, 1, 1);
      const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
      return new THREE.Color().setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
    } catch (e) {
      return fallback;
    }
  }

  buildMemory() {
    const n = this.data.atlas.tile;
    this.memoryPixels = new Uint8Array(n * n * 4);
    const tex = new THREE.DataTexture(this.memoryPixels, n, n, THREE.RGBAFormat);
    tex.magFilter = tex.minFilter = THREE.NearestFilter;
    tex.colorSpace = THREE.SRGBColorSpace;
    this.memoryTexture = tex;

    // the memory covers the 5 m x 5 m maze; plane u -> +x, v -> +y (figure frame)
    const group = new THREE.Group();
    const cells = new THREE.Mesh(
      new THREE.PlaneGeometry(5, 5).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, toneMapped: false,
        side: THREE.DoubleSide }),
    );
    cells.renderOrder = 2;
    group.add(cells);

    // faint white cell grid on top
    const px = 1024;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = px;
    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = 'rgba(255,255,255,0.45)';
    ctx.lineWidth = 1.5;
    for (let i = 0; i <= n; i++) {
      const p = (i * px) / n;
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, px); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(px, p); ctx.stroke();
    }
    const gridTex = new THREE.CanvasTexture(canvas);
    gridTex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    const grid = new THREE.Mesh(
      new THREE.PlaneGeometry(5, 5).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ map: gridTex, transparent: true, depthWrite: false, toneMapped: false,
        side: THREE.DoubleSide }),
    );
    grid.position.y = 0.001;
    grid.renderOrder = 3;
    group.add(grid);

    // thin frame, mainly to read the floating layer as one sheet
    const edge = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(5, 5).rotateX(-Math.PI / 2)),
      new THREE.LineBasicMaterial({ color: LSMM_BLUE, transparent: true, opacity: 0.7 }),
    );
    group.add(edge);
    this.memoryEdge = edge;

    this.memory = group;
    this.scene.add(group);
  }

  updateMemory(frame) {
    const { tile: n, columns } = this.data.atlas;
    const src = this.atlas.data;
    const stride = this.atlas.width * 4;
    const r0 = Math.floor(frame / columns) * n;
    const c0 = (frame % columns) * n;
    const [ur, ug, ub, ua] = UNWRITTEN_RGBA;
    const out = this.memoryPixels;
    for (let i = 0; i < n; i++) { // atlas row i (top = +y)  ->  texture row n - 1 - i (row 0 = -y)
      let s = (r0 + i) * stride + c0 * 4;
      let d = (n - 1 - i) * n * 4;
      for (let j = 0; j < n; j++, s += 4, d += 4) {
        if (src[s + 3] > 127) {
          out[d] = src[s]; out[d + 1] = src[s + 1]; out[d + 2] = src[s + 2];
          out[d + 3] = 255 * WRITTEN_ALPHA;
        } else {
          out[d] = ur; out[d + 1] = ug; out[d + 2] = ub; out[d + 3] = 255 * ua;
        }
      }
    }
    this.memoryTexture.needsUpdate = true;
  }

  buildPredictedWalls() {
    // slightly larger than the real walls, so a correct prediction shows as a blue glow around the wall
    // and a wrong one as a free-standing blue wall
    this.predicted = new THREE.Group();
    this.predictedMeshes = this.data.wall_slots.map((s) => {
      const geo = s.horizontal
        ? new THREE.BoxGeometry(1.0, WALL_HEIGHT + 0.06, 0.1)
        : new THREE.BoxGeometry(0.1, WALL_HEIGHT + 0.06, 1.0);
      const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
        color: LSMM_BLUE, transparent: true, opacity: 0, depthWrite: false,
      }));
      mesh.position.copy(fig(s.x, s.y, (WALL_HEIGHT + 0.06) / 2));
      mesh.renderOrder = 1;
      this.predicted.add(mesh);
      return mesh;
    });
    this.scene.add(this.predicted);
  }

  updatePredictedWalls(frame) {
    const probs = this.data.wall_prob[frame];
    this.predictedMeshes.forEach((m, k) => {
      const p = probs[k] < 0 ? 0 : probs[k] / 255;
      m.visible = p > 0.05;
      m.material.opacity = 0.35 * p;
    });
  }

  buildTrajectory() {
    const pts = this.data.trajectory;
    const positions = [];
    const colors = [];
    pts.forEach(([x, y], k) => {
      positions.push(x, 0.03, -y);
      const c = copper(k / (pts.length - 1));
      colors.push(c.r, c.g, c.b);
    });
    const geo = new LineGeometry();
    geo.setPositions(positions);
    geo.setColors(colors);
    const mat = new LineMaterial({ vertexColors: true, linewidth: 3, worldUnits: false });
    mat.resolution.set(this.stage.clientWidth, this.stage.clientHeight);
    this.trajectory = new Line2(geo, mat);
    this.trajectory.renderOrder = 4;
    this.scene.add(this.trajectory);
  }

  showFrame(frame) {
    const f = Math.min(this.nFrames - 1, Math.max(0, Math.round(frame)));
    // clip time (k + 1) / fps is frame k; stay just below the clip end so it does not wrap
    this.mixer.setTime(Math.min((frame + 1) / this.fps, this.clipEnd));
    if (f !== this.shownFrame) {
      this.shownFrame = f;
      this.updateMemory(f);
      this.updatePredictedWalls(f);
      this.trajectory.geometry.instanceCount = Math.max(f, 1);
      this.ui.slider.value = f;
      const t = f / this.fps;
      this.ui.time.textContent = `${t.toFixed(1)} s / ${((this.nFrames - 1) / this.fps).toFixed(1)} s`;
    }
  }

  // ---- camera ------------------------------------------------------------------------------------
  robotPosition() {
    return this.robot.getWorldPosition(new THREE.Vector3());
  }

  setCamera(mode, move = true) {
    this.cameraMode = mode;
    this.root.querySelectorAll('[data-camera]').forEach((b) => b.classList.toggle('is-active', b.dataset.camera === mode));
    if (!move) return;
    const c = this.controls;
    if (mode === 'top') {
      // straight down, figure orientation (+y up on screen)
      c.target.set(0, 0, 0);
      this.camera.position.set(0, 9.5, 0.001);
    } else if (mode === 'follow') {
      const p = this.robotPosition();
      const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(this.robot.getWorldQuaternion(new THREE.Quaternion()));
      forward.y = 0;
      forward.normalize();
      c.target.copy(p);
      this.camera.position.copy(p).addScaledVector(forward, -1.9).add(new THREE.Vector3(0, 1.2, 0));
      this.followLast = p.clone();
    } else {
      c.target.set(0.3, 0, -0.2);
      this.camera.position.copy(fig(-4.4, -4.6, 4.2));
    }
    c.update();
  }

  setMemoryMode(mode) {
    this.memoryMode = mode;
    this.root.querySelectorAll('[data-memory]').forEach((b) => b.classList.toggle('is-active', b.dataset.memory === mode));
    this.memory.visible = mode !== 'off';
    this.memory.position.y = mode === 'float' ? MEMORY_FLOAT_Y : MEMORY_FLOOR_Y;
    this.memoryEdge.visible = mode === 'float';
  }

  // ---- UI ----------------------------------------------------------------------------------------
  bindUI() {
    const { play, slider, speed } = this.ui;
    slider.max = this.nFrames - 1;
    play.addEventListener('click', () => this.setPlaying(!this.playing));
    slider.addEventListener('input', () => {
      this.frame = Number(slider.value);
      this.showFrame(this.frame);
    });
    slider.addEventListener('pointerdown', () => { this.scrubbing = true; });
    window.addEventListener('pointerup', () => { this.scrubbing = false; });
    speed.addEventListener('change', () => { this.speed = Number(speed.value); });
    this.root.querySelectorAll('[data-camera]').forEach((b) => b.addEventListener('click', () => this.setCamera(b.dataset.camera)));
    this.root.querySelectorAll('[data-memory]').forEach((b) => b.addEventListener('click', () => this.setMemoryMode(b.dataset.memory)));
    const toggle = (name, fn) => {
      const box = this.root.querySelector(`[data-toggle="${name}"]`);
      box.addEventListener('change', () => fn(box.checked));
      fn(box.checked);
    };
    toggle('prediction', (on) => { this.predicted.visible = on; });
    toggle('walls', (on) => this.innerWalls.forEach((o) => { o.visible = on; }));
    toggle('path', (on) => { this.trajectory.visible = on; });
    this.setPlaying(this.playing);
  }

  setPlaying(on) {
    this.playing = on;
    this.ui.play.textContent = on ? '❚❚' : '▶';
    this.ui.play.setAttribute('aria-label', on ? 'Pause' : 'Play');
  }

  tick() {
    const dt = Math.min(this.clock.getDelta(), 0.1);
    if (!this.visible) return; // off screen: no rendering
    if (this.playing && !this.scrubbing) {
      this.frame += dt * this.fps * this.speed;
      if (this.frame > this.nFrames - 1) this.frame = 0;
      this.showFrame(this.frame);
    }
    if (this.cameraMode === 'follow') {
      // keep the user's orbit offset, move with the robot
      const p = this.robotPosition();
      const delta = p.clone().sub(this.followLast);
      this.camera.position.add(delta);
      this.controls.target.add(delta);
      this.followLast.copy(p);
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

function start(root) {
  if (!webglAvailable()) {
    root.querySelector('.replay3d-status').textContent = 'The 3D replay needs WebGL 2, which this browser does not provide.';
    return;
  }
  const replay = new Replay(root);
  let started = false;
  new IntersectionObserver((entries) => {
    const on = entries.some((e) => e.isIntersecting);
    replay.visible = on;
    if (on && !started) {
      started = true;
      replay.init().catch((err) => {
        console.error(err);
        replay.setStatus('The 3D replay could not be loaded.');
      });
    }
  }, { rootMargin: '300px' }).observe(root);
}

document.querySelectorAll('.replay3d').forEach(start);
