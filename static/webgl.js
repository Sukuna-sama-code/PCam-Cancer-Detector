import * as THREE from './vendor/three.module.min.js';

const TAU = Math.PI * 2;
const clamp = (v, min, max) => Math.min(max, Math.max(min, v));

const FLOW_GLSL = `
  vec3 cellularFlow(vec3 p, float time, float busy) {
    float wave = p.x * 1.38 - time * 0.72;
    float amp = 0.08 + busy * 0.06;
    p.y += sin(wave + p.z * 0.85) * amp;
    p.z += cos(wave * 0.82 + p.y * 1.1) * (0.095 + busy * 0.045);
    p.x += sin(p.y * 2.0 + time * 0.42) * 0.028;
    return p;
  }
`;

function membranePoint(u, v, branch) {
  const envelope = Math.pow(Math.max(0, Math.sin(Math.PI * u)), 0.54);
  const angle = u * TAU * 1.15 + branch * Math.PI - 0.45;
  const radius = 0.28 + envelope * (0.24 + v * 0.58);
  const scallop = 1.0 + 0.048 * Math.sin(u * TAU * 3.0 + v * 2.0);
  return [
    (u - 0.5) * 5.6,
    Math.cos(angle) * radius * scallop + 0.14 * Math.sin(u * TAU),
    Math.sin(angle) * radius + 0.16 * Math.cos(u * Math.PI),
  ];
}

function createMembraneGeometry() {
  const cols = 64;
  const rows = 10;
  const positions = [];
  const uvs = [];
  const indices = [];

  for (let branch = 0; branch < 2; branch++) {
    const start = positions.length / 3;
    for (let i = 0; i <= cols; i++) {
      for (let j = 0; j <= rows; j++) {
        positions.push(...membranePoint(i / cols, j / rows, branch));
        uvs.push(i / cols, j / rows);
      }
    }
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        const a = start + i * (rows + 1) + j;
        const b = a + rows + 1;
        indices.push(a, b, a + 1, b, b + 1, a + 1);
      }
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

function createCircleLoop(radius, segments = 84) {
  const pts = [];
  for (let i = 0; i < segments; i++) {
    const a = (i / segments) * TAU;
    pts.push(Math.cos(a) * radius, Math.sin(a) * radius, 0);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  return geo;
}

function renderHeroFallback(canvas) {
  const host = canvas?.parentElement;
  if (!host) return;
  const fallback = document.createElement('div');
  fallback.className = canvas.className;
  fallback.id = canvas.id;
  fallback.dataset.webgl = 'fallback';
  fallback.dataset.paused = 'true';
  fallback.style.cssText = 'position:absolute;inset:0;display:grid;place-items:center;pointer-events:none;opacity:0.55;';
  fallback.innerHTML = `<svg width="320" height="170" viewBox="0 0 320 170" fill="none" aria-hidden="true">
    <ellipse cx="220" cy="85" rx="88" ry="38" stroke="#FF8A00" stroke-width="0.9" stroke-opacity="0.45"/>
    <ellipse cx="220" cy="85" rx="64" ry="24" stroke="#FFB800" stroke-width="0.8" stroke-dasharray="4 4" stroke-opacity="0.4"/>
    <path d="M90 112 C155 24, 232 142, 300 46" stroke="#FFB800" stroke-width="1.3" stroke-opacity="0.75"/>
    <path d="M90 52 C155 142, 232 24, 300 116" stroke="#F4F4F6" stroke-width="1.0" stroke-opacity="0.45"/>
  </svg>`;
  canvas.replaceWith(fallback);
}

export function initAetherisMotion() {
  const canvas = document.getElementById('hero-webgl');
  const heroEl = document.getElementById('hero-pavilion');
  if (!canvas || !heroEl) {
    return { setInferencePulse() {}, dispose() {} };
  }

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance',
    });
  } catch (_) {
    renderHeroFallback(canvas);
    return { setInferencePulse() {}, dispose() {} };
  }

  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 2, 0.1, 60);
  camera.position.set(0, 0.1, 7.2);

  const sculptureGroup = new THREE.Group();
  const isoGroup = new THREE.Group();
  scene.add(isoGroup, sculptureGroup);

  const uniforms = {
    uTime: { value: 0 },
    uBusy: { value: 0 },
    uDpr: { value: 1 },
  };

  // 1. Sculptural Dual-Helix Bioluminescent Lattice (Obsidian + Molten Amber Fresnel)
  const membraneGeo = createMembraneGeometry();
  const membraneMat = new THREE.ShaderMaterial({
    uniforms,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    extensions: { derivatives: true },
    vertexShader: `
      uniform float uTime;
      uniform float uBusy;
      varying vec2 vUv;
      varying vec3 vNormal;
      varying vec3 vView;
      ${FLOW_GLSL}
      void main() {
        vUv = uv;
        vec3 p = cellularFlow(position, uTime, uBusy);
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        vNormal = normalize(normalMatrix * normal);
        vView = -mv.xyz;
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: `
      uniform float uTime;
      uniform float uBusy;
      varying vec2 vUv;
      varying vec3 vNormal;
      varying vec3 vView;
      void main() {
        vec2 grid = vUv * vec2(64.0, 10.0);
        vec2 deriv = max(fwidth(grid), vec2(0.0005));
        vec2 cell = abs(fract(grid - 0.5) - 0.5) / deriv;
        float wire = 1.0 - smoothstep(0.24, 1.08, min(cell.x, cell.y));
        float rim = pow(1.0 - abs(dot(normalize(vNormal), normalize(vView))), 1.65);
        float edge = 1.0 - smoothstep(0.0, 0.03, min(vUv.y, 1.0 - vUv.y));
        float ends = smoothstep(0.0, 0.06, vUv.x) * (1.0 - smoothstep(0.94, 1.0, vUv.x));
        float pulse = 0.5 + 0.5 * sin(vUv.x * 13.0 - uTime * (1.5 + uBusy * 2.4));
        float depth = clamp(1.16 - vView.z * 0.06, 0.30, 0.96);

        vec3 amber = vec3(1.0, 0.54, 0.04);
        vec3 gold = vec3(1.0, 0.78, 0.30);
        vec3 emerald = vec3(0.22, 0.85, 0.60);
        vec3 baseCol = mix(amber, gold, clamp(rim * 0.62 + edge * 0.35 + pulse * 0.16, 0.0, 1.0));
        vec3 color = mix(baseCol, emerald, uBusy * pulse * 0.35);

        float alpha = (0.025 + rim * 0.065 + wire * (0.34 + rim * 0.30) + edge * 0.22);
        alpha *= ends * depth * (1.0 + uBusy * (0.32 + pulse * 0.45));
        gl_FragColor = vec4(color, clamp(alpha, 0.0, 0.94));
        #include <colorspace_fragment>
      }
    `,
  });
  const membraneMesh = new THREE.Mesh(membraneGeo, membraneMat);
  membraneMesh.frustumCulled = false;
  sculptureGroup.add(membraneMesh);

  // 2. Stacked-Pro Isometric Architectural Wireframe Cubes & Orbital Calipers
  const boxEdgeGeoA = new THREE.EdgesGeometry(new THREE.BoxGeometry(1.55, 1.55, 1.55));
  const boxEdgeGeoB = new THREE.EdgesGeometry(new THREE.BoxGeometry(1.08, 1.08, 1.08));
  const boxMatA = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.16 });
  const boxMatB = new THREE.LineBasicMaterial({ color: 0xff9a1f, transparent: true, opacity: 0.34 });
  const cubeA = new THREE.LineSegments(boxEdgeGeoA, boxMatA);
  const cubeB = new THREE.LineSegments(boxEdgeGeoB, boxMatB);
  cubeA.rotation.set(0.62, 0.78, 0.14);
  cubeB.rotation.set(-0.55, 0.45, 0.65);

  const ringGeo = createCircleLoop(1.92, 92);
  const ringMat = new THREE.LineBasicMaterial({ color: 0xffb800, transparent: true, opacity: 0.22 });
  const caliperRing = new THREE.LineLoop(ringGeo, ringMat);
  caliperRing.rotation.set(0.88, -0.32, 0.20);
  isoGroup.add(cubeA, cubeB, caliperRing);

  // 3. 160 Bioluminescent Nuclei Particles
  let seed = 7321;
  const rnd = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const pCount = 160;
  const pPos = new Float32Array(pCount * 3);
  const pSize = new Float32Array(pCount);
  const pPhase = new Float32Array(pCount);
  const pTint = new Float32Array(pCount);

  for (let i = 0; i < pCount; i++) {
    if (i < 118) {
      const pt = membranePoint(0.05 + rnd() * 0.90, 0.12 + rnd() * 0.78, i % 2);
      pPos[i * 3] = pt[0];
      pPos[i * 3 + 1] = pt[1];
      pPos[i * 3 + 2] = pt[2];
      pSize[i] = 2.8 + rnd() * 3.2;
    } else {
      pPos[i * 3] = (rnd() - 0.5) * 6.8;
      pPos[i * 3 + 1] = (rnd() - 0.5) * 3.0;
      pPos[i * 3 + 2] = (rnd() - 0.5) * 2.8;
      pSize[i] = 1.6 + rnd() * 2.2;
    }
    pPhase[i] = rnd() * TAU;
    pTint[i] = rnd();
  }

  const particleGeo = new THREE.BufferGeometry();
  particleGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
  particleGeo.setAttribute('aSize', new THREE.BufferAttribute(pSize, 1));
  particleGeo.setAttribute('aPhase', new THREE.BufferAttribute(pPhase, 1));
  particleGeo.setAttribute('aTint', new THREE.BufferAttribute(pTint, 1));

  const particleMat = new THREE.ShaderMaterial({
    uniforms,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: `
      uniform float uTime;
      uniform float uBusy;
      uniform float uDpr;
      attribute float aSize;
      attribute float aPhase;
      attribute float aTint;
      varying float vTwinkle;
      varying float vTint;
      ${FLOW_GLSL}
      void main() {
        vec3 p = cellularFlow(position, uTime, uBusy);
        p.y += sin(uTime * 0.65 + aPhase) * 0.04;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        vTwinkle = 0.58 + 0.42 * sin(uTime * (1.3 + uBusy * 2.2) + aPhase);
        vTint = aTint;
        gl_PointSize = aSize * uDpr * (1.0 + uBusy * 0.35) * (5.8 / max(1.0, -mv.z));
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: `
      varying float vTwinkle;
      varying float vTint;
      void main() {
        float d = length(gl_PointCoord - vec2(0.5));
        if (d > 0.5) discard;
        float core = smoothstep(0.28, 0.0, d);
        float halo = pow(1.0 - d * 2.0, 2.2);
        vec3 amber = vec3(1.0, 0.68, 0.16);
        vec3 ivory = vec3(1.0, 0.94, 0.78);
        vec3 emerald = vec3(0.25, 0.88, 0.62);
        vec3 base = vTint > 0.78 ? emerald : amber;
        vec3 col = mix(base, ivory, core);
        float alpha = (halo * 0.45 + core * 0.55) * vTwinkle * 0.85;
        gl_FragColor = vec4(col, alpha);
        #include <colorspace_fragment>
      }
    `,
  });
  const particles = new THREE.Points(particleGeo, particleMat);
  sculptureGroup.add(particles);

  const reducedQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  const state = {
    px: 0, py: 0, tpx: 0, tpy: 0,
    busy: 0, tbusy: 0,
    elapsed: 0,
    inView: true,
    reduced: reducedQuery.matches,
    disposed: false,
    raf: 0,
    lastTs: 0,
  };

  function resize() {
    const rect = heroEl.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    uniforms.uDpr.value = dpr;
    camera.aspect = w / h;
    const isCompact = heroEl.classList.contains('is-compact');
    const offsetX = w < 760 ? 0.35 : isCompact ? 2.45 : 1.95;
    sculptureGroup.position.set(offsetX, 0.0, 0);
    isoGroup.position.set(offsetX + 0.35, 0.0, -0.5);
    camera.position.z = clamp(6.6 + (2.4 - Math.min(camera.aspect, 2.4)) * 1.15, 6.2, 8.8);
    camera.updateProjectionMatrix();
    renderFrame();
  }

  function renderFrame() {
    uniforms.uTime.value = state.elapsed;
    uniforms.uBusy.value = state.busy;

    sculptureGroup.rotation.set(
      0.28 + Math.sin(state.elapsed * 0.16) * 0.08 + state.py * 0.14,
      -0.24 + Math.sin(state.elapsed * 0.13) * 0.15 + state.px * 0.18,
      0.20 + Math.sin(state.elapsed * 0.09) * 0.04,
    );
    cubeA.rotation.x = 0.62 + state.elapsed * 0.08;
    cubeA.rotation.y = 0.78 + state.elapsed * 0.11 + state.px * 0.12;
    cubeB.rotation.x = -0.55 - state.elapsed * 0.10;
    cubeB.rotation.y = 0.45 + state.elapsed * 0.14;
    caliperRing.rotation.z = 0.20 + state.elapsed * (0.05 + state.busy * 0.16);
    boxMatB.opacity = 0.32 + state.busy * 0.28;
    ringMat.opacity = 0.20 + state.busy * 0.24;

    renderer.render(scene, camera);
    canvas.dataset.webgl = 'ready';
  }

  function syncPausedAttr() {
    const paused = state.disposed || document.hidden || !state.inView || state.reduced;
    canvas.dataset.paused = paused ? 'true' : 'false';
    return paused;
  }

  function schedule() {
    if (syncPausedAttr() || state.raf) return;
    state.lastTs = performance.now();
    state.raf = requestAnimationFrame(tick);
  }

  function stopLoop() {
    if (state.raf) {
      cancelAnimationFrame(state.raf);
      state.raf = 0;
    }
    syncPausedAttr();
  }

  function tick(ts) {
    state.raf = 0;
    if (syncPausedAttr()) return;

    const dt = Math.min((ts - state.lastTs) / 1000 || 0.016, 0.05);
    state.lastTs = ts;

    const ease = 1 - Math.exp(-dt * 8.0);
    state.px += (state.tpx - state.px) * ease;
    state.py += (state.tpy - state.py) * ease;
    state.busy += (state.tbusy - state.busy) * (1 - Math.exp(-dt * 4.5));
    state.elapsed += dt * (0.82 + state.busy * 2.1);

    renderFrame();
    schedule();
  }

  function onHeroPointerMove(e) {
    if (state.reduced || e.pointerType === 'touch') return;
    const r = heroEl.getBoundingClientRect();
    state.tpx = ((e.clientX - r.left) / Math.max(1, r.width) - 0.5) * 2;
    state.tpy = ((e.clientY - r.top) / Math.max(1, r.height) - 0.5) * 2;
  }
  function onHeroPointerLeave() {
    state.tpx = 0;
    state.tpy = 0;
  }

  heroEl.addEventListener('pointermove', onHeroPointerMove, { passive: true });
  heroEl.addEventListener('pointerleave', onHeroPointerLeave, { passive: true });

  const ro = new ResizeObserver(resize);
  ro.observe(heroEl);

  const io = new IntersectionObserver(
    (entries) => {
      state.inView = Boolean(entries[0]?.isIntersecting);
      if (state.inView) schedule();
      else stopLoop();
    },
    { threshold: 0.02 },
  );
  io.observe(heroEl);

  const onVisibility = () => {
    if (document.hidden) stopLoop();
    else schedule();
  };
  document.addEventListener('visibilitychange', onVisibility);

  reducedQuery.addEventListener?.('change', () => {
    state.reduced = reducedQuery.matches;
    if (state.reduced) {
      stopLoop();
      renderFrame();
    } else {
      schedule();
    }
  });

  resize();
  schedule();

  return {
    setInferencePulse(active) {
      state.tbusy = active ? 1 : 0;
      if (state.reduced) {
        state.busy = state.tbusy;
        renderFrame();
      } else {
        schedule();
      }
    },
    dispose() {
      state.disposed = true;
      stopLoop();
      ro.disconnect();
      io.disconnect();
      heroEl.removeEventListener('pointermove', onHeroPointerMove);
      heroEl.removeEventListener('pointerleave', onHeroPointerLeave);
      document.removeEventListener('visibilitychange', onVisibility);
      membraneGeo.dispose();
      membraneMat.dispose();
      boxEdgeGeoA.dispose();
      boxEdgeGeoB.dispose();
      boxMatA.dispose();
      boxMatB.dispose();
      ringGeo.dispose();
      ringMat.dispose();
      particleGeo.dispose();
      particleMat.dispose();
      renderer.dispose();
    },
  };
}
