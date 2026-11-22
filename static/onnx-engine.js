const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];
const PATCH_DIM = 96;
const DISPLAY_DIM = 576;

let manifestCache = null;
let sessionCache = null;
let bioCache = null;
let lastSlideCache = null; // { rgb96, cam12, ttaStd, baseResult }

// Precomputed 256-entry OpenCV-compatible COLORMAP_JET LUT (RGB)
const JET_LUT = (() => {
  const lut = new Uint8Array(256 * 3);
  const clamp01 = (v) => Math.max(0, Math.min(1, v));
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    lut[i * 3] = Math.round(clamp01(1.5 - Math.abs(4 * t - 3)) * 255);
    lut[i * 3 + 1] = Math.round(clamp01(1.5 - Math.abs(4 * t - 2)) * 255);
    lut[i * 3 + 2] = Math.round(clamp01(1.5 - Math.abs(4 * t - 1)) * 255);
  }
  return lut;
})();

function roundTo(val, decimals) {
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function nextPaintFrame() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export async function initLocalEngine(onProgress) {
  if (!manifestCache) {
    const res = await fetch('./models/manifest.json');
    if (!res.ok) throw new Error(`Failed to fetch manifest.json (${res.status})`);
    manifestCache = await res.json();
  }

  if (sessionCache) {
    return buildStatusPayload();
  }

  if (!window.ort) {
    throw new Error('ONNX Runtime Web failed to load from CDN.');
  }

  window.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';
  window.ort.env.wasm.numThreads = 1;
  window.ort.env.wasm.simd = true;

  onProgress?.(0, 0, manifestCache.model_size_bytes || 0);
  const modelRes = await fetch(manifestCache.model_url);
  if (!modelRes.ok) {
    throw new Error(`Failed to fetch ONNX weights (${modelRes.status})`);
  }

  const totalBytes = Number(modelRes.headers.get('Content-Length')) || manifestCache.model_size_bytes || 0;
  let modelBytes;

  if (modelRes.body && typeof modelRes.body.getReader === 'function') {
    const reader = modelRes.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      const pct = totalBytes > 0 ? Math.min(99, Math.round((loaded / totalBytes) * 100)) : 50;
      onProgress?.(pct, loaded, totalBytes);
    }
    modelBytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      modelBytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
  } else {
    const buf = await modelRes.arrayBuffer();
    modelBytes = new Uint8Array(buf);
  }

  onProgress?.(100, modelBytes.byteLength, modelBytes.byteLength);
  sessionCache = await window.ort.InferenceSession.create(modelBytes, {
    executionProviders: ['wasm'],
  });

  return buildStatusPayload();
}

function buildStatusPayload() {
  return {
    model_loaded: Boolean(sessionCache),
    device: 'WebAssembly',
    default_threshold: manifestCache.default_threshold,
    val_auc: manifestCache.val_auc,
    test_auc: manifestCache.test_auc,
    checkpoint_metrics: manifestCache.checkpoint_metrics,
    samples: manifestCache.samples,
  };
}

async function loadImageSourceToCanvas(source) {
  const isFile = source instanceof Blob;
  const filename = isFile ? (source.name || 'uploaded.png') : String(source);
  const lower = filename.toLowerCase();
  const isTiff = lower.endsWith('.tif') || lower.endsWith('.tiff');

  if (isTiff) {
    if (!window.UTIF) throw new Error('UTIF TIFF decoder is unavailable.');
    const buf = isFile ? await source.arrayBuffer() : await (await fetch(source)).arrayBuffer();
    const ifds = window.UTIF.decode(buf);
    if (!ifds || ifds.length === 0) throw new Error('Invalid or empty TIFF file.');
    window.UTIF.decodeImage(buf, ifds[0]);
    const rgba = window.UTIF.toRGBA8(ifds[0]);
    const w = ifds[0].width;
    const h = ifds[0].height;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);
    return canvas;
  }

  const blob = isFile ? source : await (await fetch(source)).blob();
  if (!isFile && !blob.type && !blob.size) {
    throw new Error(`Could not load image from ${filename}`);
  }

  let bitmap = null;
  if (typeof createImageBitmap === 'function') {
    try {
      bitmap = await createImageBitmap(blob);
    } catch (_) {
      bitmap = null;
    }
  }

  if (bitmap) {
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close?.();
    return canvas;
  }

  const objUrl = URL.createObjectURL(blob);
  try {
    const img = await new Promise((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error(`Unable to decode image '${filename}'.`));
      el.src = objUrl;
    });
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    return canvas;
  } finally {
    URL.revokeObjectURL(objUrl);
  }
}

async function decodeToRgb96(source) {
  const nativeCanvas = await loadImageSourceToCanvas(source);
  const w = nativeCanvas.width;
  const h = nativeCanvas.height;
  const nativeCtx = nativeCanvas.getContext('2d', { willReadFrequently: true });
  const nativeData = nativeCtx.getImageData(0, 0, w, h).data;

  let sumRgb = 0;
  const pxCount = w * h;
  for (let i = 0; i < nativeData.length; i += 4) {
    sumRgb += nativeData[i] + nativeData[i + 1] + nativeData[i + 2];
  }
  const meanIntensity = sumRgb / Math.max(1, pxCount * 3);

  const c96 = document.createElement('canvas');
  c96.width = PATCH_DIM;
  c96.height = PATCH_DIM;
  const ctx96 = c96.getContext('2d', { willReadFrequently: true });
  ctx96.imageSmoothingEnabled = true;
  ctx96.imageSmoothingQuality = 'high';
  ctx96.drawImage(nativeCanvas, 0, 0, PATCH_DIM, PATCH_DIM);
  const rgb96 = ctx96.getImageData(0, 0, PATCH_DIM, PATCH_DIM).data;

  const c56 = document.createElement('canvas');
  c56.width = 56;
  c56.height = 56;
  const ctx56 = c56.getContext('2d');
  ctx56.imageSmoothingEnabled = true;
  ctx56.imageSmoothingQuality = 'high';
  ctx56.drawImage(nativeCanvas, 0, 0, 56, 56);
  const thumb56Url = c56.toDataURL('image/png');

  return { rgb96, nativeWidth: w, nativeHeight: h, meanIntensity, thumb56Url };
}

function buildRotatedNormalizedCHW(rgb96, angleDeg) {
  const planeSize = PATCH_DIM * PATCH_DIM;
  const chw = new Float32Array(3 * planeSize);
  for (let dr = 0; dr < PATCH_DIM; dr++) {
    for (let dc = 0; dc < PATCH_DIM; dc++) {
      let sr, sc;
      if (angleDeg === 0) {
        sr = dr;
        sc = dc;
      } else if (angleDeg === 90) {
        sr = dc;
        sc = PATCH_DIM - 1 - dr;
      } else if (angleDeg === 180) {
        sr = PATCH_DIM - 1 - dr;
        sc = PATCH_DIM - 1 - dc;
      } else {
        sr = PATCH_DIM - 1 - dc;
        sc = dr;
      }
      const srcIdx = (sr * PATCH_DIM + sc) * 4;
      const dstIdx = dr * PATCH_DIM + dc;
      chw[dstIdx] = (rgb96[srcIdx] / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      chw[planeSize + dstIdx] = (rgb96[srcIdx + 1] / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      chw[2 * planeSize + dstIdx] = (rgb96[srcIdx + 2] / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
  }
  return chw;
}

async function run4xInference(rgb96, onAngle) {
  if (!sessionCache) throw new Error('ONNX engine is not initialized.');
  const angles = [0, 90, 180, 270];
  const rotProbs = {};
  const probsArr = [];
  let cam12 = null;

  for (const ang of angles) {
    onAngle?.(ang);
    await nextPaintFrame();
    const chw = buildRotatedNormalizedCHW(rgb96, ang);
    const inputTensor = new window.ort.Tensor('float32', chw, [1, 3, PATCH_DIM, PATCH_DIM]);
    const outputs = await sessionCache.run({ input: inputTensor });
    const prob = sigmoid(Number(outputs.logit.data[0]));
    rotProbs[`${ang}°`] = prob;
    probsArr.push(prob);
    if (ang === 0 && outputs.cam) {
      cam12 = Float32Array.from(outputs.cam.data);
    }
  }

  const meanProb = probsArr.reduce((a, b) => a + b, 0) / probsArr.length;
  const variance = probsArr.reduce((acc, p) => acc + Math.pow(p - meanProb, 2), 0) / probsArr.length;
  const ttaStd = Math.sqrt(variance);
  return { meanProb, ttaStd, rotProbs, cam12 };
}

function cubicWeight(x) {
  const a = -0.75; // OpenCV INTER_CUBIC Catmull-Rom coefficient
  const ax = Math.abs(x);
  if (ax <= 1) return (a + 2) * ax * ax * ax - (a + 3) * ax * ax + 1;
  if (ax < 2) return a * ax * ax * ax - 5 * a * ax * ax + 8 * a * ax - 4 * a;
  return 0;
}

function resizeFloatGridCubic(src, sw, sh, dw, dh) {
  const tmp = new Float32Array(sh * dw);
  const scaleX = sw / dw;
  const scaleY = sh / dh;

  for (let x = 0; x < dw; x++) {
    const sx = (x + 0.5) * scaleX - 0.5;
    const ix = Math.floor(sx);
    const fx = sx - ix;
    const w0 = cubicWeight(fx + 1);
    const w1 = cubicWeight(fx);
    const w2 = cubicWeight(fx - 1);
    const w3 = cubicWeight(fx - 2);
    const c0 = Math.max(0, Math.min(sw - 1, ix - 1));
    const c1 = Math.max(0, Math.min(sw - 1, ix));
    const c2 = Math.max(0, Math.min(sw - 1, ix + 1));
    const c3 = Math.max(0, Math.min(sw - 1, ix + 2));

    for (let y = 0; y < sh; y++) {
      const base = y * sw;
      tmp[y * dw + x] = src[base + c0] * w0 + src[base + c1] * w1 + src[base + c2] * w2 + src[base + c3] * w3;
    }
  }

  const dst = new Float32Array(dh * dw);
  for (let y = 0; y < dh; y++) {
    const sy = (y + 0.5) * scaleY - 0.5;
    const iy = Math.floor(sy);
    const fy = sy - iy;
    const w0 = cubicWeight(fy + 1);
    const w1 = cubicWeight(fy);
    const w2 = cubicWeight(fy - 1);
    const w3 = cubicWeight(fy - 2);
    const r0 = Math.max(0, Math.min(sh - 1, iy - 1)) * dw;
    const r1 = Math.max(0, Math.min(sh - 1, iy)) * dw;
    const r2 = Math.max(0, Math.min(sh - 1, iy + 1)) * dw;
    const r3 = Math.max(0, Math.min(sh - 1, iy + 2)) * dw;
    const outRow = y * dw;

    for (let x = 0; x < dw; x++) {
      const val = tmp[r0 + x] * w0 + tmp[r1 + x] * w1 + tmp[r2 + x] * w2 + tmp[r3 + x] * w3;
      dst[outRow + x] = val < 0 ? 0 : val > 1 ? 1 : val;
    }
  }
  return dst;
}

function enhanceLuminanceAndSharpen(rgba, w, h, sharpenStrength) {
  const n = w * h;
  const lum = new Float32Array(n);
  const hist = new Uint32Array(256);

  for (let p = 0; p < n; p++) {
    const i = p * 4;
    const y = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
    lum[p] = y;
    hist[Math.max(0, Math.min(255, Math.round(y)))]++;
  }

  const loTarget = n * 0.01;
  const hiTarget = n * 0.99;
  let cum = 0;
  let lo = 0;
  let hi = 255;
  for (let b = 0; b < 256; b++) {
    cum += hist[b];
    if (cum >= loTarget) {
      lo = b;
      break;
    }
  }
  cum = 0;
  for (let b = 255; b >= 0; b--) {
    cum += hist[b];
    if (cum >= n - hiTarget) {
      hi = b;
      break;
    }
  }
  const span = Math.max(12, hi - lo);

  // Subtle 55/45 luminance contrast boost preserving chromatic H&E hue
  for (let p = 0; p < n; p++) {
    const yOrig = lum[p];
    const yStretch = Math.max(0, Math.min(255, ((yOrig - lo) / span) * 255));
    const yTarget = 0.55 * yStretch + 0.45 * yOrig;
    const delta = yTarget - yOrig;
    const i = p * 4;
    rgba[i] = Math.max(0, Math.min(255, Math.round(rgba[i] + delta)));
    rgba[i + 1] = Math.max(0, Math.min(255, Math.round(rgba[i + 1] + delta)));
    rgba[i + 2] = Math.max(0, Math.min(255, Math.round(rgba[i + 2] + delta)));
  }

  if (sharpenStrength > 0) {
    const tmp = new Float32Array(n * 3);
    // Horizontal 5-tap Gaussian [1, 4, 6, 4, 1] / 16
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const xm2 = Math.max(0, x - 2);
        const xm1 = Math.max(0, x - 1);
        const xp1 = Math.min(w - 1, x + 1);
        const xp2 = Math.min(w - 1, x + 2);
        const i0 = (row + xm2) * 4;
        const i1 = (row + xm1) * 4;
        const i2 = (row + x) * 4;
        const i3 = (row + xp1) * 4;
        const i4 = (row + xp2) * 4;
        const out = (row + x) * 3;
        tmp[out] = (rgba[i0] + 4 * rgba[i1] + 6 * rgba[i2] + 4 * rgba[i3] + rgba[i4]) * 0.0625;
        tmp[out + 1] = (rgba[i0 + 1] + 4 * rgba[i1 + 1] + 6 * rgba[i2 + 1] + 4 * rgba[i3 + 1] + rgba[i4 + 1]) * 0.0625;
        tmp[out + 2] = (rgba[i0 + 2] + 4 * rgba[i1 + 2] + 6 * rgba[i2 + 2] + 4 * rgba[i3 + 2] + rgba[i4 + 2]) * 0.0625;
      }
    }

    // Vertical 5-tap pass + Unsharp Mask in-place
    const orig = new Uint8ClampedArray(rgba);
    for (let y = 0; y < h; y++) {
      const ym2 = Math.max(0, y - 2) * w;
      const ym1 = Math.max(0, y - 1) * w;
      const y0 = y * w;
      const yp1 = Math.min(h - 1, y + 1) * w;
      const yp2 = Math.min(h - 1, y + 2) * w;
      for (let x = 0; x < w; x++) {
        const b0 = (ym2 + x) * 3;
        const b1 = (ym1 + x) * 3;
        const b2 = (y0 + x) * 3;
        const b3 = (yp1 + x) * 3;
        const b4 = (yp2 + x) * 3;
        const idx = (y0 + x) * 4;
        for (let c = 0; c < 3; c++) {
          const blur = (tmp[b0 + c] + 4 * tmp[b1 + c] + 6 * tmp[b2 + c] + 4 * tmp[b3 + c] + tmp[b4 + c]) * 0.0625;
          const sharp = orig[idx + c] + sharpenStrength * (orig[idx + c] - blur);
          rgba[idx + c] = sharp < 0 ? 0 : sharp > 255 ? 255 : Math.round(sharp);
        }
      }
    }
  }
}

function drawRoiReticle(ctx, isWhite = false) {
  const x0 = 192;
  const y0 = 192;
  const size = 192;
  const x1 = x0 + size;
  const y1 = y0 + size;
  const tick = 16;

  ctx.save();
  ctx.strokeStyle = isWhite ? 'rgb(255, 255, 255)' : 'rgb(26, 168, 108)';
  ctx.lineWidth = 2;
  ctx.strokeRect(x0, y0, size, size);

  ctx.strokeStyle = isWhite ? 'rgb(255, 255, 255)' : 'rgb(124, 232, 166)';
  ctx.lineWidth = 3;
  ctx.lineCap = 'square';
  ctx.beginPath();
  // Top-left
  ctx.moveTo(x0, y0 + tick);
  ctx.lineTo(x0, y0);
  ctx.lineTo(x0 + tick, y0);
  // Top-right
  ctx.moveTo(x1 - tick, y0);
  ctx.lineTo(x1, y0);
  ctx.lineTo(x1, y0 + tick);
  // Bottom-left
  ctx.moveTo(x0, y1 - tick);
  ctx.lineTo(x0, y1);
  ctx.lineTo(x0 + tick, y1);
  // Bottom-right
  ctx.moveTo(x1 - tick, y1);
  ctx.lineTo(x1, y1);
  ctx.lineTo(x1, y1 - tick);
  ctx.stroke();
  ctx.restore();
}

function computeAttribution(cam12, ttaStd) {
  const cam96 = resizeFloatGridCubic(cam12, 12, 12, PATCH_DIM, PATCH_DIM);
  let centerSum = 0;
  let outerSum = 0;
  let peakCam = 0;

  for (let r = 0; r < PATCH_DIM; r++) {
    for (let c = 0; c < PATCH_DIM; c++) {
      const v = cam96[r * PATCH_DIM + c];
      if (r >= 32 && r < 64 && c >= 32 && c < 64) {
        centerSum += v;
        if (v > peakCam) peakCam = v;
      } else {
        outerSum += v;
      }
    }
  }

  const centerMean = centerSum / (32 * 32);
  const outerMean = outerSum / (PATCH_DIM * PATCH_DIM - 32 * 32);
  const ttaStability = Math.max(0, 1 - ttaStd);
  return {
    center_peak: roundTo(peakCam * 100, 1),
    center_mean: roundTo(centerMean * 100, 1),
    outer_mean: roundTo(outerMean * 100, 1),
    tta_stability: roundTo(ttaStability * 100, 1),
  };
}

function renderOpticsFromCache(rgb96, cam12, enhance, sharpen) {
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = PATCH_DIM;
  srcCanvas.height = PATCH_DIM;
  const srcCtx = srcCanvas.getContext('2d');
  srcCtx.putImageData(new ImageData(new Uint8ClampedArray(rgb96), PATCH_DIM, PATCH_DIM), 0, 0);

  const dispCanvas = document.createElement('canvas');
  dispCanvas.width = DISPLAY_DIM;
  dispCanvas.height = DISPLAY_DIM;
  const dispCtx = dispCanvas.getContext('2d', { willReadFrequently: true });
  dispCtx.imageSmoothingEnabled = Boolean(enhance);
  if (enhance) dispCtx.imageSmoothingQuality = 'high';
  dispCtx.drawImage(srcCanvas, 0, 0, DISPLAY_DIM, DISPLAY_DIM);

  const baseImgData = dispCtx.getImageData(0, 0, DISPLAY_DIM, DISPLAY_DIM);
  if (enhance) {
    enhanceLuminanceAndSharpen(baseImgData.data, DISPLAY_DIM, DISPLAY_DIM, Number(sharpen || 0));
    dispCtx.putImageData(baseImgData, 0, 0);
  }

  const cleanCopy = new Uint8ClampedArray(baseImgData.data);
  drawRoiReticle(dispCtx, false);
  const roi_image = dispCanvas.toDataURL('image/png');

  // Layer-4 Grad-CAM JET blend
  const camCanvas = document.createElement('canvas');
  camCanvas.width = DISPLAY_DIM;
  camCanvas.height = DISPLAY_DIM;
  const camCtx = camCanvas.getContext('2d');
  const cam576 = resizeFloatGridCubic(cam12, 12, 12, DISPLAY_DIM, DISPLAY_DIM);
  const blendData = camCtx.createImageData(DISPLAY_DIM, DISPLAY_DIM);
  const out = blendData.data;

  for (let p = 0; p < DISPLAY_DIM * DISPLAY_DIM; p++) {
    const idx = p * 4;
    const lutIdx = Math.max(0, Math.min(255, Math.round(cam576[p] * 255))) * 3;
    out[idx] = Math.round(0.6 * cleanCopy[idx] + 0.4 * JET_LUT[lutIdx]);
    out[idx + 1] = Math.round(0.6 * cleanCopy[idx + 1] + 0.4 * JET_LUT[lutIdx + 1]);
    out[idx + 2] = Math.round(0.6 * cleanCopy[idx + 2] + 0.4 * JET_LUT[lutIdx + 2]);
    out[idx + 3] = 255;
  }
  camCtx.putImageData(blendData, 0, 0);
  drawRoiReticle(camCtx, true);
  const cam_image = camCanvas.toDataURL('image/png');

  return { roi_image, cam_image };
}

export async function analyzeSlideLocal({ sampleName, file, threshold = 0.1261, enhance = true, sharpen = 1.25, onAngle }) {
  if (!manifestCache || !sessionCache) {
    await initLocalEngine();
  }

  const t0 = performance.now();
  let sourceInput;
  let specimen;
  let source;

  if (file) {
    sourceInput = file;
    specimen = file.name || 'uploaded_patch.png';
    source = 'External Upload';
  } else if (sampleName) {
    const item = manifestCache.samples.find((s) => s.filename === sampleName);
    sourceInput = item ? item.thumb_url : `./samples/${sampleName}`;
    specimen = sampleName;
    source = 'Reference Biopsy Bank';
  } else {
    throw new Error('Provide either a reference sample or an uploaded biopsy file.');
  }

  const decoded = await decodeToRgb96(sourceInput);
  const { meanProb, ttaStd, rotProbs, cam12 } = await run4xInference(decoded.rgb96, onAngle);
  const attribution = computeAttribution(cam12, ttaStd);
  const { roi_image, cam_image } = renderOpticsFromCache(decoded.rgb96, cam12, enhance, sharpen);
  const elapsedMs = roundTo(performance.now() - t0, 1);

  const rotProbsPct = {};
  for (const [k, v] of Object.entries(rotProbs)) {
    rotProbsPct[k] = roundTo(v * 100, 2);
  }

  const result = {
    specimen,
    source,
    native_size: `${decoded.nativeWidth}×${decoded.nativeHeight} px`,
    mean_intensity: roundTo(decoded.meanIntensity, 1),
    mean_prob: roundTo(meanProb * 100, 2),
    raw_prob: roundTo(meanProb, 6),
    threshold: roundTo(threshold * 100, 2),
    is_cancer: meanProb >= threshold,
    rot_probs: rotProbsPct,
    tta_std: roundTo(ttaStd * 100, 2),
    elapsed_ms: elapsedMs,
    attribution,
    roi_image,
    cam_image,
  };

  lastSlideCache = {
    rgb96: decoded.rgb96,
    cam12,
    ttaStd,
    baseResult: result,
  };

  return result;
}

export function reRenderLastOptics(enhance = true, sharpen = 1.25) {
  if (!lastSlideCache) return null;
  const { roi_image, cam_image } = renderOpticsFromCache(
    lastSlideCache.rgb96,
    lastSlideCache.cam12,
    enhance,
    sharpen,
  );
  lastSlideCache.baseResult = {
    ...lastSlideCache.baseResult,
    roi_image,
    cam_image,
  };
  return lastSlideCache.baseResult;
}

function parseCsvLine(line) {
  const cells = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"' && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        cur += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      cells.push(cur.trim());
      cur = '';
    } else {
      cur += ch;
    }
  }
  cells.push(cur.trim());
  return cells;
}

async function parseGroundTruthCsv(csvFile) {
  const lookup = new Map();
  if (!csvFile) return lookup;
  const text = await csvFile.text();
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) return lookup;

  const header = parseCsvLine(lines[0]).map((h) => h.toLowerCase());
  const idIdx = header.indexOf('id');
  const lblIdx = header.indexOf('label');
  if (idIdx < 0 || lblIdx < 0) return lookup;

  for (let i = 1; i < lines.length; i++) {
    const cols = parseCsvLine(lines[i]);
    if (cols.length <= Math.max(idIdx, lblIdx)) continue;
    const lbl = Number.parseInt(cols[lblIdx], 10);
    if (!Number.isNaN(lbl)) lookup.set(String(cols[idIdx]), lbl);
  }
  return lookup;
}

function escapeCsvCell(val) {
  const s = String(val ?? '');
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export async function runBatchLocal({ files, useBuiltin = true, gtCsvFile = null, threshold = 0.1261, onProgress }) {
  if (!manifestCache || !sessionCache) {
    await initLocalEngine();
  }

  const t0 = performance.now();
  const gtLookup = await parseGroundTruthCsv(gtCsvFile);
  const items = [];

  if (files && files.length > 0) {
    for (const f of files) {
      if (f.size > 0) items.push({ filename: f.name || 'patch.png', file: f });
    }
  } else if (useBuiltin) {
    for (const s of manifestCache.samples) {
      items.push({ filename: s.filename, url: s.thumb_url });
    }
  }

  if (items.length === 0) {
    throw new Error('No slides provided for batch screening.');
  }

  const rows = [];
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    onProgress?.(i + 1, items.length);
    const decoded = await decodeToRgb96(item.file || item.url);
    const { meanProb, rotProbs } = await run4xInference(decoded.rgb96);
    const pred = meanProb >= threshold ? 1 : 0;
    const stem = item.filename.replace(/\.[^.]+$/, '');
    const lower = item.filename.toLowerCase();
    const trueLbl = gtLookup.has(stem)
      ? gtLookup.get(stem)
      : (lower.includes('cancer') ? 1 : (lower.includes('normal') ? 0 : null));

    rows.push({
      specimen: item.filename,
      thumb_b64: decoded.thumb56Url,
      verdict: pred === 1 ? 'METASTATIC' : 'BENIGN',
      prob_pct: roundTo(meanProb * 100, 2),
      risk_tier: meanProb >= 0.75 ? 'High Risk' : (meanProb >= threshold ? 'Moderate Risk' : 'Low Risk'),
      ground_truth: trueLbl === 1 ? 'Metastatic (1)' : (trueLbl === 0 ? 'Benign (0)' : 'Unlabeled'),
      audit: trueLbl !== null ? (pred === trueLbl ? 'Correct' : 'Mismatch') : '—',
      rot_0: roundTo(rotProbs['0°'] * 100, 1),
      rot_90: roundTo(rotProbs['90°'] * 100, 1),
      rot_180: roundTo(rotProbs['180°'] * 100, 1),
      rot_270: roundTo(rotProbs['270°'] * 100, 1),
      _pred: pred,
      _ok: trueLbl !== null ? (pred === trueLbl ? 1 : 0) : null,
    });
  }

  rows.sort((a, b) => b.prob_pct - a.prob_pct);
  const posN = rows.reduce((s, r) => s + r._pred, 0);
  const labeled = rows.filter((r) => r._ok !== null).map((r) => r._ok);
  const verifiedAcc = labeled.length
    ? roundTo((labeled.reduce((a, b) => a + b, 0) / labeled.length) * 100, 2)
    : null;

  const uiRows = rows.map(({ _pred, _ok, ...rest }) => rest);
  const csvCols = ['specimen', 'verdict', 'prob_pct', 'risk_tier', 'ground_truth', 'audit', 'rot_0', 'rot_90', 'rot_180', 'rot_270'];
  const csvLines = [csvCols.join(',')];
  for (const r of uiRows) {
    csvLines.push(csvCols.map((c) => escapeCsvCell(r[c])).join(','));
  }
  const meanRisk = roundTo(uiRows.reduce((s, r) => s + r.prob_pct, 0) / uiRows.length, 2);

  return {
    total: uiRows.length,
    positive: posN,
    negative: uiRows.length - posN,
    verified_accuracy: verifiedAcc,
    mean_risk: meanRisk,
    elapsed_ms: roundTo(performance.now() - t0, 1),
    rows: uiRows,
    csv: csvLines.join('\n') + '\n',
  };
}

export async function initBiomarkerLocal() {
  if (!bioCache) {
    const res = await fetch('./models/biomarker_rf.json');
    if (!res.ok) throw new Error(`Failed to fetch biomarker_rf.json (${res.status})`);
    bioCache = await res.json();
  }
  return {
    accuracy: bioCache.accuracy,
    n_samples: bioCache.n_samples,
    features: bioCache.features,
    defaults: bioCache.defaults,
    ranges: bioCache.ranges,
    presets: bioCache.presets,
    benign: bioCache.benign,
    malignant: bioCache.malignant,
  };
}

export async function predictBiomarkerLocal(values) {
  if (!bioCache) await initBiomarkerLocal();

  const missing = bioCache.features.filter((f) => !(f in values));
  if (missing.length > 0) {
    throw new Error(`Missing biomarker features: ${missing.join(', ')}`);
  }

  // Match scikit-learn internal float32 node threshold comparison
  const row = bioCache.features.map((f) => Math.fround(Number(values[f])));
  let sumProb = 0;

  for (const tr of bioCache.trees) {
    let node = 0;
    const { f, t, l, r, p } = tr;
    while (f[node] >= 0) {
      node = row[f[node]] <= t[node] ? l[node] : r[node];
    }
    sumProb += p[node];
  }

  const probMal = sumProb / bioCache.trees.length;
  const riskTier = probMal >= 0.7 ? 'High Risk' : (probMal >= 0.35 ? 'Borderline' : 'Low Risk');

  return {
    malignant_prob: roundTo(probMal * 100, 2),
    benign_prob: roundTo((1 - probMal) * 100, 2),
    risk_tier: riskTier,
    importances: bioCache.importances,
  };
}
