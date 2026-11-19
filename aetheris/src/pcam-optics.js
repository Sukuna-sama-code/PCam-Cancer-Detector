// Separable Lanczos-4 upsampling. Optical changes apply only to the displayed patch.
const N = 96,
  S = 576;
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
function reflect(i, n) {
  return i < 0 ? -i : i >= n ? 2 * n - 2 - i : i;
}
function sinc(x) {
  return Math.abs(x) < 1e-9 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
}
const kernels = Array.from({ length: S }, (_, i) => {
  const x = (i + 0.5) / 6 - 0.5,
    start = Math.floor(x) - 3;
  const weights = Array.from(
    { length: 8 },
    (_, j) => sinc(x - start - j) * sinc((x - start - j) / 4),
  );
  const sum = weights.reduce((a, b) => a + b, 0);
  return weights.map((w, j) => [clamp(start + j, 0, N - 1), w / sum]);
});
function lanczos(rgba) {
  const h = new Float32Array(N * S * 3),
    out = new Uint8ClampedArray(S * S * 4);
  for (let y = 0; y < N; y++)
    for (let x = 0; x < S; x++)
      for (let c = 0; c < 3; c++) {
        let v = 0;
        for (const [j, w] of kernels[x]) v += w * rgba[4 * (y * N + j) + c];
        h[(y * S + x) * 3 + c] = v;
      }
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++) {
      let i = (y * S + x) * 4;
      out[i + 3] = 255;
      for (let c = 0; c < 3; c++) {
        let v = 0;
        for (const [j, w] of kernels[y]) v += w * h[(j * S + x) * 3 + c];
        out[i + c] = v;
      }
    }
  return out;
}
const linear = Array.from({ length: 256 }, (_, i) => {
  let v = i / 255;
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
});
const transfer = (v) =>
  255 *
  (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.max(0, v) ** (1 / 2.4) - 0.055);
const lab = (v) =>
  v > 216 / 24389 ? Math.cbrt(v) : ((24389 / 27) * v + 16) / 116;
const invLab = (v) => (v > 6 / 29 ? v ** 3 : ((116 * v - 16) * 27) / 24389);
function contrast(rgb) {
  const count = S * S,
    tile = S / 8,
    hist = Array.from({ length: 64 }, () => new Uint32Array(256));
  const L = new Uint8ClampedArray(count),
    aa = new Float32Array(count),
    bb = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const r = linear[rgb[4 * i]],
      g = linear[rgb[4 * i + 1]],
      b = linear[rgb[4 * i + 2]];
    const fx = lab((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047),
      fy = lab(0.2126729 * r + 0.7151522 * g + 0.072175 * b),
      fz = lab((0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883);
    L[i] = (116 * fy - 16) * 2.55;
    aa[i] = fx - fy;
    bb[i] = fy - fz;
    hist[Math.floor(Math.floor(i / S) / tile) * 8 + Math.floor((i % S) / tile)][
      L[i]
    ]++;
  }
  const limit = Math.max(1, Math.floor((1.6 * tile * tile) / 256));
  for (const lut of hist) {
    let overflow = 0;
    for (let j = 0; j < 256; j++) {
      overflow += Math.max(0, lut[j] - limit);
      lut[j] = Math.min(lut[j], limit);
    }
    const bonus = Math.floor(overflow / 256),
      rem = overflow % 256,
      step = rem ? Math.floor(256 / rem) : 256;
    let used = 0,
      sum = 0;
    for (let j = 0; j < 256; j++) {
      sum += lut[j] + bonus + (used < rem && j % step === 0 ? (used++, 1) : 0);
      lut[j] = Math.round((sum * 255) / (tile * tile));
    }
  }
  for (let i = 0; i < count; i++) {
    const x = (i % S) / tile - 0.5,
      y = Math.floor(i / S) / tile - 0.5,
      ix = Math.floor(x),
      iy = Math.floor(y),
      tx = x - ix,
      ty = y - iy;
    const lookup = (a, b) => hist[clamp(b, 0, 7) * 8 + clamp(a, 0, 7)][L[i]];
    const enhanced =
      (1 - ty) * ((1 - tx) * lookup(ix, iy) + tx * lookup(ix + 1, iy)) +
      ty * ((1 - tx) * lookup(ix, iy + 1) + tx * lookup(ix + 1, iy + 1));
    const fy = ((0.55 * enhanced + 0.45 * L[i]) / 2.55 + 16) / 116,
      X = 0.95047 * invLab(fy + aa[i]),
      Y = invLab(fy),
      Z = 1.08883 * invLab(fy - bb[i]);
    rgb[4 * i] = transfer(3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z);
    rgb[4 * i + 1] = transfer(-0.969266 * X + 1.8760108 * Y + 0.041556 * Z);
    rgb[4 * i + 2] = transfer(0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z);
  }
  return rgb;
}
function sharpen(rgb, strength) {
  if (!strength) return rgb;
  const weights = Array.from({ length: 9 }, (_, j) =>
      Math.exp(-((j - 4) ** 2) / (2 * 1.4 ** 2)),
    ),
    sum = weights.reduce((a, b) => a + b);
  const tmp = new Float32Array(S * S * 3),
    out = new Uint8ClampedArray(rgb);
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++)
      for (let c = 0; c < 3; c++) {
        let v = 0;
        for (let j = -4; j <= 4; j++)
          v += weights[j + 4] * rgb[4 * (y * S + reflect(x + j, S)) + c];
        tmp[3 * (y * S + x) + c] = v / sum;
      }
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++)
      for (let c = 0; c < 3; c++) {
        let v = 0;
        for (let j = -4; j <= 4; j++)
          v += weights[j + 4] * tmp[3 * (reflect(y + j, S) * S + x) + c];
        const i = 4 * (y * S + x) + c;
        out[i] = rgb[i] + strength * (rgb[i] - v / sum);
      }
  return out;
}
let lastPixels, enhanced;
self.onmessage = ({ data: { pixels, strength, id } }) => {
  if (pixels) {
    lastPixels = pixels;
    enhanced = contrast(lanczos(lastPixels));
  }
  const output = sharpen(enhanced, strength);
  self.postMessage({ pixels: output, id });
};
