// The graph points directly into the released PyTorch ZIP's uncompressed tensor storages.
const bundleURL =
  "https://media.githubusercontent.com/media/Sukuna-sama-code/PCam-Cancer-Detector/f2b42f72768a2f088ff73cd4982f6d1bb477fa5c/pcam_deployment_bundle.pt";
const cdn = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
const runtime = new URL("./runtime/", self.location.href).href;
let session, ort, offline;
async function assets() {
  if (offline !== undefined) return offline;
  try {
    const r = await fetch("./runtime/assets.json");
    offline = r.ok ? await r.json() : null;
  } catch {
    offline = null;
  }
  return offline;
}
async function load(file, manifest) {
  if (session && !file) return;
  self.postMessage({ status: "Loading the inference runtime…" });
  const local = await assets();
  const base = local ? runtime : cdn;
  ort ??= await import(base + "ort.wasm.min.mjs");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = base;
  self.postMessage({ status: "Verifying the 86 MB trained checkpoint…" });
  let bytes;
  if (file) bytes = await file.arrayBuffer();
  else if (local?.parts?.length) {
    const whole = new Uint8Array(manifest.size);
    let offset = 0;
    for (const part of local.parts) {
      const res = await fetch(runtime + part);
      if (!res.ok) throw Error("Checkpoint segment unavailable.");
      const segment = new Uint8Array(await res.arrayBuffer());
      whole.set(segment, offset);
      offset += segment.length;
    }
    bytes = whole.buffer;
  } else {
    const response = await fetch(bundleURL);
    if (!response.ok)
      throw Error(
        "Checkpoint download failed. Select the local deployment bundle to retry.",
      );
    bytes = await response.arrayBuffer();
  }
  const digest = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    (b) => b.toString(16).padStart(2, "0"),
  ).join("");
  if (digest !== manifest.sha256)
    throw Error(
      "This file does not match the released PCam checkpoint. Select pcam_deployment_bundle.pt.",
    );
  const next = await ort.InferenceSession.create(
    new URL("./pcam-assets/network.onnx", self.location.href).href,
    {
      executionProviders: ["wasm"],
      externalData: [{ path: "pcam.pt", data: new Uint8Array(bytes) }],
    },
  );
  if (session) await session.release();
  session = next;
}
self.onmessage = async ({ data: { pixels, file, manifest } }) => {
  try {
    await load(file, manifest);
    const scores = [];
    let cam;
    const means = [0.485, 0.456, 0.406],
      sd = [0.229, 0.224, 0.225];
    for (let k = 0; k < 4; k++) {
      self.postMessage({ status: "Running " + k * 90 + "° rotation…" });
      const tensor = new Float32Array(3 * 96 * 96);
      for (let y = 0; y < 96; y++)
        for (let x = 0; x < 96; x++) {
          const pos = [
            y * 96 + x,
            x * 96 + 95 - y,
            (95 - y) * 96 + 95 - x,
            (95 - x) * 96 + y,
          ][k];
          for (let c = 0; c < 3; c++)
            tensor[c * 9216 + y * 96 + x] =
              (pixels[pos * 4 + c] / 255 - means[c]) / sd[c];
        }
      const input = new ort.Tensor("float32", tensor, [1, 3, 96, 96]);
      const result = await session.run({ image: input });
      scores.push(result.score.data[0]);
      if (k === 0) {
        const scale = Math.max(...result.cam.data) + 1e-8;
        cam = Array.from(result.cam.data, (v) => v / scale);
      }
      input.dispose();
      result.score.dispose();
      result.cam.dispose();
    }
    self.postMessage({ result: { scores, cam } });
  } catch (error) {
    self.postMessage({ error: error.message || String(error) });
  }
};
