import React, { useEffect, useRef, useState } from "react";
import {
  Microscope,
  SlidersHorizontal,
  Focus,
  Layers,
  Activity,
  Upload,
  RotateCcw,
  Maximize2,
} from "lucide-react";
import UTIF from "utif";
import manifest from "../pcam-assets/model.json";
import bank from "../pcam-assets/references.json";
const initial = manifest.samples[0],
  baseline = Number(manifest.threshold.toFixed(3));
const percent = (n) => (100 * n).toFixed(2) + "%";
function surface(w = 96, h = w) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}
async function readPatch(source, filename) {
  let image;
  if (source instanceof File && /\.tiff?$/i.test(filename)) {
    const bytes = await source.arrayBuffer(),
      [page] = UTIF.decode(bytes);
    if (!page || page.t256?.[0] * page.t257?.[0] > 16e6)
      throw Error("Choose one tissue patch rather than a whole-slide image.");
    UTIF.decodeImage(bytes, page);
    const rgba = UTIF.toRGBA8(page);
    image = surface(page.width, page.height);
    image
      .getContext("2d")
      .putImageData(
        new ImageData(new Uint8ClampedArray(rgba), page.width, page.height),
        0,
        0,
      );
  } else {
    const blob =
      source instanceof File
        ? source
        : await fetch(source).then((r) => {
            if (!r.ok) throw Error("Reference image unavailable.");
            return r.blob();
          });
    image = await createImageBitmap(blob);
  }
  try {
    const c = surface(),
      ctx = c.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(image, 0, 0, 96, 96);
    return {
      pixels: ctx.getImageData(0, 0, 96, 96).data,
      width: image.width,
      height: image.height,
    };
  } finally {
    image.close?.();
  }
}
function render(view, src, heat, roi) {
  const ctx = view.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, 576, 576);
  ctx.drawImage(src, 0, 0, 576, 576);
  if (heat) {
    const overlay = surface(12),
      g = overlay.getContext("2d"),
      img = g.createImageData(12, 12);
    heat.forEach((v, i) => {
      for (let c = 0; c < 3; c++)
        img.data[4 * i + c] =
          255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - [3, 2, 1][c])));
      img.data[4 * i + 3] = 255;
    });
    g.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.globalAlpha = 0.4;
    ctx.drawImage(overlay, 0, 0, 576, 576);
    ctx.globalAlpha = 1;
  }
  ctx.strokeStyle = roi;
  ctx.lineWidth = 3;
  ctx.strokeRect(192, 192, 192, 192);
}
function Images({ pixels, cam, enhance, strength }) {
  const plain = useRef(),
    hot = useRef(),
    [bitmap, setBitmap] = useState(null),
    [zoom, Z] = useState(""),
    dialog = useRef(),
    worker = useRef(),
    last = useRef(),
    seq = useRef(0);
  useEffect(() => () => worker.current?.terminate(), []);
  useEffect(() => {
    if (!pixels) return;
    const id = ++seq.current;
    let cancelled = false;
    const raw = surface();
    raw
      .getContext("2d")
      .putImageData(new ImageData(new Uint8ClampedArray(pixels), 96, 96), 0, 0);
    setBitmap(raw);
    if (enhance) {
      worker.current ??= new Worker("./pcam-optics.js");
      const sendPixels = last.current !== pixels;
      worker.current.onmessage = ({ data }) => {
        if (cancelled || data.id !== id) return;
        const c = surface(576);
        c.getContext("2d").putImageData(
          new ImageData(data.pixels, 576, 576),
          0,
          0,
        );
        setBitmap(c);
      };
      worker.current.postMessage({
        id,
        pixels: sendPixels ? pixels : null,
        strength,
      });
      last.current = pixels;
    }
    return () => {
      cancelled = true;
    };
  }, [pixels, enhance, strength]);
  useEffect(() => {
    if (!bitmap) return;
    if (plain.current) render(plain.current, bitmap, null, "#39d8b1");
    if (cam && hot.current) render(hot.current, bitmap, cam, "white");
  }, [bitmap, cam]);
  useEffect(() => {
    if (zoom) dialog.current?.showModal();
    else dialog.current?.close();
  }, [zoom]);
  return (
    <>
      {[
        ["H&E brightfield", plain, Focus, "Central 32 × 32 px region"],
        ["Grad-CAM", hot, Layers, "Layer 4 · positive-logit attribution"],
      ].map(([title, ref, I, detail], idx) => (
        <article key={title} className="panel viewport">
          <div className="viewtitle">
            <I size={18} />
            <h3>{title}</h3>
            <button
              aria-label={"Enlarge " + title}
              disabled={!pixels || (idx === 1 && !cam)}
              onClick={() => Z(ref.current.toDataURL())}
            >
              <Maximize2 size={16} />
            </button>
          </div>
          {pixels && (idx === 0 || cam) ? (
            <canvas
              ref={ref}
              width="576"
              height="576"
              aria-label={title + " of current specimen"}
            />
          ) : (
            <div className="imagewait">
              {idx === 1 ? "Awaiting model output" : "Awaiting specimen"}
            </div>
          )}
          <small>{detail}</small>
        </article>
      ))}
      <dialog ref={dialog} onClose={() => Z("")}>
        <button className="subbtn" onClick={() => Z("")}>
          Close image
        </button>
        {zoom && (
          <img
            src={zoom}
            alt="Enlarged patch with central ROI"
            style={{ width: "100%" }}
          />
        )}
      </dialog>
    </>
  );
}
export function Detector({ bridge, openNative }) {
  const [spec, select] = useState({ name: initial }),
    [pixels, P] = useState(null),
    [nativeSize, setNativeSize] = useState(null),
    [result, R] = useState(null),
    [status, U] = useState(""),
    [busy, B] = useState(false),
    [threshold, T] = useState(baseline),
    [enhance, H] = useState(true),
    [strength, S] = useState(1.25),
    [weights, W] = useState(null),
    [engine, A] = useState(false);
  const version = useRef(0),
    worker = useRef(),
    upload = useRef(),
    bundle = useRef(),
    weightsSent = useRef(null);
  function stop() {
    version.current++;
    worker.current?.terminate();
    worker.current = null;
    weightsSent.current = null;
    A(false);
    B(false);
  }
  useEffect(
    () => () => {
      worker.current?.terminate();
    },
    [],
  );
  useEffect(() => {
    let active = true;
    P(null);
    setNativeSize(null);
    R(null);
    U("Loading specimen…");
    readPatch(spec.file || "./pcam-assets/" + spec.name, spec.name)
      .then((data) => {
        if (!active) return;
        P(data.pixels);
        setNativeSize([data.width, data.height]);
        R(spec.file ? null : bank[spec.name]);
        U("");
      })
      .catch((e) => {
        if (active) U(e.message);
      });
    return () => {
      active = false;
    };
  }, [spec]);
  function change(next) {
    if (busy) stop();
    select(next);
  }
  function analyze() {
    if (!pixels || busy) return;
    const id = ++version.current;
    R(null);
    B(true);
    U("Starting analysis…");
    worker.current ??= new Worker("./pcam-worker.js", { type: "module" });
    const fail = (message) => {
      if (version.current !== id) return;
      stop();
      U(message);
    };
    worker.current.onerror = (e) =>
      fail(
        "Inference runtime failed to start. " +
          (e.message || "Reload and retry."),
      );
    worker.current.onmessage = ({ data }) => {
      if (version.current !== id) return;
      if (data.error) {
        fail(data.error);
        return;
      }
      if (data.result) {
        R({ ...data.result, live: true });
        weightsSent.current = weights;
        B(false);
        A(true);
        U("");
      } else U(data.status);
    };
    worker.current.postMessage({
      pixels,
      file: weightsSent.current === weights ? null : weights,
      manifest,
    });
  }
  const mean = result?.scores.reduce((a, b) => a + b, 0) / 4,
    above = mean >= threshold;
  const intensity = pixels
    ? pixels.reduce((sum, v, i) => sum + (i % 4 === 3 ? 0 : v), 0) /
      (96 * 96 * 3)
    : null;
  return (
    <div className="pcam">
      <div className="sectionhead">
        <div>
          <span className="kicker">
            <Microscope size={15} /> COMPUTATIONAL PATHOLOGY
          </span>
          <h2>PCam tissue analysis</h2>
        </div>
        {bridge && (
          <button className="subbtn" onClick={openNative}>
            Python workstation
          </button>
        )}
      </div>
      <div className="acquire panel">
        <label className="field">
          Reference biopsy
          <select
            value={spec.file ? "" : spec.name}
            onChange={(e) => {
              if (e.target.value) change({ name: e.target.value });
              else upload.current.click();
            }}
          >
            <option value="">External patch…</option>
            {manifest.samples.map((n) => (
              <option value={n} key={n}>
                {n.startsWith("normal") ? "Normal" : "Metastatic"} reference ·{" "}
                {n}
              </option>
            ))}
          </select>
        </label>
        <div>
          <input
            type="file"
            hidden
            ref={upload}
            accept=".png,.jpg,.jpeg,.tif,.tiff"
            onChange={(e) => {
              const file = e.target.files[0];
              if (file) change({ name: file.name, file });
              e.target.value = "";
            }}
          />
          <button className="subbtn" onClick={() => upload.current.click()}>
            <Upload size={15} />
            Upload patch
          </button>
        </div>
        <button
          className="primary"
          disabled={!pixels || busy}
          onClick={analyze}
        >
          <Activity size={16} />
          {busy ? "Analyzing…" : "Run 4× analysis"}
        </button>
      </div>
      <p className="specmeta">
        <strong>{spec.name}</strong>
        <span>{spec.file ? "External upload" : "Reference bank"}</span>
        <span>
          {nativeSize && (nativeSize[0] !== 96 || nativeSize[1] !== 96)
            ? nativeSize.join(" × ") + " px resized to 96 × 96 px"
            : "Input 96 × 96 px"}
        </span>
        <span>ROI 32 × 32 px</span>
        {intensity !== null && (
          <span>Mean intensity {intensity.toFixed(1)} / 255</span>
        )}
      </p>
      {status && (
        <div role="status" className="panel pcstatus">
          {status}
        </div>
      )}
      <div className="pcgrid">
        <Images
          pixels={pixels}
          cam={result?.cam}
          enhance={enhance}
          strength={strength}
        />
        <article className="panel pcresult">
          <span className="kicker">
            <Activity size={15} /> MODEL OUTPUT
          </span>
          <h3>Patch classification</h3>
          {result ? (
            <>
              <div className={"verdict " + (above ? "flagged" : "")}>
                {above
                  ? "Flagged for metastasis review"
                  : "Below decision threshold"}
              </div>
              <span className="score">{percent(mean)}</span>
              <small>Mean positive-class score · 4 rotations</small>
              <progress
                max="1"
                value={mean}
                aria-label="Mean positive-class score"
              />
              <small>Decision cutoff {percent(threshold)}</small>
              <details open>
                <summary>Rotation scores</summary>
                <div className="rotations">
                  {result.scores.map((v, i) => (
                    <div key={i}>
                      <small>{90 * i}°</small>
                      <b>{percent(v)}</b>
                    </div>
                  ))}
                </div>
              </details>
              <small>
                {result.live
                  ? "Computed in this browser"
                  : "Saved output from the trained checkpoint"}
              </small>
            </>
          ) : (
            <p>Run analysis to see the four scores and activation map.</p>
          )}
        </article>
      </div>
      <section className="panel pccal">
        <div className="sectionhead">
          <div>
            <span className="kicker">
              <SlidersHorizontal size={15} /> WORKSTATION SETTINGS
            </span>
            <h3>System calibration & optics</h3>
          </div>
          <button
            className="subbtn"
            onClick={() => {
              T(baseline);
              H(true);
              S(1.25);
            }}
          >
            <RotateCcw size={14} />
            Reset
          </button>
        </div>
        <div className="calfields">
          <div>
            <strong>Model checkpoint</strong>
            <small>{weights?.name || "pcam_deployment_bundle.pt"}</small>
            <small>
              {engine
                ? "Browser inference online · CPU / WASM"
                : busy
                  ? "Starting inference engine"
                  : "Center-aware ResNet-34 · on demand"}
            </small>
            <input
              hidden
              type="file"
              ref={bundle}
              accept=".pt,.pth"
              onChange={(e) => {
                const f = e.target.files[0];
                if (f) {
                  stop();
                  W(f);
                  R(spec.file ? null : bank[spec.name]);
                  U(
                    "Local checkpoint selected. Run analysis to verify its fingerprint.",
                  );
                }
                e.target.value = "";
              }}
            />
            <button className="textbtn" onClick={() => bundle.current.click()}>
              Use local checkpoint
            </button>
          </div>
          <label className="field">
            Decision threshold <output>{threshold.toFixed(3)}</output>
            <input
              type="range"
              min=".01"
              max=".99"
              step=".001"
              value={threshold}
              onChange={(e) => T(+e.target.value)}
            />
            <small>
              Checkpoint cutoff {manifest.threshold.toFixed(3)} · Youden’s J
            </small>
          </label>
          <div>
            <label className="toggle">
              <input
                type="checkbox"
                checked={enhance}
                onChange={(e) => H(e.target.checked)}
              />{" "}
              HD optical enhancement
            </label>
            <small>6× Lanczos-4 + CLAHE · display only</small>
            <label className="field">
              Nuclear edge sharpening · {strength.toFixed(2)}
              <input
                type="range"
                min="0"
                max="2.5"
                step=".05"
                value={strength}
                disabled={!enhance}
                onChange={(e) => S(+e.target.value)}
              />
            </label>
          </div>
        </div>
      </section>
      <p className="pcnote">
        Research use. Scores describe a single H&E lymph-node patch. A
        pathologist must interpret the tissue in context. Uploaded patches are
        processed in this browser.
      </p>
    </div>
  );
}
