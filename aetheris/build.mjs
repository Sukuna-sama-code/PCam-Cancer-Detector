import { build } from "esbuild";
import { mkdir, writeFile, readFile, copyFile, rm } from "node:fs/promises";
import postcss from "postcss";
import tailwind from "tailwindcss";
await rm("dist", { force: true, recursive: true });
await mkdir("dist", { recursive: true });
await build({
  entryPoints: ["src/app.jsx"],
  bundle: true,
  minify: true,
  outfile: "dist/app.js",
  define: { "process.env.NODE_ENV": '"production"' },
  legalComments: "eof",
});
const css = await postcss([
  tailwind({ content: ["src/*.jsx"], corePlugins: { preflight: false } }),
]).process(await readFile("src/style.css", "utf8"), { from: "src/style.css" });
await writeFile("dist/style.css", css.css);
for (const [file, source] of Object.entries({
  "serif.woff2":
    "instrument-serif/files/instrument-serif-latin-400-normal.woff2",
  "italic.woff2":
    "instrument-serif/files/instrument-serif-latin-400-italic.woff2",
  "inter.woff2": "inter/files/inter-latin-400-normal.woff2",
}))
  await copyFile("node_modules/@fontsource/" + source, "dist/" + file);
for (const font of ["instrument-serif", "inter"])
  await copyFile(
    "node_modules/@fontsource/" + font + "/LICENSE",
    "dist/" + font + "-license.txt",
  );
await copyFile("index.html", "dist/index.html");

const { cp } = await import("node:fs/promises");
await cp("pcam-assets", "dist/pcam-assets", { recursive: true });
for (const name of ["pcam-worker", "pcam-optics"])
  await copyFile("src/" + name + ".js", "dist/" + name + ".js");
// Package a self-contained runtime and checkpoint for private static hosting.
if (process.env.PCAM_CHECKPOINT) {
  const { createHash } = await import("node:crypto");
  const model = JSON.parse(await readFile("pcam-assets/model.json", "utf8"));
  const weights = await readFile(process.env.PCAM_CHECKPOINT);
  if (createHash("sha256").update(weights).digest("hex") !== model.sha256)
    throw Error("Wrong PCam checkpoint");
  await mkdir("dist/runtime", { recursive: true });
  const names = [
    "ort.wasm.min.mjs",
    "ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.wasm",
  ];
  for (const name of names)
    await copyFile(
      "node_modules/onnxruntime-web/dist/" + name,
      "dist/runtime/" + name,
    );
  const parts = [];
  for (let offset = 0; offset < weights.length; offset += 8 * 1024 * 1024) {
    const name = "weights-" + parts.length + ".bin";
    parts.push(name);
    await writeFile(
      "dist/runtime/" + name,
      weights.subarray(offset, offset + 8 * 1024 * 1024),
    );
  }
  await writeFile("dist/runtime/assets.json", JSON.stringify({ parts }));
}
