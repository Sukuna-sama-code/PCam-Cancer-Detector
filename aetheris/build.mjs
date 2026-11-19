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
