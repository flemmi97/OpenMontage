// usage: node qa_stills.mjs <entry> <compId> <propsJson> <publicDir> <outDir> <scale> <t1,t2,...>
import { bundle } from "@remotion/bundler";
import { renderStill, selectComposition } from "@remotion/renderer";
import fs from "fs"; import path from "path";
const [entry, id, propsPath, publicDir, outDir, scale, times] = process.argv.slice(2);
const inputProps = JSON.parse(fs.readFileSync(propsPath, "utf8"));
const serveUrl = await bundle({ entryPoint: path.resolve(entry), publicDir: path.resolve(publicDir) });
const browserExecutable = "/opt/pw-browsers/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell";
const composition = await selectComposition({ serveUrl, id, inputProps, browserExecutable });
fs.mkdirSync(outDir, { recursive: true });
for (const t of times.split(",")) {
  const frame = Math.min(composition.durationInFrames - 1, Math.round(parseFloat(t) * 30));
  await renderStill({ composition, serveUrl, output: path.join(outDir, `s_${t}.png`), frame, inputProps, scale: parseFloat(scale), browserExecutable });
}
console.log("done", times.split(",").length);
