// Stages the MediaPipe runtime next to the app so vision never depends on a CDN.
//
// The WASM binary and the JS glue that loads it must come from the exact same package version.
// Loading the glue from npm and the WASM from `@mediapipe/tasks-vision@latest` silently pairs
// mismatched builds, and a blocked or offline network leaves face tracking dead on arrival.
import { copyFile, mkdir, readFile, stat } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const wasmDir = path.join(root, "public", "vision", "wasm");

const RUNTIME_FILES = [
  "vision_wasm_internal.js",
  "vision_wasm_internal.wasm",
  "vision_wasm_module_internal.js",
  "vision_wasm_module_internal.wasm",
];

const MODEL_FILES = ["blaze_face_short_range.tflite", "efficientdet_lite0.tflite"];

async function sizeOf(target) {
  try {
    return (await stat(target)).size;
  } catch {
    return null;
  }
}

async function main() {
  await mkdir(wasmDir, { recursive: true });
  let packageDir = null;
  for (const name of RUNTIME_FILES) {
    const from = require.resolve(`@mediapipe/tasks-vision/${name}`);
    packageDir = path.dirname(path.dirname(from));
    const to = path.join(wasmDir, name);
    if ((await sizeOf(to)) === (await sizeOf(from))) continue;
    await copyFile(from, to);
  }
  const manifest = JSON.parse(await readFile(path.join(packageDir, "package.json"), "utf8"));
  console.log(`[vision] staged MediaPipe ${manifest.version} runtime into public/vision/wasm`);

  const missing = [];
  for (const name of MODEL_FILES) {
    if ((await sizeOf(path.join(root, "public", "vision", "models", name))) === null) {
      missing.push(name);
    }
  }
  if (missing.length) {
    console.warn(`[vision] missing committed model assets: ${missing.join(", ")}`);
  }
}

// A staging failure must not break `npm install`; the runtime reports its own vision state.
main().catch((err) => console.warn(`[vision] could not stage runtime: ${err.message}`));
