import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const publicVision = path.resolve(process.cwd(), "public", "vision");
const prepareScript = path.resolve(process.cwd(), "scripts", "prepare-vision-assets.mjs");

function sha256(file: string) {
  return createHash("sha256").update(readFileSync(file)).digest("hex");
}

describe("committed vision models", () => {
  // Exact byte lengths of the official float16 builds. A truncated or silently swapped model
  // would still load, so the size is the guard that tracking quality never changes.
  const models = [
    ["blaze_face_short_range.tflite", 229746],
    ["efficientdet_lite0.tflite", 7254339],
  ] as const;

  it.each(models)("ships %s at its original size", (name, bytes) => {
    expect(statSync(`${publicVision}/models/${name}`).size).toBe(bytes);
  });
});

describe("staged vision runtime", () => {
  const runtime = [
    "vision_wasm_internal.js",
    "vision_wasm_internal.wasm",
    "vision_wasm_module_internal.js",
    "vision_wasm_module_internal.wasm",
  ];

  it("is byte-identical to the installed package, so glue and binary can never skew", () => {
    execFileSync(process.execPath, [prepareScript], { stdio: "pipe" });
    for (const name of runtime) {
      const installed = require.resolve(`@mediapipe/tasks-vision/${name}`);
      expect(sha256(`${publicVision}/wasm/${name}`)).toBe(sha256(installed));
    }
  });
});
