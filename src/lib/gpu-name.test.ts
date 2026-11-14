import { describe, expect, it } from "vitest";
import { shortGpuName } from "./render";

describe("shortGpuName", () => {
  it.each([
    [
      "ANGLE (NVIDIA, NVIDIA GeForce RTX 2050 Direct3D11 vs_5_0 ps_5_0, D3D11-31.0.15.3623)",
      "NVIDIA GeForce RTX 2050",
    ],
    [
      "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)",
      "SwiftShader",
    ],
    ["ANGLE (Apple, ANGLE Metal Renderer: Apple M1 Pro, Unspecified Version)", "Apple M1 Pro"],
    ["Mesa Intel(R) UHD Graphics 620 (KBL GT2)", "Mesa Intel(R) UHD Graphics 620 (KBL GT2)"],
  ])("%s → %s", (raw, expected) => {
    expect(shortGpuName(raw)).toBe(expected);
  });
});