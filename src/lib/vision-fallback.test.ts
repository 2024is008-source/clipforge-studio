import { beforeEach, describe, expect, it, vi } from "vitest";

type Delegate = "GPU" | "CPU";

const created: Delegate[] = [];
let gpuThrowsOnInference = false;
let gpuFailsToBuild = false;
let cpuThrowsOnInference = false;
let filesetBase: string | null = null;
let filesetFailure: unknown = null;
const modelPaths: string[] = [];

function makeTask(delegate: Delegate, kind: string) {
  return {
    detectForVideo: (_video: HTMLVideoElement, ts: number) => {
      if (delegate === "GPU" && gpuThrowsOnInference) {
        throw new Error("GL context lost");
      }
      if (delegate === "CPU" && cpuThrowsOnInference) {
        throw new Error("model input is invalid");
      }
      return { detections: [{ kind, delegate, ts }] };
    },
    close: vi.fn(),
  };
}

vi.mock("@mediapipe/tasks-vision", () => ({
  FilesetResolver: {
    forVisionTasks: async (base: string) => {
      filesetBase = base;
      if (filesetFailure) throw filesetFailure;
      return {};
    },
  },
  FaceDetector: {
    createFromOptions: async (
      _fs: unknown,
      options: { baseOptions: { delegate: Delegate; modelAssetPath: string } },
    ) => {
      const delegate = options.baseOptions.delegate;
      modelPaths.push(options.baseOptions.modelAssetPath);
      created.push(delegate);
      if (delegate === "GPU" && gpuFailsToBuild) throw new Error("no WebGL2");
      return makeTask(delegate, "face");
    },
  },
  ObjectDetector: {
    createFromOptions: async (
      _fs: unknown,
      options: { baseOptions: { delegate: Delegate; modelAssetPath: string } },
    ) => {
      modelPaths.push(options.baseOptions.modelAssetPath);
      return makeTask(options.baseOptions.delegate, "object");
    },
  },
}));

async function load() {
  vi.resetModules();
  return import("./reframe");
}

const video = {} as HTMLVideoElement;

beforeEach(() => {
  created.length = 0;
  modelPaths.length = 0;
  filesetBase = null;
  filesetFailure = null;
  gpuThrowsOnInference = false;
  gpuFailsToBuild = false;
  cpuThrowsOnInference = false;
});

describe("vision delegate portability", () => {
  it("uses the GPU delegate when it works", async () => {
    const { getFaceDetector } = await load();
    const { det, objDet, delegate } = await getFaceDetector();
    expect(delegate).toBe("GPU");
    expect(det.detectForVideo(video, 1).detections).toHaveLength(1);
    expect(objDet?.detectForVideo(video, 1).detections).toHaveLength(1);
    expect(created).toEqual(["GPU"]);
  });

  it("falls back to CPU when the GPU delegate cannot be created", async () => {
    gpuFailsToBuild = true;
    const { getFaceDetector } = await load();
    const { det, delegate } = await getFaceDetector();
    expect(delegate).toBe("CPU");
    expect(det.detectForVideo(video, 1).detections).toHaveLength(1);
    expect(created).toEqual(["GPU", "CPU"]);
  });

  it("keeps tracking when the GPU delegate is created but fails on a real frame", async () => {
    gpuThrowsOnInference = true;
    const { getFaceDetector, onVisionDelegateChange, lastVisionNotice } = await load();
    const labels: string[] = [];
    onVisionDelegateChange((label) => labels.push(label));

    const view = await getFaceDetector();
    expect(view.delegate).toBe("GPU");
    // The frame that fails is reported empty rather than aborting the whole analysis.
    expect(view.det.detectForVideo(video, 1).detections).toEqual([]);

    await vi.waitFor(async () => expect((await getFaceDetector()).delegate).toBe("CPU"));
    gpuThrowsOnInference = false;
    const recovered = await getFaceDetector();
    expect(recovered.det.detectForVideo(video, 2).detections).toHaveLength(1);
    expect(recovered.objDet?.detectForVideo(video, 2).detections).toHaveLength(1);
    expect(lastVisionNotice()).toContain("continuing on CPU");
    expect(labels).toContain("CPU (Face + Person/Object)");
  });

  it("rebuilds the CPU pair only once across many failing frames", async () => {
    gpuThrowsOnInference = true;
    const { getFaceDetector } = await load();
    const { det } = await getFaceDetector();
    for (let i = 0; i < 25; i++) det.detectForVideo(video, i);
    await vi.waitFor(() => expect(created.filter((d) => d === "CPU")).toHaveLength(1));
    expect(created).toEqual(["GPU", "CPU"]);
  });

  it("surfaces a CPU inference error instead of hiding it", async () => {
    gpuFailsToBuild = true;
    cpuThrowsOnInference = true;
    const { getFaceDetector } = await load();
    const { det, delegate } = await getFaceDetector();
    expect(delegate).toBe("CPU");
    expect(() => det.detectForVideo(video, 1)).toThrow("model input is invalid");
    expect(created).toEqual(["GPU", "CPU"]);
  });
});

describe("vision assets are served by the app, not a CDN", () => {
  it("loads the runtime and both models from same-origin paths", async () => {
    const { getFaceDetector } = await load();
    await getFaceDetector();
    expect(filesetBase).toBe("/vision/wasm");
    expect(modelPaths).toEqual([
      "/vision/models/blaze_face_short_range.tflite",
      "/vision/models/efficientdet_lite0.tflite",
    ]);
    expect([...modelPaths, filesetBase].join(" ")).not.toMatch(/https?:\/\//);
  });

  it("names the missing asset when the runtime fails to load", async () => {
    // FilesetResolver rejects with the <script> element's error Event, which has no message.
    filesetFailure = {
      target: { src: "http://localhost:8080/vision/wasm/vision_wasm_internal.js" },
    };
    const { getFaceDetector, lastVisionError } = await load();
    await expect(getFaceDetector()).rejects.toBeTruthy();
    expect(lastVisionError()).toBe(
      "could not load http://localhost:8080/vision/wasm/vision_wasm_internal.js",
    );
  });

  it("retries from scratch after a failure and clears the reported error", async () => {
    filesetFailure = new Error("offline");
    const { getFaceDetector, lastVisionError } = await load();
    await expect(getFaceDetector()).rejects.toThrow("offline");
    expect(lastVisionError()).toBe("offline");

    filesetFailure = null;
    const { delegate } = await getFaceDetector();
    expect(delegate).toBe("GPU");
    expect(lastVisionError()).toBeNull();
  });
});
