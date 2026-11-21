import { useCallback, useRef } from "react";
import { cubicBezier, type Bezier } from "@/lib/motion";

const SIZE = 150;
const PAD = 26;
const INNER = SIZE - PAD * 2;
// y is allowed to overshoot below 0 and above 1 so "back"/elastic-style curves are reachable.
const Y_MIN = -0.6;
const Y_MAX = 1.6;
const Y_SPAN = Y_MAX - Y_MIN;

const toPx = (x: number, y: number): [number, number] => [
  PAD + x * INNER,
  PAD + (Y_MAX - y) / Y_SPAN * INNER,
];
const fromPx = (px: number, py: number): [number, number] => {
  const x = Math.min(1, Math.max(0, (px - PAD) / INNER));
  const y = Math.min(Y_MAX, Math.max(Y_MIN, Y_MAX - ((py - PAD) / INNER) * Y_SPAN));
  return [x, y];
};

function curvePoints(ease: (x: number) => number): string {
  const steps = 48;
  const pts: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const x = i / steps;
    const [px, py] = toPx(x, ease(x));
    pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
  }
  return pts.join(" ");
}

/**
 * Cubic-bezier easing picker: drag the two control handles to shape the pop curve. Handles snap
 * horizontally to [0,1] (a function must stay single-valued) and vertically to [0.4,-] so overshoot
 * reads as a bounce. The dashed y=1 line marks the settled value.
 */
export function CurveEditor({ value, onChange }: { value: Bezier; onChange: (b: Bezier) => void }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<0 | 1 | null>(null);

  const [x1, y1, x2, y2] = value;
  const ease = cubicBezier(x1, y1, x2, y2);
  const [p1x, p1y] = toPx(x1, y1);
  const [p2x, p2y] = toPx(x2, y2);
  const [ox, oy] = toPx(0, 0);
  const [ex, ey] = toPx(1, 1);

  const update = useCallback(
    (event: PointerEvent | React.PointerEvent) => {
      const svg = svgRef.current;
      const slot = drag.current;
      if (!svg || slot === null) return;
      const rect = svg.getBoundingClientRect();
      const scale = SIZE / rect.width;
      const [nx, ny] = fromPx((event.clientX - rect.left) * scale, (event.clientY - rect.top) * scale);
      if (slot === 0) onChange([nx, ny, x2, y2]);
      else onChange([x1, y1, nx, ny]);
    },
    [onChange, x1, y1, x2, y2],
  );

  const start = (slot: 0 | 1) => (e: React.PointerEvent) => {
    drag.current = slot;
    const move = (ev: PointerEvent) => update(ev);
    const stop = () => {
      drag.current = null;
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
  };

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      className="w-full touch-none select-none rounded-md border border-border bg-background"
      role="img"
      aria-label="Easing curve editor"
    >
      {/* unit guides */}
      <line x1={PAD} y1={oy} x2={SIZE - PAD} y2={oy} stroke="var(--border)" strokeDasharray="3 3" />
      <line x1={PAD} y1={ey} x2={SIZE - PAD} y2={ey} stroke="var(--border)" strokeDasharray="3 3" />
      <line x1={PAD} y1={PAD} x2={PAD} y2={SIZE - PAD} stroke="var(--border)" />
      {/* control arms */}
      <line x1={ox} y1={oy} x2={p1x} y2={p1y} stroke="var(--muted-foreground)" strokeWidth={1} />
      <line x1={ex} y1={ey} x2={p2x} y2={p2y} stroke="var(--muted-foreground)" strokeWidth={1} />
      {/* the eased curve itself */}
      <polyline points={curvePoints(ease)} fill="none" stroke="var(--primary)" strokeWidth={2.5} />
      {/* endpoints */}
      <circle cx={ox} cy={oy} r={3} fill="var(--muted-foreground)" />
      <circle cx={ex} cy={ey} r={3} fill="var(--muted-foreground)" />
      {/* draggable handles */}
      <circle cx={p1x} cy={p1y} r={8} fill="var(--primary)" opacity={0.25} onPointerDown={start(0)} style={{ cursor: "grab" }} />
      <circle cx={p1x} cy={p1y} r={4.5} fill="var(--primary)" onPointerDown={start(0)} style={{ cursor: "grab" }} />
      <circle cx={p2x} cy={p2y} r={8} fill="var(--hot)" opacity={0.25} onPointerDown={start(1)} style={{ cursor: "grab" }} />
      <circle cx={p2x} cy={p2y} r={4.5} fill="var(--hot)" onPointerDown={start(1)} style={{ cursor: "grab" }} />
    </svg>
  );
}
