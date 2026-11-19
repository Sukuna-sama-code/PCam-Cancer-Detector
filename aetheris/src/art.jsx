import React, { useRef, useEffect } from "react";
// Draw the warped cellular membrane once per resize. Motion composites the canvas layer.
export function Membrane() {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    function draw() {
      const w = c.clientWidth,
        h = c.clientHeight;
      if (!w || !h) return;
      const d = Math.min(devicePixelRatio || 1, 2);
      c.width = w * d;
      c.height = h * d;
      const g = c.getContext("2d");
      g.scale(d, d);
      const size = Math.min(w * 0.31, h * 0.43),
        cx = w * 0.53,
        cy = h * 0.53;
      function point(u, v) {
        const ripple =
          0.17 * Math.cos(5 * u + v) + 0.1 * Math.sin(3 * u - 2 * v);
        const r = 0.82 + (0.39 + ripple) * Math.cos(v),
          x = r * Math.cos(u),
          y = r * Math.sin(u),
          z = (0.57 + ripple) * Math.sin(v) + 0.2 * Math.sin(3 * u);
        const a = y * 0.63 - z * 0.77,
          b = y * 0.77 + z * 0.63;
        return [
          cx + (x * 0.92 + a * 0.39) * size * 1.22,
          cy + (a * 0.92 - x * 0.39) * size * 1.22,
          b,
        ];
      }
      const lines = [];
      for (let i = 0; i < 142; i++)
        for (let j = 0; j < 48; j++) {
          const u = (i / 142) * Math.PI * 2,
            v = (j / 48) * Math.PI * 2,
            p = point(u, v);
          lines.push([p, point(u, v + Math.PI / 24), false]);
          if (i % 2 === 0) lines.push([p, point(u + Math.PI / 71, v), true]);
        }
      lines.sort((a, b) => a[0][2] - b[0][2]);
      for (const [p, q, cross] of lines) {
        const light = Math.max(0.08, Math.min(1, (p[2] + 1.1) / 2.1));
        g.strokeStyle =
          "rgba(" +
          [
            Math.round(108 + 125 * light),
            Math.round(155 + 75 * light),
            Math.round(112 + 68 * light),
            (0.1 + 0.64 * light) * (cross ? 0.51 : 1),
          ].join(",") +
          ")";
        g.lineWidth = cross ? 0.55 : 0.9;
        g.beginPath();
        g.moveTo(p[0], p[1]);
        g.lineTo(q[0], q[1]);
        g.stroke();
      }
      let seed = 9;
      const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      for (let i = 0; i < 110; i++) {
        g.fillStyle = "rgba(215,224,192," + (0.12 + rand() * 0.58) + ")";
        g.beginPath();
        g.arc(rand() * w, rand() * h, 0.4 + rand(), 0, 7);
        g.fill();
      }
    }
    const observer = new ResizeObserver(draw);
    observer.observe(c);
    draw();
    return () => observer.disconnect();
  }, []);
  return (
    <div className="specimen" aria-hidden="true">
      <i className="orbit orbit1" />
      <i className="orbit orbit2" />
      <canvas ref={ref} />
      <i className="pearl p1" />
      <i className="pearl p2" />
      <i className="pearl p3" />
    </div>
  );
}
