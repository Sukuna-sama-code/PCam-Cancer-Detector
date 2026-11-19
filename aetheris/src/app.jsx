import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Orbit } from "lucide-react";
import { Membrane } from "./art";
import { Detector } from "./detector";

const embedded =
  window.parent !== window &&
  new URLSearchParams(location.search).get("streamlitUrl") !== null;
function emit(type, data) {
  window.parent.postMessage(
    { isStreamlitMessage: true, type: "streamlit:" + type, ...data },
    "*",
  );
}
function App() {
  const [bridge, setBridge] = useState(embedded);
  const shell = useRef(null);
  useEffect(() => {
    if (window.parent === window) return;
    function msg(e) {
      if (e.source === window.parent && e.data?.type === "streamlit:render")
        setBridge(true);
    }
    window.addEventListener("message", msg);
    emit("componentReady", { apiVersion: 1 });
    const ro = new ResizeObserver(() =>
      emit("setFrameHeight", { height: document.documentElement.offsetHeight }),
    );
    ro.observe(document.body);
    return () => {
      ro.disconnect();
      window.removeEventListener("message", msg);
    };
  }, []);
  useEffect(() => {
    const el = shell.current,
      mq = matchMedia("(prefers-reduced-motion: reduce)"),
      fine = matchMedia("(pointer: fine)");
    let raf = 0,
      x = 0,
      y = 0;
    function move(e) {
      if (mq.matches || !fine.matches) return;
      x = (e.clientX / innerWidth - 0.5) * 10;
      y = (e.clientY / innerHeight - 0.5) * 8;
      if (!raf)
        raf = requestAnimationFrame(() => {
          el.style.setProperty("--px", x + "px");
          el.style.setProperty("--py", y + "px");
          raf = 0;
        });
    }
    window.addEventListener("pointermove", move, { passive: true });
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("pointermove", move);
    };
  }, []);

  function openNative() {
    emit("setComponentValue", { value: "detector", dataType: "json" });
  }
  return (
    <div ref={shell} className="shell">
      <div className="ambient" aria-hidden="true" />
      <header>
        <div className="bar">
          <div className="brand">
            <span className="brandmark">
              <Orbit size={30} />
            </span>
            <div>
              Aetheris<small>PCAM DETECTOR</small>
            </div>
          </div>
          <span className="demo">
            <i />
            Pathology workspace
          </span>
        </div>
      </header>
      <main>
        <section className="hero">
          <div className="hero-copy relative">
            <span className="overline">
              <span className="shortline" />
              LYMPH-NODE PATCH ANALYSIS
            </span>
            <h1>
              PCam <em>Histopathology</em>
            </h1>
            <p>
              Examine a lymph-node patch with the trained center-aware
              ResNet-34, four rotational views and layer-4 Grad-CAM.
            </p>
          </div>
          <Membrane />
        </section>
        <section className="detector-workspace">
          <Detector bridge={bridge} openNative={openNative} />
        </section>
      </main>
    </div>
  );
}
createRoot(document.getElementById("root")).render(<App />);
