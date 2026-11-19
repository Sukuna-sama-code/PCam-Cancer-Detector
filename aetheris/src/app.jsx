import React, { useState, useRef, useEffect } from "react";
import { createRoot } from "react-dom/client";
import {
  Orbit,
  Layers,
  Route,
  Compass,
  UserRound,
  Dna,
  Activity,
  Files,
  ShieldCheck,
  Check,
  FileText,
  UploadCloud,
  X,
  Download,
  SlidersHorizontal,
  MapPin,
  ScanLine,
  BookOpen,
  ChevronDown,
  Microscope,
  CircleHelp,
  FlaskConical,
  ClipboardList,
  Leaf,
  LockKeyhole,
} from "lucide-react";
import { Membrane } from "./art";
import { sample, pathways, providers } from "./data";
const tabs = [
  ["Intake", Layers],
  ["Diagnostic pathways", Route],
  ["Care navigator", Compass],
];
const steps = [
  ["Demographics", "Age and background", UserRound],
  ["Family history", "Inherited risk factors", Dna],
  ["Symptoms", "Changes you have noticed", Activity],
  ["Lab reports", "Supporting documents", Files],
];
const titles = [
  "Your health profile",
  "Family history",
  "Localized symptoms",
  "Lab reports",
];
const subtitles = [
  "Start with the details that shape your screening discussion.",
  "Include patterns you know about in close relatives.",
  "Choose the change you most want to discuss.",
  "Attach a report to list in your doctor prep brief.",
];
const embedded =
  window.parent !== window &&
  new URLSearchParams(location.search).get("streamlitUrl") !== null;
function emit(type, data) {
  window.parent.postMessage(
    { isStreamlitMessage: true, type: "streamlit:" + type, ...data },
    "*",
  );
}
function field(label, value, update, choices, opts = {}) {
  const id = label.replaceAll(" ", "-");
  return (
    <label className="field" htmlFor={id}>
      {label}
      {choices ? (
        <span className="selectwrap">
          <select
            id={id}
            value={value}
            onChange={(e) => update(e.target.value)}
          >
            {choices.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
          <ChevronDown size={15} />
        </span>
      ) : (
        <input
          id={id}
          type="number"
          required
          min={0}
          max={opts.max || 120}
          value={value}
          onChange={(e) => update(e.target.value)}
          {...opts}
        />
      )}
    </label>
  );
}
function tilt(e) {
  if (
    e.pointerType !== "mouse" ||
    matchMedia("(prefers-reduced-motion: reduce)").matches
  )
    return;
  const t = e.currentTarget,
    r = t.getBoundingClientRect();
  t.style.transform =
    "perspective(1000px) rotateX(" +
    (0.5 - (e.clientY - r.top) / r.height) * 2.7 +
    "deg) rotateY(" +
    ((e.clientX - r.left) / r.width - 0.5) * 3.8 +
    "deg)";
}
function App() {
  const [tab, T] = useState(0),
    [step, S] = useState(0),
    [done, D] = useState(0),
    [p, P] = useState({ ...sample }),
    [dialog, G] = useState(""),
    [city, C] = useState("All cities"),
    [service, V] = useState("All specialties"),
    [chosen, H] = useState(null),
    [error, E] = useState(""),
    [over, O] = useState(false),
    [bridge, B] = useState(embedded);
  const modal = useRef(null),
    file = useRef(null),
    shell = useRef(null);
  const paths = pathways(p),
    update = (k, v) => {
      P((old) => ({ ...old, [k]: v }));
      D((n) => Math.min(n, step));
    };
  useEffect(() => {
    if (window.parent === window) return;
    function msg(e) {
      if (e.source === window.parent && e.data?.type === "streamlit:render")
        B(true);
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
    if (dialog) modal.current?.showModal();
    else modal.current?.close();
  }, [dialog]);
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
  function attach(f) {
    if (!f) return;
    if (!/\.(pdf|png|jpe?g)$/i.test(f.name) || f.size > 10 * 1024 * 1024) {
      E("Choose a PDF, JPG or PNG under 10 MB.");
      return;
    }
    E("");
    update("report", f.name);
  }
  function advance(e) {
    e.preventDefault();
    D(Math.max(done, step + 1));
    if (step === 3) T(1);
    else S(step + 1);
  }
  function changeTab(i) {
    T(i);
    E("");
  }
  const brief = [
    "AETHERIS SPATIAL DIAGNOSTICS",
    "Doctor preparation brief · discussion aid",
    "",
    "Age: " + p.age + " | Sex at birth: " + p.sex,
    "Smoking: " +
      p.smoking +
      (p.smoking === "Never" ? "" : " | Pack-years: " + p.packs) +
      (p.smoking === "Former" ? " | Years since quitting: " + p.quit : ""),
    "Family history: " + p.family,
    "Current concern: " + p.symptom,
    "Report listed (contents not analyzed): " + (p.report || "None"),
    "Intake reviewed: " + done + "/4 stages",
    "Care destination: " +
      (chosen
        ? chosen.name + (chosen.certificate ? "" : " (fictional example)")
        : "Undecided"),
    "",
    "Topics for consultation:",
    ...paths
      .filter((x) => x.score > 0)
      .map((x) => "• " + x.name + " — " + x.tag),
    "• Which screening or assessment is appropriate for me?",
    "• What are the benefits, limitations and next steps?",
    "",
    "Educational workspace. A licensed clinician must decide diagnosis and care.",
  ].join("\n");
  function download() {
    const u = URL.createObjectURL(
      new Blob([brief], { type: "text/plain;charset=utf-8" }),
    );
    const a = document.createElement("a");
    a.href = u;
    a.download = "Aetheris-doctor-prep-brief.txt";
    a.click();
    setTimeout(() => URL.revokeObjectURL(u), 2000);
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
              Aetheris<small>SPATIAL DIAGNOSTICS</small>
            </div>
          </div>
          <div className="header-actions shrink-0">
            {bridge && (
              <button
                className="subbtn"
                onClick={() =>
                  emit("setComponentValue", {
                    value: "detector",
                    dataType: "json",
                  })
                }
              >
                <Microscope size={16} />
                PCam detector
              </button>
            )}
            <span className="demo">
              <i /> Demo workspace
            </span>
            <button
              className="help"
              aria-label="About this workspace"
              onClick={() => G("about")}
            >
              <CircleHelp size={19} />
            </button>
          </div>
        </div>
      </header>
      <main>
        <section className="hero">
          <div className="hero-copy relative">
            <span className="overline">
              <span className="shortline" /> CLINICAL WORKSPACE
            </span>
            <h1>
              Spatial <em>Diagnostics</em>
            </h1>
            <p>
              From your health history to a more informed conversation with your
              clinician.
            </p>
          </div>
          <Membrane />
        </section>
        <nav aria-label="Workspace views">
          <div className="tabs" role="tablist">
            <span className="slider" style={{ "--active": tab }} />
            {tabs.map(([label, I], i) => (
              <button
                key={label}
                id={"tab" + i}
                role="tab"
                aria-controls="workspace"
                aria-selected={i === tab}
                tabIndex={i === tab ? 0 : -1}
                onKeyDown={(e) => {
                  const n = {
                    ArrowRight: (i + 1) % 3,
                    ArrowLeft: (i + 2) % 3,
                    Home: 0,
                    End: 2,
                  }[e.key];
                  if (n !== undefined) {
                    e.preventDefault();
                    changeTab(n);
                    document.getElementById("tab" + n).focus();
                  }
                }}
                onClick={() => changeTab(i)}
              >
                <I size={17} />
                {label}
              </button>
            ))}
          </div>
        </nav>
        <section
          id="workspace"
          role="tabpanel"
          aria-labelledby={"tab" + tab}
          className="view"
          key={tab}
        >
          {tab === 0 ? (
            <>
              <div className="sectionhead">
                <div>
                  <span className="kicker">
                    <ClipboardList size={14} /> PATIENT INTAKE
                  </span>
                  <h2>Build your health profile</h2>
                </div>
                <span className="pill">
                  <i /> {done === 4 ? "Intake complete" : "In progress"}
                </span>
              </div>
              <div className="intakegrid">
                <aside className="panel journey">
                  <div className="asidehead">
                    <h3>Intake progress</h3>
                    <span>{done} / 4</span>
                  </div>
                  <div className="segments">
                    {steps.map((_, i) => (
                      <i key={i} className={i < done ? "filled" : ""} />
                    ))}
                  </div>
                  <ol>
                    {steps.map(([name, detail, I], i) => (
                      <li key={name}>
                        <button
                          disabled={i > done}
                          className={
                            "stage " +
                            (step === i ? "selected " : "") +
                            (i < done ? "completed" : "")
                          }
                          aria-current={step === i ? "step" : undefined}
                          onClick={() => {
                            S(i);
                            E("");
                          }}
                        >
                          <span className="stepicon">
                            {i < done ? <Check size={17} /> : <I size={17} />}
                          </span>
                          <span>
                            <strong>{name}</strong>
                            <small>{detail}</small>
                          </span>
                        </button>
                      </li>
                    ))}
                  </ol>
                </aside>
                <div className="stack" style={{ "--depth": Math.min(done, 2) }}>
                  <form
                    key={step}
                    onSubmit={advance}
                    className="panel formcard"
                  >
                    <div className="cardtop flex items-center">
                      <span className="iconplate">
                        {React.createElement(steps[step][2], { size: 23 })}
                      </span>
                      <span className="micro">STEP 0{step + 1} / 04</span>
                    </div>
                    <h3>{titles[step]}</h3>
                    <p>{subtitles[step]}</p>
                    <div className="fields">
                      {step === 0 && (
                        <>
                          {field(
                            "Age (years)",
                            p.age,
                            (v) => update("age", v),
                            null,
                            { min: 18, max: 110 },
                          )}
                          {field(
                            "Sex assigned at birth",
                            p.sex,
                            (v) => update("sex", v),
                            ["Female", "Male", "Intersex", "Prefer not to say"],
                          )}
                          {field(
                            "Smoking history",
                            p.smoking,
                            (v) => update("smoking", v),
                            ["Never", "Former", "Current"],
                          )}
                          {p.smoking !== "Never" &&
                            field(
                              "Pack-years",
                              p.packs,
                              (v) => update("packs", v),
                              null,
                              { step: "any", max: 300 },
                            )}
                          {p.smoking === "Former" &&
                            field(
                              "Years since quitting",
                              p.quit,
                              (v) => update("quit", v),
                              null,
                              { max: Number(p.age) || 110 },
                            )}
                        </>
                      )}
                      {step === 1 &&
                        field(
                          "Close relative’s cancer history",
                          p.family,
                          (v) => update("family", v),
                          [
                            "Unsure",
                            "None known",
                            "Breast or ovarian cancer",
                            "Other cancer",
                          ],
                        )}
                      {step === 2 &&
                        field(
                          "Current concern",
                          p.symptom,
                          (v) => update("symptom", v),
                          [
                            "None reported",
                            "Changing skin lesion",
                            "Persistent cough",
                            "Breast change",
                            "Other persistent symptom",
                          ],
                        )}
                      {step === 3 && (
                        <div
                          className={"drop " + (over ? "over" : "")}
                          onDragOver={(e) => {
                            e.preventDefault();
                            O(true);
                          }}
                          onDragLeave={() => O(false)}
                          onDrop={(e) => {
                            e.preventDefault();
                            O(false);
                            attach(e.dataTransfer.files[0]);
                          }}
                        >
                          <UploadCloud />
                          <strong>{p.report || "Drop a report here"}</strong>
                          <small>
                            PDF, JPG or PNG · Up to 10 MB. Only the filename is
                            used.
                          </small>
                          <input
                            hidden
                            type="file"
                            accept=".pdf,.jpg,.jpeg,.png"
                            ref={file}
                            onChange={(e) => {
                              attach(e.target.files[0]);
                              e.target.value = "";
                            }}
                          />
                          <button
                            type="button"
                            className="subbtn"
                            onClick={() => file.current.click()}
                          >
                            Choose file
                          </button>
                          {p.report && (
                            <button
                              type="button"
                              className="textbtn"
                              onClick={() => update("report", "")}
                            >
                              Remove
                            </button>
                          )}
                          {error && <span role="alert">{error}</span>}
                        </div>
                      )}
                    </div>
                    <div className="formfoot">
                      {step > 0 ? (
                        <button
                          className="textbtn"
                          type="button"
                          onClick={() => S(step - 1)}
                        >
                          Back
                        </button>
                      ) : (
                        <small>
                          <LockKeyhole size={14} /> Session only
                        </small>
                      )}
                      <button className="primary">
                        {step === 3 ? "View pathways" : "Continue"}
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            </>
          ) : tab === 1 ? (
            <>
              <div className="sectionhead">
                <div>
                  <span className="kicker">
                    <Route size={14} /> GUIDELINE REVIEW
                  </span>
                  <h2>Diagnostic pathways</h2>
                </div>
                <button className="subbtn" onClick={() => changeTab(0)}>
                  Review intake
                </button>
              </div>
              <p className="intro">
                Discussion topics for the current profile. Meters count matching
                criteria. They do not estimate cancer risk.
              </p>
              <div className="deck">
                {paths.map((x, i) => (
                  <article
                    className="panel path"
                    key={x.name}
                    onPointerMove={tilt}
                    onPointerLeave={(e) =>
                      (e.currentTarget.style.transform = "")
                    }
                  >
                    <div className="cardtop flex items-center">
                      <span className="iconplate">
                        {React.createElement(
                          x.name === "Liquid biopsy"
                            ? FlaskConical
                            : x.name === "Dermoscopy"
                              ? Microscope
                              : ScanLine,
                          { size: 22 },
                        )}
                      </span>
                      <span className="pill">{x.tag}</span>
                    </div>
                    <span className="kicker">{x.type}</span>
                    <h3>{x.name}</h3>
                    <p>{x.desc}</p>
                    {x.meter !== null && (
                      <div className="meter">
                        <small>
                          Guideline criteria matched{" "}
                          <b>
                            {x.meter}/{x.total}
                          </b>
                        </small>
                        <progress value={x.meter} max={x.total} />
                      </div>
                    )}
                    <details>
                      <summary>
                        <BookOpen size={15} />
                        Clinical explainability
                      </summary>
                      <p>{x.why}</p>
                      <a href={x.url} target="_blank" rel="noreferrer">
                        {x.source}
                      </a>
                    </details>
                    <details>
                      <summary>
                        <ClipboardList size={15} />
                        Preparation guide
                      </summary>
                      <p>{x.prep}</p>
                    </details>
                  </article>
                ))}
              </div>
            </>
          ) : (
            <>
              <div className="sectionhead">
                <div>
                  <span className="kicker">
                    <Compass size={14} /> SPECIALIST DIRECTORY
                  </span>
                  <h2>Care navigator</h2>
                </div>
                <button className="primary" onClick={() => G("brief")}>
                  <FileText size={16} />
                  Doctor prep brief
                </button>
              </div>
              <p className="intro">
                Explore a documented referral option and example centers.
                Profiles matching your discussion topics appear first.
              </p>
              <div className="filters panel">
                <SlidersHorizontal size={20} />
                {field("Location", city, C, [
                  "All cities",
                  "Bengaluru",
                  "Mysuru",
                ])}
                {field("Specialty", service, V, [
                  "All specialties",
                  ...new Set(providers.map((x) => x.specialty)),
                ])}
              </div>
              <div className="deck">
                {[...providers]
                  .sort(
                    (a, b) =>
                      (paths.find((x) => x.name === b.service)?.score || 0) -
                      (paths.find((x) => x.name === a.service)?.score || 0),
                  )
                  .filter(
                    (x) =>
                      (city === "All cities" || x.city === city) &&
                      (service === "All specialties" ||
                        service === x.specialty),
                  )
                  .map((x) => (
                    <article key={x.name} className="panel provider">
                      <div className={"avatar " + x.tone}>{x.initials}</div>
                      <span className="kicker">
                        {x.specialty} ·{" "}
                        {x.certificate ? "HOSPITAL DIRECTORY" : "DEMO"}
                      </span>
                      <h3>{x.name}</h3>
                      <p>
                        <MapPin size={15} />
                        {x.city} · {x.service}
                      </p>
                      {x.certificate && (
                        <p>
                          <a
                            href={x.certificate}
                            target="_blank"
                            rel="noreferrer"
                          >
                            NABH certificate
                          </a>{" "}
                          ·{" "}
                          <a
                            href={x.serviceUrl}
                            target="_blank"
                            rel="noreferrer"
                          >
                            MRI service
                          </a>
                        </p>
                      )}
                      <button
                        className="subbtn"
                        onClick={() => {
                          H(x);
                          G("brief");
                        }}
                      >
                        Prepare for this visit
                      </button>
                    </article>
                  ))}
              </div>
              {!providers.some(
                (x) =>
                  (city === "All cities" || x.city === city) &&
                  (service === "All specialties" || service === x.specialty),
              ) && (
                <div className="panel empty">
                  No example centers match both filters.{" "}
                  <button
                    className="textbtn"
                    onClick={() => {
                      C("All cities");
                      V("All specialties");
                    }}
                  >
                    Clear filters
                  </button>
                </div>
              )}
            </>
          )}
        </section>
      </main>
      <footer>
        <div>
          <ShieldCheck size={19} />
          <span>
            Clinical decision support only. Discuss screening and diagnosis with
            a licensed physician.
          </span>
        </div>
        <button className="textbtn" onClick={() => G("about")}>
          About Aetheris
        </button>
      </footer>
      <dialog
        ref={modal}
        onCancel={() => G("")}
        onClose={() => G("")}
        aria-labelledby="modal-title"
      >
        <div className="modalhead">
          <span className="iconplate">
            {dialog === "brief" ? <FileText /> : <Leaf />}
          </span>
          <button
            autoFocus
            className="help"
            aria-label="Close dialog"
            onClick={() => G("")}
          >
            <X size={20} />
          </button>
        </div>
        <h2 id="modal-title">
          {dialog === "brief" ? "Doctor prep brief" : "About this workspace"}
        </h2>
        {dialog === "brief" ? (
          <>
            <pre>{brief}</pre>
            <button className="primary" onClick={download}>
              <Download size={16} />
              Export doctor prep brief
            </button>
          </>
        ) : (
          <>
            <p>
              Aetheris Spatial Diagnostics uses an editable fictional adult
              profile and published US and UK guidance to prepare discussion
              topics. Local recommendations may differ. Report contents are
              never processed.
            </p>
            <p>
              Changes stay in this browser session. Reloading restores the
              example.
            </p>
            {bridge && (
              <button
                className="primary"
                onClick={() =>
                  emit("setComponentValue", {
                    value: "detector",
                    dataType: "json",
                  })
                }
              >
                <Microscope size={16} />
                Open PCam research detector
              </button>
            )}
          </>
        )}
      </dialog>
    </div>
  );
}
createRoot(document.getElementById("root")).render(<App />);
