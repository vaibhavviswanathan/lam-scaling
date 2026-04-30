// LAM latent action tour — frontend.

const $ = (id) => document.getElementById(id);

// ---- state ----

const state = {
  tag: null,
  cache: null,         // {tag, branch, n, transitions, motion_order, unique_codes}
  currentIdx: 0,       // currently displayed transition (driven by tour slider)
  anchorA: null,
  anchorB: null,
  alpha: 0.5,
  playing: false,
  abortInterp: null,   // AbortController for the latest interpolate request
  projection: "pca",   // "pca" | "tsne" | "umap"
};

const PROJ_LABELS = {
  pca: { x: "PC1", y: "PC2" },
  tsne: { x: "t-SNE 1", y: "t-SNE 2" },
  umap: { x: "UMAP 1", y: "UMAP 2" },
};

function projCoord(t) {
  return t[state.projection] || t.pca || [0, 0];
}

// ---- API ----

async function api(url, opts = {}) {
  const res = await fetch(url, { ...opts, headers: { "Content-Type": "application/json", ...(opts.headers || {}) } });
  if (!res.ok) throw new Error(`${url}: ${res.status} ${await res.text()}`);
  return res.json();
}

async function fetchCheckpoints() {
  return api("/api/checkpoints");
}

async function fetchCache(tag) {
  return api(`/api/cache/${tag}`);
}

async function fetchInterpolation(tag, anchorA, anchorB, alpha, signal) {
  return api("/api/interpolate", {
    method: "POST",
    body: JSON.stringify({ tag, anchor_a: anchorA, anchor_b: anchorB, alpha }),
    signal,
  });
}

const framePath = (idx, which) => `/frames/${String(idx).padStart(5, "0")}_${which}.jpg`;

// ---- rendering helpers ----

// Soft cross-fade when swapping frames.
function setImg(el, src) {
  if (!src) {
    el.removeAttribute("src");
    return;
  }
  if (el.src && el.src.endsWith(src)) return;
  el.classList.add("loading");
  const next = new Image();
  next.onload = () => {
    el.src = src;
    el.classList.remove("loading");
  };
  next.onerror = () => {
    el.removeAttribute("src");
    el.classList.remove("loading");
  };
  next.src = src;
}

function infoText(c, idx, percentile) {
  const t = c.transitions[idx];
  const parts = [
    `transition #${idx}`,
    `motion ${t.motion.toFixed(2)}`,
    `pct ${percentile.toFixed(0)}%`,
  ];
  if (typeof t.code_id === "number") parts.push(`code id ${t.code_id}`);
  return parts.join("  ·  ");
}

// ---- scatter ----

function renderScatter() {
  const c = state.cache;
  if (!c) return;

  const coords = c.transitions.map(projCoord);
  const x = coords.map((p) => p[0]);
  const y = coords.map((p) => p[1]);
  const motion = c.transitions.map((t) => t.motion);
  const ids = c.transitions.map((t) => t.id);

  const main = {
    x, y,
    mode: "markers",
    type: "scattergl",
    marker: {
      size: 7,
      color: motion,
      colorscale: "Viridis",
      colorbar: { title: "motion", thickness: 12, len: 0.85 },
      line: { width: 0 },
      opacity: 0.85,
    },
    text: ids.map((id, i) => `transition #${id}<br>motion ${motion[i].toFixed(2)}` +
      (typeof c.transitions[i].code_id === "number" ? `<br>code ${c.transitions[i].code_id}` : "")),
    hoverinfo: "text",
    name: "transitions",
    showlegend: false,
  };

  const traces = [main];
  // Highlight current selection
  if (state.currentIdx != null) {
    const cur = projCoord(c.transitions[state.currentIdx]);
    traces.push({
      x: [cur[0]], y: [cur[1]],
      mode: "markers",
      type: "scatter",
      marker: { size: 14, color: "rgba(13,148,136,0.95)", line: { color: "#fff", width: 2 } },
      hoverinfo: "skip",
      showlegend: false,
    });
  }
  // Highlight A and B
  for (const [idx, color, label] of [
    [state.anchorA, "#dc2626", "A"],
    [state.anchorB, "#2563eb", "B"],
  ]) {
    if (idx == null) continue;
    const p = projCoord(c.transitions[idx]);
    traces.push({
      x: [p[0]], y: [p[1]],
      mode: "markers+text",
      type: "scatter",
      marker: { size: 16, color, line: { color: "#fff", width: 2 } },
      text: [label],
      textposition: "middle center",
      textfont: { color: "#fff", size: 11, family: "ui-sans-serif" },
      hoverinfo: "skip",
      showlegend: false,
    });
  }

  const title = c.label + (c.unique_codes != null ? `  ·  unique codes here: ${c.unique_codes}/64` : "");

  const labels = PROJ_LABELS[state.projection] || PROJ_LABELS.pca;
  const layout = {
    title: { text: title, font: { size: 14, color: "#374151" }, x: 0.02 },
    margin: { l: 40, r: 30, t: 40, b: 40 },
    xaxis: { title: labels.x, zeroline: false, gridcolor: "#e5e7eb", color: "#6b7280" },
    yaxis: { title: labels.y, zeroline: false, gridcolor: "#e5e7eb", color: "#6b7280" },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#fafafa",
    font: { family: "ui-sans-serif, system-ui, sans-serif", size: 12, color: "#374151" },
    hovermode: "closest",
    dragmode: "pan",
  };
  Plotly.react("scatter", traces, layout, { displayModeBar: false, responsive: true });
}

// Click on scatter -> jump tour slider
function attachScatterClick() {
  const el = $("scatter");
  el.on("plotly_click", (ev) => {
    if (!ev || !ev.points || !ev.points.length) return;
    const p = ev.points[0];
    // Only react to clicks on the main trace (curveNumber 0).
    if (p.curveNumber !== 0) return;
    const idx = p.pointNumber;
    setCurrentIdx(idx);
  });
}

// ---- tour controls ----

function pctForIdx(c, idx) {
  // motion_order: ranks are 0..N-1; find idx's rank
  const order = c.motion_order;
  for (let i = 0; i < order.length; i++) {
    if (order[i] === idx) return (i / (order.length - 1)) * 100;
  }
  return 0;
}

function idxForPct(c, p) {
  const rank = Math.round((p / 100) * (c.motion_order.length - 1));
  const r = Math.max(0, Math.min(c.motion_order.length - 1, rank));
  return c.motion_order[r];
}

function setCurrentIdx(idx, opts = {}) {
  const c = state.cache;
  if (!c) return;
  state.currentIdx = idx;
  setImg($("frame-t"), framePath(idx, "t"));
  setImg($("frame-tp1"), framePath(idx, "tp1"));
  const p = pctForIdx(c, idx);
  $("info-md").textContent = infoText(c, idx, p);
  if (!opts.fromSlider) {
    $("pct-slider").value = p.toFixed(1);
  }
  renderScatter();
}

// ---- interpolation controls ----

function updateAnchorImg(slot) {
  const idx = slot === "A" ? state.anchorA : state.anchorB;
  const img = slot === "A" ? $("anchor-a-img") : $("anchor-b-img");
  const idLabel = slot === "A" ? $("anchor-a-id") : $("anchor-b-id");
  if (idx == null) {
    img.removeAttribute("src");
    idLabel.textContent = "—";
  } else {
    setImg(img, framePath(idx, "tp1"));
    idLabel.textContent = `#${idx}`;
  }
}

function bothAnchorsSet() {
  return state.anchorA != null && state.anchorB != null;
}

function updateInterpEnable() {
  const ok = bothAnchorsSet();
  $("alpha-slider").disabled = !ok;
  $("play-btn").disabled = !ok || state.playing;
  $("swap-btn").disabled = !ok;
  if (!ok) {
    $("interp-info").textContent = "pin both A and B to enable interpolation";
    $("interp-img").removeAttribute("src");
  }
}

async function runInterpolation(alpha) {
  if (!bothAnchorsSet()) return;
  if (state.abortInterp) state.abortInterp.abort();
  const ctrl = new AbortController();
  state.abortInterp = ctrl;
  try {
    const r = await fetchInterpolation(
      state.tag, state.anchorA, state.anchorB, alpha, ctrl.signal,
    );
    if (ctrl.signal.aborted) return;
    setImg($("interp-img"), framePath(r.nearest_idx, "tp1"));
    $("interp-info").textContent =
      `α = ${alpha.toFixed(2)}  ·  retrieved #${r.nearest_idx}  ·  ` +
      `d(pred, nearest f_t+1) = ${r.d_pred_to_nearest_tp1.toFixed(2)}  ·  ` +
      `d(pred, nearest f_t) = ${r.d_pred_to_nearest_t.toFixed(2)}`;
  } catch (e) {
    if (e.name !== "AbortError") {
      $("interp-info").textContent = `error: ${e.message}`;
    }
  }
}

// ---- play loop ----

async function playAtoB() {
  if (!bothAnchorsSet() || state.playing) return;
  state.playing = true;
  const btn = $("play-btn");
  btn.classList.add("playing");
  btn.textContent = "■ stop";
  $("alpha-slider").disabled = true;
  let stopped = false;
  const stopHandler = () => { stopped = true; };
  btn.addEventListener("click", stopHandler, { once: true });

  try {
    // 1) hold A's real frame
    setImg($("interp-img"), framePath(state.anchorA, "tp1"));
    $("interp-info").textContent = `▶ start: A (transition #${state.anchorA}, real frame)`;
    $("alpha-slider").value = "0";
    await sleep(550);
    if (stopped) return;

    // 2) sweep alpha
    const N = 24;
    for (let i = 0; i <= N; i++) {
      if (stopped) return;
      const alpha = i / N;
      $("alpha-slider").value = String(alpha);
      // We deliberately await each step so the displayed frame matches the slider.
      try {
        const r = await fetchInterpolation(state.tag, state.anchorA, state.anchorB, alpha);
        if (stopped) return;
        setImg($("interp-img"), framePath(r.nearest_idx, "tp1"));
        $("interp-info").textContent =
          `▶ playing  ·  α = ${alpha.toFixed(2)}  ·  retrieved #${r.nearest_idx}`;
      } catch (e) {
        $("interp-info").textContent = `error: ${e.message}`;
        break;
      }
      await sleep(95);
    }
    if (stopped) return;

    // 3) hold B's real frame
    setImg($("interp-img"), framePath(state.anchorB, "tp1"));
    $("interp-info").textContent = `▶ end: B (transition #${state.anchorB}, real frame)`;
    $("alpha-slider").value = "1";
    await sleep(550);
  } finally {
    btn.removeEventListener("click", stopHandler);
    btn.classList.remove("playing");
    btn.textContent = "▶ play A → B";
    state.playing = false;
    updateInterpEnable();
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- wiring ----

async function loadTag(tag) {
  state.tag = tag;
  $("proj-status").textContent = "loading projections (first switch may take a few seconds for t-SNE + UMAP)…";
  try {
    state.cache = await fetchCache(tag);
    $("proj-status").textContent = "";
  } catch (e) {
    $("proj-status").textContent = `error loading: ${e.message}`;
    throw e;
  }
  // reset anchors when branch changes
  state.anchorA = null;
  state.anchorB = null;
  updateAnchorImg("A");
  updateAnchorImg("B");
  $("interp-img").removeAttribute("src");
  $("alpha-slider").value = "0.5";
  state.alpha = 0.5;
  // re-render with current slider position
  const idx = idxForPct(state.cache, parseFloat($("pct-slider").value));
  setCurrentIdx(idx);
  updateInterpEnable();
}

async function init() {
  const cps = await fetchCheckpoints();
  const sel = $("branch-select");
  for (const c of cps) {
    const opt = document.createElement("option");
    opt.value = c.tag;
    opt.textContent = c.label;
    sel.appendChild(opt);
  }
  // default to gaussian_h100
  const initial = cps.find((c) => c.tag === "gaussian_h100") || cps[0];
  sel.value = initial.tag;

  await loadTag(initial.tag);
  attachScatterClick();

  // tour slider
  $("pct-slider").addEventListener("input", () => {
    const p = parseFloat($("pct-slider").value);
    if (!state.cache) return;
    const idx = idxForPct(state.cache, p);
    setCurrentIdx(idx, { fromSlider: true });
  });

  // branch dropdown
  sel.addEventListener("change", async (ev) => {
    await loadTag(ev.target.value);
  });

  // pin / pair / swap
  $("pin-a-btn").addEventListener("click", () => {
    state.anchorA = state.currentIdx;
    updateAnchorImg("A");
    renderScatter();
    updateInterpEnable();
    if (bothAnchorsSet()) runInterpolation(parseFloat($("alpha-slider").value));
  });
  $("pin-b-btn").addEventListener("click", () => {
    state.anchorB = state.currentIdx;
    updateAnchorImg("B");
    renderScatter();
    updateInterpEnable();
    if (bothAnchorsSet()) runInterpolation(parseFloat($("alpha-slider").value));
  });
  $("random-pair-btn").addEventListener("click", () => {
    if (!state.cache) return;
    const order = state.cache.motion_order;
    state.anchorA = order[Math.floor(0.05 * order.length)];
    state.anchorB = order[Math.floor(0.95 * order.length)];
    updateAnchorImg("A");
    updateAnchorImg("B");
    renderScatter();
    updateInterpEnable();
    runInterpolation(parseFloat($("alpha-slider").value));
  });
  $("swap-btn").addEventListener("click", () => {
    [state.anchorA, state.anchorB] = [state.anchorB, state.anchorA];
    updateAnchorImg("A");
    updateAnchorImg("B");
    renderScatter();
    if (bothAnchorsSet()) runInterpolation(parseFloat($("alpha-slider").value));
  });

  // alpha slider
  $("alpha-slider").addEventListener("input", () => {
    if (!bothAnchorsSet()) return;
    if (state.playing) return;  // play handles updates itself
    const a = parseFloat($("alpha-slider").value);
    state.alpha = a;
    runInterpolation(a);
  });

  // play
  $("play-btn").addEventListener("click", () => {
    if (state.playing) return;  // the in-play "■ stop" overrides via stopHandler
    playAtoB();
  });

  // projection toggle
  document.querySelectorAll(".toggle[data-proj]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const proj = btn.dataset.proj;
      if (state.projection === proj) return;
      state.projection = proj;
      document.querySelectorAll(".toggle[data-proj]").forEach((b) => {
        const active = b.dataset.proj === proj;
        b.classList.toggle("active", active);
        b.setAttribute("aria-selected", active ? "true" : "false");
      });
      renderScatter();
    });
  });
}

window.addEventListener("DOMContentLoaded", init);
