/* Sentiment Studio front end - vanilla JS, no build step.
   Talks to the FastAPI backend in 03_web_app.py (all endpoints under /api). */
"use strict";

// ── Helpers ───────────────────────────────────────────────────────────
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 0) => `${(100 * v).toFixed(d)}%`;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
// backtick-quoted words in server explanations -> <code>
const richText = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>");

const ICON_PATHS = {
  copy: '<path d="M4 6h16M4 12h10M4 18h13"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-9 9"/>',
  table: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M9 10v10"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
  sparkles: '<path d="M12 3l1.8 4.7 4.7 1.8-4.7 1.8L12 16l-1.8-4.7-4.7-1.8 4.7-1.8z"/><path d="M19 15l.8 2.2 2.2.8-2.2.8L19 21l-.8-2.2-2.2-.8 2.2-.8z"/>',
  target: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
  pen: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
  upload: '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  alert: '<path d="M12 7v6M12 17h.01"/>',
  download: '<path d="M12 4v12M7 11l5 5 5-5M4 20h16"/>',
  clipboard: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  line: '<path d="M5 19L19 5"/>',
  rect: '<rect x="4" y="6" width="16" height="12" rx="1"/>',
  circle: '<circle cx="12" cy="12" r="8"/>',
  text: '<path d="M5 6h14M12 6v13"/>',
  crop: '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/>',
  undo: '<path d="M9 14L4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-3"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/>',
};
const icon = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICON_PATHS[name] || ""}</svg>`;

const TIER_TEXT = { high: "Good", medium: "Fair", low: "Needs work" };
const TIER_COLOR = { high: "var(--good-bar)", medium: "var(--fair-bar)", low: "var(--bad-bar)" };
const SENT_COLOR = { positive: "var(--pos)", neutral: "var(--neu)", negative: "var(--neg)" };
const tierOf = (score, hi = 70, mid = 45) => (score >= hi ? "high" : score >= mid ? "medium" : "low");

class ApiError extends Error {
  constructor(title, hint) { super(title); this.title = title; this.hint = hint || ""; }
}

async function api(path, { json, form, method } = {}) {
  const opts = { method: method || (json || form ? "POST" : "GET") };
  if (json) { opts.headers = { "Content-Type": "application/json" }; opts.body = JSON.stringify(json); }
  if (form) opts.body = form;
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new ApiError("Can't reach the server.", "Is `python 03_web_app.py` still running?");
  }
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok) {
    if (data && data.error) throw new ApiError(data.error, data.hint || data.detail);
    if (data && data.detail) throw new ApiError("The request was rejected.", JSON.stringify(data.detail));
    throw new ApiError(`Request failed (${res.status}).`, "");
  }
  return data;
}

function formOf(fields) {
  const fd = new FormData();
  for (const [k, v] of Object.entries(fields)) if (v !== undefined && v !== null) fd.append(k, v);
  return fd;
}

function toast(title, msg = "", kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `<b>${esc(title)}</b>${msg ? richText(msg) : ""}`;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), kind === "error" ? 7000 : 3500);
}

function errorNote(err) {
  const title = err instanceof ApiError ? err.title : "Something went wrong.";
  const hint = err instanceof ApiError ? err.hint : String(err);
  return `<div class="note note-error fade-in">${icon("alert")}<div><b>${esc(title)}</b>${hint ? `<div>${richText(hint)}</div>` : ""}</div></div>`;
}

async function withBusy(btn, fn) {
  btn.classList.add("loading");
  btn.disabled = true;
  try { return await fn(); } finally { btn.classList.remove("loading"); btn.disabled = false; }
}

function download(href, filename) {
  const a = document.createElement("a");
  a.href = href; a.download = filename;
  document.body.append(a); a.click(); a.remove();
}

function downloadText(text, filename, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  download(url, filename);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function dataUrlToFile(dataUrl, name) {
  const blob = await (await fetch(dataUrl)).blob();
  return new File([blob], name, { type: blob.type });
}

// ── Components ────────────────────────────────────────────────────────
function ring(score, { tier, caption = "/ 100", small = false } = {}) {
  const t = tier || tierOf(score);
  const r = small ? 28 : 58, size = small ? 64 : 132, c = 2 * Math.PI * r;
  const off = c * (1 - clamp(score, 0, 100) / 100);
  return `<div class="ring ${small ? "small" : ""}">
    <svg viewBox="0 0 ${size} ${size}"><circle class="track" cx="${size / 2}" cy="${size / 2}" r="${r}"/>
      <circle class="value" cx="${size / 2}" cy="${size / 2}" r="${r}" stroke="${TIER_COLOR[t]}"
        stroke-dasharray="${c}" stroke-dashoffset="${c}" data-offset="${off}"/></svg>
    <div class="ring-label"><b>${Math.round(score)}</b>${small ? "" : `<span>${esc(caption)}</span>`}</div></div>`;
}

// animate rings/bars after they're in the DOM
function animate(root) {
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $$(".ring .value", root).forEach((el) => el.setAttribute("stroke-dashoffset", el.dataset.offset));
    $$(".bar > i[data-w]", root).forEach((el) => { el.style.width = el.dataset.w; });
  }));
}

const bar = (value, cls) => `<div class="bar bar-${cls}"><i style="width:0" data-w="${clamp(value, 0, 100)}%"></i></div>`;
const badge = (text, cls) => `<span class="badge badge-${cls}">${esc(text)}</span>`;
const heuristicTag = `<span class="tag tag-heuristic" title="Rule-based score, not a trained/validated model">Heuristic</span>`;

function dropzone({ accept, title, sub, compact = false, onFile }) {
  const wrap = document.createElement("div");
  wrap.innerHTML = `<label class="dropzone ${compact ? "compact" : ""}" tabindex="0">
      ${icon("upload")}<div><strong>${esc(title)}</strong><span>${esc(sub)}</span></div>
      <input type="file" accept="${accept}" hidden></label>`;
  const zone = wrap.firstElementChild, input = $("input", zone);
  const take = (file) => { if (file) onFile(file); input.value = ""; };
  input.addEventListener("change", () => take(input.files[0]));
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", (e) => { e.preventDefault(); zone.classList.remove("drag"); take(e.dataTransfer.files[0]); });
  return zone;
}

const isImage = (f) => /^image\/(png|jpe?g|webp|gif|bmp)$/i.test(f.type) || /\.(png|jpe?g|webp)$/i.test(f.name);

function compareView(beforeUrl, beforeScore, afterUrl, afterScore, afterLabel = "After") {
  const d = afterScore - beforeScore;
  return `<div class="compare fade-in">
    <figure><div class="preview"><img src="${beforeUrl}" alt="Before"></div>
      <figcaption><span>Before</span>${badge(`${Math.round(beforeScore)} · ${TIER_TEXT[tierOf(beforeScore)]}`, tierOf(beforeScore))}</figcaption></figure>
    <figure><div class="preview"><img src="${afterUrl}" alt="${esc(afterLabel)}"></div>
      <figcaption><span>${esc(afterLabel)} <span class="delta ${d > 0.5 ? "up" : d < -0.5 ? "down" : ""}">${d >= 0 ? "+" : ""}${d.toFixed(0)}</span></span>
      ${badge(`${Math.round(afterScore)} · ${TIER_TEXT[tierOf(afterScore)]}`, tierOf(afterScore))}</figcaption></figure>
  </div>`;
}

function setupFeatureGate(container, module, featureName, install, where = "afterbegin") {
  const dep = (STATUS.data?.optional_deps || []).find((d) => d.module === module);
  if (dep && !dep.available) {
    container.insertAdjacentHTML(where, `<div class="note note-warn mt">${icon("alert")}<div><b>${esc(featureName)} isn't installed.</b>
      <div>Run <code>pip install ${esc(install)}</code> and restart the app to enable it.</div></div></div>`);
    return false;
  }
  return true;
}

// ── System status (sidebar) ──────────────────────────────────────────
const STATUS = { data: null };
async function refreshStatus() {
  try {
    const s = await api("/api/status");
    STATUS.data = s;
    const m = s.sentiment_model;
    const dot = $("#model-dot"), label = $("#model-status");
    if (!m.ready) {
      dot.className = "dot dot-loading";
      label.textContent = "Loading sentiment model…";
    } else if (m.using_fallback) {
      dot.className = "dot dot-warn";
      label.innerHTML = 'Public baseline model<br><span class="muted">Run <code>02_model_training.py</code> to use your fine-tuned model.</span>';
    } else {
      dot.className = "dot dot-ok";
      label.textContent = "Fine-tuned sentiment model";
    }
    $("#deps-list").innerHTML = s.optional_deps.map((d) =>
      `<li><span class="dot ${d.available ? "dot-ok" : "dot-off"}"></span>${esc(d.description)}</li>`).join("");
    if (!m.ready) setTimeout(refreshStatus, 3000);
  } catch {
    $("#model-dot").className = "dot dot-off";
    $("#model-status").textContent = "Server unreachable";
    setTimeout(refreshStatus, 5000);
  }
}

// ══════════════════════════════════════════════════════════════════════
// Pages
// ══════════════════════════════════════════════════════════════════════
const S = { copy: {}, image: {}, bulk: {}, website: {}, generate: {}, precise: {}, canvas: {} };

// ── Ad Copy ──────────────────────────────────────────────────────────
function pageCopy(view) {
  const st = S.copy;
  view.innerHTML = `
    <p class="lead">Check the tone of an ad or post, see a heuristic engagement score with the reasons behind it,
      and get a rewrite that is checked so it never invents prices, discounts or deadlines.</p>
    <div class="card">
      <label class="field" for="copy-text">Ad or post copy</label>
      <textarea id="copy-text" class="input" placeholder="e.g. Boost your productivity with our AI tool — try it free today!"></textarea>
      <div class="grid grid-2 mt" style="align-items:end">
        <div>
          <label class="field" for="copy-offer">Real offer details <span class="muted">(optional)</span></label>
          <input id="copy-offer" class="input" placeholder="e.g. 20% off until Sunday, free shipping over ₹999">
          <div class="hint">The rewrite may only mention prices, discounts, deadlines or guarantees found in your copy or here.</div>
        </div>
        <div class="row row-end"><span class="muted small">⌘/Ctrl + Enter</span>
          <button class="btn btn-primary" id="copy-run">${icon("arrow")}Analyze copy</button></div>
      </div>
    </div>
    <div id="copy-out"></div>`;
  const text = $("#copy-text"), offer = $("#copy-offer"), out = $("#copy-out"), run = $("#copy-run");
  text.value = st.text || ""; offer.value = st.offer || "";
  text.addEventListener("input", () => { st.text = text.value; });
  offer.addEventListener("input", () => { st.offer = offer.value; });

  const go = () => withBusy(run, async () => {
    if (!text.value.trim()) { text.focus(); toast("Enter some ad copy first."); return; }
    out.innerHTML = `<div class="grid grid-2 mt"><div class="skeleton" style="height:170px"></div><div class="skeleton" style="height:170px"></div></div>`;
    try {
      st.result = await api("/api/copy/analyze", { json: { text: text.value, offer_details: offer.value } });
      st.resultFor = text.value;
      renderCopyResult(out, st.result, text.value);
    } catch (e) { st.result = null; out.innerHTML = `<div class="mt">${errorNote(e)}</div>`; }
  });
  run.addEventListener("click", go);
  text.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) go(); });
  if (st.result) renderCopyResult(out, st.result, st.resultFor);
}

function renderCopyResult(out, r, original) {
  const s = r.sentiment, e = r.engagement, w = r.rewrite;
  const order = ["positive", "neutral", "negative"];
  const factorClass = (t) => (/missing|too short|too long|long and|no urgency|less persuasive/i.test(t) ? "warn" : "good");
  out.innerHTML = `
    <div class="grid grid-2 mt fade-in">
      <div class="card kpi">
        <div class="kpi-label">Sentiment</div>
        <div class="row">${badge(s.label.toUpperCase(), s.label)}<span class="kpi-sub">Confidence ${pct(s.confidence, 1)}</span></div>
        <div class="dist mt">${order.map((k) => `<div class="dist-row"><span>${k[0].toUpperCase() + k.slice(1)}</span>
          ${bar(100 * (s.scores[k] || 0), k)}<b>${pct(s.scores[k] || 0, 1)}</b></div>`).join("")}</div>
      </div>
      <div class="card">
        <div class="kpi-label">Engagement ${heuristicTag}</div>
        <div class="score-block mt">${ring(e.engagement_score, { tier: e.engagement_label })}
          <div>${badge(TIER_TEXT[e.engagement_label], e.engagement_label)}
            <p class="kpi-sub">Rule-based: call-to-action, urgency, length, tone and punctuation. Not a trained engagement model.</p></div></div>
      </div>
    </div>
    <div class="grid grid-2 mt fade-in">
      <div class="card"><div class="card-title">Why this sentiment</div><p class="kpi-sub" style="margin:0">${richText(r.explanation)}</p></div>
      <div class="card"><div class="card-title">Engagement factors</div>
        <ul class="list">${e.reasons.map((t) => `<li class="${factorClass(t)}">${esc(t)}</li>`).join("")}</ul></div>
    </div>
    <div class="card mt fade-in">
      <div class="row"><div class="card-title">Suggested rewrite</div><span class="spacer"></span>
        <button class="btn btn-secondary" id="copy-copy">${icon("clipboard")}Copy</button></div>
      <div class="quote mt">${esc(w.refined_text)}</div>
      <div class="metrics mt">
        <div class="metric"><span>Before</span><b>${w.before_score.toFixed(0)} / 100</b></div>
        <div class="metric"><span>After</span><b>${w.after_score.toFixed(0)} / 100</b></div>
        <div class="metric"><span>Change</span><b class="delta ${w.score_change > 0 ? "up" : w.score_change < 0 ? "down" : ""}">${w.score_change >= 0 ? "+" : ""}${w.score_change.toFixed(0)}</b></div>
        <div class="metric"><span>Method</span><b style="font-size:.85rem">${esc(w.method)}</b></div>
      </div>
      <div class="note ${w.grounded ? "note-good" : "note-warn"} mt">${icon(w.grounded ? "check" : "alert")}<div>${w.grounded
        ? "<b>Grounding check passed.</b> The rewrite adds no prices, discounts, deadlines or guarantees beyond your copy and offer details."
        : "<b>Review before using.</b> The rewrite contains claims not found in your input."}</div></div>
      ${w.rejected_candidates.length ? `<details class="mt"><summary class="muted">${w.rejected_candidates.length} generated rewrite(s) rejected for invented claims</summary>
        <ul class="list mt">${w.rejected_candidates.map((c) => `<li class="bad">${esc(c.text)}<br><span class="muted">Unsupported: ${c.unsupported.map((u) => `<code>${esc(u)}</code>`).join(" ")}</span></li>`).join("")}</ul></details>` : ""}
      <p class="muted small" style="margin:12px 0 0">Scores are the same heuristic as above, not measured results.</p>
    </div>
    <div class="card mt fade-in"><div class="card-title">Recommendation</div><p class="kpi-sub" style="margin:0">${esc(r.recommendation)}</p></div>`;
  $("#copy-copy", out).addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(w.refined_text); toast("Rewrite copied to clipboard.", "", "success"); }
    catch { toast("Couldn't access the clipboard."); }
  });
  animate(out);
}

// ── Image Analysis ───────────────────────────────────────────────────
const CHECK_TITLES = { focus: "Focus / sharpness", exposure: "Exposure", contrast: "Contrast", resolution: "Resolution", noise: "Noise / grain", framing: "Aspect ratio" };

function pageImage(view) {
  const st = S.image;
  view.innerHTML = `
    <p class="lead">Check whether a creative is technically fit to run: focus, exposure, contrast, grain, resolution and framing
      are measured the way a feed displays them, and serious defects like blur can't be averaged away.</p>
    <div id="img-drop"></div>
    <div id="img-out"></div>`;
  $("#img-drop").append(dropzone({
    accept: "image/png,image/jpeg,image/webp", title: st.file ? "Upload a different image" : "Drop an ad image here, or click to upload",
    sub: "JPG, PNG or WebP · up to 25 MB", compact: !!st.file,
    onFile: (f) => { if (!isImage(f)) return toast("That doesn't look like an image.", "Upload a JPG, PNG or WebP."); analyzeImage(f); },
  }));
  if (st.file) renderImageResult();

  async function analyzeImage(file) {
    if (st.url) URL.revokeObjectURL(st.url);
    Object.assign(st, { file, url: URL.createObjectURL(file), result: null, styled: null, error: null, loading: true });
    pageImage(view);
    try { st.result = await api("/api/image/analyze", { form: formOf({ image: file }) }); }
    catch (e) { st.error = e; }
    st.loading = false;
    if (ROUTER.current === "image") renderImageResult();
  }

  function renderImageResult() {
    const out = $("#img-out");
    const r = st.result;
    const left = `<div class="card"><div class="preview"><img src="${st.url}" alt="Uploaded creative"></div>
      <div class="row mt"><span class="file-chip">${icon("image")}${esc(st.file.name)}</span></div>
      ${r ? measurements(r.features) : ""}</div>`;
    if (st.loading) {
      out.innerHTML = `<div class="split mt">${left}<div class="stack"><div class="skeleton" style="height:180px"></div><div class="skeleton" style="height:320px"></div></div></div>`;
      return;
    }
    if (st.error) { out.innerHTML = `<div class="split mt">${left}<div>${errorNote(st.error)}</div></div>`; return; }
    const cap = r.capped ? r.checks.find((c) => c.name === r.cap_check) : null;
    const reasons = r.capped ? r.reasons.slice(1) : r.reasons; // first reason repeats the cap note
    out.innerHTML = `
      <div class="split mt fade-in">${left}
        <div class="stack">
          <div class="card">
            <div class="kpi-label">Creative quality ${heuristicTag}</div>
            <div class="score-block mt">${ring(r.score, { tier: r.label })}
              <div class="stack" style="flex:1">
                <div>${badge(TIER_TEXT[r.label], r.label)}</div>
                ${cap ? `<div class="note note-warn">${icon("alert")}<div><b>Capped by ${esc(CHECK_TITLES[cap.name].toLowerCase())}.</b> A defect this serious can't be offset by strengths elsewhere.</div></div>` : ""}
                <div class="score-parts small">
                  <div><span class="muted">Technical checks</span><b>${r.feature_score.toFixed(0)}</b></div>
                  <div><span class="muted">CLIP appeal</span><b>${r.used_clip ? r.clip_score.toFixed(0) : "n/a"}</b></div></div>
              </div></div>
          </div>
          ${r.trained_model ? `<div class="card"><div class="kpi-label">Predicted engagement <span class="tag tag-model">Trained model</span></div>
            <div class="kpi-value mt">${pct(r.trained_model.probability_top_third)}</div>
            <div class="kpi-sub">Chance this image lands in its creator's top third of posts. Validated on unseen accounts:
              AUC ${r.trained_model.auc.toFixed(2)} (95% CI ${r.trained_model.auc_ci[0].toFixed(2)}–${r.trained_model.auc_ci[1].toFixed(2)}).</div></div>` : ""}
          <div class="card"><div class="card-title">Technical checks</div>
            <p class="card-sub">Measured at feed size (long side 1080 px). A failed <span class="crit">Critical</span> check caps the
              overall score at 30; heavy grain or very low contrast cap it at 45.</p>
            <div class="checks">${r.checks.map(checkRow).join("")}</div></div>
        </div>
      </div>
      <div class="card mt fade-in"><div class="card-title">What to fix</div>
        <ul class="list">${reasons.map((t) => `<li class="${/capped|blurry|underexposed|washed out|very /i.test(t) ? "bad" : /note:|solid/i.test(t) ? "good" : "warn"}">${esc(t)}</li>`).join("")}</ul></div>
      <div class="card mt fade-in" id="img-style">
        <div class="card-title">Improve your creative</div>
        <p class="card-sub">Deterministic pixel adjustments, re-scored with the same checks. Auto-fix only corrects what the checks flagged; real blur can't be recovered.</p>
        <div class="chips" id="img-presets">${(STATUS.data?.style_presets || ["Auto-fix issues"]).map((p) =>
          `<button class="chip ${p === (st.preset || "Auto-fix issues") ? "active" : ""}" data-p="${esc(p)}">${esc(p)}</button>`).join("")}</div>
        <div class="row mt"><button class="btn btn-primary" id="img-apply">${icon("sparkles")}Apply style</button></div>
        <div id="img-styled"></div>
      </div>`;
    $$("#img-presets .chip", out).forEach((b) => b.addEventListener("click", () => {
      st.preset = b.dataset.p;
      $$("#img-presets .chip", out).forEach((x) => x.classList.toggle("active", x === b));
    }));
    $("#img-apply", out).addEventListener("click", (e) => withBusy(e.currentTarget, async () => {
      const preset = st.preset || "Auto-fix issues";
      try {
        const res = await api("/api/image/style", { form: formOf({ image: st.file, preset }) });
        st.styled = { ...res, preset };
        renderStyled();
      } catch (err) { $("#img-styled", out).innerHTML = `<div class="mt">${errorNote(err)}</div>`; }
    }));
    if (st.styled) renderStyled();
    animate(out);

    function renderStyled() {
      const s = st.styled, box = $("#img-styled", out);
      box.innerHTML = `<div class="mt">${compareView(st.url, s.before, s.image, s.after, s.preset)}</div>
        <div class="row mt">${s.after - s.before <= 0 ? `<span class="muted small">This style didn't raise the score for this image — try another preset.</span>` : ""}
          <span class="spacer"></span>
          <button class="btn btn-secondary" id="img-use">${icon("refresh")}Analyze styled version</button>
          <button class="btn btn-primary" id="img-dl">${icon("download")}Download</button></div>`;
      const name = s.preset.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
      $("#img-dl", box).addEventListener("click", () => download(s.image, `${name}.png`));
      $("#img-use", box).addEventListener("click", async () => analyzeImage(await dataUrlToFile(s.image, `${name}.png`)));
    }
  }
}

function checkRow(c) {
  const ic = c.status === "pass" ? "check" : c.status === "warn" ? "alert" : "x";
  return `<div class="check check-${c.status}">
    <span class="check-icon">${icon(ic)}</span>
    <span class="check-name">${esc(CHECK_TITLES[c.name] || c.name)}${c.critical ? '<span class="crit">Critical</span>' : ""}</span>
    <span class="check-score">${Math.round(c.score)}</span>
    ${bar(c.score, c.status)}
    <span class="check-msg">${esc(c.message)}</span></div>`;
}

function measurements(f) {
  const items = [
    ["Resolution", `${f.width} × ${f.height}`],
    ["Aspect ratio", f.aspect_ratio.toFixed(2)],
    ["Blur index", `${f.blur.toFixed(2)} <span class="muted">(0 sharp – 1 blurred)</span>`],
    ["Noise σ", f.noise.toFixed(1)],
    ["Brightness", `${Math.round(f.brightness)} / 255`],
    ["Tonal range", `${Math.round(f.dynamic_range)} / 255`],
    ["Saturation", `${Math.round(f.saturation)} / 255`],
  ];
  return `<div class="eyebrow mt">Measurements</div><div class="metrics">${items.map(([k, v]) =>
    `<div class="metric"><span>${k}</span><b>${v}</b></div>`).join("")}</div>`;
}

// ── Bulk Analysis ────────────────────────────────────────────────────
function pageBulk(view) {
  const st = S.bulk;
  view.innerHTML = `
    <p class="lead">Analyze a whole campaign at once. Upload a CSV with your ad copy (a column named <code>text</code>, <code>caption</code>,
      <code>copy</code> or similar); an optional <code>offer</code> column grounds the rewrites.</p>
    <div class="card">
      <div id="bulk-drop"></div>
      <div class="row mt">
        ${st.file ? `<span class="file-chip">${icon("table")}${esc(st.file.name)}</span>` : ""}
        <label class="switch"><input type="checkbox" id="bulk-rewrite" ${st.rewrite !== false ? "checked" : ""}>Generate grounded rewrites (slower)</label>
        <span class="spacer"></span>
        <button class="btn btn-primary" id="bulk-run" ${st.file && !st.job ? "" : "disabled"}>${icon("arrow")}Run analysis</button>
      </div>
    </div>
    <div id="bulk-out"></div>`;
  $("#bulk-drop").append(dropzone({
    accept: ".csv,text/csv", title: st.file ? "Choose a different CSV" : "Drop a CSV here, or click to upload", sub: "Comma-separated, UTF-8",
    compact: true, onFile: (f) => { Object.assign(st, { file: f, result: null, error: null, job: null }); pageBulk(view); },
  }));
  $("#bulk-rewrite").addEventListener("change", (e) => { st.rewrite = e.target.checked; });
  $("#bulk-run").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    btn.classList.add("loading"); btn.disabled = true;
    Object.assign(st, { result: null, error: null });
    try {
      st.job = await api("/api/bulk", { form: formOf({ file: st.file, rewrite: st.rewrite !== false }) });
      st.job.progress = 0;
      renderBulk();
      await pollBulk();
    } catch (err) { st.error = err; }
    btn.classList.remove("loading"); btn.disabled = false;
    if (ROUTER.current === "bulk") renderBulk();
  });
  renderBulk();

  async function pollBulk() {
    for (;;) {
      await new Promise((r) => setTimeout(r, 700));
      const j = await api(`/api/bulk/${st.job.job_id}`);
      st.job.progress = j.progress;
      if (j.status === "done") { st.result = j.result; st.job = null; return; }
      if (j.status === "error") { st.job = null; throw new ApiError(j.error, j.hint); }
      if (ROUTER.current === "bulk") { const p = $("#bulk-progress"); if (p) { p.style.width = pct(j.progress); $("#bulk-pct").textContent = pct(j.progress); } }
    }
  }

  function renderBulk() {
    const out = $("#bulk-out");
    if (!out) return;
    if (st.error) { out.innerHTML = `<div class="mt">${errorNote(st.error)}</div>`; return; }
    if (st.job) {
      out.innerHTML = `<div class="card mt"><div class="row"><b>Analyzing ${st.job.rows.toLocaleString()} rows</b>
        <span class="muted">· column <code>${esc(st.job.text_column)}</code></span><span class="spacer"></span><b id="bulk-pct">${pct(st.job.progress)}</b></div>
        <div class="progress mt"><i id="bulk-progress" style="width:${pct(st.job.progress)}"></i></div>
        <p class="muted small" style="margin:10px 0 0">You can switch pages — the analysis keeps running.</p></div>`;
      return;
    }
    if (!st.result) { out.innerHTML = ""; return; }
    const r = st.result, c = r.counts, n = r.rows || 1;
    out.innerHTML = `
      <div class="grid grid-4 mt fade-in">
        <div class="card kpi"><div class="kpi-label">Rows analyzed</div><div class="kpi-value">${r.rows.toLocaleString()}</div><div class="kpi-sub">Column <code>${esc(r.text_column)}</code></div></div>
        ${["positive", "neutral", "negative"].map((k) => `<div class="card kpi"><div class="kpi-label">${k}</div>
          <div class="kpi-value" style="color:${SENT_COLOR[k]}">${c[k].toLocaleString()}</div><div class="kpi-sub">${pct(c[k] / n)} of rows</div></div>`).join("")}
      </div>
      <div class="grid grid-2 mt fade-in">
        <div class="card"><div class="card-title">Sentiment mix</div><div class="donut-wrap mt">${donut(c)}
          <div class="legend">${["positive", "neutral", "negative"].map((k) => `<div><i style="background:${SENT_COLOR[k]}"></i>${k} · ${pct(c[k] / n)}</div>`).join("")}</div></div></div>
        <div class="card"><div class="card-title">Engagement ${heuristicTag}</div>
          <div class="dist mt">${["high", "medium", "low"].map((k) => `<div class="dist-row"><span>${TIER_TEXT[k]}</span>
            ${bar(100 * r.engagement_counts[k] / n, k)}<b>${r.engagement_counts[k].toLocaleString()}</b></div>`).join("")}</div></div>
      </div>
      <div class="card mt fade-in"><div class="card-title">Model confidence by sentiment</div>
        <p class="card-sub">How sure the model is about each prediction. Low-confidence rows are worth a manual look.</p>${histogram(r.confidence)}</div>
      <div class="card mt fade-in">
        <div class="row"><div class="tabs" id="bulk-tabs">
          <button class="active" data-t="sentiment">Sentiment</button><button data-t="engagement">Engagement</button>
          ${r.columns.includes("refined_text") ? '<button data-t="rewrites">Rewrites</button>' : ""}</div>
          <span class="spacer"></span><span class="muted small">First ${Math.min(200, r.rows)} rows</span>
          <button class="btn btn-primary" id="bulk-dl">${icon("download")}Download full CSV</button></div>
        <div class="table-wrap mt" id="bulk-table"></div>
      </div>`;
    const tables = {
      sentiment: [[r.text_column, "Text", "wide"], ["sentiment", "Sentiment"], ["confidence", "Confidence", "num"], ["sentiment_reason", "Why", "wide"]],
      engagement: [[r.text_column, "Text", "wide"], ["engagement_label", "Engagement"], ["heuristic_engagement_score", "Score", "num"], ["engagement_reason", "Factors", "wide"]],
      rewrites: [[r.text_column, "Original", "wide"], ["refined_text", "Rewrite", "wide"], ["heuristic_score_change", "Score change", "num"], ["rewrites_rejected_for_invented_claims", "Rejected", "num"]],
    };
    const show = (t) => {
      $("#bulk-table", out).innerHTML = `<table><thead><tr>${tables[t].map(([, h]) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${
        r.preview.map((row) => `<tr>${tables[t].map(([k, , cls]) => {
          let v = row[k];
          if (k === "sentiment") v = badge(v, v);
          else if (k === "engagement_label") v = badge(TIER_TEXT[v] || v, v);
          else if (k === "confidence") v = pct(v, 1);
          else if (k === "heuristic_score_change") v = `${v >= 0 ? "+" : ""}${v}`;
          else v = esc(v);
          return `<td class="${cls || ""}">${v}</td>`;
        }).join("")}</tr>`).join("")}</tbody></table>`;
    };
    $$("#bulk-tabs button", out).forEach((b) => b.addEventListener("click", () => {
      $$("#bulk-tabs button", out).forEach((x) => x.classList.toggle("active", x === b)); show(b.dataset.t);
    }));
    show("sentiment");
    $("#bulk-dl", out).addEventListener("click", () => downloadText(r.csv, "sentiment_results.csv", "text/csv"));
    animate(out);
  }
}

function donut(counts) {
  const keys = ["positive", "neutral", "negative"], total = keys.reduce((a, k) => a + counts[k], 0) || 1;
  const r = 52, c = 2 * Math.PI * r;
  let offset = 0;
  const arcs = keys.map((k) => {
    const len = (counts[k] / total) * c;
    const arc = `<circle cx="75" cy="75" r="${r}" fill="none" stroke="${SENT_COLOR[k]}" stroke-width="20"
      stroke-dasharray="${len} ${c - len}" stroke-dashoffset="${-offset}" transform="rotate(-90 75 75)"/>`;
    offset += len;
    return arc;
  }).join("");
  return `<svg class="donut" viewBox="0 0 150 150"><circle cx="75" cy="75" r="${r}" fill="none" stroke="var(--muted-bg)" stroke-width="20"/>${arcs}
    <text x="75" y="72" text-anchor="middle" font-family="Plus Jakarta Sans" font-weight="800" font-size="22" fill="var(--text)">${total.toLocaleString()}</text>
    <text x="75" y="92" text-anchor="middle" font-size="11" fill="var(--text-3)">rows</text></svg>`;
}

function histogram(rows) {
  const bins = 10, lo = 0.3, keys = ["positive", "neutral", "negative"];
  const counts = Array.from({ length: bins }, () => ({ positive: 0, neutral: 0, negative: 0 }));
  for (const row of rows) {
    const i = clamp(Math.floor(((row.confidence - lo) / (1 - lo)) * bins), 0, bins - 1);
    if (counts[i][row.sentiment] !== undefined) counts[i][row.sentiment] += 1;
  }
  const max = Math.max(1, ...counts.map((b) => b.positive + b.neutral + b.negative));
  return `<div class="hist">${counts.map((b) => `<div title="${keys.map((k) => `${k}: ${b[k]}`).join(", ")}">${keys.map((k) =>
    `<span style="height:${(130 * b[k]) / max}px;background:${SENT_COLOR[k]}"></span>`).join("")}</div>`).join("")}</div>
    <div class="hist-axis"><span>${pct(lo)}</span><span>Confidence</span><span>100%</span></div>
    <div class="legend mt" style="flex-direction:row;gap:16px">${keys.map((k) => `<div><i style="background:${SENT_COLOR[k]}"></i>${k}</div>`).join("")}</div>`;
}

// ── Website Audit ────────────────────────────────────────────────────
function pageWebsite(view) {
  const st = S.website;
  view.innerHTML = `
    <p class="lead">Audit a single landing page: search-engine basics (SEO), signals that help AI answer engines read and cite it (GEO),
      Google's real performance data, and the tone of its copy.</p>
    <div class="card">
      <label class="field" for="web-url">Page URL</label>
      <div class="row" style="flex-wrap:nowrap">
        <input id="web-url" class="input" placeholder="e.g. https://www.example.com/pricing" inputmode="url">
        <button class="btn btn-primary" id="web-run">${icon("arrow")}Audit page</button>
      </div>
      <div class="row mt">
        <label class="switch"><input type="checkbox" id="web-psi" ${st.psi !== false ? "checked" : ""}>Google PageSpeed (slower, ~30–60 s)</label>
        <label class="switch"><input type="checkbox" id="web-sent" ${st.sent !== false ? "checked" : ""}>Copy sentiment</label>
      </div>
    </div>
    <div id="web-out"></div>`;
  const url = $("#web-url"), out = $("#web-out"), run = $("#web-run");
  url.value = st.url || "";
  url.addEventListener("input", () => { st.url = url.value; });
  $("#web-psi").addEventListener("change", (e) => { st.psi = e.target.checked; });
  $("#web-sent").addEventListener("change", (e) => { st.sent = e.target.checked; });
  const go = () => withBusy(run, async () => {
    if (!url.value.trim()) { url.focus(); toast("Enter a page URL first."); return; }
    out.innerHTML = `<div class="card mt"><div class="row"><b>Fetching the page and running checks…</b></div>
      <p class="muted small">${st.psi !== false ? "Google PageSpeed usually takes 30–60 seconds." : ""}</p>
      <div class="grid grid-3 mt">${'<div class="skeleton" style="height:120px"></div>'.repeat(3)}</div></div>`;
    try {
      st.result = await api("/api/website", { json: { url: url.value.trim(), pagespeed: st.psi !== false, sentiment: st.sent !== false } });
      renderWebsite(out, st.result);
    } catch (e) { st.result = null; out.innerHTML = `<div class="mt">${errorNote(e)}</div>`; }
  });
  run.addEventListener("click", go);
  url.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  if (st.result) renderWebsite(out, st.result);
}

function renderWebsite(out, res) {
  const tierW = (s) => (s == null ? null : tierOf(s, 80, 50));
  const kpi = (key, title, heuristic) => {
    const s = res.scores[key], t = tierW(s);
    if (t == null) return `<div class="card kpi"><div class="kpi-label">${title}</div>${badge("Unavailable", "none")}
      <div class="kpi-sub">${esc((res.performance.error || "Not run").slice(0, 120))}</div></div>`;
    return `<div class="card"><div class="kpi-label">${title} ${heuristic ? heuristicTag : '<span class="tag tag-model">Google Lighthouse</span>'}</div>
      <div class="score-block mt">${ring(s, { tier: t, small: true })}<div>${badge(TIER_TEXT[t], t)}
      <div class="kpi-sub">${heuristic ? "Weighted checklist" : "Lab performance score"}</div></div></div></div>`;
  };
  const checks = res.checks, toFix = checks.filter((c) => c.status === "fail" || c.status === "warn")
    .sort((a, b) => (a.status === "fail" ? 0 : 1) - (b.status === "fail" ? 0 : 1)).slice(0, 6);
  const perf = res.performance, sent = res.sentiment;
  const rating = { good: "good", "needs improvement": "needs", poor: "poor" };
  out.innerHTML = `
    <p class="muted small mt">Analysed <a href="${esc(res.final_url)}" target="_blank" rel="noopener noreferrer">${esc(res.final_url)}</a>
      · ${res.word_count.toLocaleString()} words of main content · ${esc(res.fetched_at)} UTC
      ${res.redirect_chain.length ? `· redirected from ${esc(res.redirect_chain[0])}` : ""}</p>
    <div class="grid grid-3 fade-in">${kpi("seo", "SEO", true)}${kpi("geo", "GEO", true)}${kpi("performance", "Performance", false)}</div>
    <p class="muted small">${esc(res.score_note)}</p>
    <div class="grid grid-2 mt fade-in">
      <div class="card"><div class="card-title">Top fixes</div>
        ${toFix.length ? `<ul class="list">${toFix.map((c) => `<li class="${c.status === "fail" ? "bad" : "warn"}"><b>${esc(c.category)} · ${esc(c.name)}</b> — ${esc(c.fix)}</li>`).join("")}</ul>`
          : `<p class="kpi-sub">Every check passed.</p>`}</div>
      <div class="card"><div class="card-title">Tone of the page copy</div>
        ${sent.available ? `<div class="row mt">${badge(sent.label.toUpperCase(), sent.label)}<span class="kpi-sub">${pct(sent.confidence)} across ${sent.chunks} text chunks</span></div>
          <div class="dist mt">${["positive", "neutral", "negative"].map((k) => `<div class="dist-row"><span>${k}</span>${bar(100 * sent.scores[k], k)}<b>${pct(sent.scores[k])}</b></div>`).join("")}</div>`
          : `<p class="kpi-sub">${esc(sent.error || "Not run.")}</p>`}</div>
    </div>
    ${perf.available ? `<div class="section fade-in"><div class="eyebrow">Core Web Vitals · ${esc(perf.strategy || "mobile")}</div><div class="grid grid-3">${
      [["lcp_ms", "LCP · loading"], ["cls", "CLS · visual stability"], ["inp_ms", "INP · responsiveness"]].map(([k, name]) => {
        const v = perf.vitals[k];
        if (!v || v.value == null) return `<div class="card kpi"><div class="kpi-label">${name}</div>${badge("No data", "none")}<div class="kpi-sub">Not enough real-user data</div></div>`;
        return `<div class="card kpi"><div class="kpi-label">${name}</div><div class="kpi-value">${k === "cls" ? v.value.toFixed(3) : `${Math.round(v.value)} ms`}</div>
          <div class="row">${badge(v.rating, rating[v.rating] || "none")}<span class="muted small">${esc(v.source)} data</span></div></div>`;
      }).join("")}</div></div>` : ""}
    <div class="card mt fade-in">
      <div class="row"><div class="card-title">All ${checks.length} checks</div><span class="spacer"></span>
        <div class="tabs" id="web-filter">${["all", "fail", "warn", "pass"].map((f, i) => `<button class="${i ? "" : "active"}" data-f="${f}">${f[0].toUpperCase() + f.slice(1)} ${
          f === "all" ? checks.length : checks.filter((c) => c.status === f).length}</button>`).join("")}</div></div>
      <div class="table-wrap mt" id="web-table"></div>
      <details class="mt"><summary class="muted">How the scores are calculated</summary>
        <p class="kpi-sub">Each check scores pass = 1, warn = 0.5, fail = 0, multiplied by its weight; checks that couldn't run are left out.
        Score = weighted points ÷ weights that ran × 100. These are checklists, not predictions of search ranking or AI citations.</p>
        <p class="small"><b>SEO weights:</b> ${Object.entries(res.weights.seo).map(([k, v]) => `${esc(k)} ${v}`).join(", ")}<br>
        <b>GEO weights:</b> ${Object.entries(res.weights.geo).map(([k, v]) => `${esc(k)} ${v}`).join(", ")}</p></details>
      <div class="row row-end mt"><button class="btn btn-secondary" id="web-dl">${icon("download")}Download JSON</button></div>
    </div>`;
  const table = (f) => {
    const rows = checks.filter((c) => f === "all" || c.status === f);
    $("#web-table", out).innerHTML = `<table><thead><tr><th>Area</th><th>Check</th><th>Result</th><th>Value</th><th>Details</th><th>How to fix</th></tr></thead><tbody>${
      rows.map((c) => `<tr><td>${esc(c.category)}</td><td><b>${esc(c.name)}</b></td><td>${badge(c.status, c.status)}</td>
        <td class="num">${esc(c.value)}</td><td class="wide">${esc(c.detail)}</td><td class="wide">${esc(c.fix)}</td></tr>`).join("")}</tbody></table>`;
  };
  $$("#web-filter button", out).forEach((b) => b.addEventListener("click", () => {
    $$("#web-filter button", out).forEach((x) => x.classList.toggle("active", x === b)); table(b.dataset.f);
  }));
  table("all");
  $("#web-dl", out).addEventListener("click", () => downloadText(JSON.stringify(res, null, 2), "website_analysis.json", "application/json"));
  animate(out);
}

// ── Generative Editing ───────────────────────────────────────────────
function pageGenerate(view) {
  const st = S.generate;
  st.strength ??= 0.55; st.steps ??= 25; st.guidance ??= 7.5;
  view.innerHTML = `
    <p class="lead">Restyle an image from a text prompt with Stable Diffusion (img2img). This is generative AI, not a pixel tweak —
      the model reinterprets the whole image, so results vary. The first run downloads the model (~4–5 GB).</p>
    <div id="gen-gate"></div>
    <div class="split" id="gen-body">
      <div class="card"><div id="gen-drop"></div>${st.file ? `<div class="preview mt"><img src="${st.url}" alt="Original"></div>` : ""}</div>
      <div class="card">
        <label class="field" for="gen-prompt">Describe the new style or scene</label>
        <input id="gen-prompt" class="input" placeholder="e.g. on a warm sunset beach, golden hour lighting">
        <div class="stack mt">
          ${slider("strength", "Strength (how much changes)", 0.2, 0.9, 0.05, st.strength)}
          ${slider("steps", "Inference steps", 10, 50, 5, st.steps)}
          ${slider("guidance", "Guidance scale", 1, 15, 0.5, st.guidance)}
        </div>
        <p class="muted small" id="gen-est"></p>
        <button class="btn btn-primary" id="gen-run">${icon("sparkles")}Generate</button>
      </div>
    </div>
    <div id="gen-out"></div>`;
  if (!setupFeatureGate($("#gen-gate"), "diffusers", "Generative editing", "diffusers accelerate")) $("#gen-run").disabled = true;
  $("#gen-drop").append(dropzone({
    accept: "image/png,image/jpeg,image/webp", title: st.file ? "Replace image" : "Upload an image", sub: "JPG, PNG or WebP", compact: true,
    onFile: (f) => { if (!isImage(f)) return toast("That doesn't look like an image."); if (st.url) URL.revokeObjectURL(st.url);
      Object.assign(st, { file: f, url: URL.createObjectURL(f), result: null }); pageGenerate(view); },
  }));
  const prompt = $("#gen-prompt");
  prompt.value = st.prompt || "";
  prompt.addEventListener("input", () => { st.prompt = prompt.value; });
  const updateEstimate = async () => {
    try { $("#gen-est").textContent = `Estimated time: ${(await api(`/api/generate/estimate?steps=${st.steps}`)).estimate}`; } catch { /* optional */ }
  };
  $$("input[type=range]", view).forEach((inp) => inp.addEventListener("input", () => {
    st[inp.dataset.k] = Number(inp.value); $(`output[for=${inp.id}]`).textContent = inp.value;
    if (inp.dataset.k === "steps") updateEstimate();
  }));
  updateEstimate();
  $("#gen-run").addEventListener("click", (e) => withBusy(e.currentTarget, async () => {
    if (!st.file) return toast("Upload an image first.");
    if (!prompt.value.trim()) { prompt.focus(); return toast("Describe the new style or scene first."); }
    $("#gen-out").innerHTML = `<div class="card mt"><b>Generating…</b><p class="muted small">This can take several minutes on a CPU.</p><div class="skeleton" style="height:240px"></div></div>`;
    try {
      st.result = await api("/api/generate", { form: formOf({ image: st.file, prompt: prompt.value, strength: st.strength, steps: st.steps, guidance: st.guidance }) });
      renderGen();
    } catch (err) { $("#gen-out").innerHTML = `<div class="mt">${errorNote(err)}</div>`; }
  }));
  if (st.result) renderGen();

  function renderGen() {
    const box = $("#gen-out");
    if (!box) return;
    box.innerHTML = `<div class="card mt fade-in"><div class="card-title">Result</div>
      <p class="card-sub">Quality scores are the same creative-quality heuristic used on the Image Analysis page.</p>
      ${compareView(st.url, st.result.before, st.result.image, st.result.after, "Generated")}
      <div class="row row-end mt"><button class="btn btn-primary" id="gen-dl">${icon("download")}Download</button></div></div>`;
    $("#gen-dl").addEventListener("click", () => download(st.result.image, "generative_edit.png"));
  }
}

const slider = (k, label, min, max, step, value) => `<div class="ctrl"><label for="sl-${k}">${label}</label><output for="sl-${k}">${value}</output>
  <input type="range" id="sl-${k}" data-k="${k}" min="${min}" max="${max}" step="${step}" value="${value}" style="grid-column:1/3"></div>`;

// ── Precise Editing ──────────────────────────────────────────────────
function pagePrecise(view) {
  const st = S.precise;
  view.innerHTML = `
    <p class="lead">Exact, non-generative edits: put the subject on a precise background colour, or detect and replace text in the image.
      Colours and text come out exact rather than approximate.</p>
    <div class="split">
      <div class="card"><div id="pr-drop"></div>
        ${st.file ? `<div class="preview mt" id="pr-preview"><img src="${st.url}" alt="Image being edited" id="pr-img"></div>` : ""}</div>
      <div class="stack">
        <div class="card" id="pr-bg"><div class="card-title">Background colour</div>
          <p class="card-sub">Cuts out the subject and places it on an exact solid colour.</p>
          <div class="row"><input type="color" id="pr-color" value="${st.bg || "#FFFFFF"}"><code id="pr-color-code">${st.bg || "#FFFFFF"}</code>
            <span class="spacer"></span><button class="btn btn-primary" id="pr-swap">Swap background</button></div></div>
        <div class="card" id="pr-text"><div class="card-title">Text</div>
          <p class="card-sub">Detects text so you can replace it. Works best on flat, simple backgrounds.</p>
          <button class="btn btn-secondary" id="pr-detect">${icon("target")}Detect text</button>
          <div id="pr-regions"></div></div>
      </div>
    </div>
    <div id="pr-out"></div>`;
  const okBg = setupFeatureGate($("#pr-bg"), "rembg", "Background removal", "rembg", "beforeend");
  const okOcr = setupFeatureGate($("#pr-text"), "easyocr", "Text detection", "easyocr", "beforeend");
  $("#pr-drop").append(dropzone({
    accept: "image/png,image/jpeg,image/webp", title: st.file ? "Replace image" : "Upload an image", sub: "JPG, PNG or WebP", compact: true,
    onFile: (f) => { if (!isImage(f)) return toast("That doesn't look like an image."); setBase(f); },
  }));
  const color = $("#pr-color");
  color.addEventListener("input", () => { st.bg = color.value.toUpperCase(); $("#pr-color-code").textContent = st.bg; });
  $("#pr-swap").disabled = !okBg;
  $("#pr-detect").disabled = !okOcr;

  function setBase(file) {
    if (st.url) URL.revokeObjectURL(st.url);
    Object.assign(st, { file, url: URL.createObjectURL(file), regions: null, chosen: 0, result: null });
    pagePrecise(view);
  }

  $("#pr-swap").addEventListener("click", (e) => withBusy(e.currentTarget, async () => {
    if (!st.file) return toast("Upload an image first.");
    try {
      const r = await api("/api/precise/background", { form: formOf({ image: st.file, color: st.bg || "#FFFFFF" }) });
      st.result = { ...r, label: "Background swapped", name: "background_swapped.png" }; renderResult();
    } catch (err) { $("#pr-out").innerHTML = `<div class="mt">${errorNote(err)}</div>`; }
  }));
  $("#pr-detect").addEventListener("click", (e) => withBusy(e.currentTarget, async () => {
    if (!st.file) return toast("Upload an image first.");
    try {
      const r = await api("/api/precise/detect-text", { form: formOf({ image: st.file }) });
      Object.assign(st, { regions: r.regions, size: [r.width, r.height], chosen: 0 });
      if (!r.regions.length) toast("No text detected in this image.");
      renderRegions();
    } catch (err) { $("#pr-regions").innerHTML = `<div class="mt">${errorNote(err)}</div>`; }
  }));

  function renderRegions() {
    const box = $("#pr-regions"), prev = $("#pr-preview");
    $$(".overlay-box", prev).forEach((x) => x.remove());
    if (!st.regions || !st.regions.length) { box.innerHTML = ""; return; }
    const img = $("#pr-img");
    const place = () => {
      $$(".overlay-box", prev).forEach((x) => x.remove());
      const sx = img.clientWidth / st.size[0], sy = img.clientHeight / st.size[1], ox = img.offsetLeft, oy = img.offsetTop;
      st.regions.forEach((r, i) => {
        const [x1, y1, x2, y2] = r.bbox, d = document.createElement("div");
        d.className = `overlay-box ${i === st.chosen ? "active" : ""}`;
        Object.assign(d.style, { left: `${ox + x1 * sx}px`, top: `${oy + y1 * sy}px`, width: `${(x2 - x1) * sx}px`, height: `${(y2 - y1) * sy}px` });
        prev.append(d);
      });
    };
    img.complete ? place() : img.addEventListener("load", place, { once: true });
    window.addEventListener("resize", place);
    const r = st.regions[st.chosen];
    box.innerHTML = `<div class="stack mt">
      <div><label class="field" for="pr-sel">Detected text</label><select class="input" id="pr-sel">${st.regions.map((x, i) =>
        `<option value="${i}" ${i === st.chosen ? "selected" : ""}>"${esc(x.text)}" · ${pct(x.confidence)} confidence</option>`).join("")}</select></div>
      <div><label class="field" for="pr-new">Replace with</label><input class="input" id="pr-new" value="${esc(r.text)}"></div>
      <div class="row"><label class="field" style="margin:0">Text colour</label><input type="color" id="pr-tcolor" value="#000000">
        <span class="spacer"></span><button class="btn btn-primary" id="pr-replace">Replace text</button></div></div>`;
    $("#pr-sel").addEventListener("change", (e) => { st.chosen = Number(e.target.value); renderRegions(); });
    $("#pr-replace").addEventListener("click", (e) => withBusy(e.currentTarget, async () => {
      try {
        const res = await api("/api/precise/replace-text", { form: formOf({ image: st.file, bbox: st.regions[st.chosen].bbox.join(","),
          text: $("#pr-new").value, color: $("#pr-tcolor").value }) });
        st.result = { ...res, label: "Text replaced", name: "text_replaced.png" }; renderResult();
      } catch (err) { $("#pr-out").innerHTML = `<div class="mt">${errorNote(err)}</div>`; }
    }));
  }

  function renderResult() {
    const box = $("#pr-out"), r = st.result;
    if (!box || !r) return;
    box.innerHTML = `<div class="card mt fade-in"><div class="card-title">Result</div>
      ${compareView(st.url, r.before, r.image, r.after, r.label)}
      <div class="row row-end mt"><button class="btn btn-secondary" id="pr-continue">${icon("refresh")}Keep editing this result</button>
        <button class="btn btn-primary" id="pr-dl">${icon("download")}Download</button></div></div>`;
    $("#pr-dl").addEventListener("click", () => download(r.image, r.name));
    $("#pr-continue").addEventListener("click", async () => setBase(await dataUrlToFile(r.image, r.name)));
  }
  if (st.regions) renderRegions();
  renderResult();
}

// ── Canvas Editor (runs entirely in the browser) ─────────────────────
function pageCanvas(view) {
  const st = S.canvas;
  st.tool ??= "pen"; st.stroke ??= "#FF3B30"; st.width ??= 4; st.fill ??= "#FFFFFF"; st.fillOpacity ??= 0; st.shapes ??= []; st.text ??= "";
  const TOOLS = [["pen", "Draw", "pen"], ["line", "Line", "line"], ["rect", "Box", "rect"], ["ellipse", "Circle", "circle"], ["text", "Text", "text"], ["crop", "Crop", "crop"]];
  view.innerHTML = `
    <p class="lead">Annotate, add text boxes or crop. Drag on the image to draw; with <b>Text</b>, drag a box where the text should go.
      Everything happens in your browser.</p>
    <div class="card" id="cv-drop-card"><div id="cv-drop"></div></div>
    <div class="editor mt ${st.base ? "" : "hidden"}" id="cv-editor">
      <div class="card toolbar">
        <div><div class="eyebrow">Tool</div><div class="tools">${TOOLS.map(([id, label, ic]) =>
          `<button class="tool ${st.tool === id ? "active" : ""}" data-tool="${id}">${icon(ic)}${label}</button>`).join("")}</div></div>
        <div class="ctrl"><label>Stroke / text colour</label><input type="color" id="cv-stroke" value="${st.stroke}"></div>
        <div class="ctrl"><label for="cv-width">Stroke width</label><output for="cv-width">${st.width}</output>
          <input type="range" id="cv-width" min="1" max="20" value="${st.width}" style="grid-column:1/3"></div>
        <div class="ctrl"><label>Fill / text box colour</label><input type="color" id="cv-fill" value="${st.fill}"></div>
        <div class="ctrl"><label for="cv-op">Fill opacity</label><output for="cv-op">${st.fillOpacity}</output>
          <input type="range" id="cv-op" min="0" max="1" step="0.05" value="${st.fillOpacity}" style="grid-column:1/3"></div>
        <div><label class="field" for="cv-text">Text for text boxes</label><input class="input" id="cv-text" placeholder="e.g. NEW ARRIVALS" value="${esc(st.text)}"></div>
        <div class="row"><button class="btn btn-secondary" id="cv-undo" title="Undo (⌘/Ctrl+Z)">${icon("undo")}Undo</button>
          <button class="btn btn-ghost" id="cv-clear">${icon("trash")}Clear</button></div>
        <button class="btn btn-secondary ${st.crop ? "" : "hidden"}" id="cv-apply-crop">${icon("crop")}Apply crop</button>
        <button class="btn btn-primary" id="cv-dl">${icon("download")}Download PNG</button>
      </div>
      <div class="stage"><canvas id="cv"></canvas></div>
    </div>`;
  $("#cv-drop").append(dropzone({
    accept: "image/png,image/jpeg,image/webp", title: st.base ? "Start over with another image" : "Upload an image to edit", sub: "JPG, PNG or WebP",
    compact: !!st.base, onFile: (f) => {
      if (!isImage(f)) return toast("That doesn't look like an image.");
      const img = new Image();
      img.onload = () => { Object.assign(st, { base: img, shapes: [], crop: null }); pageCanvas(view); };
      img.src = URL.createObjectURL(f);
    },
  }));
  if (!st.base) return;

  const cv = $("#cv"), ctx = cv.getContext("2d");
  cv.width = st.base.naturalWidth || st.base.width; cv.height = st.base.naturalHeight || st.base.height;
  // stroke widths are chosen on screen, so scale them to the image's real size
  const unit = () => cv.width / cv.getBoundingClientRect().width;

  const draw = (shapes = st.shapes, withCrop = true) => {
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(st.base, 0, 0, cv.width, cv.height);
    for (const s of shapes) drawShape(ctx, s);
    if (withCrop && st.crop) {
      const { x, y, w, h } = norm(st.crop);
      ctx.save(); ctx.fillStyle = "rgba(18,22,40,.45)";
      ctx.fillRect(0, 0, cv.width, y); ctx.fillRect(0, y + h, cv.width, cv.height - y - h);
      ctx.fillRect(0, y, x, h); ctx.fillRect(x + w, y, cv.width - x - w, h);
      ctx.setLineDash([8 * unit(), 6 * unit()]); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * unit(); ctx.strokeRect(x, y, w, h); ctx.restore();
    }
  };
  draw();

  const point = (e) => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height]; };
  let current = null;
  cv.addEventListener("pointerdown", (e) => {
    cv.setPointerCapture(e.pointerId);
    const [x, y] = point(e);
    if (st.tool === "crop") { st.crop = { x1: x, y1: y, x2: x, y2: y }; current = "crop"; return; }
    if (st.tool === "text" && !st.text.trim()) { toast("Type the text first.", "Enter it in the “Text for text boxes” field, then drag a box."); return; }
    const u = unit();
    current = { tool: st.tool, stroke: st.stroke, width: st.width * u, fill: hexA(st.fill, st.fillOpacity), boxFill: st.fill,
      text: st.text, x1: x, y1: y, x2: x, y2: y, points: [[x, y]] };
  });
  cv.addEventListener("pointermove", (e) => {
    if (!current) return;
    const [x, y] = point(e);
    if (current === "crop") { st.crop.x2 = x; st.crop.y2 = y; draw(); return; }
    current.x2 = x; current.y2 = y;
    if (current.tool === "pen") current.points.push([x, y]);
    draw([...st.shapes, current]);
  });
  const end = () => {
    if (!current) return;
    if (current === "crop") {
      const { w, h } = norm(st.crop);
      if (w < 4 || h < 4) st.crop = null;
      $("#cv-apply-crop").classList.toggle("hidden", !st.crop);
    } else {
      const { w, h } = norm(current);
      if (current.tool === "pen" ? current.points.length > 1 : w > 2 || h > 2) st.shapes.push(current);
    }
    current = null; draw();
  };
  cv.addEventListener("pointerup", end);
  cv.addEventListener("pointercancel", end);

  $$(".tool", view).forEach((b) => b.addEventListener("click", () => {
    st.tool = b.dataset.tool; $$(".tool", view).forEach((x) => x.classList.toggle("active", x === b));
    if (st.tool !== "crop" && st.crop) { st.crop = null; $("#cv-apply-crop").classList.add("hidden"); draw(); }
  }));
  $("#cv-stroke").addEventListener("input", (e) => { st.stroke = e.target.value; });
  $("#cv-fill").addEventListener("input", (e) => { st.fill = e.target.value; });
  $("#cv-text").addEventListener("input", (e) => { st.text = e.target.value; });
  $("#cv-width").addEventListener("input", (e) => { st.width = Number(e.target.value); $("output[for=cv-width]").textContent = e.target.value; });
  $("#cv-op").addEventListener("input", (e) => { st.fillOpacity = Number(e.target.value); $("output[for=cv-op]").textContent = e.target.value; });
  const undo = () => { if (st.crop) st.crop = null; else st.shapes.pop(); $("#cv-apply-crop").classList.toggle("hidden", !st.crop); draw(); };
  $("#cv-undo").addEventListener("click", undo);
  $("#cv-clear").addEventListener("click", () => { st.shapes = []; st.crop = null; $("#cv-apply-crop").classList.add("hidden"); draw(); });
  $("#cv-apply-crop").addEventListener("click", () => {
    const { x, y, w, h } = norm(st.crop);
    draw(st.shapes, false); // flatten drawings into the crop
    const out = document.createElement("canvas");
    out.width = Math.round(w); out.height = Math.round(h);
    out.getContext("2d").drawImage(cv, x, y, w, h, 0, 0, w, h);
    const img = new Image();
    img.onload = () => { Object.assign(st, { base: img, shapes: [], crop: null }); pageCanvas(view); toast("Cropped.", "", "success"); };
    img.src = out.toDataURL("image/png");
  });
  $("#cv-dl").addEventListener("click", () => { draw(st.shapes, false); download(cv.toDataURL("image/png"), "canvas_edited.png"); draw(); });
  CANVAS_KEYS.undo = undo;
}

const CANVAS_KEYS = { undo: null };
document.addEventListener("keydown", (e) => {
  if (ROUTER.current === "canvas" && CANVAS_KEYS.undo && (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z"
      && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) { e.preventDefault(); CANVAS_KEYS.undo(); }
});

function norm(s) {
  return { x: Math.min(s.x1, s.x2), y: Math.min(s.y1, s.y2), w: Math.abs(s.x2 - s.x1), h: Math.abs(s.y2 - s.y1) };
}

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

function drawShape(ctx, s) {
  ctx.save();
  ctx.strokeStyle = s.stroke; ctx.lineWidth = s.width; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.fillStyle = s.fill;
  const { x, y, w, h } = norm(s);
  if (s.tool === "pen") {
    ctx.beginPath(); s.points.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py))); ctx.stroke();
  } else if (s.tool === "line") {
    ctx.beginPath(); ctx.moveTo(s.x1, s.y1); ctx.lineTo(s.x2, s.y2); ctx.stroke();
  } else if (s.tool === "rect") {
    ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
  } else if (s.tool === "ellipse") {
    ctx.beginPath(); ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  } else if (s.tool === "text") {
    // a solid box in the fill colour with the text fitted and centred inside (like the old "text box" action)
    ctx.fillStyle = s.boxFill; ctx.fillRect(x, y, w, h);
    let size = Math.max(8, h * 0.7);
    const font = (px) => `700 ${px}px Inter, -apple-system, sans-serif`;
    ctx.font = font(size);
    while (size > 8 && ctx.measureText(s.text).width > w * 0.92) { size -= 1; ctx.font = font(size); }
    ctx.fillStyle = s.stroke; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(s.text, x + w / 2, y + h / 2, w * 0.95);
  }
  ctx.restore();
}

// ══════════════════════════════════════════════════════════════════════
// Router + shell
// ══════════════════════════════════════════════════════════════════════
const PAGES = [
  { id: "copy", group: "Analyze", title: "Ad Copy", icon: "copy", render: pageCopy },
  { id: "image", group: "Analyze", title: "Image Analysis", icon: "image", render: pageImage },
  { id: "bulk", group: "Analyze", title: "Bulk Analysis", icon: "table", render: pageBulk },
  { id: "website", group: "Analyze", title: "Website Audit", icon: "globe", render: pageWebsite },
  { id: "generate", group: "Create & Edit", title: "Generative Editing", icon: "sparkles", render: pageGenerate },
  { id: "precise", group: "Create & Edit", title: "Precise Editing", icon: "target", render: pagePrecise },
  { id: "canvas", group: "Create & Edit", title: "Canvas Editor", icon: "pen", render: pageCanvas },
];

const ROUTER = {
  current: null,
  go() {
    const id = location.hash.replace(/^#\/?/, "") || "copy";
    const page = PAGES.find((p) => p.id === id) || PAGES[0];
    this.current = page.id;
    $$("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.id === page.id));
    $("#crumb").textContent = page.group;
    $("#page-title").textContent = page.title;
    document.title = `${page.title} · Sentiment Studio`;
    CANVAS_KEYS.undo = null;
    const view = $("#view");
    view.innerHTML = "";
    page.render(view);
    closeMenu();
    window.scrollTo(0, 0);
  },
};

function buildNav() {
  const groups = [...new Set(PAGES.map((p) => p.group))];
  $("#nav").innerHTML = groups.map((g) => `<div class="nav-group">${esc(g)}</div>${PAGES.filter((p) => p.group === g).map((p) =>
    `<a href="#/${p.id}" data-id="${p.id}">${icon(p.icon)}<span>${esc(p.title)}</span></a>`).join("")}`).join("");
}

function closeMenu() { $("#sidebar").classList.remove("open"); $("#backdrop").classList.remove("open"); }

buildNav();
$("#menu-btn").addEventListener("click", () => { $("#sidebar").classList.add("open"); $("#backdrop").classList.add("open"); });
$("#backdrop").addEventListener("click", closeMenu);
window.addEventListener("scroll", () => $(".topbar").classList.toggle("scrolled", window.scrollY > 4), { passive: true });
window.addEventListener("hashchange", () => ROUTER.go());
refreshStatus().then(() => {
  // pages that gate on optional dependencies need the status; re-render once it's in
  if (["generate", "precise", "image"].includes(ROUTER.current) && !$(".loading")) ROUTER.go();
});
ROUTER.go();
