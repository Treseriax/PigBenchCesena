
let SUMMARY = null;
let PRED = [];
let CONF = [];
let CLS = [];

function id(x) { return document.getElementById(x); }

async function loadAll() {
  SUMMARY = await (await fetch("/api/summary")).json();
  PRED = await (await fetch("/api/predictions")).json();
  CONF = await (await fetch("/api/confusions")).json();
  CLS = await (await fetch("/api/class_summary")).json();

  populateFilters();
  renderSummary();
  renderSidebars();
  renderGrid();
}

function unique(arr) {
  return [...new Set(arr.filter(x => x !== undefined && x !== null && String(x).trim() !== ""))].sort();
}

function populateSelect(el, values) {
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    el.appendChild(opt);
  }
}

function populateFilters() {
  populateSelect(id("policyFilter"), unique(PRED.map(r => r.split_policy)));
  populateSelect(id("actualFilter"), unique(PRED.map(r => r.actual)));
  populateSelect(id("predFilter"), unique(PRED.map(r => r.predicted)));
}

function renderSummary() {
  let lines = [];
  lines.push(`Prediction rows: ${SUMMARY.prediction_rows}`);
  lines.push(`Correct rows: ${SUMMARY.correct_rows}`);
  lines.push(`Error rows: ${SUMMARY.error_rows}`);
  lines.push("");
  for (const [k, v] of Object.entries(SUMMARY.by_policy)) {
    lines.push(`${k}: total=${v.total}, correct=${v.correct}, errors=${v.errors}`);
  }
  lines.push("");
  lines.push(SUMMARY.claim_boundary);
  id("summary").textContent = lines.join("\n");
}

function setFilter(actual, predicted, policy) {
  if (policy) id("policyFilter").value = policy;
  if (actual) id("actualFilter").value = actual;
  if (predicted) id("predFilter").value = predicted;
  id("statusFilter").value = "errors";
  renderGrid();
}

function renderSidebars() {
  const c = id("confusions");
  c.innerHTML = "";
  for (const r of CONF.slice(0, 25)) {
    const div = document.createElement("div");
    div.className = "rowlink";
    div.textContent = `${r.split_policy} | ${r.actual} → ${r.predicted}: ${r.error_count}`;
    div.onclick = () => setFilter(r.actual, r.predicted, r.split_policy);
    c.appendChild(div);
  }

  const cls = id("classes");
  cls.innerHTML = "";
  const sorted = [...CLS].sort((a, b) => Number(a.recall) - Number(b.recall)).slice(0, 25);
  for (const r of sorted) {
    const div = document.createElement("div");
    div.className = "rowlink";
    div.textContent = `${r.split_policy} | ${r.actual}: recall=${Number(r.recall).toFixed(3)}, support=${r.support}`;
    div.onclick = () => setFilter(r.actual, "", r.split_policy);
    cls.appendChild(div);
  }
}

function keep(r) {
  const p = id("policyFilter").value;
  const a = id("actualFilter").value;
  const pr = id("predFilter").value;
  const st = id("statusFilter").value;
  const q = id("searchBox").value.trim().toLowerCase();

  if (p && r.split_policy !== p) return false;
  if (a && r.actual !== a) return false;
  if (pr && r.predicted !== pr) return false;

  const correct = String(r.correct).toLowerCase() === "true";
  if (st === "errors" && correct) return false;
  if (st === "correct" && !correct) return false;

  if (q) {
    const hay = [
      r.canonical_gt_object_id,
      r.scan_frame_id,
      r.video_id,
      r.actual,
      r.predicted,
      r.split_policy,
      r.crop_type,
      r.model_type
    ].join(" ").toLowerCase();
    if (!hay.includes(q)) return false;
  }

  return true;
}

function renderGrid() {
  const rows = PRED.filter(keep);
  const grid = id("grid");
  grid.innerHTML = "";

  for (const r of rows) {
    const correct = String(r.correct).toLowerCase() === "true";
    const card = document.createElement("div");
    card.className = "card " + (correct ? "correct" : "error");

    const imgUrl = "/image?path=" + encodeURIComponent(r.crop_path);

    card.innerHTML = `
      <div class="badges">
        <span class="badge ${correct ? "correct" : "error"}">${correct ? "CORRECT" : "ERROR"}</span>
        <span class="badge">${r.split_policy}</span>
        <span class="badge">${r.crop_type}</span>
        <span class="badge">${r.model_type}</span>
      </div>

      <img src="${imgUrl}">

      <div class="meta">
        actual: <b>${r.actual}</b> &nbsp; predicted: <b>${r.predicted}</b><br>
        confidence-like score: ${Number(r.confidence_like_score).toFixed(5)}<br>
        scanframe: ${r.scan_frame_id}<br>
        object: ${r.canonical_gt_object_id}<br>
        video: ${r.video_id}
      </div>
    `;

    grid.appendChild(card);
  }
}

for (const el of ["policyFilter", "actualFilter", "predFilter", "statusFilter", "searchBox"]) {
  window.addEventListener("DOMContentLoaded", () => {
    id(el).addEventListener("input", renderGrid);
    id(el).addEventListener("change", renderGrid);
  });
}

loadAll();
