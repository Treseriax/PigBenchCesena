
let DATA = null;
let CURRENT = null;

function byId(id) { return document.getElementById(id); }

function clean(v) {
  if (v === null || v === undefined) return "";
  return String(v);
}

function catClass(cat) {
  if (cat === "strict_gold_classification") return "strict";
  if (cat === "caution_analysis") return "caution";
  return "nonusable";
}

async function loadData() {
  const resp = await fetch("/api/data");
  DATA = await resp.json();

  populateBehaviourFilter();
  renderSidebar();

  if (!CURRENT && DATA.scanframes.length) {
    selectScanframe(DATA.scanframes[0].scan_frame_id);
  }
}

function populateBehaviourFilter() {
  const set = new Set();
  for (const sf of DATA.scanframes || []) {
    for (const o of sf.objects || []) {
      if (o.behaviour_code) set.add(o.behaviour_code);
    }
  }

  const sel = byId("behaviourFilter");
  const current = sel.value;
  sel.innerHTML = '<option value="">All</option>';
  [...set].sort().forEach(b => {
    const opt = document.createElement("option");
    opt.value = b;
    opt.textContent = b;
    sel.appendChild(opt);
  });
  sel.value = current;
}

function scanframeMatches(sf) {
  const cat = byId("categoryFilter").value;
  const beh = byId("behaviourFilter").value;
  const q = byId("searchBox").value.trim().toLowerCase();
  const onlyProblems = byId("onlyProblemClips").checked;

  if (q && !sf.scan_frame_id.toLowerCase().includes(q)) return false;

  const objs = sf.objects || [];

  if (cat && !objs.some(o => o.final_gt_v2_category === cat)) return false;
  if (beh && !objs.some(o => o.behaviour_code === beh)) return false;

  if (onlyProblems) {
    const hasProblem = objs.some(o => o.final_gt_v2_category !== "strict_gold_classification");
    if (!hasProblem) return false;
  }

  return true;
}

function renderSidebar() {
  const list = byId("scanList");
  list.innerHTML = "";

  let counts = {
    scanframes: 0,
    strict_objects: 0,
    caution_objects: 0,
    nonusable_objects: 0,
  };

  for (const sf of DATA.scanframes || []) {
    if (!scanframeMatches(sf)) continue;

    counts.scanframes += 1;
    const objs = sf.objects || [];
    counts.strict_objects += objs.filter(o => o.final_gt_v2_category === "strict_gold_classification").length;
    counts.caution_objects += objs.filter(o => o.final_gt_v2_category === "caution_analysis").length;
    counts.nonusable_objects += objs.filter(o => o.final_gt_v2_category === "nonusable_fix_or_excluded").length;

    const div = document.createElement("div");
    div.className = "scanItem";
    if (CURRENT && CURRENT.scan_frame_id === sf.scan_frame_id) div.classList.add("active");

    const strict = objs.filter(o => o.final_gt_v2_category === "strict_gold_classification").length;
    const caution = objs.filter(o => o.final_gt_v2_category === "caution_analysis").length;
    const nonusable = objs.filter(o => o.final_gt_v2_category === "nonusable_fix_or_excluded").length;

    div.innerHTML = `
      <div><b>${sf.scan_frame_id}</b></div>
      <div class="small">${sf.video_id}</div>
      <span class="badge strict">strict=${strict}</span>
      <span class="badge caution">caution=${caution}</span>
      <span class="badge nonusable">nonusable=${nonusable}</span>
    `;

    div.onclick = () => selectScanframe(sf.scan_frame_id);
    list.appendChild(div);
  }

  byId("summary").textContent =
    `scanframes: ${counts.scanframes}\n` +
    `strict objects: ${counts.strict_objects}\n` +
    `caution objects: ${counts.caution_objects}\n` +
    `nonusable objects: ${counts.nonusable_objects}`;
}

function selectScanframe(scanId) {
  CURRENT = DATA.scanframes.find(x => x.scan_frame_id === scanId);

  byId("title").textContent = CURRENT.scan_frame_id;
  byId("subtitle").textContent = CURRENT.video_id;

  const video = byId("video");
  video.src = CURRENT.clip_path ? `/video?path=${encodeURIComponent(CURRENT.clip_path)}` : "";
  video.load();

  renderObjects();
  renderSidebar();
  setTimeout(drawOverlay, 250);
}

function visibleObject(o) {
  if (o.final_gt_v2_category === "strict_gold_classification" && !byId("showStrict").checked) return false;
  if (o.final_gt_v2_category === "caution_analysis" && !byId("showCaution").checked) return false;
  if (o.final_gt_v2_category === "nonusable_fix_or_excluded" && !byId("showNonusable").checked) return false;
  return true;
}

function renderObjects() {
  const box = byId("objects");
  box.innerHTML = "";

  const objs = CURRENT.objects || [];

  for (const o of objs) {
    const cls = catClass(o.final_gt_v2_category);
    const div = document.createElement("div");
    div.className = "objCard";

    div.innerHTML = `
      <div class="objHeader">
        <div>
          <b class="${cls}">${o.canonical_colour_label_norm}</b>
          <span> / behaviour: <b>${o.behaviour_code}</b></span>
        </div>
        <div class="${cls}">${o.final_gt_v2_category}</div>
      </div>
      <div class="kv">
        object: ${o.canonical_gt_object_id}<br>
        candidate box: ${o.manual_assigned_candidate_box_id}<br>
        bbox status: ${o.manual_bbox_status}<br>
        identity status: ${o.manual_identity_status}<br>
        gt status: ${o.manual_gt_v2_status}<br>
        classification gate: ${o.classification_gate}<br>
        split: ${o.recommended_split || "-"}<br>
        note: ${o.reviewer_note || "-"}
      </div>
    `;

    box.appendChild(div);
  }
}

function resizeCanvas() {
  const video = byId("video");
  const canvas = byId("overlay");
  const rect = video.getBoundingClientRect();

  canvas.width = rect.width;
  canvas.height = rect.height;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
}

function drawOverlay() {
  const video = byId("video");
  const canvas = byId("overlay");
  const ctx = canvas.getContext("2d");

  resizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!CURRENT || !video.videoWidth || !video.videoHeight) return;

  const sx = canvas.width / video.videoWidth;
  const sy = canvas.height / video.videoHeight;

  for (const o of CURRENT.objects || []) {
    if (!visibleObject(o)) continue;
    if (!o.has_valid_bbox || !o.bbox) continue;

    const [x1, y1, x2, y2] = o.bbox;
    const cls = catClass(o.final_gt_v2_category);

    if (cls === "strict") ctx.strokeStyle = "lime";
    else if (cls === "caution") ctx.strokeStyle = "yellow";
    else ctx.strokeStyle = "red";

    ctx.lineWidth = cls === "strict" ? 2 : 3;
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);

    if (byId("showLabels").checked) {
      ctx.fillStyle = ctx.strokeStyle;
      ctx.font = "14px Arial";
      const label = `${o.canonical_colour_label_norm} / ${o.behaviour_code}`;
      ctx.fillText(label, x1 * sx + 4, y1 * sy + 16);
    }
  }
}

byId("categoryFilter").onchange = renderSidebar;
byId("behaviourFilter").onchange = renderSidebar;
byId("searchBox").oninput = renderSidebar;
byId("onlyProblemClips").onchange = renderSidebar;
byId("refreshBtn").onclick = loadData;

for (const id of ["showStrict", "showCaution", "showNonusable", "showLabels"]) {
  byId(id).onchange = drawOverlay;
}

byId("video").addEventListener("loadedmetadata", drawOverlay);
byId("video").addEventListener("timeupdate", drawOverlay);
window.addEventListener("resize", drawOverlay);

document.querySelectorAll("#seekButtons button").forEach(btn => {
  btn.onclick = () => {
    byId("video").currentTime = Number(btn.dataset.seek);
    drawOverlay();
  };
});

loadData();
