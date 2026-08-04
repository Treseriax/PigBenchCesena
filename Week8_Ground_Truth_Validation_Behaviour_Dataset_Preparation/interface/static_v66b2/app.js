
let DATA = null;
let CURRENT_SCAN = null;
let CURRENT_TARGET = null;
let drawing = false;
let startPoint = null;
let currentBox = null;

function byId(id) { return document.getElementById(id); }
function clean(v) { return v === null || v === undefined ? "" : String(v); }

async function loadData() {
  const resp = await fetch("/api/data");
  DATA = await resp.json();
  renderList();

  if (!CURRENT_TARGET && DATA.scanframes.length && DATA.scanframes[0].targets.length) {
    selectTarget(DATA.scanframes[0].scan_frame_id, DATA.scanframes[0].targets[0].canonical_gt_object_id);
  }
}

function renderList() {
  const list = byId("list");
  list.innerHTML = "";

  let targetCount = 0;
  for (const sf of DATA.scanframes) targetCount += sf.targets.length;

  byId("summary").textContent = `scanframes: ${DATA.scanframes.length}\ntargets: ${targetCount}`;

  for (const sf of DATA.scanframes) {
    for (const t of sf.targets) {
      const div = document.createElement("div");
      div.className = "item";
      if (CURRENT_TARGET && CURRENT_TARGET.canonical_gt_object_id === t.canonical_gt_object_id) div.classList.add("active");

      div.innerHTML = `
        <b>${t.canonical_gt_object_id}</b><br>
        <span class="small">${sf.video_id}</span><br>
        <span class="small">${t.behaviour_code} / ${t.manual_bbox_status} / ${t.manual_reviewer_note}</span>
      `;

      div.onclick = () => selectTarget(sf.scan_frame_id, t.canonical_gt_object_id);
      list.appendChild(div);
    }
  }
}

function selectTarget(scanId, objectId) {
  CURRENT_SCAN = DATA.scanframes.find(s => s.scan_frame_id === scanId);
  CURRENT_TARGET = CURRENT_SCAN.targets.find(t => t.canonical_gt_object_id === objectId);

  byId("title").textContent = CURRENT_TARGET.canonical_gt_object_id;
  byId("subtitle").textContent = `${CURRENT_SCAN.video_id} / ${CURRENT_TARGET.canonical_colour_label_norm} / ${CURRENT_TARGET.behaviour_code}`;

  const video = byId("video");
  video.src = CURRENT_SCAN.clip_path ? `/video?path=${encodeURIComponent(CURRENT_SCAN.clip_path)}` : "";
  video.load();

  byId("objTitle").textContent = `${CURRENT_TARGET.canonical_colour_label_norm} / ${CURRENT_TARGET.behaviour_code}`;
  byId("objInfo").innerHTML = `
    old candidate: ${CURRENT_TARGET.manual_assigned_candidate_box_id || "-"}<br>
    old bbox status: ${CURRENT_TARGET.manual_bbox_status}<br>
    gt status: ${CURRENT_TARGET.manual_gt_v2_status}<br>
    note: ${CURRENT_TARGET.manual_reviewer_note}
  `;

  setInputsFromTarget();
  renderList();
  setTimeout(drawOverlay, 250);
}

function setInputsFromTarget() {
  byId("x1").value = clean(CURRENT_TARGET.manual_bbox_x1);
  byId("y1").value = clean(CURRENT_TARGET.manual_bbox_y1);
  byId("x2").value = clean(CURRENT_TARGET.manual_bbox_x2);
  byId("y2").value = clean(CURRENT_TARGET.manual_bbox_y2);

  const vals = [Number(byId("x1").value), Number(byId("y1").value), Number(byId("x2").value), Number(byId("y2").value)];
  if (vals.every(v => Number.isFinite(v)) && vals[2] > vals[0] && vals[3] > vals[1]) {
    currentBox = vals;
  } else {
    currentBox = null;
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

  if (!video.videoWidth || !video.videoHeight) return;

  const sx = canvas.width / video.videoWidth;
  const sy = canvas.height / video.videoHeight;

  if (currentBox) {
    const [x1, y1, x2, y2] = currentBox;
    ctx.strokeStyle = "yellow";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    ctx.fillStyle = "yellow";
    ctx.font = "14px Arial";
    ctx.fillText(`${CURRENT_TARGET.canonical_colour_label_norm} / ${CURRENT_TARGET.behaviour_code}`, x1 * sx + 4, y1 * sy + 16);
  }
}

function canvasToVideoCoords(evt) {
  const video = byId("video");
  const canvas = byId("overlay");
  const rect = canvas.getBoundingClientRect();

  const cx = evt.clientX - rect.left;
  const cy = evt.clientY - rect.top;

  const vx = cx * video.videoWidth / canvas.width;
  const vy = cy * video.videoHeight / canvas.height;

  return [vx, vy];
}

byId("overlay").addEventListener("mousedown", evt => {
  if (!byId("video").videoWidth) return;
  drawing = true;
  startPoint = canvasToVideoCoords(evt);
});

byId("overlay").addEventListener("mousemove", evt => {
  if (!drawing || !startPoint) return;
  const p = canvasToVideoCoords(evt);
  const x1 = Math.min(startPoint[0], p[0]);
  const y1 = Math.min(startPoint[1], p[1]);
  const x2 = Math.max(startPoint[0], p[0]);
  const y2 = Math.max(startPoint[1], p[1]);
  currentBox = [x1, y1, x2, y2];
  updateInputs();
  drawOverlay();
});

byId("overlay").addEventListener("mouseup", evt => {
  drawing = false;
  startPoint = null;
});

function updateInputs() {
  if (!currentBox) return;
  byId("x1").value = currentBox[0].toFixed(3);
  byId("y1").value = currentBox[1].toFixed(3);
  byId("x2").value = currentBox[2].toFixed(3);
  byId("y2").value = currentBox[3].toFixed(3);
}

["x1", "y1", "x2", "y2"].forEach(id => {
  byId(id).oninput = () => {
    const vals = [Number(byId("x1").value), Number(byId("y1").value), Number(byId("x2").value), Number(byId("y2").value)];
    if (vals.every(v => Number.isFinite(v)) && vals[2] > vals[0] && vals[3] > vals[1]) {
      currentBox = vals;
      drawOverlay();
    }
  };
});

byId("saveBtn").onclick = async () => {
  const payload = {
    canonical_gt_object_id: CURRENT_TARGET.canonical_gt_object_id,
    manual_bbox_x1: byId("x1").value,
    manual_bbox_y1: byId("y1").value,
    manual_bbox_x2: byId("x2").value,
    manual_bbox_y2: byId("y2").value,
    manual_reviewer_note: byId("note").value || "BBox adjusted in v66b2"
  };

  const resp = await fetch("/api/save_bbox", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });

  const out = await resp.json();
  if (out.ok) {
    byId("saveMsg").textContent = "saved";
  } else {
    byId("saveMsg").textContent = out.error || "save failed";
  }
};

byId("refreshBtn").onclick = loadData;
byId("video").addEventListener("loadedmetadata", drawOverlay);
window.addEventListener("resize", drawOverlay);

loadData();
