
let DATA = null;
let ASSIGNMENTS = [];
let CURRENT = null;
let selectedBoxId = null;

const colourOrder = ["green", "blue", "purple", "red_neck", "red_tail", "no_colour"];

function byId(id) { return document.getElementById(id); }

function clean(v) {
  if (v === null || v === undefined) return "";
  return String(v);
}

function assignmentKey(scan, colour) {
  return `${scan}||${colour}`;
}

function buildAssignmentMap() {
  const m = {};
  for (const r of ASSIGNMENTS) {
    m[assignmentKey(r.scan_frame_id, r.canonical_colour_label_norm)] = r;
  }
  return m;
}

async function loadAll() {
  const dataResp = await fetch("/api/data");
  DATA = await dataResp.json();

  const assResp = await fetch("/api/assignments");
  const assData = await assResp.json();
  ASSIGNMENTS = assData.assignments || [];

  renderSidebar();
  if (!CURRENT && DATA.scanframes.length) {
    selectScanframe(DATA.scanframes[0].scan_frame_id);
  } else if (CURRENT) {
    selectScanframe(CURRENT.scan_frame_id);
  }
}

function renderSidebar() {
  const filter = byId("filterStatus").value;
  const query = byId("searchBox").value.trim().toLowerCase();

  const list = byId("scanList");
  list.innerHTML = "";

  const counts = {};
  for (const sf of DATA.scanframes) {
    counts[sf.status] = (counts[sf.status] || 0) + 1;
  }

  byId("statusSummary").textContent =
    Object.entries(counts).map(([k, v]) => `${k}: ${v}`).join("\n");

  for (const sf of DATA.scanframes) {
    if (filter && sf.status !== filter) continue;
    if (query && !sf.scan_frame_id.toLowerCase().includes(query)) continue;

    const div = document.createElement("div");
    div.className = "scanItem";
    if (CURRENT && CURRENT.scan_frame_id === sf.scan_frame_id) div.classList.add("active");

    const boxCount = (sf.candidate_boxes || []).length;
    const targetCount = (sf.canonical_targets || []).length;

    div.innerHTML = `
      <div><b>${sf.scan_frame_id}</b></div>
      <div class="small">${sf.video_id}</div>
      <div class="badge">${sf.status}</div>
      <div class="small">targets=${targetCount}, boxes=${boxCount}</div>
    `;

    div.onclick = () => selectScanframe(sf.scan_frame_id);
    list.appendChild(div);
  }
}

function selectScanframe(scanId) {
  CURRENT = DATA.scanframes.find(x => x.scan_frame_id === scanId);
  selectedBoxId = null;

  byId("title").textContent = CURRENT.scan_frame_id;
  byId("subtitle").innerHTML = `
    ${CURRENT.video_id} — <span class="${CURRENT.status.startsWith("P3") ? "okText" : "warningText"}">${CURRENT.status}</span>
    — gate: ${CURRENT.classification_gate}
  `;

  const clipPath = CURRENT.clip_meta && CURRENT.clip_meta.clip_path ? CURRENT.clip_meta.clip_path : "";
  const video = byId("video");
  video.src = clipPath ? `/video?path=${encodeURIComponent(clipPath)}` : "";
  video.load();

  renderTargets();
  resizeCanvas();
  setTimeout(drawOverlay, 200);
  renderSidebar();
}

function renderTargets() {
  const container = byId("targetRows");
  container.innerHTML = "";

  const amap = buildAssignmentMap();

  const targets = [...(CURRENT.canonical_targets || [])].sort((a, b) => {
    return colourOrder.indexOf(a.canonical_colour_label_norm) - colourOrder.indexOf(b.canonical_colour_label_norm);
  });

  for (const t of targets) {
    const key = assignmentKey(t.scan_frame_id, t.canonical_colour_label_norm);
    const a = amap[key] || t;

    const card = document.createElement("div");
    card.className = "targetCard";

    const options = [`<option value="">-- no candidate selected --</option>`];
    for (const b of CURRENT.candidate_boxes || []) {
      const id = clean(b.candidate_box_id);
      const prev = clean(b.deprecated_previous_visual_marker_colour);
      options.push(`<option value="${id}" ${clean(a.manual_assigned_candidate_box_id) === id ? "selected" : ""}>${id} / old=${prev}</option>`);
    }

    card.innerHTML = `
      <div class="targetHeader">
        <div>
          <div class="colourName">${t.canonical_colour_label_norm}</div>
          <div class="small">raw: ${t.canonical_colour_label_raw}</div>
        </div>
        <div class="behaviour">behaviour: <b>${t.behaviour_code}</b></div>
      </div>

      <div class="targetGrid">
        <label>Candidate box</label>
        <select class="candidateBox">${options.join("")}</select>

        <label>BBox status</label>
        <select class="bboxStatus">
          ${optionList(["", "bbox_ok", "bbox_wrong", "bbox_missing", "bbox_extra_false_positive", "bbox_needs_manual_redraw", "bbox_uncertain"], a.manual_bbox_status)}
        </select>

        <label>Identity status</label>
        <select class="identityStatus">
          ${optionList(["", "identity_confirmed", "identity_uncertain", "identity_switch", "identity_not_visible", "identity_missing"], a.manual_identity_status)}
        </select>

        <label>GT v2 status</label>
        <select class="gtStatus">
          ${optionList(["", "gold_usable", "silver_usable_with_caution", "red_exclude", "fix_required", "unknown_pending_review"], a.manual_gt_v2_status)}
        </select>

        <label>Classification use</label>
        <select class="classificationUse">
          ${optionList(["", "use_for_classification", "use_for_classification_with_caution", "exclude_from_classification", "pending_review"], a.manual_classification_use)}
        </select>

        <label>Reviewer</label>
        <input class="reviewedBy" value="${escapeHtml(clean(a.manual_reviewed_by) || "oyavuz")}">

        <label>Note</label>
        <textarea class="reviewerNote">${escapeHtml(clean(a.manual_reviewer_note))}</textarea>
      </div>

      <div style="margin-top:8px;">
        <button class="saveBtn">Save assignment</button>
        <button class="focusBtn">Focus selected box</button>
        <span class="saveMsg"></span>
      </div>

      <div class="small" style="margin-top:6px;">
        Excel cell: row ${t.excel_behaviour_row_0based}, col ${t.excel_behaviour_col_0based}
      </div>
    `;

    const candidateSelect = card.querySelector(".candidateBox");
    candidateSelect.onchange = () => {
      selectedBoxId = candidateSelect.value;
      drawOverlay();
    };

    card.querySelector(".focusBtn").onclick = () => {
      selectedBoxId = candidateSelect.value;
      drawOverlay();
    };

    card.querySelector(".saveBtn").onclick = async () => {
      const payload = {
        scan_frame_id: t.scan_frame_id,
        canonical_colour_label_norm: t.canonical_colour_label_norm,
        manual_assigned_candidate_box_id: candidateSelect.value,
        manual_bbox_status: card.querySelector(".bboxStatus").value,
        manual_identity_status: card.querySelector(".identityStatus").value,
        manual_colour_status: "canonical_excel_colour_confirmed",
        manual_behaviour_status: "canonical_excel_behaviour_confirmed",
        manual_gt_v2_status: card.querySelector(".gtStatus").value,
        manual_classification_use: card.querySelector(".classificationUse").value,
        manual_review_priority: CURRENT.status,
        manual_reviewer_note: card.querySelector(".reviewerNote").value,
        manual_reviewed_by: card.querySelector(".reviewedBy").value
      };

      const resp = await fetch("/api/save_assignment", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });

      const msg = card.querySelector(".saveMsg");
      const out = await resp.json();

      if (out.ok) {
        msg.textContent = "saved";
        msg.className = "saveMsg okText";
        await refreshAssignmentsOnly();
      } else {
        msg.textContent = out.error || "save failed";
        msg.className = "saveMsg problemText";
      }
    };

    container.appendChild(card);
  }
}

function optionList(values, selected) {
  return values.map(v => `<option value="${v}" ${clean(selected) === v ? "selected" : ""}>${v || "--"}</option>`).join("");
}

function escapeHtml(s) {
  return clean(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function refreshAssignmentsOnly() {
  const assResp = await fetch("/api/assignments");
  const assData = await assResp.json();
  ASSIGNMENTS = assData.assignments || [];
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

  if (!CURRENT || !byId("showBoxes").checked) return;
  if (!video.videoWidth || !video.videoHeight) return;

  const sx = canvas.width / video.videoWidth;
  const sy = canvas.height / video.videoHeight;

  for (const b of CURRENT.candidate_boxes || []) {
    const x1 = Number(b.bbox_x1) * sx;
    const y1 = Number(b.bbox_y1) * sy;
    const x2 = Number(b.bbox_x2) * sx;
    const y2 = Number(b.bbox_y2) * sy;

    const id = clean(b.candidate_box_id);
    const selected = selectedBoxId && id === selectedBoxId;

    ctx.lineWidth = selected ? 4 : 2;
    ctx.strokeStyle = selected ? "yellow" : "lime";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    if (byId("showLabels").checked) {
      ctx.fillStyle = selected ? "yellow" : "lime";
      ctx.font = "14px Arial";
      ctx.fillText(id, x1 + 4, y1 + 16);
    }
  }
}

byId("filterStatus").onchange = renderSidebar;
byId("searchBox").oninput = renderSidebar;
byId("refreshBtn").onclick = loadAll;
byId("showBoxes").onchange = drawOverlay;
byId("showLabels").onchange = drawOverlay;

byId("video").addEventListener("loadedmetadata", drawOverlay);
byId("video").addEventListener("timeupdate", drawOverlay);
window.addEventListener("resize", drawOverlay);

document.querySelectorAll("#playButtons button").forEach(btn => {
  btn.onclick = () => {
    const t = Number(btn.dataset.seek);
    byId("video").currentTime = t;
    drawOverlay();
  };
});

loadAll();
