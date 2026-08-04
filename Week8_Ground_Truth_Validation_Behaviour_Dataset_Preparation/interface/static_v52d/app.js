let clips = [];
let selectedClip = null;
let anchors = [];
let tracks = [];
let tracksByFrame = new Map();

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const colourMap = {
  blue: "#1e5bff",
  green: "#32e64b",
  cyan: "#00e5ff",
  red: "#ff2525",
  pink: "#ff3dd8",
  purple: "#a855ff",
  unknown: "#c8c8c8",
  not_visible: "#999999",
  uncertain: "#ffcc00",
  unassigned: "#aaaaaa"
};

function getColour(c) {
  c = (c || "unknown").toLowerCase();
  return colourMap[c] || "#ffffff";
}

function htmlEscape(s) {
  return String(s ?? "").replace(/[&<>"']/g, m => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[m]));
}

async function loadClips() {
  const res = await fetch("/api/clips");
  const data = await res.json();
  clips = data.clips || [];
  renderClipList();
}

function renderClipList() {
  const list = document.getElementById("clipList");
  const q = document.getElementById("clipSearch").value.toLowerCase();
  const filter = document.getElementById("qualityFilter").value;

  list.innerHTML = "";

  clips
    .filter(c => filter === "all" || c.quality_tier === filter)
    .filter(c => {
      const s = `${c.scan_frame_id} ${c.video_id}`.toLowerCase();
      return s.includes(q);
    })
    .forEach(c => {
      const div = document.createElement("div");
      div.className = "clipItem " + (c.quality_tier || "");
      div.innerHTML = `
        <b>${htmlEscape(c.scan_frame_id)}</b>
        <span>${htmlEscape(c.video_id)}</span>
        <small>${htmlEscape(c.quality_tier)} | draw ${Number(c.draw_ok_ratio).toFixed(2)} | review ${Number(c.review_needed_ratio).toFixed(2)}</small>
      `;
      div.onclick = () => selectClip(c.scan_frame_id);
      list.appendChild(div);
    });
}

async function selectClip(scan) {
  const clipRes = await fetch(`/api/clip?scan=${encodeURIComponent(scan)}`);
  const clipData = await clipRes.json();

  const annRes = await fetch(`/api/annotations?scan=${encodeURIComponent(scan)}`);
  const annData = await annRes.json();

  selectedClip = clipData.clip;
  selectedClip.qa = clipData.qa || {};
  anchors = annData.anchors || [];
  tracks = annData.tracks || [];

  tracksByFrame = new Map();
  for (const t of tracks) {
    const f = Number(t.frame_index_in_clip || 0);
    if (!tracksByFrame.has(f)) tracksByFrame.set(f, []);
    tracksByFrame.get(f).push(t);
  }

  document.getElementById("clipTitle").textContent = `${scan} — ${selectedClip.video_id || ""}`;
  document.getElementById("clipStats").innerHTML = `
    <b>quality:</b> ${htmlEscape(selectedClip.qa.quality_tier || "")}
    | <b>draw:</b> ${Number(selectedClip.qa.draw_ok_ratio || 0).toFixed(3)}
    | <b>stable:</b> ${Number(selectedClip.qa.stable_tracklet_ratio || 0).toFixed(3)}
    | <b>fallback:</b> ${Number(selectedClip.qa.recall_fallback_ratio || 0).toFixed(3)}
    | <b>missing:</b> ${Number(selectedClip.qa.missing_ratio || 0).toFixed(3)}
    | <b>review:</b> ${Number(selectedClip.qa.review_needed_ratio || 0).toFixed(3)}
  `;

  video.src = `/video?scan=${encodeURIComponent(scan)}`;
  video.load();
}

function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
}

function frameIndex() {
  const fps = Number(selectedClip?.fps_used || 25);
  return Math.round(video.currentTime * fps);
}

function scaleBox(box) {
  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;
  const sx = canvas.width / vw;
  const sy = canvas.height / vh;
  return {
    x1: box.x1 * sx,
    y1: box.y1 * sy,
    x2: box.x2 * sx,
    y2: box.y2 * sy
  };
}

function drawBox(box, colour, lineWidth, dashed, label) {
  const b = scaleBox(box);
  ctx.save();
  ctx.strokeStyle = colour;
  ctx.lineWidth = lineWidth;
  if (dashed) ctx.setLineDash([7, 5]);
  ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

  if (document.getElementById("showLabels").checked && label) {
    ctx.font = "13px Arial";
    ctx.fillStyle = colour;
    ctx.fillText(label, b.x1 + 3, Math.max(14, b.y1 - 4));
  }
  ctx.restore();
}

function makeLabel(r, source) {
  const parts = [];
  if (document.getElementById("showColour").checked) parts.push(r.visual_marker_colour || "unknown");
  if (document.getElementById("showBehaviour").checked) parts.push(r.behaviour_code || "");
  parts.push(source);
  return `${r.behaviour_pig_id || r.final_box_id || ""}/${parts.filter(Boolean).join("/")}`;
}

function drawOverlay() {
  resizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!selectedClip || !video.videoWidth) {
    requestAnimationFrame(drawOverlay);
    return;
  }

  const f = frameIndex();
  const current = tracksByFrame.get(f) || [];

  let stableCount = 0;
  let fallbackCount = 0;
  let missingCount = 0;
  let reviewCount = 0;

  if (document.getElementById("showAnchor").checked) {
    for (const a of anchors) {
      drawBox(a, "#ffffff", 1.5, true, makeLabel(a, "anchor"));
    }
  }

  for (const r of current) {
    if (r.hybrid_source === "stable_tracklet") stableCount++;
    if (r.hybrid_source === "recall_fallback") fallbackCount++;
    if (r.hybrid_source === "missing") missingCount++;
    if (r.review_needed) reviewCount++;

    if (!r.draw_ok) continue;

    const colour = getColour(r.visual_marker_colour);
    if (r.hybrid_source === "stable_tracklet" && document.getElementById("showStable").checked) {
      drawBox(r, colour, 3, false, makeLabel(r, "stable"));
    }
    if (r.hybrid_source === "recall_fallback" && document.getElementById("showFallback").checked) {
      drawBox(r, colour, 1.5, true, makeLabel(r, "fallback"));
    }
  }

  if (document.getElementById("showReview").checked) {
    document.getElementById("frameInfo").innerHTML = `
      <b>time:</b> ${video.currentTime.toFixed(2)}s
      | <b>frame:</b> ${f}
      | <b>stable:</b> ${stableCount}
      | <b>fallback:</b> ${fallbackCount}
      | <b>missing:</b> ${missingCount}
      | <b>review:</b> ${reviewCount}
    `;
  } else {
    document.getElementById("frameInfo").innerHTML = `<b>time:</b> ${video.currentTime.toFixed(2)}s | <b>frame:</b> ${f}`;
  }

  requestAnimationFrame(drawOverlay);
}

async function saveNote() {
  if (!selectedClip) return;

  const payload = {
    scan_frame_id: selectedClip.scan_frame_id,
    video_id: selectedClip.video_id || "",
    time_sec: video.currentTime.toFixed(3),
    frame_index_in_clip: frameIndex(),
    issue_type: document.getElementById("issueType").value,
    severity: document.getElementById("severity").value,
    note: document.getElementById("noteText").value
  };

  const res = await fetch("/api/save_note", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  document.getElementById("saveStatus").textContent = data.ok ? `Saved: ${data.saved_to}` : `Error: ${data.error}`;
  if (data.ok) document.getElementById("noteText").value = "";
}

document.getElementById("clipSearch").addEventListener("input", renderClipList);
document.getElementById("qualityFilter").addEventListener("change", renderClipList);
document.getElementById("saveNote").addEventListener("click", saveNote);

for (const id of ["showAnchor", "showStable", "showFallback", "showReview", "showLabels", "showBehaviour", "showColour"]) {
  document.getElementById(id).addEventListener("change", drawOverlay);
}

video.addEventListener("loadedmetadata", resizeCanvas);
window.addEventListener("resize", resizeCanvas);

loadClips();
drawOverlay();
