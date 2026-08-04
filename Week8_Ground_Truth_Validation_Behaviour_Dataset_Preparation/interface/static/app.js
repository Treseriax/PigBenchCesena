let clips = [];
let currentClip = null;

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const clipSelect = document.getElementById("clipSelect");
const clipSearch = document.getElementById("clipSearch");

const layerIds = ["showBoxes", "showPigId", "showColour", "showBehaviour", "showTimestamp", "showFlags"];

const colourMap = {
  blue: "#2b6cff",
  green: "#22aa55",
  cyan: "#00bcd4",
  red: "#ff3333",
  pink: "#ff4fd8",
  purple: "#9b59ff",
  unknown: "#aaaaaa",
  not_visible: "#777777",
  uncertain: "#ffcc00",
  unassigned: "#999999"
};

function getLayers() {
  const layers = {};
  for (const id of layerIds) {
    layers[id] = document.getElementById(id).checked;
  }
  return layers;
}

async function loadClips() {
  const res = await fetch("/api/clips");
  const data = await res.json();
  clips = data.clips;
  document.getElementById("statusBadge").textContent = `${data.clip_count} clips loaded`;
  renderClipOptions();
}

function renderClipOptions() {
  const q = clipSearch.value.toLowerCase();
  clipSelect.innerHTML = "";

  for (const c of clips) {
    const label = `${c.scan_frame_id} | ${c.split || "unmapped"} | ${c.behaviour_set || "-"} | objects=${c.object_count}`;
    if (q && !label.toLowerCase().includes(q)) continue;

    const opt = document.createElement("option");
    opt.value = c.scan_frame_id;
    opt.textContent = label;
    clipSelect.appendChild(opt);
  }

  if (!currentClip && clipSelect.options.length > 0) {
    clipSelect.selectedIndex = 0;
    loadClip(clipSelect.value);
  }
}

async function loadClip(scan) {
  const res = await fetch(`/api/clip/${encodeURIComponent(scan)}`);
  currentClip = await res.json();

  video.src = `/media/${encodeURIComponent(scan)}.mp4`;
  video.load();

  document.getElementById("scanInfo").textContent = currentClip.scan_frame_id;
  document.getElementById("videoInfo").textContent = currentClip.video_id || "-";
  document.getElementById("splitInfo").textContent = currentClip.split || "unmapped";
  document.getElementById("behaviourInfo").textContent = currentClip.behaviour_set || "-";
  document.getElementById("objectInfo").textContent = `${currentClip.objects.length}`;

  renderObjectTable();
  drawOverlay();
}

function renderObjectTable() {
  const tbody = document.querySelector("#objectTable tbody");
  tbody.innerHTML = "";

  for (const o of currentClip.objects || []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${o.final_box_id || "-"}</td>
      <td>${o.visual_marker_colour || "-"}</td>
      <td>${o.behaviour_pig_id || "-"}</td>
      <td>${o.behaviour_code || "-"}</td>
      <td>${o.identity_status || "-"}</td>
      <td>${(o.validation_flags || []).join("|")}</td>
    `;
    tbody.appendChild(tr);
  }
}

function resizeCanvasToVideo() {
  const rect = video.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
}

function labelTextForObject(o, layers) {
  const parts = [];
  if (layers.showPigId) parts.push(o.behaviour_pig_id || "pig?");
  if (layers.showColour) parts.push(o.visual_marker_colour || "colour?");
  if (layers.showBehaviour) parts.push(o.behaviour_code || "beh?");
  if (layers.showFlags && o.validation_flags && o.validation_flags.length) {
    parts.push(o.validation_flags.join("|"));
  }
  return parts.join(" / ");
}

function drawOverlay() {
  resizeCanvasToVideo();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!currentClip || !video.videoWidth || !video.videoHeight) {
    requestAnimationFrame(drawOverlay);
    return;
  }

  const layers = getLayers();
  const sx = canvas.width / video.videoWidth;
  const sy = canvas.height / video.videoHeight;

  if (layers.showBoxes) {
    for (const o of currentClip.objects || []) {
      const b = o.bbox_xyxy || [0, 0, 0, 0];
      const x1 = b[0] * sx;
      const y1 = b[1] * sy;
      const x2 = b[2] * sx;
      const y2 = b[3] * sy;
      const w = x2 - x1;
      const h = y2 - y1;

      const colour = colourMap[o.visual_marker_colour] || "#ffffff";
      ctx.strokeStyle = colour;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, w, h);

      const label = labelTextForObject(o, layers);
      if (label) {
        ctx.font = "14px Arial";
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(0,0,0,0.75)";
        ctx.fillRect(x1, Math.max(0, y1 - 22), textWidth + 10, 20);
        ctx.fillStyle = colour;
        ctx.fillText(label, x1 + 5, Math.max(14, y1 - 7));
      }
    }
  }

  const fps = currentClip.fps_used || 25;
  const estFrame = Math.floor(video.currentTime * fps);
  document.getElementById("timeInfo").textContent = `${video.currentTime.toFixed(2)} s`;
  document.getElementById("frameInfo").textContent = `${estFrame}`;

  if (layers.showTimestamp) {
    ctx.font = "16px Arial";
    const msg = `t=${video.currentTime.toFixed(2)}s | frame≈${estFrame}`;
    ctx.fillStyle = "rgba(0,0,0,0.75)";
    ctx.fillRect(10, 10, ctx.measureText(msg).width + 16, 26);
    ctx.fillStyle = "white";
    ctx.fillText(msg, 18, 29);
  }

  requestAnimationFrame(drawOverlay);
}

async function saveNote() {
  if (!currentClip) return;

  const fps = currentClip.fps_used || 25;
  const layers = getLayers();

  const payload = {
    scan_frame_id: currentClip.scan_frame_id,
    video_id: currentClip.video_id || "",
    video_current_time_sec: Number(video.currentTime.toFixed(3)),
    estimated_frame_index_in_clip: Math.floor(video.currentTime * fps),
    issue_type: document.getElementById("issueType").value,
    severity: document.getElementById("severity").value,
    note: document.getElementById("noteText").value,
    visible_layers: Object.entries(layers).filter(([k, v]) => v).map(([k]) => k).join("|")
  };

  const res = await fetch("/api/note", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  const status = document.getElementById("noteStatus");

  if (data.ok) {
    status.textContent = "Saved.";
    document.getElementById("noteText").value = "";
  } else {
    status.textContent = "Save failed.";
  }
}

clipSelect.addEventListener("change", () => loadClip(clipSelect.value));
clipSearch.addEventListener("input", renderClipOptions);
document.getElementById("saveNote").addEventListener("click", saveNote);

for (const id of layerIds) {
  document.getElementById(id).addEventListener("change", drawOverlay);
}

video.addEventListener("loadedmetadata", drawOverlay);
window.addEventListener("resize", drawOverlay);

loadClips();
drawOverlay();
