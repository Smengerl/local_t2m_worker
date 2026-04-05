"""
FastAPI web server — REST API + browser frontend for the batch queue.

Endpoints:
  GET  /              → HTML dashboard (single-page app)
  GET  /api/jobs      → list all jobs
  GET  /api/jobs/{id} → single job details
  POST /api/jobs      → enqueue a new job
  DELETE /api/jobs/{id} → delete a job (only pending jobs)
  POST /api/jobs/{id}/retry → re-enqueue a failed job
  POST /api/clear-finished  → remove all done/failed jobs
  GET  /api/stats     → counts per status
  GET  /outputs/{filename}  → serve generated images

Usage:
    python -m batch.server              # default: http://localhost:8000
    python -m batch.server --port 9000
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from batch.queue import (
    clear_finished,
    delete_job,
    enqueue,
    get_job,
    list_jobs,
    stats,
    update_job,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Image Generation Queue", version="1.0")

_OUTPUTS_DIR = _ROOT / "outputs"
_OUTPUTS_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(_OUTPUTS_DIR)), name="outputs")


# ── Request / response models ─────────────────────────────────────────────────

class EnqueueRequest(BaseModel):
    config: str = "configs/sd15_default.json"
    prompt: str
    negative_prompt: str = ""
    output: Optional[str] = None
    model_id: Optional[str] = None
    adapter_id: Optional[str] = None
    lora_id: Optional[str] = None
    lora_scale: Optional[float] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/stats")
def api_stats() -> dict[str, int]:
    return stats()


@app.get("/api/jobs")
def api_list_jobs() -> list[dict[str, Any]]:
    return list_jobs()


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/jobs", status_code=201)
def api_enqueue(req: EnqueueRequest) -> dict[str, Any]:
    job = enqueue(
        config=req.config,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        output=req.output,
        model_id=req.model_id,
        adapter_id=req.adapter_id,
        lora_id=req.lora_id,
        lora_scale=req.lora_scale,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
    )
    return job


@app.delete("/api/jobs/{job_id}")
def api_delete_job(job_id: str) -> dict[str, str]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running job")
    delete_job(job_id)
    return {"deleted": job_id}


@app.post("/api/jobs/{job_id}/retry")
def api_retry_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("failed", "done"):
        raise HTTPException(status_code=409, detail="Only failed or done jobs can be retried")
    updated = update_job(
        job_id,
        status="pending",
        started_at=None,
        finished_at=None,
        result_path=None,
        error=None,
    )
    return updated


@app.post("/api/clear-finished")
def api_clear_finished() -> dict[str, int]:
    removed = clear_finished()
    return {"removed": removed}


# ── Config discovery ──────────────────────────────────────────────────────────

_CONFIGS_DIR = _ROOT / "configs"

def _load_config_list() -> list[dict[str, str]]:
    """Return all JSON files in configs/ as [{value, label, description}], sorted by filename."""
    entries: list[dict[str, str]] = []
    for path in sorted(_CONFIGS_DIR.glob("*.json")):
        try:
            import json as _json
            data = _json.loads(path.read_text())
            desc = data.get("description") or path.stem
            pipeline = data.get("pipeline_type", "")
            label = f"[{pipeline.upper()}] {desc}" if pipeline else desc
        except Exception:
            desc = path.stem
            label = path.stem
        entries.append({"value": f"configs/{path.name}", "label": label, "description": desc})
    return entries


@app.get("/api/configs")
def api_configs() -> list[dict[str, str]]:
    """List all available config files with human-readable labels."""
    return _load_config_list()


# ── HTML frontend ─────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Generation Queue</title>
<style>
  :root {
    --bg: #0f1117; --card: #1a1d27; --border: #2a2d3e;
    --accent: #7c6af7; --green: #22c55e; --red: #ef4444;
    --yellow: #eab308; --blue: #3b82f6; --text: #e2e8f0; --muted: #64748b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif;
         font-size: 14px; padding: 24px; }
  h1 { font-size: 1.4rem; font-weight: 700; margin-bottom: 20px; color: var(--accent); }
  h2 { font-size: 1rem; font-weight: 600; margin-bottom: 12px; }

  /* Stats bar */
  .stats { display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }
  .stat { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
          padding: 10px 18px; display: flex; flex-direction: column; align-items: center; }
  .stat .num { font-size: 1.6rem; font-weight: 700; }
  .stat .lbl { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; }
  .stat.pending .num { color: var(--yellow); }
  .stat.running .num { color: var(--blue); }
  .stat.done    .num { color: var(--green); }
  .stat.failed  .num { color: var(--red); }

  /* Layout: form left, queue right */
  .layout { display: grid; grid-template-columns: 340px 1fr; gap: 20px; }
  .layout > * { min-width: 0; }
  @media(max-width: 800px) { .layout { grid-template-columns: 1fr; } }

  /* Card */
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
          padding: 18px; }

  /* Form */
  label { display: block; font-size: 0.75rem; color: var(--muted); margin-bottom: 4px;
          text-transform: uppercase; letter-spacing: .04em; }
  input, select, textarea {
    width: 100%; background: var(--bg); border: 1px solid var(--border);
    border-radius: 6px; color: var(--text); padding: 7px 10px; font-size: 13px;
    margin-bottom: 12px; font-family: inherit;
  }
  textarea { resize: vertical; min-height: 64px; }
  input:focus, select:focus, textarea:focus { outline: 2px solid var(--accent); }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }

  /* Buttons */
  button {
    cursor: pointer; border: none; border-radius: 6px;
    padding: 8px 16px; font-size: 13px; font-weight: 600; font-family: inherit;
  }
  .btn-primary { background: var(--accent); color: #fff; width: 100%; padding: 10px; }
  .btn-primary:hover { filter: brightness(1.15); }
  .btn-sm { padding: 4px 10px; font-size: 11px; }
  .btn-danger { background: #3f1212; color: var(--red); }
  .btn-retry  { background: #1a2a3f; color: var(--blue); }
  .btn-clear  { background: #1f1f1f; color: var(--muted); font-size: 12px;
                margin-bottom: 12px; }

  /* Queue table */
  .queue-header { display: flex; justify-content: space-between; align-items: center;
                  margin-bottom: 12px; }
  .job { border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px;
         margin-bottom: 8px; cursor: pointer; transition: border-color .15s;
         overflow: hidden; min-width: 0; }
  .job:hover { border-color: var(--accent); }
  .job-top { display: flex; align-items: center; gap: 10px; }
  .badge { border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 700;
           text-transform: uppercase; flex-shrink: 0; }
  .badge.pending { background:#3a2e00; color: var(--yellow); }
  .badge.running { background:#0d1f3c; color: var(--blue); }
  .badge.done    { background:#052015; color: var(--green); }
  .badge.failed  { background:#2a0808; color: var(--red); }
  .job-prompt { flex: 1; min-width: 0; overflow: hidden; white-space: nowrap;
                text-overflow: ellipsis; font-weight: 500; }
  .job-cfg    { font-size: 11px; color: var(--muted); margin-left: auto; flex-shrink: 0; }
  .job-meta   { font-size: 11px; color: var(--muted); margin-top: 6px; }
  .job-actions { margin-top: 8px; display: flex; gap: 6px; }

  /* Progress bar — only visible while running */
  .job-progress { margin-top: 10px; display: none; }
  .job-progress.visible { display: block; }
  .job-progress-labels { display: flex; justify-content: space-between;
                         font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .job-progress progress { width: 100%; height: 6px; border-radius: 3px;
                            accent-color: var(--accent); }

  /* Detail panel */
  .detail { margin-top: 10px; }
  .detail pre { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
                padding: 10px; font-size: 11px; overflow-x: auto; white-space: pre-wrap;
                word-break: break-all; }
  .detail-prompt { font-size: 13px; font-weight: 500; line-height: 1.5;
                   white-space: pre-wrap; word-break: break-word;
                   margin-bottom: 10px; color: var(--text); }
  .detail-table { width: 100%; border-collapse: collapse; font-size: 12px;
                  color: var(--muted); margin-top: 6px; }
  .detail-table td { padding: 3px 6px 3px 0; vertical-align: top; }
  .detail-key { color: var(--muted); white-space: nowrap; padding-right: 12px;
                font-size: 11px; text-transform: uppercase; letter-spacing: .04em;
                width: 1%; }
  .detail-table td:last-child { color: var(--text); word-break: break-word; }
  .detail img { max-width: 100%; border-radius: 6px; margin-top: 10px;
                border: 1px solid var(--border); }

  /* Log panel */
  .job-log { margin-top: 12px; }
  .job-log-header { font-size: 11px; text-transform: uppercase; letter-spacing: .04em;
                    color: var(--muted); margin-bottom: 4px; }
  .job-log-body { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
                  padding: 8px 10px; font-family: monospace; font-size: 11px; line-height: 1.55;
                  max-height: 260px; overflow-y: auto; white-space: pre-wrap;
                  word-break: break-all; color: #c8d0da; }
  .job-thumb {
    width: 64px; height: 64px; object-fit: cover; border-radius: 6px; flex-shrink: 0;
    border: 1px solid var(--border); cursor: zoom-in; margin-left: auto;
  }

  /* Lightbox */
  #lightbox {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,.85);
    z-index: 1000; align-items: center; justify-content: center; cursor: zoom-out;
  }
  #lightbox.open { display: flex; }
  #lightbox img {
    max-width: 92vw; max-height: 92vh; border-radius: 10px;
    box-shadow: 0 8px 40px rgba(0,0,0,.8); cursor: default;
  }

  /* Empty state */
  .empty { text-align: center; color: var(--muted); padding: 40px; }

  /* Spinner */
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border);
             border-top-color: var(--accent); border-radius: 50%;
             animation: spin .7s linear infinite; vertical-align: middle; }
</style>
</head>
<body>
<h1>🖼 Image Generation Queue</h1>

<!-- Stats -->
<div class="stats" id="stats">
  <div class="stat pending"><span class="num" id="s-pending">–</span><span class="lbl">Pending</span></div>
  <div class="stat running"><span class="num" id="s-running">–</span><span class="lbl">Running</span></div>
  <div class="stat done"   ><span class="num" id="s-done">–</span><span class="lbl">Done</span></div>
  <div class="stat failed" ><span class="num" id="s-failed">–</span><span class="lbl">Failed</span></div>
</div>

<div class="layout">

  <!-- ── Enqueue form ── -->
  <div class="card">
    <h2>New Job</h2>
    <form id="enqueue-form">
      <label>Prompt *</label>
      <textarea id="f-prompt" required placeholder="a sunset over mountains"></textarea>

      <label>Negative Prompt</label>
      <input id="f-negative" type="text" placeholder="blurry, low quality">

      <label>Config</label>
      <select id="f-config">
        <option value="">Loading configs…</option>
      </select>
      <div id="f-config-hint" style="font-size:11px; color:var(--muted); margin-top:-8px; margin-bottom:12px; line-height:1.4; min-height:16px;"></div>

      <details style="margin-bottom:12px">
        <summary style="cursor:pointer;color:var(--muted);font-size:12px;margin-bottom:8px">
          Advanced overrides
        </summary>
        <div style="margin-top:8px">
          <label>Model ID override</label>
          <input id="f-model-id" type="text" placeholder="(from config)">
          <label>LoRA ID override</label>
          <input id="f-lora-id" type="text" placeholder="(from config)">
          <div class="row">
            <div>
              <label>Steps</label>
              <input id="f-steps" type="number" min="1" placeholder="(from config)">
            </div>
            <div>
              <label>Guidance scale</label>
              <input id="f-guidance" type="number" step="0.5" placeholder="(from config)">
            </div>
          </div>
          <div class="row">
            <div>
              <label>LoRA scale</label>
              <input id="f-lora-scale" type="number" step="0.05" min="0" max="1" placeholder="(from config)">
            </div>
            <div>
              <label>Output path</label>
              <input id="f-output" type="text" placeholder="(auto)">
            </div>
          </div>
        </div>
      </details>

      <button type="submit" class="btn-primary">➕ Add to Queue</button>
      <div id="form-msg" style="margin-top:8px;font-size:12px;min-height:18px"></div>
    </form>
  </div>

  <!-- ── Queue list ── -->
  <div class="card">
    <div class="queue-header">
      <h2>Queue</h2>
      <button class="btn-clear" onclick="clearFinished()">🗑 Clear finished</button>
    </div>
    <div id="job-list"><div class="empty">Loading…</div></div>
  </div>

</div>

<script>
const $ = id => document.getElementById(id);

// ── Helpers ────────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const r = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

function badge(status) {
  return `<span class="badge ${status}">${status}</span>`;
}

function relTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  const diff = Math.round((Date.now() - d) / 1000);
  if (diff < 60)   return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
  return d.toLocaleTimeString();
}

function durStr(startIso, endIso) {
  if (!startIso || !endIso) return null;
  const secs = Math.round((new Date(endIso) - new Date(startIso)) / 1000);
  if (secs < 60) return `${secs}s`;
  return `${Math.floor(secs / 60)}m ${secs % 60}s`;
}

// ── Stats ──────────────────────────────────────────────────────────────────
async function refreshStats() {
  try {
    const s = await apiFetch('/api/stats');
    $('s-pending').textContent = s.pending;
    $('s-running').textContent = s.running;
    $('s-done').textContent    = s.done;
    $('s-failed').textContent  = s.failed;
  } catch (e) { /* ignore stats errors silently */ }
}

// ── Job list ───────────────────────────────────────────────────────────────
let expandedId = null;

async function refreshJobs() {
  let jobs;
  try {
    jobs = await apiFetch('/api/jobs');
  } catch (e) {
    $('job-list').innerHTML = `<div class="empty" style="color:var(--red)">⚠ Server nicht erreichbar: ${escHtml(String(e.message))}</div>`;
    return;
  }
  const list  = $('job-list');

  if (jobs.length === 0) {
    list.innerHTML = '<div class="empty">Queue is empty.</div>';
    return;
  }

  // Render newest-first (reverse), but keep the FIFO logic in the backend
  const ordered = [...jobs].reverse();

  list.innerHTML = ordered.map(j => {
    const cfg = j.config.replace('configs/', '').replace('.json', '');
    const expanded = j.id === expandedId;
    const actions = [];

    if (j.status === 'pending') {
      actions.push(`<button class="btn-sm btn-danger" onclick="deleteJob('${j.id}',event)">Delete</button>`);
    }
    if (j.status === 'failed' || j.status === 'done') {
      actions.push(`<button class="btn-sm btn-retry" onclick="retryJob('${j.id}',event)">Re-run</button>`);
    }
    if (j.status === 'running') {
      actions.push(`<span class="spinner"></span>`);
    }

    // ── progress bar (only while running and total is known) ──────────────
    const step  = j.progress_step  || 0;
    const total = j.progress_total || 0;
    const pct   = total > 0 ? Math.round(step / total * 100) : 0;
    const showProgress = j.status === 'running' && total > 0;
    const progressBlock = `
      <div class="job-progress${showProgress ? ' visible' : ''}">
        <div class="job-progress-labels">
          <span>Generating…</span>
          <span>${step} / ${total} steps&nbsp;(${pct}%)</span>
        </div>
        <progress value="${step}" max="${Math.max(total, 1)}"></progress>
      </div>`;

    let detail = '';
    if (expanded) {
      const dur = durStr(j.started_at, j.finished_at);
      const rows = [
        ['Config',    escHtml(j.config.replace('configs/','').replace('.json',''))],
        ['Steps',     j.steps ?? '—'],
        ['Guidance',  j.guidance_scale ?? '—'],
        j.negative_prompt ? ['Negative', escHtml(j.negative_prompt)] : null,
        ['Added',     j.added_at  ? new Date(j.added_at).toLocaleString()  : '—'],
        j.started_at  ? ['Started',   new Date(j.started_at).toLocaleString()]  : null,
        j.finished_at ? ['Finished',  new Date(j.finished_at).toLocaleString()] : null,
        dur           ? ['Duration',  dur] : null,
      ].filter(Boolean);

      const tableRows = rows.map(([k,v]) =>
        `<tr><td class="detail-key">${k}</td><td>${v}</td></tr>`).join('');

      const errorBlock = j.error
        ? `<pre style="color:var(--red);margin-top:10px">${escHtml(j.error)}</pre>`
        : '';

      const logLines = j.log_lines ?? [];
      const logBlock = logLines.length > 0 ? `
        <div class="job-log">
          <div class="job-log-header">Log</div>
          <div class="job-log-body" id="log-${j.id}">${escHtml(logLines.join('\\n'))}</div>
        </div>` : '';

      detail = `
        <div class="detail">
          <div class="detail-prompt">${escHtml(j.prompt)}</div>
          <table class="detail-table">${tableRows}</table>
          ${logBlock}
          ${errorBlock}
        </div>`;
    }

    const thumbSrc = (j.status === 'done' && j.result_path)
      ? `/outputs/${j.result_path.split('/').pop()}`
      : null;
    const inlineThumb = thumbSrc
      ? `<img class="job-thumb" src="${thumbSrc}" alt="result"
              onclick="event.stopPropagation(); openLightbox('${thumbSrc}')">`
      : '';

    return `
      <div class="job" id="job-${j.id}" onclick="toggleExpand('${j.id}')">
        <div class="job-top">
          ${badge(j.status)}
          <span class="job-prompt">${escHtml(j.prompt)}</span>
          <span class="job-cfg">${escHtml(cfg)}</span>
          ${inlineThumb}
        </div>
        <div class="job-meta">${j.finished_at ? 'finished ' + relTime(j.finished_at) : 'added ' + relTime(j.added_at)}${j.status === 'done' && durStr(j.started_at, j.finished_at) ? '  ·  ⏱ ' + durStr(j.started_at, j.finished_at) : ''}</div>
        <div class="job-actions">${actions.join('')}</div>
        ${progressBlock}
        ${detail}
      </div>`;
  }).join('');

  // Auto-scroll all visible log panels to the bottom (live tail)
  document.querySelectorAll('.job-log-body').forEach(el => {
    el.scrollTop = el.scrollHeight;
  });
}

function escHtml(s) {
  return String(s ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

function toggleExpand(id) {
  expandedId = expandedId === id ? null : id;
  refreshJobs();
}

// ── Actions ────────────────────────────────────────────────────────────────
async function deleteJob(id, evt) {
  evt.stopPropagation();
  await apiFetch(`/api/jobs/${id}`, { method: 'DELETE' });
  if (expandedId === id) expandedId = null;
  refresh();
}

async function retryJob(id, evt) {
  evt.stopPropagation();
  await apiFetch(`/api/jobs/${id}/retry`, { method: 'POST' });
  refresh();
}

async function clearFinished() {
  await apiFetch('/api/clear-finished', { method: 'POST' });
  expandedId = null;
  refresh();
}

// ── Config dropdown (dynamic) ──────────────────────────────────────────────
async function loadConfigs() {
  try {
    const configs = await apiFetch('/api/configs');
    const sel = $('f-config');
    const configDescs = {};
    sel.innerHTML = configs.map(c => {
      configDescs[c.value] = c.description || '';
      return `<option value="${escHtml(c.value)}">${escHtml(c.label)}</option>`;
    }).join('');
    const hint = $('f-config-hint');
    const updateHint = () => { hint.textContent = configDescs[sel.value] || ''; };
    sel.addEventListener('change', updateHint);
    updateHint();
  } catch (err) {
    console.warn('Could not load configs:', err);
  }
}

// ── Form submit ────────────────────────────────────────────────────────────
$('enqueue-form').addEventListener('submit', async e => {
  e.preventDefault();
  const msg = $('form-msg');
  msg.textContent = '';

  const body = {
    config:          $('f-config').value,
    prompt:          $('f-prompt').value.trim(),
    negative_prompt: $('f-negative').value.trim(),
    output:          $('f-output').value.trim() || null,
    model_id:        $('f-model-id').value.trim() || null,
    lora_id:         $('f-lora-id').value.trim() || null,
    lora_scale:      $('f-lora-scale').value ? +$('f-lora-scale').value : null,
    steps:           $('f-steps').value ? +$('f-steps').value : null,
    guidance_scale:  $('f-guidance').value ? +$('f-guidance').value : null,
  };

  try {
    const job = await apiFetch('/api/jobs', {
      method: 'POST',
      body: JSON.stringify(body),
    });
    msg.style.color = 'var(--green)';
    msg.textContent = `✅ Queued  id=${job.id.slice(0,8)}…`;
    $('f-prompt').value = '';
    refresh();
  } catch (err) {
    msg.style.color = 'var(--red)';
    msg.textContent = `❌ ${err.message}`;
  }
});

// ── Lightbox ───────────────────────────────────────────────────────────────
function openLightbox(src) {
  const lb = document.getElementById('lightbox');
  document.getElementById('lightbox-img').src = src;
  lb.classList.add('open');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
  document.getElementById('lightbox-img').src = '';
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// ── Polling ────────────────────────────────────────────────────────────────
async function refresh() {
  await Promise.all([refreshStats(), refreshJobs()]);
}

async function init() {
  await loadConfigs();   // populate config dropdown first
  refresh();
  setInterval(refresh, 3000);   // auto-refresh every 3 s
}

init();
</script>

<div id="lightbox" onclick="closeLightbox()">
  <img id="lightbox-img" src="" alt="full size" onclick="event.stopPropagation()">
</div>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def frontend() -> str:
    return _HTML


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the queue web server.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000,
                        help="Bind port (default: 8000).")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload (development only).")
    args = parser.parse_args()

    uvicorn.run(
        "batch.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
