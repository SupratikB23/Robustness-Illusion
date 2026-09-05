async function get(p) { const r = await fetch(p); if (!r.ok) throw new Error("missing " + p); return r.json(); }
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const num = (v, d=3) => (typeof v === "number" && isFinite(v)) ? v.toFixed(d) : "n/a";

function drawCurve(canvas, c) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width = canvas.offsetWidth * 2, H = canvas.height = 520;
  const X = s => 60 + s * (W - 90), Y = v => H - 40 - v * (H - 80);
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "#ccc"; ctx.beginPath(); ctx.moveTo(60, Y(1)); ctx.lineTo(W - 30, Y(1)); ctx.stroke();
  const band = (lo, hi, color) => {
    ctx.fillStyle = color; ctx.beginPath();
    c.strength.forEach((s, i) => i ? ctx.lineTo(X(s), Y(hi[i])) : ctx.moveTo(X(s), Y(hi[i])));
    [...c.strength.keys()].reverse().forEach(i => ctx.lineTo(X(c.strength[i]), Y(lo[i])));
    ctx.closePath(); ctx.fill();
  };
  band(c.output_lo, c.output_hi, "rgba(30,120,220,.15)");
  band(c.feature_lo, c.feature_hi, "rgba(220,80,30,.15)");
  const line = (key, color, dash=[]) => {
    ctx.strokeStyle = color; ctx.setLineDash(dash); ctx.lineWidth = 3; ctx.beginPath();
    c.strength.forEach((s, i) => i ? ctx.lineTo(X(s), Y(c[key][i])) : ctx.moveTo(X(s), Y(c[key][i])));
    ctx.stroke(); ctx.setLineDash([]);
  };
  line("output_mean", "#1e78dc"); line("feature_mean", "#dc501e"); line("null_b", "#888", [8, 6]);
  ctx.fillStyle = "#333"; ctx.font = "22px system-ui";
  c.strength.forEach(s => ctx.fillText(String(s), X(s) - 8, H - 12));
}

async function main() {
  try {
    const curves = await get("data/curves.json");
    const wrap = document.getElementById("curve-list");
    for (const [t, c] of Object.entries(curves.curves)) {
      const r = curves.rii[t];
      const div = document.createElement("div");
      div.className = "card";
      div.innerHTML = `<h3>${esc(t)} — RII ${num(r.mean)} [${num(r.lo)}, ${num(r.hi)}]</h3><canvas></canvas>`;
      wrap.appendChild(div);
      drawCurve(div.querySelector("canvas"), c);
    }
  } catch { document.getElementById("curve-list").textContent = "Run the pipeline to generate data/curves.json."; }

  try {
    const slider = await get("data/slider.json");
    const selT = document.getElementById("sel-t"), selS = document.getElementById("sel-s");
    const ts = [...new Set(slider.frames.map(f => f.transform))];
    ts.forEach(t => { const o = document.createElement("option"); o.value = o.textContent = t; selT.appendChild(o); });
    const listFor = () => slider.frames.filter(x => x.transform === selT.value).sort((a, b) => a.strength - b.strength);
    const render = () => {
      const list = listFor();
      selS.max = Math.max(0, list.length - 1);
      const f = list[+selS.value];
      if (!f) return;
      const img = document.getElementById("sl-img");
      if (f.image) { img.src = "data/" + f.image; img.alt = `${f.transform} strength ${f.strength}`; }
      else { img.removeAttribute("src"); img.alt = "frame not precomputed"; }
      document.getElementById("sl-num").innerHTML =
        `strength <b>${esc(f.strength)}</b> (${+selS.value + 1}/${list.length})<br>` +
        `output <b>${num(f.output_stability)}</b><br>features <b>${num(f.feature_stability)}</b>`;
      const li = (ids, cls) => ids.slice(0, 32).map(i => `<span class="${cls}">${esc(i)}</span>`).join(" ");
      document.getElementById("sl-feat").innerHTML =
        `<div>kept: ${li(f.kept, "")}</div><div>entered: ${li(f.entered, "entered")}</div><div>left: ${li(f.left, "left")}</div>`;
    };
    selT.onchange = render; selS.oninput = render; render();
  } catch { /* no slider data yet */ }

  try {
    const cases = await get("data/cases.json");
    const banner = cases.pending_hand_analysis
      ? `<div class="card"><b>Pending hand analysis.</b> Cases below are finder drafts, not curated results.</div>` : "";
    document.getElementById("case-list").innerHTML = banner + cases.cases.map(c =>
      `<div class="card"><b>${esc(c.transform)}</b> · ${esc(c.image_id)} · strength ${esc(c.strength)}<br>` +
      `output ${num(c.output_stability, 2)}, features ${num(c.feature_stability, 2)}<br>${esc(c.description)}</div>`).join("");
  } catch { document.getElementById("case-list").textContent = "No cases yet."; }
}
main();
