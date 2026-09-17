// The browser half of the demo: input, span layout, panels. No taxonomy logic
// lives here — api/resolve.py resolves, this draws.

const GROUPS = ["structured_identifiers", "sentence_markers", "logical_structures", "corruption", "semantical"];
const LABEL = {
  structured_identifiers: "structured identifiers",
  sentence_markers: "sentence markers",
  logical_structures: "logical structures",
  corruption: "corruption · detected, not repaired",
  semantical: "language",
};
const COLOR = g => getComputedStyle(document.documentElement).getPropertyValue("--" + g).trim();
const SAMPLES = [
  "hello can you please patch CVE-2024-3094 auf 192.168.1.1 tonight AND rollback, no downtime faster than the previus fix",
  "show me invoices > 5000 EUR from Q3 2024 sorted by IBAN DE89370400440532013000",
  "why does ERR_CONNECTION_REFUSED happen when I curl https://api.acme.io/v2/users?limit=50 <div> after upgrading to v2.1.0-rc1",
  "необходимо patch the driver for ICD-10 J45.909 before 2026-07-08T10:00Z",
];

const $ = id => document.getElementById(id);
const box = $("in");

/* ---------------------------------------------------------------- engine */

// Every span and stat comes from the Python function in api/resolve.py — the
// real banks, one implementation. This file only draws them.
async function resolve(text) {
  const res = await fetch("/api/resolve", { method: "POST", body: JSON.stringify({ text }) });
  if (!res.ok) throw new Error(`resolve: ${res.status}`);
  return res.json();
}

/* ---------------------------------------------------------------- render */

$("chips").innerHTML = SAMPLES.map((s, i) =>
  `<button class="chip" data-i="${i}">${s.split(" ").slice(0, 4).join(" ")}…</button>`).join("");
$("chips").onclick = e => {
  const i = e.target.dataset.i;
  if (i === undefined) return;
  box.value = SAMPLES[i];
  run();
};

$("legend").innerHTML = GROUPS.map(g =>
  `<span><i style="background:${COLOR(g)}"></i>${LABEL[g]}</span>`).join("");

// Every claimed span, flattened to one layer per group. Language only shows
// where it differs from the carrier — a monolingual query is one span over
// the whole text, which is a fact for the sidebar, not an underline.
function collect(data) {
  const out = [];
  for (const [group, types] of Object.entries(data.spans || {}))
    for (const [type, spans] of Object.entries(types))
      for (const [text, start, end] of spans) out.push({ group, type, text, start, end });

  const segs = languageSegments(data);
  const carrier = carrierOf(segs);
  for (const s of segs)
    if (s.language !== carrier)
      out.push({ group: "semantical", type: s.language, text: s.text, start: s.start, end: s.end });

  return out.sort((a, b) => a.start - b.start || b.end - a.end);
}

const languageSegments = data =>
  Object.values((data.segments || {}).semantical || {}).flat()
    .map(([text, start, end, language]) => ({ text, start, end, language }));

function carrierOf(segs) {
  const chars = {};
  for (const s of segs) chars[s.language] = (chars[s.language] || 0) + (s.end - s.start);
  return Object.keys(chars).sort((a, b) => chars[b] - chars[a])[0];
}

// Split the text at every span boundary, so a char covered by two groups is
// one element carrying both — overlap is the point, not an edge case.
function render(text, spans) {
  const cuts = [...new Set([0, text.length, ...spans.flatMap(s => [s.start, s.end])])].sort((a, b) => a - b);
  const out = $("out");
  out.textContent = "";
  if (!text) {
    out.innerHTML = '<p class="empty">Nothing to resolve yet.</p>';
    return;
  }
  for (let i = 0; i < cuts.length - 1; i++) {
    const [a, b] = [cuts[i], cuts[i + 1]];
    const hits = spans.filter(s => s.start < b && s.end > a);
    const piece = text.slice(a, b);
    if (!hits.length) { out.append(piece); continue; }

    const el = document.createElement("span");
    el.className = "seg";
    el.textContent = piece;
    el.title = hits.map(h => `${h.group} · ${h.type}`).join("\n");
    el.style.boxShadow = hits.map((h, k) => `0 ${3 + k * 4}px 0 ${COLOR(h.group)}`).join(", ");
    hits.forEach((h, k) => {
      if (h.start !== a) return;  // label once, on the span's first piece
      const lbl = document.createElement("i");
      lbl.className = "lbl";
      lbl.textContent = h.type;
      lbl.style.color = COLOR(h.group);
      lbl.style.bottom = `calc(100% + ${k * 14}px)`;
      el.append(lbl);
    });
    out.append(el);
  }
}

function tallies(data) {
  const rows = GROUPS.flatMap(g => Object.entries((data.tfs || {})[g] || {})
    .sort((a, b) => b[1] - a[1])
    .map(([type, n]) => `<div class="tally"><i style="background:${COLOR(g)}"></i>${type}<em>${n}</em></div>`));
  $("tallies").innerHTML = rows.join("") || '<p class="empty">No spans claimed.</p>';
}

function languages(data) {
  const segs = languageSegments(data);
  if (!segs.length) { $("langs").innerHTML = '<p class="empty">Not measured.</p>'; return; }
  const carrier = carrierOf(segs);
  const by = {};
  for (const s of segs) (by[s.language] ||= []).push(s.text.trim());
  $("langs").innerHTML = Object.entries(by).map(([lang, texts]) => {
    const note = lang === carrier ? "carrier" : `"${texts.slice(0, 3).join('", "')}"`;
    const color = lang === carrier ? "var(--fg)" : COLOR("semantical");
    return `<div class="lang"><b style="color:${color}">${lang}</b>
      <span>${texts.length} span${texts.length > 1 ? "s" : ""} · ${note}</span></div>`;
  }).join("") + (data.is_code_switched ? `<div class="lang"><span>code-switched</span></div>` : "");
}

const FRACTION = /(_ratio|_share|_rate)$/;

function stats(data) {
  const rows = Object.values(data.stats || {}).flatMap(types => Object.values(types).flat());
  $("stats").innerHTML = rows.map(([name, value]) => {
    const shown = Number.isInteger(value) ? value : value.toFixed(2);
    const bar = FRACTION.test(name)
      ? `<div class="bar"><i style="width:${Math.min(100, value * 100)}%"></i></div>` : "";
    return `<div class="row"><b>${name}</b><span>${shown}</span></div>${bar}`;
  }).join("") || '<p class="empty">Not measured.</p>';
}

let seq = 0;
async function run() {
  const text = box.value;
  const mine = ++seq;
  let data;
  try {
    data = await resolve(text);
  } catch (err) {
    $("warn").textContent = " · " + err.message;
    return;
  }
  if (mine !== seq) return;  // a later keystroke already answered
  $("mode").textContent = (data.engines || []).join(" · ");
  $("warn").textContent = data.warning ? " · " + data.warning : "";
  render(text, collect(data));
  tallies(data);
  languages(data);
  stats(data);
}

let timer;
box.oninput = () => { clearTimeout(timer); timer = setTimeout(run, 60); };
box.value = SAMPLES[0];
run();
