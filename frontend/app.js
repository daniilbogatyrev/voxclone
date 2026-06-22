const $ = (id) => document.getElementById(id);
const setStatus = (m) => { $("status").textContent = m; };

async function loadLanguages() {
  const { languages } = await (await fetch("/languages")).json();
  $("language").innerHTML = languages
    .map((l) => `<option value="${l.code}">${l.label}</option>`)
    .join("");
}

async function loadVoices() {
  const lang = $("language").value;
  const { voices, default: def } = await (
    await fetch(`/voices?language=${encodeURIComponent(lang)}`)
  ).json();
  $("voice").innerHTML = voices
    .map((v) => `<option value="${v.key}">${v.label}</option>`)
    .join("");
  if (def) $("voice").value = def;
}

$("language").addEventListener("change", loadVoices);

$("go").addEventListener("click", async () => {
  const text = $("text").value.trim();
  if (!text) { setStatus("Enter some text."); return; }
  const voice = $("voice").value;
  const language = $("language").value;
  const label = $("voice").selectedOptions[0]?.textContent ?? voice;
  setStatus(`Generating with “${label}”… (first clip for a voice takes ~30–90 s)`);
  $("go").disabled = true;
  try {
    const resp = await fetch("/clone", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voice, text, language }),
    });
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try { detail = (await resp.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    const blob = await resp.blob();
    $("player").src = URL.createObjectURL(blob);
    $("player").play();
    setStatus("Done.");
  } catch (e) {
    setStatus("Error: " + e.message);
  } finally {
    $("go").disabled = false;
  }
});

$("restart").addEventListener("click", async () => {
  setStatus("Restarting voice…");
  try {
    await fetch("/restart", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voice: $("voice").value }),
    });
    setStatus("Voice reset — the next Generate reloads it.");
  } catch (e) {
    setStatus("Error: " + e.message);
  }
});

(async () => {
  try {
    await loadLanguages();
    await loadVoices();
    setStatus("Ready.");
  } catch (e) {
    setStatus("Could not reach the server: " + e.message);
  }
})();
