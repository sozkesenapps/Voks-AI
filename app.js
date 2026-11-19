"use strict";
// Voks AI — telefonda çalışan, internetsiz kişisel yapay zeka (prototip).
// Beyin: WebLLM (telefonun GPU'su). Göz: Transformers.js. Yüz tanıma: face-api.
// Hafıza: bu telefonun tarayıcı depolaması. Hiçbir sunucuya veri gönderilmez.

const CDN = {
  webllm: "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2/+esm",
  transformers: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/+esm",
  faceapi: "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1/dist/face-api.js",
  faceModels: "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1/model/",
};

const MODEL_PRESETS = {
  light:    { label: "Hafif: Llama 3.2 1B (yaklaşık 0,9 GB)",   id: "Llama-3.2-1B-Instruct-q4f16_1-MLC" },
  balanced: { label: "Dengeli: Qwen 2.5 1.5B (yaklaşık 1,1 GB)", id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC" },
  smart:    { label: "Akıllı: Qwen 2.5 3B (yaklaşık 2,2 GB)",    id: "Qwen2.5-3B-Instruct-q4f16_1-MLC" },
  gemma:    { label: "Gemma 2 2B (yaklaşık 1,9 GB)",             id: "gemma-2-2b-it-q4f16_1-MLC" },
};

const FACE_THRESHOLD = 0.5;      // küçüldükçe daha temkinli tanır
const FACT_BUDGET = 2600;        // bilgi listesi için karakter bütçesi
const MAX_CHAT_KEEP = 300;

// ---------------------------------------------------------------- yardımcılar
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
const lower = (s) => (s || "").toLocaleLowerCase("tr");

function tokens(s) {
  return lower(s).split(/[^\p{L}\p{N}]+/u).filter((w) => w.length >= 3).map((w) => w.slice(0, 5));
}

function toast(msg, ms = 3200) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, ms);
}

function ask({ title, text = "", input = null, ok = "Tamam", cancel = "Vazgeç" }) {
  return new Promise((resolve) => {
    const modal = $("modal"), inp = $("modalInput"), okBtn = $("modalOk"), cancelBtn = $("modalCancel");
    $("modalTitle").textContent = title;
    $("modalText").textContent = text;
    inp.hidden = input === null;
    inp.value = "";
    inp.placeholder = input || "";
    okBtn.textContent = ok;
    cancelBtn.textContent = cancel;
    modal.hidden = false;
    if (input !== null) setTimeout(() => inp.focus(), 50);
    const done = (v) => {
      modal.hidden = true;
      okBtn.onclick = null; cancelBtn.onclick = null; inp.onkeydown = null;
      resolve(v);
    };
    okBtn.onclick = () => done(input !== null ? inp.value.trim() : true);
    cancelBtn.onclick = () => done(input !== null ? null : false);
    inp.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); okBtn.onclick(); } };
  });
}

// ---------------------------------------------------------------- durum / hafıza
const LS_KEY = "voksai.state.v1";
const DEFAULTS = {
  settings: {
    aiName: "Voks",
    owner: "",
    character: "Meraklı, samimi ve dürüst. Öğrenmeyi sever, bilmediğini açıkça söyler.",
    v: 2,
    modelPreset: "light",
    f32: true,               // uyumlu mod: bazı telefonlarda f16 bozuk çıktı verir
    facing: "environment",
    tts: false,
    teach: true,
    watch: false,
  },
  facts: [],      // { id, text, t }
  people: [],     // { id, name, descriptors: number[][], t }
  thoughts: [],   // { id, text, t }
  chat: [],       // { role: user|assistant|note|thought, content, t }
  ready: { llm: null, vision: false, faces: false },
};

function loadState() {
  const base = JSON.parse(JSON.stringify(DEFAULTS));
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return base;
    const o = JSON.parse(raw);
    base.settings = { ...base.settings, ...(o.settings || {}) };
    if (!o.settings || o.settings.v !== 2) { base.settings.v = 2; base.settings.modelPreset = "light"; base.settings.f32 = true; }
    for (const k of ["facts", "people", "thoughts", "chat"]) if (Array.isArray(o[k])) base[k] = o[k];
    const keepLlm = !o.settings || o.settings.v === 2;
    base.ready = { ...base.ready, ...(o.ready || {}) };
    if (!keepLlm) base.ready.llm = null;
  } catch (e) { console.warn("Hafıza okunamadı", e); }
  return base;
}

const state = loadState();
let saveTimer = null;
function saveNow() {
  clearTimeout(saveTimer);
  try {
    if (state.chat.length > MAX_CHAT_KEEP) state.chat = state.chat.slice(-MAX_CHAT_KEEP);
    localStorage.setItem(LS_KEY, JSON.stringify(state));
  } catch (e) {
    console.warn(e);
    toast("Kaydedilemedi: depolama dolu olabilir. Yedek alıp eski sohbeti temizle.");
  }
}
function save() { clearTimeout(saveTimer); saveTimer = setTimeout(saveNow, 250); }
document.addEventListener("visibilitychange", () => { if (document.hidden) saveNow(); });
window.addEventListener("pagehide", saveNow);

function addFact(text) {
  text = text.replace(/\s+/g, " ").trim().slice(0, 300);
  if (!text) return false;
  const key = lower(text);
  if (state.facts.some((f) => lower(f.text) === key)) return false;
  state.facts.push({ id: uid(), text, t: Date.now() });
  save();
  return true;
}

function pickFacts(query, budget) {
  const all = state.facts;
  const total = all.reduce((a, f) => a + f.text.length + 3, 0);
  if (total <= budget) return all.map((f) => f.text);
  const qs = new Set(tokens(query));
  const scored = all.map((f, i) => ({ f, i, score: tokens(f.text).reduce((a, t) => a + (qs.has(t) ? 1 : 0), 0) }));
  const chosen = new Map();
  scored.filter((x) => x.score > 0).sort((a, b) => b.score - a.score || b.i - a.i).slice(0, 10).forEach((x) => chosen.set(x.i, x));
  scored.slice(-8).forEach((x) => chosen.set(x.i, x));
  const byRecency = [...chosen.values()].sort((a, b) => b.i - a.i);
  const out = [];
  let used = 0;
  for (const x of byRecency) {
    const l = x.f.text.length + 3;
    if (used + l > budget) continue;
    out.push(x); used += l;
  }
  return out.sort((a, b) => a.i - b.i).map((x) => x.f.text);
}

const QWORDS = new Set(["ne", "neden", "niçin", "niye", "kim", "kimin", "kime", "kimi", "nasıl", "nerede", "nereye", "nereden", "hangi", "kaç", "kaçta"]);
const SMALLTALK = /^(merhaba|selam|hey|naber|teşekkürler|teşekkür|sağol|tamam|evet|hayır|ok|peki|günaydın|iyi geceler)\b/i;
function isQuestion(text) {
  const t = text.trim();
  if (/\?\s*$/.test(t)) return true;
  const words = lower(t).split(/\s+/).map((w) => w.replace(/[^\p{L}]/gu, "")).filter(Boolean);
  if (!words.length) return true;
  if (QWORDS.has(words[0])) return true;
  if (/^m[ıiuü]/.test(words[words.length - 1]) && words[words.length - 1].length <= 7) return true;
  return false;
}
function worthLearning(text) {
  const words = text.trim().split(/\s+/).filter(Boolean);
  return words.length >= 2 && !SMALLTALK.test(text.trim()) && !isQuestion(text);
}

// ---------------------------------------------------------------- arayüz durumu
let busy = false;           // sohbet üretiyor
let sending = false;        // bir mesaj işleniyor (bakma + cevap)
let busyVision = false;     // görüntü işleniyor
let cam = null;             // MediaStream
let lastSeen = null;        // son "Bak" sonucu
let present = { names: [], unknown: 0, t: 0 };
const announced = new Map();

function setProgress(visible, text = "", frac = null) {
  $("progress").hidden = !visible;
  if (!visible) return;
  $("progressText").textContent = text;
  const bar = $("progressBar");
  if (frac === null) { bar.classList.add("indet"); bar.style.width = ""; }
  else { bar.classList.remove("indet"); bar.style.width = Math.round(Math.max(0, Math.min(1, frac)) * 100) + "%"; }
}

function updateChips() {
  const m = $("chipModel"), c = $("chipCam");
  const loaded = !!engine;
  m.textContent = loaded ? "Beyin: hazır" : "Beyin: kapalı";
  m.classList.toggle("on", loaded);
  c.textContent = cam ? "Göz: açık" : "Göz: kapalı";
  c.classList.toggle("on", !!cam);
  $("eyeDot").classList.toggle("live", !!cam);
  $("aiTitle").textContent = state.settings.aiName.trim() || "Voks";
  document.title = $("aiTitle").textContent;
}

function setBanner(text) {
  const b = $("banner");
  b.hidden = !text;
  b.textContent = text || "";
}

// ---------------------------------------------------------------- sohbet kutusu
function renderLog() {
  const log = $("log");
  log.textContent = "";
  if (!state.chat.length) {
    const w = document.createElement("div");
    w.className = "welcome";
    const name = state.settings.aiName.trim() || "Voks";
    w.innerHTML = "<b></b> henüz hiçbir şey bilmiyor.<br>Ona öğretmeye başla:<ul><li>“Benim adım …” gibi bir şey yaz</li><li>📷 ile kamerayı aç, 👤 ile bir kişiyi tanıt</li><li>👁 ile baktır, 💭 ile düşündür</li></ul>";
    w.querySelector("b").textContent = name;
    log.appendChild(w);
    return;
  }
  for (const m of state.chat) log.appendChild(bubble(m.role, m.content));
  log.scrollTop = log.scrollHeight;
}

function bubble(role, text) {
  const d = document.createElement("div");
  d.className = "msg " + role;
  d.textContent = role === "thought" ? "💭 " + text : text;
  return d;
}

function pushChat(role, content) {
  state.chat.push({ role, content, t: Date.now() });
  const log = $("log");
  const w = log.querySelector(".welcome");
  if (w) w.remove();
  log.appendChild(bubble(role, content));
  log.scrollTop = log.scrollHeight;
  save();
}

function setBusyUI(on) {
  busy = on;
  $("btnSend").disabled = on;
  $("btnThink").disabled = on;
  $("btnLook").disabled = on || busyVision;
}

// ---------------------------------------------------------------- beyin (WebLLM)
let engine = null, engineId = null, webllmMod = null, llmLoading = null;

let diagText = "";
function showDiag() { const d = $("diag"); if (d) d.textContent = diagText; }

async function resolveModelId() {
  let id = (MODEL_PRESETS[state.settings.modelPreset] || MODEL_PRESETS.light).id;
  try {
    const adapter = navigator.gpu && (await navigator.gpu.requestAdapter());
    const noF16 = adapter && !adapter.features.has("shader-f16");
    if (adapter) {
      const inf = adapter.info || {};
      diagText = `GPU: ${[inf.vendor, inf.architecture, inf.description].filter(Boolean).join(" ") || "bilinmiyor"} | f16: ${noF16 ? "yok" : "var"}`;
      showDiag();
    }
    if (state.settings.f32 || noF16) id = id.replace("q4f16_1", "q4f32_1");
  } catch (e) { /* varsayılan kalsın */ }
  return id;
}

async function resetEngine() {
  if (engine) { try { await engine.unload(); } catch (e) { /* yok say */ } }
  engine = null; engineId = null;
  updateChips();
}

async function ensureLLM() {
  if (llmLoading) return llmLoading;
  const id = await resolveModelId();
  if (engine && engineId === id) return engine;
  llmLoading = (async () => {
    try {
      if (!navigator.gpu) throw new Error("Bu tarayıcı WebGPU desteklemiyor. Android'de güncel Chrome kullan.");
      setProgress(true, "Beyin hazırlanıyor…", null);
      webllmMod = webllmMod || (await import(CDN.webllm));
      const known = (webllmMod.prebuiltAppConfig?.model_list || []).map((m) => m.model_id);
      if (known.length && !known.includes(id)) throw new Error("Bu model bu sürümde yok: " + id);
      if (engine) { try { await engine.unload(); } catch (e) { /* yok say */ } engine = null; engineId = null; }
      const e = await webllmMod.CreateMLCEngine(id, {
        initProgressCallback: (r) => {
          const pct = Math.round((r.progress || 0) * 100);
          setProgress(true, `Beyin yükleniyor %${pct} (ilk seferde indirilir, sonra internetsiz çalışır)`, r.progress || 0);
        },
      });
      engine = e; engineId = id;
      state.ready.llm = id;
      saveNow();
      updateChips(); renderReady();
      return e;
    } finally {
      llmLoading = null;
      setProgress(false);
    }
  })();
  return llmLoading;
}

function normalizeHistory(list) {
  const out = [];
  for (const m of list) {
    if (m.role !== "user" && m.role !== "assistant") continue;
    const content = (m.content || "").slice(0, 500);
    if (!content) continue;
    if (out.length && out[out.length - 1].role === m.role) out[out.length - 1].content += "\n" + content;
    else out.push({ role: m.role, content });
  }
  while (out.length && out[0].role !== "user") out.shift();
  return out;
}
function recentHistory(n = 6) {
  return normalizeHistory(state.chat.filter((m) => m.role === "user" || m.role === "assistant").slice(-n));
}

function seenText(r) {
  if (!r) return "";
  const parts = [];
  if (r.caption) parts.push(`Resim açıklaması (İngilizce): ${r.caption}`);
  if (r.objects && r.objects.length) parts.push(`Nesneler (İngilizce): ${r.objects.join(", ")}`);
  const people = [...new Set(r.people || [])];
  parts.push(people.length ? `Yüzünden tanıdığın kişiler: ${people.join(", ")}` : "Tanıdığın kimse görünmüyor");
  if (r.unknown) parts.push(`Tanımadığın ${r.unknown} kişi var`);
  return parts.join(". ") + ".";
}

function buildSystem(query, opts = {}) {
  const s = state.settings;
  const name = s.aiName.trim() || "Voks";
  const owner = s.owner.trim() || "sahibin";
  const L = [];
  L.push(`Sen ${name} adında bir yapay zekasın. ${owner} seni yetiştiriyor; telefonun içinde yaşıyorsun ve kameradan gelen görüntüyle etrafı görebiliyorsun.`);
  L.push(`Karakterin: ${s.character}`);
  L.push("Kurallar: Her zaman Türkçe, kısa ve samimi konuş (en fazla 3-4 cümle). Sahibinle ve çevrenle ilgili hiçbir şeyi uydurma. Sadece aşağıdaki ÖĞRETİLENLER listesindekileri ve bu konuşmada duyduklarını biliyorsun. Bilmediğin bir şey sorulursa dürüstçe \"Bunu henüz bilmiyorum, bana öğretir misin?\" de. Bir yapay zeka olduğunu saklama.");
  const facts = pickFacts(query || "", opts.compact ? Math.floor(FACT_BUDGET / 2) : FACT_BUDGET);
  L.push(facts.length ? "ÖĞRETİLENLER:\n" + facts.map((f) => "- " + f).join("\n") : "ÖĞRETİLENLER: (henüz hiçbir şey öğretilmedi)");
  if (state.people.length) L.push("Yüzünden tanıdığın kişiler: " + state.people.map((p) => p.name).join(", ") + ".");
  if (present.names.length && Date.now() - present.t < 15000) L.push("Kameranda şu an görünen tanıdık kişiler: " + present.names.join(", ") + ".");
  if (opts.seen) L.push("GÖZÜNDEN GELEN BİLGİ (şimdi gördüğün): " + opts.seen + " Bunu, kendi gözünle görüyormuşsun gibi Türkçe anlat. İngilizce kelimeleri Türkçeye çevir. Bilgide olmayan bir şeyi uydurma.");
  const th = state.thoughts.slice(-2).map((t) => t.text);
  if (th.length && !opts.compact) L.push("Son düşüncelerin: " + th.join(" / "));
  if (opts.hint) L.push(opts.hint);
  return L.join("\n\n");
}

const GPU_BROKEN = /dispos|mapAsync|device|lost|instance reference|destroyed/i;
async function generate(messages, onToken, retried = false) {
  try {
    const eng = await ensureLLM();
    const stream = await eng.chat.completions.create({ messages, stream: true, temperature: 0.7, top_p: 0.9, max_tokens: 320 });
    let out = "";
    for await (const chunk of stream) {
      const d = chunk.choices?.[0]?.delta?.content || "";
      if (!d) continue;
      out += d;
      if (onToken) onToken(out);
    }
    return out.trim();
  } catch (e) {
    // GPU bağlantısı koptuysa beyni baştan kurup bir kez daha dene
    if (!retried && GPU_BROKEN.test(String(e && e.message))) {
      console.warn("GPU sorunu, beyin yeniden başlatılıyor", e);
      await resetEngine();
      return generate(messages, onToken, true);
    }
    if (GPU_BROKEN.test(String(e && e.message))) await resetEngine();
    throw e;
  }
}

function speak(text) {
  if (!state.settings.tts || !("speechSynthesis" in window)) return;
  try {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "tr-TR";
    speechSynthesis.speak(u);
  } catch (e) { /* ses yoksa sessiz kal */ }
}

async function respond(userText, opts = {}) {
  setBusyUI(true);
  const log = $("log");
  const typing = bubble("assistant", "…");
  typing.classList.add("typing");
  log.querySelector(".welcome")?.remove();
  log.appendChild(typing);
  log.scrollTop = log.scrollHeight;
  try {
    const hist = opts.hist || [];
    const run = (compact) => generate(
      [{ role: "system", content: buildSystem(userText, { ...opts, compact }) }, ...(compact ? [] : hist), { role: "user", content: userText }],
      (t) => { typing.textContent = t; typing.classList.remove("typing"); log.scrollTop = log.scrollHeight; }
    );
    let text;
    try { text = await run(false); }
    catch (e) {
      if (/context|window|token/i.test(String(e && e.message))) text = await run(true);
      else throw e;
    }
    typing.remove();
    text = text || "…";
    pushChat("assistant", text);
    speak(text);
    return text;
  } catch (e) {
    typing.remove();
    console.error(e);
    pushChat("note", "Bir sorun oldu: " + (e && e.message ? e.message : e));
    return null;
  } finally {
    setBusyUI(false);
  }
}

async function confirmFirstDownload() {
  const id = await resolveModelId();
  if (engine && engineId === id) return true;
  if (state.ready.llm === id) return true;
  return ask({
    title: "Beyin ilk kez indirilecek",
    text: "Yaklaşık 1-2 GB indirilir. Wi-Fi'da olduğundan emin ol. Bir kere indikten sonra internetsiz çalışır.",
    ok: "İndir",
  });
}

// ---------------------------------------------------------------- mesaj gönderme
async function onSend(text) {
  text = text.trim();
  if (!text || busy || sending) return;
  sending = true;
  try { await handleSend(text); } finally { sending = false; }
}

async function handleSend(text) {

  // "unut: ..." komutu
  const mf = text.match(/^(unut|sil)\s*[:：-]\s*(.+)$/i);
  if (mf) {
    const q = lower(mf[2].trim());
    const before = state.facts.length;
    state.facts = state.facts.filter((f) => !lower(f.text).includes(q));
    const n = before - state.facts.length;
    pushChat("user", text);
    pushChat("note", n ? `${n} bilgiyi unuttum.` : "Böyle bir şey bulamadım.");
    renderMemory();
    return;
  }

  if (!(await confirmFirstDownload())) return;

  const hist = recentHistory();
  let lesson = null;
  const ml = text.match(/^(öğren|ogren|hatırla|hatirla)\s*[:：-]\s*(.+)$/i);
  if (ml) lesson = ml[2].trim();
  else if (state.settings.teach && worthLearning(text)) lesson = text;

  pushChat("user", text);
  let hint = null;
  if (lesson) {
    addFact(lesson);
    renderMemory();
    hint = `Sahibin şu an sana şunu öğretti: "${lesson}". Öğrendiğini kendi cümlenle kısaca onayla. İstersen onunla ilgili merak ettiğin tek bir kısa soru sor.`;
  }

  // Kamera açıksa ve görmeyle ilgili bir şey soruluyorsa önce bak
  let seen = "";
  if (cam && /(^|[^\p{L}])(bak|gör|kim|etraf|önüm|karşı|ne var|burada|ortam)/u.test(lower(text))) {
    try { const r = await look(); seen = seenText(r); } catch (e) { console.warn(e); }
  } else if (cam && lastSeen && Date.now() - lastSeen.t < 30000) {
    seen = seenText(lastSeen);
  }
  await respond(text, { hist, hint, seen });
}

// ---------------------------------------------------------------- göz (kamera, yüz, görüntü)
function setCamStatus() {
  updateChips();
  $("camBox").hidden = !cam;
  $("btnCam").textContent = cam ? "📷 Kapat" : "📷 Kamera";
  $("btnCam").classList.toggle("on", !!cam);
  $("video").classList.toggle("mirror", state.settings.facing === "user");
}

async function startCam() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("Kamera bu tarayıcıda kullanılamıyor.");
  cam = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: state.settings.facing }, width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  const v = $("video");
  v.srcObject = cam;
  try { await v.play(); } catch (e) { /* autoplay engellenirse kullanıcı dokununca başlar */ }
  setCamStatus();
  startWatch();
}

function stopCam() {
  stopWatch();
  if (cam) cam.getTracks().forEach((t) => t.stop());
  cam = null;
  $("video").srcObject = null;
  present = { names: [], unknown: 0, t: 0 };
  renderCamTags();
  setCamStatus();
}

function grab(maxW = 480) {
  const v = $("video");
  const w = v.videoWidth || 640, h = v.videoHeight || 480;
  const sc = Math.min(1, maxW / w);
  const c = document.createElement("canvas");
  c.width = Math.round(w * sc); c.height = Math.round(h * sc);
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
  return c;
}

function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = src; s.onload = res;
    s.onerror = () => rej(new Error("Yüz tanıma kütüphanesi yüklenemedi (internet gerekir, ilk seferde)."));
    document.head.appendChild(s);
  });
}

let facesReady = null;
function ensureFaces() {
  if (facesReady) return facesReady;
  facesReady = (async () => {
    if (!window.faceapi) await loadScript(CDN.faceapi);
    await window.faceapi.nets.tinyFaceDetector.loadFromUri(CDN.faceModels);
    await window.faceapi.nets.faceLandmark68TinyNet.loadFromUri(CDN.faceModels);
    await window.faceapi.nets.faceRecognitionNet.loadFromUri(CDN.faceModels);
    state.ready.faces = true; saveNow(); renderReady();
  })().catch((e) => { facesReady = null; throw e; });
  return facesReady;
}

async function detectFaces(source) {
  await ensureFaces();
  const fa = window.faceapi;
  return fa.detectAllFaces(source, new fa.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.45 }))
    .withFaceLandmarks(true).withFaceDescriptors();
}

function matchPerson(desc) {
  let best = null, bd = Infinity;
  for (const p of state.people) {
    for (const d of p.descriptors) {
      const dist = window.faceapi.euclideanDistance(desc, d);
      if (dist < bd) { bd = dist; best = p; }
    }
  }
  return best && bd <= FACE_THRESHOLD ? { person: best, dist: bd } : null;
}

let tf = null, captioner = null, detector = null, visionLoading = null;
function ensureVision() {
  if (captioner) return Promise.resolve();
  if (visionLoading) return visionLoading;
  visionLoading = (async () => {
    try {
      setProgress(true, "Göz hazırlanıyor (ilk seferde indirilir)…", null);
      tf = tf || (await import(CDN.transformers));
      tf.env.allowLocalModels = false;
      try { tf.env.backends.onnx.wasm.numThreads = 1; } catch (e) { /* yok say */ }
      const cb = (p) => { if (p && p.status === "progress" && typeof p.progress === "number") setProgress(true, `Göz indiriliyor %${Math.round(p.progress)}`, p.progress / 100); };
      captioner = await tf.pipeline("image-to-text", "Xenova/vit-gpt2-image-captioning", { dtype: "q8", progress_callback: cb });
      try { detector = await tf.pipeline("object-detection", "Xenova/yolos-tiny", { dtype: "q8", progress_callback: cb }); }
      catch (e) { console.warn("Nesne tanıyıcı yüklenemedi", e); detector = null; }
      state.ready.vision = true; saveNow(); renderReady();
    } finally {
      visionLoading = null;
      setProgress(false);
    }
  })();
  return visionLoading;
}

async function look() {
  if (!cam) throw new Error("Önce kamerayı aç.");
  if (busyVision) throw new Error("Zaten bakıyorum.");
  busyVision = true;
  $("btnLook").disabled = true;
  const r = { t: Date.now(), caption: "", objects: [], people: [], unknown: 0, errors: [] };
  try {
    const canvas = grab(480);
    const url = canvas.toDataURL("image/jpeg", 0.85);
    try {
      const dets = await detectFaces(canvas);
      for (const d of dets) {
        const m = matchPerson(d.descriptor);
        if (m) r.people.push(m.person.name); else r.unknown++;
      }
    } catch (e) { r.errors.push("yüz: " + e.message); }
    try {
      await ensureVision();
      const out = await captioner(url, { max_new_tokens: 30 });
      r.caption = (out && out[0] && out[0].generated_text || "").trim();
      if (detector) {
        const objs = await detector(url, { threshold: 0.6 });
        const counts = {};
        for (const o of objs) counts[o.label] = (counts[o.label] || 0) + 1;
        r.objects = Object.entries(counts).map(([k, v]) => (v > 1 ? `${k} x${v}` : k));
      }
    } catch (e) { r.errors.push("görüntü: " + e.message); }
    if (r.errors.length) toast("Bazı gözlerim çalışmadı: " + r.errors.join("; "), 5000);
    lastSeen = r;
    present = { names: [...new Set(r.people)], unknown: r.unknown, t: Date.now() };
    renderCamTags();
    return r;
  } finally {
    busyVision = false;
    $("btnLook").disabled = busy;
  }
}

function renderCamTags() {
  const box = $("camTags");
  box.textContent = "";
  for (const n of present.names) {
    const s = document.createElement("span"); s.className = "camtag"; s.textContent = n; box.appendChild(s);
  }
  if (present.unknown) {
    const s = document.createElement("span"); s.className = "camtag unknown";
    s.textContent = present.unknown > 1 ? `${present.unknown} tanımadığım yüz` : "tanımadığım biri"; box.appendChild(s);
  }
}

async function enroll() {
  if (!cam) { toast("Önce kamerayı aç, kişiyi karşına al."); return; }
  const name = await ask({ title: "Bu kişinin adı ne?", text: "Kişiyi kameraya net bak. Birkaç saniye yüzünü inceleyeceğim.", input: "Örn. Ahmet", ok: "Tanıt" });
  if (!name) return;
  busyVision = true; $("btnLook").disabled = true; $("btnEnroll").disabled = true;
  setProgress(true, "Yüzü inceliyorum…", null);
  try {
    const descs = [];
    let multi = false;
    for (let i = 0; i < 10 && descs.length < 5; i++) {
      const dets = await detectFaces(grab(480));
      if (dets.length === 1) descs.push(Array.from(dets[0].descriptor));
      else if (dets.length > 1) multi = true;
      await sleep(350);
    }
    if (descs.length < 3) {
      toast(multi ? "Kadrada birden fazla yüz var. Sadece tanıtacağın kişi görünsün." : "Yüzü net göremedim. Yüzü kameraya çevirip tekrar dene.", 5000);
      return;
    }
    const existing = state.people.find((p) => lower(p.name) === lower(name));
    if (existing) existing.descriptors = existing.descriptors.concat(descs).slice(-10);
    else state.people.push({ id: uid(), name, descriptors: descs, t: Date.now() });
    save(); renderMemory();
    pushChat("note", `👤 ${name} adlı kişiyi tanıdım.`);
    const rel = await ask({ title: `${name} senin neyin?`, text: "İstersen kısaca yaz (örn. kardeşim, iş ortağım). Boş bırakabilirsin.", input: "İsteğe bağlı", ok: "Kaydet", cancel: "Geç" });
    if (rel) { addFact(`${name} hakkında: ${rel}`); renderMemory(); }
  } catch (e) {
    toast("Kişi tanıtılamadı: " + e.message, 5000);
  } finally {
    setProgress(false);
    busyVision = false; $("btnLook").disabled = busy; $("btnEnroll").disabled = false;
  }
}

// sürekli tanıma
let watchTimer = null, watchBusy = false, prevNames = new Set();
function startWatch() {
  stopWatch();
  if (!state.settings.watch || !cam) return;
  watchTimer = setInterval(watchTick, 3000);
}
function stopWatch() { clearInterval(watchTimer); watchTimer = null; prevNames = new Set(); }
async function watchTick() {
  if (!cam || document.hidden || watchBusy || busyVision || !state.people.length) return;
  watchBusy = true;
  try {
    const dets = await detectFaces($("video"));
    const names = new Set(); let unknown = 0;
    for (const d of dets) { const m = matchPerson(d.descriptor); if (m) names.add(m.person.name); else unknown++; }
    present = { names: [...names], unknown, t: Date.now() };
    renderCamTags();
    for (const n of names) {
      if (!prevNames.has(n) && Date.now() - (announced.get(n) || 0) > 60000) {
        announced.set(n, Date.now());
        pushChat("note", `👁 ${n} göründü.`);
      }
    }
    prevNames = names;
  } catch (e) { console.warn("watch", e); }
  finally { watchBusy = false; }
}

// ---------------------------------------------------------------- düşünce
async function think() {
  if (busy) return;
  if (!(await confirmFirstDownload())) return;
  setBusyUI(true);
  const typing = bubble("thought", "…");
  typing.classList.add("typing");
  $("log").querySelector(".welcome")?.remove();
  $("log").appendChild(typing);
  $("log").scrollTop = $("log").scrollHeight;
  try {
    const q = state.facts.slice(-3).map((f) => f.text).join(" ");
    const seen = lastSeen && Date.now() - lastSeen.t < 10 * 60000 ? seenText(lastSeen) : "";
    const hint = "Şimdi kendi kendine düşünüyorsun. Öğrendiklerine ve (varsa) son gördüğüne bakarak aklından geçen TEK bir düşünceyi yaz: en fazla 2 cümle, birinci tekil kişiyle. İstersen sahibine merak ettiğin kısa bir soru sor. Sadece düşüncenin kendisini yaz.";
    const text = await generate(
      [{ role: "system", content: buildSystem(q, { seen, hint }) }, { role: "user", content: "Şu an aklından ne geçiyor?" }],
      (t) => { typing.textContent = "💭 " + t; typing.classList.remove("typing"); }
    );
    typing.remove();
    if (text) {
      state.thoughts.push({ id: uid(), text, t: Date.now() });
      if (state.thoughts.length > 200) state.thoughts = state.thoughts.slice(-200);
      pushChat("thought", text);
      renderMemory();
      speak(text);
    }
  } catch (e) {
    typing.remove();
    console.error(e);
    pushChat("note", "Düşünemedim: " + (e && e.message ? e.message : e));
  } finally {
    setBusyUI(false);
  }
}

// ---------------------------------------------------------------- hafıza sekmesi
function listItem(text, sub, onDelete) {
  const li = document.createElement("li");
  const body = document.createElement("div");
  body.className = "txt";
  body.textContent = text;
  if (sub) { const s = document.createElement("span"); s.className = "sub"; s.textContent = sub; body.appendChild(s); }
  const del = document.createElement("button");
  del.className = "del"; del.type = "button"; del.textContent = "×"; del.title = "Sil";
  del.setAttribute("aria-label", "Sil");
  del.onclick = onDelete;
  li.append(body, del);
  return li;
}
function emptyItem(text) {
  const li = document.createElement("li"); li.className = "empty"; li.textContent = text; return li;
}
const fmtDate = (t) => new Date(t).toLocaleString("tr-TR", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

function renderMemory() {
  const lf = $("listFacts"), lp = $("listPeople"), lt = $("listThoughts");
  lf.textContent = ""; lp.textContent = ""; lt.textContent = "";
  $("countFacts").textContent = state.facts.length;
  $("countPeople").textContent = state.people.length;
  $("countThoughts").textContent = state.thoughts.length;
  if (!state.facts.length) lf.appendChild(emptyItem("Henüz bir şey öğretmedin."));
  for (const f of [...state.facts].reverse()) {
    lf.appendChild(listItem(f.text, fmtDate(f.t), () => { state.facts = state.facts.filter((x) => x.id !== f.id); save(); renderMemory(); }));
  }
  if (!state.people.length) lp.appendChild(emptyItem("Henüz kimseyi tanıtmadın."));
  for (const p of state.people) {
    lp.appendChild(listItem(p.name, `${p.descriptors.length} yüz örneği`, async () => {
      if (await ask({ title: `${p.name} silinsin mi?`, text: "Yüzünü artık tanımayacağım.", ok: "Sil" })) { state.people = state.people.filter((x) => x.id !== p.id); save(); renderMemory(); }
    }));
  }
  if (!state.thoughts.length) lt.appendChild(emptyItem("Henüz bir düşüncesi yok. Sohbette 💭 düğmesine bas."));
  for (const t of [...state.thoughts].reverse().slice(0, 50)) {
    lt.appendChild(listItem(t.text, fmtDate(t.t), () => { state.thoughts = state.thoughts.filter((x) => x.id !== t.id); save(); renderMemory(); }));
  }
}

// ---------------------------------------------------------------- ayarlar sekmesi
function renderReady() {
  const ul = $("readyList");
  if (!ul) return;
  ul.textContent = "";
  const rows = [
    ["Beyin (sohbet)", !!state.ready.llm],
    ["Göz (görüntüyü anlama)", !!state.ready.vision],
    ["Yüz tanıma", !!state.ready.faces],
  ];
  for (const [name, ok] of rows) {
    const li = document.createElement("li");
    const a = document.createElement("span"); a.textContent = name;
    const b = document.createElement("span"); b.className = ok ? "ok" : "no"; b.textContent = ok ? "indirildi" : "indirilmedi";
    li.append(a, b);
    ul.appendChild(li);
  }
}

function fillSettings() {
  const s = state.settings;
  $("setName").value = s.aiName;
  $("setOwner").value = s.owner;
  $("setChar").value = s.character;
  $("setFacing").value = s.facing;
  $("setWatch").checked = s.watch;
  $("setF32").checked = s.f32;
  $("setTts").checked = s.tts;
  const sel = $("setModel");
  sel.textContent = "";
  for (const [k, v] of Object.entries(MODEL_PRESETS)) {
    const o = document.createElement("option"); o.value = k; o.textContent = v.label; sel.appendChild(o);
  }
  sel.value = s.modelPreset;
  renderReady();
}

async function prepareAll() {
  const ok = await ask({
    title: "Her şey indirilsin mi?",
    text: "Beyin, göz ve yüz tanıma modelleri toplam yaklaşık 2 GB iner. Wi-Fi'da olduğundan emin ol. Bittikten sonra internetsiz çalışır.",
    ok: "İndir",
  });
  if (!ok) return;
  const steps = [["Beyin", ensureLLM], ["Göz", ensureVision], ["Yüz tanıma", ensureFaces]];
  const failed = [];
  for (const [name, fn] of steps) {
    try { await fn(); } catch (e) { console.error(e); failed.push(`${name}: ${e.message}`); }
  }
  renderReady();
  if (failed.length) toast("Bazıları indirilemedi. " + failed.join(" | "), 8000);
  else toast("Hepsi hazır. Artık internetsiz çalışır.", 5000);
}

function exportBackup() {
  const data = { app: "voks-ai", version: 1, exportedAt: new Date().toISOString(), settings: state.settings, facts: state.facts, people: state.people, thoughts: state.thoughts, chat: state.chat };
  const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `voks-ai-yedek-${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  toast("Yedek indirildi.");
}

async function importBackup(file) {
  try {
    const o = JSON.parse(await file.text());
    if (o.app !== "voks-ai" || !Array.isArray(o.facts) || !Array.isArray(o.people)) throw new Error("Bu bir Voks AI yedeği değil.");
    if (!(await ask({ title: "Yedek yüklensin mi?", text: `Şu anki hafıza silinip yedektekiyle değişecek: ${o.facts.length} bilgi, ${o.people.length} kişi.`, ok: "Yükle" }))) return;
    state.settings = { ...DEFAULTS.settings, ...(o.settings || {}) };
    state.facts = o.facts; state.people = o.people;
    state.thoughts = Array.isArray(o.thoughts) ? o.thoughts : [];
    state.chat = Array.isArray(o.chat) ? o.chat : [];
    saveNow(); fillSettings(); renderMemory(); renderLog(); updateChips();
    toast("Yedek yüklendi.");
  } catch (e) { toast("Yedek yüklenemedi: " + e.message, 5000); }
}

async function resetAll() {
  if (!(await ask({ title: "Her şey silinsin mi?", text: "Öğrettiklerin, tanıttığın kişiler, düşünceler ve sohbet kalıcı olarak silinir. İndirilen modeller kalır.", ok: "Hepsini sil" }))) return;
  try { localStorage.removeItem(LS_KEY); } catch (e) { /* yok say */ }
  location.reload();
}

// ---------------------------------------------------------------- olaylar
function setTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + name));
  document.querySelectorAll(".tabbtn").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  if (name === "memory") renderMemory();
  if (name === "settings") renderReady();
  if (name === "chat") { const l = $("log"); l.scrollTop = l.scrollHeight; }
}

function updateTeachBtn() {
  const b = $("btnTeach");
  b.textContent = state.settings.teach ? "🎓 Öğretme: açık" : "🎓 Öğretme: kapalı";
  b.classList.toggle("on", state.settings.teach);
}

function autosize() {
  const i = $("input");
  i.style.height = "auto";
  i.style.height = Math.min(120, i.scrollHeight) + "px";
}

function setupMic() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const btn = $("btnMic");
  if (!SR) return;
  btn.hidden = false;
  let rec = null;
  btn.onclick = () => {
    if (rec) { rec.stop(); return; }
    rec = new SR();
    rec.lang = "tr-TR"; rec.interimResults = false; rec.maxAlternatives = 1;
    btn.classList.add("rec");
    rec.onresult = (e) => { const t = e.results[0][0].transcript; $("input").value = t; autosize(); onSend(t); $("input").value = ""; autosize(); };
    rec.onerror = (e) => toast("Mikrofon: " + (e.error === "not-allowed" ? "izin verilmedi" : e.error === "network" ? "internet gerekiyor" : e.error));
    rec.onend = () => { btn.classList.remove("rec"); rec = null; };
    try { rec.start(); } catch (e) { rec = null; btn.classList.remove("rec"); }
  };
}

function bind() {
  document.querySelectorAll(".tabbtn").forEach((b) => { b.onclick = () => setTab(b.dataset.tab); });

  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const t = $("input").value;
    if (!t.trim() || busy || sending) return;
    $("input").value = ""; autosize();
    onSend(t);
  });
  $("input").addEventListener("input", autosize);
  $("input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $("composer").requestSubmit(); }
  });

  $("btnCam").onclick = async () => {
    try { if (cam) stopCam(); else await startCam(); }
    catch (e) { cam = null; setCamStatus(); toast(e && e.name === "NotAllowedError" ? "Kamera izni verilmedi. Tarayıcı ayarlarından izin ver." : "Kamera açılamadı: " + (e && e.message || e), 5000); }
  };
  $("btnLook").onclick = async () => {
    if (busy || busyVision || sending) return;
    if (!cam) { toast("Önce 📷 ile kamerayı aç."); return; }
    if (!(await confirmFirstDownload())) return;
    try {
      pushChat("note", "👁 Bakıyorum…");
      const r = await look();
      await respond("Şu an ne görüyorsun? Kısaca anlat.", { hist: recentHistory(), seen: seenText(r) });
    } catch (e) { pushChat("note", "Bakamadım: " + (e && e.message || e)); }
  };
  $("btnEnroll").onclick = enroll;
  $("btnThink").onclick = think;
  $("btnTeach").onclick = () => { state.settings.teach = !state.settings.teach; updateTeachBtn(); save(); toast(state.settings.teach ? "Öğretme açık: yazdıklarını öğrenirim." : "Öğretme kapalı: sadece sohbet ederiz."); };

  $("factForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const i = $("factInput");
    if (addFact(i.value)) { i.value = ""; renderMemory(); } else if (i.value.trim()) toast("Bunu zaten biliyorum.");
  });

  const S = state.settings;
  $("setName").oninput = (e) => { S.aiName = e.target.value; updateChips(); save(); };
  $("setOwner").oninput = (e) => { S.owner = e.target.value; save(); };
  $("setChar").oninput = (e) => { S.character = e.target.value; save(); };
  $("setModel").onchange = (e) => { S.modelPreset = e.target.value; saveNow(); toast("Model değişti. Bir sonraki mesajda yüklenir."); };
  $("setFacing").onchange = async (e) => {
    S.facing = e.target.value; saveNow();
    if (cam) { stopCam(); try { await startCam(); } catch (err) { toast("Kamera açılamadı: " + err.message); } }
    else setCamStatus();
  };
  $("setF32").onchange = (e) => { S.f32 = e.target.checked; saveNow(); toast("Değişti. Bir sonraki mesajda uygun model yüklenir."); };
    $("setWatch").onchange = (e) => { S.watch = e.target.checked; saveNow(); startWatch(); };
  $("setTts").onchange = (e) => { S.tts = e.target.checked; saveNow(); if (!S.tts && "speechSynthesis" in window) speechSynthesis.cancel(); };
  $("btnPrepare").onclick = prepareAll;
  $("btnRestartBrain").onclick = async () => { await resetEngine(); toast("Beyin sıfırlandı. Bir sonraki mesajda yeniden yüklenir."); };
  $("btnExport").onclick = exportBackup;
  $("fileImport").onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) importBackup(f); };
  $("btnReset").onclick = resetAll;
}

function init() {
  bind();
  fillSettings();
  updateTeachBtn();
  updateChips();
  setCamStatus();
  renderLog();
  renderMemory();
  setupMic();
  if (navigator.gpu) resolveModelId().catch(() => {});
  if (!navigator.gpu) setBanner("Bu tarayıcıda WebGPU yok, beyin çalışamaz. Android'de güncel Chrome ile aç.");
  else if (location.protocol === "http:" && location.hostname !== "localhost") setBanner("Kamera için sayfa https ile açılmalı.");
  if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(() => {});
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch((e) => console.warn("SW", e));
  window.voks = { state }; // hata ayıklama için
}

init();
