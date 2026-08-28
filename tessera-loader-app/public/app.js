/**
 * ====================================================================================================
 * ⚡ REAL TESSERA CLIENT APPLICATION (REAL HUGGINGFACE INFERENCE IN BUN)
 * ====================================================================================================
 */

let ws = null;
let isGenerating = false;
let currentAssistantMsgElement = null;
let activeModel = {
  id: "onnx-community/Qwen2.5-0.5B-Instruct",
  name: "Qwen 2.5 - 0.5B Instruct",
  precision: "q4 (Quantized)",
  sizeMb: "350 MB",
  paramCount: "490M"
};

document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) lucide.createIcons();
  initWebSocket();
  fetchModels();
  fetchMemoryStats();
  initInputListeners();
});

// ====================================================================================================
// 1. WEBSOCKET CONNECTION
// ====================================================================================================
function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("✓ Connected to Real Tessera WebSocket Engine");
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === "load_progress") {
      showDownloadProgress(data.percent, data.file);
    } else if (data.type === "model_loaded") {
      hideDownloadProgress();
      updateModelBanner(data.model);
      alert(`✓ Model [${data.model.name}] Loaded Successfully into Memory!`);
    } else if (data.type === "load_error") {
      hideDownloadProgress();
      alert("Failed to load model: " + data.error);
    } else if (data.type === "recalled_memory") {
      renderRecalledMemory(data.chunks, data.recallLatencyMs);
    } else if (data.type === "token") {
      appendAssistantToken(data.token);
    } else if (data.type === "done") {
      finishAssistantGeneration(data.latencyMs);
    } else if (data.type === "ingested") {
      fetchMemoryStats();
    } else if (data.type === "error") {
      alert("Error: " + data.error);
      isGenerating = false;
      toggleSendButton(true);
    }
  };

  ws.onclose = () => {
    setTimeout(initWebSocket, 2000);
  };
}

// ====================================================================================================
// 2. REAL MODEL CHOOSER & LOADER
// ====================================================================================================
function onModelSelectChange() {
  const select = document.getElementById("model-select");
  const customBar = document.getElementById("custom-model-bar");
  if (select.value === "custom") {
    customBar.classList.remove("hidden");
    document.getElementById("custom-model-input").focus();
  } else {
    customBar.classList.add("hidden");
  }
}

function loadSelectedModel() {
  const select = document.getElementById("model-select");
  if (select.value === "custom") {
    loadCustomModel();
    return;
  }
  triggerModelLoad(select.value);
}

function loadCustomModel() {
  const input = document.getElementById("custom-model-input");
  const modelId = input.value.trim();
  if (!modelId) {
    alert("Please enter a Hugging Face Repo ID or local path!");
    return;
  }
  triggerModelLoad(modelId);
}

function triggerModelLoad(modelId) {
  showDownloadProgress(0, "Fetching model weights...");
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "load_model", modelId }));
  }
}

function showDownloadProgress(percent, file) {
  const bar = document.getElementById("download-progress-bar");
  bar.classList.remove("hidden");
  document.getElementById("download-percent").textContent = `${percent}%`;
  document.getElementById("download-progress-fill").style.width = `${percent}%`;
  document.getElementById("download-status-text").innerHTML = `
    <i data-lucide="loader-2" class="w-3.5 h-3.5 animate-spin text-indigo-400"></i>
    <span>Loading ${file || 'weights'}... (${percent}%)</span>
  `;
  if (window.lucide) lucide.createIcons();
}

function hideDownloadProgress() {
  const bar = document.getElementById("download-progress-bar");
  bar.classList.add("hidden");
}

async function fetchModels() {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    if (data.activeId) {
      const match = data.supportedModels.find((m) => m.id === data.activeId);
      if (match) updateModelBanner(match);
    }
  } catch (err) {
    console.error("Failed to fetch models:", err);
  }
}

function updateModelBanner(model) {
  activeModel = model;
  document.getElementById("banner-model-name").textContent = model.name || model.id;
  document.getElementById("banner-precision").textContent = model.precision || "Dynamic";
  document.getElementById("banner-vram").textContent = model.sizeMb || "Dynamic";
  document.getElementById("banner-params").textContent = model.paramCount || "Dynamic";
}

// ====================================================================================================
// 3. PROMPT SUBMISSION & "SEE MORE" COLLAPSE SYSTEM
// ====================================================================================================
function handleSend() {
  if (isGenerating) return;
  const inputEl = document.getElementById("prompt-input");
  const prompt = inputEl.value.trim();
  if (!prompt) return;

  inputEl.value = "";
  updateCounters("");

  // Render User Message with Virtualized "See More" if text is long
  renderUserMessage(prompt);

  // Send to Real Neural Engine
  isGenerating = true;
  toggleSendButton(false);
  createAssistantMessageContainer();

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "chat", prompt }));
  }
}

function renderUserMessage(text) {
  const container = document.getElementById("chat-container");
  const words = text.split(/\s+/).filter(w => w.length > 0);
  const isLarge = words.length > 80;
  const uniqueId = "msg-" + Math.random().toString(36).substr(2, 9);

  const wrapper = document.createElement("div");
  wrapper.className = "flex justify-end animate-fade-in";

  let contentHtml = "";
  if (isLarge) {
    // Collapsible Prompt Card with Word Count Badge
    contentHtml = `
      <div class="max-w-2xl bg-indigo-950/40 border border-indigo-500/30 rounded-2xl p-4 shadow-lg">
        <div class="flex items-center justify-between gap-3 mb-2 pb-2 border-b border-indigo-500/20 text-xs text-indigo-300 font-medium">
          <span class="flex items-center gap-1.5">
            <i data-lucide="file-text" class="w-3.5 h-3.5"></i>
            Ingested Large Document / Manuscript
          </span>
          <span class="px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 font-semibold text-[11px]">
            ${words.length.toLocaleString()} words (${text.length.toLocaleString()} chars)
          </span>
        </div>
        <div id="${uniqueId}-content" class="prompt-collapsed text-xs sm:text-sm text-slate-200 whitespace-pre-wrap font-sans leading-relaxed">
          ${escapeHtml(text)}
        </div>
        <button onclick="togglePromptCollapse('${uniqueId}-content', this)" class="mt-2 text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1">
          <span>Show Full Text (${words.length.toLocaleString()} words)</span>
          <i data-lucide="chevron-down" class="w-3.5 h-3.5"></i>
        </button>
      </div>
    `;
  } else {
    contentHtml = `
      <div class="max-w-xl bg-indigo-600 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm shadow-md leading-relaxed whitespace-pre-wrap">
        ${escapeHtml(text)}
      </div>
    `;
  }

  wrapper.innerHTML = contentHtml;
  container.appendChild(wrapper);
  if (window.lucide) lucide.createIcons();
  scrollToBottom();
}

function togglePromptCollapse(elementId, btn) {
  const el = document.getElementById(elementId);
  if (!el) return;

  if (el.classList.contains("prompt-collapsed")) {
    el.classList.remove("prompt-collapsed");
    el.classList.add("prompt-expanded");
    btn.innerHTML = `<span>Show Less</span> <i data-lucide="chevron-up" class="w-3.5 h-3.5"></i>`;
  } else {
    el.classList.remove("prompt-expanded");
    el.classList.add("prompt-collapsed");
    btn.innerHTML = `<span>Show Full Text</span> <i data-lucide="chevron-down" class="w-3.5 h-3.5"></i>`;
  }
  if (window.lucide) lucide.createIcons();
}

// ====================================================================================================
// 4. MEMORY RECALL & ASSISTANT STREAMING RENDERER
// ====================================================================================================
function renderRecalledMemory(chunks, latencyMs) {
  if (!chunks || chunks.length === 0) return;
  const container = document.getElementById("chat-container");
  const uniqueId = "mem-" + Math.random().toString(36).substr(2, 9);

  const card = document.createElement("div");
  card.className = "max-w-2xl bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 animate-fade-in text-xs";

  card.innerHTML = `
    <div class="flex items-center justify-between cursor-pointer" onclick="toggleAccordion('${uniqueId}')">
      <div class="flex items-center gap-2 text-cyan-400 font-medium">
        <i data-lucide="brain-circuit" class="w-4 h-4"></i>
        <span>Meridian Recalled ${chunks.length} Memory Chunks</span>
        <span class="text-slate-500 font-normal">in ${latencyMs.toFixed(1)}ms</span>
      </div>
      <i id="${uniqueId}-icon" data-lucide="chevron-down" class="w-4 h-4 text-slate-400 transition-transform"></i>
    </div>
    <div id="${uniqueId}-body" class="hidden mt-3 pt-3 border-t border-slate-800/80 space-y-2 max-h-60 overflow-y-auto pr-1">
      ${chunks.map((c, i) => `
        <div class="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/60">
          <div class="flex items-center justify-between text-[11px] text-slate-400 mb-1">
            <span class="font-mono text-cyan-300 font-semibold">Chunk #${c.id}</span>
            <span>RRF Score: <strong class="text-slate-200">${c.rrfScore.toFixed(3)}</strong> (Dense: ${c.denseScore.toFixed(2)} | BM25: ${c.bm25Score.toFixed(1)})</span>
          </div>
          <p class="text-slate-300 text-[11px] leading-relaxed line-clamp-3">${escapeHtml(c.text)}</p>
        </div>
      `).join("")}
    </div>
  `;

  container.appendChild(card);
  if (window.lucide) lucide.createIcons();
  scrollToBottom();
}

function toggleAccordion(id) {
  const body = document.getElementById(`${id}-body`);
  const icon = document.getElementById(`${id}-icon`);
  if (!body) return;

  if (body.classList.contains("hidden")) {
    body.classList.remove("hidden");
    icon.classList.add("rotate-180");
  } else {
    body.classList.add("hidden");
    icon.classList.remove("rotate-180");
  }
}

function createAssistantMessageContainer() {
  const container = document.getElementById("chat-container");
  const wrapper = document.createElement("div");
  wrapper.className = "flex gap-3.5 max-w-3xl animate-fade-in";

  wrapper.innerHTML = `
    <div class="w-8 h-8 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center flex-shrink-0 text-emerald-400 mt-0.5">
      <i data-lucide="bot" class="w-4 h-4"></i>
    </div>
    <div class="flex-1 space-y-1">
      <div class="flex items-center gap-2 text-xs text-slate-400">
        <span class="font-semibold text-slate-200">Tessera</span>
        <span class="text-[10px] px-1.5 py-0.2 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">${activeModel.name}</span>
      </div>
      <div class="assistant-content text-sm text-slate-100 leading-relaxed font-sans prose-custom whitespace-pre-wrap"></div>
    </div>
  `;

  container.appendChild(wrapper);
  currentAssistantMsgElement = wrapper.querySelector(".assistant-content");
  if (window.lucide) lucide.createIcons();
  scrollToBottom();
}

function appendAssistantToken(token) {
  if (currentAssistantMsgElement) {
    currentAssistantMsgElement.textContent += token;
    scrollToBottom();
  }
}

function finishAssistantGeneration(latencyMs) {
  isGenerating = false;
  toggleSendButton(true);
  currentAssistantMsgElement = null;
}

// ====================================================================================================
// 5. MEMORY STATS & CHUNK INSPECTOR MODAL
// ====================================================================================================
async function fetchMemoryStats() {
  try {
    const res = await fetch("/api/memory");
    const data = await res.json();
    const stats = data.stats;

    document.getElementById("stat-chunks").textContent = stats.totalChunks.toLocaleString();
    document.getElementById("stat-words").textContent = stats.totalWords.toLocaleString();

    const pill = document.getElementById("memory-pill");
    if (stats.totalChunks > 0) {
      pill.classList.remove("hidden");
    }
  } catch (err) {
    console.error("Failed to fetch memory stats:", err);
  }
}

async function openChunkModal() {
  const modal = document.getElementById("chunk-modal");
  const modalBody = document.getElementById("chunk-modal-body");
  modal.classList.remove("hidden");

  try {
    const res = await fetch("/api/memory");
    const data = await res.json();
    const chunks = data.chunksPreview || [];

    if (chunks.length === 0) {
      modalBody.innerHTML = `<p class="text-slate-500 text-center py-8">No chunks currently stored in Meridian Memory.</p>`;
    } else {
      modalBody.innerHTML = chunks.map(c => `
        <div class="p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex flex-col gap-1.5">
          <div class="flex items-center justify-between text-xs">
            <span class="font-mono text-cyan-400 font-semibold">Chunk #${c.id}</span>
            <span class="text-slate-400">${c.wordCount} words • ~${c.tokenCount} tokens</span>
          </div>
          <p class="text-slate-300 leading-relaxed">${escapeHtml(c.text)}</p>
        </div>
      `).join("");
    }

    document.getElementById("modal-footer-stats").textContent = 
      `${data.stats.totalChunks} chunks • ${data.stats.totalWords.toLocaleString()} words • ${data.stats.uniqueBm25Terms} BM25 terms`;
  } catch (err) {
    modalBody.innerHTML = `<p class="text-red-400 text-center py-4">Failed to load chunks.</p>`;
  }
}

function closeChunkModal() {
  document.getElementById("chunk-modal").classList.add("hidden");
}

async function clearMemory() {
  if (!confirm("Are you sure you want to wipe all Meridian long-term memory?")) return;
  try {
    await fetch("/api/clear-memory", { method: "POST" });
    fetchMemoryStats();
    alert("Meridian memory wiped successfully.");
  } catch (err) {
    console.error("Failed to clear memory:", err);
  }
}

// ====================================================================================================
// 6. UTILITIES & INPUT LISTENERS
// ====================================================================================================
function initInputListeners() {
  const inputEl = document.getElementById("prompt-input");

  inputEl.addEventListener("input", () => {
    updateCounters(inputEl.value);
  });

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });
}

function updateCounters(text) {
  const chars = text.length;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  document.getElementById("char-counter").textContent = `${chars.toLocaleString()} chars`;
  document.getElementById("word-counter").textContent = `${words.toLocaleString()} words`;
}

function toggleSendButton(enabled) {
  const btn = document.getElementById("send-btn");
  btn.disabled = !enabled;
}

function insertSampleQuery(q) {
  const inputEl = document.getElementById("prompt-input");
  inputEl.value = q;
  updateCounters(q);
  inputEl.focus();
}

function scrollToBottom() {
  const container = document.getElementById("chat-container");
  container.scrollTop = container.scrollHeight;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
