/**
 * ====================================================================================================
 * 🚀 REAL TESSERA MODEL LOADER & CHAT SERVER (BUN + @huggingface/transformers)
 * ====================================================================================================
 * Features:
 * - Real Model Downloading, Caching & In-Memory Execution
 * - Real Token Streaming over WebSockets
 * - Model Chooser API (Qwen 2.5, LLaMA 3.2, Custom HuggingFace Repos)
 * - Meridian Hybrid Vector Memory & Okapi BM25 Indexing
 * ====================================================================================================
 */

import { MeridianEngine } from "./engine/mrm_memory";
import { RealModelEngine, SUPPORTED_MODELS } from "./engine/real_model_engine";

const PORT = Number(process.env.PORT) || 3000;

// Initialize Core Singletons
const meridian = new MeridianEngine();
const realEngine = new RealModelEngine(meridian);

console.log("=" .repeat(80));
console.log("  🧠 REAL TESSERA MODEL LOADER SERVER (BUN + HUGGING FACE ENGINE)");
console.log("=" .repeat(80));
console.log(`✓ Active Runtime:   Bun ${Bun.version} on ${process.platform} (${process.arch})`);
console.log(`✓ Real Inference:   @huggingface/transformers (ONNX Runtime / WASM / WebGPU)`);
console.log(`✓ Inbuilt Memory:   Meridian Hybrid (HNSW Dense + Okapi BM25 RRF)`);
console.log(`✓ Server URL:       http://localhost:${PORT}\n`);

// Pre-initialize embedding extractor in background
realEngine.initEmbeddingExtractor().catch(err => {
  console.warn("Embedding extractor init error:", err);
});

const server = Bun.serve<{ id: string }>({
  port: PORT,

  websocket: {
    open(ws) {
      ws.data = { id: Math.random().toString(36).substring(2, 9) };
    },
    async message(ws, message) {
      try {
        const data = JSON.parse(typeof message === "string" ? message : new TextDecoder().decode(message));

        if (data.type === "ping") {
          ws.send(JSON.stringify({ type: "pong" }));
          return;
        }

        // Real Model Loading Request over WebSocket (for live download progress)
        if (data.type === "load_model") {
          const modelId = data.modelId;
          ws.send(JSON.stringify({ type: "load_status", status: "downloading", message: `Downloading ${modelId}...` }));
          
          try {
            const info = await realEngine.loadModel(modelId, (p) => {
              if (p.status === "progress" && p.total) {
                const percent = Math.round((p.loaded / p.total) * 100);
                ws.send(JSON.stringify({
                  type: "load_progress",
                  percent,
                  file: p.file || "weights.onnx",
                  loaded: p.loaded,
                  total: p.total
                }));
              }
            });
            ws.send(JSON.stringify({ type: "model_loaded", model: info }));
          } catch (err: any) {
            ws.send(JSON.stringify({ type: "load_error", error: err.message }));
          }
          return;
        }

        // Real Chat & Ingestion Request
        if (data.type === "chat") {
          const prompt = (data.prompt || "").trim();
          if (!prompt) return;

          // Check if large document (> 150 words) -> Auto-Ingest into Meridian
          const words = prompt.split(/\s+/).filter((w: string) => w.length > 0);
          if (words.length > 150) {
            const ingestResult = meridian.ingestText(prompt);
            ws.send(JSON.stringify({
              type: "ingested",
              chunks: ingestResult.count,
              words: ingestResult.words,
              timeMs: ingestResult.timeMs,
              stats: meridian.getStats()
            }));
            ws.send(JSON.stringify({
              type: "token",
              token: `📖 Successfully ingested manuscript into Meridian Vector Memory! (${ingestResult.words.toLocaleString()} words across ${ingestResult.count} chunks in ${ingestResult.timeMs.toFixed(1)}ms). You can now ask questions about the text.`
            }));
            ws.send(JSON.stringify({ type: "done", latencyMs: ingestResult.timeMs }));
            return;
          }

          // 1. Recall relevant context from Meridian Hybrid Memory
          const { recalled, latencyMs: recallMs } = meridian.hybridRecall(prompt, 8);
          ws.send(JSON.stringify({
            type: "recalled_memory",
            chunks: recalled,
            recallLatencyMs: recallMs
          }));

          // 2. Stream Real Token Generation from the loaded neural model
          const t0 = performance.now();
          for await (const token of realEngine.streamChat(prompt, recalled)) {
            ws.send(JSON.stringify({ type: "token", token }));
          }
          const genLatencyMs = performance.now() - t0;
          ws.send(JSON.stringify({ type: "done", latencyMs: genLatencyMs }));
        }
      } catch (err: any) {
        ws.send(JSON.stringify({ type: "error", error: err?.message || "Unknown error" }));
      }
    }
  },

  async fetch(req, server) {
    const url = new URL(req.url);

    // WebSocket Upgrade
    if (url.pathname === "/ws") {
      const success = server.upgrade(req);
      if (success) return undefined;
      return new Response("WebSocket upgrade failed", { status: 400 });
    }

    // Static Assets
    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(Bun.file("public/index.html"), { headers: { "Content-Type": "text/html; charset=utf-8" } });
    }
    if (url.pathname === "/style.css") {
      return new Response(Bun.file("public/style.css"), { headers: { "Content-Type": "text/css; charset=utf-8" } });
    }
    if (url.pathname === "/app.js") {
      return new Response(Bun.file("public/app.js"), { headers: { "Content-Type": "application/javascript; charset=utf-8" } });
    }

    // REST API: GET /api/models
    if (url.pathname === "/api/models" && req.method === "GET") {
      return Response.json({
        activeId: realEngine.activeModelId || "onnx-community/Qwen2.5-0.5B-Instruct",
        supportedModels: SUPPORTED_MODELS,
        isLoading: realEngine.isLoading,
        loadProgress: realEngine.loadProgress
      });
    }

    // REST API: POST /api/load-model
    if (url.pathname === "/api/load-model" && req.method === "POST") {
      const body = await req.json();
      const modelId = body.modelId || "onnx-community/Qwen2.5-0.5B-Instruct";
      try {
        const info = await realEngine.loadModel(modelId);
        return Response.json({ success: true, model: info });
      } catch (err: any) {
        return Response.json({ success: false, error: err.message }, { status: 500 });
      }
    }

    // REST API: POST /api/ingest
    if (url.pathname === "/api/ingest" && req.method === "POST") {
      const body = await req.json();
      const text = body.text || "";
      if (!text.trim()) return Response.json({ error: "Empty text" }, { status: 400 });

      const res = meridian.ingestText(text, body.chunkSize || 250, body.overlap || 40);
      return Response.json({
        success: true,
        ...res,
        stats: meridian.getStats()
      });
    }

    // REST API: GET /api/memory
    if (url.pathname === "/api/memory" && req.method === "GET") {
      const chunks = Array.from(meridian.corpusById.values()).slice(0, 100);
      return Response.json({
        stats: meridian.getStats(),
        chunksPreview: chunks
      });
    }

    // REST API: POST /api/clear-memory
    if (url.pathname === "/api/clear-memory" && req.method === "POST") {
      meridian.clear();
      return Response.json({ success: true, stats: meridian.getStats() });
    }

    return new Response("Not Found", { status: 404 });
  }
});
