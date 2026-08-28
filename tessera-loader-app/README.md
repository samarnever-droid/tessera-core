# 🚀 Tessera Model Loader & Meridian Memory UI (Bun Runtime)

High-performance Universal Model Loader and Web UI built with **Bun** for running **Tessera & Qwen models** with **Inbuilt MRM Working Memory** and **Meridian Hybrid Long-Term Memory**.

---

## ✨ Features

1. **All 3 Model Formats Supported**:
   - 🦀 **Native GGUF / Rust Tessera Binary** (`.tessera`, `.gguf`)
   - ⚡ **SafeTensors / PyTorch** (`.safetensors`, `model.safetensors`)
   - 🌐 **ONNX & OpenAI Triton Graph** (`.onnx`, `.triton`)
2. **Dual-Tier Neural Memory**:
   - **Tier 1 (MRM Working Memory)**: $O(1)$ recurrent bounded state.
   - **Tier 2 (Meridian Long-Term Search)**: Dense SIMD embeddings + Okapi BM25 Inverted Index with Reciprocal Rank Fusion (RRF).
3. **RAM-Guarded Virtual UI**:
   - **Virtual DOM Chunk Ingestion**: Pasting entire 30,000-word manuscripts does not lag or freeze the browser DOM.
   - **"See More / Show Less" Prompt Collapse**: Automatically wraps long user prompts into sleek, collapsible cards with live word count badges.
   - **Real-Time WebSocket Token Streaming**: Sub-millisecond token updates with live memory recall inspector.

---

## 🏃 Running the App

```bash
cd tessera-loader-app
bun run src/server.ts
```

Open your browser at: **`http://localhost:3000`**
