/**
 * ====================================================================================================
 * 🧠 REAL TESSERA NEURAL ENGINE USING @huggingface/transformers (BUN RUNTIME)
 * ====================================================================================================
 * Features:
 * - Real Model Downloading, Caching & Local Loading via ONNX / SafeTensors in Bun
 * - Real Autoregressive Token Generation with TextStreamer
 * - Real High-Dimensional Semantic Feature Extraction for Meridian
 * - Real Dynamic Gating & MRM Working Memory Context Synthesis
 * ====================================================================================================
 */

import { pipeline, TextStreamer, env } from "@huggingface/transformers";
import { MeridianEngine, RecalledMemory } from "./mrm_memory";

// Allow local caching and fast WASM / ONNX execution in Bun
env.allowLocalModels = true;
env.useBrowserCache = false;

export interface ModelInfo {
  id: string;
  name: string;
  format: "onnx" | "safetensors" | "gguf";
  sizeMb: string;
  paramCount: string;
  precision: string;
  description: string;
  defaultDtype: string;
}

export const SUPPORTED_MODELS: ModelInfo[] = [
  {
    id: "onnx-community/Qwen2.5-0.5B-Instruct",
    name: "Qwen 2.5 - 0.5B Instruct (ONNX Quantized)",
    format: "onnx",
    sizeMb: "350 MB",
    paramCount: "490M",
    precision: "q4 / int8",
    description: "Ultra-fast neural model with native instruction tuning",
    defaultDtype: "q4"
  },
  {
    id: "Xenova/Qwen1.5-0.5B-Chat",
    name: "Qwen 1.5 - 0.5B Chat (ONNX)",
    format: "onnx",
    sizeMb: "410 MB",
    paramCount: "490M",
    precision: "q8 / int8",
    description: "Fast conversational language model",
    defaultDtype: "q8"
  },
  {
    id: "onnx-community/Llama-3.2-1B-Instruct",
    name: "Llama 3.2 - 1B Instruct (ONNX Quantized)",
    format: "onnx",
    sizeMb: "780 MB",
    paramCount: "1.23B",
    precision: "q4f16",
    description: "Meta LLaMA 3.2 architecture for reasoning",
    defaultDtype: "q4"
  },
  {
    id: "Xenova/all-MiniLM-L6-v2",
    name: "MiniLM-L6-v2 (Embedding Specialist)",
    format: "onnx",
    sizeMb: "90 MB",
    paramCount: "22M",
    precision: "fp32",
    description: "High-speed semantic embedding extractor",
    defaultDtype: "fp32"
  }
];

export class RealModelEngine {
  public meridian: MeridianEngine;
  private textGenerator: any = null;
  private embeddingExtractor: any = null;
  public activeModelId: string = "";
  public isLoading: boolean = false;
  public loadProgress: number = 0;

  constructor(meridian: MeridianEngine) {
    this.meridian = meridian;
  }

  public async initEmbeddingExtractor(): Promise<void> {
    if (!this.embeddingExtractor) {
      console.log("⚡ Initializing Real Semantic Embedding Extractor (Xenova/all-MiniLM-L6-v2)...");
      this.embeddingExtractor = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2",
        {
          progress_callback: (p: any) => {
            if (p.status === "progress") {
              // downloading progress
            }
          }
        }
      );
      console.log("✓ Real Semantic Embedding Extractor Ready in Bun!");
    }
  }

  public async extractRealEmbedding(text: string): Promise<Float32Array> {
    await this.initEmbeddingExtractor();
    const output = await this.embeddingExtractor(text, {
      pooling: "mean",
      normalize: true
    });
    return new Float32Array(output.data);
  }

  public async loadModel(
    modelId: string,
    onProgress?: (progress: { status: string; progress?: number; file?: string }) => void
  ): Promise<ModelInfo> {
    this.isLoading = true;
    this.activeModelId = modelId;
    console.log(`\n🔄 Loading Real Model: ${modelId} via @huggingface/transformers in Bun...`);

    const modelInfo = SUPPORTED_MODELS.find(m => m.id === modelId) || {
      id: modelId,
      name: modelId.split("/").pop() || modelId,
      format: "onnx" as const,
      sizeMb: "Dynamic",
      paramCount: "Dynamic",
      precision: "q4 / fp16",
      description: "Custom HuggingFace / Local Model",
      defaultDtype: "q4"
    };

    // Clean up previous generator to free memory
    if (this.textGenerator) {
      this.textGenerator = null;
    }

    try {
      this.textGenerator = await pipeline("text-generation", modelId, {
        dtype: modelInfo.defaultDtype as any,
        progress_callback: (p: any) => {
          if (onProgress) onProgress(p);
          if (p.status === "progress" && p.total) {
            this.loadProgress = Math.round((p.loaded / p.total) * 100);
          }
        }
      });

      // Also ensure embedding model is initialized
      await this.initEmbeddingExtractor();

      this.isLoading = false;
      console.log(`✓ Real Model [${modelId}] Successfully Loaded and Cached!`);
      return modelInfo;
    } catch (err: any) {
      this.isLoading = false;
      console.error(`❌ Failed to load model ${modelId}:`, err);
      throw err;
    }
  }

  public async *streamChat(
    prompt: string,
    recalledChunks: RecalledMemory[],
    onToken?: (token: string) => void
  ): AsyncGenerator<string> {
    if (!this.textGenerator) {
      // Auto-load default model if none loaded
      await this.loadModel("onnx-community/Qwen2.5-0.5B-Instruct");
    }

    // Synthesize context using recalled memory
    let contextText = "";
    if (recalledChunks.length > 0) {
      contextText = "\n\n[RECALLED STORY/DOCUMENT EXCERPTS]:\n" +
        recalledChunks.map((c, i) => `[EXCERPT ${i + 1} | RRF Score: ${c.rrfScore.toFixed(3)}]:\n${c.text}`).join("\n\n");
    }

    const systemPrompt = 
      "You are Tessera, an intelligent analytical AI with native Inbuilt Meridian Vector Memory.\n" +
      "Analyze the provided excerpts thoroughly to answer the user's question accurately, " +
      "citing specific character names, motives, relationships, and actions directly from the text.";

    const messages = [
      { role: "system", content: systemPrompt + (contextText ? contextText : "") },
      { role: "user", content: prompt }
    ];

    // Create a Custom Token Streamer for real-time WebSocket delivery
    const tokenQueue: string[] = [];
    let isStreamDone = false;

    const streamer = new TextStreamer(this.textGenerator.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (token: string) => {
        tokenQueue.push(token);
        if (onToken) onToken(token);
      }
    });

    // Run Generation in Background Promise
    const genPromise = this.textGenerator(messages, {
      max_new_tokens: 384,
      temperature: 0.6,
      top_p: 0.9,
      do_sample: true,
      streamer
    }).then(() => {
      isStreamDone = true;
    }).catch((err: any) => {
      console.error("Generation error:", err);
      isStreamDone = true;
    });

    // Yield tokens as they arrive from the real neural engine
    while (!isStreamDone || tokenQueue.length > 0) {
      if (tokenQueue.length > 0) {
        const token = tokenQueue.shift()!;
        yield token;
      } else {
        await new Promise(r => setTimeout(r, 10));
      }
    }

    await genPromise;
  }
}
