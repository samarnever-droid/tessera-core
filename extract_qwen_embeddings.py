"""
Download Qwen model and extract real semantic embeddings for Tessera-Meridian with batching.
"""

import os
import json
import struct
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading {MODEL_NAME} via HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32, trust_remote_code=True)
model.eval()

# Sample real knowledge corpus
DOCUMENTS = [
    {"id": 1, "title": "Meridian Vector Architecture", "text": "Meridian is a low-latency high-throughput vector database featuring Lehman-Yao B+ Trees, AVX2 SIMD Euclidean routing, and hardware POPCNT Binary Quantization."},
    {"id": 2, "title": "Tessera Deep Learning", "text": "Tessera is a frontier neural architecture integrating Microsoft Differential Attention, Adaptive RoPE, and native inbuilt long-term vector memory."},
    {"id": 3, "title": "Cahill Serializable Snapshot Isolation", "text": "Serializable Snapshot Isolation uses write-intent locks and SIREAD locks to prevent write skew anomalies in distributed transaction engines."},
    {"id": 4, "title": "Quantum Error Correction", "text": "Surface codes use topological 2D lattices of physical qubits to detect phase flips and bit flips without measuring quantum eigenstates directly."},
    {"id": 5, "title": "High Performance Computing", "text": "Direct memory access and cache line prefetching enable multi-core processors to sustain gigabytes per second memory throughput."},
]

topics = [
    "distributed systems database replication consensus raft paxos",
    "compiler optimization LLVM intermediate representation register allocation",
    "neural network transformer self attention key value caching feedforward",
    "operating system kernel virtual memory page tables TLB context switch",
    "cryptographic hashing elliptic curve digital signature zero knowledge proof"
]

all_docs = list(DOCUMENTS)
doc_id = 100
for i in range(2000):
    topic = topics[i % len(topics)]
    all_docs.append({
        "id": doc_id,
        "title": f"Document #{doc_id}",
        "text": f"Knowledge base entry #{doc_id} covering {topic} with index parameter {i*17} and shard routing hash {hex(i*31)}."
    })
    doc_id += 1

print(f"Total documents to embed: {len(all_docs)}")

def get_embeddings_batched(texts: list[str], batch_size: int = 64) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.inference_mode():
            outputs = model(**inputs)
            # Masked mean pooling
            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embs = torch.sum(outputs.last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = (sum_embs / sum_mask).numpy()
            
            # Normalize
            norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True) + 1e-8
            normed = mean_pooled / norms
            all_embs.append(normed)
        if (i + batch_size) % 512 == 0 or (i + batch_size) >= len(texts):
            print(f"  -> Processed {min(i + batch_size, len(texts))} / {len(texts)} texts...")
    return np.vstack(all_embs)

texts = [doc["text"] for doc in all_docs]
embs = get_embeddings_batched(texts, batch_size=64)
dim = embs.shape[1]
print(f"Computed embeddings shape: {embs.shape} (Dimension: {dim})")

QUERIES = [
    {"target_id": 1, "query": "What vector database uses AVX2 SIMD and Lehman-Yao B+ Trees?", "type": "Semantic Paraphrase"},
    {"target_id": 2, "query": "Tell me about the Tessera architecture with Differential Attention and vector memory.", "type": "Semantic Paraphrase"},
    {"target_id": 3, "query": "How does Cahill Serializable Snapshot Isolation prevent write skew anomalies?", "type": "Exact Lexical"},
    {"target_id": 4, "query": "Topological 2D lattices of physical qubits detecting phase flips in surface codes.", "type": "Rare Identifier"},
    {"target_id": 5, "query": "DMA and cache line prefetching for high bandwidth memory throughput.", "type": "Adversarial Context"},
]

query_texts = [q["query"] for q in QUERIES]
query_embs = get_embeddings_batched(query_texts, batch_size=len(query_texts))

query_data = []
for idx, q in enumerate(QUERIES):
    query_data.append({
        "target_id": q["target_id"],
        "query_text": q["query"],
        "query_type": q["type"],
        "vector": query_embs[idx].tolist()
    })

# Export binary file
out_bin_path = "qwen_real_embeddings.bin"
with open(out_bin_path, "wb") as f:
    f.write(struct.pack("<II", len(all_docs), dim))
    for idx, doc in enumerate(all_docs):
        f.write(struct.pack("<Q", doc["id"]))
        f.write(embs[idx].astype(np.float32).tobytes())

with open("qwen_queries.json", "w") as f:
    json.dump(query_data, f, indent=2)

print(f"✓ Successfully exported {len(all_docs)} real Qwen embeddings to {out_bin_path} ({os.path.getsize(out_bin_path) / (1024*1024):.2f} MB)")
print("✓ Successfully exported queries to qwen_queries.json")
