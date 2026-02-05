import json
import os
import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai

# -----------------------------
# CONFIG
# -----------------------------

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = "text-embedding-004"
DIM = 768

# -----------------------------
# LOAD RAG CHUNKS
# -----------------------------

with open("rag_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# -----------------------------
# CREATE FAISS INDEX
# -----------------------------

index = faiss.IndexFlatL2(DIM)
metadata_store = []

# -----------------------------
# EMBED + STORE
# -----------------------------

for chunk in chunks:
    text = chunk["content"]

    response = client.models.embed_content(
        model=MODEL,
        contents=text
    )

    vector = np.array(
        response.embeddings[0].values,
        dtype="float32"
    )

    if vector.shape[0] != DIM:
        raise ValueError("Embedding dimension mismatch")

    index.add(vector.reshape(1, -1))
    metadata_store.append(chunk)

# -----------------------------
# SAVE
# -----------------------------

faiss.write_index(index, "rag_index.faiss")

with open("metadata_store.json", "w", encoding="utf-8") as f:
    json.dump(metadata_store, f, indent=2, ensure_ascii=False)

print("✅ Embeddings created & stored successfully")
