import faiss
import json
import numpy as np
import os
from dotenv import load_dotenv
from google import genai

# -------------------------
# Load env & Gemini client
# -------------------------
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = "models/text-embedding-004"
DIM = 768

# -------------------------
# Load FAISS index
# -------------------------
index = faiss.read_index("rag_index.faiss")

# -------------------------
# Load metadata store
# -------------------------
with open("metadata_store.json", "r", encoding="utf-8") as f:
    metadata_store = json.load(f)

# -------------------------
# Search function
# -------------------------
def search(query, top_k=3):
    response = client.models.embed_content(
        model=MODEL,
        contents=query
    )

    vector = np.array(response.embeddings[0].values).astype("float32")
    distances, indices = index.search(vector.reshape(1, -1), top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        r = metadata_store[idx].copy()   # important: copy to avoid mutation
        r["distance"] = float(dist)
        results.append(r)

    return results

# -------------------------
# Handle search results
# -------------------------
def handle_results(results):
    if not results:
        print("❌ No match found")
        return

    # sort by distance
    results = sorted(results, key=lambda x: x["distance"])

    # ---------- CONFIRMATION CASE ----------
    if len(results) > 1 and abs(results[0]["distance"] - results[1]["distance"]) < 0.05:
        print("\nMultiple matches found. Please confirm:\n")

        for i, r in enumerate(results, 1):
            meta = r["metadata"]

            # 👤 USER → short info only
            if r["type"] == "user":
                print(
                    f"{i}️⃣ {meta.get('user_name', '').title()} | "
                    f"{meta.get('designation')} | "
                    f"{meta.get('hierarchy')}"
                )

            # 👥 GROUP → ONLY name + level (no content)
            elif r["type"] == "group":
                print(
                    f"{i}️⃣ {meta.get('group_name', '').title()} | "
                    f"Level: {meta.get('level')}"
                )

        choice = input("\nSelect option (1/2/3): ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(results):
            selected = results[int(choice) - 1]
            meta = selected["metadata"]

            print("\n✅ Selected:")

            # 👤 USER (NO LEVEL)
            if selected["type"] == "user":
                print(f"Name: {meta.get('user_name', '').title()}")
                print(f"Designation: {meta.get('designation')}")
                print(f"Hierarchy: {meta.get('hierarchy')}")

            # 👥 GROUP (FULL DETAILS)
            elif selected["type"] == "group":
                print(f"Group Name: {meta.get('group_name', '').title()}")
                print(f"Level: {meta.get('level')}")
                print(f"Description: {selected['content']}")
        else:
            print("❌ Invalid selection")

    # ---------- SINGLE MATCH CASE ----------
    else:
        selected = results[0]
        meta = selected["metadata"]

        print("\n✅ Match found:")

        # 👤 USER → NO LEVEL
        if selected["type"] == "user":
            print(f"Name: {meta.get('user_name', '').title()}")
            print(f"Designation: {meta.get('designation')}")
            print(f"Hierarchy: {meta.get('hierarchy')}")

        # 👥 GROUP
        elif selected["type"] == "group":
            print(f"Group Name: {meta.get('group_name', '').title()}")
            print(f"Level: {meta.get('level')}")
            print(f"Description: {selected['content']}")


# -------------------------
# Main loop
# -------------------------
if __name__ == "__main__":
    query = input("🔍 Search: ").strip()
    results = search(query)
    handle_results(results)
