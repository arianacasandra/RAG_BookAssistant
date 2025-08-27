import os
import json
import requests
import chromadb
from typing import List, Dict

# --- Ollama settings ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# --- Chroma collection ---
client = chromadb.Client()
collection = client.create_collection(name="my_collection")

# --- Load book summaries JSON ---
with open('books_prompt_result.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

ids, documents, metadatas = [], [], []
for data in json_data["books"]:
    book_id = str(data.get('id', ''))
    title = data.get('title', '')
    summary = data.get('summary', '')

    doc = f"Title: {title}\nSummary: {summary}".strip()
    documents.append(doc)
    ids.append(book_id)
    metadatas.append({"id": book_id, "title": title})

# --- Ollama embedding helpers ---
def _ollama_embed(text: str, model: str = EMBED_MODEL) -> List[float]:
    """Get an embedding from Ollama for a single text."""
    url = f"{OLLAMA_HOST}/api/embeddings"
    payload = {"model": model, "prompt": text}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "embedding" not in data:
        raise RuntimeError(f"Ollama embeddings: unexpected response: {data}")
    return data["embedding"]

def embed_batch_ollama(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """Naive batch embedding (Ollama doesn’t support native batch embeddings)."""
    vectors = []
    for t in texts:
        vectors.append(_ollama_embed(t, model=model))
    return vectors

# --- Upsert into Chroma ---
embeddings = embed_batch_ollama(documents, model=EMBED_MODEL)
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print("Sample entries:")
print(collection.peek())

# --- Embedding + search ---
def embed_text(text: str, model: str = EMBED_MODEL) -> List[float]:
    return _ollama_embed(text, model=model)

def search_books(query: str, k: int = 1, model: str = EMBED_MODEL) -> List[Dict]:
    q_emb = embed_text(query, model=model)
    res = collection.query(query_embeddings=[q_emb], n_results=k)

    hits = []
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    ids_  = res["ids"][0]
    dists = res.get("distances", [[None]])[0]

    for i in range(len(ids_)):
        doc_text = docs[i] or ""
        summary_part = doc_text.split("Summary:", 1)[-1].strip() if "Summary:" in doc_text else doc_text
        hits.append({
            "id": metas[i].get("id", ids_[i]),
            "title": metas[i].get("title", "Unknown"),
            "score": dists[i],
            "summary_snippet": summary_part[:350] + ("..." if len(summary_part) > 350 else "")
        })
    return hits

# --- Quick index: title -> summary (exact lookup) ---
title_to_summary: Dict[str, str] = {}
for data in json_data["books"]:
    t = (data.get("title") or "").strip()
    s = (data.get("summary") or "").strip()
    if t:
        title_to_summary[t] = s

def get_summary_by_title(title: str) -> str:
    """
    Return the full summary for an EXACT TITLE.
    - Exact match (case-insensitive fallback).
    """
    if title in title_to_summary:
        return title_to_summary[title]

    tl = title.lower().strip()
    for k, v in title_to_summary.items():
        if k.lower().strip() == tl:
            return v

    return f"The title «{title}» was not found in the local database."

# --- Load inappropriate words list ---
with open("bad_words.json", "r", encoding="utf-8") as f:
    BAD_WORDS = set(json.load(f)["bad_words"])

def contains_inappropriate_language(text: str) -> bool:
    """Check if the user input contains offensive words."""
    words = text.lower().split()
    return any(w in BAD_WORDS for w in words)

def tts_speak(text: str, rate: int = 175, volume: float = 1.0) -> None:
    """Speak the text out loud."""
    if not pyttsx3:
        print("⚠️ TTS not available. Install with: pip install pyttsx3")
        return
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    engine.say(text)
    engine.runAndWait()

def tts_save(text: str, out_path: str = "summary.wav", rate: int = 175, volume: float = 1.0) -> str:
    """Save the spoken text to a WAV file and return the path."""
    if not pyttsx3:
        print("⚠️ TTS not available. Install with: pip install pyttsx3")
        return ""
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    return out_path

# --- CLI loop ---
if __name__ == "__main__":
    while True:
        user_q = input("\nWhat book are you looking for / what are you interested in?\n> ").strip()
        if not user_q:
            break

        # --- Check for inappropriate language ---
        if contains_inappropriate_language(user_q):
            print("I’m here to help, but let’s keep the conversation respectful 🙂")
            continue

        # --- Otherwise, proceed with normal search ---
        results = search_books(user_q, k=1)
        if not results:
            print("No matches found. Please try different keywords.")
            continue

        print("\nTop matches:")
        for i, r in enumerate(results, start=1):
            print(f"\n{i}) {r['title']}  (score: {r['score']})")

            # Full summary from JSON
            full_summary = get_summary_by_title(r['title'])
            print(f"\nFull summary:\n{full_summary}")
            choice = input("\n🔊 Do you want to listen to this summary? (p=play, s=save, b=both, n=skip)\n> ").strip().lower()
            if choice == "p":
                tts_speak(full_summary)
            elif choice == "s":
                filename = f"summary_{r['title'].replace(' ', '_')}.wav"
                out = tts_save(full_summary, filename)
                if out:
                    print(f"✅ Audio saved to: {out}")
            elif choice == "b":
                filename = f"summary_{r['title'].replace(' ', '_')}.wav"
                out = tts_save(full_summary, filename)
                if out:
                    print(f"✅ Audio saved to: {out}")
                tts_speak(full_summary)
            else:
                print("Skipped audio.")

