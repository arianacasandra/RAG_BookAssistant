import os, json, tempfile, requests
from typing import List
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import chromadb

# --- Config ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static",
)
CORS(app)

# --- Load data ---
with open("books_prompt_result.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

title_to_summary = {b["title"]: b["summary"] for b in json_data["books"]}

with open("bad_words.json", "r", encoding="utf-8") as f:
    BAD_WORDS = set(json.load(f)["bad_words"])

def contains_inappropriate_language(text: str) -> bool:
    words = text.lower().split()
    return any(w in BAD_WORDS for w in words)

# --- Chroma setup ---
client = chromadb.Client()
collection = client.create_collection(name="my_collection")

ids, docs, metas = [], [], []
for b in json_data["books"]:
    ids.append(str(b["id"]))
    t, s = b["title"], b["summary"]
    docs.append(f"Title: {t}\nSummary: {s}")
    metas.append({"title": t})

def _ollama_embed(text: str, model: str = EMBED_MODEL) -> List[float]:
    url = f"{OLLAMA_HOST}/api/embeddings"
    r = requests.post(url, json={"model": model, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]

# index documents once at startup
embeddings = [_ollama_embed(d) for d in docs]
collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

# --- TTS with pyttsx3 ---
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

def text_to_wav_file(text: str, rate: int = 175, volume: float = 1.0) -> str:
    if not pyttsx3:
        raise RuntimeError("TTS not available. Install pyttsx3.")
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    path = tmp.name
    tmp.close()
    engine.save_to_file(text, path)
    engine.runAndWait()
    return path

# --- Routes ---
@app.route("/")
def index():
    return render_template("chat.html")

@app.post("/chat")
def chat():
    data = request.get_json(force=True)
    msg = (data.get("message") or "").strip()

    if not msg:
        return jsonify({"reply": "Please type something."})

    if contains_inappropriate_language(msg):
        return jsonify({"reply": "I’m here to help, but let’s keep the conversation respectful 🙂"})

    # Semantic search for best match
    q_emb = _ollama_embed(msg)
    res = collection.query(query_embeddings=[q_emb], n_results=1)
    title = res["metadatas"][0][0]["title"]
    summary = title_to_summary.get(title, "No summary found.")
    reply = f"Best match: {title}\n\n{summary}"
    return jsonify({"reply": reply})

@app.post("/search")
def search():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    k = int(data.get("k", 3))

    if not query:
        return jsonify({"error": "query is required"}), 400

    if contains_inappropriate_language(query):
        return jsonify({"blocked": True,
                        "message": "I’m here to help, but let’s keep the conversation respectful 🙂"})

    q_emb = _ollama_embed(query)
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

    return jsonify({"results": hits})

@app.get("/summary")
def summary():
    title = (request.args.get("title") or "").strip()
    if not title:
        return jsonify({"error": "title query param is required"}), 400

    # exact match
    if title in title_to_summary:
        return jsonify({"title": title, "summary": title_to_summary[title]})

    # case-insensitive fallback
    tl = title.lower()
    for k, v in title_to_summary.items():
        if k.lower() == tl:
            return jsonify({"title": k, "summary": v})

    return jsonify({"error": f"The title «{title}» was not found."}), 404

@app.post("/tts")
def tts():
    if not pyttsx3:
        return jsonify({"error": "TTS not available. Install pyttsx3."}), 500

    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    rate = int(data.get("rate", 175))
    volume = float(data.get("volume", 1.0))

    if not text:
        return jsonify({"error": "text is required"}), 400

    try:
        wav_path = text_to_wav_file(text, rate=rate, volume=volume)
        return send_file(wav_path, mimetype="audio/wav",
                         as_attachment=False, download_name="tts.wav")
    except Exception as e:
        return jsonify({"error": f"TTS failed: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
