import chromadb
import json
from openai import OpenAI
import os
from typing import List, Dict


client = chromadb.Client()
collection = client.create_collection(name="my_collection")
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


with open('books_prompt_result.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)
    
ids, documents, metadatas = [], [], []

for data in json_data["books"]:
    book_id = str(data.get('id', ''))
    title = data.get('title', '')
    summary = data.get('summary', '')

    doc = f"Title: {title} \n Summary: {summary}".strip()
    documents.append(doc)
    ids.append(book_id)
    metadatas.append({"id": book_id, "title": title})

def embed_batch(texts, model="text-embedding-3-small", batch_size=100):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = openai_client.embeddings.create(
            model=model,
            input=batch
        )
        vectors.extend([d.embedding for d in resp.data])
    return vectors

embeddings = embed_batch(documents, model="text-embedding-3-small", batch_size=100)
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print("Sample entries:")
print(collection.peek())

def embed_text(text: str, model: str = "text-embedding-3-small") -> List[float]:
    return openai_client.embeddings.create(model=model, input=text).data[0].embedding

def search_books(query: str, k: int = 1, model: str = "text-embedding-3-small") -> List[Dict]:
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

# === DEMO ===
if __name__ == "__main__":
    while True:
        user_q = input("\nWhat book are you looking for / what are you interested in? (e.g., friendship and magic, war, dystopia)\n> ").strip()
        if not user_q:
            break
        results = search_books(user_q, k=1)
        if not results:
            print("No matches found. Please try different keywords.")
            continue

        print("\nBest Match:")
        for i, r in enumerate(results, start=1):
            print(f"\n{i}) {r['title']}  (scor: {r['score']})")
