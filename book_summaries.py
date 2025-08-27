import os
import json
import re
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

prompt_text = """
You are to respond ONLY with valid JSON. 
Do not include any text or code fences outside of the JSON.

Return an object with a top-level key "books" that contains a list of 10 objects.
Each object must have:
- "id": a string from "1" to "10"
- "title": the exact title of a book that actually exists in real life (fiction or non-fiction, widely published)
- "summary": approximately 300 words that follow ALL the rules below.

Rules for "summary":
1. The book must be a real, published work — do NOT invent titles or summaries.
2. Clearly present the setting, main character(s), and the central conflict without revealing major spoilers.
3. Naturally hint at the key themes and tone (e.g., hope, tragedy, mystery, resilience).
4. Conclude with a sentence that invites curiosity or wonder about the story’s outcome.
5. Do NOT include the book title anywhere in the summary.
6. Write in an engaging, vivid style that feels like a compelling back-cover blurb.
7. Avoid generic phrasing; make each summary distinct and memorable.

Example format (do not reuse the example content):
{
  "books": [
    {
      "id": "1",
      "title": "Example of a Real Published Book",
      "summary": "Engaging ~300 word summary here..."
    }
  ]
}
"""

response = client.responses.create(
    model="gpt-4o",
    input=prompt_text
)

raw = response.output_text.strip()

if raw.startswith("```"):
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())

try:
    payload = json.loads(raw)
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON from model: {e}\nRaw output:\n{raw}")

if "books" not in payload or len(payload["books"]) != 10:
    raise ValueError("The JSON does not contain exactly 10 books.")

final_data = {
    "books": payload["books"]
}

with open("books_prompt_result.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)

print("Structured book list saved to books_prompt_result.json")
