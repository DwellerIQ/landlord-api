from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from supabase import create_client
import anthropic
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def get_relevant_chunks(question, num_chunks=5):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    question_embedding = response.data[0].embedding

    result = supabase.rpc("match_documents", {
        "query_embedding": question_embedding,
        "match_threshold": 0.5,
        "match_count": num_chunks
    }).execute()

    return result.data

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    question = body.get("question", "")

    if not question:
        return {"error": "No question provided"}

    chunks = get_relevant_chunks(question)
    context = "\n\n".join([chunk["content"] for chunk in chunks])

    message = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""You are a specialized assistant helping Chicago and Illinois landlords understand their legal rights and responsibilities.

Use ONLY the following legal text to answer the question. If the answer is not in the provided text, say so clearly and recommend consulting an attorney.

LEGAL CONTEXT:
{context}

LANDLORD QUESTION:
{question}

Provide a clear, practical answer. Reference specific ordinance sections where relevant."""
            }
        ]
    )

    answer = message.content[0].text

    return {
        "answer": answer,
        "sources": [chunk["doc_type"] for chunk in chunks]
    }

@app.get("/")
async def root():
    return {"status": "Landlord API is running"}
