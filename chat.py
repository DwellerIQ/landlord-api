from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from supabase import create_client
import anthropic
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
from fastapi.responses import Response

@app.options("/api/chat")
async def options_chat():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )
@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    question = body.get("question", "")
    history = body.get("history", [])

    if not question:
        return {"error": "No question provided"}

    # Get relevant legal chunks based on the current question
    chunks = get_relevant_chunks(question)
    context = "\n\n".join([chunk["content"] for chunk in chunks])

    # Build system prompt
    system_prompt = f"""You are a knowledgeable assistant helping Chicago and Illinois landlords understand their legal rights and responsibilities.

Use ONLY the following legal text to answer questions. If the answer is not in the provided text, say so clearly and recommend consulting an attorney.

Keep your answers conversational, practical and concise. Avoid heavy formatting with lots of headers. Use plain paragraphs with occasional bullet points only when listing multiple distinct items. Always reference specific ordinance sections when relevant.

LEGAL CONTEXT:
{context}"""

    # Build messages array with full conversation history
    messages = []

    # Add previous conversation history
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Add current question
    messages.append({
        "role": "user",
        "content": question
    })

    # Ask Claude with full history
    message = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=messages
    )

    answer = message.content[0].text

    return {
        "answer": answer,
        "sources": [chunk["doc_type"] for chunk in chunks]
    }

@app.get("/")
async def root():
    return {"status": "Landlord API is running"}
