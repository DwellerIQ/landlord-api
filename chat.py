from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from supabase import create_client
import anthropic
import stripe
import os
import json
import uuid
from datetime import datetime

openai_client = OpenAI(api_key=os.environ.get(""))
supabase = create_client(os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_KEY", ""))
claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")

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
stripe.api_key = os.environ["STRIPE_SECRET_KEY"]

NOTICE_KEYWORDS = [
    "five-day notice", "5-day notice", "10-day notice", "ten-day notice",
    "written notice", "notice to terminate", "notice of termination",
    "eviction notice", "notice to quit", "entry notice", "notice of entry",
    "notice to cure", "notice to pay"
]

NOTICE_TYPES = {
    "non_payment": "5-Day Notice to Pay Rent",
    "lease_violation": "10-Day Notice to Cure Lease Violation",
    "termination": "Notice of Termination of Tenancy",
    "entry": "Notice of Landlord Entry",
    "foreclosure": "Notice of Foreclosure Action"
}

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

def detect_notice_needed(answer):
    answer_lower = answer.lower()
    for keyword in NOTICE_KEYWORDS:
        if keyword in answer_lower:
            return True
    return False

def detect_notice_type(answer):
    answer_lower = answer.lower()
    if "5-day" in answer_lower or "five-day" in answer_lower or "non-payment" in answer_lower or "unpaid rent" in answer_lower:
        return "non_payment"
    elif "10-day" in answer_lower or "ten-day" in answer_lower or "lease violation" in answer_lower:
        return "lease_violation"
    elif "entry" in answer_lower or "enter" in answer_lower or "access" in answer_lower:
        return "entry"
    elif "foreclosure" in answer_lower:
        return "foreclosure"
    else:
        return "termination"

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

    chunks = get_relevant_chunks(question)
    context = "\n\n".join([chunk["content"] for chunk in chunks])

    system_prompt = f"""You are a knowledgeable assistant helping Chicago and Illinois landlords understand their legal rights and responsibilities.

Use ONLY the following legal text to answer questions. If the answer is not in the provided text, say so clearly and recommend consulting an attorney.

Keep your answers conversational, practical and concise. Avoid heavy formatting with lots of headers. Use plain paragraphs with occasional bullet points only when listing multiple distinct items. Always reference specific ordinance sections when relevant.

LEGAL CONTEXT:
{context}"""

    capped_history = history[-6:] if len(history) > 6 else history

    messages = []
    for msg in capped_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    messages.append({
        "role": "user",
        "content": question
    })

    message = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=messages
    )

    answer = message.content[0].text
    notice_needed = detect_notice_needed(answer)
    notice_type = detect_notice_type(answer) if notice_needed else None

    return {
        "answer": answer,
        "sources": [chunk["doc_type"] for chunk in chunks],
        "notice_needed": notice_needed,
        "notice_type": notice_type,
        "notice_type_label": NOTICE_TYPES.get(notice_type, "") if notice_type else ""
    }

@app.options("/api/create-payment")
async def options_payment():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )

@app.post("/api/create-payment")
async def create_payment(request: Request):
    body = await request.json()
    notice_type = body.get("notice_type", "")
    conversation_history = body.get("history", [])
    notice_details = body.get("notice_details", {})

    # Create a unique session ID for this notice request
    session_id = str(uuid.uuid4())

    # Store pending notice request in Supabase
    supabase.table("notice_requests").insert({
        "session_id": session_id,
        "notice_type": notice_type,
        "conversation_history": json.dumps(conversation_history),
        "notice_details": json.dumps(notice_details),
        "status": "pending",
        "created_at": datetime.utcnow().isoformat()
    }).execute()

    payment_link = os.environ["STRIPE_PAYMENT_LINK"]

    return {
        "payment_url": f"{payment_link}?client_reference_id={session_id}",
        "session_id": session_id
    }

@app.post("/api/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.environ["STRIPE_WEBHOOK_SECRET"]
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        session_id = session.get("client_reference_id")

        if session_id:
            # Get the pending notice request
            result = supabase.table("notice_requests").select("*").eq("session_id", session_id).execute()

            if result.data:
                notice_request = result.data[0]
                notice_type = notice_request["notice_type"]
                history = json.loads(notice_request["conversation_history"])
                details = json.loads(notice_request["notice_details"])

                # Generate the notice
                notice_content = await generate_notice_content(notice_type, history, details)

                # Save completed notice
                supabase.table("notice_requests").update({
                    "status": "completed",
                    "notice_content": notice_content,
                    "completed_at": datetime.utcnow().isoformat()
                }).eq("session_id", session_id).execute()

    return {"status": "ok"}

async def generate_notice_content(notice_type, history, details):
    chunks = get_relevant_chunks(f"landlord notice {notice_type} Chicago Illinois requirements")
    context = "\n\n".join([chunk["content"] for chunk in chunks])

    conversation_summary = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in history[-6:]
    ])

    prompt = f"""Based on the following Chicago/Illinois landlord-tenant law, generate a professional, legally compliant {NOTICE_TYPES.get(notice_type, 'landlord notice')}.

LEGAL CONTEXT:
{context}

CONVERSATION CONTEXT:
{conversation_summary}

NOTICE DETAILS PROVIDED:
{json.dumps(details, indent=2)}

Generate a complete, professional notice document. Include:
- Proper heading and title
- Date line
- Landlord and tenant information fields (use placeholders like [LANDLORD NAME] if not provided)
- The full legal notice text with all required information per Chicago RLTO
- Signature line
- Important legal disclaimer at the bottom

Format it cleanly as a formal document."""

    message = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text

@app.options("/api/get-notice")
async def options_get_notice():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )

@app.post("/api/get-notice")
async def get_notice(request: Request):
    body = await request.json()
    session_id = body.get("session_id", "")

    if not session_id:
        return {"error": "No session ID provided"}

    result = supabase.table("notice_requests").select("*").eq("session_id", session_id).execute()

    if not result.data:
        return {"status": "not_found"}

    notice = result.data[0]

    if notice["status"] == "pending":
        return {"status": "pending"}

    return {
        "status": "completed",
        "notice_content": notice["notice_content"],
        "notice_type": notice["notice_type"],
        "notice_type_label": NOTICE_TYPES.get(notice["notice_type"], ""),
        "created_at": notice["created_at"]
    }

@app.get("/api/documents/{user_id}")
async def get_documents(user_id: str):
    result = supabase.table("notice_requests").select(
        "session_id, notice_type, notice_type_label, created_at, status"
    ).eq("user_id", user_id).eq("status", "completed").order("created_at", desc=True).execute()

    return {"documents": result.data}

@app.get("/")
async def root():
    return {"status": "Landlord API is running"}
