from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from supabase import create_client
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import anthropic
import httpx
import os
import json
import uuid
import time
from collections import defaultdict
from datetime import datetime

# ‚îÄ‚îÄ Startup: initialize shared clients once ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ
_openai = None
_supabase = None
_claude = None

def get_clients():
    global _openai, _supabase, _claude
    if _openai is None:
        _openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if _supabase is None:
        _supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    if _claude is None:
        _claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _openai, _supabase, _claude

# ‚îÄ‚îÄ Rate limiting ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ
limiter = Limiter(key_func=get_remote_address)

_member_request_times: dict = defaultdict(list)
MEMBER_HOURLY_LIMIT = int(os.environ.get("MEMBER_HOURLY_LIMIT", "10"))
MEMBER_DAILY_LIMIT = int(os.environ.get("MEMBER_DAILY_LIMIT", "25"))

def check_member_rate_limit(member_id: str) -> str | None:
    """Returns None if allowed, otherwise a user-facing limit message."""
    now = time.time()
    times = _member_request_times[member_id]
    times[:] = [t for t in times if t > now - 86400]
    if len(times) >= MEMBER_DAILY_LIMIT:
        return f"Daily limit of {MEMBER_DAILY_LIMIT} questions reached. Your limit resets in 24 hours."
    if sum(1 for t in times if t > now - 3600) >= MEMBER_HOURLY_LIMIT:
        return f"Limit of {MEMBER_HOURLY_LIMIT} questions per hour reached. Please try again later."
    times.append(now)
    return None

# ‚îÄ‚îÄ App setup ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = [
    "https://dwelleriq.com",
    "https://www.dwelleriq.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-DIQ-Key"],
)

# ‚îÄ‚îÄ Auth ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ
_DIQ_API_KEY = os.environ.get("DIQ_API_KEY", "")

# Token verification cache: {token: (member_id, expiry)}
_token_cache: dict = {}
TOKEN_CACHE_TTL = 300  # 5 minutes

async def verify_memberstack_token(token: str) -> str | None:
    """Call Memberstack's API to verify a JWT. Returns member_id or None."""
    if token in _token_cache:
        member_id, expiry = _token_cache[token]
        if time.time() < expiry:
            return member_id
        del _token_cache[token]

    ms_key = os.environ.get("MEMBERSTACK_SECRET_KEY", "")
    if not ms_key:
        return None

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                "https://admin.memberstack.com/members/verify-token",
                json={"token": token},
                headers={"x-api-key": ms_key, "Content-Type": "application/json"},
            )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            member_id = data.get("id")
            if member_id:
                _token_cache[token] = (member_id, time.time() + TOKEN_CACHE_TTL)
                return member_id
    except Exception:
        pass
    return None

async def require_auth(request: Request) -> str:
    """
    Validates the request and returns the verified member_id.
    - Always checks X-DIQ-Key if DIQ_API_KEY is configured.
    - If MEMBERSTACK_SECRET_KEY is configured, also requires a valid
      Memberstack JWT in the Authorization: Bearer header.
    """
    if _DIQ_API_KEY:
        if request.headers.get("X-DIQ-Key", "") != _DIQ_API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

    ms_key = os.environ.get("MEMBERSTACK_SECRET_KEY", "")
    if ms_key:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authentication required")
        token = auth_header[7:]
        member_id = await verify_memberstack_token(token)
        if not member_id:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return member_id

    # MS key not configured yet ‚Äî fall back to member_id from body
    return ""

# ‚îÄ‚îÄ Domain data ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ
NOTICE_KEYWORDS = [
    "five-day notice", "5-day notice", "10-day notice", "ten-day notice",
    "written notice", "notice to terminate", "notice of termination",
    "eviction notice", "notice to quit", "entry notice", "notice of entry",
    "notice to cure", "notice to pay",
    "provide notice", "send notice", "serve notice", "issue a notice",
    "send a written", "provide written", "written notification",
    "notify the tenant", "notify your tenant", "must notify",
    "notice of abandoned", "notice regarding", "notice of entry",
    "notice of non-renewal", "non-renewal notice", "notice to vacate",
    "notice of rent increase", "rent increase notice"
]

NOTICE_TYPES = {
    "non_payment": "5-Day Notice to Pay Rent",
    "lease_violation": "10-Day Notice to Cure Lease Violation",
    "termination": "Notice of Termination of Tenancy",
    "entry": "Notice of Landlord Entry",
    "foreclosure": "Notice of Foreclosure Action",
    "rent_increase": "Notice of Rent Increase",
    "non_renewal": "Notice of Non-Renewal of Lease"
}

# ‚îÄ‚îÄ Helpers ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ
def get_relevant_chunks(question, num_chunks=8):
    openai_client, supabase, _ = get_clients()
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    question_embedding = response.data[0].embedding
    result = supabase.rpc("match_documents", {
        "query_embedding": question_embedding,
        "match_threshold": 0.35,
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
    elif "rent increase" in answer_lower or "raise the rent" in answer_lower or "increase rent" in answer_lower:
        return "rent_increase"
    elif "non-renewal" in answer_lower or "not renew" in answer_lower or "renewal notice" in answer_lower:
        return "non_renewal"
    elif "entry" in answer_lower or "enter" in answer_lower or "access" in answer_lower:
        return "entry"
    elif "foreclosure" in answer_lower:
        return "foreclosure"
    else:
        return "termination"

# ‚îÄ‚îÄ Routes ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ‚îÄ
@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/api/chat")
@limiter.limit("30/hour")
async def chat(request: Request, verified_member_id: str = Depends(require_auth)):
    openai_client, supabase, claude_client = get_clients()
    body = await request.json()
    question = body.get("question", "")
    history = body.get("history", [])
    # Use server-verified member_id; fall back to client-supplied only if MS key not set
    member_id = verified_member_id or body.get("member_id", "anonymous")

    if not question:
        return JSONResponse(status_code=400, content={"error": "No question provided"})

    if len(question) > 2000:
        return JSONResponse(status_code=400, content={"error": "Message too long (max 2000 characters)"})

    limit_message = check_member_rate_limit(member_id)
    if limit_message:
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limited", "message": limit_message}
        )

    chunks = get_relevant_chunks(question, num_chunks=5)
    context = "\n\n".join([chunk["content"] for chunk in chunks])

    system_prompt = f"""You are a Chicago and Illinois landlord-tenant law expert. Answer questions directly and completely.

Use the legal context below as your primary source and cite specific ordinance sections when relevant. If the context covers the question, use it. If the context is sparse or silent on a topic, answer from your knowledge of Chicago and Illinois landlord-tenant law ‚Äî do not refuse to answer just because the retrieved context is incomplete.

Match answer length to the question. No filler, no padding. Only recommend consulting an attorney for litigation strategy or genuinely unique fact patterns outside standard Chicago/Cook County/Illinois landlord-tenant law.

LEGAL CONTEXT:
{context}"""

    capped_history = history[-6:] if len(history) > 6 else history
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in capped_history]
    messages.append({"role": "user", "content": question})

    try:
        message = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            system=system_prompt,
            messages=messages
        )
        answer = message.content[0].text
    except Exception as e:
        error_msg = str(e).lower()
        if "rate" in error_msg or "limit" in error_msg or "overload" in error_msg:
            return JSONResponse(
                status_code=429,
                content={"error": "high_demand", "message": "We're experiencing high demand right now. Please try again in a moment."}
            )
        return JSONResponse(
            status_code=500,
            content={"error": "server_error", "message": "Something went wrong. Please try again."}
        )

    notice_needed = detect_notice_needed(answer)
    notice_type = detect_notice_type(answer) if notice_needed else None

    return {
        "answer": answer,
        "sources": [chunk["doc_type"] for chunk in chunks],
        "notice_needed": notice_needed,
        "notice_type": notice_type,
        "notice_type_label": NOTICE_TYPES.get(notice_type, "") if notice_type else ""
    }

@app.post("/api/create-payment")
@limiter.limit("5/day")
async def create_payment(request: Request, _: str = Depends(require_auth)):
    _, supabase, _ = get_clients()
    body = await request.json()
    notice_type = body.get("notice_type", "")
    conversation_history = body.get("history", [])
    notice_details = body.get("notice_details", {})

    session_id = str(uuid.uuid4())

    supabase.table("notice_requests").insert({
        "session_id": session_id,
        "notice_type": notice_type,
        "conversation_history": json.dumps(conversation_history),
        "notice_details": json.dumps(notice_details),
        "status": "pending",
        "created_at": datetime.utcnow().isoformat()
    }).execute()

    payment_link = os.environ.get("STRIPE_PAYMENT_LINK", "")

    return {
        "payment_url": f"{payment_link}?client_reference_id={session_id}",
        "session_id": session_id
    }

@app.post("/api/stripe-webhook")
async def stripe_webhook(request: Request):
    import stripe
    stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
    _, supabase, _ = get_clients()

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.environ.get("STRIPE_WEBHOOK_SECRET", "")
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        session_id = session.get("client_reference_id")

        if session_id:
            result = supabase.table("notice_requests").select("*").eq("session_id", session_id).execute()

            if result.data:
                notice_request = result.data[0]
                notice_type = notice_request["notice_type"]
                history = json.loads(notice_request["conversation_history"])
                details = json.loads(notice_request["notice_details"])
                notice_content = await generate_notice_content(notice_type, history, details)

                supabase.table("notice_requests").update({
                    "status": "completed",
                    "notice_content": notice_content,
                    "completed_at": datetime.utcnow().isoformat()
                }).eq("session_id", session_id).execute()

    return {"status": "ok"}

async def generate_notice_content(notice_type, history, details):
    _, _, claude_client = get_clients()
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
- Landlord and tenant information fields
- The full legal notice text with all required information per Chicago RLTO
- Signature line
- Legal disclaimer at the bottom

Format it cleanly as a formal document."""

    message = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text

@app.post("/api/get-notice")
async def get_notice(request: Request, _: str = Depends(require_auth)):
    _, supabase, _ = get_clients()
    body = await request.json()
    session_id = body.get("session_id", "")

    if not session_id:
        return JSONResponse(status_code=400, content={"error": "No session ID provided"})

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
async def get_documents(user_id: str, request: Request, _: str = Depends(require_auth)):
    _, supabase, _ = get_clients()
    result = supabase.table("notice_requests").select(
        "session_id, notice_type, created_at, status"
    ).eq("user_id", user_id).eq("status", "completed").order("created_at", desc=True).execute()

    return {"documents": result.data}
