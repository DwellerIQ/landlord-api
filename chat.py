
import os
from openai import OpenAI
from supabase import create_client
import anthropic
import json

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def get_relevant_chunks(question, num_chunks=5):
    # Convert question to embedding
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    question_embedding = response.data[0].embedding

    # Search Supabase for relevant chunks
    result = supabase.rpc("match_documents", {
        "query_embedding": question_embedding,
        "match_threshold": 0.5,
        "match_count": num_chunks
    }).execute()

    return result.data

def handler(request):
    # Handle CORS
    if request.method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": ""
        }

    try:
        body = request.json()
        question = body.get("question", "")

        if not question:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No question provided"})
            }

        # Get relevant legal chunks
        chunks = get_relevant_chunks(question)
        context = "\n\n".join([chunk["content"] for chunk in chunks])

        # Ask Claude
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
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "answer": answer,
                "sources": [chunk["doc_type"] for chunk in chunks]
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
