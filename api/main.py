from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse


import asyncio
import json
from lib import perplexity
# import perplexity
from typing import List, Optional
from datetime import datetime

# Initialize Perplexity client
try:
    with open("perplexity_cookies.json", "r", encoding="utf-8") as f:
        perplexity_cookies = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    print("Cookies file not found or invalid. Using empty cookies.")
    perplexity_cookies = {}

perplexity_cli = perplexity.Client(perplexity_cookies)
app = FastAPI(
    title="Perplexity AI API", description="Stream Perplexity AI responses using SSE"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_answer(res):
    """Extract answer from Perplexity API response."""
    text = res.get("text", [])
    backend_uuid = res.get("backend_uuid", None)
    if isinstance(text, dict):
        answer = text.get("answer")
        if answer:
            return {"answer": answer, "backend_uuid": backend_uuid}
    elif isinstance(text, list):
        for item in text:
            if isinstance(item, dict) and item.get("step_type") == "FINAL":
                answer_str = item.get("content", {}).get("answer")
                if answer_str:
                    answer = json.loads(answer_str).get("answer")
                    return {"answer": answer, "backend_uuid": backend_uuid}

    return {"answer": None, "backend_uuid": backend_uuid}


def save_resp(res, question_id):
    """Save response to file for logging/debugging."""
    dir_name = "responses"
    file_name = f"{question_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(f"{dir_name}/{file_name}", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    except Exception:
        # Silently fail if we can't save
        pass


async def generate_sse_stream(
    query: str,
    mode: str,
    model: Optional[str],
    sources: List[str],
    language: str,
    follow_up: Optional[dict],
    incognito: bool,
):
    """Generate SSE stream from Perplexity responses."""
    previous_content = ""
    response_count = 0

    # Start the streaming search in a sync function
    try:
        for stream in perplexity_cli.search(
            query,
            mode=mode,
            model=model,
            sources=sources,
            files={},
            stream=True,
            language=language,
            follow_up=follow_up,
            incognito=incognito,
        ):
            response_count += 1
            save_resp(
                stream,
                f"API-{datetime.now().strftime('%Y%m%d%H%M%S')}-{response_count}",
            )

            ans_data = extract_answer(stream)
            ans_string = ans_data.get("answer", "")
            backend_uuid = ans_data.get("backend_uuid")
            if ans_string:
                # Only send new content that hasn't been sent yet
                if len(ans_string) > len(previous_content):
                    new_content = ans_string[len(previous_content) :]
                    # Format for SSE
                    event_data = json.dumps(
                        {
                            "type": "content",
                            "content": new_content,
                            "backend_uuid": backend_uuid,
                            "done": False,
                        }
                    )
                    yield f"data: {event_data}\n\n"
                    await asyncio.sleep(
                        0.01
                    )  # Small delay to prevent overwhelming the client
                    previous_content = ans_string

        # Send completion event
        event_data = json.dumps({"type": "content", "content": "", "done": True})
        yield f"data: {event_data}\n\n"

    except Exception as e:
        error_data = json.dumps({"type": "error", "error": str(e)})
        yield f"data: {error_data}\n\n"


@app.get("/api/query")
async def query(
    q: str = Query(..., description="Query to send to Perplexity"),
    mode: str = Query(
        "auto",
        description="Search mode",
        enum=["auto", "writing", "coding", "research"],
    ),
    model: Optional[str] = Query(None, description="Model to use"),
    sources: str = Query("web", description="Sources (comma-separated)"),
    language: str = Query("en-US", description="Language"),
    incognito: bool = Query(False, description="Use incognito mode"),
):
    """Stream Perplexity AI responses as Server-Sent Events (SSE)."""

    sources_list = [s.strip() for s in sources.split(",")]

    return StreamingResponse(
        generate_sse_stream(
            query=q,
            mode=mode,
            model=model,
            sources=sources_list,
            language=language,
            follow_up=None,  # No follow-up for initial query
            incognito=incognito,
        ),
        media_type="text/event-stream",
    )


@app.get("/api/followup")
async def followup(
    q: str = Query(..., description="Follow-up query"),
    backend_uuid: str = Query(..., description="UUID of the previous response"),
    mode: str = Query(
        "auto",
        description="Search mode",
        enum=["auto", "writing", "coding", "research"],
    ),
    model: Optional[str] = Query(None, description="Model to use"),
    sources: str = Query("web", description="Sources (comma-separated)"),
    language: str = Query("en-US", description="Language"),
    incognito: bool = Query(False, description="Use incognito mode"),
):
    """Stream follow-up responses as Server-Sent Events (SSE)."""

    # Create follow-up structure expected by perplexity
    follow_up_data = {"backend_uuid": backend_uuid, "attachments": []}

    sources_list = [s.strip() for s in sources.split(",")]

    return StreamingResponse(
        generate_sse_stream(
            query=q,
            mode=mode,
            model=model,
            sources=sources_list,
            language=language,
            follow_up=follow_up_data,
            incognito=incognito,
        ),
        media_type="text/event-stream",
    )


@app.get("/api/threads")
def get_threads(limit: int = 20, offset: int = 0, search_term: str = ""):
    """Fetch a list of threads from Perplexity AI."""
    try:
        threads = perplexity_cli.get_threads(
            limit=limit, offset=offset, search_term=search_term
        )
        return JSONResponse(content=threads)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
