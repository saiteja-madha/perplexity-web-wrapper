import json
from datetime import datetime


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
