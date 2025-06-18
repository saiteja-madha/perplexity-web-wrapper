import re
import sys
import json
import random
import mimetypes
from uuid import uuid4
from curl_cffi import requests, CurlMime


class Client:
    """
    A client for interacting with the Perplexity AI API.
    """

    def __init__(self, cookies={}):
        # Initialize an HTTP session with default headers and optional cookies
        self.session = requests.Session(
            headers={
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "accept-language": "en-US,en;q=0.9",
                "cache-control": "max-age=0",
                "dnt": "1",
                "priority": "u=0, i",
                "sec-ch-ua": '"Not;A=Brand";v="24", "Chromium";v="128"',
                "sec-ch-ua-arch": '"x86"',
                "sec-ch-ua-bitness": '"64"',
                "sec-ch-ua-full-version": '"128.0.6613.120"',
                "sec-ch-ua-full-version-list": '"Not;A=Brand";v="24.0.0.0", "Chromium";v="128.0.6613.120"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-model": '""',
                "sec-ch-ua-platform": '"Windows"',
                "sec-ch-ua-platform-version": '"19.0.0"',
                "sec-fetch-dest": "document",
                "sec-fetch-mode": "navigate",
                "sec-fetch-site": "same-origin",
                "sec-fetch-user": "?1",
                "upgrade-insecure-requests": "1",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            },
            cookies=cookies,
            impersonate="chrome",
        )
        self.own = bool(cookies)
        self.copilot = 0 if not cookies else float("inf")
        self.file_upload = 0 if not cookies else float("inf")
        self.signin_regex = re.compile(
            r'"(https://www\.perplexity\.ai/api/auth/callback/email\?callbackUrl=.*?)"'
        )
        self.timestamp = format(random.getrandbits(32), "08x")
        self.session.get("https://www.perplexity.ai/api/auth/session")

    def search(
        self,
        query,
        mode="auto",
        model=None,
        sources=["web"],
        files={},
        stream=False,
        language="en-US",
        follow_up=None,
        incognito=False,
    ):
        """
        Executes a search query on Perplexity AI.

        Parameters:
        - query: The search query string.
        - mode: Search mode ('auto', 'pro', 'reasoning', 'deep research').
        - model: Specific model to use for the query.
        - sources: List of sources ('web', 'scholar', 'social').
        - files: Dictionary of files to upload.
        - stream: Whether to stream the response.
        - language: Language code (ISO 639).
        - follow_up: Information for follow-up queries.
        - incognito: Whether to enable incognito mode.
        """
        # Validate input parameters
        assert mode in ["auto", "pro", "reasoning", "deep research"], (
            "Invalid search mode."
        )
        assert (
            model
            in {
                "auto": [None],
                "pro": [
                    None,
                    "sonar",
                    "gpt-4.5",
                    "gpt-4o",
                    "claude 3.7 sonnet",
                    "gemini 2.0 flash",
                    "grok-2",
                ],
                "reasoning": [None, "r1", "o3-mini", "claude 3.7 sonnet"],
                "deep research": [None],
            }[mode]
            if self.own
            else True
        ), "Invalid model for the selected mode."
        assert all([source in ("web", "scholar", "social") for source in sources]), (
            "Invalid sources."
        )
        assert (
            self.copilot > 0 if mode in ["pro", "reasoning", "deep research"] else True
        ), "No remaining pro queries."
        assert self.file_upload - len(files) >= 0 if files else True, (
            "File upload limit exceeded."
        )

        # Update query and file upload counters
        self.copilot = (
            self.copilot - 1
            if mode in ["pro", "reasoning", "deep research"]
            else self.copilot
        )
        self.file_upload = self.file_upload - len(files) if files else self.file_upload

        # Upload files and prepare the query payload
        uploaded_files = []
        for filename, file in files.items():
            file_type = mimetypes.guess_type(filename)[0]
            file_upload_info = (
                self.session.post(
                    "https://www.perplexity.ai/rest/uploads/create_upload_url?version=2.18&source=default",
                    json={
                        "content_type": file_type,
                        "file_size": sys.getsizeof(file),
                        "filename": filename,
                        "force_image": False,
                        "source": "default",
                    },
                )
            ).json()

            # Upload the file to the server
            mp = CurlMime()
            for key, value in file_upload_info["fields"].items():
                mp.addpart(name=key, data=value)
            mp.addpart(
                name="file", content_type=file_type, filename=filename, data=file
            )

            upload_resp = self.session.post(
                file_upload_info["s3_bucket_url"], multipart=mp
            )

            if not upload_resp.ok:
                raise Exception("File upload error", upload_resp)

            # Extract the uploaded file URL
            if "image/upload" in file_upload_info["s3_object_url"]:
                uploaded_url = re.sub(
                    r"/private/s--.*?--/v\d+/user_uploads/",
                    "/private/user_uploads/",
                    upload_resp.json()["secure_url"],
                )
            else:
                uploaded_url = file_upload_info["s3_object_url"]

            uploaded_files.append(uploaded_url)

        # Prepare the JSON payload for the query
        json_data = {
            "query_str": query,
            "params": {
                "attachments": uploaded_files + follow_up["attachments"]
                if follow_up
                else uploaded_files,
                "frontend_context_uuid": str(uuid4()),
                "frontend_uuid": str(uuid4()),
                "is_incognito": incognito,
                "language": language,
                "last_backend_uuid": follow_up["backend_uuid"] if follow_up else None,
                "mode": "concise" if mode == "auto" else "copilot",
                "model_preference": {
                    "auto": {None: "turbo"},
                    "pro": {
                        None: "pplx_pro",
                        "sonar": "experimental",
                        "gpt-4.5": "gpt45",
                        "gpt-4o": "gpt4o",
                        "claude 3.7 sonnet": "claude2",
                        "gemini 2.0 flash": "gemini2flash",
                        "grok-2": "grok",
                    },
                    "reasoning": {
                        None: "pplx_reasoning",
                        "r1": "r1",
                        "o3-mini": "o3mini",
                        "claude 3.7 sonnet": "claude37sonnetthinking",
                    },
                    "deep research": {None: "pplx_alpha"},
                }[mode][model],
                "source": "default",
                "sources": sources,
                "version": "2.18",
            },
        }

        # Send the query request and handle the response
        resp = self.session.post(
            "https://www.perplexity.ai/rest/sse/perplexity_ask",
            json=json_data,
            stream=True,
        )
        chunks = []

        def stream_response(resp):
            """
            Generator for streaming responses.
            """
            for chunk in resp.iter_lines(delimiter=b"\r\n\r\n"):
                content = chunk.decode("utf-8")

                if content.startswith("event: message\r\n"):
                    content_json = json.loads(
                        content[len("event: message\r\ndata: ") :]
                    )
                    if "text" in content_json:
                        content_json["text"] = json.loads(content_json["text"])

                    chunks.append(content_json)
                    yield chunks[-1]

                elif content.startswith("event: end_of_stream\r\n"):
                    return

        if stream:
            return stream_response(resp)

        for chunk in resp.iter_lines(delimiter=b"\r\n\r\n"):
            content = chunk.decode("utf-8")

            if content.startswith("event: message\r\n"):
                content_json = json.loads(content[len("event: message\r\ndata: ") :])
                if "text" in content_json:
                    content_json["text"] = json.loads(content_json["text"])

                chunks.append(content_json)

            elif content.startswith("event: end_of_stream\r\n"):
                return chunks[-1]

    def get_threads(self, limit=20, offset=0, search_term=""):
        """
        Fetches a list of threads from Perplexity AI.

        Parameters:
        - limit: Number of threads to fetch (default 20)
        - offset: Offset for pagination (default 0)
        - search_term: Search term to filter threads (default empty)
        """
        url = "https://www.perplexity.ai/rest/thread/list_ask_threads?version=2.18&source=default"
        payload = {"limit": limit, "offset": offset, "search_term": search_term}
        resp = self.session.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_thread_details_by_slug(self, slug, query_params=None):
        """
        Fetches thread details using the provided slug from the new endpoint.

        Parameters:
        - slug: The thread slug (string)
        - query_params: Optional dict of query parameters to override defaults
        """
        from urllib.parse import urlencode

        default_params = {
            "with_parent_info": "true",
            "with_schematized_response": "true",
            "version": "2.18",
            "source": "default",
            "limit": 100,
            "offset": 0,
            "from_first": "true",
            "supported_block_use_cases": [
                "answer_modes",
                "media_items",
                "knowledge_cards",
                "inline_entity_cards",
                "place_widgets",
                "finance_widgets",
                "sports_widgets",
                "shopping_widgets",
                "jobs_widgets",
                "search_result_widgets",
                "clarification_responses",
                "inline_images",
                "inline_assets",
                "inline_finance_widgets",
                "placeholder_cards",
                "diff_blocks",
                "inline_knowledge_cards",
            ],
        }
        # Merge user params
        params = dict(default_params)
        if query_params:
            for k, v in query_params.items():
                params[k] = v
        # Handle list params for supported_block_use_cases
        query_items = []
        for k, v in params.items():
            if isinstance(v, list):
                for item in v:
                    query_items.append((k, item))
            else:
                query_items.append((k, v))
        query_string = urlencode(query_items)
        url = f"https://www.perplexity.ai/rest/thread/{slug}?{query_string}"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()
