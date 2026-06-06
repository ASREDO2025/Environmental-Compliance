import os
import anthropic
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Prompt Generator Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

SYSTEM_PROMPT = """You are an expert prompt engineer. When given raw text input, you will:
1. Analyze the text to understand its intent and context
2. Identify improvements and additions needed
3. Refine and rewrite the prompt for maximum clarity and effectiveness

Output ONLY two optimized prompts with no commentary, analysis, or explanation.
Format your response exactly as:

CLAUDE OPTIMIZED PROMPT:
[optimized prompt for Claude]

GPT-5X OPTIMIZED PROMPT:
[optimized prompt for GPT-5X]

Claude prompts should leverage: constitutional AI alignment, nuanced instruction-following, multi-turn context, XML tags for structure, and explicit reasoning requests.
GPT-5X prompts should leverage: system/user role separation, chain-of-thought triggers, precise formatting directives, and explicit output structure requirements."""


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/generate")
async def generate(raw_text: str = Form(...)):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return StreamingResponse(
            iter(["data: ERROR: ANTHROPIC_API_KEY is not set.\n\n"]),
            media_type="text/event-stream",
        )

    def stream_response():
        client = anthropic.Anthropic(api_key=api_key)
        with client.messages.stream(
            model="claude-opus-4-8",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": raw_text}],
        ) as stream:
            for text in stream.text_stream:
                escaped = text.replace("\n", "\\n")
                yield f"data: {escaped}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")
