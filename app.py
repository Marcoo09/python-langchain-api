from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx, json

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    model = "llama3.1"  # force llama3.1 for now
    messages = body.get("messages", [])
    prompt = body.get("prompt") or "You are a helpful assistant."
    temperature = body.get("temperature", 0.7)

    # flatten messages into a single prompt string
    user_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    full_prompt = f"{prompt}\n{user_prompt}"

    payload = {
        "model": model,
        "prompt": full_prompt,
        "options": {"temperature": temperature}
    }

    async def generate():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "http://localhost:11434/api/generate",
                json=payload,
            ) as r:
                if r.status_code != 200:
                    yield f"[Error: {r.status_code} {await r.aread()}]"
                    return

                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except Exception:
                        yield line

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/models")
async def models():
    return JSONResponse([
        {"id": "llama3.1", "name": "LLaMA 3.1"},
        {"id": "mistral", "name": "Mistral 7B"},
    ])
