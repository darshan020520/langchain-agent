
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from agent import agent_executor

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {"message": "LangChain Agent Server is running."}

@app.post("/invoke")
async def invoke(request: Request):
    query = request.query_params.get("content")
    session_id = request.query_params.get("session_id", "default")

    if not query:
        return {"error": "Missing 'content' query param."}

    print(f"[INFO] Query: {query} | session: {session_id}")

    async def generate_response():
        try:
            streamer = await agent_executor.stream_invoke(query, session_id)
            async for token in streamer:
                print(f"[STREAM] {token}")  # <-- ADD THIS
                yield token
        except Exception as e:
            print(f"[ERROR] Agent execution failed: {e}")
            yield f"\n\n[Error: {str(e)}]"

    return StreamingResponse(generate_response(), media_type="text/plain")

