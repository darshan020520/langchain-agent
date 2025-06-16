import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from agent import agent_executor

app = FastAPI()

# Allow frontend access with specific origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {"message": "LangChain Agent Server is running."}

@app.post("/invoke")
async def invoke(request: Request):
    try:
        query = request.query_params.get("content")
        session_id = request.query_params.get("session_id", "default")

        if not query:
            return {"error": "Missing 'content' query param."}

        print(f"[INFO] Query: {query} | session: {session_id}")

        async def generate_response():
            try:
                streamer = await agent_executor.stream_invoke(query, session_id)
                async for token in streamer:
                    if token != "<<DONE>>":
                        print(f"[STREAM] Sending token: {token}")
                        # Format as SSE with proper newlines
                        yield f"data: {token}\n\n"
                    else:
                        print("[STREAM] Sending DONE token")
                        yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"[ERROR] Agent execution failed: {e}")
                error_step = f'<step><step_name>error</step_name>{{"error": "{str(e)}"}}</step>'
                yield f"data: {error_step}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except Exception as e:
        print(f"[ERROR] Request handling failed: {e}")
        return {"error": str(e)}

