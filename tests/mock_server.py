"""Tiny fake OpenAI-compatible streaming server so the benchmark can be smoke-tested without a GPU.
Run:  python tests/mock_server.py  (listens on :8000)"""
import asyncio
import json
from aiohttp import web


async def chat(request):
    body = await request.json()
    n = min(body.get("max_tokens", 32), 32)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await asyncio.sleep(0.05)
    for i in range(n):
        chunk = {"choices": [{"delta": {"content": f"tok{i} "}}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await asyncio.sleep(0.005)
    usage = {"choices": [], "usage": {"completion_tokens": n}}
    await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


app = web.Application()
app.router.add_post("/v1/chat/completions", chat)
if __name__ == "__main__":
    web.run_app(app, port=8000)
