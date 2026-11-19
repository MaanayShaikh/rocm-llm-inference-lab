"""Gradio demo:  python -m rag.app  (opens on http://localhost:7860)"""
import os
import gradio as gr
from rag.engine import RagEngine

engine = RagEngine(f"http://localhost:{os.environ.get('PORT', '8000')}",
                   os.environ.get("MODEL", "Qwen/Qwen3-8B"))


def ask(question):
    answer, hits = engine.answer(question)
    sources = "\n".join(f"[{h['source']}:{h['chunk']}] (score {h['score']:.2f})" for h in hits)
    return answer, sources


with gr.Blocks(title="ROCm LLM Inference Lab: RAG demo") as demo:
    gr.Markdown("# RAG on vLLM + AMD ROCm\nAsk about the documents in `data/docs`.")
    q = gr.Textbox(label="Question", placeholder="How do I start vLLM in the ROCm container?")
    a = gr.Textbox(label="Answer", lines=8)
    s = gr.Textbox(label="Retrieved sources", lines=4)
    gr.Button("Ask").click(ask, q, [a, s])
    q.submit(ask, q, [a, s])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
