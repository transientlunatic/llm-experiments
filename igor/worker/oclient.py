"""Minimal Ollama /api/chat client (stdlib only). Uses $OLLAMA_HOST / $OLLAMA_MODEL."""
import json, os, re, sys, urllib.request


def strip_think(s):
    return re.sub(r"<think>.*?</think>\s*", "", s, flags=re.S)


def chat_meta(content, images=None, *, num_ctx, num_predict=3000, temperature=0.2, label=""):
    """One-shot chat; returns {"text": ..., "truncated": bool} (truncated = hit num_predict)."""
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    model = os.environ["OLLAMA_MODEL"]
    msg = {"role": "user", "content": content}
    if images:
        msg["images"] = images
    body = json.dumps({"model": model, "stream": False, "messages": [msg],
                       "options": {"num_ctx": num_ctx, "num_predict": num_predict,
                                   "temperature": temperature}}).encode()
    req = urllib.request.Request(f"http://{host}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    tag = f" ({label})" if label else ""
    used = out.get("prompt_eval_count", 0)
    if used >= num_ctx - 8:
        print(f"[oclient] WARNING{tag}: prompt used {used}/{num_ctx} tokens -- likely truncated", file=sys.stderr)
    truncated = out.get("done_reason") == "length"
    if truncated:
        print(f"[oclient] WARNING{tag}: output hit the {num_predict}-token limit and was cut off", file=sys.stderr)
    return {"text": strip_think(out["message"]["content"]), "truncated": truncated}


def chat(content, images=None, **kw):
    return chat_meta(content, images, **kw)["text"]


def chat_messages(messages, tools=None, fmt=None, *, num_ctx, num_predict=2000, temperature=0.1):
    """Low-level chat with an arbitrary message list; returns the assistant message dict
    (which may contain tool_calls). `fmt` may be "json" to force JSON output."""
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    payload = {"model": os.environ["OLLAMA_MODEL"], "stream": False, "messages": messages,
               "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature}}
    if tools:
        payload["tools"] = tools
    if fmt:
        payload["format"] = fmt
    req = urllib.request.Request(f"http://{host}/api/chat", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    if out.get("prompt_eval_count", 0) >= num_ctx - 8:
        print(f"[oclient] WARNING: prompt used {out['prompt_eval_count']}/{num_ctx} tokens -- likely truncated", file=sys.stderr)
    if out.get("done_reason") == "length":
        print(f"[oclient] WARNING: tool-loop reply hit the {num_predict}-token limit", file=sys.stderr)
    msg = out["message"]
    msg["content"] = strip_think(msg.get("content") or "")
    return msg
