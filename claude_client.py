import anthropic
from config import API_KEY, MAX_TOKENS

client = anthropic.Anthropic(api_key=API_KEY)

def chat(model: str, system: str, user: str,
         tools: list = None) -> anthropic.types.Message:
    kwargs = dict(
        model=model, max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    if tools:
        kwargs["tools"] = tools
    return client.messages.create(**kwargs)

def stream_chat(model: str, system: str, user: str):
    with client.messages.stream(
        model=model, max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}]
    ) as s:
        for text in s.text_stream:
            yield text

def get_text(msg) -> str:
    for b in msg.content:
        if b.type == "text":
            return b.text
    return ""

def get_tool_call(msg) -> dict | None:
    for b in msg.content:
        if b.type == "tool_use":
            return {"id": b.id, "name": b.name, "input": b.input}
    return None

def token_cost(msg, model: str) -> dict:
    pricing = {
        "claude-sonnet-4-20250514":  (3.0, 15.0),
        "claude-haiku-4-5-20251001": (0.25, 1.25),
    }
    inp, out = pricing.get(model, (3.0, 15.0))
    cost = (msg.usage.input_tokens / 1e6 * inp +
            msg.usage.output_tokens / 1e6 * out)
    return {
        "input_tokens":  msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
        "cost_usd":      round(cost, 6)
    }
