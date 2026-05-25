from claude_client import stream_chat
from config import MODEL_MAIN

PROMPTS = {
    "executive": """Create a crisp executive summary with:
- 3 bullet key findings (most important first)
- 1 strategic implication
- 1 recommended next action
Use plain business language. No jargon. Max 200 words.""",

    "technical": """Create a technical summary with:
- System/architecture overview
- Key components and interactions
- Technical risks and mitigations
Max 300 words.""",

    "board": """Create a board-level briefing with:
- Business impact (quantified where possible)
- Risk assessment (1 sentence)
- Decision required from leadership
Max 150 words. Extremely concise."""
}

def summarise(content: str, audience: str = "executive"):
    system = PROMPTS.get(audience, PROMPTS["executive"])
    user   = f"Summarise the following:\n\n{content[:3000]}"
    return stream_chat(MODEL_MAIN, system, user)
