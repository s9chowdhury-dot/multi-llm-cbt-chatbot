# app.py
import gradio as gr
from graph import build_graph

# Build LangGraph once at startup (fast + avoids rebuilding per message)
GRAPH = build_graph()

DISCLAIMER = (
    "⚠️ This demo is not a substitute for professional mental health care. "
    "If you feel in danger or may harm yourself, call your local emergency number "
    "or your local crisis line immediately."
)

def respond(
    message: str,
    history: list[dict[str, str]],
    system_message: str,
):
    """
    Gradio ChatInterface with type="messages" supplies:
      - message: the user's latest message (string)
      - history: list of {"role": "...", "content": "..."} dicts
    We convert that to the state expected by graph.py and return out["final_response"].
    """

    # Start with an optional system message for tone/style (kept separate from safety logic in graph.py)
    messages: list[dict[str, str]] = []
    if system_message and system_message.strip():
        messages.append({"role": "system", "content": system_message.strip()})

    # Include previous turns (already in {"role","content"} format)
    # Roles are typically "user" / "assistant" for Gradio "messages" history.
    messages.extend(history or [])

    # Add the new user message
    messages.append({"role": "user", "content": message})

    # Invoke LangGraph
    state = {"messages": messages}
    out = GRAPH.invoke(state)

    # Return final_response (fallbacks included for robustness)
    final = out.get("final_response")
    if final and isinstance(final, str):
        return final

    # Fallback: try draft_response or stringify the whole output
    return out.get("draft_response") or str(out)


chatbot = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="Therapy-Style Support Chatbot (Demo)",
    description=DISCLAIMER,
    save_history=True,
    editable=True,
    additional_inputs=[
        gr.Textbox(
            value="You are a warm, collaborative CBT-style support chatbot. "
                  "Ask Socratic questions, avoid lecturing, and keep responses concise.",
            label="System message (tone/style)",
        ),
    ],
)

with gr.Blocks() as demo:
    chatbot.render()

if __name__ == "__main__":
    demo.launch()