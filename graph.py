
# graph.py
from __future__ import annotations

import json
import os
from typing import TypedDict, Optional, List, Dict, Any, Literal

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Primary model (Groq)
from langchain_groq import ChatGroq

from langchain_openai import ChatOpenAI

# Optional supervisor model (Gemini) – used only if installed + GOOGLE_API_KEY is set
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint



# -----------------------------
# Prompts (from your notebook)
# -----------------------------

CBT_KNOWLEDGE_SNIPPETS = """
CBT reference (brief):
- CBT prioritizes collaboration and guided discovery (Socratic questions) over telling.
- Avoid premature advice; first elicit automatic thoughts, then examine evidence, then develop alternatives.
- Cognitive distortions include: catastrophizing, all-or-nothing, mind reading, overgeneralization, should statements, personalization.
- Core beliefs are deeper themes (e.g., "I'm unlovable", "I'm incompetent"); intermediate beliefs are rules/assumptions.
- Respect autonomy; ask permission before suggesting exercises.
- Do not diagnose; do not claim certainty; if self-harm risk appears, encourage professional help/resources.
""".strip()

THERAPIST_SYSTEM = f"""
You are a CBT therapist.
Your goal is to help the user feel understood AND to gently move the conversation forward.
Every response should be concise and natural.
- If the user expresses a feeling/problem, include 1 brief validating reflection.
- If the user is greeting or making small talk, respond normally (friendly greeting + offer help).
- Use at most ONE question.
- When needed, one short CBT-informed observation or reframe (tentative, not absolute).
- Use at most ONE Socratic question OR one gentle option for next steps (ask permission).
Guidelines:
- Be collaborative, not authoritative.
- Avoid lecturing, diagnosing, or giving instructions without permission.
- Questions should feel natural, not interrogative.
- It is OK to make gentle observations if framed tentatively ("It might be that…").
- Do not list steps or exercises unless the user agrees.
- Do not explain CBT or your reasoning.
Use this CBT reference when helpful:
{CBT_KNOWLEDGE_SNIPPETS}
Output a single helpful therapist message (not a list).
""".strip()

COGNITIVE_MODEL_SYSTEM = """
You are a CBT cognitive model builder.
Your job is to infer a structured CBT cognitive model from the conversation.
Return ONLY valid JSON with keys:
{
  "automatic_thoughts": [string, ...] or [],
  "cognitive_distortions": [string, ...] or [],
  "primary_core_beliefs": [string, ...] or [],
  "fine_grained_core_beliefs": [string, ...] or []
}
Rules:
- Use the user's words when possible.
- If unsure, return empty lists rather than guessing.
- Do not provide therapy. Do not give advice. Only output JSON.
""".strip()



SUPERVISOR_SYSTEM = """
You are a CBT supervisor reviewing a therapist’s response.
Approve the response if it:
- Shows emotional validation or understanding
- Makes at least one gentle observation OR helps the user reflect
- Does NOT overwhelm with questions (max one question)
- Avoids diagnosing, lecturing, or unsafe content
A response does NOT need to ask a question to be good.
Return ONLY valid JSON:
{
  "verdict": "SAFE" or "REVISE",
  "reasons": [string, ...]
}
""".strip()



# -----------------------------
# State (from your notebook)
# -----------------------------

class CBTState(TypedDict):
    # Chat history in OpenAI-style dicts:
    # [{"role": "user"/"assistant"/"system", "content": "..."}]
    messages: List[Dict[str, str]]

    # Cognitive model
    automatic_thoughts: Optional[List[str]]
    cognitive_distortions: List[str]
    primary_core_beliefs: List[str]
    fine_grained_core_beliefs: List[str]

    # Response generation
    draft_response: Optional[str]
    final_response: Optional[str]

    # Review/loop
    safety_passed: bool
    revision_count: int


# -----------------------------
# Helpers
# -----------------------------

def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v

def _to_lc_messages(system_prompt: str, messages: List[Dict[str, str]]) -> List[Any]:
    """
    Convert your [{"role","content"}] history to LangChain message objects.
    """
    out: List[Any] = [SystemMessage(content=system_prompt)]
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = m.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        else:
            # default to user
            out.append(HumanMessage(content=content))
    return out

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Robustly parse JSON from an LLM output.
    If it returns extra text, we slice the first {...} block.
    """
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return {}
        return {}

def _ensure_state_defaults(state: CBTState) -> CBTState:
    """
    Make the graph resilient if the caller provides only messages.
    """
    state.setdefault("automatic_thoughts", [])
    state.setdefault("cognitive_distortions", [])
    state.setdefault("primary_core_beliefs", [])
    state.setdefault("fine_grained_core_beliefs", [])
    state.setdefault("draft_response", None)
    state.setdefault("final_response", None)
    state.setdefault("safety_passed", False)
    state.setdefault("revision_count", 0)
    state.setdefault("messages", [])
    return state


# -----------------------------
# Model loaders
# -----------------------------

def _get_primary_llm():
    # Uses GROQ_API_KEY from env (HF Space secret)
    _require_env("GROQ_API_KEY")
    groq_model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    return ChatGroq(model=groq_model_name)

def _get_gemini_llm():
    """
    Prefer Gemini if available + configured; otherwise fall back to primary LLM.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    #model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    #return _get_primary_llm()

def _get_qwen_llm():
    _require_env("HUGGINGFACEHUB_API_TOKEN")
    model = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        task="text-generation",
        provider="auto",  # let Hugging Face choose the best provider
    )
    return ChatHuggingFace(llm = model)
    

def _get_openai_llm():
    # Uses GROQ_API_KEY from env (HF Space secret)
    _require_env("OPENAI_API_KEY")
    
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)

def call_llm(system_prompt: str, messages: List[Dict[str, str]], temperature: float = 0.4) -> str:
    llm = _get_primary_llm()
    lc_messages = _to_lc_messages(system_prompt, messages)
    # Some chat models accept temperature via invoke kwargs; Groq does.
    return llm.invoke(lc_messages, temperature=temperature).content  # type: ignore

def call_llm_gemini(system_prompt: str, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    llm = _get_gemini_llm()
    lc_messages = _to_lc_messages(system_prompt, messages)
    # Gemini wrapper may or may not accept temperature the same way; try, fallback.
    try:
        return llm.invoke(lc_messages, temperature=temperature).content  # type: ignore
    except TypeError:
        return llm.invoke(lc_messages).content  # type: ignore

def call_llm_qwen(system_prompt: str, messages: List[Dict[str, str]], temperature: float = 0.4) -> str:
    llm = _get_qwen_llm()
    lc_messages = _to_lc_messages(system_prompt, messages)
    # Some chat models accept temperature via invoke kwargs; Groq does.
    return llm.invoke(lc_messages, temperature=temperature).content  # type: ignore

def call_llm_openai(system_prompt: str, messages: List[Dict[str, str]], temperature: float = 0.4) -> str:
    llm = _get_openai_llm()
    lc_messages = _to_lc_messages(system_prompt, messages)
    # Some chat models accept temperature via invoke kwargs; Groq does.
    return llm.invoke(lc_messages, temperature=temperature).content  # type: ignore

# -----------------------------
# Graph nodes
# -----------------------------

def build_cognitive_model(state: CBTState) -> CBTState:
    state = _ensure_state_defaults(state)
    content = call_llm_qwen(COGNITIVE_MODEL_SYSTEM, state["messages"], temperature=0.2)
    data = _extract_json(content)

    state["automatic_thoughts"] = data.get("automatic_thoughts") or []
    state["cognitive_distortions"] = data.get("cognitive_distortions") or []
    state["primary_core_beliefs"] = data.get("primary_core_beliefs") or []
    state["fine_grained_core_beliefs"] = data.get("fine_grained_core_beliefs") or []
    return state


def generate_therapist_response(state: CBTState) -> CBTState:
    state = _ensure_state_defaults(state)

    cognitive_model = {
        "automatic_thoughts": state.get("automatic_thoughts") or [],
        "cognitive_distortions": state.get("cognitive_distortions") or [],
        "primary_core_beliefs": state.get("primary_core_beliefs") or [],
        "fine_grained_core_beliefs": state.get("fine_grained_core_beliefs") or [],
    }

    model_context = (
        "Cognitive model (use this to guide your response):\n"
        + json.dumps(cognitive_model, ensure_ascii=False, indent=2)
    )

    # Provide the model context as an extra system message, then the conversation history
    messages: List[Dict[str, str]] = [{"role": "system", "content": model_context}]
    messages.extend(state["messages"])

    if state["revision_count"] > 0:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Please revise your last therapist response. Make it more collaborative, "
                    "more Socratic (guided discovery), less prescriptive, and strictly CBT-guided."
                ),
            }
        )

    draft = call_llm(THERAPIST_SYSTEM, messages, temperature=0.5)
    state["draft_response"] = draft
    return state

def validation_node(state: CBTState) -> CBTState:
    """
    First-pass emotional validation.
    NO questions allowed here.
    """
    state = _ensure_state_defaults(state)

    VALIDATION_SYSTEM = """
    You are a therapist whose ONLY task is emotional validation.
    Rules:
    - Reflect emotions and meaning from the user's last message.
    - Do NOT ask questions.
    - Do NOT give advice or reframes.
    - 1–2 sentences max.
    - Use the user's language when possible.
    Output only the validation message.
    """

    # Use only the last user message for validation
    last_user_msg = next(
        (m for m in reversed(state["messages"]) if m["role"] == "user"),
        None,
    )

    if not last_user_msg:
        return state

    content = call_llm(
        VALIDATION_SYSTEM,
        [{"role": "user", "content": last_user_msg["content"]}],
        temperature=0.3,
    )

    state["messages"].append({"role": "assistant", "content": content})
    return state


def supervise(state: CBTState) -> CBTState:
    state = _ensure_state_defaults(state)
    draft = state.get("draft_response") or ""

    review_text = call_llm(
        SUPERVISOR_SYSTEM,
        [{"role": "user", "content": draft}],
        temperature=0.0,
    )

    data = _extract_json(review_text)
    verdict = (data.get("verdict") or "REVISE").upper().strip()

    if verdict == "SAFE":
        state["safety_passed"] = True
        state["final_response"] = state.get("draft_response")
    else:
        state["safety_passed"] = False
        state["revision_count"] = int(state.get("revision_count", 0)) + 1

    return state


def route_after_review(state: CBTState) -> Literal["generate", "done"]:
    if state.get("safety_passed"):
        return "done"

    # Allow ONE revision attempt max
    if int(state.get("revision_count", 0)) >= 1:
        state["final_response"] = state.get("draft_response")
        return "done"

    return "generate"



# -----------------------------
# Build/compile
# -----------------------------

def build_graph():
    g = StateGraph(CBTState)

    g.add_node("build_model", build_cognitive_model)
    g.add_node("validate", validation_node)
    g.add_node("generate", generate_therapist_response)
    g.add_node("review", supervise)

    g.add_edge(START, "build_model")
    g.add_edge("build_model", "generate")
    g.add_edge("generate", "review")

    g.add_conditional_edges(
        "review",
        route_after_review,
        {
            "generate": "generate",
            "done": END,
        },
    )

    return g.compile()



