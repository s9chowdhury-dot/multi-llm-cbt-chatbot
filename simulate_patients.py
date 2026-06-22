# simulate_patients.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any

# Use your existing therapist graph
from graph import build_graph

# Use your existing Groq/LangChain helpers from graph.py
# (We import these to avoid duplicating env/model logic)
from graph import _get_primary_llm, _to_lc_messages


# -----------------------------
# Config: your selected indices
# -----------------------------
SELECTED_IDX = [
    2, 5, 9, 13, 15, 16, 19, 20, 21, 22, 23, 25, 26, 31, 34, 36, 38, 39,
    41, 42, 43, 48, 49, 56, 65, 69, 70, 71, 73, 77, 78, 79, 80, 82, 87, 89,
    91, 92, 102, 106
]


# -----------------------------
# Data loading + selection
# -----------------------------
def load_patients(path: str = "patients.json") -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_cases_by_index(rows: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    # de-dup while preserving order
    seen = set()
    unique = []
    for i in indices:
        if i not in seen:
            unique.append(i)
            seen.add(i)

    valid = [i for i in unique if 0 <= i < len(rows)]
    invalid = [i for i in unique if i not in valid]
    if invalid:
        print(f"⚠️ Skipping out-of-range indices (dataset size={len(rows)}): {invalid}")

    return [rows[i] for i in valid]


# -----------------------------
# Patient prompt builder (your format)
# -----------------------------
def build_patient_prompt(row: dict[str, Any]) -> str:
    core_beliefs = "\n".join([f"- {b}" for b in row.get("core_belief_fine_grained", [])])
    thoughts = (row.get("thoughts") or "").strip()

    return f"""
Imagine you are XXX, a patient who has been experiencing mental health challenges.
You have been attending therapy sessions for several weeks.

Your task is to engage in a conversation with the therapist as XXX would during a
cognitive behavioral therapy (CBT) session.

Align your responses with XXX’s background information provided in the
‘Patient History’ section below.

Your responses should be guided by the cognitive conceptualization information
provided, but avoid directly referencing or naming any CBT concepts, as a real
patient would not explicitly think in those terms.

Patient History:
{row.get("ori_text","")}

Cognitive Conceptualization Information (for internal guidance only):

Core Beliefs:
{core_beliefs}

Situation:
{row.get("situation","")}

Typical Thoughts:
{thoughts}

You will be asked about your experiences over the past week.
Engage in a conversation with the therapist regarding the situation above.

Use the provided thoughts and beliefs as a reference for how you interpret
events and yourself, but do not disclose this structure directly.
Instead, allow your responses to be informed by these elements in a natural,
implicit way, enabling the therapist to infer your thought patterns over time.

Adhere to the following guidelines:
1. Use natural, conversational language, including hesitations, pauses, emotional expressions.
2. Do not use clinical/therapy terminology. Do not name “core beliefs” or “automatic thoughts.”
3. Gradually reveal deeper concerns over multiple turns.
4. Maintain consistency with XXX’s profile, especially the core beliefs listed above.
5. You may minimize, resist, deflect, joke, or feel embarrassed when topics feel sensitive.
6. Do not resolve your own problems without the therapist’s guidance.
7. Respond only as the patient.
8. Do not copy long phrases verbatim from the Patient History; paraphrase naturally.
""".strip()


# -----------------------------
# Patient simulator call
# -----------------------------
def call_patient_llm(patient_system_prompt: str, convo: list[dict[str, str]], temperature: float = 0.8) -> str:
    """
    convo is OpenAI-style dict messages: [{"role":"user"/"assistant"/"system","content":"..."}]
    """
    llm = _get_primary_llm()
    lc_messages = _to_lc_messages(patient_system_prompt, convo)
    return llm.invoke(lc_messages, temperature=temperature).content  # type: ignore


# -----------------------------
# Simulation runner
# -----------------------------
GRAPH = build_graph()

DEFAULT_THERAPIST_OPENING = "Hi—good to see you. How has this past week been for you?"


def simulate_session(
    row: dict[str, Any],
    turns: int = 10,
    therapist_opening: str = DEFAULT_THERAPIST_OPENING,
) -> dict[str, Any]:
    """
    Runs a patient↔therapist conversation for `turns` patient turns.
    Returns a dict that you can save as JSON.
    """
    patient_system = build_patient_prompt(row)

    transcript: list[dict[str, str]] = []

    # Start with therapist opening (so patient doesn't monologue first)
    if therapist_opening:
        transcript.append({"role": "assistant", "content": therapist_opening})

    last_graph_out: dict[str, Any] = {}

    for _ in range(turns):
        # 1) patient response as "user"
        patient_convo = transcript + [
            {"role": "user", "content": "Respond as the patient to the therapist’s last message."}
        ]
        patient_msg = call_patient_llm(patient_system, patient_convo, temperature=0.85)
        transcript.append({"role": "user", "content": patient_msg})

        # 2) therapist response via your LangGraph
        last_graph_out = GRAPH.invoke({"messages": transcript})
        therapist_msg = (last_graph_out.get("final_response") or last_graph_out.get("draft_response") or "").strip()
        transcript.append({"role": "assistant", "content": therapist_msg})

    # Package results
    result = {
        "patient_id": row.get("id"),
        "source_row": {
            "id": row.get("id"),
            "ori_text": row.get("ori_text"),
            "situation": row.get("situation"),
            "thoughts": row.get("thoughts"),
            "core_belief_fine_grained": row.get("core_belief_fine_grained"),
        },
        "patient_prompt": patient_system,
        "therapist_opening": therapist_opening,
        "turns": turns,
        "transcript": transcript,
        "last_cognitive_model": {
            "automatic_thoughts": last_graph_out.get("automatic_thoughts", []),
            "cognitive_distortions": last_graph_out.get("cognitive_distortions", []),
            "primary_core_beliefs": last_graph_out.get("primary_core_beliefs", []),
            "fine_grained_core_beliefs": last_graph_out.get("fine_grained_core_beliefs", []),
        },
    }
    return result


# -----------------------------
# Saving helpers
# -----------------------------
def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_transcript_json(result: dict[str, Any], out_dir: str = "transcripts/individual") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pid = result.get("patient_id") or "unknown"
    path = Path(out_dir) / f"patient_{pid}_{_timestamp()}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def save_transcript_txt(result: dict[str, Any], out_dir: str = "transcripts/individual_txt") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pid = result.get("patient_id") or "unknown"
    path = Path(out_dir) / f"patient_{pid}_{_timestamp()}.txt"

    lines: list[str] = []
    for msg in result["transcript"]:
        speaker = "PATIENT" if msg["role"] == "user" else "THERAPIST"
        lines.append(f"{speaker}:\n{msg['content']}\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def save_batch_json(results: list[dict[str, Any]], out_path: str = "transcripts/batch_40.json") -> str:
    Path("transcripts").mkdir(exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


# -----------------------------
# Main: run your 40 cases
# -----------------------------
def run_batch(
    patients_path: str = "patients.json",
    turns: int = 10,
    save_txt: bool = False,
) -> list[dict[str, Any]]:
    patients = load_patients(patients_path)
    selected = select_cases_by_index(patients, SELECTED_IDX)
    print(f"Loaded {len(patients)} patients, selected {len(selected)} cases.")

    results: list[dict[str, Any]] = []
    for i, row in enumerate(selected, start=1):
        print(f"[{i}/{len(selected)}] Simulating patient id={row.get('id')}...")
        res = simulate_session(row=row, turns=turns)
        results.append(res)

        save_transcript_json(res)
        if save_txt:
            save_transcript_txt(res)

    save_batch_json(results)
    print("✅ Saved transcripts to transcripts/ ...")
    return results


if __name__ == "__main__":
    # Run 40 cases × 10 turns each, save JSON per patient + batch JSON
    run_batch(patients_path="patients.json", turns=10, save_txt=False)
