import json
from pathlib import Path

def load_patients(path="core_fine_test.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def get_patient_by_id(patients, pid: str):
    return next(p for p in patients if p["id"] == pid)

def build_patient_prompt(row: dict) -> str:
    core_beliefs = "\n".join([f"- {b}" for b in row.get("core_belief_fine_grained", [])])
    thoughts = row.get("thoughts", "").strip()

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
    """.strip()

SELECTED_IDX = [2, 5, 9, 13, 15, 16, 19, 20, 21, 22, 23, 25, 26, 31, 34, 36, 38, 39,
                41, 42, 43, 48, 49, 56, 65, 69, 70, 71, 73, 77, 78, 79, 80, 82, 87, 89,
                91, 92, 102, 106]

def select_cases_by_index(rows: list[dict], indices: list[int]) -> list[dict]:
    # remove duplicates while preserving order
    seen = set()
    unique = []
    for i in indices:
        if i not in seen:
            unique.append(i)
            seen.add(i)

    # keep only valid indices
    valid = [i for i in unique if 0 <= i < len(rows)]
    invalid = [i for i in unique if i not in valid]


    return [rows[i] for i in valid]

selected_rows = select_cases_by_index(patients, SELECTED_IDX)
print("Selected:", len(selected_rows))
print("Selected IDs:", [r.get("id") for r in selected_rows[:5]], "...")

