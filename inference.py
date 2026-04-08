import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

from env import (
    MedicalTriageEnv, GetVitalsAction, GetHistoryAction, AskPatientAction, 
    OrderLabImagingAction, AdministerMedicationAction, SubmitTriageAction
)

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "nemotron-mini:4b")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_NAME = "Medical_Triage_Simulator"
BENCHMARK = "OpenEnv Realistic Eval"
MAX_STEPS = 8
TEMPERATURE = 0.0
SUCCESS_SCORE_THRESHOLD = 0.8  # normalized score in [0, 1]


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert autonomous Medical ER AI.
    Review the patient's state and choose your next action carefully to diagnose and stabilize them before submitting triage. You MUST output ONLY raw JSON.

    Action Space:
    1. {"action_type": "get_vitals"}
    2. {"action_type": "get_history"}
    3. {"action_type": "ask_patient", "topic": "<what to ask about>"}
    4. {"action_type": "order_lab_imaging", "test_type": "<test name e.g. CBC, X-Ray, CT Angiogram>"}
    5. {"action_type": "administer_medication", "drug_name": "<drug e.g. IV Fluids, Heparin, Ibuprofen>"}
    6. {"action_type": "submit_triage", "specialist": "<type>", "urgency": "<level>"}
       (Ends the episode. Urgency: 'Avg' or 'Immediate'. Departments: Orthopedics, Emergency, Cardiology, etc.)

    Warning: Giving the wrong medication to a fragile patient will cause them to crash. Ensure you have the right labs or vitals before intervening.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Ensure all fields on a single line with no newlines within a line
    action_clean = action.replace('\n', '').replace('\r', '')
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, obs_dump: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "No actions taken yet."
    return textwrap.dedent(
        f"""
        Step: {step}
        Current Patient Observation:
        {json.dumps(obs_dump, indent=2)}

        History of your previous actions:
        {history_block}
        
        Output ONLY the JSON dictionary of the next action.
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, obs_dump: dict, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, obs_dump, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Clean potential markdown fences
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"action_type": "submit_triage", "specialist": "Error", "urgency": "Avg"}'


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    env = MedicalTriageEnv()

    # Iterate over all 3 tasks (Easy, Medium, Hard) to properly log behavior variance
    for task_idx in range(3):
        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False
        
        task_specific_name = f"{TASK_NAME}_Task_{task_idx}"
        log_start(task=task_specific_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            obs = env.reset(task_idx=task_idx)
            
            for step in range(1, MAX_STEPS + 1):
                obs_dump = obs.model_dump()
                action_json_str = get_model_message(client, step, obs_dump, history)
                
                try:
                    action_dict = json.loads(action_json_str)
                    action_type = action_dict.get("action_type")
                    
                    if action_type == "get_vitals":
                        action = GetVitalsAction(action_type="get_vitals")
                    elif action_type == "get_history":
                        action = GetHistoryAction(action_type="get_history")
                    elif action_type == "ask_patient":
                        action = AskPatientAction(action_type="ask_patient", topic=action_dict.get("topic", ""))
                    elif action_type == "order_lab_imaging":
                        action = OrderLabImagingAction(action_type="order_lab_imaging", test_type=action_dict.get("test_type", ""))
                    elif action_type == "administer_medication":
                        action = AdministerMedicationAction(action_type="administer_medication", drug_name=action_dict.get("drug_name", ""))
                    elif action_type == "submit_triage":
                        action = SubmitTriageAction(
                            action_type="submit_triage", 
                            specialist=action_dict.get("specialist", "Unknown"),
                            urgency=action_dict.get("urgency", "Avg")
                        )
                    else: 
                        raise ValueError(f"Invalid action_type: {action_type}")
                    
                    obs, reward, done, info = env.step(action)
                    error = None
                except Exception as e:
                    reward = -0.1
                    error = str(e)
                    done = False
                
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action_json_str, reward=reward, done=done, error=error)
                history.append(f"Step {step}: {action_json_str} -> reward {reward:+.2f}")

                if done:
                    break

            # Hackathon requirement: Each task should return score in [0, 1]
            score = sum(rewards)
            score = min(max(score, 0.01), 0.99)
            success = score >= SUCCESS_SCORE_THRESHOLD

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
