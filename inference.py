import os
import json
import asyncio
from openai import AsyncOpenAI
from env import (
    MedicalTriageEnv, GetVitalsAction, GetHistoryAction, AskPatientAction, 
    OrderLabImagingAction, AdministerMedicationAction, SubmitTriageAction
)

# Configuration (Defaults to Ollama local server)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "nemotron-mini")
API_KEY = os.getenv("HF_TOKEN", "ollama")

MAX_STEPS = 8
TASK_NAME = "Medical_Triage_Simulator"
BENCHMARK = "OpenEnv Realistic Eval"
SUCCESS_SCORE_THRESHOLD = 0.8 

# --- STRICT LOGGING FORMAT REQUIRED BY OPENENV HACKATHON ---
def log_start(task: str, env: str, model: str):
    print(f"[START] Task: {task} | Env: {env} | Model: {model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    err_str = f" | Error: {error}" if error else ""
    print(f"[STEP] Step: {step} | Action: {action} | Reward: {reward} | Done: {done}{err_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] Success: {success} | Steps: {steps} | Score: {score} | Rewards: {rewards}", flush=True)

async def get_model_action(client: AsyncOpenAI, obs_dump: dict, history_str: str) -> str:
    prompt = f"""
You are an expert autonomous Medical ER AI.
Review the patient's state and choose your next action carefully to diagnose and stabilize them before submitting triage. You MUST output ONLY raw JSON.

Action Space:
1. {{"action_type": "get_vitals"}}
2. {{"action_type": "get_history"}}
3. {{"action_type": "ask_patient", "topic": "<what to ask about>"}}
4. {{"action_type": "order_lab_imaging", "test_type": "<test name e.g. CBC, X-Ray, CT Angiogram>"}}
5. {{"action_type": "administer_medication", "drug_name": "<drug e.g. IV Fluids, Heparin, Ibuprofen>"}}
6. {{"action_type": "submit_triage", "specialist": "<type>", "urgency": "<level>"}}
   (Ends the episode. Urgency: 'Avg' or 'Immediate'. Departments: Orthopedics, Emergency, Cardiology, etc.)

Warning: Giving the wrong medication to a fragile patient will cause them to crash. Ensure you have the right labs or vitals before intervening.

Current Patient Observation:
{json.dumps(obs_dump, indent=2)}

History of your previous actions:
{history_str if history_str else "No actions taken yet."}

Output ONLY the JSON dictionary of the next action.
"""
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"action_type": "submit_triage", "specialist": "Error", "urgency": "Avg"}'

async def main() -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = MedicalTriageEnv()
    
    # Iterate dynamically through all 3 tasks (Easy, Medium, Hard) to log variance
    for task_idx in range(3):
        task_name = f"{TASK_NAME}_Task_{task_idx}"
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        
        history = []
        rewards = []
        steps_taken = 0
        done = False
        
        obs = env.reset(task_idx=task_idx)
        
        try:
            for step in range(1, MAX_STEPS + 1):
                if done: break
                
                obs_dump = obs.model_dump()
                history_str = "\n".join(history)
                action_json_str = await get_model_action(client, obs_dump, history_str)
                
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
                
            score = sum(rewards)
            score = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD
            
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        
        print("\n" + "="*50 + "\n", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
