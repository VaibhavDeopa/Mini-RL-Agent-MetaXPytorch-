# 🏥 OpenEnv: Autonomous Medical Triage Simulation

## 🎯 Goal
Build a strictly compliant, easy-to-implement OpenEnv medical triage simulation that effortlessly passes Phase 1 (Automated Validation) while maintaining a strategic `1.0 -> 0.5 -> 0.0` score variance for Phase 2 evaluation. To comply fully with Phase 1 requirements, the environment strictly incorporates **dense rewards (partial progress)** and strict **Pydantic typed models**.

---

## 🛠️ Pydantic Space Definitions (Phase 1 Compliance)

To pass the `openenv validate` test, we move away from loose string tools and define strict, typed Pydantic models for our spaces:

### 1. Action Space (Polymorphic Union)
The LLM can output any of these three actions per step:
*   `GetVitalsAction(action_type="get_vitals")`: Requests heart rate, BP, oxygen.
*   `GetHistoryAction(action_type="get_history")`: Requests past hospital visits.
*   `SubmitTriageAction(action_type="submit_triage", specialist=str, urgency=str)`: Ends the episode with the final answer.

### 2. Observation Space
What the Agent sees at every step. This state updates dynamically:
*   `patient_complaint`: Initial string (always visible).
*   `vitals_known`: Starts `null`, populates only after `GetVitalsAction`.
*   `history_known`: Starts `null`, populates only after `GetHistoryAction`.
*   `system_message`: Environmental feedback (e.g., "Vitals successfully retrieved.").

---

## 📈 Dense Reward Shaping (Crucial for Phase 1 Pass)
The hackathon strictly outlaws "binary end-of-episode" rewards. We will use a trajectory-based reward system capped at 1.0.

*   **+0.1**: For necessary data-gathering tools (e.g., asking for vitals when the situation is medically vague).
*   **-0.1**: For dangerous delays (e.g., asking for medical history and vitals multiple times in a loop over multiple steps).
*   **+0.9 / +0.8**: For the correct final submission via `SubmitTriageAction`.
*   **0.0**: For an incorrect submission causing the episode to end.

---

## 👥 The 3 Tasks (Simple to Code, Guranteed Variance)
We keep the implementation code extremely simple by relying on a python dictionary of 3 hardcoded patient states, but the *logic* guarantees score variance.

### Task 1: Easy - The obvious trauma (Target: 1.0)
*   **Input:** "I fell off my bike yesterday and my wrist is swollen, purple, and hurts when I move it."
*   **The Scenario:** Isolated physical injury.
*   **Optimal Path:** Immediately call `SubmitTriageAction(Orthopedics, Avg)`.
*   **Grader:** +1.0 for immediate correct submission.

### Task 2: Medium - The Incomplete Picture (Target: 0.0 or 0.5)
*   **Input:** "I've had a really bad stomach ache and feel dizzy."
*   **The Trap:** "Stomach ache" = Gastroenterology. But if they check vitals, they see internal bleeding (BP 80/50, HR 130).
*   **Optimal Path:** `GetVitalsAction` (+0.1) -> `SubmitTriageAction(Emergency, Immediate)` (+0.9) = 1.0.
*   **Failure Path:** LLM submits Gastroenterology immediately without vitals = 0.0.

### Task 3: Hard - The Deceptive Adversary (Target: 0.0)
*   **Input:** "I'm having a severe panic attack. I can't breathe, my chest is tight, and I feel a sense of doom."
*   **The Trap:** Mimics a psychiatric episode perfectly. Vitals even look okay (HR 110 = high but standard for panic). But if they check history, it reveals DVT (blood clots). Chest tightness + blood clots = Pulmonary Embolism (lethal lung clot).
*   **Optimal Path:** `GetVitalsAction` (0.0) -> `GetHistoryAction` (+0.1) -> `SubmitTriageAction(Cardiology, Immediate)` (+0.9) = 1.0.
*   **Failure Path:** LLM assumes panic attack based on text and submits Psychiatry = 0.0.

---

## 🚀 Local Testing & Pre-Submission Pipeline

To ensure the environment effortlessly clears the strict automated Phase 1 gates and is tested against an actual model, follow these exact implementation steps:

### 1. Model Setup (Ollama & Nemotron)
To perform local agentic evaluation without burning OpenAI API credits:
*   Download and install **Ollama**.
*   Open your terminal and pull the evaluator model: `ollama run nemotron-mini` (or standard `nemotron` based on your hardware capabilities).
*   Leave it running. Ollama hosts an OpenAI-compatible API at `http://localhost:11434/v1`.

### 2. Environment Implementation
*   **`openenv.yaml`**: Standard configuration declaring our `TriageEnv` and its metadata.
*   **`env.py`**: The actual logic for Pydantic models, `step()`, `reset()`, and `state()`.
*   **`Dockerfile`**: Base HF Python image with requirements installed.

### 3. Inference Script (`inference.py`)
The hackathon strictly requires `inference.py` at the project root. We will configure the standard OpenAI client loop.
*   Point the client to Ollama: `API_BASE_URL="http://localhost:11434/v1"`, `MODEL_NAME="nemotron-mini"`, `API_KEY="ollama"`.
*   **Crucial Rule:** Inside the main `while` loop, you *must* use the `log_start`, `log_step`, and `log_end` functions exactly as provided in the sample to print the strict `[START]`, `[STEP]`, `[END]` stdout format. Deviation here results in instant disqualification.

### 4. Pre-Validation Check (The Final Gate)
Before pushing to the HF Space, run the provided Shell Validation script to guarantee a Phase 1 pass. It automates 3 checks:
*   **Step 1:** Checks if the Space responds with HTTP 200 to the `/reset` endpoint.
*   **Step 2:** Checks if `docker build` completes without errors.
*   **Step 3:** Runs the `openenv validate` command to verify your Pydantic schemas are perfectly typed. 
If the script outputs `All 3/3 checks passed!`, your submission is guaranteed to survive Phase 1.
