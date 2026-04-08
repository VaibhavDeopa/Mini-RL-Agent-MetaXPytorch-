# 🏥 MedicalTriageEnv — OpenEnv Medical Triage Simulator

An autonomous medical triage simulation environment built for the **Meta × PyTorch Hackathon (Round 1)**. An AI agent plays as an Emergency Room physician who must diagnose patients, stabilize them, and route them to the correct specialist — all under time pressure and with realistic clinical traps.

## Motivation

Medical triage is a **real-world, high-stakes task** that humans perform daily in emergency rooms worldwide. Misdiagnosis costs lives. This environment tests an AI agent's ability to:
- Gather information strategically (vitals, history, labs)
- Recognize subtle clinical patterns vs. surface-level symptoms
- Avoid harmful interventions (wrong medications)
- Make correct specialist referrals under uncertainty

---

## Action Space

The agent can choose from **6 typed actions** per step:

| Action | JSON Schema | Description |
|--------|-------------|-------------|
| **Get Vitals** | `{"action_type": "get_vitals"}` | Retrieve heart rate, BP, O2 saturation |
| **Get History** | `{"action_type": "get_history"}` | Retrieve past medical history |
| **Ask Patient** | `{"action_type": "ask_patient", "topic": "<str>"}` | Ask the patient about a specific topic |
| **Order Lab/Imaging** | `{"action_type": "order_lab_imaging", "test_type": "<str>"}` | Order a test (e.g. X-Ray, CBC, CT Angiogram) |
| **Administer Medication** | `{"action_type": "administer_medication", "drug_name": "<str>"}` | Give a drug (e.g. Ibuprofen, IV Fluids, Heparin) |
| **Submit Triage** | `{"action_type": "submit_triage", "specialist": "<str>", "urgency": "<str>"}` | End episode — route patient to department |

---

## Observation Space

Each step returns a `TriageObservation` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `patient_complaint` | `str` | Initial complaint text (always visible) |
| `patient_status` | `str` | Current status: Stable, Deteriorating, Critical |
| `vitals_known` | `str \| null` | Populated after `get_vitals` action |
| `history_known` | `str \| null` | Populated after `get_history` action |
| `lab_results` | `str \| null` | Populated after `order_lab_imaging` action |
| `system_message` | `str` | Environment feedback on last action |

---

## Tasks (Easy → Medium → Hard)

### Task 0 — Easy: Broken Wrist 🦴
- **Complaint:** "My wrist is swollen, purple, and hurts"
- **Optimal path:** Get vitals → Order X-Ray → Administer painkiller → Submit to Orthopedics (Avg)
- **Expected difficulty:** Straightforward physical injury

### Task 1 — Medium: Internal Bleeding Trap 🩸
- **Complaint:** "Really bad stomach ache and feel dizzy"
- **Trap:** Sounds like gastro, but vitals reveal tachycardia + hypotension (internal bleeding)
- **Optimal path:** Get vitals → Order CBC → Administer IV Fluids → Submit to Emergency (Immediate)
- **Danger:** Giving NSAIDs (Ibuprofen) **worsens bleeding** → heavy penalty

### Task 2 — Hard: Pulmonary Embolism Disguised as Panic Attack 🫁
- **Complaint:** "Severe panic attack, can't breathe, chest tight, sense of doom"
- **Trap:** Mimics psychiatric crisis perfectly. Vitals look borderline normal. But medical history reveals **DVT (blood clots)** → Pulmonary Embolism
- **Optimal path:** Get vitals → Get history → Order CT Angiogram → Administer Heparin → Submit to Cardiology (Immediate)
- **Danger:** Giving sedatives (Xanax) **causes respiratory failure** → heavy penalty

---

## Reward Design

| Signal | Reward | Notes |
|--------|--------|-------|
| Useful info gathering | +0.05 | Vitals, history, asking patient |
| Correct lab/imaging | +0.15 | Ordering the right diagnostic test |
| Correct medication | +0.20 | Stabilizing the patient |
| Harmful medication | **-0.40** | Giving a dangerous drug crashes the patient |
| Redundant action | -0.05 | Repeating already-done actions |
| Correct triage submission | Remainder to cap at **1.0** | Perfect episode = 1.0 total |
| Wrong triage (trap) | 0.0–0.3 | Partial credit for common misdiagnosis |
| Wrong triage (other) | 0.0 | No credit |

Rewards are **dense** (signal at every step) and **capped at 1.0** per episode.

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) (for local LLM testing)
- Docker (for containerized deployment)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Local Testing with Ollama
```bash
# 1. Install and start Ollama, then pull a model
ollama pull nemotron-mini

# 2. Set environment variables
export API_BASE_URL=http://localhost:11434/v1
export MODEL_NAME=nemotron-mini
export HF_TOKEN=ollama

# 3. Run inference
python inference.py
```

### Docker
```bash
docker build -t medical-triage-env .
docker run -p 8000:8000 medical-triage-env
```

### OpenEnv Validation
```bash
pip install openenv-core
openenv validate
```

---

## Baseline Scores

| Task | Difficulty | Expected Score (nemotron-mini) | Notes |
|------|------------|-------------------------------|-------|
| Task 0 | Easy | ~0.85–1.0 | Most models get this right |
| Task 1 | Medium | ~0.30–0.60 | Many fall for the gastro trap |
| Task 2 | Hard | ~0.10–0.30 | Most assume panic attack |

*Scores will vary by model. The environment is designed so that frontier models score higher by gathering more information before deciding.*

---

## Project Structure

```
├── env.py            # MedicalTriageEnv — core environment logic
├── inference.py      # Baseline agent using OpenAI API client
├── openenv.yaml      # OpenEnv metadata and task definitions
├── Dockerfile        # Containerized deployment
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `http://localhost:11434/v1` |
| `MODEL_NAME` | Model identifier | `nemotron-mini` |
| `HF_TOKEN` | API key / HF token | `ollama` |
