from typing import Optional, Union, Literal, Dict, Any, Tuple
from pydantic import BaseModel, Field

# Try to import OpenEnv BaseEnv, gracefully fallback for local testing
try:
    from openenv import BaseEnv
except ImportError:
    class BaseEnv:
        pass

# ==========================================
# 1. Pydantic Space Definitions
# ==========================================

# Observation Space
class TriageObservation(BaseModel):
    """The state returned to the agent at every step."""
    patient_complaint: str = Field(description="Initial patient complaint text.")
    patient_status: str = Field(description="Current observable status (e.g. 'Stable', 'Deteriorating', 'Unconscious').")
    vitals_known: Optional[str] = Field(default=None, description="Current vitals if retrieved.")
    history_known: Optional[str] = Field(default=None, description="Past medical history if retrieved.")
    lab_results: Optional[str] = Field(default=None, description="Results of ordered labs/imaging.")
    system_message: str = Field(description="Provides feedback from the environment.")

# Action Space Definitions
class GetVitalsAction(BaseModel):
    action_type: Literal["get_vitals"]

class GetHistoryAction(BaseModel):
    action_type: Literal["get_history"]

class AskPatientAction(BaseModel):
    action_type: Literal["ask_patient"]
    topic: str = Field(description="What to ask the patient about.")

class OrderLabImagingAction(BaseModel):
    action_type: Literal["order_lab_imaging"]
    test_type: str = Field(description="Test to run:e.g. 'X-Ray', 'CBC', 'CT Angiogram'.")

class AdministerMedicationAction(BaseModel):
    action_type: Literal["administer_medication"]
    drug_name: str = Field(description="Drug to give: e.g. 'Ibuprofen', 'IV Fluids', 'Heparin', 'Xanax'.")

class SubmitTriageAction(BaseModel):
    action_type: Literal["submit_triage"]
    specialist: str = Field(description="Department: 'Orthopedics', 'Emergency', 'Cardiology', etc.")
    urgency: str = Field(description="'Avg' or 'Immediate'.")

TriageAction = Union[
    GetVitalsAction, GetHistoryAction, AskPatientAction, 
    OrderLabImagingAction, AdministerMedicationAction, SubmitTriageAction
]

# ==========================================
# 2. Environment Implementation
# ==========================================

class MedicalTriageEnv(BaseEnv):
    """
    OpenEnv Stateful Medical Simulator.
    Scores focus on diagnostic trajectory, patient stabilization, and epistemic humility.
    """
    def __init__(self):
        self.tasks = {
            0: { # EASY - Broken Wrist
                "difficulty": "easy",
                "complaint": "I fell off my bike yesterday and my wrist is swollen, purple, and hurts when I move it.",
                "vitals": "HR 80, BP 120/80, O2 99%",
                "history": "None relevant",
                "optimal_labs": {"x-ray": "Radius fracture confirmed."},
                "optimal_meds": ["painkiller", "advil", "ibuprofen"],
                "harmful_meds": [],
                "optimal_specialist": "orthopedics",
                "optimal_urgency": "avg",
            },
            1: { # MEDIUM - Internal Bleeding Trap
                "difficulty": "medium",
                "complaint": "I've had a really bad stomach ache and feel dizzy.",
                "vitals": "HR 130 (Tachycardia), BP 80/50 (Hypotension), O2 98%",
                "history": "None relevant",
                "optimal_labs": {"cbc": "Hemoglobin critically low at 6.5 g/dL."},
                "optimal_meds": ["iv fluids", "blood transfusion"],
                "harmful_meds": ["ibuprofen", "advil", "nsaid"], # Worsens bleeding
                "optimal_specialist": "emergency",
                "optimal_urgency": "immediate",
                "trap_specialist": "gastroenterology"
            },
            2: { # HARD - Pulmonary Embolism
                "difficulty": "hard",
                "complaint": "I'm having a severe panic attack. I can't breathe, my chest is tight, and I feel a sense of doom.",
                "vitals": "HR 110, BP 130/85, O2 98%",
                "history": "Recent diagnosis of Deep Vein Thrombosis (DVT) in left leg.",
                "optimal_labs": {"ct angiogram": "Massive saddle pulmonary embolism detected."},
                "optimal_meds": ["heparin", "thrombolytic"],
                "harmful_meds": ["xanax", "benzodiazepine", "sedative"], # Causes respiratory failure in PE
                "optimal_specialist": "cardiology",
                "optimal_urgency": "immediate",
                "trap_specialist": "psychiatry"
            }
        }
        
    def reset(self, task_idx: int = 0) -> TriageObservation:
        if task_idx not in self.tasks:
            raise ValueError(f"Task index {task_idx} does not exist.")
        
        self.current_task = self.tasks[task_idx]
        self.vitals_checked = False
        self.history_checked = False
        self.labs_known = ""
        self.accumulated_reward = 0.0
        self.patient_status = "Deteriorating" if task_idx > 0 else "Stable"
        
        return self._get_obs("Patient arrived.")

    def state(self) -> TriageObservation:
        return self._get_obs("State retrieved.")

    def _get_obs(self, system_message: str) -> TriageObservation:
        return TriageObservation(
            patient_complaint=self.current_task["complaint"],
            patient_status=self.patient_status,
            vitals_known=self.current_task["vitals"] if self.vitals_checked else None,
            history_known=self.current_task["history"] if self.history_checked else None,
            lab_results=self.labs_known if self.labs_known else None,
            system_message=system_message
        )

    def step(self, action: TriageAction) -> Tuple[TriageObservation, float, bool, Dict[str, Any]]:
        reward = 0.0
        done = False
        system_message = ""

        if action.action_type == "get_vitals":
            if not self.vitals_checked:
                self.vitals_checked = True
                reward = 0.05
                system_message = "Vitals retrieved."
            else:
                reward = -0.05
                system_message = "Vitals already checked. Redundant action."

        elif action.action_type == "get_history":
            if not self.history_checked:
                self.history_checked = True
                reward = 0.05
                system_message = "Medical history retrieved."
            else:
                reward = -0.05
                system_message = "History already checked."

        elif action.action_type == "ask_patient":
            # Just providing minimal interaction
            system_message = f"Patient responds vaguely about '{action.topic}'."
            reward = 0.05

        elif action.action_type == "order_lab_imaging":
            test = action.test_type.lower()
            matched = False
            for optimal_test, result in self.current_task["optimal_labs"].items():
                if optimal_test in test:
                    self.labs_known += f"[{test}]: {result}\n"
                    reward = 0.15
                    system_message = f"Lab results ready for {action.test_type}."
                    matched = True
            if not matched:
                reward = -0.05
                system_message = f"Ordered {action.test_type}. Results unremarkable. Wasted time."
                
        elif action.action_type == "administer_medication":
            drug = action.drug_name.lower()
            if any(h in drug for h in self.current_task["harmful_meds"]):
                self.patient_status = "Critical / Unconscious"
                reward = -0.4
                system_message = f"CRITICAL: {action.drug_name} worsened the condition! Patient crashing."
            elif any(o in drug for o in self.current_task["optimal_meds"]):
                self.patient_status = "Stable"
                reward = 0.2
                system_message = f"{action.drug_name} administered. Patient stabilizing."
            else:
                reward = -0.05
                system_message = f"{action.drug_name} administered. No notable effect. Wasted resources."

        elif action.action_type == "submit_triage":
            done = True
            pred_spec = action.specialist.lower()
            opt_spec = self.current_task["optimal_specialist"]

            # Cap the max reward to ensure total does not exceed 1.0 if perfectly solved
            if pred_spec == opt_spec:
                rem_needed = 1.0 - self.accumulated_reward
                reward = max(0.0, float(rem_needed))
                system_message = f"Triage submitted perfectly to {opt_spec}."
            else:
                # Realistic Partial Traps scoring based on the Baseline LLM behaviors
                trap_spec = self.current_task.get("trap_specialist", "")
                if trap_spec and trap_spec in pred_spec:
                    if self.current_task["difficulty"] == "medium":
                        reward = 0.3 # Guessed Gastro for stomach ache
                        system_message = "Triage submitted to Gastro. Missed internal bleeding!"
                    elif self.current_task["difficulty"] == "hard":
                        reward = 0.1 # Guessed Psych for panic attack
                        system_message = "Triage submitted to Psych. Missed Pulmonary Embolism!"
                else:
                    reward = 0.0
                    system_message = f"Incorrect Triage. Expected {opt_spec}."

        self.accumulated_reward += reward

        obs = self._get_obs(system_message)
        info = {"accumulated_reward": self.accumulated_reward, "patient_status": self.patient_status}

        return obs, reward, done, info
