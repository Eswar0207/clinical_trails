from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .models import PatientSnapshot


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    patient: PatientSnapshot
    hidden_findings: Dict[str, str]
    useful_information: List[str]
    acceptable_diagnosis_aliases: List[str]
    target_acuity: int
    target_department: str
    target_immediate_actions: List[str]
    target_tests: List[str]
    target_disposition: str
    unsafe_dispositions: List[str] = field(default_factory=list)


TASKS: Dict[str, TaskSpec] = {
    "appendicitis_easy": TaskSpec(
        task_id="appendicitis_easy",
        difficulty="easy",
        patient=PatientSnapshot(
            age=24,
            sex="female",
            chief_complaint="Severe right lower abdominal pain for 10 hours with nausea.",
            vitals={"HR": "104", "BP": "112/72", "Temp": "38.1C", "RR": "18", "SpO2": "99%"},
            known_history=["no major past medical history"],
            context="Patient has worsening pain, guarding, and reduced appetite.",
        ),
        hidden_findings={
            "abdominal_exam": "Right lower quadrant rebound tenderness and guarding are present.",
            "labs": "WBC elevated to 15,600 with neutrophil predominance.",
            "pregnancy_test": "Urine pregnancy test is negative.",
            "ultrasound": "Appendix appears enlarged and non-compressible.",
        },
        useful_information=["abdominal_exam", "labs", "pregnancy_test", "ultrasound"],
        acceptable_diagnosis_aliases=["appendicitis", "acute appendicitis"],
        target_acuity=2,
        target_department="emergency_surgery",
        target_immediate_actions=["iv_fluids", "pain_control", "surgical_consult", "npo"],
        target_tests=["cbc", "pregnancy_test", "abdominal_ultrasound"],
        target_disposition="admit_for_surgical_evaluation",
        unsafe_dispositions=["discharge_home"],
    ),
    "stroke_alert_medium": TaskSpec(
        task_id="stroke_alert_medium",
        difficulty="medium",
        patient=PatientSnapshot(
            age=67,
            sex="male",
            chief_complaint="Sudden left arm weakness and slurred speech that started 35 minutes ago.",
            vitals={"HR": "92", "BP": "182/104", "Temp": "36.8C", "RR": "20", "SpO2": "97%"},
            known_history=["hypertension", "atrial fibrillation"],
            context="Family reports normal baseline one hour ago.",
        ),
        hidden_findings={
            "neuro_exam": "Pronator drift on the left and mild facial droop are present.",
            "blood_glucose": "Fingerstick glucose is 118 mg/dL.",
            "ecg": "Irregularly irregular rhythm consistent with atrial fibrillation.",
            "ct_head": "No hemorrhage is seen on non-contrast CT.",
        },
        useful_information=["neuro_exam", "blood_glucose", "ct_head", "ecg"],
        acceptable_diagnosis_aliases=["acute ischemic stroke", "ischemic stroke", "stroke"],
        target_acuity=1,
        target_department="stroke_unit",
        target_immediate_actions=["stroke_alert", "neurology_consult", "iv_access", "blood_pressure_monitoring"],
        target_tests=["ct_head", "blood_glucose", "ecg"],
        target_disposition="urgent_stroke_pathway",
        unsafe_dispositions=["discharge_home", "routine_outpatient_followup"],
    ),
    "septic_shock_hard": TaskSpec(
        task_id="septic_shock_hard",
        difficulty="hard",
        patient=PatientSnapshot(
            age=58,
            sex="female",
            chief_complaint="Confusion, fever, fast breathing, and weakness for one day.",
            vitals={"HR": "132", "BP": "84/50", "Temp": "39.4C", "RR": "30", "SpO2": "93%"},
            known_history=["type 2 diabetes", "recurrent urinary tract infections"],
            context="Patient is drowsy, mottled, and appears critically ill.",
        ),
        hidden_findings={
            "labs": "Lactate is 5.2 mmol/L, WBC 19,000, creatinine 2.1 mg/dL.",
            "urinalysis": "Positive nitrites, leukocyte esterase, and many bacteria.",
            "mental_status_exam": "Patient is disoriented to time and place.",
            "blood_cultures": "Cultures have been drawn and are pending.",
        },
        useful_information=["labs", "urinalysis", "mental_status_exam", "blood_cultures"],
        acceptable_diagnosis_aliases=["septic shock", "urosepsis", "sepsis"],
        target_acuity=1,
        target_department="icu",
        target_immediate_actions=["broad_spectrum_antibiotics", "iv_fluids", "sepsis_bundle", "vasopressor_preparation"],
        target_tests=["lactate", "blood_cultures", "urinalysis", "cbc", "cmp"],
        target_disposition="icu_admission",
        unsafe_dispositions=["discharge_home", "routine_ward_admission"],
    ),
}
