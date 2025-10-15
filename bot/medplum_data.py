import os
import base64
import requests
import random
import time
from datetime import datetime, timedelta
from faker import Faker
from typing import Optional, Dict, List

# Initialize Faker and API
fake = Faker()
# Medplum FHIR API base URL server
BASE_URL = os.getenv("MEDPLUM_URL", "http://medplum-server:8103/fhir/R4")
TOKEN = os.getenv("MEDPLUM_TOKEN")


HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/fhir+json"
}

NUM_PATIENTS = 1
DELAY = 0.5  # seconds between uploads
RETRY_COUNT = 3  # number of retries for failed API calls


# --------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------

def log(message: str) -> None:
    """Print log with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def post_resource(resource_type: str, data: dict) -> Optional[dict]:
    """POST FHIR resource with retry and logging."""
    url = f"{BASE_URL}/{resource_type}"
    for attempt in range(RETRY_COUNT):
        try:
            resp = requests.post(url, json=data, headers=HEADERS, timeout=30)
            if resp.status_code in (200, 201):
                result = resp.json()
                log(f"✅ {resource_type} created: {result.get('id')}")
                return result
            else:
                # Log full OperationOutcome for debugging (trim if too verbose)
                try:
                    txt = resp.json()
                except Exception:
                    txt = resp.text
                log(f"⚠️  Failed {resource_type} ({resp.status_code}): {txt}")
        except Exception as e:
            log(f"❌ Error creating {resource_type}: {e}")

        time.sleep(1)  # wait before retry
    return None


# --------------------------------------------------------------------
# Data Generation Functions
# --------------------------------------------------------------------

def generate_patient() -> dict:
    """Generate a synthetic FHIR Patient resource."""
    gender = fake.random_element(["male", "female"])
    title = "Mr." if gender == "male" else "Ms."
    first_name = fake.first_name_male() if gender == "male" else fake.first_name_female()
    last_name = fake.last_name()
    full_name = f"{title} {first_name}{random.randint(1, 999)} {last_name}{random.randint(1, 999)}"

    birth_date = fake.date_of_birth(minimum_age=18, maximum_age=85)
    deceased_date = birth_date + timedelta(days=random.randint(20000, 30000)) if random.random() < 0.1 else None

    patient = {
        "resourceType": "Patient",
        "identifier": [
            {"system": "https://github.com/synthetichealth/synthea", "value": str(fake.uuid4())},
            {"system": "http://hospital.smarthealthit.org", "value": str(fake.uuid4())},
        ],
        "name": [{"use": "official", "text": full_name}],
        "telecom": [{"system": "phone", "value": fake.phone_number(), "use": "home"}],
        "gender": gender,
        "birthDate": str(birth_date),
        "address": [{
            "line": [fake.street_address()],
            "city": fake.city(),
            "state": fake.state(),
            "postalCode": fake.postcode(),
            "country": fake.country()
        }],
        "maritalStatus": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus",
                "code": fake.random_element(["S", "M", "D", "W"]),
            }]
        },
        "communication": [{"language": {"coding": [{"code": "en", "display": "English"}]}}],
    }

    if deceased_date:
        patient["deceasedDateTime"] = deceased_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    return patient


def generate_encounter(patient_id: str, date: datetime) -> dict:
    """Generate an Encounter resource."""
    return {
        "resourceType": "Encounter",
        "status": "finished",
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB", "display": "Ambulatory"},
        "subject": {"reference": f"Patient/{patient_id}"},
        "period": {"start": date.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": date.strftime("%Y-%m-%dT%H:%M:%SZ")},
        "reasonCode": [{"text": "Annual physical examination"}]
    }


def generate_observations(patient_id: str, encounter_id: str, exam_date: datetime) -> List[Dict]:
    """Generate multiple vital and lab Observation resources."""
    observation_templates = [
        {"code": "8302-2", "display": "Body height", "value": fake.random_int(150, 200), "unit": "cm"},
        {"code": "29463-7", "display": "Body weight", "value": fake.random_int(45, 130), "unit": "kg"},
        {"code": "8867-4", "display": "Heart rate", "value": fake.random_int(60, 120), "unit": "beats/min"},
        {"code": "8310-5", "display": "Body temperature", "value": round(random.uniform(36, 39), 1), "unit": "°C"},
        {"code": "9279-1", "display": "Respiratory rate", "value": fake.random_int(12, 28), "unit": "breaths/min"},
        {"code": "29271-4", "display": "Vision", "value": random.choice(["20/20", "20/30", "20/40"])},
        {"code": "38213-9", "display": "Hearing", "value": random.choice(["Normal", "Mild Loss", "Impaired"])},
        {"code": "2345-7", "display": "Glucose [Mass/volume] in Blood", "value": round(random.uniform(70, 180), 1), "unit": "mg/dL"},
        {"code": "17861-6", "display": "Calcium [Mass/volume] in Serum or Plasma", "value": round(random.uniform(8.5, 10.5), 2), "unit": "mg/dL"},
        {"code": "777-3", "display": "Platelets", "value": fake.random_int(150000, 450000), "unit": "/µL"},
        {"code": "785-6", "display": "MCHC", "value": round(random.uniform(32, 36), 1), "unit": "g/dL"},
        {"code": "789-8", "display": "Red Blood Cells", "value": round(random.uniform(4.0, 6.0), 2), "unit": "million/µL"},
        {"code": "6690-2", "display": "White Blood Cells", "value": fake.random_int(4000, 11000), "unit": "/µL"},
        {"code": "2085-9", "display": "Cholesterol [Mass/volume] in Serum or Plasma", "value": round(random.uniform(120, 250), 1), "unit": "mg/dL"},
        {"code": "2571-8", "display": "Creatinine [Mass/volume] in Serum or Plasma", "value": round(random.uniform(0.6, 1.3), 2), "unit": "mg/dL"},
        {"code": "2160-0", "display": "Hemoglobin A1c/Hemoglobin.total in Blood", "value": round(random.uniform(4.5, 6.5), 1), "unit": "%"},
        {"code": "718-7", "display": "Hemoglobin [Mass/volume] in Blood", "value": round(random.uniform(12, 17), 1), "unit": "g/dL"},
        {"code": "789-8", "display": "Hematocrit", "value": round(random.uniform(36, 50), 1), "unit": "%"},
        {"code": "48065-7", "display": "C-Reactive Protein (CRP)", "value": round(random.uniform(0, 10), 1), "unit": "mg/L"},
        {"code": "20570-8", "display": "Sodium [Moles/volume] in Serum or Plasma", "value": round(random.uniform(135, 145), 1), "unit": "mmol/L"},
        {"code": "2075-0", "display": "Potassium [Moles/volume] in Serum or Plasma", "value": round(random.uniform(3.5, 5.0), 1), "unit": "mmol/L"},
        {"code": "2751-8", "display": "Albumin [Mass/volume] in Serum or Plasma", "value": round(random.uniform(3.5, 5.0), 2), "unit": "g/dL"},
        {"code": "14682-9", "display": "Bilirubin.total in Serum or Plasma", "value": round(random.uniform(0.3, 1.2), 2), "unit": "mg/dL"},
    ]

    created_obs = []
    for od in observation_templates:
        # Randomly skip test
        value = od["value"] if random.random() < 0.8 else None

        obs = {
            "resourceType": "Observation",
            "status": "final" if value is not None else "registered",
            "code": {"coding": [{"system": "http://loinc.org", "code": od["code"], "display": od["display"]}]},
            "subject": {"reference": f"Patient/{patient_id}"},
            "encounter": {"reference": f"Encounter/{encounter_id}"},
            "effectiveDateTime": exam_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        if random.random() < 0.7:
            if "unit" in od:
                obs["valueQuantity"] = {
                "value": od["value"],
                "unit": od["unit"],
                "system": "http://unitsofmeasure.org",
                "code": od["unit"]
            }
            else:
                obs["valueString"] = od["value"]
        else:
        # Test not conducted yet → set value to null (FHIR-compliant)
            if "unit" in od:
                obs["valueQuantity"] = {
                "value": None,
                "unit": od["unit"],
                "system": "http://unitsofmeasure.org",
                "code": od["unit"]
            }
            else:
                obs["valueString"] = "Not Conducted"
        obs_res = post_resource("Observation", obs)
        
        if obs_res:
            created_obs.append({"reference": f"Observation/{obs_res['id']}"})

    return created_obs


def generate_conditions(patient_id: str) -> List[Dict]:
    """Generate medical history conditions."""
    conditions = [
    {"code": "44054006", "display": "Diabetes mellitus type 2"},
    {"code": "38341003", "display": "Hypertension"},
    {"code": "195967001", "display": "Asthma"},
    {"code": "73211009", "display": "Viral pneumonia"},
    {"code": "233604007", "display": "Pneumonia"},
    {"code": "22298006", "display": "Myocardial infarction"},
    {"code": "84114007", "display": "Depressive disorder"},
    {"code": "301011002", "display": "Anxiety disorder"},
    {"code": "698247007", "display": "COVID-19"},
    {"code": "109838007", "display": "Migraine"},
    {"code": "1801000119103", "display": "Chronic kidney disease"},
    {"code": "44054006", "display": "Diabetes mellitus type 2"},
    {"code": "235595009", "display": "Obesity"},
    {"code": "13645005", "display": "Rheumatoid arthritis"},
    {"code": "271327008", "display": "Anemia"},
    {"code": "399211009", "display": "Hypothyroidism"},
    {"code": "195967001", "display": "Asthma"},
    {"code": "254637007", "display": "Acute bronchitis"},
    {"code": "162864005", "display": "Tuberculosis"},
    {"code": "10509002", "display": "Chronic obstructive pulmonary disease"},
    {"code": "44054006", "display": "Type 2 diabetes mellitus"},
    {"code": "48176007", "display": "Cirrhosis of liver"},
    {"code": "125605004", "display": "Epilepsy"},
    {"code": "414545008", "display": "Coronary artery disease"},
    {"code": "40930008", "display": "Hyperlipidemia"}
    ]
    cond_refs = []
    num_conditions = random.randint(1, 4)
    selected_conditions = random.sample(conditions, num_conditions)

    for c in selected_conditions:
        cond = {
            "resourceType": "Condition",
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "verificationStatus": {"coding": [{"code": "confirmed"}]},
            "category": [{"coding": [{"code": "problem-list-item"}]}],
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": c["code"], "display": c["display"]}]},
            "subject": {"reference": f"Patient/{patient_id}"},
            "onsetDateTime": fake.date_between(start_date='-10y', end_date='-1y').strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        cond_res = post_resource("Condition", cond)
        if cond_res and "id" in cond_res:
            cond_refs.append({"reference": f"Condition/{cond_res['id']}"})
    return cond_refs


# nurse note text (shared baseline)



def create_diagnostic_reports(patient_id: str, encounter_id: str, obs_refs: List[Dict], cond_refs: List[Dict]):
    """Create multiple DiagnosticReports for observations and history, including a nurse note as a presentedForm."""
    now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    practitioner_name = f"Dr. {fake.first_name()} {fake.last_name()}, {random.choice(['MD', 'PhD', 'DDS', 'PharmD'])}"
    # Base64-encode nurse note for presentedForm.data (FHIR base64Binary)
    staff_name = f". {fake.first_name()} {fake.last_name()}"
    reports = [
        {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "code": {"text": "Patient Medical History Summary"},
            "subject": {"reference": f"Patient/{patient_id}"},
            "encounter": {"reference": f"Encounter/{encounter_id}"},
            "issued": now_iso,
            "performer": [{"display": practitioner_name}],
            "result": cond_refs
        },
        {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "code": {"text": "Vital Signs Panel"},
            "subject": {"reference": f"Patient/{patient_id}"},
            "encounter": {"reference": f"Encounter/{encounter_id}"},
            "issued": now_iso,
            "performer": [{"display": practitioner_name}],
            "result": obs_refs
        },
        {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "code": {"text": "Clinical Narrative"},
            "subject": {"reference": f"Patient/{patient_id}"},
            "encounter": {"reference": f"Encounter/{encounter_id}"},
            "issued": now_iso,
            "performer": [{"display": practitioner_name}],
            "conclusion": (
                "# Chief Complaint\n- Cough and congestion\n\n"
                "# History of Present Illness\nMild bronchitis with seasonal allergies.\n\n"
                "# Medications\n- Amoxicillin\n- Cetirizine\n\n"
                "# Assessment and Plan\nRest, hydration, and follow-up after 7 days."
            )
        },
        # Nurse note as DiagnosticReport.presentedForm (visible in Medplum Reports)
        {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "code": {"text": "Nurse Note"},
            "subject": {"reference": f"Patient/{patient_id}"},
            "encounter": {"reference": f"Encounter/{encounter_id}"},
            "issued": now_iso,
            "performer": [{"display": staff_name}],
            "presentedForm": [{
            "contentType": "text/markdown",
            "language": "en",
            "title": "Nurse Note"
            }],
            "conclusion": (
                "# Chief Complaint\n- Cough and congestion\n\n"
                "# History of Present Illness\nMild bronchitis with seasonal allergies.\n\n"
                "# Medications\n- Amoxicillin\n- Cetirizine\n\n"
                "# Assessment and Plan\nRest, hydration, and follow-up after 7 days."
            )
        }
    ]

    for rep in reports:
        post_resource("DiagnosticReport", rep)



# --------------------------------------------------------------------
# Main Workflow
# --------------------------------------------------------------------

def create_patient_with_data():
    """End-to-end patient data generation and upload."""
    patient = post_resource("Patient", generate_patient())
    if not patient:
        return
    patient_id = patient["id"]

    exam_date = fake.date_time_between(start_date='-2y', end_date='now')
    encounter = post_resource("Encounter", generate_encounter(patient_id, exam_date))
    if not encounter:
        return
    encounter_id = encounter["id"]
    
    obs_refs = generate_observations(patient_id, encounter_id, exam_date)
    cond_refs = generate_conditions(patient_id)
    create_diagnostic_reports(patient_id, encounter_id, obs_refs, cond_refs)

    
    log(f"🎉 Completed upload for Patient/{patient_id}\n")


if __name__ == "__main__":
    log("🚀 Starting Medplum Synthetic Patient Upload")
    for i in range(NUM_PATIENTS):
        log(f"--- Generating patient {i + 1} of {NUM_PATIENTS} ---")
        create_patient_with_data()
        time.sleep(DELAY)
    log("✅ All patients uploaded successfully!")
