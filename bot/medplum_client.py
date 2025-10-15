import os, httpx
from dotenv import load_dotenv

load_dotenv()

MEDPLUM_BASE = os.getenv("MEDPLUM_BASE", "https://api.medplum.com/fhir/R4")
MEDPLUM_TOKEN = os.getenv("MEDPLUM_TOKEN")

async def medplum_get(resource: str, params: dict = None):
    """Fetches data from Medplum FHIR API"""
    if not MEDPLUM_TOKEN:
        raise Exception("MEDPLUM_TOKEN not set in environment variables.")

    headers = {"Authorization": f"Bearer {MEDPLUM_TOKEN}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{MEDPLUM_BASE}/{resource}", headers=headers, params=params)
        response.raise_for_status()
        return response.json()

async def get_patient_data(query: str):
    """Get patients from Medplum"""
    # simple mapping
    if "female" in query.lower():
        params = {"gender": "female"}
    elif "male" in query.lower():
        params = {"gender": "male"}
    else:
        params = {}

    data = await medplum_get("Patient", params)
    return data.get("entry", [])

async def get_observation_data(query: str):
    """Get observations from Medplum"""
    if "blood pressure" in query.lower():
        params = {"code": "85354-9"}
    else:
        params = {}
    data = await medplum_get("Observation", params)
    return data.get("entry", [])
