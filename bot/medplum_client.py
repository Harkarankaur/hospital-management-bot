import os, httpx
from dotenv import load_dotenv

load_dotenv()
MEDPLUM_BASE_URL = os.getenv("MEDPLUM_BASE_URL")

MEDPLUM_TOKEN = os.getenv("MEDPLUM_TOKEN")

async def medplum_get(resource: str, params: dict = None):
    """Fetches data from Medplum FHIR API"""
    if not MEDPLUM_TOKEN:
        raise Exception("MEDPLUM_TOKEN not set in environment variables.")

    headers = {"Authorization": f"Bearer {MEDPLUM_TOKEN}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{MEDPLUM_BASE_URL}/{resource}", headers=headers, params=params)
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
async def get_resource(resource: str, params: dict = None):
    """Generic fetch from Medplum"""
    data = await medplum_get(resource, params)
    return data.get("entry", [])

RESOURCE_MAP = {
    "patient": get_patient_data,
    "observation": get_observation_data,
    "condition": lambda q: get_resource("Condition", {"code": "diabetes"})
}

async def query_medplum(query: str):
    for k, func in RESOURCE_MAP.items():
        if k in query.lower():
            return await func(query)
    return await get_resource("Observation")
