# app/seed_medplum.py
import os
import json
import requests
import psycopg2
from dotenv import load_dotenv
from typing import List

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

MEDPLUM_BASE = os.getenv("MEDPLUM_BASE_URL") or os.getenv("MEDPLUM_BASE")  # accept either name
MEDPLUM_TOKEN = os.getenv("MEDPLUM_TOKEN")

HEADERS = {"Authorization": f"Bearer {MEDPLUM_TOKEN}", "Content-Type": "application/fhir+json"}

# Utility: get psycopg2 connection
def get_conn():
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASS]):
        raise RuntimeError("DB credentials missing in env")
    return psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)

# Create tables if not exist (simple schema)
def create_tables():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id SERIAL PRIMARY KEY,
        fhir_id VARCHAR(128) UNIQUE,
        name TEXT,
        gender VARCHAR(20),
        birth_date DATE,
        raw_json JSONB
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS doctors (
        id SERIAL PRIMARY KEY,
        fhir_id VARCHAR(128) UNIQUE,
        name TEXT,
        qualification TEXT,
        raw_json JSONB
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conditions (
        id SERIAL PRIMARY KEY,
        fhir_id VARCHAR(128) UNIQUE,
        patient_fhir_id VARCHAR(128),
        diagnosis TEXT,
        clinical_status TEXT,
        raw_json JSONB
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        id SERIAL PRIMARY KEY,
        fhir_id VARCHAR(128) UNIQUE,
        patient_fhir_id VARCHAR(128),
        practitioner_fhir_id VARCHAR(128),
        status TEXT,
        start_time TIMESTAMP,
        raw_json JSONB
    );
    """)
    cur.execute("""
        ALTER TABLE patients
        ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();
    """)
    conn.commit()
    cur.close()
    conn.close()

# Simple fetcher with pagination support
def fetch_resource(resource: str, count: int = 50) -> List[dict]:
    if not MEDPLUM_BASE or not MEDPLUM_TOKEN:
        raise RuntimeError("Medplum not configured")
    base = MEDPLUM_BASE.rstrip('/')
    # If base already includes /fhir/R4, avoid doubling
    if not base.lower().endswith("/fhir/r4"):
        base = base + "/fhir/R4"
    url = f"{base}/{resource}?_count={count}"
    all_entries = []
    while url:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        entries = data.get("entry", [])
        all_entries.extend(entries)
        # find next link
        next_link = None
        for l in data.get("link", []):
            if l.get("relation") == "next":
                next_link = l.get("url")
                break
        url = next_link
    return all_entries

# Insert helpers (use ON CONFLICT DO NOTHING)
def seed_patients():
    rows = fetch_resource("Patient")
    conn = get_conn(); cur = conn.cursor()
    for e in rows:
        r = e.get("resource", {})
        fhir_id = r.get("id")
        # build name safely
        name = None
        try:
            if isinstance(r.get("name"), list) and len(r.get("name")) > 0:
                n0 = r["name"][0]
                name = n0.get("text") or " ".join(n0.get("given", []) + [n0.get("family","")]).strip()
        except Exception:
            name = None
        cur.execute("""
            INSERT INTO patients (fhir_id, name, gender, birth_date, raw_json)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (fhir_id) DO UPDATE
              SET name = EXCLUDED.name,
                  gender = EXCLUDED.gender,
                  birth_date = EXCLUDED.birth_date,
                  raw_json = EXCLUDED.raw_json
        """, (fhir_id, name, r.get("gender"), r.get("birthDate"), json.dumps(r)))
    conn.commit(); cur.close(); conn.close()
    return len(rows)

def seed_doctors():
    rows = fetch_resource("Practitioner")
    conn = get_conn(); cur = conn.cursor()
    for e in rows:
        r = e.get("resource", {})
        fhir_id = r.get("id")
        name = None
        try:
            if isinstance(r.get("name"), list) and len(r.get("name")) > 0:
                n0 = r["name"][0]
                name = n0.get("text") or " ".join(n0.get("given", []) + [n0.get("family","")]).strip()
        except Exception:
            name = None
        qualification = None
        try:
            qualification = r.get("qualification", [{}])[0].get("code", {}).get("text")
        except Exception:
            qualification = None
        cur.execute("""
            INSERT INTO doctors (fhir_id, name, qualification, raw_json)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (fhir_id) DO UPDATE
              SET name = EXCLUDED.name,
                  qualification = EXCLUDED.qualification,
                  raw_json = EXCLUDED.raw_json
        """, (fhir_id, name, qualification, json.dumps(r)))
    conn.commit(); cur.close(); conn.close()
    return len(rows)

def seed_conditions():
    rows = fetch_resource("Condition")
    conn = get_conn(); cur = conn.cursor()
    for e in rows:
        r = e.get("resource", {})
        fhir_id = r.get("id")
        patient_ref = r.get("subject", {}).get("reference", "")
        patient_fhir_id = patient_ref.replace("Patient/", "") if patient_ref else None
        diagnosis = r.get("code", {}).get("text")
        clinical_status = None
        try:
            clinical_status = r.get("clinicalStatus", {}).get("coding", [])[0].get("code")
        except Exception:
            clinical_status = None
        cur.execute("""
            INSERT INTO conditions (fhir_id, patient_fhir_id, diagnosis, clinical_status, raw_json)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (fhir_id) DO UPDATE
              SET patient_fhir_id = EXCLUDED.patient_fhir_id,
                  diagnosis = EXCLUDED.diagnosis,
                  clinical_status = EXCLUDED.clinical_status,
                  raw_json = EXCLUDED.raw_json
        """, (fhir_id, patient_fhir_id, diagnosis, clinical_status, json.dumps(r)))
    conn.commit(); cur.close(); conn.close()
    return len(rows)

def seed_appointments():
    rows = fetch_resource("Appointment")
    conn = get_conn(); cur = conn.cursor()
    for e in rows:
        r = e.get("resource", {})
        fhir_id = r.get("id")
        patient_ref = next((p.get("actor", {}).get("reference", "") for p in r.get("participant", []) if p.get("actor", {}).get("reference","").startswith("Patient/")), "")
        practitioner_ref = next((p.get("actor", {}).get("reference", "") for p in r.get("participant", []) if p.get("actor", {}).get("reference","").startswith("Practitioner/")), "")
        start = r.get("start") or r.get("startDateTime") or None
        status = r.get("status")
        cur.execute("""
            INSERT INTO appointments (fhir_id, patient_fhir_id, practitioner_fhir_id, status, start_time, raw_json)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (fhir_id) DO UPDATE
              SET patient_fhir_id = EXCLUDED.patient_fhir_id,
                  practitioner_fhir_id = EXCLUDED.practitioner_fhir_id,
                  status = EXCLUDED.status,
                  start_time = EXCLUDED.start_time,
                  raw_json = EXCLUDED.raw_json
        """, (fhir_id, patient_ref.replace("Patient/", "") if patient_ref else None,
              practitioner_ref.replace("Practitioner/", "") if practitioner_ref else None,
              status, start, json.dumps(r)))
    conn.commit(); cur.close(); conn.close()
    return len(rows)

def run_all():
    create_tables()
    result = {}
    try:
        result['patients'] = seed_patients()
    except Exception as e:
        result['patients_error'] = str(e)
    try:
        result['doctors'] = seed_doctors()
    except Exception as e:
        result['doctors_error'] = str(e)
    try:
        result['conditions'] = seed_conditions()
    except Exception as e:
        result['conditions_error'] = str(e)
    try:
        result['appointments'] = seed_appointments()
    except Exception as e:
        result['appointments_error'] = str(e)
    return result
# --- Manual data insert (example: patient) ---
def insert_manual_patient(name, gender, birth_date):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO patients (name, gender, birth_date)
        VALUES (%s, %s, %s)
        RETURNING id;
    """, (name, gender, birth_date))
    conn.commit()
    new_id = cur.fetchone()[0]
    cur.close()
    conn.close()
    return new_id

# --- Generic fetch function ---
def fetch_all_data(table_name):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table_name} ORDER BY id DESC;")
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return [dict(zip(columns, r)) for r in rows]

if __name__ == "__main__":
    res = run_all()
    print("Seeding complete:", res)
