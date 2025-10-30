##worked adding db file to this not working yet
import os
import streamlit as st
import requests
import psycopg2
from dotenv import load_dotenv
import httpx
# LangChain imports
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from fastapi import FastAPI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import pandas as pd
import asyncio
from db import create_tables, fetch_all_data, insert_manual_patient


load_dotenv()
app = FastAPI()

####end points
@app.get("/health")
def health():
    return {"status": "ok"}
@app.get("/dbtest")
def db_test():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS")
        )
        conn.close()
        return {"db": "ok"}
    except Exception as e:
        return {"db": "fail", "error": str(e)}

@app.get("/medplumtest")
def medplum_test():
    try:
        headers = {"Authorization": f"Bearer {os.getenv('MEDPLUM_TOKEN')}"}
        r = requests.get(f"{os.getenv('MEDPLUM_BASE_URL')}/Patient", headers=headers, timeout=15)
        r.raise_for_status()
        return {"medplum": "ok"}
    except Exception as e:
        return {"medplum": "fail", "error": str(e)}
@app.get("/seed-medplum")
def seed_medplum_endpoint():
    try:
        result = db.run_all()
        return {"status": "ok", "result": result}
    except Exception as e:
        return {"status": "fail", "error": str(e)}
# ============= CONFIG ==================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
MEDPLUM_BASE_URL = os.getenv("MEDPLUM_BASE_URL")
MEDPLUM_TOKEN = os.getenv("MEDPLUM_TOKEN")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

# ============= HELPERS ==================

def safe_sql(sql: str):
    """Ensure SQL is SELECT-only."""
    bad = ["insert", "update", "delete", "drop", "alter", "create", "truncate"]
    s = sql.strip().lower()
    return s.startswith("select") and not any(b in s for b in bad)

def execute_sql(sql: str):
    if not safe_sql(sql):
        raise RuntimeError("Unsafe SQL detected.")
    conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return [dict(zip(cols, r)) for r in rows]

# ============= LANGCHAIN CHAINS ==================

def get_sql_chain():
    """LangChain SQL agent for Text→SQL→Execution."""
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASS]):
        return None
    uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
    db = SQLDatabase.from_uri(uri)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    return SQLDatabaseChain.from_llm(llm, db, verbose=False)

def get_rag_chain():
    """RAG retriever for schema / FHIR documentation."""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    if not os.path.exists(CHROMA_PERSIST_DIR):
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)

    # If empty, seed minimal context
    if vectordb._collection.count() == 0:
        docs = {
            "Patient.txt": "FHIR Patient: demographic info (id, name, birthDate, gender).",
            "Condition.txt": "FHIR Condition: clinical problem linked to a patient.",
            "Observation.txt": "FHIR Observation: vital signs or test results.",
            "Schema.txt": "DB tables: patients(id, name, dob, gender), conditions(id, patient_id, diagnosis)."
        }
        for fname, txt in docs.items():
            path = f"./{fname}"
            with open(path, "w") as f:
                f.write(txt)
        vectordb = Chroma.from_texts(docs.values(), embedding=embeddings, persist_directory=CHROMA_PERSIST_DIR)
        vectordb.persist()

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
# ============= MEDPLUM ==================
async def medplum_get(resource: str, params: dict = None):
    """Fetch FHIR resource and convert to readable format."""
    if not MEDPLUM_BASE_URL or not MEDPLUM_TOKEN:
        raise RuntimeError("Medplum not configured")

    headers = {"Authorization": f"Bearer {MEDPLUM_TOKEN}"}
    endpoint = resource
    url = f"{MEDPLUM_BASE_URL.rstrip('/')}/{endpoint}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers, params=params or {"_count": 20})
        resp.raise_for_status()
        data = resp.json()
        return fhir_to_readable(resource, data)

def fhir_to_readable(resource_type: str, data: dict):
    """Convert FHIR bundle to readable dicts by resource type."""
    readable = []

    entries = data.get("entry", [])
    for entry in entries:
        res = entry.get("resource", {})

        # Patient
        if resource_type.lower() == "patient":
            readable.append({
                "id": res.get("id"),
                "name": " ".join([
                    *res.get("name", [{}])[0].get("given", []),
                    res.get("name", [{}])[0].get("family", "")
                ]).strip(),
                "gender": res.get("gender", ""),
                "birthDate": res.get("birthDate", ""),
                "phone": res.get("telecom", [{}])[0].get("value", ""),
                "address": res.get("address", [{}])[0].get("city", ""),
            })

        # Practitioner
        elif resource_type.lower() == "practitioner":
            readable.append({
                "id": res.get("id"),
                "name": " ".join([
                    *res.get("name", [{}])[0].get("given", []),
                    res.get("name", [{}])[0].get("family", "")
                ]).strip(),
                "qualification": ", ".join([
                    q.get("code", {}).get("text", "")
                    for q in res.get("qualification", [])
                ]) or "N/A",
                "mail": res.get("telecom", [{}])[0].get("value", ""),
            })

        # Condition
        elif resource_type.lower() == "condition":
            readable.append({
                "id": res.get("id"),
                "patient": res.get("subject", {}).get("reference", ""),
                "diagnosis": res.get("code", {}).get("text", ""),
                "onsetDateTime": res.get("onsetDateTime", ""),
            })

        # Observation
        elif resource_type.lower() == "observation":
            readable.append({
                "id": res.get("id"),
                "patient": res.get("subject", {}).get("reference", ""),
                "type": res.get("code", {}).get("text", ""),
                "value": res.get("valueQuantity", {}).get("value", ""),
                "unit": res.get("valueQuantity", {}).get("unit", ""),
                "date": res.get("effectiveDateTime", ""),
            })

        # Appointment
        elif resource_type.lower() == "appointment":
            readable.append({
                "id": res.get("id"),
                "status": res.get("status", ""),
                "patient": next(
                    (a.get("actor", {}).get("reference", "")
                    for a in res.get("participant", [])
                    if "Patient" in a.get("actor", {}).get("reference", "")), ""),
                "practitioner": next(
                    (a.get("actor", {}).get("reference", "")
                    for a in res.get("participant", [])
                    if "Practitioner" in a.get("actor", {}).get("reference", "")), ""),
                "start": res.get("start", ""),
                "end": res.get("end", ""),
            })

        # Encounter
        elif resource_type.lower() == "encounter":
            readable.append({
                "id": res.get("id"),
                "patient": res.get("subject", {}).get("reference", ""),
                "status": res.get("status", ""),
                "class": res.get("class", {}).get("code", ""),
                "start": res.get("period", {}).get("start", ""),
                "end": res.get("period", {}).get("end", ""),
            })

        # Default (fallback)
        else:
            readable.append(res)

    return readable

async def fetch_all_medplum_resources():
    """Fetch all major resource types"""
    resource_types = [
        "Patient",
        "Practitioner",
        "Observation",
        "Condition",
        "Appointment",
        "Encounter",
        "Medication",
        "AllergyIntolerance",
        "Procedure"
    ]

    all_resources = {}

    for resource_type in resource_types:
        try:
            response = await medplum_get(resource_type)
            all_resources[resource_type] = response
        except Exception as e:
            all_resources[resource_type] = {"error": str(e)}

    return all_resources

async def get_patients(query: str):
    return await medplum_get("Patient")

async def get_practitioners(query: str):
    return await medplum_get("Practitioner")

async def get_observations(query: str):
    return await medplum_get("Observation")

async def get_conditions(query: str):
    return await medplum_get("Condition")

async def query_medplum_async(query: str):
    """Router to handle Medplum resource fetching"""
    q = query.lower().strip()

    # Fetch all resources
    if "all resources" in q or "everything" in q or "all data" in q:
        return await fetch_all_medplum_resources()

    # Fetch specific resource
    resource_map = {
        "patient": "Patient",
        "practitioner": "Practitioner",
        "doctor": "Practitioner",
        "observation": "Observation",
        "vital": "Observation",
        "condition": "Condition",
        "disease": "Condition",
        "appointment": "Appointment",
        "encounter": "Encounter",
        "medication": "Medication",
        "allergy": "AllergyIntolerance",
        "procedure": "Procedure",
    }

    for key, resource in resource_map.items():
        if key in q:
            return await medplum_get(resource)

    # Default fallback
    return {"message": "Specify what data to fetch (e.g. 'list patients', 'get all resources', etc.)"}




# ============= ORCHESTRATOR ==================

def route_query(query: str, source: str):
    """Simple heuristic router: chooses SQL, Medplum, or RAG."""
    q = query.lower()
    if source == "auto":
        if any(k in q for k in ["patient", "doctor", "appointment", "hospital", "ward", "report", "record", "admission", "age", "gender", "disease", "condition", "count", "average", "total", "list", "show", "get"]):
            source = "db"
        elif any(k in q for k in ["fhir", "medplum", "observation", "practitioner", "vital"]):
            source = "medplum"
        else:
            source = "rag"
    return source

# ============= STREAMLIT UI ==================

st.set_page_config(page_title="Bot", layout="wide")
st.title("Medical Bot")

query = st.text_area("Enter your question:", height=120)
source = st.selectbox("Select source:", ["auto", "db", "medplum", "rag"])
run = st.button("Run Query")

st.sidebar.markdown("### Environment Status")
st.sidebar.json({
    "OpenAI": bool(OPENAI_API_KEY),
    "DB Connected": bool(DB_HOST and DB_NAME),
    "Medplum Configured": bool(MEDPLUM_BASE_URL and MEDPLUM_TOKEN),
})

if run and query:
    st.info("Processing query... please wait.")
    try:
        chosen = route_query(query, source)
        st.write(f"**Chosen Source:** `{chosen}`")

        if chosen == "db":
            sql_chain = get_sql_chain()
            if not sql_chain:
                st.error("DB not configured.")
            else:
                response = sql_chain.run(query)
                st.subheader("Database Result")
                st.write(response)
        elif chosen == "medplum":
            try:
                result = asyncio.run(query_medplum_async(query))
                st.subheader("Medplum FHIR Response")

                if isinstance(result, dict):
                    for resource_type, data in result.items():
                        st.markdown(f"### {resource_type}")
                        if isinstance(data, dict) and "error" in data:
                            st.error(f"Error: {data['error']}")
                        elif isinstance(data, list) and len(data) > 0:
                            df = pd.DataFrame(data)
                            st.dataframe(df)
                        else:
                            st.warning("No records found.")
                else:
                    df = pd.DataFrame(result)
                    st.dataframe(df)

            except Exception as e:
                st.error(f"Medplum query failed: {e}")


        elif chosen == "rag":
            rag_chain = get_rag_chain()
            answer = rag_chain({"query": query})
            st.subheader("RAG Answer")
            st.write(answer["result"])
            st.caption("Sources:")
            for doc in answer["source_documents"]:
                st.text(doc.metadata.get("source", "unknown"))

    except Exception as e:
        st.error(str(e))

st.markdown("---")
st.caption("🔒 Safe, single-file agent demo — built with LangChain + Streamlit + PostgreSQL + Medplum")


