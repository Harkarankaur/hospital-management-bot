import os
import streamlit as st
import requests
import psycopg2
from dotenv import load_dotenv

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_experimental.sql import SQLDatabase, SQLDatabaseChain

# ============= CONFIG ==================
load_dotenv()

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

def medplum_query(query: str):
    if not (MEDPLUM_BASE_URL and MEDPLUM_TOKEN):
        raise RuntimeError("Medplum not configured.")
    headers = {"Authorization": f"Bearer {MEDPLUM_TOKEN}", "Accept": "application/fhir+json"}
    url = f"{MEDPLUM_BASE_URL.rstrip('/')}/{query.lstrip('/')}"
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()

# ============= ORCHESTRATOR ==================

def route_query(query: str, source: str):
    """Simple heuristic router: chooses SQL, Medplum, or RAG."""
    q = query.lower()
    if source == "auto":
        if any(k in q for k in ["patient", "condition", "fhir", "medplum", "observation"]):
            source = "medplum" if MEDPLUM_TOKEN else "rag"
        elif any(k in q for k in ["count", "average", "report", "total"]):
            source = "db"
        else:
            source = "rag"
    return source

# ============= STREAMLIT UI ==================

st.set_page_config(page_title="LangChain SQL + RAG + Medplum Bot", layout="wide")
st.title("🧠 Intelligent Text-to-SQL + RAG + Medplum Agent")

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
                if "diabetes" in query.lower():
                    endpoint = "Condition?code:text=diabetes&_include=Condition:subject"
                elif "patient" in query.lower():
                    endpoint = "Patient"
                else:
                    endpoint = "Observation"
                result = medplum_query(endpoint)
                st.subheader("Medplum FHIR Response")
                st.json(result)
            except Exception as e:
                st.error(f"Medplum error: {e}")

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
