from openai import OpenAI
import os
import psycopg2
import os
from fastapi import HTTPException
import requests
import os
import streamlit as st



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_sql(nl_query: str) -> str:
    prompt = f"""
    Convert the following natural language query into a safe SQL SELECT query.
    Do not use INSERT, UPDATE, DELETE, DROP, ALTER, or CREATE.
    Query: {nl_query}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a SQL expert assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )
    sql = response.choices[0].message.content.strip()
    return sql

def execute_sql(sql: str):
    forbidden = ["insert", "update", "delete", "drop", "alter", "create"]
    if not sql.lower().startswith("select") or any(word in sql.lower() for word in forbidden):
        raise HTTPException(status_code=400, detail="Unsafe SQL detected")

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS")
    )
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return [dict(zip(columns, row)) for row in rows]


def query_medplum(nl_query: str):
    base_url = os.getenv("MEDPLUM_BASE_URL")
    token = os.getenv("MEDPLUM_TOKEN")

    # Simple mapping: extend later
    if "diabetes" in nl_query.lower():
        resource = "Condition?code:text=diabetes&_include=Condition:subject"
    elif "patient" in nl_query.lower():
        resource = "Patient"
    else:
        resource = "Patient"

    url = f"{base_url}/{resource}"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

def orchestrate(query: str, source: str = "db"):
    if source == "db":
        sql = generate_sql(query)
        result = execute_sql(sql)
        return {"sql": sql, "data": result}
    elif source == "medplum":
        result = query_medplum(query)
        return {"data": result}
    else:
        raise ValueError("Unknown source")

st.title("Medplum Text-to-SQL Agentic Bot")

query = st.text_input("Enter your question:")

source = st.selectbox("Source", ["db", "medplum"])

if st.button("Submit") and query:
    try:
        result = orchestrate(query, source)
        st.subheader("Result")
        st.json(result)
    except Exception as e:
        st.error(str(e))
