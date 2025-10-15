import streamlit as st
import requests
import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MEDPLUM_TOKEN = os.getenv("MEDPLUM_TOKEN")
MEDPLUM_BASE_URL = os.getenv("MEDPLUM_BASE_URL")

st.title("Simple Medplum Text-to-SQL Bot")

# User input
user_question = st.text_area("Enter your question:")

def medplum_search(query: str):
    """
    Simple Medplum query simulation:
    Convert question to FHIR query and call Medplum API
    """
    # Hardcoded mapping for now
    if "diabetes" in query.lower():
        fhir_path = "Condition?code:text=diabetes&_include=Condition:subject"
    elif "patient" in query.lower():
        fhir_path = "Patient"
    else:
        fhir_path = "Patient"

    url = f"{MEDPLUM_BASE_URL}/{fhir_path}"
    headers = {"Authorization": f"Bearer {MEDPLUM_TOKEN}"}

    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

# Button to submit question
if st.button("Submit"):
    if user_question.strip() == "":
        st.warning("Please enter a question first!")
    else:
        st.subheader("Medplum Response")
        result = medplum_search(user_question)
        st.json(result)

        # Optional: Generate SQL using OpenAI
        st.subheader("Optional SQL Generation (OpenAI)")
        try:
            prompt = f"Convert this request into a safe SQL SELECT query: {user_question}"
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a SQL assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0
            )
            sql_result = response["choices"][0]["message"]["content"].strip()
            st.code(sql_result)
        except Exception as e:
            st.error(f"OpenAI error: {str(e)}")
