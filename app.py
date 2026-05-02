import streamlit as st
import joblib
import re

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Fake Job Detector", layout="wide")

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# ==============================
# CLEAN FUNCTION
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    return text

# ==============================
# SESSION STATE INIT
# ==============================
if "title" not in st.session_state:
    st.session_state.title = ""
if "description" not in st.session_state:
    st.session_state.description = ""
if "requirements" not in st.session_state:
    st.session_state.requirements = ""
if "company" not in st.session_state:
    st.session_state.company = ""

# ==============================
# FUNCTION TO LOAD EXAMPLE
# ==============================
def load_example():
    st.session_state.title = "Work From Home"
    st.session_state.description = "Earn money quickly with no experience required"
    st.session_state.requirements = "No skills needed"
    st.session_state.company = "Unknown company"

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("📌 About Project")
st.sidebar.write("Fake Job Detection using ML + TF-IDF")

# ==============================
# MAIN TITLE
# ==============================
st.title("🕵️ Fake Job Posting Detection System")
st.info("Click 'Load Example' to auto-fill sample job data")

# ==============================
# BUTTON (IMPORTANT FIX)
# ==============================
st.button("📂 Load Example", on_click=load_example)

# ==============================
# INPUT FIELDS (AFTER BUTTON)
# ==============================
col1, col2 = st.columns(2)

with col1:
    title = st.text_input("Job Title", key="title")
    description = st.text_area("Job Description", key="description")

with col2:
    requirements = st.text_area("Requirements", key="requirements")
    company = st.text_area("Company Profile", key="company")

# ==============================
# PREDICTION
# ==============================
if st.button("🔍 Predict"):
    input_text = title + " " + description + " " + requirements + " " + company
    input_text = clean_text(input_text)

    vec = tfidf.transform([input_text])
    pred = model.predict(vec)

    try:
        prob = model.predict_proba(vec)[0]
        confidence = round(max(prob) * 100, 2)
    except:
        confidence = "N/A"

    st.markdown("## 🔎 Result")

    if pred[0] == 1:
        st.error("⚠️ FAKE JOB DETECTED")
    else:
        st.success("✅ REAL JOB")

    st.write(f"Confidence: {confidence}")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("🚀 Built using Streamlit")