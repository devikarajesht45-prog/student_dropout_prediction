"""
Student Success Predictor — Streamlit Dashboard
=================================================
Binary classification (Graduate vs Dropout) built on the UCI
"Predict students' dropout and academic success" dataset.

Expected repo layout (matches the project structure):

    STUDENT_DROPOUT_PREDICTION/
    ├── app.py                     <- this file (place at project root)
    ├── data/
    │   └── student_dropout_data.csv
    ├── models/
    │   ├── label_encoder.pkl
    │   ├── scaler.pkl
    │   └── Voting Classifier.pkl
    └── notebooks/
        ├── DataMapping.ipynb
        └── ModelBuilding.ipynb

ASSUMPTIONS (please verify against your ModelBuilding.ipynb — see the
"Setup notes" section at the bottom of the sidebar in the running app):
  1. `label_encoder.pkl` is a dict of {column_name: fitted LabelEncoder},
     one per categorical column — the common pattern when several
     columns are label-encoded in one notebook. If yours is a single
     LabelEncoder object instead, see the `load_encoders()` function
     below — it auto-detects both shapes.
  2. `scaler.pkl` is a single StandardScaler/MinMaxScaler fit on the
     numeric columns (everything NOT mapped to text in DataMapping.ipynb),
     in the column order defined in NUMERIC_COLS below.
  3. `Voting Classifier.pkl` was trained on the full ordered feature
     set FEATURE_ORDER (categorical columns already label-encoded,
     numeric columns already scaled), with the "Enrolled" class
     removed from the target beforehand — i.e. it's a binary
     classifier over {"Dropout", "Graduate"}.
  4. The model exposes `predict_proba` and `.classes_`.

If your training order or encoder shapes differ, adjust FEATURE_ORDER,
NUMERIC_COLS, CATEGORICAL_COLS and load_encoders() accordingly — those
are the only pieces tied to how ModelBuilding.ipynb was written.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .main-header {
        padding: 1.75rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5f8a 55%, #3b82c4 100%);
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
    .main-header p { margin: 0.35rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

    .section-card {
        background: #ffffff0d;
        border: 1px solid rgba(150,150,150,0.18);
        border-radius: 14px;
        padding: 1.1rem 1.3rem 0.6rem 1.3rem;
        margin-bottom: 1.1rem;
    }
    .section-title {
        font-weight: 700;
        font-size: 1.02rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-weight: 700;
        font-size: 1.05rem;
        background: linear-gradient(135deg, #2c5f8a 0%, #3b82c4 100%);
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #24507a 0%, #3272ab 100%);
        color: white;
    }

    .result-card {
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    .result-card h2 { margin: 0; font-size: 2rem; }
    .result-card p { margin: 0.3rem 0 0 0; opacity: 0.92; }

    .prob-row { display: flex; align-items: center; gap: 0.8rem; margin: 0.5rem 0; }
    .prob-label { width: 110px; font-weight: 600; font-size: 0.95rem; }
    .prob-bar-bg { flex: 1; background: rgba(150,150,150,0.2); border-radius: 8px; height: 22px; overflow: hidden; }
    .prob-bar-fill { height: 100%; border-radius: 8px; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; color: white; font-size: 0.8rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Success Predictor</h1>
    <p>Live binary classification (Graduate vs. Dropout) — UCI Student Dropout Dataset</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Feature schema (mirrors DataMapping.ipynb)
# --------------------------------------------------------------------------
FEATURE_ORDER = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance\t", "Previous qualification",
    "Previous qualification (grade)", "Nacionality",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", "Admission grade",
    "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder",
    "Age at enrollment", "International",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP",
]
 
CATEGORICAL_COLS = [
    "Marital status", "Application mode", "Course",
    "Daytime/evening attendance\t", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
    "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "International",
]

NUMERIC_COLS = [c for c in FEATURE_ORDER if c not in CATEGORICAL_COLS]
 
# Category options — copied verbatim from DataMapping.ipynb so the strings
# sent to the LabelEncoder(s) exactly match what they were fit on.
MARITAL_STATUS = ["Single", "Married", "Widower", "Divorced", "Facto Union", "Legally Separated"]

APPLICATION_MODE = [
    "1st phase-general contingent", "Ordinance No.612/93",
    "1st phase-special contingent(Azores Island)", "Holders of other higher courses",
    "Ordinance No.854-B/99", "International student(bachelor)",
    "1st phase-special contingent(Madeira Island)", "2nd phase-general contingent",
    "3rd phase-general contingent", "Ordinance No.533-A/99,item b2)(Different Plan)",
    "Ordinance No.533-A/99,item b3)(Other Institution)", "Over 23 years old",
    "Transfer", "Change of course", "Technological specialization diploma holders",
    "Change of institution/couse", "Short cycle diploma holders",
    "Change of institution/course (International)",
]

COURSE = [
    "Biofuel Production Technologies", "Animation and Multimedia Design",
    "Social service(evening)", "Agronomy", "Communication Design",
    "Veterinary Nursing", "Informatics Engineering", "Equinculture", "Management",
    "Social Service", "Tourism", "Nrsing", "Oral Hygiene",
    "Advertising and Marketing Management", "Journalism and Communication",
    "Basic Education", "Management(evening attendance)",
]

DAYTIME_EVENING = ["daytime", "evening"]

PREVIOUS_QUALIFICATION = [
    "Secondary education", "Higher education - bachelor's degree",
    "Higher education - degree", "Higher education - master's",
    "Higher education - doctorate", "Frequency of higher education",
    "12th year of schooling - not completed", "11th year of schooling - not completed",
    "Other - 11th year of schooling", "10th year of schooling",
    "10th year of schooling - not completed",
    "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
    "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
    "Technological specialization course", "Higher education - degree (1st cycle)",
    "Professional higher technical course", "Higher education - master (2nd cycle)",
]

NATIONALITY = [
    "Portuguese", "German", "Spanish", "Italian", "Dutch", "English", "Lithuanian",
    "Angolan", "Cape Verdean", "Guinean", "Mozambican", "Santomean", "Turkish",
    "Brazilian", "Romanian", "Moldova (Republic of)", "Mexican", "Ukrainian",
    "Russian", "Cuban", "Colombian",
]

PARENT_QUALIFICATION = [
    "Secondary Education - 12th Year of Schooling or Eq.",
    "Higher Education - Bachelor's Degree", "Higher Education - Degree",
    "Higher Education - Master's", "Higher Education - Doctorate",
    "Frequency of Higher Education", "12th Year of Schooling - Not Completed",
    "11th Year of Schooling - Not Completed", "7th Year (Old)",
    "Other - 11th Year of Schooling", "10th Year of Schooling",
    "General commerce course", "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    "Technical-professional course", "7th year of schooling",
    "2nd cycle of the general high school course",
    "9th Year of Schooling - Not Completed", "8th year of schooling", "Unknown",
    "Can't read or write", "Can read without having a 4th year of schooling",
    "Basic education 1st cycle (4th/5th year) or equiv.",
    "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    "Technological specialization course", "Higher education - degree (1st cycle)",
    "Specialized higher studies course", "Professional higher technical course",
    "Higher Education - Master (2nd cycle)", "Higher Education - Doctorate (3rd cycle)",
]

MOTHER_OCCUPATION = [
    "Student",
    "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
    "Specialists in Intellectual and Scientific Activities",
    "Intermediate Level Technicians and Professions", "Administrative staff",
    "Personal Services, Security and Safety Workers and Sellers",
    "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
    "Skilled Workers in Industry, Construction and Craftsmen",
    "Installation and Machine Operators and Assembly Workers", "Unskilled Workers",
    "Armed Forces Professions", "Other Situation", "(blank)", "Health professionals",
    "teachers", "Specialists in information and communication technologies (ICT)",
    "Intermediate level science and engineering technicians and professions",
    "Technicians and professionals, of intermediate level of health",
    "Intermediate level technicians from legal, social, sports, cultural and similar services",
    "Office workers, secretaries in general and data processing operators",
    "Data, accounting, statistical, financial services and registry-related operators",
    "Other administrative support staff", "personal service workers", "sellers",
    "Personal care workers and the like",
    "Skilled construction workers and the like, except electricians",
    "Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like",
    "Workers in food processing, woodworking, clothing and other industries and crafts",
    "cleaning workers", "Unskilled workers in agriculture, animal production, fisheries and forestry",
    "Unskilled workers in extractive industry, construction, manufacturing and transport",
    "Meal preparation assistants",
]

FATHER_OCCUPATION = MOTHER_OCCUPATION + [
    "Armed Forces Officers", "Armed Forces Sergeants", "Other Armed Forces personnel",
    "Directors of administrative and commercial services",
    "Hotel, catering, trade and other services directors",
    "Specialists in the physical sciences, mathematics, engineering and related techniques",
    "Specialists in finance, accounting, administrative organization, public and commercial relations",
    "Information and communication technology technicians",
    "Protection and security services personnel",
    "Market-oriented farmers and skilled agricultural and animal production workers",
    "Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence",
    "Skilled workers in metallurgy, metalworking and similar",
    "Skilled workers in electricity and electronics",
    "Fixed plant and machine operators", "assembly workers",
    "Vehicle drivers and mobile equipment operators",
    "Street vendors (except food) and street service providers",
]

YES_NO = ["Yes", "No"]
GENDER = ["Male", "Female"]

# --------------------------------------------------------------------------
# Load model artifacts
# --------------------------------------------------------------------------
MODELS_DIR = "models"

@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    def _load(path):
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    model_path = os.path.join(MODELS_DIR, "Voting Classifier.pkl")

    missing = [p for p in [scaler_path, encoder_path, model_path] if not os.path.exists(p)]
    if missing:
        return None, None, None, missing

    scaler = _load(scaler_path)
    encoders_raw = _load(encoder_path)
    model = _load(model_path)
    return scaler, encoders_raw, model, []


def get_encoder_for_column(encoders_raw, column):
    """Handle both a dict-of-encoders and a single shared LabelEncoder."""
    if isinstance(encoders_raw, dict):
        return encoders_raw.get(column)
    return encoders_raw  # single encoder shared across columns (fallback)


scaler, encoders_raw, model, missing_files = load_artifacts()

if missing_files:
    st.error(
        "Couldn't find the following model file(s) — make sure `app.py` sits "
        "at the project root, next to the `models/` folder:\n\n"
        + "\n".join(f"- `{m}`" for m in missing_files)
    )
    st.stop()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.write(
        "This dashboard loads your trained **Voting Classifier**, "
        "**scaler**, and **label encoder(s)** to predict, in real time, "
        "whether a student is likely to **Graduate** or **Drop out**."
    )
    st.markdown("---")
    st.markdown("### ⚙️ Setup notes")
    st.caption(
        "Categorical fields are encoded with `label_encoder.pkl`, numeric "
        "fields are scaled with `scaler.pkl`, then the row is passed to "
        "`Voting Classifier.pkl`. "
    )
    st.markdown("---")
    st.caption("Built with Streamlit 🎈")

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
st.markdown('<div class="section-card"><div class="section-title">👤 Personal & Demographic</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    gender = st.selectbox("Gender", GENDER)
with c2:
    marital_status = st.selectbox("Marital status", MARITAL_STATUS)
with c3:
    age = st.number_input("Age at enrollment", min_value=15, max_value=80, value=20, step=1)
with c4:
    nationality = st.selectbox("Nationality", NATIONALITY)
c5, c6 = st.columns(2)
with c5:
    displaced = st.selectbox("Displaced", YES_NO)
with c6:
    special_needs = st.selectbox("Educational special needs", YES_NO)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">🎓 Academic Background</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    application_mode = st.selectbox("Application mode", APPLICATION_MODE)
    course = st.selectbox("Course", COURSE)
    attendance = st.selectbox("Daytime/evening attendance", DAYTIME_EVENING)
with c2:
    application_order = st.number_input("Application order", min_value=0, max_value=9, value=1, step=1)
    previous_qualification = st.selectbox("Previous qualification", PREVIOUS_QUALIFICATION)
    previous_qualification_grade = st.number_input("Previous qualification (grade)", min_value=0.0, max_value=200.0, value=130.0, step=0.5)
admission_grade = st.number_input("Admission grade", min_value=0.0, max_value=200.0, value=130.0, step=0.5)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">👪 Family Background</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    mother_qualification = st.selectbox("Mother's qualification", PARENT_QUALIFICATION)
    mother_occupation = st.selectbox("Mother's occupation", MOTHER_OCCUPATION)
with c2:
    father_qualification = st.selectbox("Father's qualification", PARENT_QUALIFICATION)
    father_occupation = st.selectbox("Father's occupation", FATHER_OCCUPATION)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">💳 Financial Status</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    debtor = st.selectbox("Debtor", YES_NO)
with c2:
    tuition_up_to_date = st.selectbox("Tuition fees up to date", YES_NO)
with c3:
    scholarship = st.selectbox("Scholarship holder", YES_NO)
with c4:
    international = st.selectbox("International", YES_NO)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">📚 Curricular Performance</div>', unsafe_allow_html=True)
sem1_c, sem2_c = st.columns(2)
with sem1_c:
    st.markdown("**1st Semester**")
    cu1_credited = st.number_input("Credited (1st sem)", 0, 40, 0, key="cu1_cred")
    cu1_enrolled = st.number_input("Enrolled (1st sem)", 0, 40, 6, key="cu1_enr")
    cu1_evaluations = st.number_input("Evaluations (1st sem)", 0, 50, 8, key="cu1_eval")
    cu1_approved = st.number_input("Approved (1st sem)", 0, 40, 6, key="cu1_app")
    cu1_grade = st.number_input("Grade (1st sem)", 0.0, 20.0, 12.0, step=0.1, key="cu1_grade")
    cu1_without_eval = st.number_input("Without evaluations (1st sem)", 0, 40, 0, key="cu1_woeval")
with sem2_c:
    st.markdown("**2nd Semester**")
    cu2_credited = st.number_input("Credited (2nd sem)", 0, 40, 0, key="cu2_cred")
    cu2_enrolled = st.number_input("Enrolled (2nd sem)", 0, 40, 6, key="cu2_enr")
    cu2_evaluations = st.number_input("Evaluations (2nd sem)", 0, 50, 8, key="cu2_eval")
    cu2_approved = st.number_input("Approved (2nd sem)", 0, 40, 6, key="cu2_app")
    cu2_grade = st.number_input("Grade (2nd sem)", 0.0, 20.0, 12.0, step=0.1, key="cu2_grade")
    cu2_without_eval = st.number_input("Without evaluations (2nd sem)", 0, 40, 0, key="cu2_woeval")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">📈 Macroeconomic Indicators</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    unemployment_rate = st.number_input("Unemployment rate (%)", -5.0, 40.0, 10.8, step=0.1)
with c2:
    inflation_rate = st.number_input("Inflation rate (%)", -5.0, 15.0, 1.4, step=0.1)
with c3:
    gdp = st.number_input("GDP", -6.0, 6.0, 1.7, step=0.1)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")
predict_clicked = st.button("🔮 Predict Outcome", use_container_width=True)

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
if predict_clicked:
    raw_row = {
        "Marital status": marital_status,
        "Application mode": application_mode,
        "Course": course,
        "Daytime/evening attendance\t": attendance,
        "Previous qualification": previous_qualification,
        "Nacionality": nationality,
        "Mother's qualification": mother_qualification,
        "Father's qualification": father_qualification,
        "Mother's occupation": mother_occupation,
        "Father's occupation": father_occupation,
        "Displaced": displaced,
        "Educational special needs": special_needs,
        "Debtor": debtor,
        "Tuition fees up to date": tuition_up_to_date,
        "Gender": gender,
        "Scholarship holder": scholarship,
        "International": international,
        "Application order": application_order,
        "Previous qualification (grade)": previous_qualification_grade,
        "Admission grade": admission_grade,
        "Age at enrollment": age,
        "Curricular units 1st sem (credited)": cu1_credited,
        "Curricular units 1st sem (enrolled)": cu1_enrolled,
        "Curricular units 1st sem (evaluations)": cu1_evaluations,
        "Curricular units 1st sem (approved)": cu1_approved,
        "Curricular units 1st sem (grade)": cu1_grade,
        "Curricular units 1st sem (without evaluations)": cu1_without_eval,
        "Curricular units 2nd sem (credited)": cu2_credited,
        "Curricular units 2nd sem (enrolled)": cu2_enrolled,
        "Curricular units 2nd sem (evaluations)": cu2_evaluations,
        "Curricular units 2nd sem (approved)": cu2_approved,
        "Curricular units 2nd sem (grade)": cu2_grade,
        "Curricular units 2nd sem (without evaluations)": cu2_without_eval,
        "Unemployment rate": unemployment_rate,
        "Inflation rate": inflation_rate,
        "GDP": gdp,
    }

    try:
        df = pd.DataFrame([raw_row])[FEATURE_ORDER]

        # 1. Encode categorical columns
        for col in CATEGORICAL_COLS:
            enc = get_encoder_for_column(encoders_raw, col)
            if enc is None:
                st.error(f"No fitted LabelEncoder found for column '{col}' in label_encoder.pkl.")
                st.stop()
            df[col] = enc.transform(df[col].astype(str))

        # 2. Scale numeric columns
        df[FEATURE_ORDER] = scaler.transform(df[FEATURE_ORDER])
 
        # 3. Predict
        X = df[FEATURE_ORDER].values
        proba = model.predict_proba(X)[0]
        classes = list(getattr(model, "classes_", [0, 1]))

        # Map class labels (int-encoded or string) to friendly names
        target_encoder = None
        if isinstance(encoders_raw, dict):
            target_encoder = encoders_raw.get("Target") or encoders_raw.get("target")

        def friendly_label(c):
            if target_encoder is not None and isinstance(c, (int, np.integer)):
                try:
                    return str(target_encoder.inverse_transform([c])[0])
                except Exception:
                    pass
            label = str(c)
            if label in ("0", "1"):
                # last-resort guess if class labels are plain 0/1 with no encoder
                return "Dropout" if label == "0" else "Graduate"
            return label

        class_probs = {friendly_label(c): p for c, p in zip(classes, proba)}
        grad_prob = next((v for k, v in class_probs.items() if "grad" in k.lower()), None)
        drop_prob = next((v for k, v in class_probs.items() if "drop" in k.lower()), None)
        if grad_prob is None or drop_prob is None:
            # fallback: just take the two probabilities in class order
            vals = list(class_probs.values())
            grad_prob, drop_prob = (vals[0], vals[1]) if len(vals) > 1 else (vals[0], 1 - vals[0])

        predicted = "Graduate" if grad_prob >= drop_prob else "Dropout"
        confidence = max(grad_prob, drop_prob) * 100

        st.markdown("---")
        if predicted == "Graduate":
            gradient = "linear-gradient(135deg, #1d7a4a 0%, #2fa86a 100%)"
            emoji = "🎓"
            message = "This student is predicted to Graduate"
        else:
            gradient = "linear-gradient(135deg, #a33131 0%, #d6543f 100%)"
            emoji = "⚠️"
            message = "This student is predicted to Drop out"

        st.markdown(f"""
        <div class="result-card" style="background:{gradient};">
            <h2>{emoji} {message}</h2>
            <p>Model confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        res_c1, res_c2 = st.columns([1, 1])
        with res_c1:
            st.metric("🎓 Graduate probability", f"{grad_prob*100:.1f}%")
        with res_c2:
            st.metric("⚠️ Dropout probability", f"{drop_prob*100:.1f}%")

        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label">Graduate</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{grad_prob*100:.1f}%; background:#2fa86a;">{grad_prob*100:.1f}%</div>
            </div>
        </div>
        <div class="prob-row">
            <div class="prob-label">Dropout</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{drop_prob*100:.1f}%; background:#d6543f;">{drop_prob*100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 View encoded feature row sent to the model"):
            st.dataframe(df[FEATURE_ORDER], use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "This usually means the column order, encoder shape, or scaler "
            "columns don't match how `ModelBuilding.ipynb` was written. "
            "Check the FEATURE_ORDER / NUMERIC_COLS lists and the "
            "`get_encoder_for_column()` function at the top of app.py."
        )
