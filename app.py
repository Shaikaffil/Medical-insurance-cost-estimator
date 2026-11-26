"""
Medical Insurance Cost Estimator — Step-by-Step Interactive Wizard
Model: Linear Regression (R² = 0.9057)
"""

import base64
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Medical Insurance Cost Estimator & Risk Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "saved_models" / "linear_regression_best_model.pkl"
SCALER_PATH = BASE / "models" / "saved_models" / "scaler.pkl"
BG_PATH = BASE / "assets" / "health_bg.jpg"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


@st.cache_data
def get_bg_css():
    if BG_PATH.exists():
        b64 = base64.b64encode(BG_PATH.read_bytes()).decode("utf-8")
        return (
            f'background-image: linear-gradient(135deg, rgba(129, 140, 248, 0.45) 0%, '
            f'rgba(52, 211, 153, 0.35) 50%, rgba(251, 146, 60, 0.38) 100%), '
            f'url("data:image/jpeg;base64,{b64}") !important; '
            f'background-size: cover !important; background-position: center !important; background-attachment: fixed !important;'
        )
    return "background: linear-gradient(135deg, #C7D2FE 0%, #A7F3D0 50%, #FED7AA 100%) !important;"


model, scaler = load_model()
bg_style = get_bg_css()

# ─── Session State Defaults ───────────────────────────────────────────
defaults = {
    "step": 1,
    "age": 30,
    "sex": "Male",
    "children": 0,
    "bmi": 27.5,
    "smoker": "No — Non-smoker",
    "region": "Southeast",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Styling ──────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

html, body, .stApp {
    __BG_STYLE__
    font-family: 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif !important;
    color: #1E1B4B;
}
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
}
footer { visibility: hidden !important; }
.block-container, div[data-testid="stMainBlockContainer"] {
    padding: 0.75rem 2rem 1.2rem 2rem !important;
    max-width: 1050px !important;
    margin: 0 auto !important;
}

.hero-bar {
    background: linear-gradient(115deg, #1E1B4B 0%, #312E81 50%, #4F46E5 100%);
    border-radius: 18px;
    padding: 1.2rem 2.2rem;
    color: #FFFFFF;
    margin-bottom: 1rem;
    border-bottom: 4px solid #F97316;
    box-shadow: 0 6px 24px rgba(49, 46, 129, 0.22);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.6rem;
}
.hero-title { font-size: 2.2rem; font-weight: 800; margin: 0; line-height: 1.2; }
.hero-title span { color: #34D399; }
.hero-sub { font-size: 1.2rem; color: #E0E7FF; margin: 0.2rem 0 0 0; }
.h-badge-green {
    background: rgba(16, 185, 129, 0.22);
    border: 1px solid #34D399;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-size: 1.08rem;
    font-weight: 700;
    color: #A7F3D0;
}

.wizard-card {
    background: rgba(255, 255, 255, 0.96);
    border: 2px solid #C7D2FE;
    border-radius: 18px;
    padding: 2rem 2.4rem;
    box-shadow: 0 8px 28px rgba(30, 27, 75, 0.1);
    margin-bottom: 1rem;
    text-align: center;
}
.step-kicker {
    display: inline-block;
    background: #4F46E5;
    color: #FFFFFF;
    font-size: 0.95rem;
    font-weight: 800;
    padding: 0.35rem 1.1rem;
    border-radius: 999px;
    margin-bottom: 0.6rem;
}
.step-q {
    font-size: 1.75rem;
    font-weight: 800;
    color: #1E1B4B;
    margin: 0 0 0.4rem 0;
    line-height: 1.25;
}
.step-hint {
    font-size: 1.1rem;
    color: #4338CA;
    margin: 0;
}

.answered-bar {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}
.ans-chip {
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid #C7D2FE;
    color: #312E81;
    padding: 0.38rem 1.1rem;
    border-radius: 999px;
    font-size: 1.0rem;
    font-weight: 700;
}

label,
[data-testid="stWidgetLabel"] p,
.stRadio label,
.stSelectbox label,
.stNumberInput label,
.stSlider label {
    font-size: 1.18rem !important;
    font-weight: 700 !important;
    color: #1E1B4B !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
.stNumberInput input {
    font-size: 1.12rem !important;
    min-height: 60px !important;
    border-radius: 12px !important;
}
.stRadio div[role="radiogroup"] label p {
    font-size: 1.12rem !important;
    font-weight: 600 !important;
}
[data-testid="stThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
}

.pill-info {
    background: rgba(245, 247, 255, 0.96);
    border: 1px solid #C7D2FE;
    border-radius: 12px;
    padding: 0.85rem 1.3rem;
    font-size: 1.1rem;
    font-weight: 700;
    margin-top: 0.6rem;
    text-align: center;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1E1B4B 0%, #3730A3 55%, #4F46E5 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 1.18rem !important;
    letter-spacing: 0.3px !important;
    padding: 0.85rem 1.6rem !important;
    min-height: 62px !important;
    border-radius: 12px !important;
    width: 100%;
    box-shadow: 0 4px 16px rgba(49, 46, 129, 0.26) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #312E81 0%, #4F46E5 100%) !important;
    box-shadow: 0 8px 22px rgba(79, 70, 229, 0.36) !important;
    transform: translateY(-2px);
}
div.stButton > button[kind="secondary"] {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 1.5px solid #C7D2FE !important;
    color: #312E81 !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    padding: 0.82rem 1.5rem !important;
    min-height: 62px !important;
    border-radius: 12px !important;
    width: 100%;
    box-shadow: 0 2px 8px rgba(30, 27, 75, 0.07) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button[kind="secondary"]:hover {
    background: #E0E7FF !important;
    border-color: #6366F1 !important;
    color: #1E1B4B !important;
}

.quote-hero {
    background: linear-gradient(135deg, #1E1B4B 0%, #3730A3 60%, #4F46E5 100%);
    color: #FFFFFF;
    border-radius: 18px;
    padding: 2rem 2.4rem;
    border-top: 5px solid #34D399;
    box-shadow: 0 10px 30px rgba(49, 46, 129, 0.24);
    margin-bottom: 1rem;
    text-align: center;
}
.quote-kicker {
    font-size: 1.0rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #A5B4FC;
}
.quote-price {
    font-size: 3.5rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.1;
    margin: 0.3rem 0;
}
.quote-monthly { font-size: 1.25rem; font-weight: 700; color: #34D399; }

.summary-card {
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid #C7D2FE;
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1rem;
    text-align: center;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.9rem;
}
.s-item {
    background: #F5F7FF;
    border: 1px solid #E0E7FF;
    border-radius: 10px;
    padding: 0.85rem 1rem;
}
.s-lbl { font-size: 0.9rem; color: #4F46E5; font-weight: 700; }
.s-val { font-size: 1.1rem; color: #1E1B4B; font-weight: 800; }

.savings-banner {
    background: linear-gradient(90deg, #ECFDF5 0%, #D1FAE5 100%);
    border: 1px solid #6EE7B7;
    color: #065F46;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    font-size: 1.08rem;
    font-weight: 700;
    margin-bottom: 1rem;
    text-align: center;
}

.family-quote-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(245,247,255,0.96) 100%);
    border: 2px solid #C7D2FE;
    border-left: 6px solid #10B981;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 6px 20px rgba(30, 27, 75, 0.1);
    height: 100%;
    text-align: center;
}
.fq-text {
    font-size: 1.3rem;
    font-weight: 800;
    color: #1E1B4B;
    line-height: 1.4;
    margin: 0 0 0.5rem 0;
}
.fq-sub {
    font-size: 1.08rem;
    color: #4338CA;
    font-weight: 600;
    margin: 0 0 0.75rem 0;
}
.fq-pills {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem;
}
.fq-pill {
    background: #ECFDF5;
    border: 1px solid #6EE7B7;
    color: #065F46;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-size: 0.95rem;
    font-weight: 700;
}
div[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 2px solid #C7D2FE !important;
    box-shadow: 0 6px 18px rgba(30, 27, 75, 0.12) !important;
    max-height: 185px !important;
    object-fit: cover !important;
}
</style>
""".replace("__BG_STYLE__", bg_style),
    unsafe_allow_html=True,
)

# ─── Header ───────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-bar">
    <div>
        <h1 class="hero-title">🛡️ Medical Insurance <span>Cost Estimator & Risk Analyzer</span></h1>
        <p class="hero-sub">Answer one quick question at a time to get your personalized ML cost estimate & summary.</p>
    </div>
    <span class="h-badge-green">Linear Regression • R² = 0.9057</span>
</div>
""",
    unsafe_allow_html=True,
)

# ─── Happy Family + Motivating Health Quote ───────────────────────────
img_path = BASE / "assets" / "happy_family.jpg"
q_col1, q_col2 = st.columns([0.65, 1.35], gap="small")
with q_col1:
    if img_path.exists():
        st.image(str(img_path), width="stretch")
with q_col2:
    st.markdown(
        """
    <div class="family-quote-card">
        <div class="fq-text">“The greatest wealth is health — protecting your family today secures every tomorrow.”</div>
        <p class="fq-sub">Health insurance isn’t an expense; it’s peace of mind for the moments that matter most.</p>
        <div class="fq-pills">
            <span class="fq-pill">💚 Family Peace of Mind</span>
            <span class="fq-pill">🛡️ Financial Protection</span>
            <span class="fq-pill">⚡ 60-Second Estimate</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

step = st.session_state.step
total_steps = 6

# Progress bar
st.progress(min((step - 1) / total_steps, 1.0))

# Show already-answered chips during steps 2..6
if 1 < step <= total_steps:
    chips = []
    if step > 1:
        chips.append(f'<span class="ans-chip">🎂 Age: {st.session_state.age}</span>')
    if step > 2:
        chips.append(f'<span class="ans-chip">⚧ Gender: {st.session_state.sex}</span>')
    if step > 3:
        chips.append(f'<span class="ans-chip">👶 Dependents: {st.session_state.children}</span>')
    if step > 4:
        chips.append(f'<span class="ans-chip">📊 BMI: {st.session_state.bmi:.1f}</span>')
    if step > 5:
        s_short = "Yes" if "Yes" in st.session_state.smoker else "No"
        chips.append(f'<span class="ans-chip">🚬 Smoker: {s_short}</span>')
    st.markdown(f'<div class="answered-bar">{"".join(chips)}</div>', unsafe_allow_html=True)

# ─── Step-by-Step Single-Field Wizard ─────────────────────────────────
if step == 1:
    st.markdown(
        """
    <div class="wizard-card">
        <span class="step-kicker">Step 1 of 6</span>
        <div class="step-q">🎂 How old is the policyholder?</div>
        <p class="step-hint">Age is one of the strongest numeric predictors of medical insurance charges.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.session_state.age = st.slider("Select Age (Years)", 18, 64, st.session_state.age)
    if st.button("Next: Gender →", type="primary"):
        st.session_state.step = 2
        st.rerun()

elif step == 2:
    st.markdown(
        """
    <div class="wizard-card">
        <span class="step-kicker">Step 2 of 6</span>
        <div class="step-q">⚧ What is your gender?</div>
        <p class="step-hint">Select biological sex as recorded for insurance underwriting.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.session_state.sex = st.radio(
        "Select Gender",
        ["Male", "Female"],
        index=0 if st.session_state.sex == "Male" else 1,
        horizontal=True,
    )
    b1, b2 = st.columns(2)
    with b1:
        if st.button("← Back"):
            st.session_state.step = 1
            st.rerun()
    with b2:
        if st.button("Next: Dependents →", type="primary"):
            st.session_state.step = 3
            st.rerun()

elif step == 3:
    st.markdown(
        """
    <div class="wizard-card">
        <span class="step-kicker">Step 3 of 6</span>
        <div class="step-q">👶 How many children or dependents?</div>
        <p class="step-hint">Choose the number of dependents covered under the policy (0–5).</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.session_state.children = st.select_slider(
        "Number of Dependents", options=[0, 1, 2, 3, 4, 5], value=st.session_state.children
    )
    fam = 1 + st.session_state.children
    st.markdown(
        f'<div class="pill-info" style="color:#312E81;">{"👤" * fam} &nbsp; Total Covered Family Members: {fam}</div>',
        unsafe_allow_html=True,
    )
    b1, b2 = st.columns(2)
    with b1:
        if st.button("← Back"):
            st.session_state.step = 2
            st.rerun()
    with b2:
        if st.button("Next: BMI →", type="primary"):
            st.session_state.step = 4
            st.rerun()

elif step == 4:
    st.markdown(
        """
    <div class="wizard-card">
        <span class="step-kicker">Step 4 of 6</span>
        <div class="step-q">📊 What is your Body Mass Index (BMI)?</div>
        <p class="step-hint">BMI ≥ 30 combined with smoking strongly increases medical charges.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    bmi_val = st.number_input(
        "Enter BMI (kg/m²)", min_value=15.0, max_value=55.0, value=float(st.session_state.bmi), step=0.5, format="%.1f"
    )
    st.session_state.bmi = bmi_val
    if bmi_val < 18.5:
        b_cat, b_col = "Underweight", "#2563EB"
    elif bmi_val < 25.0:
        b_cat, b_col = "Normal Weight", "#059669"
    elif bmi_val < 30.0:
        b_cat, b_col = "Overweight", "#D97706"
    else:
        b_cat, b_col = "Obese (BMI ≥ 30)", "#DC2626"
    st.markdown(f'<div class="pill-info" style="color:{b_col};">BMI Category: {b_cat}</div>', unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("← Back"):
            st.session_state.step = 3
            st.rerun()
    with b2:
        if st.button("Next: Smoking Status →", type="primary"):
            st.session_state.step = 5
            st.rerun()

elif step == 5:
    st.markdown(
        """
    <div class="wizard-card">
        <span class="step-kicker">Step 5 of 6</span>
        <div class="step-q">🚬 Do you smoke?</div>
        <p class="step-hint">Smoking is the #1 cost driver in our dataset (~3.8x higher average charges).</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    opts = ["No — Non-smoker", "Yes — Active smoker"]
    st.session_state.smoker = st.radio(
        "Smoking Status", opts, index=0 if st.session_state.smoker == opts[0] else 1, horizontal=True
    )
    b1, b2 = st.columns(2)
    with b1:
        if st.button("← Back"):
            st.session_state.step = 4
            st.rerun()
    with b2:
        if st.button("Next: Region →", type="primary"):
            st.session_state.step = 6
            st.rerun()

elif step == 6:
    st.markdown(
        """
    <div class="wizard-card">
        <span class="step-kicker">Step 6 of 6</span>
        <div class="step-q">📍 Which region do you live in?</div>
        <p class="step-hint">Final step — click Estimate Cost to run the Linear Regression model.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    regs = ["Northeast", "Northwest", "Southeast", "Southwest"]
    st.session_state.region = st.selectbox("Residential Region", regs, index=regs.index(st.session_state.region))
    b1, b2 = st.columns(2)
    with b1:
        if st.button("← Back"):
            st.session_state.step = 5
            st.rerun()
    with b2:
        if st.button("Estimate My Medical Cost →", type="primary"):
            st.session_state.step = 7
            st.rerun()

# ─── Step 7: Prediction Result + Complete Summary ─────────────────────
else:
    age = st.session_state.age
    sex = st.session_state.sex
    children = st.session_state.children
    bmi = st.session_state.bmi
    smoker_val = "yes" if "Yes" in st.session_state.smoker else "no"
    region = st.session_state.region

    obese_smoker = int(bmi >= 30.0 and smoker_val == "yes")
    feature_dict = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "obese_smoker": obese_smoker,
        "sex_male": int(sex.lower() == "male"),
        "smoker_yes": int(smoker_val == "yes"),
        "region_northwest": int(region.lower() == "northwest"),
        "region_southeast": int(region.lower() == "southeast"),
        "region_southwest": int(region.lower() == "southwest"),
    }

    row = pd.DataFrame([feature_dict])[list(scaler.feature_names_in_)]
    predicted = max(0.0, float(model.predict(scaler.transform(row))[0]))
    monthly = predicted / 12.0

    ns_row = pd.DataFrame([dict(feature_dict, smoker_yes=0, obese_smoker=0)])[list(scaler.feature_names_in_)]
    ns_predicted = max(0.0, float(model.predict(scaler.transform(ns_row))[0]))
    savings = max(0.0, predicted - ns_predicted)

    if bmi < 18.5:
        bmi_cat = "Underweight"
    elif bmi < 25.0:
        bmi_cat = "Normal Weight"
    elif bmi < 30.0:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obese"

    if obese_smoker:
        risk_lbl = "High Risk (Obese + Smoker)"
    elif smoker_val == "yes" or bmi >= 30.0:
        risk_lbl = "Moderate Risk"
    else:
        risk_lbl = "Low Risk"

    st.markdown(
        f"""
    <div class="quote-hero">
        <div class="quote-kicker">Estimated Annual Medical Insurance Cost</div>
        <div class="quote-price">${predicted:,.2f}</div>
        <div class="quote-monthly">≈ ${monthly:,.2f} / month &nbsp;•&nbsp; {risk_lbl}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if smoker_val == "yes" and savings > 0:
        st.markdown(
            f"""
        <div class="savings-banner">
            💡 Non-Smoker Comparison: With the same age, BMI, and region as a non-smoker,
            your estimate drops to <strong>${ns_predicted:,.2f}/yr</strong> (saving <strong>${savings:,.2f}/yr</strong>).
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
    <div class="summary-card">
        <div style="font-size:1.3rem; font-weight:800; color:#1E1B4B; margin-bottom:0.9rem;">
            📋 Complete Estimate Summary
        </div>
        <div class="summary-grid">
            <div class="s-item"><div class="s-lbl">🎂 Age</div><div class="s-val">{age} years</div></div>
            <div class="s-item"><div class="s-lbl">⚧ Gender</div><div class="s-val">{sex}</div></div>
            <div class="s-item"><div class="s-lbl">👶 Dependents</div><div class="s-val">{children}</div></div>
            <div class="s-item"><div class="s-lbl">📊 BMI</div><div class="s-val">{bmi:.1f} ({bmi_cat})</div></div>
            <div class="s-item"><div class="s-lbl">🚬 Smoker</div><div class="s-val">{"Yes" if smoker_val == "yes" else "No"}</div></div>
            <div class="s-item"><div class="s-lbl">📍 Region</div><div class="s-val">{region}</div></div>
            <div class="s-item"><div class="s-lbl">🛡️ Risk Tier</div><div class="s-val">{risk_lbl}</div></div>
            <div class="s-item"><div class="s-lbl">📅 Monthly Est.</div><div class="s-val">${monthly:,.2f}</div></div>
            <div class="s-item"><div class="s-lbl">🤖 Model</div><div class="s-val">Linear Reg (R² 0.9057)</div></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    b1, b2 = st.columns(2)
    with b1:
        if st.button("← Edit Inputs", key="edit_inputs_btn"):
            st.session_state.step = 6
            st.rerun()
    with b2:
        if st.button("🔄 Start New Estimate", type="primary", key="restart_btn"):
            st.session_state.step = 1
            st.rerun()

