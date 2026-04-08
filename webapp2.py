import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2d3348; border-radius: 12px;
        padding: 20px; text-align: center; margin-bottom: 10px;
    }
    .metric-card h2 { color: #7dd3fc; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #94a3b8; margin: 4px 0 0; font-size: 0.85rem; }
    .churn-high {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 2px solid #ef4444; border-radius: 16px; padding: 24px; text-align: center;
    }
    .churn-low {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 2px solid #22c55e; border-radius: 16px; padding: 24px; text-align: center;
    }
    .churn-label { font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700; }
    .churn-prob  { font-size: 3rem; font-weight: 700; font-family: 'Space Mono', monospace; }
    .section-header {
        font-family: 'Space Mono', monospace; font-size: 1rem; color: #7dd3fc;
        letter-spacing: 0.1em; text-transform: uppercase;
        border-bottom: 1px solid #2d3348; padding-bottom: 8px; margin-bottom: 16px;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white; border: none; border-radius: 10px; padding: 14px;
        font-family: 'Space Mono', monospace; font-size: 1rem; font-weight: 700;
        letter-spacing: 0.05em; cursor: pointer; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ── Data Source URL ───────────────────────────────────────────────────────────
DATA_URLS = [
    "https://raw.githubusercontent.com/varaprasad197/Customer-churn-predictor/main/tele_comm.csv",
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/srees1988/predict-churn-py/main/customer_churn_data.csv",
]

# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['TenureGroup'] = pd.cut(
        df['tenure'], bins=[0, 12, 24, 48, float('inf')],
        labels=['New', 'Regular', 'Established', 'Loyal']
    )
    df['IsFirstYear'] = (df['tenure'] <= 12).astype(int)
    df['IsLongTerm']  = (df['tenure'] >= 24).astype(int)
    df['AvgMonthlyCharge'] = df.apply(
        lambda x: x['TotalCharges'] / x['tenure'] if x['tenure'] > 0 else x['MonthlyCharges'], axis=1
    )
    df['CustomerLTV'] = df['TotalCharges'] + (df['MonthlyCharges'] * 6)
    add_svc = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['NumAdditionalServices'] = df[add_svc].apply(lambda x: (x == 'Yes').sum(), axis=1)
    df['HasSecurityBundle']  = ((df['OnlineSecurity'] == 'Yes') & (df['OnlineBackup'] == 'Yes')).astype(int)
    df['HasStreamingBundle'] = ((df['StreamingTV'] == 'Yes') & (df['StreamingMovies'] == 'Yes')).astype(int)
    df['InternetUser']       = (df['InternetService'] != 'No').astype(int)
    df['FiberOpticUser']     = (df['InternetService'] == 'Fiber optic').astype(int)
    df['ServicesPerMonth']   = df['NumAdditionalServices'] / (df['tenure'] + 1)
    df['IsMonthToMonth']    = (df['Contract'] == 'Month-to-month').astype(int)
    df['ContractType']      = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['ElectronicPayment'] = df['PaymentMethod'].str.contains('electronic check|automatic', case=False).astype(int)
    df['PaymentRisk']       = df['PaymentMethod'].map({
        'Electronic check': 3, 'Mailed check': 2,
        'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1
    })
    if df['PaperlessBilling'].dtype == 'object':
        df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    df['PaperlessHighRisk'] = ((df['PaperlessBilling'] == 1) & (df['PaymentMethod'] == 'Electronic check')).astype(int)
    if df['Partner'].dtype == 'object':
        df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    if df['Dependents'].dtype == 'object':
        df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
    df['HasFamily']  = ((df['Partner'] == 1) | (df['Dependents'] == 1)).astype(int)
    monthly_median = df['MonthlyCharges'].median()
    df['HighCostLowTenure'] = (
        (df['MonthlyCharges'] > monthly_median) & (df['tenure'] < 12)
    ).astype(int)
    tenure_max = df['tenure'].max() if df['tenure'].max() > 0 else 1
    df['EngagementScore'] = (
        df['NumAdditionalServices'] * 0.3 +
        df['ContractType'] * 0.4 +
        (df['tenure'] / tenure_max) * 0.3
    )
    return df


# ── Load & Train ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    raw_df = None
    used_url = None

    for url in DATA_URLS:
        try:
            raw_df = pd.read_csv(url)
            used_url = url
            break
        except Exception:
            continue

    if raw_df is None:
        return None, None, None, None, None, "❌ Could not load dataset from any source. Check your internet connection."

    df = raw_df.copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = engineer_features(df)

    X = pd.get_dummies(df.drop(columns=['Churn']), drop_first=True)
    y = df['Churn']
    encoded_columns = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=1000, random_state=42)
    lr_l1.fit(X_scaled, y)
    coef_mask = lr_l1.coef_[0] != 0
    selected_features = [c for c, m in zip(encoded_columns, coef_mask) if m]

    if len(selected_features) == 0:
        selected_features = encoded_columns

    sel_idx = [i for i, c in enumerate(encoded_columns) if c in selected_features]
    X_sel   = X_scaled[:, sel_idx]

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_sel, y)

    probs_all = model.predict_proba(X_sel)[:, 1]
    best_thresh, best_cost = 0.5, float('inf')
    for t in np.arange(0.2, 0.8, 0.01):
        preds_t = (probs_all >= t).astype(int)
        fp = ((preds_t == 1) & (y == 0)).sum()
        fn = ((preds_t == 0) & (y == 1)).sum()
        cost = fp * 100 + fn * 500
        if cost < best_cost:
            best_cost, best_thresh = cost, t

    info = {
        "source_url": used_url,
        "n_rows": len(raw_df),
        "n_features": len(selected_features),
        "threshold": round(best_thresh, 2),
        "auc": round(roc_auc_score(y, probs_all), 4),
        "churn_rate": round(y.mean() * 100, 1),
    }

    return model, scaler, selected_features, encoded_columns, best_thresh, info


# ── Run training ──────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading dataset from GitHub & training model..."):
    model, scaler, selected_features, encoded_columns, threshold, train_info = load_and_train()

model_ready = isinstance(train_info, dict)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Churn Predictor")
    st.markdown("Telecom Customer Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔮 Single Prediction", "📂 Batch Prediction", "📊 About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    if model_ready:
        st.success("✅ Model ready")
        st.caption(f"**Source:** `tele_comm.csv` (GitHub)")
        st.caption(f"**Rows:** {train_info['n_rows']:,}")
        st.caption(f"**AUC:** {train_info['auc']}")
        st.caption(f"**Threshold:** {train_info['threshold']}")
        st.caption(f"**Selected features:** {train_info['n_features']}")
        st.caption(f"**Dataset churn rate:** {train_info['churn_rate']}%")
    else:
        st.error(train_info)


# ── Predict helper ────────────────────────────────────────────────────────────
def predict_single(input_dict):
    df_in  = pd.DataFrame([input_dict])
    df_fe  = engineer_features(df_in)
    df_enc = pd.get_dummies(df_fe, drop_first=True)
    df_enc = df_enc.reindex(columns=encoded_columns, fill_value=0).astype(float)
    X_scaled = scaler.transform(df_enc)
    sel_idx  = [i for i, c in enumerate(encoded_columns) if c in selected_features]
    X_sel    = X_scaled[:, sel_idx]
    prob     = model.predict_proba(X_sel)[0, 1]
    pred     = int(prob >= threshold)
    return prob, pred


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Single Prediction":
    st.markdown("# 🔮 Single Customer Prediction")
    st.markdown("Fill in the customer details below and hit **Predict** to get an instant churn probability.")
    st.markdown("---")

    col_l, col_r = st.columns([2, 1], gap="large")

    with col_l:
        st.markdown('<div class="section-header">👤 Demographics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        gender     = c1.selectbox("Gender",         ["Male", "Female"])
        senior     = c2.selectbox("Senior Citizen",  ["No", "Yes"])
        partner    = c3.selectbox("Partner",         ["No", "Yes"])
        dependents = c1.selectbox("Dependents",      ["No", "Yes"])
        tenure     = c2.slider("Tenure (months)", 0, 72, 12)

        st.markdown('<div class="section-header">💳 Account Info</div>', unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        contract        = c4.selectbox("Contract",        ["Month-to-month", "One year", "Two year"])
        payment_method  = c5.selectbox("Payment Method",  [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        c6, c7 = st.columns(2)
        paperless        = c6.selectbox("Paperless Billing", ["No", "Yes"])
        monthly_charges  = c7.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total_charges    = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                           float(monthly_charges * tenure), step=10.0)

        st.markdown('<div class="section-header">🌐 Internet & Services</div>', unsafe_allow_html=True)
        c8, c9 = st.columns(2)
        internet_service  = c8.selectbox("Internet Service",   ["DSL", "Fiber optic", "No"])
        online_security   = c9.selectbox("Online Security",    ["No", "Yes", "No internet service"])
        c10, c11 = st.columns(2)
        online_backup     = c10.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
        device_protection = c11.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        c12, c13 = st.columns(2)
        tech_support      = c12.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
        streaming_tv      = c13.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
        streaming_movies  = st.selectbox("Streaming Movies",   ["No", "Yes", "No internet service"])

        st.markdown('<div class="section-header">📞 Phone Service</div>', unsafe_allow_html=True)
        c14, c15 = st.columns(2)
        phone_service  = c14.selectbox("Phone Service",  ["Yes", "No"])
        multiple_lines = c15.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    with col_r:
        st.markdown("### Run Prediction")
        st.markdown("Complete the form and click below.")

        if st.button("⚡ PREDICT CHURN"):
            if not model_ready:
                st.error("Model not available. Check sidebar for details.")
            else:
                input_data = {
                    "gender": gender,
                    "SeniorCitizen": 1 if senior == "Yes" else 0,
                    "Partner": partner, "Dependents": dependents,
                    "tenure": tenure, "PhoneService": phone_service,
                    "MultipleLines": multiple_lines, "InternetService": internet_service,
                    "OnlineSecurity": online_security, "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection, "TechSupport": tech_support,
                    "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
                    "Contract": contract, "PaperlessBilling": paperless,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
                }
                try:
                    prob, pred = predict_single(input_data)
                    pct = prob * 100

                    if pred == 1:
                        st.markdown(f"""
                        <div class="churn-high">
                            <div class="churn-label" style="color:#fca5a5;">⚠️ HIGH CHURN RISK</div>
                            <div class="churn-prob" style="color:#ef4444;">{pct:.1f}%</div>
                            <div style="color:#fca5a5;font-size:0.85rem;">probability of churning</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="churn-low">
                            <div class="churn-label" style="color:#86efac;">✅ LOW CHURN RISK</div>
                            <div class="churn-prob" style="color:#22c55e;">{pct:.1f}%</div>
                            <div style="color:#86efac;font-size:0.85rem;">probability of churning</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("---")

                    fig, ax = plt.subplots(figsize=(4, 0.6))
                    fig.patch.set_alpha(0)
                    ax.barh([""], [prob],       color="#ef4444", height=0.5)
                    ax.barh([""], [1 - prob],   left=[prob], color="#22c55e", height=0.5)
                    ax.set_xlim(0, 1); ax.axis('off')
                    st.pyplot(fig, use_container_width=True)

                    st.markdown("**Key inputs summary:**")
                    summary = pd.DataFrame({
                        "Feature": ["Contract", "Tenure", "Monthly $", "Internet", "Payment"],
                        "Value":   [contract, f"{tenure} mo", f"${monthly_charges:.2f}",
                                    internet_service, payment_method]
                    })
                    st.dataframe(summary, hide_index=True, use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Batch Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Batch Prediction":
    st.markdown("# 📂 Batch Prediction")
    st.markdown("Upload a CSV with the same columns as `tele_comm.csv` (the `Churn` column is optional).")
    st.markdown("---")

    use_train_data = st.checkbox("🗂️ Use the loaded `tele_comm.csv` dataset directly for batch scoring")

    if use_train_data and model_ready:
        @st.cache_data(show_spinner=False)
        def fetch_raw_data():
            for url in DATA_URLS:
                try:
                    return pd.read_csv(url)
                except Exception:
                    continue
            return None

        df_batch_raw = fetch_raw_data()
        if df_batch_raw is not None:
            st.success(f"✅ Loaded **{len(df_batch_raw):,}** customers from `tele_comm.csv`")
            uploaded_df = df_batch_raw
        else:
            st.error("Could not fetch dataset.")
            uploaded_df = None
    else:
        uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
        uploaded_df = pd.read_csv(uploaded) if uploaded else None

    if uploaded_df is not None:
        df_batch = uploaded_df.copy()

        if 'customerID' in df_batch.columns:
            ids = df_batch['customerID']
            df_batch = df_batch.drop(columns=['customerID'])
        else:
            ids = pd.Series(range(len(df_batch)), name='Index')

        ground_truth = None
        if 'Churn' in df_batch.columns:
            ground_truth = df_batch['Churn'].map({'Yes': 1, 'No': 0}) if df_batch['Churn'].dtype == 'object' else df_batch['Churn']
            df_batch = df_batch.drop(columns=['Churn'])

        df_batch['TotalCharges'] = pd.to_numeric(df_batch['TotalCharges'], errors='coerce')
        df_batch.loc[df_batch['tenure'] == 0, 'TotalCharges'] = 0
        df_batch['TotalCharges'].fillna(df_batch['MonthlyCharges'], inplace=True)

        if not model_ready:
            st.error("Model not available.")
        else:
            if st.button("⚡ RUN BATCH PREDICTION"):
                with st.spinner("Predicting..."):
                    try:
                        df_fe  = engineer_features(df_batch)
                        df_enc = pd.get_dummies(df_fe, drop_first=True)
                        df_enc = df_enc.reindex(columns=encoded_columns, fill_value=0).astype(float)
                        X_scaled = scaler.transform(df_enc)
                        sel_idx  = [i for i, c in enumerate(encoded_columns) if c in selected_features]
                        X_sel    = X_scaled[:, sel_idx]
                        probs    = model.predict_proba(X_sel)[:, 1]
                        preds    = (probs >= threshold).astype(int)

                        results = df_batch.copy()
                        results.insert(0, 'CustomerID',       ids.values)
                        results['ChurnProbability'] = (probs * 100).round(2)
                        results['ChurnPrediction']  = preds
                        results['RiskLevel'] = pd.cut(
                            probs, bins=[0, 0.3, 0.6, 1.0],
                            labels=['🟢 Low', '🟡 Medium', '🔴 High']
                        )

                        c1, c2, c3, c4 = st.columns(4)
                        c1.markdown(f'<div class="metric-card"><h2>{len(results):,}</h2><p>Total Customers</p></div>', unsafe_allow_html=True)
                        c2.markdown(f'<div class="metric-card"><h2>{preds.sum():,}</h2><p>Predicted Churners</p></div>', unsafe_allow_html=True)
                        c3.markdown(f'<div class="metric-card"><h2>{preds.mean()*100:.1f}%</h2><p>Churn Rate</p></div>', unsafe_allow_html=True)
                        c4.markdown(f'<div class="metric-card"><h2>{probs.mean()*100:.1f}%</h2><p>Avg Risk Score</p></div>', unsafe_allow_html=True)

                        if ground_truth is not None:
                            auc = roc_auc_score(ground_truth, probs)
                            st.info(f"📈 **AUC on this dataset:** {auc:.4f}  |  Ground truth churn rate: {ground_truth.mean()*100:.1f}%")

                        st.markdown("---")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            fig, ax = plt.subplots(figsize=(5, 3), facecolor='#0f1117')
                            ax.set_facecolor('#0f1117')
                            ax.hist(probs, bins=30, color='#7dd3fc', edgecolor='#1e2130')
                            ax.axvline(threshold, color='#ef4444', linestyle='--', label=f'Threshold ({threshold:.2f})')
                            ax.set_xlabel('Churn Probability', color='white')
                            ax.set_ylabel('Count', color='white')
                            ax.set_title('Probability Distribution', color='white', fontweight='bold')
                            ax.tick_params(colors='white')
                            ax.legend(facecolor='#1e2130', labelcolor='white')
                            for spine in ax.spines.values(): spine.set_color('#2d3348')
                            st.pyplot(fig, use_container_width=True)

                        with col_b:
                            risk_counts = results['RiskLevel'].value_counts()
                            fig, ax = plt.subplots(figsize=(5, 3), facecolor='#0f1117')
                            ax.set_facecolor('#0f1117')
                            colors_risk = ['#22c55e', '#eab308', '#ef4444']
                            ax.bar(risk_counts.index, risk_counts.values,
                                   color=colors_risk[:len(risk_counts)], edgecolor='#1e2130')
                            ax.set_title('Risk Level Breakdown', color='white', fontweight='bold')
                            ax.tick_params(colors='white')
                            for spine in ax.spines.values(): spine.set_color('#2d3348')
                            st.pyplot(fig, use_container_width=True)

                        st.markdown("### 🔴 Top 20 Highest Risk Customers")
                        top20 = results[['CustomerID', 'ChurnProbability', 'ChurnPrediction',
                                         'RiskLevel', 'Contract', 'tenure', 'MonthlyCharges']]\
                                .sort_values('ChurnProbability', ascending=False).head(20)
                        st.dataframe(top20, hide_index=True, use_container_width=True)

                        csv_out = results.to_csv(index=False)
                        st.download_button("⬇️ Download Full Results CSV", csv_out,
                                           "churn_predictions.csv", "text/csv")

                    except Exception as e:
                        st.error(f"Batch prediction error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — About  (enhanced with live metrics + redesigned layout)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 About":

    # ── Extra CSS for About page ──────────────────────────────────────────────
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&display=swap');

        .about-hero {
            background: linear-gradient(135deg, #0d1b2a 0%, #1a1040 50%, #0d1b2a 100%);
            border: 1px solid #2d3348;
            border-radius: 20px;
            padding: 40px 36px 32px;
            margin-bottom: 32px;
            position: relative;
            overflow: hidden;
        }
        .about-hero::before {
            content: '';
            position: absolute;
            top: -60px; right: -60px;
            width: 240px; height: 240px;
            background: radial-gradient(circle, rgba(124,58,237,0.18) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .about-hero::after {
            content: '';
            position: absolute;
            bottom: -40px; left: -40px;
            width: 180px; height: 180px;
            background: radial-gradient(circle, rgba(37,99,235,0.14) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .about-title {
            font-family: 'Syne', 'Space Mono', monospace;
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #7dd3fc, #a78bfa, #7dd3fc);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: shimmer 3s linear infinite;
            margin-bottom: 8px;
        }
        @keyframes shimmer {
            0% { background-position: 0% center; }
            100% { background-position: 200% center; }
        }
        .about-subtitle {
            color: #64748b;
            font-size: 0.95rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-family: 'Space Mono', monospace;
        }

        /* ── Stat cards ── */
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin: 24px 0;
        }
        .stat-card {
            background: linear-gradient(145deg, #131929, #1c2237);
            border: 1px solid #252d42;
            border-radius: 14px;
            padding: 20px 16px;
            text-align: center;
            position: relative;
            transition: transform 0.2s, border-color 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-3px);
            border-color: #3b4fd8;
        }
        .stat-card .stat-value {
            font-family: 'Syne', 'Space Mono', monospace;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 6px;
        }
        .stat-card .stat-label {
            font-size: 0.72rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #64748b;
        }
        .stat-card .stat-bar-bg {
            background: #1e2130;
            border-radius: 4px;
            height: 5px;
            margin-top: 10px;
            overflow: hidden;
        }
        .stat-card .stat-bar-fill {
            height: 5px;
            border-radius: 4px;
        }
        .color-auc     { color: #7dd3fc; }
        .color-acc     { color: #a78bfa; }
        .color-prec    { color: #34d399; }
        .color-recall  { color: #fb923c; }
        .color-f1      { color: #f472b6; }
        .color-thresh  { color: #fbbf24; }
        .fill-auc      { background: #7dd3fc; }
        .fill-acc      { background: #a78bfa; }
        .fill-prec     { background: #34d399; }
        .fill-recall   { background: #fb923c; }
        .fill-f1       { background: #f472b6; }
        .fill-thresh   { background: #fbbf24; }

        /* ── Section blocks ── */
        .info-block {
            background: linear-gradient(145deg, #131929, #1c2237);
            border: 1px solid #252d42;
            border-radius: 16px;
            padding: 24px 22px;
            margin-bottom: 18px;
        }
        .info-block-title {
            font-family: 'Space Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #7dd3fc;
            border-bottom: 1px solid #252d42;
            padding-bottom: 10px;
            margin-bottom: 16px;
        }
        .pipeline-step {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 12px;
        }
        .pipeline-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: #7dd3fc;
            margin-top: 6px;
            flex-shrink: 0;
        }
        .pipeline-text {
            font-size: 0.88rem;
            color: #cbd5e1;
            line-height: 1.55;
        }
        .pipeline-text strong { color: #e2e8f0; }

        /* ── Feature tags ── */
        .tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
        .feature-tag {
            background: #1e2a3a;
            border: 1px solid #334155;
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.76rem;
            color: #7dd3fc;
            font-family: 'Space Mono', monospace;
        }

        /* ── Source badge ── */
        .source-badge {
            background: #0f2a1a;
            border: 1px solid #166534;
            border-radius: 10px;
            padding: 10px 14px;
            font-family: 'Space Mono', monospace;
            font-size: 0.78rem;
            color: #4ade80;
            word-break: break-all;
            margin-top: 8px;
        }

        /* ── Metric explanation row ── */
        .metric-explain {
            display: flex;
            gap: 6px;
            align-items: flex-start;
            margin-bottom: 8px;
        }
        .metric-explain-dot {
            width: 10px; height: 10px;
            border-radius: 2px;
            flex-shrink: 0;
            margin-top: 4px;
        }
        .metric-explain-text { font-size: 0.84rem; color: #94a3b8; line-height: 1.5; }
        .metric-explain-text strong { color: #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero banner ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="about-hero">
        <div class="about-title">📡 Telecom Churn Predictor</div>
        <div class="about-subtitle">Model Intelligence Dashboard · Logistic Regression · Binary Classification</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Compute full metrics from training data ───────────────────────────────
    if model_ready:
        @st.cache_data(show_spinner=False)
        def compute_metrics():
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            for url in DATA_URLS:
                try:
                    raw = pd.read_csv(url)
                    break
                except Exception:
                    continue
            df_m = raw.copy()
            if 'customerID' in df_m.columns:
                df_m = df_m.drop(columns=['customerID'])
            df_m['TotalCharges'] = pd.to_numeric(df_m['TotalCharges'], errors='coerce')
            df_m.loc[df_m['tenure'] == 0, 'TotalCharges'] = 0
            df_m['TotalCharges'].fillna(df_m['MonthlyCharges'], inplace=True)
            df_m['Churn'] = df_m['Churn'].map({'Yes': 1, 'No': 0})
            df_m = engineer_features(df_m)
            X_m = pd.get_dummies(df_m.drop(columns=['Churn']), drop_first=True)
            y_m = df_m['Churn']
            X_m = X_m.reindex(columns=encoded_columns, fill_value=0).astype(float)
            X_sc = scaler.transform(X_m)
            sel_idx = [i for i, c in enumerate(encoded_columns) if c in selected_features]
            X_sel = X_sc[:, sel_idx]
            probs_m = model.predict_proba(X_sel)[:, 1]
            preds_m = (probs_m >= threshold).astype(int)
            return {
                "accuracy":  round(accuracy_score(y_m, preds_m) * 100, 1),
                "precision": round(precision_score(y_m, preds_m) * 100, 1),
                "recall":    round(recall_score(y_m, preds_m) * 100, 1),
                "f1":        round(f1_score(y_m, preds_m) * 100, 1),
                "auc":       round(roc_auc_score(y_m, probs_m) * 100, 1),
                "threshold": round(threshold * 100, 1),
            }

        with st.spinner("Computing metrics..."):
            metrics = compute_metrics()

        # ── Six stat cards ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-value color-auc">{metrics['auc']}%</div>
                <div class="stat-label">ROC-AUC Score</div>
                <div class="stat-bar-bg"><div class="stat-bar-fill fill-auc" style="width:{metrics['auc']}%"></div></div>
            </div>
            <div class="stat-card">
                <div class="stat-value color-acc">{metrics['accuracy']}%</div>
                <div class="stat-label">Accuracy</div>
                <div class="stat-bar-bg"><div class="stat-bar-fill fill-acc" style="width:{metrics['accuracy']}%"></div></div>
            </div>
            <div class="stat-card">
                <div class="stat-value color-prec">{metrics['precision']}%</div>
                <div class="stat-label">Precision</div>
                <div class="stat-bar-bg"><div class="stat-bar-fill fill-prec" style="width:{metrics['precision']}%"></div></div>
            </div>
            <div class="stat-card">
                <div class="stat-value color-recall">{metrics['recall']}%</div>
                <div class="stat-label">Recall</div>
                <div class="stat-bar-bg"><div class="stat-bar-fill fill-recall" style="width:{metrics['recall']}%"></div></div>
            </div>
            <div class="stat-card">
                <div class="stat-value color-f1">{metrics['f1']}%</div>
                <div class="stat-label">F1 Score</div>
                <div class="stat-bar-bg"><div class="stat-bar-fill fill-f1" style="width:{metrics['f1']}%"></div></div>
            </div>
            <div class="stat-card">
                <div class="stat-value color-thresh">{metrics['threshold']}%</div>
                <div class="stat-label">Decision Threshold</div>
                <div class="stat-bar-bg"><div class="stat-bar-fill fill-thresh" style="width:{metrics['threshold']}%"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metric explanations ───────────────────────────────────────────────
        st.markdown("""
        <div class="info-block">
            <div class="info-block-title">📐 What Each Metric Means for This Project</div>
            <div class="metric-explain">
                <div class="metric-explain-dot" style="background:#7dd3fc;"></div>
                <div class="metric-explain-text"><strong>ROC-AUC</strong> — Measures how well the model ranks churners above non-churners across all thresholds. &gt;0.80 is production-ready for telecom churn.</div>
            </div>
            <div class="metric-explain">
                <div class="metric-explain-dot" style="background:#a78bfa;"></div>
                <div class="metric-explain-text"><strong>Accuracy</strong> — Overall correct predictions. Less informative here due to class imbalance (~26% churn rate), so AUC + Recall take priority.</div>
            </div>
            <div class="metric-explain">
                <div class="metric-explain-dot" style="background:#34d399;"></div>
                <div class="metric-explain-text"><strong>Precision</strong> — Of customers flagged as churners, what % actually churn? High precision = fewer wasted retention offers.</div>
            </div>
            <div class="metric-explain">
                <div class="metric-explain-dot" style="background:#fb923c;"></div>
                <div class="metric-explain-text"><strong>Recall</strong> — Of all customers who actually churn, what % do we catch? A missed churner costs ~5× more than a false alarm (FN=$500 vs FP=$100).</div>
            </div>
            <div class="metric-explain">
                <div class="metric-explain-dot" style="background:#f472b6;"></div>
                <div class="metric-explain-text"><strong>F1 Score</strong> — Harmonic mean of Precision + Recall. The best single summary metric for imbalanced churn datasets.</div>
            </div>
            <div class="metric-explain">
                <div class="metric-explain-dot" style="background:#fbbf24;"></div>
                <div class="metric-explain-text"><strong>Decision Threshold</strong> — Business-optimized cut-off (minimizing FP×$100 + FN×$500). Lower than default 0.5 to maximise recall on high-cost false negatives.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("❌ Model could not be loaded — metrics unavailable.")

    # ── ML Pipeline + Data Source ─────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")

    with col1:
        pipeline_steps = [
            ("Dataset", "tele_comm.csv loaded live from GitHub (varaprasad197/Customer-churn-predictor)"),
            ("Fallback", "IBM Telco CSV (same schema) — no manual file management needed"),
            ("Feature Engineering", "20+ engineered features — tenure groups, LTV, engagement score, payment risk"),
            ("Feature Selection", "L1 (Lasso) Logistic Regression to drop zero-weight columns"),
            ("Model", "Logistic Regression with class_weight='balanced' to handle imbalance"),
            ("Threshold", "Optimized for minimum business cost (FP=$100, FN=$500)"),
            ("Deployment", "No .pkl files required — model trains at startup automatically"),
        ]
        steps_html = "".join([
            f'<div class="pipeline-step"><div class="pipeline-dot"></div>'
            f'<div class="pipeline-text"><strong>{k}</strong> — {v}</div></div>'
            for k, v in pipeline_steps
        ])
        st.markdown(f"""
        <div class="info-block">
            <div class="info-block-title">🧠 ML Pipeline</div>
            {steps_html}
        </div>
        """, unsafe_allow_html=True)

        tags = ["TenureGroup", "CustomerLTV", "EngagementScore", "PaymentRisk",
                "IsMonthToMonth", "FiberOpticUser", "HasSecurityBundle",
                "HasStreamingBundle", "NumAdditionalServices", "HighCostLowTenure"]
        tags_html = "".join([f'<span class="feature-tag">{t}</span>' for t in tags])
        st.markdown(f"""
        <div class="info-block">
            <div class="info-block-title">🔧 Engineered Features (sample)</div>
            <div class="tag-row">{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if model_ready:
            st.markdown(f"""
            <div class="info-block">
                <div class="info-block-title">📡 Data Source</div>
                <div style="color:#94a3b8;font-size:0.84rem;margin-bottom:6px;">✅ Loaded successfully from:</div>
                <div class="source-badge">{train_info['source_url']}</div>
                <div style="margin-top:18px;">
                    <table style="width:100%;border-collapse:collapse;">
                        <tr style="border-bottom:1px solid #252d42;">
                            <td style="padding:8px 4px;color:#64748b;font-size:0.82rem;font-family:'Space Mono',monospace;">TOTAL ROWS</td>
                            <td style="padding:8px 4px;text-align:right;color:#e2e8f0;font-family:'Space Mono',monospace;font-size:0.88rem;">{train_info['n_rows']:,}</td>
                        </tr>
                        <tr style="border-bottom:1px solid #252d42;">
                            <td style="padding:8px 4px;color:#64748b;font-size:0.82rem;font-family:'Space Mono',monospace;">SELECTED FEATURES</td>
                            <td style="padding:8px 4px;text-align:right;color:#e2e8f0;font-family:'Space Mono',monospace;font-size:0.88rem;">{train_info['n_features']}</td>
                        </tr>
                        <tr style="border-bottom:1px solid #252d42;">
                            <td style="padding:8px 4px;color:#64748b;font-size:0.82rem;font-family:'Space Mono',monospace;">TRAIN AUC</td>
                            <td style="padding:8px 4px;text-align:right;color:#7dd3fc;font-family:'Space Mono',monospace;font-size:0.88rem;">{train_info['auc']}</td>
                        </tr>
                        <tr style="border-bottom:1px solid #252d42;">
                            <td style="padding:8px 4px;color:#64748b;font-size:0.82rem;font-family:'Space Mono',monospace;">OPTIMAL THRESHOLD</td>
                            <td style="padding:8px 4px;text-align:right;color:#fbbf24;font-family:'Space Mono',monospace;font-size:0.88rem;">{train_info['threshold']}</td>
                        </tr>
                        <tr>
                            <td style="padding:8px 4px;color:#64748b;font-size:0.82rem;font-family:'Space Mono',monospace;">DATASET CHURN RATE</td>
                            <td style="padding:8px 4px;text-align:right;color:#fb923c;font-family:'Space Mono',monospace;font-size:0.88rem;">{train_info['churn_rate']}%</td>
                        </tr>
                    </table>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Dataset could not be loaded.")

        st.markdown("""
        <div class="info-block">
            <div class="info-block-title">💡 Design Decisions</div>
            <div class="metric-explain">
                <div class="metric-explain-dot" style="background:#fbbf24;margin-top:5px;"></div>
                <div class="metric-explain-text"><strong>Balanced class weights</strong> prevent the model from ignoring the minority (churn) class.</div>
            </div>
            <div class="metric-explain">
                <div class="metric-explain-dot" style="background:#fb923c;margin-top:5px;"></div>
                <div class="metric-explain-text"><strong>Business-cost threshold</strong> lowers the cut-off so high-value churners are not missed (FN penalty is 5× FP).</div>
            </div>
            <div class="metric-explain">
                <div class="metric-explain-dot" style="background:#34d399;margin-top:5px;"></div>
                <div class="metric-explain-text"><strong>L1 feature selection</strong> reduces noise and avoids overfitting on correlated telecom features.</div>
            </div>
            <div class="metric-explain">
                <div class="metric-explain-dot" style="background:#a78bfa;margin-top:5px;"></div>
                <div class="metric-explain-text"><strong>LTV & EngagementScore</strong> proxy long-term customer value signals beyond raw monthly charges.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Code section (unchanged) ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 Deployment Steps")
    st.code("""
# 1. Install dependencies
pip install streamlit scikit-learn pandas numpy matplotlib

# 2. Run locally — NO .pkl files needed, model trains automatically
streamlit run app.py

# 3. Deploy on Streamlit Cloud
#    - Push app.py + requirements.txt to GitHub
#    - Go to share.streamlit.io → Connect repo → Deploy
    """, language="bash")

    st.markdown("### 📦 requirements.txt")
    st.code("""streamlit
pandas
numpy
matplotlib
scikit-learn""", language="text")

    st.info("💡 **Tip**: The app fetches `tele_comm.csv` directly from GitHub at startup — no manual data management needed. Results are cached so the model only trains once per session.")
