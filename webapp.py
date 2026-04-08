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
# Primary: varaprasad197's repo; fallback: IBM raw CSV (identical schema)
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
    """Download tele_comm.csv from GitHub and train a LogisticRegression model."""
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

    # ── Preprocess ──
    df = raw_df.copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Feature engineering
    df = engineer_features(df)

    # One-hot encode
    X = pd.get_dummies(df.drop(columns=['Churn']), drop_first=True)
    y = df['Churn']

    encoded_columns = list(X.columns)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # L1-based feature selection via LogisticRegression
    lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=1000, random_state=42)
    lr_l1.fit(X_scaled, y)
    coef_mask = lr_l1.coef_[0] != 0
    selected_features = [c for c, m in zip(encoded_columns, coef_mask) if m]

    if len(selected_features) == 0:
        selected_features = encoded_columns  # fallback: use all

    sel_idx = [i for i, c in enumerate(encoded_columns) if c in selected_features]
    X_sel   = X_scaled[:, sel_idx]

    # Final model
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_sel, y)

    # Business-cost optimal threshold
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

    # Option to use the training dataset itself
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

                        # Summary metrics
                        c1, c2, c3, c4 = st.columns(4)
                        c1.markdown(f'<div class="metric-card"><h2>{len(results):,}</h2><p>Total Customers</p></div>', unsafe_allow_html=True)
                        c2.markdown(f'<div class="metric-card"><h2>{preds.sum():,}</h2><p>Predicted Churners</p></div>', unsafe_allow_html=True)
                        c3.markdown(f'<div class="metric-card"><h2>{preds.mean()*100:.1f}%</h2><p>Churn Rate</p></div>', unsafe_allow_html=True)
                        c4.markdown(f'<div class="metric-card"><h2>{probs.mean()*100:.1f}%</h2><p>Avg Risk Score</p></div>', unsafe_allow_html=True)

                        # Optional: show AUC if ground truth available
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
# PAGE 3 — About
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 About":
    st.markdown("# 📊 About This Project")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 ML Pipeline")
        st.markdown("""
- **Dataset**: `tele_comm.csv` loaded live from GitHub (varaprasad197/Customer-churn-predictor)
- **Fallback sources**: IBM Telco CSV (same schema) — no manual file management needed
- **Feature Engineering**: 20+ engineered features — tenure groups, LTV, engagement score, payment risk
- **Feature Selection**: L1 (Lasso) Logistic Regression
- **Model**: Logistic Regression with `class_weight='balanced'`
- **Threshold**: Optimized for minimum business cost (FP=$100, FN=$500)
- **No `.pkl` files required** — model trains at startup automatically
        """)

    with col2:
        st.markdown("### 📡 Data Source")
        if model_ready:
            st.success(f"✅ Data loaded from:\n\n`{train_info['source_url']}`")
            st.markdown(f"""
| Metric | Value |
|---|---|
| Total rows | {train_info['n_rows']:,} |
| Selected features | {train_info['n_features']} |
| Train AUC | {train_info['auc']} |
| Optimal threshold | {train_info['threshold']} |
| Dataset churn rate | {train_info['churn_rate']}% |
""")
        else:
            st.error("Dataset could not be loaded.")

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
