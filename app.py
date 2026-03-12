import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, auc,
                             f1_score, accuracy_score, recall_score,
                             precision_score, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from scipy.stats import loguniform

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# ─── Title ─────────────────────────────────────────────────────────────────────
st.title("📡 Customer Churn Prediction")
st.markdown("**Binary Classification using Logistic Regression** — Telecom Dataset")
st.markdown("---")

# ─── Helper: engineer features ─────────────────────────────────────────────────
def engineer_features(df, median_charge, tenure_max):
    df = df.copy()
    df['IsFirstYear'] = (df['tenure'] <= 12).astype(int)
    df['AvgMonthlyCharge'] = df.apply(
        lambda x: x['TotalCharges'] / x['tenure'] if x['tenure'] > 0 else x['MonthlyCharges'],
        axis=1)
    add_svcs = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['NumAdditionalServices'] = df[add_svcs].apply(
        lambda x: (x == 'Yes').sum() if isinstance(x.iloc[0], str) else x.sum(), axis=1)
    df['FiberOpticUser'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['PaymentRisk'] = df['PaymentMethod'].map({
        'Electronic check': 3,
        'Mailed check': 2,
        'Bank transfer (automatic)': 1,
        'Credit card (automatic)': 1})
    df['HighCostLowTenure'] = (
        (df['MonthlyCharges'] > median_charge) & (df['tenure'] < 12)
    ).astype(int)
    df['HasFamily'] = ((df['Partner'] == 1) | (df['Dependents'] == 1)).astype(int)
    return df

# ─── Load default telecom dataset ───────────────────────────────────────────────
@st.cache_data
def load_default_dataset():
    """Load telecom dataset from GitHub"""
    url = "https://raw.githubusercontent.com/varaprasad197/Customer-churn-predictor/main/tele-comm.csv"
    df = pd.read_csv(url)
    return df

# ─── Cache: load + train ────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    df = load_default_dataset()

    # ── Clean ──────────────────────────────────────────────────────────────────
    df = df.drop(columns=['customerID'], errors='ignore')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

    # ── Split ──────────────────────────────────────────────────────────────────
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ── Feature engineering ────────────────────────────────────────────────────
    train_median_charge = X_train['MonthlyCharges'].median()
    train_tenure_max    = X_train['tenure'].max()
    X_train = engineer_features(X_train, train_median_charge, train_tenure_max)
    X_test  = engineer_features(X_test,  train_median_charge, train_tenure_max)

    # ── Encode ─────────────────────────────────────────────────────────────────
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test  = pd.get_dummies(X_test,  drop_first=True)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # ── Scale ──────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── SMOTE ──────────────────────────────────────────────────────────────────
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # ── L1 feature selection ───────────────────────────────────────────────────
    l1_selector = LogisticRegression(
        penalty='l1', solver='liblinear', C=0.1, max_iter=1000, random_state=42)
    l1_selector.fit(X_train_res, y_train_res)
    l1_mask           = l1_selector.coef_[0] != 0
    selected_features = X_train.columns[l1_mask].tolist()
    X_train_sel = X_train_res[:,  l1_mask]
    X_test_sel  = X_test_scaled[:, l1_mask]

    # ── Hyperparameter tuning ──────────────────────────────────────────────────
    param_dist = {
        'C':        loguniform(0.01, 10),
        'penalty':  ['l1', 'l2'],
        'solver':   ['liblinear'],
        'max_iter': [500, 1000]
    }
    random_search = RandomizedSearchCV(
        LogisticRegression(random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='recall',
        n_jobs=-1,
        random_state=42)
    random_search.fit(X_train_sel, y_train_res)
    best_model = random_search.best_estimator_
    best_model.fit(X_train_sel, y_train_res)

    # ── Predict ────────────────────────────────────────────────────────────────
    y_prob = best_model.predict_proba(X_test_sel)[:, 1]
    THRESHOLD = 0.5
    y_pred = (y_prob >= THRESHOLD).astype(int)

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_prob),
    }
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cm_vals = {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}

    return (df, X_train, X_test, y_train, y_test, y_pred, y_prob,
            metrics, cm_vals, selected_features, best_model,
            scaler, l1_mask, train_median_charge, train_tenure_max,
            X_train_sel, X_test_sel, THRESHOLD)

# ─── Load data and train model ──────────────────────────────────────────────────
with st.spinner('🔄 Loading dataset and training model...'):
    (df, X_train, X_test, y_train, y_test, y_pred, y_prob,
     metrics, cm_vals, selected_features, best_model,
     scaler, l1_mask, train_median_charge, train_tenure_max,
     X_train_sel, X_test_sel, THRESHOLD) = load_and_train()

st.success('✅ Model trained successfully!')

# ─── Sidebar info ───────────────────────────────────────────────────────────────
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "This app trains a **Logistic Regression** model on the "
    "telecom churn dataset with SMOTE balancing, L1 feature "
    "selection, and hyperparameter tuning.")

# ─── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview",
    "📈 Model Performance", 
    "🔍 Feature Insights",
    "🔮 Predict Churn"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{df['Churn'].mean():.1%}")
    col3.metric("Features", df.shape[1])
    col4.metric("Train/Test Split", "80/20")
    
    st.markdown("#### Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("#### Churn Distribution")
    churn_counts = df['Churn'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(['No Churn', 'Churn'], [churn_counts[0], churn_counts[1]], 
           color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Churn Distribution')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📈 Model Performance")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Recall", f"{metrics['recall']:.3f}")
    m3.metric("Precision", f"{metrics['precision']:.3f}")
    m4.metric("F1 Score", f"{metrics['f1']:.3f}")
    m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    
    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array([[cm_vals['TN'], cm_vals['FP']], [cm_vals['FN'], cm_vals['TP']]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'],
                cbar_kws={'label': 'Count'})
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    st.markdown("#### ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#3498db', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', label='Random classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    st.markdown("#### Precision-Recall Curve")
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_vals, precision_vals, color='#e74c3c', lw=2.5, label='Precision-Recall curve')
    ax.axhline(y=metrics['precision'], color='#95a5a6', linestyle='--', label=f'Precision = {metrics["precision"]:.3f}')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="best")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Feature Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔍 Feature Insights")
    st.markdown(f"**Selected Features (L1):** {len(selected_features)} out of {X_train.shape[1]}")
    
    coefs = best_model.coef_[0]
    feat_imp_df = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': coefs
    }).sort_values('Coefficient')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in feat_imp_df['Coefficient']]
    ax.barh(feat_imp_df['Feature'], feat_imp_df['Coefficient'], color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title('Feature Coefficients (L1 Selected)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Coefficient Value')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    st.markdown("""
    **Interpretation:**
    - 🔴 **Positive coefficients** → feature increases churn probability  
    - 🔵 **Negative coefficients** → feature decreases churn probability
    """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Predict Churn for a New Customer
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔮 Predict Churn for a New Customer")
    st.markdown("Fill in the customer details below:")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender         = st.selectbox("Gender", ["Female", "Male"])
            senior         = st.selectbox("Senior Citizen", [0, 1])
            partner        = st.selectbox("Has Partner", ["Yes", "No"])
            dependents     = st.selectbox("Has Dependents", ["Yes", "No"])
            tenure         = st.slider("Tenure (months)", 0, 72, 12)
            phone_service  = st.selectbox("Phone Service", ["Yes", "No"])

        with col2:
            multiple_lines = st.selectbox("Multiple Lines",
                                          ["No", "Yes", "No phone service"])
            internet       = st.selectbox("Internet Service",
                                          ["DSL", "Fiber optic", "No"])
            online_sec     = st.selectbox("Online Security",
                                          ["No", "Yes", "No internet service"])
            online_bkp     = st.selectbox("Online Backup",
                                          ["No", "Yes", "No internet service"])
            device_prot    = st.selectbox("Device Protection",
                                          ["No", "Yes", "No internet service"])
            tech_support   = st.selectbox("Tech Support",
                                          ["No", "Yes", "No internet service"])

        with col3:
            streaming_tv   = st.selectbox("Streaming TV",
                                          ["No", "Yes", "No internet service"])
            streaming_mv   = st.selectbox("Streaming Movies",
                                          ["No", "Yes", "No internet service"])
            contract       = st.selectbox("Contract",
                                          ["Month-to-month", "One year", "Two year"])
            paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment        = st.selectbox("Payment Method",
                                          ["Electronic check", "Mailed check",
                                           "Bank transfer (automatic)",
                                           "Credit card (automatic)"])
            monthly_charge = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
            total_charges  = st.number_input("Total Charges ($)", 0.0, 10000.0, 780.0)

        submitted = st.form_submit_button("🎯 Predict Churn", type="primary")

    if submitted:
        # Build raw row
        row = {
            'gender': 1 if gender == 'Female' else 0,
            'SeniorCitizen': senior,
            'Partner': 1 if partner == 'Yes' else 0,
            'Dependents': 1 if dependents == 'Yes' else 0,
            'tenure': tenure,
            'PhoneService': 1 if phone_service == 'Yes' else 0,
            'MultipleLines': multiple_lines,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': online_bkp,
            'DeviceProtection': device_prot,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_mv,
            'Contract': contract,
            'PaperlessBilling': 1 if paperless == 'Yes' else 0,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charge,
            'TotalCharges': total_charges,
        }
        input_df = pd.DataFrame([row])

        # Feature engineering
        input_eng = engineer_features(input_df, train_median_charge, train_tenure_max)

        # One-hot encode matching train columns
        X_train_raw = df.drop('Churn', axis=1)
        for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
            if X_train_raw[col].dtype == object:
                X_train_raw[col] = X_train_raw[col].map({'Yes': 1, 'No': 0})
        if 'gender' in X_train_raw.columns:
            X_train_raw['gender'] = X_train_raw['gender'].map({'Female': 1, 'Male': 0})
        X_train_raw = engineer_features(X_train_raw, train_median_charge, train_tenure_max)
        X_train_raw = pd.get_dummies(X_train_raw, drop_first=True)

        input_eng = pd.get_dummies(input_eng, drop_first=True)
        input_eng, _ = input_eng.align(X_train_raw, join='right', axis=1, fill_value=0)

        # Scale → select → predict
        input_scaled = scaler.transform(input_eng)
        input_sel    = input_scaled[:, l1_mask]
        prob         = best_model.predict_proba(input_sel)[0, 1]
        pred         = int(prob >= THRESHOLD)

        st.markdown("---")

        # ── Result banner ──────────────────────────────────────────────────────
        if pred == 1:
            st.error(f"⚠️  **HIGH CHURN RISK** — This customer is likely to leave")
        else:
            st.success(f"✅  **LOW CHURN RISK** — This customer is likely to stay")

        # ── Key metrics row ────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Probability",  f"{prob:.1%}")
        m2.metric("Stay Probability",   f"{1-prob:.1%}")
        risk_label = "High 🔴" if prob >= 0.7 else ("Medium 🟡" if prob >= THRESHOLD else "Low 🟢")
        m3.metric("Risk Level", risk_label)

        # ── Gauge bar ─────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 1.6))
        bar_color = '#e74c3c' if prob >= 0.7 else ('#f39c12' if prob >= THRESHOLD else '#2ecc71')
        ax.barh([''], [prob],       color=bar_color,   height=0.5, label='Churn probability')
        ax.barh([''], [1 - prob],   left=[prob], color='#ecf0f1', height=0.5)
        ax.axvline(THRESHOLD, color='#2c3e50', linestyle='--', linewidth=1.5,
                   label=f'Threshold ({THRESHOLD})')
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_title(f"Churn Probability: {prob:.1%}", fontweight='bold', fontsize=13)
        ax.legend(loc='lower right', fontsize=9)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        # ── Risk factor summary ────────────────────────────────────────────────
        st.markdown("#### 📋 Customer Risk Profile")
        flags = []
        if contract == "Month-to-month":
            flags.append(("🔴", "Month-to-month contract", "Highest churn risk contract type"))
        if internet == "Fiber optic":
            flags.append(("🔴", "Fiber optic internet", "Fiber users churn more often"))
        if tenure <= 12:
            flags.append(("🔴", "New customer (≤12 months)", "First year is highest-risk period"))
        if monthly_charge > train_median_charge:
            flags.append(("🟡", f"Above-median monthly charge (${monthly_charge:.0f})", "Higher charges increase churn risk"))
        if online_sec == "No" and internet != "No":
            flags.append(("🟡", "No Online Security", "Lack of add-ons linked to churn"))
        if tech_support == "No" and internet != "No":
            flags.append(("🟡", "No Tech Support", "Lack of support linked to churn"))
        if payment == "Electronic check":
            flags.append(("🟡", "Electronic check payment", "Associated with higher churn"))
        if partner == "Yes" or dependents == "Yes":
            flags.append(("🟢", "Has partner/dependents", "Family ties reduce churn"))
        if contract in ["One year", "Two year"]:
            flags.append(("🟢", f"{contract} contract", "Longer contracts mean lower churn"))
        if tenure > 24:
            flags.append(("🟢", f"Loyal customer ({tenure} months)", "Long tenure = lower churn risk"))

        if flags:
            for icon, factor, reason in flags:
                st.markdown(f"{icon} **{factor}** — *{reason}*")
        else:
            st.markdown("No notable risk factors detected.")
