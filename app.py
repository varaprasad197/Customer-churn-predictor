import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
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

# ─── Cache: load + train ────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # ── Clean ──────────────────────────────────────────────────────────────────
    df.drop(columns=['customerID'], inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        if df[col].dtype == object:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
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

# ─── Sidebar: data upload ───────────────────────────────────────────────────────
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload tele_comm.csv", type=["csv"],
    help="Upload the telecom churn CSV file")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "This app trains a Logistic Regression model on the "
    "telecom churn dataset with SMOTE balancing, L1 feature "
    "selection, and hyperparameter tuning.")

# ─── Main content ───────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Please upload **tele_comm.csv** in the sidebar to get started.")
    st.markdown("""
    ### What this app does
    1. **Loads & cleans** the telecom churn dataset  
    2. **Engineers 8 features** (IsFirstYear, AvgMonthlyCharge, etc.)  
    3. **Handles class imbalance** with SMOTE  
    4. **Selects features** via L1 regularisation  
    5. **Tunes hyperparameters** with RandomizedSearchCV  
    6. **Evaluates** the model with full metrics + visualisations  
    7. **Lets you predict** churn for a new customer  
    """)
    st.stop()

# ─── Train ─────────────────────────────────────────────────────────────────────
with st.spinner("Training model… this may take ~30 seconds"):
    (df, X_train, X_test, y_train, y_test, y_pred, y_prob,
     metrics, cm_vals, selected_features, best_model,
     scaler, l1_mask, train_median_charge, train_tenure_max,
     X_train_sel, X_test_sel, THRESHOLD) = load_and_train(uploaded_file)

st.success("✅ Model trained successfully!")

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dataset Overview", "📈 Model Performance", "🔍 Feature Insights", "🔮 Predict Churn"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Churn Rate", f"{df['Churn'].mean():.1%}")

    st.markdown("#### First 5 rows")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("#### Class Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    counts = df['Churn'].value_counts()
    ax.bar(['No Churn', 'Churn'], counts.values, color=['steelblue', 'tomato'])
    ax.set_ylabel("Count")
    ax.set_title("Churn Distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha='center', fontweight='bold')
    st.pyplot(fig, use_container_width=False)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance (Test Set)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{metrics['accuracy']:.4f}")
    c2.metric("Recall",    f"{metrics['recall']:.4f}")
    c3.metric("Precision", f"{metrics['precision']:.4f}")
    c4.metric("F1 Score",  f"{metrics['f1']:.4f}")
    c5.metric("ROC-AUC",   f"{metrics['roc_auc']:.4f}")

    st.markdown("---")

    # Confusion matrix + ROC + PR curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Evaluation — SMOTE + Logistic Regression',
                 fontsize=13, fontweight='bold')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    axes[0].set_title(f'Confusion Matrix\n(threshold={THRESHOLD})')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color='steelblue', linewidth=2,
                 label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random classifier')
    axes[1].set_title('ROC-AUC Curve')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # PR curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc   = auc(recall_vals, precision_vals)
    baseline = y_test.mean()
    axes[2].plot(recall_vals, precision_vals, color='darkorange', linewidth=2,
                 label=f'PR AUC = {pr_auc:.3f}')
    axes[2].fill_between(recall_vals, precision_vals, alpha=0.08, color='darkorange')
    axes[2].axhline(baseline, color='gray', linestyle='--', linewidth=1,
                    label=f'Baseline = {baseline:.2f}')
    axes[2].scatter([metrics['recall']], [metrics['precision']],
                    color='red', zorder=5, s=80,
                    label=f"Threshold={THRESHOLD}\nRecall={metrics['recall']:.2f}, "
                          f"Prec={metrics['precision']:.2f}")
    axes[2].set_title('Precision-Recall Curve')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Classification report
    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred,
                                   target_names=['No Churn', 'Churn'],
                                   output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Feature Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Feature Insights")

    st.markdown(f"**Features selected by L1 regularisation:** {len(selected_features)}")
    st.write(selected_features)

    # Feature importance from coefficients
    coefs = best_model.coef_[0]
    feat_imp = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': coefs
    }).sort_values('Coefficient', key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['tomato' if c > 0 else 'steelblue' for c in feat_imp['Coefficient']]
    ax.barh(feat_imp['Feature'], feat_imp['Coefficient'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('Feature Coefficients (Logistic Regression)\nPositive = higher churn risk',
                 fontsize=12)
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

        submitted = st.form_submit_button("Predict Churn", type="primary")

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
