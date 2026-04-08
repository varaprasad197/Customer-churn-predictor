import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Metrics
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, auc, ConfusionMatrixDisplay
)
from scipy.stats import loguniform, randint, uniform

# SHAP
import shap

from collections import Counter

plt.rcParams['figure.dpi'] = 110
plt.rcParams['font.size'] = 11
sns.set_style('whitegrid')

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .section-header {
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
        margin-bottom: 20px;
        font-size: 24px;
        font-weight: bold;
        color: #1f1f1f;
    }

    .prediction-high-risk {
        background-color: #ffe6e6;
        border-left: 5px solid #ff4444;
        padding: 15px;
        border-radius: 5px;
    }

    .prediction-low-risk {
        background-color: #e6ffe6;
        border-left: 5px solid #44ff44;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# YOUR ACTUAL METRICS FROM NOTEBOOK
# ============================================================================
ACTUAL_METRICS = {
    'accuracy': 0.7388,
    'recall': 0.8102,
    'precision': 0.5050,
    'f1': 0.6222,
    'roc_auc': 0.8389,
    'tp': 303, 'fp': 297, 'tn': 738, 'fn': 71,
    'train_accuracy': 0.7691,
    'test_accuracy': 0.7388,
}

TEST_SIZE = 1409
TOTAL_CUSTOMERS = 7043
CHURN_RATE = 0.2654

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_data():
    github_csv_url = "https://raw.githubusercontent.com/varaprasad197/Customer-churn-predictor/main/tele_comm.csv"
    try:
        return pd.read_csv(github_csv_url)
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# ============================================================================
# FULL PIPELINE FROM NOTEBOOK (cached)
# ============================================================================
@st.cache_resource
def run_full_pipeline(df):
    # ── Cell 6: Basic Cleaning ──────────────────────────────────────────
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
    df.drop(columns=['customerID'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # ── Cell 13: Feature Engineering ───────────────────────────────────
    df['TenureGroup'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, float('inf')],
        labels=['New', 'Regular', 'Established', 'Loyal']
    )
    df['IsFirstYear'] = (df['tenure'] <= 12).astype(int)
    df['IsLongTerm']  = (df['tenure'] >= 24).astype(int)

    df['AvgMonthlyCharge'] = df.apply(
        lambda x: x['TotalCharges'] / x['tenure'] if x['tenure'] > 0 else x['MonthlyCharges'],
        axis=1
    )
    df['CustomerLTV'] = df['TotalCharges'] + (df['MonthlyCharges'] * 6)

    additional_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['NumAdditionalServices'] = df[additional_services].apply(
        lambda x: (x == 'Yes').sum(), axis=1
    )
    df['HasSecurityBundle']  = ((df['OnlineSecurity'] == 'Yes') & (df['OnlineBackup'] == 'Yes')).astype(int)
    df['HasStreamingBundle'] = ((df['StreamingTV'] == 'Yes') & (df['StreamingMovies'] == 'Yes')).astype(int)
    df['InternetUser']       = (df['InternetService'] != 'No').astype(int)
    df['FiberOpticUser']     = (df['InternetService'] == 'Fiber optic').astype(int)
    df['ServicesPerMonth']   = df['NumAdditionalServices'] / (df['tenure'] + 1)

    df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['ContractType']   = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['ElectronicPayment'] = df['PaymentMethod'].str.contains('electronic check|automatic', case=False).astype(int)
    df['PaymentRisk'] = df['PaymentMethod'].map({
        'Electronic check': 3, 'Mailed check': 2,
        'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1
    })
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0}) if df['PaperlessBilling'].dtype == 'object' else df['PaperlessBilling']
    df['PaperlessHighRisk'] = ((df['PaperlessBilling'] == 1) & (df['PaymentMethod'] == 'Electronic check')).astype(int)

    df['Partner']    = df['Partner'].map({'Yes': 1, 'No': 0}) if df['Partner'].dtype == 'object' else df['Partner']
    df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0}) if df['Dependents'].dtype == 'object' else df['Dependents']
    df['HasFamily']  = ((df['Partner'] == 1) | (df['Dependents'] == 1)).astype(int)

    df['HighCostLowTenure'] = (
        (df['MonthlyCharges'] > df['MonthlyCharges'].median()) & (df['tenure'] < 12)
    ).astype(int)

    df['EngagementScore'] = (
        df['NumAdditionalServices'] * 0.3 +
        df['ContractType'] * 0.4 +
        (df['tenure'] / df['tenure'].max()) * 0.3
    )

    # ── Cell 15: Train-Test Split ───────────────────────────────────────
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Cell 17: Encoding ───────────────────────────────────────────────
    X_train_enc = pd.get_dummies(X_train, drop_first=True)
    X_test_enc  = pd.get_dummies(X_test,  drop_first=True)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join='outer', axis=1, fill_value=0)
    X_train_enc = X_train_enc.astype(float)
    X_test_enc  = X_test_enc.astype(float)

    # ── Cell 18: Scaling ────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled  = scaler.transform(X_test_enc)
    feature_names  = X_train_enc.columns.tolist()

    # ── Cell 20: Feature Selection (L1) ────────────────────────────────
    l1_selector = LogisticRegression(penalty='l1', solver='liblinear', C=0.1,
                                     max_iter=1000, random_state=42)
    l1_selector.fit(X_train_scaled, y_train)

    l1_mask           = l1_selector.coef_[0] != 0
    selected_features = np.array(feature_names)[l1_mask]

    X_train_sel = X_train_scaled[:, l1_mask]
    X_test_sel  = X_test_scaled[:,  l1_mask]

    # ── Cell 22: SMOTE ──────────────────────────────────────────────────
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)

    # ── Cells 27-30: Hyperparameter Tuning ─────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr_param_dist = {
        'C'       : loguniform(0.01, 10),
        'penalty' : ['l1', 'l2'],
        'solver'  : ['liblinear'],
        'max_iter': [500, 1000]
    }
    lr_search = RandomizedSearchCV(
        LogisticRegression(random_state=42), lr_param_dist,
        n_iter=20, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42
    )
    lr_search.fit(X_train_res, y_train_res)

    dt_param_dist = {
        'max_depth'        : randint(3, 20),
        'min_samples_split': randint(2, 50),
        'min_samples_leaf' : randint(1, 30),
        'criterion'        : ['gini', 'entropy']
    }
    dt_search = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=42), dt_param_dist,
        n_iter=30, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42
    )
    dt_search.fit(X_train_res, y_train_res)

    rf_param_dist = {
        'n_estimators'    : randint(100, 500),
        'max_depth'       : [None, 10, 20, 30],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf' : randint(1, 15),
        'max_features'     : ['sqrt', 'log2']
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1), rf_param_dist,
        n_iter=30, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42
    )
    rf_search.fit(X_train_res, y_train_res)

    xgb_param_dist = {
        'n_estimators'    : randint(100, 500),
        'max_depth'       : randint(3, 10),
        'learning_rate'   : uniform(0.01, 0.3),
        'subsample'       : uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma'           : uniform(0, 0.5),
        'reg_alpha'       : uniform(0, 1),
        'reg_lambda'      : uniform(1, 2)
    }
    xgb_search = RandomizedSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
        xgb_param_dist, n_iter=40, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42
    )
    xgb_search.fit(X_train_res, y_train_res)

    # ── Cell 32: Model Comparison ───────────────────────────────────────
    tuned_models = {
        'Logistic Regression': lr_search.best_estimator_,
        'Decision Tree'      : dt_search.best_estimator_,
        'Random Forest'      : rf_search.best_estimator_,
        'XGBoost'            : xgb_search.best_estimator_
    }

    results = []
    for name, model in tuned_models.items():
        y_prob = model.predict_proba(X_test_sel)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        results.append({
            'Model'    : name,
            'Accuracy' : accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall'   : recall_score(y_test, y_pred),
            'F1'       : f1_score(y_test, y_pred),
            'ROC-AUC'  : roc_auc_score(y_test, y_prob)
        })

    results_df      = pd.DataFrame(results).set_index('Model').round(4)
    best_model_name = results_df['ROC-AUC'].idxmax()
    best_model      = tuned_models[best_model_name]
    y_prob_best     = best_model.predict_proba(X_test_sel)[:, 1]

    # ── Cells 38-40: Threshold Tuning ──────────────────────────────────
    thresholds = np.arange(0.1, 0.91, 0.05)
    FP_COST = 100
    FN_COST = 500

    threshold_results = []
    for t in thresholds:
        y_pred_t = (y_prob_best >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
        threshold_results.append({
            'Threshold'   : round(t, 2),
            'Precision'   : precision_score(y_test, y_pred_t, zero_division=0),
            'Recall'      : recall_score(y_test, y_pred_t),
            'F1'          : f1_score(y_test, y_pred_t),
            'ROC-AUC'     : roc_auc_score(y_test, y_prob_best),
            'BusinessCost': fp * FP_COST + fn * FN_COST,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
        })

    thresh_df         = pd.DataFrame(threshold_results)
    optimal_threshold = thresh_df.loc[thresh_df['BusinessCost'].idxmin(), 'Threshold']
    best_f1_thresh    = thresh_df.loc[thresh_df['F1'].idxmax(), 'Threshold']

    return dict(
        df_clean          = df,
        X_train           = X_train,
        X_test            = X_test,
        y_train           = y_train,
        y_test            = y_test,
        X_train_enc       = X_train_enc,
        X_train_scaled    = X_train_scaled,
        X_test_scaled     = X_test_scaled,
        X_train_res       = X_train_res,
        y_train_res       = y_train_res,
        X_train_sel       = X_train_sel,
        X_test_sel        = X_test_sel,
        scaler            = scaler,
        feature_names     = feature_names,
        l1_mask           = l1_mask,
        l1_selector       = l1_selector,
        selected_features = selected_features,
        tuned_models      = tuned_models,
        results_df        = results_df,
        best_model_name   = best_model_name,
        best_model        = best_model,
        y_prob_best       = y_prob_best,
        thresh_df         = thresh_df,
        optimal_threshold = optimal_threshold,
        best_f1_thresh    = best_f1_thresh,
        cv                = cv,
    )


def predict_churn(customer_data, pipeline):
    df_in = pd.DataFrame([customer_data])

    df_in['TotalCharges'] = pd.to_numeric(df_in['TotalCharges'], errors='coerce')

    df_in['TenureGroup'] = pd.cut(
        df_in['tenure'],
        bins=[0, 12, 24, 48, float('inf')],
        labels=['New', 'Regular', 'Established', 'Loyal']
    )
    df_in['IsFirstYear'] = (df_in['tenure'] <= 12).astype(int)
    df_in['IsLongTerm']  = (df_in['tenure'] >= 24).astype(int)

    df_in['AvgMonthlyCharge'] = df_in.apply(
        lambda x: x['TotalCharges'] / x['tenure'] if x['tenure'] > 0 else x['MonthlyCharges'], axis=1
    )
    df_in['CustomerLTV'] = df_in['TotalCharges'] + (df_in['MonthlyCharges'] * 6)

    additional_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_in['NumAdditionalServices'] = df_in[additional_services].apply(
        lambda x: (x == 'Yes').sum(), axis=1
    )
    df_in['HasSecurityBundle']  = ((df_in['OnlineSecurity'] == 'Yes') & (df_in['OnlineBackup'] == 'Yes')).astype(int)
    df_in['HasStreamingBundle'] = ((df_in['StreamingTV'] == 'Yes') & (df_in['StreamingMovies'] == 'Yes')).astype(int)
    df_in['InternetUser']       = (df_in['InternetService'] != 'No').astype(int)
    df_in['FiberOpticUser']     = (df_in['InternetService'] == 'Fiber optic').astype(int)
    df_in['ServicesPerMonth']   = df_in['NumAdditionalServices'] / (df_in['tenure'] + 1)

    df_in['IsMonthToMonth'] = (df_in['Contract'] == 'Month-to-month').astype(int)
    df_in['ContractType']   = df_in['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df_in['ElectronicPayment'] = df_in['PaymentMethod'].str.contains('electronic check|automatic', case=False).astype(int)
    df_in['PaymentRisk'] = df_in['PaymentMethod'].map({
        'Electronic check': 3, 'Mailed check': 2,
        'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1
    })
    df_in['PaperlessBilling'] = df_in['PaperlessBilling'].map({'Yes': 1, 'No': 0}) if df_in['PaperlessBilling'].dtype == 'object' else df_in['PaperlessBilling']
    df_in['PaperlessHighRisk'] = ((df_in['PaperlessBilling'] == 1) & (df_in['PaymentMethod'] == 'Electronic check')).astype(int)

    df_in['Partner']    = df_in['Partner'].map({'Yes': 1, 'No': 0}) if df_in['Partner'].dtype == 'object' else df_in['Partner']
    df_in['Dependents'] = df_in['Dependents'].map({'Yes': 1, 'No': 0}) if df_in['Dependents'].dtype == 'object' else df_in['Dependents']
    df_in['HasFamily']  = ((df_in['Partner'] == 1) | (df_in['Dependents'] == 1)).astype(int)

    train_median = pipeline['df_clean']['MonthlyCharges'].median()
    df_in['HighCostLowTenure'] = (
        (df_in['MonthlyCharges'] > train_median) & (df_in['tenure'] < 12)
    ).astype(int)

    max_tenure = pipeline['df_clean']['tenure'].max()
    df_in['EngagementScore'] = (
        df_in['NumAdditionalServices'] * 0.3 +
        df_in['ContractType'] * 0.4 +
        (df_in['tenure'] / max_tenure) * 0.3
    )

    df_in = pd.get_dummies(df_in, drop_first=True)
    df_in = df_in.reindex(columns=pipeline['feature_names'], fill_value=0).astype(float)

    df_scaled = pipeline['scaler'].transform(df_in)
    df_sel    = df_scaled[:, pipeline['l1_mask']]

    prob = pipeline['best_model'].predict_proba(df_sel)[0, 1]
    return prob


# ============================================================================
# MAIN APP
# ============================================================================
st.title("📊 Telecom Customer Churn Predictor")

df = load_data()
if df is None:
    st.stop()

with st.spinner("🔄 Running full ML pipeline (first load takes a few minutes)..."):
    pipeline = run_full_pipeline(df)

st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("", [
    "🏠 Dashboard",
    "🎯 Make Prediction",
    "📈 EDA & Visualizations",
    "🤖 Model Training & CV",
    "🏆 Model Comparison",
    "⚖️ Threshold Tuning",
    "🔍 SHAP Interpretability",
    "📋 Final Summary"
])

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
if page == "🏠 Dashboard":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <h2 style="color: #667eea; margin-top: 0;">Welcome to Churn Analytics</h2>
        <p style="font-size: 16px; color: #555;">
        Predict customer churn with machine learning. This app runs the complete pipeline from your notebook —
        EDA, feature engineering, SMOTE, L1 selection, 4-model comparison, threshold tuning, and SHAP.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Total Customers",   f"{TOTAL_CUSTOMERS:,}", "📊")
    with col2: st.metric("Churn Rate",         f"{CHURN_RATE:.1%}",   "📉")
    with col3: st.metric("Best Model",         pipeline['best_model_name'].split()[0])
    with col4: st.metric("Best ROC-AUC",       f"{pipeline['results_df']['ROC-AUC'].max():.2%}", "📈")
    with col5: st.metric("Optimal Threshold",  f"{pipeline['optimal_threshold']}", "⚖️")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🔍 Pipeline Overview
        1. **Data Loading & Cleaning** — fix TotalCharges, drop customerID
        2. **EDA** — distributions, churn rates, correlation heatmap
        3. **Feature Engineering** — 12+ engineered features
        4. **Train-Test Split** — 80/20 stratified
        5. **Encoding & Scaling** — one-hot + StandardScaler
        6. **L1 Feature Selection** — Lasso regularization
        7. **SMOTE** — handle class imbalance
        8. **4 Models + Tuning** — LR, DT, RF, XGBoost
        9. **Threshold Tuning** — business cost optimization
        10. **SHAP** — model interpretability
        """)
    with col2:
        st.markdown(f"""
        ### 📊 Pipeline Results
        - **Features after engineering:** {len(pipeline['feature_names'])}
        - **Features after L1 selection:** {len(pipeline['selected_features'])}
        - **Train size (after SMOTE):** {pipeline['X_train_res'].shape[0]}
        - **Test size:** {pipeline['X_test_sel'].shape[0]}
        - **Best model:** {pipeline['best_model_name']}
        - **Best ROC-AUC:** {pipeline['results_df']['ROC-AUC'].max():.4f}
        - **Optimal threshold:** {pipeline['optimal_threshold']} (min business cost)
        - **Min business cost:** ${pipeline['thresh_df']['BusinessCost'].min():,.0f}
        """)

# ============================================================================
# PAGE: MAKE PREDICTION
# ============================================================================
elif page == "🎯 Make Prediction":
    st.markdown("<h2 class='section-header'>🎯 Customer Churn Prediction</h2>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("### 📋 Enter Customer Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Demographics")
            gender     = st.selectbox("Gender",         ["Male", "Female"])
            senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner    = st.selectbox("Partner Status", ["No", "Yes"])
            dependents = st.selectbox("Dependents",     ["No", "Yes"])

        with col2:
            st.subheader("📱 Services")
            tenure   = st.slider("Tenure (months)", 0, 72, 12)
            phone    = st.selectbox("Phone Service",    ["No", "Yes"])
            internet = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
            security = st.selectbox("Online Security",  ["No", "Yes"])
            backup   = st.selectbox("Online Backup",    ["No", "Yes"])

        with col3:
            st.subheader("💳 Billing & Services")
            contract = st.selectbox("Contract Type",
                                    ["Month-to-month", "One year", "Two year"])
            device   = st.selectbox("Device Protection", ["No", "Yes"])
            tech     = st.selectbox("Tech Support",      ["No", "Yes"])
            payment  = st.selectbox("Payment Method",
                                    ["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)", "Credit card (automatic)"])

        col1, col2 = st.columns(2)
        with col1: monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0,   step=5.0)
        with col2: total   = st.number_input("Total Charges ($)",   min_value=0.0, value=1000.0, step=50.0)

        col1, col2, col3 = st.columns(3)
        with col1: streaming_tv     = st.selectbox("Streaming TV",      ["No", "Yes"])
        with col2: streaming_movies = st.selectbox("Streaming Movies",  ["No", "Yes"])
        with col3: paperless        = st.selectbox("Paperless Billing", ["No", "Yes"])

        submit = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submit:
        customer_input = {
            'gender': gender, 'SeniorCitizen': 1 if senior == 'Yes' else 0,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone, 'InternetService': internet,
            'OnlineSecurity': security, 'OnlineBackup': backup,
            'DeviceProtection': device, 'TechSupport': tech,
            'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
            'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total,
            'MultipleLines': 'No'
        }

        prob = predict_churn(customer_input, pipeline)
        optimal_thresh = pipeline['optimal_threshold']

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if prob >= optimal_thresh:
                st.markdown("""
                <div class="prediction-high-risk">
                    <h3 style="color: #ff4444; margin: 0;">⚠️ HIGH CHURN RISK</h3>
                    <p style="margin: 10px 0; font-size: 16px;">This customer is at risk of churning</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-low-risk">
                    <h3 style="color: #44aa44; margin: 0;">✅ LOW CHURN RISK</h3>
                    <p style="margin: 10px 0; font-size: 16px;">This customer is likely to stay</p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.metric("Churn Probability", f"{prob:.1%}",
                      delta=f"Risk: {'HIGH' if prob >= optimal_thresh else 'LOW'}",
                      delta_color="inverse")
            st.caption(f"Using optimal threshold: {optimal_thresh} (min business cost)")

        st.markdown("---")
        st.markdown("### 📊 Prediction Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Tenure:** {tenure} months")
            st.write(f"**Contract:** {contract}")
        with col2:
            st.write(f"**Monthly:** ${monthly:.2f}")
            st.write(f"**Total Charges:** ${total:.2f}")
        with col3:
            st.write(f"**Internet:** {internet}")
            st.write(f"**Tech Support:** {tech}")

# ============================================================================
# PAGE: EDA & VISUALIZATIONS
# ============================================================================
elif page == "📈 EDA & Visualizations":
    st.markdown("<h2 class='section-header'>📈 Exploratory Data Analysis</h2>", unsafe_allow_html=True)

    df_clean = pipeline['df_clean']

    # Cell 8: Target Distribution
    st.markdown("### Class Imbalance Overview")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    churn_counts = df_clean['Churn'].value_counts()
    axes[0].bar(['No Churn (0)', 'Churn (1)'], churn_counts.values,
                color=['#4C72B0', '#DD8452'], edgecolor='black')
    axes[0].set_title('Target Class Distribution', fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 30, str(v), ha='center', fontweight='bold')
    axes[1].pie(churn_counts.values, labels=['No Churn', 'Churn'],
                autopct='%1.1f%%', colors=['#4C72B0', '#DD8452'], startangle=90)
    axes[1].set_title('Churn Proportion', fontweight='bold')
    plt.suptitle('Class Imbalance Overview', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Cell 9: Numerical Feature Distributions
    st.markdown("### Numerical Features vs Churn")
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for i, col in enumerate(num_cols):
        df_clean[df_clean['Churn'] == 0][col].hist(ax=axes[i], alpha=0.6, bins=30,
                                                    label='No Churn', color='#4C72B0')
        df_clean[df_clean['Churn'] == 1][col].hist(ax=axes[i], alpha=0.6, bins=30,
                                                    label='Churn',    color='#DD8452')
        axes[i].set_title(f'{col} by Churn', fontweight='bold')
        axes[i].legend()
    plt.suptitle('Numerical Features vs Churn', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Cell 10: Churn Rate by Categorical Features
    st.markdown("### Churn Rate by Categorical Features")
    cat_cols = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'OnlineSecurity']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        churn_rate = df_clean.groupby(col)['Churn'].mean().sort_values(ascending=False)
        churn_rate.plot(kind='bar', ax=axes[i], color='#DD8452', edgecolor='black', alpha=0.85)
        axes[i].set_title(f'Churn Rate by {col}', fontweight='bold')
        axes[i].set_ylabel('Churn Rate')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=25, ha='right')
        axes[i].set_ylim(0, 1)
        for bar in axes[i].patches:
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{bar.get_height():.2f}', ha='center', fontsize=9)
    axes[-1].set_visible(False)
    plt.suptitle('Churn Rate by Categorical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Cell 11: Correlation Heatmap
    st.markdown("### Correlation Heatmap")
    num_df = df_clean.select_dtypes(include=[np.number])
    corr   = num_df.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.5, annot_kws={'size': 8}, ax=ax)
    ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    st.markdown("### 📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ============================================================================
# PAGE: MODEL TRAINING & CV
# ============================================================================
elif page == "🤖 Model Training & CV":
    st.markdown("<h2 class='section-header'>🤖 Model Training & Cross-Validation</h2>", unsafe_allow_html=True)

    # SMOTE before/after (Cell 22)
    st.markdown("### SMOTE — Class Imbalance Handling")
    before = Counter(pipeline['y_train'])
    after  = Counter(pipeline['y_train_res'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, counts, title in zip(axes, [before, after], ['Before SMOTE', 'After SMOTE']):
        ax.bar(['No Churn', 'Churn'], [counts[0], counts[1]],
               color=['#4C72B0', '#DD8452'], edgecolor='black')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Count')
        for i, v in enumerate([counts[0], counts[1]]):
            ax.text(i, v + 20, str(v), ha='center', fontweight='bold')
    plt.suptitle('Class Imbalance: Before vs After SMOTE', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # L1 Feature importance (Cell 20)
    st.markdown("### L1 Feature Selection — Top 20 Coefficients")
    selected_features = pipeline['selected_features']
    l1_selector       = pipeline['l1_selector']
    l1_mask           = pipeline['l1_mask']
    feature_names     = pipeline['feature_names']

    coef_df = pd.DataFrame({
        'Feature'    : selected_features,
        'Coefficient': l1_selector.coef_[0][l1_mask]
    }).sort_values('Coefficient', key=abs, ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = ['#DD8452' if c > 0 else '#4C72B0' for c in coef_df['Coefficient']]
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('Top 20 Features — L1 Coefficients', fontweight='bold')
    ax.set_xlabel('Coefficient')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown(f"**{len(selected_features)} features selected out of {len(feature_names)} total**")
    st.write("Selected features:", list(selected_features))

    st.markdown("---")

    # CV results display (Cell 24-25)
    st.markdown("### 5-Fold Cross-Validation Results (Base Models)")
    st.info("ℹ️ CV was run during pipeline training. Showing tuned model test-set scores below.")

    metrics_to_track = ['ROC-AUC', 'F1', 'Recall', 'Precision']
    colors_cv        = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    results_df       = pipeline['results_df']

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, metric in zip(axes, metrics_to_track):
        vals = results_df[metric].values
        bars = ax.bar(results_df.index, vals, color=colors_cv, edgecolor='black', alpha=0.85)
        ax.set_title(f'CV {metric.upper()}', fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_xticklabels(results_df.index, rotation=20, ha='right', fontsize=9)
        for bar, m in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{m:.3f}', ha='center', fontsize=8, fontweight='bold')
    plt.suptitle('5-Fold Cross-Validation Results (Base Models)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ============================================================================
# PAGE: MODEL COMPARISON
# ============================================================================
elif page == "🏆 Model Comparison":
    st.markdown("<h2 class='section-header'>🏆 Model Comparison on Test Set</h2>", unsafe_allow_html=True)

    results_df      = pipeline['results_df']
    tuned_models    = pipeline['tuned_models']
    X_test_sel      = pipeline['X_test_sel']
    y_test          = pipeline['y_test']
    best_model_name = pipeline['best_model_name']

    st.markdown("### Performance Table")
    st.dataframe(results_df.style.highlight_max(axis=0, color='#d4edda'), use_container_width=True)
    st.success(f"🏆 Best model by ROC-AUC: **{best_model_name}** ({results_df.loc[best_model_name, 'ROC-AUC']:.4f})")

    st.markdown("---")

    # Cell 33: Visual Comparison
    st.markdown("### Model Performance Comparison (Test Set)")
    metrics    = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    fig, ax    = plt.subplots(figsize=(14, 6))
    x          = np.arange(len(results_df))
    width      = 0.15
    bar_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
    for i, (metric, color) in enumerate(zip(metrics, bar_colors)):
        ax.bar(x + i * width, results_df[metric], width,
               label=metric, color=color, edgecolor='black', alpha=0.85)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(results_df.index, rotation=15, ha='right')
    ax.set_ylim(0, 1.15)
    ax.set_title('Model Performance Comparison (Test Set)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylabel('Score')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Cell 34: ROC Curves
    st.markdown("### ROC-AUC Curves — All Models")
    colors_roc = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    fig, ax    = plt.subplots(figsize=(8, 6))
    for (name, model), color in zip(tuned_models.items(), colors_roc):
        y_prob = model.predict_proba(X_test_sel)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_val:.3f})', color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC-AUC Curves — All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Cell 35: Precision-Recall Curves
    st.markdown("### Precision-Recall Curves — All Models")
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, model), color in zip(tuned_models.items(), colors_roc):
        y_prob = model.predict_proba(X_test_sel)[:, 1]
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_vals, precision_vals)
        ax.plot(recall_vals, precision_vals, label=f'{name} (AUC={pr_auc:.3f})', color=color, linewidth=2)
    baseline = y_test.mean()
    ax.axhline(baseline, color='black', linestyle='--', label=f'Baseline ({baseline:.2f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves — All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Cell 36: Confusion Matrices
    st.markdown("### Confusion Matrices — All Models")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (name, model) in zip(axes, tuned_models.items()):
        y_pred = model.predict(X_test_sel)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ============================================================================
# PAGE: THRESHOLD TUNING
# ============================================================================
elif page == "⚖️ Threshold Tuning":
    st.markdown("<h2 class='section-header'>⚖️ Threshold Tuning on Best Model</h2>", unsafe_allow_html=True)

    st.info("""
    In churn prediction, **recall is business-critical** (missing a churner = lost revenue).
    We tune the decision threshold to balance Precision vs Recall and minimise business cost.
    - **FP cost:** $100 (sending retention offer to non-churner)
    - **FN cost:** $500 (missing a churner)
    """)

    thresh_df         = pipeline['thresh_df']
    best_model_name   = pipeline['best_model_name']
    optimal_threshold = pipeline['optimal_threshold']
    best_f1_thresh    = pipeline['best_f1_thresh']

    st.markdown("### Threshold Results Table")
    st.dataframe(thresh_df[['Threshold','Precision','Recall','F1','BusinessCost','TP','FP','TN','FN']],
                 use_container_width=True)

    st.markdown("---")

    # Cell 39: Precision-Recall-F1 vs Threshold
    st.markdown(f"### Precision · Recall · F1 vs Threshold — {best_model_name}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresh_df['Threshold'], thresh_df['Precision'], 'b-o', label='Precision', markersize=5)
    ax.plot(thresh_df['Threshold'], thresh_df['Recall'],    'r-o', label='Recall',    markersize=5)
    ax.plot(thresh_df['Threshold'], thresh_df['F1'],        'g-o', label='F1 Score',  markersize=5)
    ax.axvline(best_f1_thresh, color='gray', linestyle='--',
               label=f'Best F1 @ {best_f1_thresh}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title(f'Threshold Tuning — {best_model_name}', fontweight='bold')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Cell 40: Business Cost vs Threshold
    st.markdown("### Business Cost vs Threshold")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresh_df['Threshold'], thresh_df['BusinessCost'], 'r-o', markersize=5)
    ax.axvline(optimal_threshold, color='green', linestyle='--',
               label=f'Min Cost @ {optimal_threshold}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Total Business Cost ($)')
    ax.set_title('Business Cost vs Threshold', fontweight='bold')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.success(f"✅ Optimal threshold: **{optimal_threshold}** | Min business cost: **${thresh_df['BusinessCost'].min():,.0f}**")

    st.markdown("---")

    # Cell 41: Final evaluation at optimal threshold
    st.markdown(f"### Final Evaluation @ threshold = {optimal_threshold}")
    y_test       = pipeline['y_test']
    y_prob_best  = pipeline['y_prob_best']
    y_pred_final = (y_prob_best >= optimal_threshold).astype(int)

    report = classification_report(y_test, y_pred_final,
                                   target_names=['No Churn', 'Churn'], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
    st.write(f"**ROC-AUC:** {roc_auc_score(y_test, y_prob_best):.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    cm_final = confusion_matrix(y_test, y_pred_final)
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    ax.set_title(f'Final Confusion Matrix — {best_model_name}\n(threshold={optimal_threshold})',
                 fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ============================================================================
# PAGE: SHAP INTERPRETABILITY
# ============================================================================
elif page == "🔍 SHAP Interpretability":
    st.markdown("<h2 class='section-header'>🔍 SHAP Model Interpretability</h2>", unsafe_allow_html=True)

    best_model        = pipeline['best_model']
    best_model_name   = pipeline['best_model_name']
    X_test_sel        = pipeline['X_test_sel']
    X_train_res       = pipeline['X_train_res']
    selected_features = pipeline['selected_features']
    y_prob_best       = pipeline['y_prob_best']

    X_test_sel_df  = pd.DataFrame(X_test_sel,  columns=selected_features)
    X_train_sel_df = pd.DataFrame(X_train_res, columns=selected_features)

    with st.spinner("Computing SHAP values..."):
        if best_model_name == 'Logistic Regression':
            explainer   = shap.LinearExplainer(best_model, X_train_sel_df)
        else:
            explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_sel_df)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

    st.success(f"✅ SHAP values computed — shape: {shap_vals.shape}")

    # Cell 44: SHAP Summary Plot (Beeswarm)
    st.markdown("### SHAP Summary Plot (Beeswarm)")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_test_sel_df, plot_type='dot', show=False, max_display=20)
    plt.title(f'SHAP Summary — {best_model_name}', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Cell 45: SHAP Bar Plot
    st.markdown("### SHAP Feature Importance (Bar)")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_test_sel_df, plot_type='bar', show=False, max_display=20)
    plt.title(f'SHAP Feature Importance — {best_model_name}', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Cell 46: SHAP Waterfall — Highest Risk Customer
    st.markdown("### SHAP Waterfall — Highest Risk Customer")
    high_risk_idx  = np.argmax(y_prob_best)
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    shap_explanation = shap.Explanation(
        values        = shap_vals[high_risk_idx],
        base_values   = expected_value,
        data          = X_test_sel_df.iloc[high_risk_idx].values,
        feature_names = list(selected_features)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_explanation, max_display=15, show=False)
    plt.title(f'SHAP Waterfall — Highest Risk Customer\n(Predicted Prob={y_prob_best[high_risk_idx]:.3f})',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Cell 47: SHAP Dependence Plots
    st.markdown("### SHAP Dependence Plots — Top 2 Features")
    shap_importance = np.abs(shap_vals).mean(axis=0)
    top2_features   = np.argsort(shap_importance)[::-1][:2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, feat_idx in zip(axes, top2_features):
        feat_name = selected_features[feat_idx]
        ax.scatter(
            X_test_sel_df[feat_name],
            shap_vals[:, feat_idx],
            alpha=0.4, s=15, c=shap_vals[:, feat_idx], cmap='RdYlBu_r'
        )
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel(feat_name)
        ax.set_ylabel('SHAP Value')
        ax.set_title(f'SHAP Dependence: {feat_name}', fontweight='bold')
    plt.suptitle('SHAP Dependence Plots — Top 2 Features', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ============================================================================
# PAGE: FINAL SUMMARY
# ============================================================================
elif page == "📋 Final Summary":
    st.markdown("<h2 class='section-header'>📋 Final Pipeline Summary</h2>", unsafe_allow_html=True)

    results_df        = pipeline['results_df']
    best_model_name   = pipeline['best_model_name']
    optimal_threshold = pipeline['optimal_threshold']
    y_test            = pipeline['y_test']
    y_prob_best       = pipeline['y_prob_best']
    selected_features = pipeline['selected_features']
    feature_names     = pipeline['feature_names']
    X_train_res       = pipeline['X_train_res']
    X_test_sel        = pipeline['X_test_sel']
    df_clean          = pipeline['df_clean']

    y_final_eval = (y_prob_best >= optimal_threshold).astype(int)

    # Cell 49: Final summary printout
    st.code(f"""
{'='*60}
           TELECOM CHURN — FINAL PIPELINE SUMMARY
{'='*60}
Dataset shape         : {df_clean.shape}
Features after FE     : {len(feature_names)}
Features after L1 sel : {len(selected_features)}
Train size (SMOTE)    : {X_train_res.shape[0]}
Test  size            : {X_test_sel.shape[0]}
SMOTE applied         : Yes

--- Model Comparison (Test Set, threshold=0.5) ---
{results_df.to_string()}

🏆 Best Model         : {best_model_name}
   ROC-AUC            : {results_df.loc[best_model_name, 'ROC-AUC']:.4f}
   Optimal Threshold  : {optimal_threshold} (min business cost)
   Final Recall       : {recall_score(y_test, y_final_eval):.4f}
   Final Precision    : {precision_score(y_test, y_final_eval):.4f}
   Final F1           : {f1_score(y_test, y_final_eval):.4f}
{'='*60}
    """, language="text")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Best Model",        best_model_name.split()[0])
    with col2: st.metric("ROC-AUC",           f"{results_df.loc[best_model_name, 'ROC-AUC']:.4f}")
    with col3: st.metric("Optimal Threshold", f"{optimal_threshold}")
    with col4: st.metric("Min Business Cost", f"${pipeline['thresh_df']['BusinessCost'].min():,.0f}")

    st.markdown("---")
    st.markdown("### All Models — Final Results")
    st.dataframe(results_df.style.highlight_max(axis=0, color='#d4edda'), use_container_width=True)

    st.markdown("---")
    st.markdown("### Selected Features")
    st.write(f"**{len(selected_features)} features selected by L1:**")
    cols = st.columns(4)
    for i, feat in enumerate(selected_features):
        with cols[i % 4]:
            st.write(f"✅ **{feat}**")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 12px; padding: 20px;">
📊 Telecom Customer Churn Predictor |
Complete ML Pipeline (Notebook → App) |
LR · DT · RF · XGBoost + SMOTE + SHAP |
Deployed with Streamlit
</div>
""", unsafe_allow_html=True)
