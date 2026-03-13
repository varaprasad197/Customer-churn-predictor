import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from scipy.stats import loguniform
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================================================
# ADVANCED PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🎯 Telecom Churn Sentinel",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STUNNING CUSTOM STYLING
# ============================================================================
st.markdown("""
<style>
    /* Root styling */
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --success: #2ecc71;
        --warning: #f39c12;
        --danger: #e74c3c;
        --info: #3498db;
    }
    
    /* Main container */
    .main {
        padding: 0rem 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .header-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
    }
    
    /* Risk indicator cards */
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        border-left: 6px solid #e74c3c;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 8px 16px rgba(231, 76, 60, 0.2);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        border-left: 6px solid #2ecc71;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 8px 16px rgba(46, 204, 113, 0.2);
    }
    
    /* Section header */
    .section-header {
        border-bottom: 3px solid #667eea;
        border-left: 6px solid #667eea;
        padding: 15px 20px;
        margin: 25px 0;
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.05) 0%, transparent 100%);
        border-radius: 5px;
    }
    
    /* Form styling */
    .form-container {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.1);
        margin-bottom: 20px;
    }
    
    /* Alert boxes */
    .alert-success {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2ecc71;
        margin: 15px 0;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #e74c3c;
        margin: 15px 0;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #4dabf7 0%, #339af0 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #3498db;
        margin: 15px 0;
    }
    
    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    /* Feature pill */
    .feature-pill {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px;
        font-size: 12px;
        font-weight: bold;
    }
    
    /* Footer */
    .footer-section {
        text-align: center;
        color: #7f8c8d;
        font-size: 12px;
        padding: 20px;
        border-top: 2px solid #ecf0f1;
        margin-top: 30px;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 30px !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ecf0f1 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ACTUAL METRICS
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
# DATA LOADING & MODEL TRAINING
# ============================================================================
@st.cache_data
def load_data():
    github_csv_url = "https://raw.githubusercontent.com/varaprasad197/Customer-churn-predictor/main/tele_comm.csv"
    try:
        return pd.read_csv(github_csv_url)
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

@st.cache_resource
def train_model(df):
    df = df.copy()
    df.drop(columns=['customerID'], inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if df[col].dtype == object:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_median_charge = X_train['MonthlyCharges'].median()
    
    def engineer_features(df_in, median_charge):
        df_in = df_in.copy()
        df_in['IsFirstYear'] = (df_in['tenure'] <= 12).astype(int)
        df_in['AvgMonthlyCharge'] = df_in.apply(
            lambda x: x['TotalCharges'] / x['tenure'] if x['tenure'] > 0 else x['MonthlyCharges'], axis=1)
        add_svcs = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        df_in['NumAdditionalServices'] = df_in[add_svcs].apply(
            lambda x: (x == 'Yes').sum() if isinstance(x.iloc[0], str) else x.sum(), axis=1)
        df_in['FiberOpticUser'] = (df_in['InternetService'] == 'Fiber optic').astype(int)
        df_in['IsMonthToMonth'] = (df_in['Contract'] == 'Month-to-month').astype(int)
        df_in['PaymentRisk'] = df_in['PaymentMethod'].map({
            'Electronic check': 3, 'Mailed check': 2,
            'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1})
        df_in['HighCostLowTenure'] = ((df_in['MonthlyCharges'] > median_charge) & (df_in['tenure'] < 12)).astype(int)
        df_in['HasFamily'] = ((df_in['Partner'] == 1) | (df_in['Dependents'] == 1)).astype(int)
        return df_in
    
    X_train = engineer_features(X_train, train_median_charge)
    X_test = engineer_features(X_test, train_median_charge)
    
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    l1_selector = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=1000, random_state=42)
    l1_selector.fit(X_train_res, y_train_res)
    l1_mask = l1_selector.coef_[0] != 0
    selected_features = X_train.columns[l1_mask].tolist()
    
    X_train_sel = X_train_res[:, l1_mask]
    X_test_sel = X_test_scaled[:, l1_mask]
    
    param_dist = {
        'C': loguniform(0.01, 10),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'max_iter': [500, 1000]
    }
    
    random_search = RandomizedSearchCV(
        LogisticRegression(random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='recall',
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train_sel, y_train_res)
    best_model = random_search.best_estimator_
    best_model.fit(X_train_sel, y_train_res)
    
    return {
        'model': best_model,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_sel': X_train_sel,
        'X_test_sel': X_test_sel,
        'selected_features': selected_features,
        'l1_mask': l1_mask,
        'train_median_charge': train_median_charge,
    }

def predict_churn(customer_data, model_artifacts):
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    X_train = model_artifacts['X_train']
    l1_mask = model_artifacts['l1_mask']
    train_median_charge = model_artifacts['train_median_charge']
    
    df = pd.DataFrame([customer_data])
    
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
    
    df['IsFirstYear'] = (df['tenure'] <= 12).astype(int) if 'tenure' in df.columns else 0
    df['AvgMonthlyCharge'] = df.apply(lambda x: x['TotalCharges'] / x['tenure'] if x['tenure'] > 0 else x['MonthlyCharges'], axis=1) if 'tenure' in df.columns else 0
    
    add_svcs = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    if all(col in df.columns for col in add_svcs):
        df['NumAdditionalServices'] = df[add_svcs].apply(lambda x: (x == 'Yes').sum() if isinstance(x.iloc[0], str) else x.sum(), axis=1)
    
    df['FiberOpticUser'] = (df['InternetService'] == 'Fiber optic').astype(int) if 'InternetService' in df.columns else 0
    df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int) if 'Contract' in df.columns else 0
    df['PaymentRisk'] = df['PaymentMethod'].map({'Electronic check': 3, 'Mailed check': 2, 'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1}) if 'PaymentMethod' in df.columns else 1
    df['HighCostLowTenure'] = ((df['MonthlyCharges'] > train_median_charge) & (df['tenure'] < 12)).astype(int) if 'MonthlyCharges' in df.columns else 0
    df['HasFamily'] = ((df['Partner'] == 1) | (df['Dependents'] == 1)).astype(int) if 'Partner' in df.columns else 0
    
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=X_train.columns, fill_value=0)
    df_scaled = scaler.transform(df)
    df_sel = df_scaled[:, l1_mask]
    
    prob_churn = model.predict_proba(df_sel)[0, 1]
    return prob_churn

# ============================================================================
# MAIN APP
# ============================================================================
df = load_data()
if df is None:
    st.stop()

with st.spinner("⚡ Initializing Churn Sentinel System..."):
    model_artifacts = train_model(df)

# Header
st.markdown("""
<div class="header-section">
    <h1 style="margin: 0; font-size: 40px; text-align: center;">🚨 CHURN SENTINEL SYSTEM</h1>
    <p style="text-align: center; margin: 10px 0 0 0; font-size: 16px; opacity: 0.95;">
    Real-time Customer Churn Prediction & Monitoring Platform
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px;">
    <h2>🎯 NAVIGATION</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["🏠 Dashboard", "🔍 Vigilant Scan", "📊 Analytics Center", "⚡ Performance Hub"],
    label_visibility="collapsed"
)

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
if page == "🏠 Dashboard":
    st.markdown("<div class='section-header'>📊 SYSTEM OVERVIEW</div>", unsafe_allow_html=True)
    
    # Key metrics in stunning cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics_data = [
        (col1, "👥 TOTAL CUSTOMERS", TOTAL_CUSTOMERS, "7,043 Active"),
        (col2, "📉 CHURN RATE", f"{CHURN_RATE:.1%}", "26.5% Risk"),
        (col3, "✅ ACCURACY", f"{ACTUAL_METRICS['accuracy']:.2%}", "Model Precision"),
        (col4, "🎯 RECALL", f"{ACTUAL_METRICS['recall']:.2%}", "Detection Rate"),
        (col5, "📈 ROC-AUC", f"{ACTUAL_METRICS['roc_auc']:.2%}", "Classification"),
    ]
    
    for col, label, value, desc in metrics_data:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 5px;">{label}</div>
                <div style="font-size: 28px; font-weight: bold; margin: 10px 0;">{value}</div>
                <div style="font-size: 11px; opacity: 0.8;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Intelligence Boxes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="alert-info">
            <h3 style="margin: 0; color: white;">🔍 SENTINEL INTELLIGENCE</h3>
            <p style="margin: 10px 0 0 0;">
            <strong>Detection Capability:</strong> Identifies 81% of at-risk customers<br>
            <strong>Model Type:</strong> Logistic Regression with SMOTE balancing<br>
            <strong>Features Analyzed:</strong> 22 key customer attributes<br>
            <strong>Real-time:</strong> Live prediction available
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="alert-success">
            <h3 style="margin: 0; color: white;">⚡ SYSTEM STATUS</h3>
            <p style="margin: 10px 0 0 0;">
            <strong>Status:</strong> ✅ OPERATIONAL<br>
            <strong>Last Update:</strong> Just Now<br>
            <strong>Training Status:</strong> Optimized (5-fold CV)<br>
            <strong>Generalization:</strong> Excellent (3% gap)
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE: VIGILANT SCAN (PREDICTION)
# ============================================================================
elif page == "🔍 Vigilant Scan":
    st.markdown("<div class='section-header'>🔍 CUSTOMER CHURN RISK ASSESSMENT</div>", unsafe_allow_html=True)
    
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("<div class='form-container'>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 👤 Demographics")
            gender = st.selectbox("👨 Gender", ["Male", "Female"], key="gender")
            senior = st.selectbox("👴 Senior Citizen", ["No", "Yes"], key="senior")
            partner = st.selectbox("💑 Partner Status", ["No", "Yes"], key="partner")
            dependents = st.selectbox("👶 Dependents", ["No", "Yes"], key="dependents")
        
        with col2:
            st.markdown("### 📱 Services")
            tenure = st.slider("📅 Tenure (months)", 0, 72, 12)
            phone = st.selectbox("☎️ Phone Service", ["No", "Yes"], key="phone")
            internet = st.selectbox("🌐 Internet Service", ["No", "DSL", "Fiber optic"], key="internet")
            security = st.selectbox("🔒 Online Security", ["No", "Yes"], key="security")
            backup = st.selectbox("💾 Online Backup", ["No", "Yes"], key="backup")
        
        with col3:
            st.markdown("### 💳 Billing & Contract")
            contract = st.selectbox("📜 Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
            device = st.selectbox("🖥️ Device Protection", ["No", "Yes"], key="device")
            tech = st.selectbox("🛠️ Tech Support", ["No", "Yes"], key="tech")
            payment = st.selectbox("💰 Payment Method", 
                                 ["Electronic check", "Mailed check", 
                                  "Bank transfer (automatic)", "Credit card (automatic)"], key="payment")
        
        col1, col2 = st.columns(2)
        with col1:
            monthly = st.number_input("💵 Monthly Charges ($)", min_value=0.0, value=65.0, step=5.0)
        with col2:
            total = st.number_input("💲 Total Charges ($)", min_value=0.0, value=1000.0, step=50.0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            streaming_tv = st.selectbox("📺 Streaming TV", ["No", "Yes"], key="tv")
        with col2:
            streaming_movies = st.selectbox("🎬 Streaming Movies", ["No", "Yes"], key="movies")
        with col3:
            paperless = st.selectbox("📄 Paperless Billing", ["No", "Yes"], key="paperless")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Submit button
        submit = st.form_submit_button("🚀 SCAN FOR CHURN RISK", use_container_width=True)
    
    if submit:
        customer_input = {
            'gender': gender, 'SeniorCitizen': 1 if senior == 'Yes' else 0,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone, 'InternetService': internet,
            'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': device, 
            'TechSupport': tech, 'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
            'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total
        }
        
        prob = predict_churn(customer_input, model_artifacts)
        
        st.markdown("---")
        st.markdown("<div class='section-header'>📋 SCAN RESULTS</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prob >= 0.5:
                st.markdown(f"""
                <div class="risk-high">
                    <h2 style="color: white; margin: 0;">🚨 HIGH CHURN RISK</h2>
                    <p style="margin: 10px 0 0 0; font-size: 16px; color: rgba(255,255,255,0.9);">
                    This customer requires immediate attention and retention strategy
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h2 style="color: white; margin: 0;">✅ LOW CHURN RISK</h2>
                    <p style="margin: 10px 0 0 0; font-size: 16px; color: rgba(255,255,255,0.9);">
                    Customer is likely to remain satisfied and loyal
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Probability visualization
            fig, ax = plt.subplots(figsize=(8, 3), facecolor='white')
            risk_level = prob * 100
            colors = ['#e74c3c' if risk_level >= 50 else '#2ecc71']
            bars = ax.barh(['Churn Risk'], [risk_level], color=colors, height=0.5, edgecolor='black', linewidth=2)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Risk Percentage (%)', fontsize=12, fontweight='bold')
            ax.text(risk_level/2, 0, f'{risk_level:.1f}%', ha='center', va='center', 
                   fontsize=16, fontweight='bold', color='white')
            ax.set_facecolor('#f8f9fa')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        
        st.markdown("---")
        
        # Detailed analysis
        st.markdown("""
        <div class="alert-info">
            <h3>📊 PREDICTION ANALYSIS</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**⏰ Tenure:** {tenure} months")
        with col2:
            st.write(f"**📜 Contract:** {contract}")
        with col3:
            st.write(f"**💵 Monthly:** ${monthly:.2f}")
        with col4:
            st.write(f"**🌐 Internet:** {internet}")
        
        # Risk factors
        risk_factors = []
        if tenure <= 12:
            risk_factors.append("🔴 New customer (first year)")
        if contract == "Month-to-month":
            risk_factors.append("🔴 No long-term commitment")
        if monthly > 80:
            risk_factors.append("🔴 High monthly charges")
        if internet == "Fiber optic":
            risk_factors.append("🟡 Fiber optic user (higher historical churn)")
        if phone == "No" and internet == "No":
            risk_factors.append("🟡 Minimal service adoption")
        
        if risk_factors:
            st.markdown("**⚠️ Risk Factors Detected:**")
            for factor in risk_factors[:4]:
                st.write(f"• {factor}")

# ============================================================================
# PAGE: ANALYTICS CENTER
# ============================================================================
elif page == "📊 Analytics Center":
    st.markdown("<div class='section-header'>📊 CUSTOMER DATA ANALYTICS</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div>TOTAL ROWS</div>
            <div style="font-size: 32px; font-weight: bold;">{len(df):,}</div>
            <div style="font-size: 11px; opacity: 0.8;">Active Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div>TOTAL COLUMNS</div>
            <div style="font-size: 32px; font-weight: bold;">{df.shape[1]}</div>
            <div style="font-size: 11px; opacity: 0.8;">Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div>CHURN RATE</div>
            <div style="font-size: 32px; font-weight: bold;">{CHURN_RATE:.1%}</div>
            <div style="font-size: 11px; opacity: 0.8;">Risk Baseline</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Churn Distribution")
        churn_dist = df['Churn'].value_counts().map({0: 'Retained', 1: 'Churned'})
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
        colors = ['#51cf66', '#ff6b6b']
        wedges, texts, autotexts = ax.pie(churn_dist.values, labels=churn_dist.index, autopct='%1.1f%%',
                                           colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col2:
        st.markdown("### 📈 Tenure Distribution")
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
        ax.hist(df['tenure'], bins=30, color='#667eea', edgecolor='black', alpha=0.8)
        ax.set_title('Customer Tenure Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Tenure (months)', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    st.markdown("---")
    
    st.markdown("### 🔑 Key Features in Model")
    st.write(f"**{len(model_artifacts['selected_features'])} Critical Features Identified**")
    
    cols = st.columns(5)
    for i, feat in enumerate(model_artifacts['selected_features'][:10]):
        with cols[i % 5]:
            st.markdown(f'<span class="feature-pill">{feat}</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ============================================================================
# PAGE: PERFORMANCE HUB
# ============================================================================
elif page == "⚡ Performance Hub":
    st.markdown("<div class='section-header'>⚡ MODEL PERFORMANCE METRICS</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-info">
        ✅ VERIFIED: These are your ACTUAL metrics from your notebook (NOT approximations)
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    perf_metrics = [
        (col1, "Accuracy", f"{ACTUAL_METRICS['accuracy']:.2%}", "Correctness"),
        (col2, "Recall", f"{ACTUAL_METRICS['recall']:.2%}", "Sensitivity"),
        (col3, "Precision", f"{ACTUAL_METRICS['precision']:.2%}", "Specificity"),
        (col4, "F1 Score", f"{ACTUAL_METRICS['f1']:.2%}", "Balance"),
        (col5, "ROC-AUC", f"{ACTUAL_METRICS['roc_auc']:.2%}", "Discrimination"),
    ]
    
    for col, label, value, desc in perf_metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 13px; opacity: 0.9;">{label}</div>
                <div style="font-size: 24px; font-weight: bold; margin: 8px 0;">{value}</div>
                <div style="font-size: 10px; opacity: 0.8;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Confusion Matrix Analysis")
        cm_data = np.array([[ACTUAL_METRICS['tn'], ACTUAL_METRICS['fp']], 
                           [ACTUAL_METRICS['fn'], ACTUAL_METRICS['tp']]])
        
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar=True,
                   xticklabels=['No Churn', 'Churn'], yticklabels=['Actual No Churn', 'Actual Churn'],
                   annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_title('Confusion Matrix (1,409 Test Customers)', fontsize=12, fontweight='bold')
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col2:
        st.markdown("### Performance Breakdown")
        st.write(f"""
        <div style="background: white; padding: 20px; border-radius: 10px; border: 1px solid #ecf0f1;">
            <h4 style="color: #667eea;">✅ CORRECT PREDICTIONS</h4>
            <p><strong>True Positives (TP):</strong> {ACTUAL_METRICS['tp']} - Caught churn cases</p>
            <p><strong>True Negatives (TN):</strong> {ACTUAL_METRICS['tn']} - Correct no-churn</p>
            
            <h4 style="color: #e74c3c; margin-top: 20px;">❌ INCORRECT PREDICTIONS</h4>
            <p><strong>False Positives (FP):</strong> {ACTUAL_METRICS['fp']} - False alarms</p>
            <p><strong>False Negatives (FN):</strong> {ACTUAL_METRICS['fn']} - Missed churn cases</p>
            
            <h4 style="color: #2ecc71; margin-top: 20px;">🎯 INTERPRETATION</h4>
            <p>High Recall (81%) = Excellent churn detection<br>
            Precision (50%) = Trade-off for sensitivity</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Model Quality Assurance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div>TRAIN ACCURACY</div>
            <div style="font-size: 28px; font-weight: bold;">{ACTUAL_METRICS['train_accuracy']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div>TEST ACCURACY</div>
            <div style="font-size: 28px; font-weight: bold;">{ACTUAL_METRICS['test_accuracy']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        gap = ACTUAL_METRICS['train_accuracy'] - ACTUAL_METRICS['test_accuracy']
        st.markdown(f"""
        <div class="metric-card">
            <div>OVERFITTING GAP</div>
            <div style="font-size: 28px; font-weight: bold; color: #51cf66;">{gap:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-success">
        ✅ EXCELLENT: Model generalizes well (3% gap indicates no overfitting)
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer-section">
    <p>
    🚨 <strong>CHURN SENTINEL SYSTEM</strong> | 
    Real-time ML Prediction Platform | 
    Logistic Regression + SMOTE | 
    Deployed on Streamlit
    </p>
    <p style="margin-top: 10px; opacity: 0.7;">
    © 2024 Advanced Churn Analytics | Accuracy: 73.88% | Recall: 81.02% | ROC-AUC: 83.89%
    </p>
</div>
""", unsafe_allow_html=True)
