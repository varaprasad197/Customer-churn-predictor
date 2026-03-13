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

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
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
# LOAD & TRAIN MODEL
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
st.title("📊 Telecom Customer Churn Predictor")

df = load_data()
if df is None:
    st.stop()

with st.spinner("🔄 Initializing model..."):
    model_artifacts = train_model(df)

st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("", ["🏠 Dashboard", "🎯 Make Prediction", "📈 Performance", "📊 Analytics"])

# ============================================================================
# PAGE: DASHBOARD (HOME)
# ============================================================================
if page == "🏠 Dashboard":
    # Hero Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <h2 style="color: #667eea; margin-top: 0;">Welcome to Churn Analytics</h2>
        <p style="font-size: 16px; color: #555;">
        Predict customer churn with machine learning. Our model analyzes telecom customer data 
        to identify at-risk customers and help you take proactive retention measures.
        </p>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Customers", f"{TOTAL_CUSTOMERS:,}", "📊")
    with col2:
        st.metric("Churn Rate", f"{CHURN_RATE:.1%}", "📉")
    with col3:
        st.metric("Accuracy", f"{ACTUAL_METRICS['accuracy']:.2%}", "✅")
    with col4:
        st.metric("Recall", f"{ACTUAL_METRICS['recall']:.2%}", "🎯")
    with col5:
        st.metric("ROC-AUC", f"{ACTUAL_METRICS['roc_auc']:.2%}", "📈")
    
    st.markdown("---")
    
    # Key Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 Model Insights
        
        **What the model predicts:**
        - Identifies high-risk customers likely to churn
        - Analyzes 29 customer features (reduced to 22 key ones)
        - Uses Logistic Regression with SMOTE balancing
        
        **Key Performance:**
        - Catches **81%** of actual churners (High Recall)
        - When it predicts churn, **50.5%** are correct (Precision)
        - Great at pattern recognition (ROC-AUC: 83.89%)
        """)
    
    with col2:
        st.markdown("""
        ### 💡 How to Use
        
        1. **🎯 Make Prediction** - Enter customer details
        2. **📈 Performance** - See model metrics & confusion matrix
        3. **📊 Analytics** - Explore customer data patterns
        
        **Tips:**
        - Use predictions with domain expertise
        - Monitor retention efforts
        - Adjust strategies based on customer segments
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
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
            senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
            partner = st.selectbox("Partner Status", ["No", "Yes"], key="partner")
            dependents = st.selectbox("Dependents", ["No", "Yes"], key="dependents")
        
        with col2:
            st.subheader("📱 Services")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["No", "Yes"], key="phone")
            internet = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"], key="internet")
            security = st.selectbox("Online Security", ["No", "Yes"], key="security")
            backup = st.selectbox("Online Backup", ["No", "Yes"], key="backup")
        
        with col3:
            st.subheader("💳 Billing & Services")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
            device = st.selectbox("Device Protection", ["No", "Yes"], key="device")
            tech = st.selectbox("Tech Support", ["No", "Yes"], key="tech")
            payment = st.selectbox("Payment Method", 
                                 ["Electronic check", "Mailed check", 
                                  "Bank transfer (automatic)", "Credit card (automatic)"], key="payment")
        
        col1, col2 = st.columns(2)
        with col1:
            monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0, step=5.0)
        with col2:
            total = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0, step=50.0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"], key="tv")
        with col2:
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"], key="movies")
        with col3:
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"], key="paperless")
        
        # Submit button
        submit = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
    
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
        
        # Result Display
        col1, col2 = st.columns(2)
        
        with col1:
            if prob >= 0.5:
                st.markdown("""
                <div class="prediction-high-risk">
                    <h3 style="color: #ff4444; margin: 0;">⚠️ HIGH CHURN RISK</h3>
                    <p style="margin: 10px 0; font-size: 16px;">This customer is at risk</p>
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
            # Probability gauge
            st.metric("Churn Probability", f"{prob:.1%}", 
                     delta=f"Risk Level: {'HIGH' if prob >= 0.5 else 'LOW'}",
                     delta_color="inverse")
        
        st.markdown("---")
        
        # Insights
        st.markdown("""
        ### 📊 Prediction Details
        """)
        
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
# PAGE: MODEL PERFORMANCE
# ============================================================================
elif page == "📈 Performance":
    st.markdown("<h2 class='section-header'>📈 Model Performance Metrics</h2>", unsafe_allow_html=True)
    
    st.info("✅ These are your ACTUAL metrics from your notebook (not inflated)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{ACTUAL_METRICS['accuracy']:.2%}", "Overall Correctness")
    col2.metric("Recall", f"{ACTUAL_METRICS['recall']:.2%}", "Catches Churners")
    col3.metric("Precision", f"{ACTUAL_METRICS['precision']:.2%}", "Prediction Accuracy")
    col4.metric("F1 Score", f"{ACTUAL_METRICS['f1']:.2%}", "Balanced Metric")
    col5.metric("ROC-AUC", f"{ACTUAL_METRICS['roc_auc']:.2%}", "Discrimination")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Confusion Matrix")
        cm_data = np.array([[ACTUAL_METRICS['tn'], ACTUAL_METRICS['fp']], 
                           [ACTUAL_METRICS['fn'], ACTUAL_METRICS['tp']]])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                   xticklabels=['No Churn', 'Churn'], yticklabels=['Actual No Churn', 'Actual Churn'])
        ax.set_title('Confusion Matrix (1,409 Test Customers)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col2:
        st.markdown("### Model Breakdown")
        st.write(f"""
        **✅ Correct Predictions:**
        - True Positives (TP): **{ACTUAL_METRICS['tp']}** - Correctly predicted churn
        - True Negatives (TN): **{ACTUAL_METRICS['tn']}** - Correctly predicted no-churn
        
        **❌ Incorrect Predictions:**
        - False Positives (FP): **{ACTUAL_METRICS['fp']}** - Predicted churn but didn't
        - False Negatives (FN): **{ACTUAL_METRICS['fn']}** - Predicted no-churn but did
        
        **Trade-off:** High recall (81%) means we catch most churners but have some false alarms
        """)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Model Quality Checks")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Train Accuracy", f"{ACTUAL_METRICS['train_accuracy']:.2%}")
    with col2:
        st.metric("Test Accuracy", f"{ACTUAL_METRICS['test_accuracy']:.2%}")
    with col3:
        gap = ACTUAL_METRICS['train_accuracy'] - ACTUAL_METRICS['test_accuracy']
        st.metric("Overfitting Gap", f"{gap:.2%}", "✅ Good")
    
    st.success("✅ Model generalizes well with no overfitting (3% gap is excellent)")

# ============================================================================
# PAGE: ANALYTICS
# ============================================================================
elif page == "📊 Analytics":
    st.markdown("<h2 class='section-header'>📊 Customer Data Analytics</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Churn Rate", f"{CHURN_RATE:.1%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Churn Distribution")
        churn_dist = df['Churn'].value_counts().map({0: 'Retained', 1: 'Churned'})
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71', '#e74c3c']
        churn_dist.plot(kind='bar', ax=ax, color=colors, edgecolor='black', alpha=0.7)
        ax.set_title('Customer Churn Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        ax.set_xlabel('')
        plt.xticks(rotation=0)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col2:
        st.markdown("### Tenure Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['tenure'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        ax.set_title('Customer Tenure Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Tenure (months)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    st.markdown("---")
    
    st.markdown("### 🔑 Selected Features")
    st.write(f"**{len(model_artifacts['selected_features'])} features selected by the model**")
    
    # Display features in a nice grid
    cols = st.columns(4)
    for i, feat in enumerate(model_artifacts['selected_features']):
        with cols[i % 4]:
            st.write(f"✅ **{feat}**")
    
    st.markdown("---")
    
    st.markdown("### 📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 12px; padding: 20px;">
📊 Telecom Customer Churn Predictor | 
Logistic Regression + SMOTE | 
Actual Metrics (No Inflated Data) | 
Deployed with Streamlit
</div>
""", unsafe_allow_html=True)
