import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import joblib
import os
import io
from collections import Counter

warnings.filterwarnings("ignore")

# ─── Sklearn / Imbalanced ───────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
from scipy.stats import loguniform, randint, uniform
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLES
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] {background: #0f1117;}
[data-testid="stSidebar"] * {color: #e0e6f0 !important;}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg,#1e2235,#252b40);
    border: 1px solid #3a4060;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,.35);
}
div[data-testid="metric-container"] label {color:#8b9dc3 !important; font-size:.8rem; font-weight:600; letter-spacing:.06em;}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {color:#e8f0ff !important; font-size:1.8rem; font-weight:700;}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {font-size:.8rem;}

/* ── Tab styling ── */
button[data-baseweb="tab"] {font-weight:600; font-size:.9rem; letter-spacing:.03em;}

/* ── Section headers ── */
.section-title {
    font-size:1.35rem; font-weight:700; color:#6c8cef;
    border-left:4px solid #6c8cef; padding-left:12px; margin:20px 0 10px;
}
.sub-title {font-size:1rem; font-weight:600; color:#a0b0d0; margin:14px 0 6px;}

/* ── Risk badge ── */
.risk-high  {background:#ff4b4b22;border:1px solid #ff4b4b;color:#ff4b4b;border-radius:8px;padding:8px 16px;font-weight:700;display:inline-block;}
.risk-low   {background:#00c85322;border:1px solid #00c853;color:#00c853;border-radius:8px;padding:8px 16px;font-weight:700;display:inline-block;}

/* ── Info box ── */
.info-box {background:#1a2035;border:1px solid #3a4a6b;border-radius:10px;padding:16px 20px;margin:10px 0;}

/* ── Divider ── */
hr {border-color:#2a3050 !important;}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HELPERS / CACHED FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TenureGroup"] = pd.cut(df["tenure"], bins=[0,12,24,48,float("inf")],
                               labels=["New","Regular","Established","Loyal"])
    df["IsFirstYear"]  = (df["tenure"] <= 12).astype(int)
    df["IsLongTerm"]   = (df["tenure"] >= 24).astype(int)

    df["AvgMonthlyCharge"] = df.apply(
        lambda x: x["TotalCharges"]/x["tenure"] if x["tenure"]>0 else x["MonthlyCharges"], axis=1)
    df["CustomerLTV"] = df["TotalCharges"] + df["MonthlyCharges"]*6

    additional = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                  "TechSupport","StreamingTV","StreamingMovies"]
    df["NumAdditionalServices"] = df[additional].apply(lambda x:(x=="Yes").sum(), axis=1)
    df["HasSecurityBundle"]  = ((df["OnlineSecurity"]=="Yes")&(df["OnlineBackup"]=="Yes")).astype(int)
    df["HasStreamingBundle"] = ((df["StreamingTV"]=="Yes")&(df["StreamingMovies"]=="Yes")).astype(int)
    df["InternetUser"]       = (df["InternetService"]!="No").astype(int)
    df["FiberOpticUser"]     = (df["InternetService"]=="Fiber optic").astype(int)
    df["ServicesPerMonth"]   = df["NumAdditionalServices"]/(df["tenure"]+1)

    df["IsMonthToMonth"] = (df["Contract"]=="Month-to-month").astype(int)
    df["ContractType"]   = df["Contract"].map({"Month-to-month":0,"One year":1,"Two year":2})
    df["ElectronicPayment"] = df["PaymentMethod"].str.contains("electronic check|automatic",
                                                                case=False, na=False).astype(int)
    df["PaymentRisk"] = df["PaymentMethod"].map({
        "Electronic check":3,"Mailed check":2,
        "Bank transfer (automatic)":1,"Credit card (automatic)":1})

    for col in ["PaperlessBilling","Partner","Dependents"]:
        if df[col].dtype == "object":
            df[col] = df[col].map({"Yes":1,"No":0})

    df["PaperlessHighRisk"] = ((df["PaperlessBilling"]==1)&(df["PaymentMethod"]=="Electronic check")).astype(int)
    df["HasFamily"]  = ((df["Partner"]==1)|(df["Dependents"]==1)).astype(int)
    df["HighCostLowTenure"] = (
        (df["MonthlyCharges"]>df["MonthlyCharges"].median())&(df["tenure"]<12)).astype(int)
    df["EngagementScore"] = (
        df["NumAdditionalServices"]*0.3 +
        df["ContractType"]*0.4 +
        (df["tenure"]/df["tenure"].max())*0.3)
    return df


@st.cache_resource(show_spinner=False)
def train_pipeline(df: pd.DataFrame):
    df_fe = feature_engineering(df)
    X = df_fe.drop("Churn", axis=1)
    y = df_fe["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_enc = pd.get_dummies(X_train, drop_first=True)
    X_test_enc  = pd.get_dummies(X_test,  drop_first=True)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="outer", axis=1, fill_value=0)
    X_train_enc = X_train_enc.astype(float)
    X_test_enc  = X_test_enc.astype(float)
    encoded_cols = X_train_enc.columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled  = scaler.transform(X_test_enc)

    l1_sel = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                max_iter=1000, random_state=42)
    l1_sel.fit(X_train_scaled, y_train)
    l1_mask = l1_sel.coef_[0] != 0
    selected_features = np.array(encoded_cols)[l1_mask]

    X_train_sel = X_train_scaled[:, l1_mask]
    X_test_sel  = X_test_scaled[:,  l1_mask]

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                           random_state=42, n_jobs=-1)

    results = []
    trained = {}
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        trained[name] = model
        y_prob = model.predict_proba(X_test_sel)[:,1]
        y_pred = (y_prob>=0.5).astype(int)
        results.append({
            "Model": name,
            "Accuracy":  round(accuracy_score(y_test, y_pred),4),
            "Precision": round(precision_score(y_test, y_pred),4),
            "Recall":    round(recall_score(y_test, y_pred),4),
            "F1":        round(f1_score(y_test, y_pred),4),
            "ROC-AUC":   round(roc_auc_score(y_test, y_prob),4),
        })

    results_df = pd.DataFrame(results).set_index("Model")
    best_name  = results_df["ROC-AUC"].idxmax()
    best_model = trained[best_name]

    y_prob_best = best_model.predict_proba(X_test_sel)[:,1]
    thresholds  = np.arange(0.1,0.91,0.05)
    cost_rows   = []
    for t in thresholds:
        yp = (y_prob_best>=t).astype(int)
        tn,fp,fn,tp = confusion_matrix(y_test, yp).ravel()
        cost_rows.append({
            "Threshold": round(t,2),
            "Precision": precision_score(y_test,yp,zero_division=0),
            "Recall":    recall_score(y_test,yp),
            "F1":        f1_score(y_test,yp),
            "BusinessCost": fp*100 + fn*500,
            "TN":tn,"FP":fp,"FN":fn,"TP":tp,
        })
    thresh_df   = pd.DataFrame(cost_rows)
    optimal_thr = thresh_df.loc[thresh_df["BusinessCost"].idxmin(),"Threshold"]

    return {
        "trained_models":    trained,
        "best_model_name":   best_name,
        "best_model":        best_model,
        "results_df":        results_df,
        "scaler":            scaler,
        "encoded_cols":      encoded_cols,
        "selected_features": selected_features,
        "l1_mask":           l1_mask,
        "l1_coefs":          l1_sel.coef_[0],
        "X_test_sel":        X_test_sel,
        "X_test_sel_df":     pd.DataFrame(X_test_sel, columns=selected_features),
        "X_train_res":       X_train_res,
        "y_test":            y_test,
        "y_prob_best":       y_prob_best,
        "thresh_df":         thresh_df,
        "optimal_threshold": optimal_thr,
        "df_fe":             df_fe,
    }


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return buf


PAL = ["#6c8cef","#f0883e","#3dd56d","#e45858","#a78bfa","#fbbf24"]

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📡 Churn Predictor")
    st.markdown("---")

    data_source = st.radio("Data source", ["Upload CSV", "GitHub URL"], index=1)

    df_raw = None
    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload tele_comm.csv", type="csv")
        if uploaded:
            df_raw = load_data(uploaded)
    else:
        default_url = ("https://github.com/varaprasad197/Customer-churn-predictor/blob/main/tele_comm.csv")
        url = st.text_input("CSV URL", value=default_url)
        if st.button("🚀 Load & Train", use_container_width=True):
            with st.spinner("Fetching dataset …"):
                try:
                    df_raw = load_data(url)
                    st.session_state["df_raw"] = df_raw
                except Exception as e:
                    st.error(f"Failed to load: {e}")

    if "df_raw" in st.session_state:
        df_raw = st.session_state["df_raw"]

    if df_raw is not None:
        st.success(f"✅ {len(df_raw):,} rows loaded")

    st.markdown("---")
    st.markdown("### ⚙️ Business Cost")
    fp_cost = st.slider("False Positive cost ($)", 50, 500, 100, 50,
                        help="Cost of offering retention to non-churner")
    fn_cost = st.slider("False Negative cost ($)", 100, 1000, 500, 50,
                        help="Cost of missing a churner")
    st.markdown("---")
    st.caption("Built with ❤️ · Streamlit · sklearn · SHAP")

# ════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════════════
st.title("📡 Telecom Customer Churn — ML Intelligence Hub")
st.markdown(
    "<div class='info-box'>End-to-end churn ML pipeline: EDA → Feature Engineering → "
    "SMOTE → 4-model comparison → Threshold tuning → SHAP interpretability → "
    "Live prediction.</div>", unsafe_allow_html=True)

if df_raw is None:
    st.info("👈  Load your dataset from the sidebar to get started.")
    st.stop()

# ── Train pipeline ──────────────────────────────────────────────────────────
with st.spinner("🔄 Running full ML pipeline … (first run may take ~30 s)"):
    pipe = train_pipeline(df_raw)

results_df        = pipe["results_df"]
best_name         = pipe["best_model_name"]
best_model        = pipe["best_model"]
trained_models    = pipe["trained_models"]
scaler            = pipe["scaler"]
encoded_cols      = pipe["encoded_cols"]
selected_features = pipe["selected_features"]
l1_mask           = pipe["l1_mask"]
l1_coefs          = pipe["l1_coefs"]
X_test_sel        = pipe["X_test_sel"]
X_test_sel_df     = pipe["X_test_sel_df"]
X_train_res       = pipe["X_train_res"]
y_test            = pipe["y_test"]
y_prob_best       = pipe["y_prob_best"]
thresh_df         = pipe["thresh_df"]
optimal_threshold = pipe["optimal_threshold"]
df_fe             = pipe["df_fe"]

# Recompute business cost with sidebar sliders
thresh_df["BusinessCost"] = thresh_df["FP"]*fp_cost + thresh_df["FN"]*fn_cost
optimal_threshold = thresh_df.loc[thresh_df["BusinessCost"].idxmin(),"Threshold"]

# ════════════════════════════════════════════════════════════════════════════
# KPI STRIP
# ════════════════════════════════════════════════════════════════════════════
k1,k2,k3,k4,k5 = st.columns(5)
y_final = (y_prob_best >= optimal_threshold).astype(int)
k1.metric("🏆 Best Model",   best_name)
k2.metric("ROC-AUC",         f"{results_df.loc[best_name,'ROC-AUC']:.4f}")
k3.metric("Recall @ Opt.Thr",f"{recall_score(y_test,y_final):.4f}")
k4.metric("Optimal Threshold",f"{optimal_threshold}")
k5.metric("Min Business Cost",f"${thresh_df['BusinessCost'].min():,.0f}")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 EDA",
    "🔬 Feature Analysis",
    "🤖 Model Comparison",
    "🎯 Threshold Tuning",
    "🧠 SHAP Explainability",
    "🔮 Live Prediction",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("<div class='section-title'>Exploratory Data Analysis</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # Churn distribution
    with c1:
        st.markdown("<div class='sub-title'>Target Distribution</div>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1,2,figsize=(8,3.5), facecolor="#0e1117")
        counts = df_raw["Churn"].value_counts()
        axes[0].bar(["No Churn","Churn"], counts.values, color=[PAL[0],PAL[1]], edgecolor="#333")
        axes[0].set_facecolor("#161b27"); axes[0].tick_params(colors="#aaa")
        for sp in axes[0].spines.values(): sp.set_color("#333")
        for i,v in enumerate(counts.values):
            axes[0].text(i, v+30, str(v), ha="center", color="#eee", fontweight="bold")
        axes[1].pie(counts.values, labels=["No Churn","Churn"],
                    autopct="%1.1f%%", colors=[PAL[0],PAL[1]], startangle=90,
                    textprops={"color":"#ddd"})
        axes[1].set_facecolor("#161b27")
        fig.patch.set_facecolor("#0e1117")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Numerical distributions
    with c2:
        st.markdown("<div class='sub-title'>Numerical Feature by Churn</div>", unsafe_allow_html=True)
        num_col = st.selectbox("Select feature", ["tenure","MonthlyCharges","TotalCharges"], key="eda_num")
        fig, ax = plt.subplots(figsize=(6,3.5), facecolor="#0e1117")
        ax.set_facecolor("#161b27")
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.tick_params(colors="#aaa")
        df_raw[df_raw["Churn"]==0][num_col].hist(ax=ax, alpha=0.65, bins=30,
            label="No Churn", color=PAL[0])
        df_raw[df_raw["Churn"]==1][num_col].hist(ax=ax, alpha=0.65, bins=30,
            label="Churn",    color=PAL[1])
        ax.legend(facecolor="#1a2035", labelcolor="#ddd")
        ax.set_title(f"{num_col} Distribution", color="#ddd", fontweight="bold")
        fig.patch.set_facecolor("#0e1117")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Categorical churn rates
    st.markdown("<div class='sub-title'>Churn Rate by Categorical Feature</div>", unsafe_allow_html=True)
    cat_options = ["Contract","InternetService","PaymentMethod","TechSupport",
                   "OnlineSecurity","gender","SeniorCitizen"]
    cat_sel = st.selectbox("Select category", cat_options, key="cat_eda")
    fig, ax = plt.subplots(figsize=(10,3.5), facecolor="#0e1117")
    ax.set_facecolor("#161b27")
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.tick_params(colors="#aaa")
    rates = df_raw.groupby(cat_sel)["Churn"].mean().sort_values(ascending=False)
    bars  = ax.bar(rates.index.astype(str), rates.values, color=PAL[1], edgecolor="#444", alpha=0.88)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.2f}", ha="center", color="#ddd", fontsize=9)
    ax.set_ylabel("Churn Rate", color="#aaa")
    ax.set_ylim(0,1)
    plt.xticks(rotation=25, ha="right", color="#aaa")
    ax.set_title(f"Churn Rate by {cat_sel}", color="#ddd", fontweight="bold")
    fig.patch.set_facecolor("#0e1117")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Correlation heatmap
    st.markdown("<div class='sub-title'>Correlation Heatmap (Numeric)</div>", unsafe_allow_html=True)
    num_df = df_raw.select_dtypes(include=[np.number])
    corr   = num_df.corr()
    fig, ax = plt.subplots(figsize=(12,6), facecolor="#0e1117")
    ax.set_facecolor("#161b27")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, annot_kws={"size":8,"color":"#ddd"},
                ax=ax, cbar_kws={"shrink":0.8})
    ax.tick_params(colors="#aaa")
    ax.set_title("Correlation Matrix", color="#ddd", fontweight="bold")
    fig.patch.set_facecolor("#0e1117")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — FEATURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("<div class='section-title'>Feature Engineering & Selection</div>",
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class='info-box'>
    ✅ <b>{len(encoded_cols)}</b> features after one-hot encoding &nbsp;|&nbsp;
    ✅ <b>{l1_mask.sum()}</b> features selected via L1 (Lasso) regularisation &nbsp;|&nbsp;
    ❌ <b>{(~l1_mask).sum()}</b> features dropped
    </div>""", unsafe_allow_html=True)

    # L1 coefficient bar chart
    coef_df = pd.DataFrame({
        "Feature":     selected_features,
        "Coefficient": l1_coefs[l1_mask]
    }).sort_values("Coefficient", key=abs, ascending=False).head(25)

    fig, ax = plt.subplots(figsize=(10,7), facecolor="#0e1117")
    ax.set_facecolor("#161b27")
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.tick_params(colors="#aaa")
    colors = [PAL[1] if c>0 else PAL[0] for c in coef_df["Coefficient"]]
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="#333")
    ax.axvline(0, color="#555", linewidth=0.9)
    ax.set_title("Top 25 Features — L1 Coefficients", color="#ddd", fontweight="bold")
    ax.set_xlabel("Coefficient", color="#aaa")
    pos_patch = mpatches.Patch(color=PAL[1], label="Increases Churn Risk")
    neg_patch = mpatches.Patch(color=PAL[0], label="Decreases Churn Risk")
    ax.legend(handles=[pos_patch, neg_patch], facecolor="#1a2035", labelcolor="#ddd")
    fig.patch.set_facecolor("#0e1117")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Engineered feature distributions
    st.markdown("<div class='sub-title'>Engineered Feature Deep-Dive</div>", unsafe_allow_html=True)
    eng_features = ["EngagementScore","CustomerLTV","NumAdditionalServices",
                    "AvgMonthlyCharge","ServicesPerMonth"]
    sel_eng = st.selectbox("Explore engineered feature", eng_features)
    fig, ax = plt.subplots(figsize=(10,3.5), facecolor="#0e1117")
    ax.set_facecolor("#161b27")
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.tick_params(colors="#aaa")
    df_fe[df_fe["Churn"]==0][sel_eng].hist(ax=ax, bins=35, alpha=0.65, color=PAL[0], label="No Churn")
    df_fe[df_fe["Churn"]==1][sel_eng].hist(ax=ax, bins=35, alpha=0.65, color=PAL[1], label="Churn")
    ax.set_title(f"{sel_eng} by Churn", color="#ddd", fontweight="bold")
    ax.legend(facecolor="#1a2035", labelcolor="#ddd")
    fig.patch.set_facecolor("#0e1117")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("<div class='section-title'>Model Comparison</div>", unsafe_allow_html=True)

    # Styled results table
    def highlight_best(s):
        is_max = s == s.max()
        return ["background:#1e3a2f;color:#3dd56d;font-weight:700" if v else "" for v in is_max]

    st.dataframe(
        results_df.style.apply(highlight_best).format("{:.4f}"),
        use_container_width=True)

    # Metric bar chart
    metrics_list = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
    fig, ax = plt.subplots(figsize=(13,5), facecolor="#0e1117")
    ax.set_facecolor("#161b27")
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.tick_params(colors="#aaa")
    x   = np.arange(len(results_df))
    w   = 0.14
    for i,(metric,color) in enumerate(zip(metrics_list,PAL)):
        bars = ax.bar(x+i*w, results_df[metric], w, label=metric, color=color,
                      edgecolor="#333", alpha=0.88)
    ax.set_xticks(x + w*2)
    ax.set_xticklabels(results_df.index, rotation=15, ha="right", color="#aaa")
    ax.set_ylim(0,1.15)
    ax.set_title("Model Performance — Test Set", color="#ddd", fontweight="bold")
    ax.legend(facecolor="#1a2035", labelcolor="#ddd")
    fig.patch.set_facecolor("#0e1117")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    c1,c2 = st.columns(2)

    # ROC curves
    with c1:
        st.markdown("<div class='sub-title'>ROC-AUC Curves</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,5), facecolor="#0e1117")
        ax.set_facecolor("#161b27")
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.tick_params(colors="#aaa")
        for (name,model),color in zip(trained_models.items(), PAL):
            yp = model.predict_proba(X_test_sel)[:,1]
            fpr,tpr,_ = roc_curve(y_test,yp)
            ax.plot(fpr,tpr, color=color, lw=2, label=f"{name} ({auc(fpr,tpr):.3f})")
        ax.plot([0,1],[0,1],"--",color="#555",lw=1)
        ax.set_xlabel("FPR",color="#aaa"); ax.set_ylabel("TPR",color="#aaa")
        ax.set_title("ROC Curves",color="#ddd",fontweight="bold")
        ax.legend(facecolor="#1a2035",labelcolor="#ddd",fontsize=8)
        fig.patch.set_facecolor("#0e1117")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Confusion matrices
    with c2:
        st.markdown("<div class='sub-title'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm_model = st.selectbox("Select model", list(trained_models.keys()))
        yp_cm = trained_models[cm_model].predict(X_test_sel)
        cm    = confusion_matrix(y_test, yp_cm)
        fig, ax = plt.subplots(figsize=(5,4), facecolor="#0e1117")
        ax.set_facecolor("#161b27")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Churn","Churn"],
                    yticklabels=["No Churn","Churn"],
                    annot_kws={"size":13,"weight":"bold"})
        ax.tick_params(colors="#aaa")
        ax.set_xlabel("Predicted",color="#aaa"); ax.set_ylabel("Actual",color="#aaa")
        ax.set_title(f"{cm_model}",color="#ddd",fontweight="bold")
        fig.patch.set_facecolor("#0e1117")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Precision-Recall curves
    st.markdown("<div class='sub-title'>Precision-Recall Curves</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10,4), facecolor="#0e1117")
    ax.set_facecolor("#161b27")
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.tick_params(colors="#aaa")
    for (name,model),color in zip(trained_models.items(),PAL):
        yp = model.predict_proba(X_test_sel)[:,1]
        prec,rec,_ = precision_recall_curve(y_test,yp)
        ax.plot(rec,prec,color=color,lw=2,label=f"{name} ({auc(rec,prec):.3f})")
    ax.axhline(y_test.mean(),color="#555",linestyle="--",label=f"Baseline ({y_test.mean():.2f})")
    ax.set_xlabel("Recall",color="#aaa"); ax.set_ylabel("Precision",color="#aaa")
    ax.set_title("Precision-Recall Curves",color="#ddd",fontweight="bold")
    ax.legend(facecolor="#1a2035",labelcolor="#ddd",fontsize=9)
    fig.patch.set_facecolor("#0e1117")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — THRESHOLD TUNING
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("<div class='section-title'>Threshold Tuning & Business Cost</div>",
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class='info-box'>
    💡 <b>FP cost</b> = ${fp_cost} (unnecessary retention offer) &nbsp;|&nbsp;
    💡 <b>FN cost</b> = ${fn_cost} (missed churner) &nbsp;|&nbsp;
    ⚡ <b>Optimal threshold</b> = <b>{optimal_threshold}</b> &nbsp;|&nbsp;
    💰 <b>Min cost</b> = <b>${thresh_df["BusinessCost"].min():,.0f}</b>
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)

    with c1:
        st.markdown("<div class='sub-title'>Precision / Recall / F1 vs Threshold</div>",
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#0e1117")
        ax.set_facecolor("#161b27")
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.tick_params(colors="#aaa")
        ax.plot(thresh_df["Threshold"], thresh_df["Precision"], "o-", color=PAL[0], label="Precision", ms=5)
        ax.plot(thresh_df["Threshold"], thresh_df["Recall"],    "o-", color=PAL[1], label="Recall",    ms=5)
        ax.plot(thresh_df["Threshold"], thresh_df["F1"],        "o-", color=PAL[2], label="F1",        ms=5)
        best_f1_t = thresh_df.loc[thresh_df["F1"].idxmax(),"Threshold"]
        ax.axvline(best_f1_t, color="#aaa", linestyle="--", label=f"Best F1 @ {best_f1_t}")
        ax.legend(facecolor="#1a2035", labelcolor="#ddd")
        ax.set_xlabel("Threshold",color="#aaa"); ax.set_ylabel("Score",color="#aaa")
        ax.set_title(f"Threshold Tuning — {best_name}",color="#ddd",fontweight="bold")
        ax.grid(color="#2a3050",linewidth=0.5)
        fig.patch.set_facecolor("#0e1117")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c2:
        st.markdown("<div class='sub-title'>Business Cost vs Threshold</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#0e1117")
        ax.set_facecolor("#161b27")
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.tick_params(colors="#aaa")
        ax.plot(thresh_df["Threshold"], thresh_df["BusinessCost"], "o-", color=PAL[3], ms=5)
        ax.axvline(optimal_threshold, color=PAL[2], linestyle="--",
                   label=f"Min Cost @ {optimal_threshold}")
        ax.fill_between(thresh_df["Threshold"], thresh_df["BusinessCost"],
                        thresh_df["BusinessCost"].min(), alpha=0.12, color=PAL[3])
        ax.set_xlabel("Threshold",color="#aaa"); ax.set_ylabel("Total Cost ($)",color="#aaa")
        ax.set_title("Business Cost vs Threshold",color="#ddd",fontweight="bold")
        ax.legend(facecolor="#1a2035",labelcolor="#ddd")
        ax.grid(color="#2a3050",linewidth=0.5)
        fig.patch.set_facecolor("#0e1117")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("<div class='sub-title'>Threshold Details Table</div>", unsafe_allow_html=True)
    disp_cols = ["Threshold","Precision","Recall","F1","BusinessCost","TN","FP","FN","TP"]
    st.dataframe(
        thresh_df[disp_cols].style
            .highlight_min(subset=["BusinessCost"], color="#1e3a2f")
            .highlight_max(subset=["F1"],           color="#1e2a3a")
            .format({"Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}",
                     "BusinessCost":"${:,.0f}"}),
        use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — SHAP
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("<div class='section-title'>SHAP Model Explainability</div>", unsafe_allow_html=True)

    if not SHAP_AVAILABLE:
        st.warning("Install `shap` to enable this tab:  `pip install shap`")
    else:
        @st.cache_resource(show_spinner=False)
        def compute_shap(_best_model, _X_test_df, _X_train_res, _best_name, _selected_features):
            if _best_name == "Logistic Regression":
                X_tr_df = pd.DataFrame(_X_train_res, columns=_selected_features)
                explainer = shap.LinearExplainer(_best_model, X_tr_df)
            else:
                explainer = shap.TreeExplainer(_best_model)
            sv = explainer.shap_values(_X_test_df)
            if isinstance(sv, list):
                sv = sv[1]
            return explainer, sv

        with st.spinner("Computing SHAP values …"):
            explainer, shap_vals = compute_shap(
                best_model, X_test_sel_df, X_train_res,
                best_name, list(selected_features))

        shap_importance = np.abs(shap_vals).mean(axis=0)
        top_idx         = np.argsort(shap_importance)[::-1][:15]

        c1,c2 = st.columns(2)

        with c1:
            st.markdown("<div class='sub-title'>Feature Importance (Mean |SHAP|)</div>",
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6,6), facecolor="#0e1117")
            ax.set_facecolor("#161b27")
            for sp in ax.spines.values(): sp.set_color("#333")
            ax.tick_params(colors="#aaa")
            feats  = np.array(selected_features)[top_idx]
            imps   = shap_importance[top_idx]
            ax.barh(feats[::-1], imps[::-1], color=PAL[0], edgecolor="#333")
            ax.set_xlabel("Mean |SHAP|",color="#aaa")
            ax.set_title(f"SHAP Importance — {best_name}",color="#ddd",fontweight="bold")
            fig.patch.set_facecolor("#0e1117")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c2:
            st.markdown("<div class='sub-title'>SHAP Dependence Plot</div>", unsafe_allow_html=True)
            dep_feat = st.selectbox("Feature", list(selected_features[:20]), key="shap_dep")
            fidx = list(selected_features).index(dep_feat)
            fig, ax = plt.subplots(figsize=(6,5), facecolor="#0e1117")
            ax.set_facecolor("#161b27")
            for sp in ax.spines.values(): sp.set_color("#333")
            ax.tick_params(colors="#aaa")
            sc = ax.scatter(X_test_sel_df[dep_feat], shap_vals[:,fidx],
                            c=shap_vals[:,fidx], cmap="RdYlBu_r", alpha=0.5, s=12)
            ax.axhline(0, color="#555", linestyle="--", linewidth=0.8)
            ax.set_xlabel(dep_feat,color="#aaa"); ax.set_ylabel("SHAP Value",color="#aaa")
            ax.set_title(f"Dependence: {dep_feat}",color="#ddd",fontweight="bold")
            plt.colorbar(sc, ax=ax, label="SHAP Value")
            fig.patch.set_facecolor("#0e1117")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Waterfall for highest-risk customer
        st.markdown("<div class='sub-title'>Waterfall — Highest-Risk Customer</div>",
                    unsafe_allow_html=True)
        hrisk_idx = int(np.argmax(y_prob_best))
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)): ev = ev[1] if len(ev)>1 else ev[0]

        shap_exp = shap.Explanation(
            values        = shap_vals[hrisk_idx],
            base_values   = float(ev),
            data          = X_test_sel_df.iloc[hrisk_idx].values,
            feature_names = list(selected_features)
        )
        fig, ax = plt.subplots(figsize=(10,6), facecolor="#0e1117")
        shap.waterfall_plot(shap_exp, max_display=15, show=False)
        plt.title(f"Waterfall — Predicted Churn Prob: {y_prob_best[hrisk_idx]:.3f}",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — LIVE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("<div class='section-title'>🔮 Live Customer Churn Prediction</div>",
                unsafe_allow_html=True)
    st.markdown("Fill in customer details below and hit **Predict** to get an instant churn risk score.")

    with st.form("pred_form"):
        c1,c2,c3 = st.columns(3)

        with c1:
            st.markdown("**Demographics**")
            gender         = st.selectbox("Gender",          ["Male","Female"])
            senior         = st.selectbox("Senior Citizen",  ["No","Yes"])
            partner        = st.selectbox("Partner",         ["No","Yes"])
            dependents     = st.selectbox("Dependents",      ["No","Yes"])

        with c2:
            st.markdown("**Account & Billing**")
            tenure         = st.slider("Tenure (months)", 0, 72, 12)
            contract       = st.selectbox("Contract",  ["Month-to-month","One year","Two year"])
            payment        = st.selectbox("Payment Method",
                                ["Electronic check","Mailed check",
                                 "Bank transfer (automatic)","Credit card (automatic)"])
            paperless      = st.selectbox("Paperless Billing", ["Yes","No"])
            monthly_chg    = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 1.0)
            total_chg      = st.number_input("Total Charges ($)",   0.0, 9000.0,
                                             float(monthly_chg*tenure), 10.0)

        with c3:
            st.markdown("**Services**")
            internet       = st.selectbox("Internet Service",    ["DSL","Fiber optic","No"])
            online_sec     = st.selectbox("Online Security",     ["No","Yes","No internet service"])
            online_bkp     = st.selectbox("Online Backup",       ["No","Yes","No internet service"])
            device_prot    = st.selectbox("Device Protection",   ["No","Yes","No internet service"])
            tech_sup       = st.selectbox("Tech Support",        ["No","Yes","No internet service"])
            stream_tv      = st.selectbox("Streaming TV",        ["No","Yes","No internet service"])
            stream_mov     = st.selectbox("Streaming Movies",    ["No","Yes","No internet service"])
            phone_svc      = st.selectbox("Phone Service",       ["Yes","No"])
            multi_lines    = st.selectbox("Multiple Lines",      ["No","Yes","No phone service"])

        submitted = st.form_submit_button("⚡ Predict Churn Risk", use_container_width=True)

    if submitted:
        # Build raw row
        raw_row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": 1 if senior=="Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_svc, "MultipleLines": multi_lines,
            "InternetService": internet, "OnlineSecurity": online_sec,
            "OnlineBackup": online_bkp, "DeviceProtection": device_prot,
            "TechSupport": tech_sup, "StreamingTV": stream_tv,
            "StreamingMovies": stream_mov, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_chg, "TotalCharges": total_chg,
            "Churn": 0,  # placeholder
        }])

        # Feature engineering
        row_fe = feature_engineering(raw_row)
        row_fe.drop(columns=["Churn"], inplace=True, errors="ignore")

        # Encode
        row_enc = pd.get_dummies(row_fe, drop_first=True)
        row_enc = row_enc.reindex(columns=encoded_cols, fill_value=0).astype(float)

        # Scale & select
        row_scaled = scaler.transform(row_enc)
        row_sel    = row_scaled[:, l1_mask]

        # Predict
        prob  = best_model.predict_proba(row_sel)[0,1]
        label = int(prob >= optimal_threshold)

        st.markdown("---")
        risk_class = "risk-high" if label==1 else "risk-low"
        risk_text  = "⚠️ HIGH CHURN RISK" if label==1 else "✅ LOW CHURN RISK"
        st.markdown(
            f"<div style='text-align:center;margin:20px 0'>"
            f"<span class='{risk_class}' style='font-size:1.4rem;padding:14px 30px'>"
            f"{risk_text}</span></div>", unsafe_allow_html=True)

        m1,m2,m3 = st.columns(3)
        m1.metric("Churn Probability",  f"{prob:.2%}")
        m2.metric("Decision Threshold", f"{optimal_threshold}")
        m3.metric("Prediction",         "Churn" if label else "No Churn")

        # Gauge-style probability bar
        fig, ax = plt.subplots(figsize=(8,1.4), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        ax.barh([0],[1], color="#1e2235", height=0.5)
        ax.barh([0],[prob], color="#ff4b4b" if prob>=optimal_threshold else "#3dd56d", height=0.5)
        ax.axvline(optimal_threshold, color="#fbbf24", linewidth=2,
                   label=f"Threshold = {optimal_threshold}")
        ax.set_xlim(0,1); ax.set_yticks([]); ax.set_xlabel("Churn Probability",color="#aaa")
        ax.legend(facecolor="#1a2035",labelcolor="#ddd",loc="upper right",fontsize=9)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(colors="#aaa")
        fig.patch.set_facecolor("#0e1117")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Top feature contributions
        if SHAP_AVAILABLE:
            try:
                sv_row = explainer.shap_values(
                    pd.DataFrame(row_sel, columns=selected_features))
                if isinstance(sv_row, list): sv_row = sv_row[1]
                contrib = pd.DataFrame({
                    "Feature":     selected_features,
                    "SHAP Value":  sv_row[0]
                }).sort_values("SHAP Value", key=abs, ascending=False).head(10)

                st.markdown("<div class='sub-title'>Top Feature Contributions</div>",
                            unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8,4), facecolor="#0e1117")
                ax.set_facecolor("#161b27")
                for sp in ax.spines.values(): sp.set_color("#333")
                ax.tick_params(colors="#aaa")
                colors_c = [PAL[1] if v>0 else PAL[0] for v in contrib["SHAP Value"]]
                ax.barh(contrib["Feature"][::-1], contrib["SHAP Value"][::-1],
                        color=colors_c[::-1], edgecolor="#333")
                ax.axvline(0, color="#555", linewidth=0.9)
                ax.set_xlabel("SHAP Value",color="#aaa")
                ax.set_title("Why this prediction?",color="#ddd",fontweight="bold")
                fig.patch.set_facecolor("#0e1117")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception:
                pass
