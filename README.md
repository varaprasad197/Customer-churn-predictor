# 📉 Customer Churn Prediction

An interactive Streamlit web app that predicts telecom customer churn using Logistic Regression with a full ML pipeline.

🔗 **[Live Demo Of Churn-Predictor-app](https://customer-churn-predict-app.streamlit.app/)**   <- click for real-time engagement

---

## Features

- **Upload your own CSV** — the app trains the full pipeline on the fly
- **Dataset Overview** — churn rate, class distribution, data preview
- **Model Performance** — accuracy, recall, precision, F1, ROC-AUC, confusion matrix, ROC & PR curves
- **Feature Insights** — L1-selected features and their coefficients
- **Predict Churn** — enter a single customer's details and get an instant churn probability

---

## ML Pipeline

```
Load → Clean → Engineer Features → Split → Encode → Scale
→ SMOTE (balance classes) → L1 Feature Selection → Cross-Validate
→ Hyperparameter Tune → Threshold Optimize → Evaluate
```

**Model**: Logistic Regression (L1 penalty, `liblinear` solver)  
**Resampling**: SMOTE to handle class imbalance (~26.5% churn)  
**Tuning**: RandomizedSearchCV optimized for Recall  

### Results (on Telco dataset)
| Metric | Score |
|--------|-------|
| Accuracy | 0.739 |
| Recall | 0.810 |
| Precision | 0.505 |
| F1 Score | 0.622 |
| ROC-AUC | 0.839 |

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/customer-churn-predictor.git
cd customer-churn-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dataset Format

Upload a CSV with these columns:

| Column | Type | Example |
|--------|------|---------|
| gender | string | Male / Female |
| SeniorCitizen | int | 0 / 1 |
| Partner | string | Yes / No |
| Dependents | string | Yes / No |
| tenure | int | 12 |
| PhoneService | string | Yes / No |
| MultipleLines | string | Yes / No / No phone service |
| InternetService | string | DSL / Fiber optic / No |
| OnlineSecurity | string | Yes / No / No internet service |
| OnlineBackup | string | Yes / No / No internet service |
| DeviceProtection | string | Yes / No / No internet service |
| TechSupport | string | Yes / No / No internet service |
| StreamingTV | string | Yes / No / No internet service |
| StreamingMovies | string | Yes / No / No internet service |
| Contract | string | Month-to-month / One year / Two year |
| PaperlessBilling | string | Yes / No |
| PaymentMethod | string | Electronic check / Mailed check / ... |
| MonthlyCharges | float | 65.50 |
| TotalCharges | float | 786.00 |
| Churn | string | Yes / No |

---

## Tech Stack

- **Streamlit** — web app framework
- **scikit-learn** — ML pipeline
- **imbalanced-learn** — SMOTE oversampling
- **pandas / numpy** — data processing
- **matplotlib / seaborn** — visualizations

---

## License

MIT
