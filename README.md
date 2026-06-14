# 🏦 CreditIQ — Enterprise Credit Risk Intelligence Platform

> PySpark AutoML · Bidirectional LSTM · SHAP Explainability · RBP-DCG Graph Propagation

---

## 🚀 Quick Deploy (Streamlit Cloud — FREE, 2 minutes)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "feat: enterprise credit risk dashboard"
git remote add origin https://github.com/YOUR_USERNAME/credit-risk-app.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to → **https://share.streamlit.io**
2. Click **"New app"**
3. Choose your GitHub repo
4. Set **Main file path**: `app.py`
5. Click **Deploy** ✅

> 🕐 Live in ~90 seconds. No Docker, no server config.

---

## 🏗️ Why NOT Vercel?

Vercel is optimized for **Node.js / Next.js / static frontends**. Python ML apps with heavy dependencies (PySpark, SHAP, TensorFlow, NumPy) don't run on Vercel's serverless functions due to:
- 250MB deployment size limit
- 10s max execution timeout
- No persistent process support

### Best deployment targets for this stack:

| Platform | Cost | Best For |
|---|---|---|
| **Streamlit Cloud** ⭐ | Free | Streamlit apps, fastest path |
| **Hugging Face Spaces** | Free | ML/AI demos, community |
| **Render.com** | Free tier | Full Python web apps |
| **Railway.app** | Free tier | Docker + Python |
| **Google Cloud Run** | Pay-per-use | Production scale |

---

## 📁 Project Structure

```
credit-risk-app/
├── app.py                    # 🎯 Main Streamlit dashboard (enterprise UI)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Theme + server config
├── notebooks/
│   └── Untitled2__1_.ipynb   # Original research notebook
└── README.md
```

---

## 🧠 ML Architecture

```
Raw Banking Data (20,000+ customers)
         │
         ▼
  [SMOTE Balancing]  ──── Handles class imbalance (50/50 split)
         │
         ▼
  [Feature Engineering]
  • Loan-to-Income Ratio
  • Credit Score Normalized
  • High Delinquency Flag
  • Risk Index (composite)
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
  [PySpark AutoML]                  [RBP-DCG Graph]
  • Logistic Regression             • K-NN customer graph
  • Decision Tree                   • Economic cycle kernel
  • Cross-validation                • Belief propagation
  • AUC-ROC optimization            • Uncertainty scoring
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
               [Bidirectional LSTM]
               • 128→64 unit layers
               • BatchNorm + Dropout
               • EarlyStopping (val_auc)
                        │
                        ▼
             [SHAP Explainability]
             • TreeExplainer (RF)
             • Global feature importance
             • Per-customer SHAP drivers
                        │
                        ▼
          [Enterprise Decision Engine]
          • RL-style profit optimization
          • APPROVE / REVIEW / REJECT
          • Fairness analysis by age/income
                        │
                        ▼
          [PySpark Structured Streaming]
          • Real-time CSV ingestion
          • KL-divergence drift detection
          • Live risk scoring
```

---

## 🎨 Dashboard Features

- **Portfolio Risk Gauge** — live enterprise risk score with delta indicator
- **Risk Distribution Donut** — LOW / MEDIUM / HIGH breakdown
- **Decision Breakdown** — APPROVE / REVIEW / REJECT counts
- **Credit Score vs Risk Scatter** — portfolio heatmap
- **Default Probability Histogram** — risk density by segment
- **RBP-DCG Panel** — graph propagation uncertainty signals
- **Customer Intelligence Table** — filterable, sortable, exportable
- **CSV Upload** — drop any predictions file for instant analysis

---

## 🔐 Security Notes

> ⚠️ The ngrok auth token in the original notebook (`39zJ5...`) is a **personal credential**.  
> Remove it before committing to any public repository.  
> For production, use environment variables:

```python
import os
ngrok.set_auth_token(os.environ["NGROK_TOKEN"])
```

---

## 📦 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py

# Open browser → http://localhost:8501
```

---

## 📊 Sample Data

The dashboard auto-generates **500 synthetic customers** when no CSV is provided.  
For real predictions, upload `enterprise_explainable_predictions.csv` from the notebook output.

---

*Built with PySpark MLlib · scikit-learn · TensorFlow · SHAP · Streamlit · Plotly*
