"""
╔══════════════════════════════════════════════════════════════╗
║   ENTERPRISE CREDIT RISK INTELLIGENCE PLATFORM               ║
║   Stack: PySpark AutoML · Bi-LSTM · SHAP · RBP-DCG Graph    ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time

# ──────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ — Enterprise Risk Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Enterprise Credit Risk Intelligence — AutoML + SHAP + RBP-DCG"},
)

# ──────────────────────────────────────────────
# DESIGN SYSTEM  (CSS variables + global styles)
# ──────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Tokens ── */
  :root {
    --bg-base:     #0A0E1A;
    --bg-card:     #111827;
    --bg-card-alt: #161D2E;
    --border:      #1F2D45;
    --accent:      #00D4FF;
    --accent-dim:  #0A2A38;
    --green:       #10B981;
    --amber:       #F59E0B;
    --red:         #EF4444;
    --text-1:      #F1F5F9;
    --text-2:      #94A3B8;
    --text-3:      #475569;
    --font-mono:   'JetBrains Mono', 'Courier New', monospace;
  }

  /* ── Reset Chrome injected by Streamlit ── */
  .stApp { background: var(--bg-base); color: var(--text-1); }
  .block-container { padding: 2rem 2.5rem 4rem; max-width: 1440px; }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] * { color: var(--text-1) !important; }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border);
    border-radius: 10px;
    padding: 0.75rem;
    background: var(--bg-card-alt);
  }

  /* ── Metric cards ── */
  [data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
  }
  [data-testid="stMetricLabel"]  { color: var(--text-2) !important; font-size: 0.75rem; letter-spacing: .06em; text-transform: uppercase; }
  [data-testid="stMetricValue"]  { color: var(--text-1) !important; font-size: 1.7rem; font-weight: 700; }
  [data-testid="stMetricDelta"]  { font-size: 0.8rem; }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
  .stDataFrame thead tr th { background: var(--bg-card-alt) !important; color: var(--text-2) !important; font-size: 0.72rem; letter-spacing: .07em; text-transform: uppercase; }
  .stDataFrame tbody tr:nth-child(even) td { background: var(--bg-card-alt) !important; }
  .stDataFrame tbody tr td { color: var(--text-1) !important; font-size: 0.85rem; }

  /* ── Selectbox / inputs ── */
  .stSelectbox > div > div { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 8px; color: var(--text-1) !important; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #00D4FF 0%, #0066CC 100%) !important;
    color: #000 !important;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    letter-spacing: 0.04em;
  }
  .stButton > button:hover { opacity: .88; }

  /* ── Download button ── */
  .stDownloadButton > button {
    background: var(--bg-card) !important;
    color: var(--accent) !important;
    border: 1.5px solid var(--accent) !important;
    border-radius: 8px;
    font-weight: 600;
  }

  /* ── Risk badges (inline HTML) ── */
  .badge {
    display:inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: .07em;
    text-transform: uppercase;
  }
  .badge-high   { background:#3B1212; color:#EF4444; border:1px solid #7F1D1D; }
  .badge-medium { background:#3B2F12; color:#F59E0B; border:1px solid #78350F; }
  .badge-low    { background:#132A21; color:#10B981; border:1px solid #064E3B; }

  /* ── Section headers ── */
  .section-label {
    font-size: 0.68rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: 0.4rem;
  }

  /* ── Alert banners ── */
  .alert-critical {
    background: #1F0A0A;
    border: 1px solid #7F1D1D;
    border-left: 4px solid #EF4444;
    border-radius: 8px;
    padding: 0.85rem 1.2rem;
    color: #FCA5A5;
    font-size: 0.88rem;
  }
  .alert-ok {
    background: #0A1F14;
    border: 1px solid #064E3B;
    border-left: 4px solid #10B981;
    border-radius: 8px;
    padding: 0.85rem 1.2rem;
    color: #6EE7B7;
    font-size: 0.88rem;
  }

  /* ── Divider ── */
  .divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

  /* ── Plotly chart wrapper ── */
  .js-plotly-plot { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PLOTLY THEME
# ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(color="#94A3B8", family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor="#1F2D45", linecolor="#1F2D45", zerolinecolor="#1F2D45"),
    yaxis=dict(gridcolor="#1F2D45", linecolor="#1F2D45", zerolinecolor="#1F2D45"),
    legend=dict(bgcolor="#161D2E", bordercolor="#1F2D45", borderwidth=1),
    margin=dict(l=16, r=16, t=48, b=16),
)
RISK_COLORS = {"LOW": "#10B981", "MEDIUM": "#F59E0B", "HIGH": "#EF4444"}


# ──────────────────────────────────────────────
# DATA LOADING  (cached)
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_default_data():
    """Generate realistic synthetic banking data as demo."""
    np.random.seed(42)
    n = 500
    credit_scores = np.clip(np.random.normal(650, 120, n).astype(int), 300, 850)
    incomes       = np.random.lognormal(10.5, 0.4, n).astype(int)
    delinquencies = np.random.poisson(1.5, n)
    dti           = np.round(np.random.beta(2, 5, n), 2)

    risk_raw = (
        0.4 * (650 - credit_scores) / 350 +
        0.35 * dti +
        0.25 * delinquencies / 6 +
        np.random.normal(0, 0.05, n)
    )
    default_prob = np.clip(risk_raw * 100, 3, 92)

    df = pd.DataFrame({
        "Customer_ID":        [f"C{i:06d}" for i in range(n)],
        "Age":                np.clip(np.random.normal(40, 12, n).astype(int), 21, 65),
        "Income":             incomes,
        "Loan_Amount":        np.random.lognormal(9.5, 0.5, n).astype(int),
        "Loan_Term":          np.random.choice([12, 24, 36, 48, 60], n),
        "Credit_Score":       credit_scores,
        "Employment_Years":   np.clip(np.random.normal(8, 6, n).astype(int), 0, 30),
        "Debt_to_Income":     dti,
        "Num_Credit_Lines":   np.random.poisson(4, n) + 1,
        "Num_Delinquencies":  delinquencies,
        "Home_Ownership":     np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "Loan_Purpose":       np.random.choice(["Home", "Business", "Education"], n),
        "Default_Probability": np.round(default_prob, 2),
        "RBP_Risk_Score":     np.round(np.clip(default_prob / 100 + np.random.normal(0, 0.05, n), 0, 1), 3),
        "RBP_Uncertainty":    np.round(np.random.beta(1.5, 8, n), 3),
        "RBP_Oscillation":    np.random.poisson(0.8, n),
    })

    df["Enterprise_Risk_Score"] = np.round(
        df["Default_Probability"] * 0.7 +
        (df["Debt_to_Income"] * 100 * 0.3), 2
    )

    df["Risk_Level"] = df["Enterprise_Risk_Score"].apply(
        lambda x: "HIGH" if x >= 75 else ("MEDIUM" if x >= 45 else "LOW")
    )

    def decide(row):
        score = row["Enterprise_Risk_Score"]
        unc   = row["RBP_Uncertainty"]
        profit = 100 if score < 45 else (30 if score < 75 else -200)
        if unc > 0.15: profit -= 50
        if row["Income"] < 20000: profit -= 20
        return "APPROVE" if profit > 50 else ("MANUAL REVIEW" if profit > 0 else "REJECT")

    df["Loan_Decision"] = df.apply(decide, axis=1)

    def risk_reason(row):
        reasons = []
        if row["Credit_Score"] < 580:   reasons.append("Low credit score")
        if row["Debt_to_Income"] > 0.5: reasons.append("High debt-to-income")
        if row["Num_Delinquencies"] >= 2: reasons.append("Past delinquencies")
        if row["Loan_Amount"] > 50000:  reasons.append("Large loan amount")
        if row["Enterprise_Risk_Score"] > 70: reasons.append("High ML risk score")
        return ", ".join(reasons[:3]) if reasons else "Financial profile stable"

    df["Risk_Reason"] = df.apply(risk_reason, axis=1)
    return df


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns exist."""
    df = df.copy()
    for col in ["Default_Probability", "Enterprise_Risk_Score", "Credit_Score", "Income"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Default_Probability" not in df.columns and "Enterprise_Risk_Score" in df.columns:
        df["Default_Probability"] = df["Enterprise_Risk_Score"]

    df = df.dropna(subset=["Default_Probability"])

    if "Enterprise_Risk_Score" not in df.columns:
        df["Enterprise_Risk_Score"] = df["Default_Probability"]

    if "Risk_Level" not in df.columns:
        df["Risk_Level"] = df["Enterprise_Risk_Score"].apply(
            lambda x: "HIGH" if x >= 75 else ("MEDIUM" if x >= 45 else "LOW")
        )

    if "Loan_Decision" not in df.columns:
        df["Loan_Decision"] = df["Risk_Level"].map(
            {"HIGH": "REJECT", "MEDIUM": "MANUAL REVIEW", "LOW": "APPROVE"}
        )

    if "Risk_Reason" not in df.columns:
        df["Risk_Reason"] = "—"

    return df


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 .5rem;'>
      <div style='font-size:1.3rem;font-weight:800;color:#F1F5F9;letter-spacing:-.01em;'>
        🏦 CreditIQ
      </div>
      <div style='font-size:0.72rem;color:#475569;letter-spacing:.1em;text-transform:uppercase;margin-top:2px;'>
        Enterprise Risk Platform
      </div>
    </div>
    <hr style='border:none;border-top:1px solid #1F2D45;margin:0.5rem 0 1rem;'/>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Data Source</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload predictions CSV",
        type=["csv"],
        help="Drop enterprise_explainable_predictions.csv or sample_upload.csv",
    )

    st.markdown('<hr style="border:none;border-top:1px solid #1F2D45;margin:1rem 0;"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Filters</div>', unsafe_allow_html=True)

    risk_filter = st.multiselect(
        "Risk Level",
        ["LOW", "MEDIUM", "HIGH"],
        default=["LOW", "MEDIUM", "HIGH"],
    )

    if st.button("↺  Refresh"):
        st.cache_data.clear()
        st.rerun()

    st.markdown('<hr style="border:none;border-top:1px solid #1F2D45;margin:1rem 0;"/>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.7rem;color:#475569;line-height:1.6;'>
      <b style='color:#64748B;'>Stack</b><br/>
      PySpark MLlib · AutoML<br/>
      Bidirectional LSTM<br/>
      SHAP Explainability<br/>
      RBP-DCG Graph Propagation<br/>
      SMOTE Balancing
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
with st.spinner("Loading data…"):
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        data_source = f"📂 {uploaded_file.name}"
    else:
        for fp in ["enterprise_explainable_predictions.csv", "enterprise_risk_predictions.csv"]:
            if os.path.exists(fp):
                raw_df = pd.read_csv(fp)
                data_source = f"📁 {fp}"
                break
        else:
            raw_df = load_default_data()
            data_source = "🔬 Synthetic demo data"

df = enrich_dataframe(raw_df)

# Apply sidebar filter
df_filtered = df[df["Risk_Level"].isin(risk_filter)] if risk_filter else df

# ──────────────────────────────────────────────
# PAGE HEADER
# ──────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    <h1 style='margin:0 0 .2rem;font-size:1.9rem;font-weight:800;
               background:linear-gradient(90deg,#00D4FF,#FFFFFF);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
      Credit Risk Intelligence
    </h1>
    <p style='color:#64748B;font-size:.85rem;margin:0;'>
      AutoML · Explainable AI · Graph Propagation · Real-Time Scoring
    </p>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
    <div style='text-align:right;'>
      <span style='font-size:0.7rem;color:#475569;background:#111827;
                   border:1px solid #1F2D45;padding:4px 10px;border-radius:20px;'>
        {data_source}
      </span>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# ALERT BANNER
# ──────────────────────────────────────────────
high_n = (df_filtered["Risk_Level"] == "HIGH").sum()
high_unc = (df_filtered["RBP_Uncertainty"] > 0.15).sum() if "RBP_Uncertainty" in df_filtered.columns else 0

if high_n > 0:
    st.markdown(f"""
    <div class="alert-critical">
      🚨 <b>{high_n} HIGH RISK</b> customers flagged for immediate review
      {"&nbsp;·&nbsp; ⚠️ " + str(high_unc) + " high-uncertainty cases detected" if high_unc > 0 else ""}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="alert-ok">✅ Portfolio stable — no critical risk flags</div>', unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# KPI METRICS ROW
# ──────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)

total  = len(df_filtered)
n_low  = (df_filtered["Risk_Level"] == "LOW").sum()
n_med  = (df_filtered["Risk_Level"] == "MEDIUM").sum()
n_high = (df_filtered["Risk_Level"] == "HIGH").sum()
avg_risk   = df_filtered["Enterprise_Risk_Score"].mean()
approval_rate = (df_filtered["Loan_Decision"] == "APPROVE").mean() * 100

m1.metric("Total Customers",   f"{total:,}")
m2.metric("🟢 Low Risk",       f"{n_low:,}")
m3.metric("🟡 Medium Risk",    f"{n_med:,}")
m4.metric("🔴 High Risk",      f"{n_high:,}")
m5.metric("Avg Risk Score",    f"{avg_risk:.1f}")
m6.metric("Approval Rate",     f"{approval_rate:.1f}%")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# ROW 1 — GAUGE + PIE + DECISION BAR
# ──────────────────────────────────────────────
r1c1, r1c2, r1c3 = st.columns([1.3, 1, 1.3])

# Portfolio Risk Gauge
with r1c1:
    st.markdown('<div class="section-label">Portfolio Risk Gauge</div>', unsafe_allow_html=True)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(float(avg_risk), 1),
        delta={"reference": 45, "valueformat": ".1f"},
        title={"text": "Enterprise Risk Score", "font": {"color": "#94A3B8", "size": 13}},
        number={"font": {"color": "#F1F5F9", "size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "nticks": 6},
            "bar": {"color": "#00D4FF", "thickness": 0.3},
            "bgcolor": "#111827",
            "bordercolor": "#1F2D45",
            "steps": [
                {"range": [0,  45], "color": "#0D2318"},
                {"range": [45, 75], "color": "#2A1F08"},
                {"range": [75,100], "color": "#2A0808"},
            ],
            "threshold": {
                "line": {"color": "#EF4444", "width": 2},
                "thickness": 0.8,
                "value": 75,
            },
        },
    ))
    gauge.update_layout(**PLOTLY_LAYOUT, height=260)
    st.plotly_chart(gauge, use_container_width=True)

# Risk level donut
with r1c2:
    st.markdown('<div class="section-label">Risk Distribution</div>', unsafe_allow_html=True)
    rc = df_filtered["Risk_Level"].value_counts().reset_index()
    rc.columns = ["Risk_Level", "Count"]
    pie = px.pie(
        rc, values="Count", names="Risk_Level",
        color="Risk_Level", color_discrete_map=RISK_COLORS,
        hole=0.6,
    )
    pie.update_traces(
        textfont_size=11,
        marker=dict(line=dict(color="#111827", width=2)),
    )
    pie.update_layout(**PLOTLY_LAYOUT, height=260,
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.2))
    st.plotly_chart(pie, use_container_width=True)

# Decision breakdown
with r1c3:
    st.markdown('<div class="section-label">Loan Decision Breakdown</div>', unsafe_allow_html=True)
    dec_counts = df_filtered["Loan_Decision"].value_counts()
    dec_colors = {"APPROVE": "#10B981", "MANUAL REVIEW": "#F59E0B", "REJECT": "#EF4444"}
    bar_dec = go.Figure()
    for dec, clr in dec_colors.items():
        if dec in dec_counts.index:
            bar_dec.add_trace(go.Bar(
                name=dec,
                x=[dec],
                y=[dec_counts[dec]],
                marker_color=clr,
                text=[f"{dec_counts[dec]:,}"],
                textposition="outside",
                textfont=dict(color="#F1F5F9", size=13),
            ))
    bar_dec.update_layout(**PLOTLY_LAYOUT, height=260,
                          showlegend=False, barmode="group",
                          yaxis=dict(showgrid=True, gridcolor="#1F2D45"))
    st.plotly_chart(bar_dec, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# ROW 2 — PROBABILITY HISTOGRAM + CREDIT SCATTER
# ──────────────────────────────────────────────
r2c1, r2c2 = st.columns(2)

with r2c1:
    st.markdown('<div class="section-label">Default Probability Distribution</div>', unsafe_allow_html=True)
    hist = px.histogram(
        df_filtered, x="Default_Probability",
        color="Risk_Level", color_discrete_map=RISK_COLORS,
        nbins=40, opacity=0.85,
        labels={"Default_Probability": "Default Probability (%)"},
    )
    hist.update_layout(**PLOTLY_LAYOUT, height=280,
                       bargap=0.05,
                       xaxis_title="Default Probability (%)",
                       yaxis_title="Customer Count")
    st.plotly_chart(hist, use_container_width=True)

with r2c2:
    if "Credit_Score" in df_filtered.columns:
        st.markdown('<div class="section-label">Credit Score vs Risk Score</div>', unsafe_allow_html=True)
        scatter = px.scatter(
            df_filtered.dropna(subset=["Credit_Score"]),
            x="Credit_Score", y="Enterprise_Risk_Score",
            color="Risk_Level", color_discrete_map=RISK_COLORS,
            opacity=0.55, size_max=5,
            hover_data=["Customer_ID", "Loan_Decision"] if "Customer_ID" in df_filtered.columns else None,
        )
        scatter.update_traces(marker=dict(size=4))
        scatter.update_layout(**PLOTLY_LAYOUT, height=280,
                              xaxis_title="Credit Score",
                              yaxis_title="Enterprise Risk Score")
        st.plotly_chart(scatter, use_container_width=True)
    else:
        st.markdown('<div class="section-label">Income vs Risk Score</div>', unsafe_allow_html=True)
        scatter2 = px.scatter(
            df_filtered, x="Income", y="Enterprise_Risk_Score",
            color="Risk_Level", color_discrete_map=RISK_COLORS, opacity=0.55,
        )
        scatter2.update_traces(marker=dict(size=4))
        scatter2.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(scatter2, use_container_width=True)

# ──────────────────────────────────────────────
# ROW 3 — RBP UNCERTAINTY PANEL (conditional)
# ──────────────────────────────────────────────
if "RBP_Uncertainty" in df_filtered.columns:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">RBP-DCG Graph Propagation Signals</div>', unsafe_allow_html=True)

    r3c1, r3c2 = st.columns(2)

    with r3c1:
        unc_scatter = px.scatter(
            df_filtered,
            x="Enterprise_Risk_Score", y="RBP_Uncertainty",
            color="Risk_Level", color_discrete_map=RISK_COLORS,
            opacity=0.6,
            labels={"Enterprise_Risk_Score": "Enterprise Risk Score",
                    "RBP_Uncertainty": "Belief Uncertainty (σ²)"},
            title="Risk Score vs Propagation Uncertainty",
        )
        unc_scatter.update_traces(marker=dict(size=4))
        unc_scatter.add_hline(y=0.15, line_dash="dot", line_color="#EF4444",
                              annotation_text="Uncertainty Threshold", annotation_font_color="#EF4444")
        unc_scatter.update_layout(**PLOTLY_LAYOUT, height=280,
                                  title_font=dict(color="#94A3B8", size=13))
        st.plotly_chart(unc_scatter, use_container_width=True)

    with r3c2:
        rbp_hist = px.histogram(
            df_filtered, x="RBP_Risk_Score",
            color="Risk_Level", color_discrete_map=RISK_COLORS,
            nbins=30, opacity=0.85,
            title="RBP Graph Risk Score Distribution",
            labels={"RBP_Risk_Score": "Graph Propagation Risk Score"},
        )
        rbp_hist.update_layout(**PLOTLY_LAYOUT, height=280,
                               title_font=dict(color="#94A3B8", size=13))
        st.plotly_chart(rbp_hist, use_container_width=True)

# ──────────────────────────────────────────────
# ROW 4 — CUSTOMER INTELLIGENCE TABLE
# ──────────────────────────────────────────────
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Customer Risk Intelligence</div>', unsafe_allow_html=True)

t_col1, t_col2, t_col3 = st.columns([2, 2, 1])
with t_col1:
    dec_filter = st.selectbox(
        "Filter by Decision",
        ["ALL", "APPROVE", "MANUAL REVIEW", "REJECT"],
        label_visibility="collapsed",
    )
with t_col2:
    sort_by = st.selectbox(
        "Sort by",
        ["Enterprise_Risk_Score", "Default_Probability", "Credit_Score"] + (
            ["RBP_Uncertainty"] if "RBP_Uncertainty" in df_filtered.columns else []
        ),
        label_visibility="collapsed",
    )
with t_col3:
    st.download_button(
        "⬇ Export CSV",
        df_filtered.to_csv(index=False).encode("utf-8"),
        file_name="risk_export.csv",
        mime="text/csv",
        use_container_width=True,
    )

table_df = df_filtered if dec_filter == "ALL" else df_filtered[df_filtered["Loan_Decision"] == dec_filter]
table_df = table_df.sort_values(sort_by, ascending=False).reset_index(drop=True)

display_cols = [c for c in [
    "Customer_ID", "Age", "Income", "Credit_Score",
    "Default_Probability", "Enterprise_Risk_Score",
    "RBP_Uncertainty", "Risk_Level", "Loan_Decision", "Risk_Reason"
] if c in table_df.columns]

st.dataframe(
    table_df[display_cols].head(200),
    use_container_width=True,
    height=380,
    column_config={
        "Default_Probability":   st.column_config.ProgressColumn("Default Prob %", min_value=0, max_value=100, format="%.1f%%"),
        "Enterprise_Risk_Score": st.column_config.ProgressColumn("Risk Score",     min_value=0, max_value=100, format="%.1f"),
        "Risk_Level":            st.column_config.TextColumn("Risk"),
        "Loan_Decision":         st.column_config.TextColumn("Decision"),
        "Income":                st.column_config.NumberColumn("Income", format="$%d"),
        "RBP_Uncertainty":       st.column_config.NumberColumn("Uncertainty", format="%.3f"),
    },
)

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#334155;font-size:0.72rem;padding:.5rem 0;'>
  CreditIQ Enterprise &nbsp;·&nbsp; PySpark AutoML &nbsp;·&nbsp; Bidirectional LSTM &nbsp;·&nbsp;
  SHAP Explainability &nbsp;·&nbsp; RBP-DCG Graph Propagation
</div>
""", unsafe_allow_html=True)
