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
import os

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ — Enterprise Risk Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DESIGN SYSTEM ──────────────────────────────────────────────
st.markdown("""
<style>
  :root {
    --bg-base:#0A0E1A; --bg-card:#111827; --bg-card-alt:#161D2E;
    --border:#1F2D45; --accent:#00D4FF;
    --green:#10B981; --amber:#F59E0B; --red:#EF4444;
    --text-1:#F1F5F9; --text-2:#94A3B8; --text-3:#475569;
  }
  .stApp { background:var(--bg-base) !important; color:var(--text-1); }
  .block-container { padding:2rem 2.5rem 4rem; max-width:1440px; }
  #MainMenu, footer, header { visibility:hidden; }

  [data-testid="stSidebar"] { background:var(--bg-card) !important; border-right:1px solid var(--border); }
  [data-testid="stSidebar"] * { color:var(--text-1) !important; }
  [data-testid="stFileUploader"] { border:1.5px dashed var(--border); border-radius:10px; background:var(--bg-card-alt); }

  [data-testid="stMetric"] { background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:1.1rem 1.25rem; }
  [data-testid="stMetricLabel"] { color:var(--text-2) !important; font-size:.75rem; letter-spacing:.06em; text-transform:uppercase; }
  [data-testid="stMetricValue"] { color:var(--text-1) !important; font-size:1.7rem; font-weight:700; }

  .stButton>button { background:linear-gradient(135deg,#00D4FF,#0066CC) !important; color:#000 !important; font-weight:700; border:none; border-radius:8px; }
  .stDownloadButton>button { background:var(--bg-card) !important; color:var(--accent) !important; border:1.5px solid var(--accent) !important; border-radius:8px; font-weight:600; }
  .stSelectbox>div>div { background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:8px; color:var(--text-1) !important; }

  .section-label { font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color:#00D4FF; font-weight:700; margin-bottom:.4rem; }
  .divider { border:none; border-top:1px solid var(--border); margin:1.5rem 0; }
  .alert-critical { background:#1F0A0A; border:1px solid #7F1D1D; border-left:4px solid #EF4444; border-radius:8px; padding:.85rem 1.2rem; color:#FCA5A5; font-size:.88rem; }
  .alert-ok { background:#0A1F14; border:1px solid #064E3B; border-left:4px solid #10B981; border-radius:8px; padding:.85rem 1.2rem; color:#6EE7B7; font-size:.88rem; }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY DARK THEME ──────────────────────────────────────────
PL = dict(
    paper_bgcolor="#111827", plot_bgcolor="#111827",
    font=dict(color="#94A3B8", family="Inter,sans-serif", size=12),
    xaxis=dict(gridcolor="#1F2D45", linecolor="#1F2D45", zerolinecolor="#1F2D45"),
    yaxis=dict(gridcolor="#1F2D45", linecolor="#1F2D45", zerolinecolor="#1F2D45"),
    legend=dict(bgcolor="#161D2E", bordercolor="#1F2D45", borderwidth=1),
    margin=dict(l=20, r=20, t=50, b=20),
)
RC = {"LOW": "#10B981", "MEDIUM": "#F59E0B", "HIGH": "#EF4444"}

# ── SYNTHETIC DATA (realistic spread across all 3 risk tiers) ──
@st.cache_data(show_spinner=False)
def load_default_data():
    np.random.seed(42)
    n = 600

    # Build three realistic customer segments
    n_low, n_med, n_high = 280, 200, 120

    def seg(n, score_mu, score_sd, dti_mu, delinq_rate, cs_mu, cs_sd):
        scores = np.clip(np.random.normal(score_mu, score_sd, n), 5, 95)
        return dict(
            Enterprise_Risk_Score = np.round(scores, 2),
            Default_Probability   = np.round(np.clip(scores + np.random.normal(0, 4, n), 3, 97), 2),
            Debt_to_Income        = np.round(np.clip(np.random.normal(dti_mu, 0.08, n), 0.05, 0.95), 2),
            Num_Delinquencies     = np.random.poisson(delinq_rate, n),
            Credit_Score          = np.clip(np.random.normal(cs_mu, cs_sd, n).astype(int), 300, 850),
            Income                = np.random.lognormal(10.5 - score_mu/200, 0.35, n).astype(int),
            Loan_Amount           = np.random.lognormal(9.5, 0.5, n).astype(int),
            Age                   = np.clip(np.random.normal(40, 12, n).astype(int), 21, 65),
            Employment_Years      = np.clip(np.random.normal(8, 5, n).astype(int), 0, 30),
            Loan_Term             = np.random.choice([12,24,36,48,60], n),
            Home_Ownership        = np.random.choice([0,1], n, p=[0.5,0.5]),
            Loan_Purpose          = np.random.choice(["Home","Business","Education"], n),
            RBP_Risk_Score        = np.round(np.clip(scores/100 + np.random.normal(0, 0.04, n), 0, 1), 3),
            RBP_Uncertainty       = np.round(np.random.beta(1.5 + score_mu/60, 8, n), 3),
            RBP_Oscillation       = np.random.poisson(delinq_rate * 0.5, n),
            Risk_Level            = [("LOW" if score_mu < 35 else ("MEDIUM" if score_mu < 65 else "HIGH"))] * n,
        )

    low  = seg(n_low,  22, 9,  0.20, 0.3, 730, 55)
    med  = seg(n_med,  57, 8,  0.42, 1.2, 610, 60)
    high = seg(n_high, 82, 7,  0.68, 3.1, 490, 65)

    rows = []
    for s in [low, med, high]:
        nn = len(s["Enterprise_Risk_Score"])
        for i in range(nn):
            rows.append({k: v[i] if hasattr(v, '__len__') else v for k, v in s.items()})

    df = pd.DataFrame(rows)
    df["Customer_ID"] = [f"C{i:06d}" for i in range(len(df))]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    def decide(row):
        s, u = row["Enterprise_Risk_Score"], row["RBP_Uncertainty"]
        profit = 100 if s < 45 else (30 if s < 75 else -200)
        if u > 0.15: profit -= 50
        if row["Income"] < 20000: profit -= 20
        return "APPROVE" if profit > 50 else ("MANUAL REVIEW" if profit > 0 else "REJECT")

    df["Loan_Decision"] = df.apply(decide, axis=1)

    def risk_reason(row):
        r = []
        if row["Credit_Score"] < 580:        r.append("Low credit score")
        if row["Debt_to_Income"] > 0.5:      r.append("High debt-to-income")
        if row["Num_Delinquencies"] >= 2:    r.append("Past delinquencies")
        if row["Loan_Amount"] > 50000:       r.append("Large loan amount")
        if row["Enterprise_Risk_Score"] > 70: r.append("High ML risk score")
        return ", ".join(r[:3]) if r else "Financial profile stable"

    df["Risk_Reason"] = df.apply(risk_reason, axis=1)
    return df


def enrich_dataframe(df):
    df = df.copy()
    for c in ["Default_Probability","Enterprise_Risk_Score","Credit_Score","Income"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Default_Probability" not in df.columns and "Enterprise_Risk_Score" in df.columns:
        df["Default_Probability"] = df["Enterprise_Risk_Score"]
    df = df.dropna(subset=["Default_Probability"])
    if "Enterprise_Risk_Score" not in df.columns:
        df["Enterprise_Risk_Score"] = df["Default_Probability"]
    if "Risk_Level" not in df.columns:
        df["Risk_Level"] = df["Enterprise_Risk_Score"].apply(
            lambda x: "HIGH" if x >= 75 else ("MEDIUM" if x >= 45 else "LOW"))
    if "Loan_Decision" not in df.columns:
        df["Loan_Decision"] = df["Risk_Level"].map(
            {"HIGH":"REJECT","MEDIUM":"MANUAL REVIEW","LOW":"APPROVE"})
    if "Risk_Reason" not in df.columns:
        df["Risk_Reason"] = "—"
    return df


# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 .5rem;'>
      <div style='font-size:1.3rem;font-weight:800;color:#F1F5F9;'>🏦 CreditIQ</div>
      <div style='font-size:.72rem;color:#475569;letter-spacing:.1em;text-transform:uppercase;margin-top:2px;'>Enterprise Risk Platform</div>
    </div>
    <hr style='border:none;border-top:1px solid #1F2D45;margin:.5rem 0 1rem;'/>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Data Source</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload predictions CSV", type=["csv"])

    st.markdown('<hr style="border:none;border-top:1px solid #1F2D45;margin:1rem 0;"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Filters</div>', unsafe_allow_html=True)
    risk_filter = st.multiselect("Risk Level", ["LOW","MEDIUM","HIGH"], default=["LOW","MEDIUM","HIGH"])
    if st.button("↺  Refresh"):
        st.cache_data.clear(); st.rerun()

    st.markdown('<hr style="border:none;border-top:1px solid #1F2D45;margin:1rem 0;"/>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.7rem;color:#475569;line-height:1.7;'>
      <b style='color:#64748B;'>Stack</b><br/>
      PySpark MLlib · AutoML<br/>Bidirectional LSTM<br/>
      SHAP Explainability<br/>RBP-DCG Graph Propagation<br/>SMOTE Balancing
    </div>""", unsafe_allow_html=True)

# ── LOAD DATA ──────────────────────────────────────────────────
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    data_source = f"📂 {uploaded_file.name}"
else:
    for fp in ["enterprise_explainable_predictions.csv","enterprise_risk_predictions.csv"]:
        if os.path.exists(fp):
            raw_df = pd.read_csv(fp); data_source = f"📁 {fp}"; break
    else:
        raw_df = load_default_data(); data_source = "🔬 Synthetic demo data"

df = enrich_dataframe(raw_df)
df_filtered = df[df["Risk_Level"].isin(risk_filter)] if risk_filter else df

# ── HEADER ─────────────────────────────────────────────────────
c1, c2 = st.columns([3,1])
with c1:
    st.markdown("""
    <h1 style='margin:0 0 .2rem;font-size:1.9rem;font-weight:800;
               background:linear-gradient(90deg,#00D4FF,#FFFFFF);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
      Credit Risk Intelligence</h1>
    <p style='color:#64748B;font-size:.85rem;margin:0;'>
      AutoML · Explainable AI · Graph Propagation · Real-Time Scoring</p>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div style='text-align:right;padding-top:.5rem;'>
      <span style='font-size:.7rem;color:#475569;background:#111827;
                   border:1px solid #1F2D45;padding:4px 10px;border-radius:20px;'>
        {data_source}</span></div>""", unsafe_allow_html=True)

# ── ALERT ──────────────────────────────────────────────────────
high_n = (df_filtered["Risk_Level"]=="HIGH").sum()
if high_n > 0:
    st.markdown(f'<div class="alert-critical">🚨 <b>{high_n} HIGH RISK</b> customers flagged for immediate review</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="alert-ok">✅ Portfolio stable — no critical risk flags</div>', unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── KPI ROW ────────────────────────────────────────────────────
total = len(df_filtered)
n_low  = (df_filtered["Risk_Level"]=="LOW").sum()
n_med  = (df_filtered["Risk_Level"]=="MEDIUM").sum()
n_high = (df_filtered["Risk_Level"]=="HIGH").sum()
avg_risk = df_filtered["Enterprise_Risk_Score"].mean()
appr_rate = (df_filtered["Loan_Decision"]=="APPROVE").mean()*100

m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("Total Customers",  f"{total:,}")
m2.metric("🟢 Low Risk",      f"{n_low:,}")
m3.metric("🟡 Medium Risk",   f"{n_med:,}")
m4.metric("🔴 High Risk",     f"{n_high:,}")
m5.metric("Avg Risk Score",   f"{avg_risk:.1f}")
m6.metric("Approval Rate",    f"{appr_rate:.1f}%")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── ROW 1: GAUGE · PIE · BAR ───────────────────────────────────
r1c1, r1c2, r1c3 = st.columns([1.3,1,1.3])

with r1c1:
    st.markdown('<div class="section-label">Portfolio Risk Gauge</div>', unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(float(avg_risk), 1),
        delta={"reference": 45, "valueformat":".1f"},
        title={"text":"Enterprise Risk Score","font":{"color":"#94A3B8","size":13}},
        number={"font":{"color":"#F1F5F9","size":42}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#475569","nticks":6},
            "bar":{"color":"#00D4FF","thickness":0.28},
            "bgcolor":"#111827","bordercolor":"#1F2D45",
            "steps":[
                {"range":[0,45],"color":"#0D2318"},
                {"range":[45,75],"color":"#2A1F08"},
                {"range":[75,100],"color":"#2A0808"},
            ],
            "threshold":{"line":{"color":"#EF4444","width":2},"thickness":0.8,"value":75},
        },
    ))
    fig_gauge.update_layout(**PL, height=270)
    st.plotly_chart(fig_gauge, use_container_width=True)

with r1c2:
    st.markdown('<div class="section-label">Risk Distribution</div>', unsafe_allow_html=True)
    rc_df = df_filtered["Risk_Level"].value_counts().reset_index()
    rc_df.columns = ["Risk_Level","Count"]
    fig_pie = px.pie(rc_df, values="Count", names="Risk_Level",
                     color="Risk_Level", color_discrete_map=RC, hole=0.58)
    fig_pie.update_traces(textfont_size=12,
                          marker=dict(line=dict(color="#111827", width=2)))
    fig_pie.update_layout(**PL, height=270, showlegend=True)
        fig_pie.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.25))
    st.plotly_chart(fig_pie, use_container_width=True)

with r1c3:
    st.markdown('<div class="section-label">Loan Decision Breakdown</div>', unsafe_allow_html=True)
    dec_map = {"APPROVE":"#10B981","MANUAL REVIEW":"#F59E0B","REJECT":"#EF4444"}
    dec_df  = df_filtered["Loan_Decision"].value_counts().reset_index()
    dec_df.columns = ["Decision","Count"]
    dec_df["Color"] = dec_df["Decision"].map(dec_map)
    fig_bar = go.Figure()
    for _, row in dec_df.iterrows():
        fig_bar.add_trace(go.Bar(
            name=row["Decision"], x=[row["Decision"]], y=[row["Count"]],
            marker_color=row["Color"],
            text=[f"{row['Count']:,}"], textposition="outside",
            textfont=dict(color="#F1F5F9", size=14),
        ))
    fig_bar.update_layout(**PL, height=270, showlegend=False, barmode="group",
                          yaxis=dict(showgrid=True, gridcolor="#1F2D45"))
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── ROW 2: HISTOGRAM · SCATTER ─────────────────────────────────
r2c1, r2c2 = st.columns(2)

with r2c1:
    st.markdown('<div class="section-label">Default Probability Distribution</div>', unsafe_allow_html=True)
    fig_hist = px.histogram(
        df_filtered, x="Default_Probability",
        color="Risk_Level", color_discrete_map=RC,
        nbins=40, opacity=0.85,
        category_orders={"Risk_Level":["LOW","MEDIUM","HIGH"]},
    )
    fig_hist.update_layout(**PL, height=290,
                           xaxis_title="Default Probability (%)",
                           yaxis_title="Customers", bargap=0.04)
    st.plotly_chart(fig_hist, use_container_width=True)

with r2c2:
    if "Credit_Score" in df_filtered.columns:
        st.markdown('<div class="section-label">Credit Score vs Risk Score</div>', unsafe_allow_html=True)
        fig_sc = px.scatter(
            df_filtered.dropna(subset=["Credit_Score"]),
            x="Credit_Score", y="Enterprise_Risk_Score",
            color="Risk_Level", color_discrete_map=RC, opacity=0.6,
            hover_data={c:True for c in ["Customer_ID","Loan_Decision","Income"] if c in df_filtered.columns},
        )
        fig_sc.update_traces(marker=dict(size=5))
        fig_sc.add_vline(x=580, line_dash="dot", line_color="#475569",
                         annotation_text="580 threshold", annotation_font_color="#475569")
        fig_sc.add_hline(y=75, line_dash="dot", line_color="#EF4444",
                         annotation_text="High risk", annotation_font_color="#EF4444")
        fig_sc.update_layout(**PL, height=290,
                             xaxis_title="Credit Score", yaxis_title="Enterprise Risk Score")
        st.plotly_chart(fig_sc, use_container_width=True)

# ── ROW 3: DTI BOX · INCOME VIOLIN ────────────────────────────
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

r3c1, r3c2 = st.columns(2)

with r3c1:
    if "Debt_to_Income" in df_filtered.columns:
        st.markdown('<div class="section-label">Debt-to-Income by Risk Tier</div>', unsafe_allow_html=True)
        fig_box = px.box(
            df_filtered, x="Risk_Level", y="Debt_to_Income",
            color="Risk_Level", color_discrete_map=RC,
            category_orders={"Risk_Level":["LOW","MEDIUM","HIGH"]},
            points="outliers",
        )
        fig_box.update_layout(**PL, height=270, showlegend=False,
                              xaxis_title="Risk Tier", yaxis_title="Debt-to-Income Ratio")
        st.plotly_chart(fig_box, use_container_width=True)

with r3c2:
    if "Income" in df_filtered.columns:
        st.markdown('<div class="section-label">Income Distribution by Risk Tier</div>', unsafe_allow_html=True)
        fig_vio = px.violin(
            df_filtered, x="Risk_Level", y="Income",
            color="Risk_Level", color_discrete_map=RC,
            category_orders={"Risk_Level":["LOW","MEDIUM","HIGH"]},
            box=True, points=False,
        )
        fig_vio.update_layout(**PL, height=270, showlegend=False,
                              xaxis_title="Risk Tier", yaxis_title="Annual Income ($)")
        st.plotly_chart(fig_vio, use_container_width=True)

# ── ROW 4: RBP-DCG PANEL ───────────────────────────────────────
if "RBP_Uncertainty" in df_filtered.columns:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">RBP-DCG Graph Propagation Signals</div>', unsafe_allow_html=True)
    r4c1, r4c2 = st.columns(2)

    with r4c1:
        fig_unc = px.scatter(
            df_filtered,
            x="Enterprise_Risk_Score", y="RBP_Uncertainty",
            color="Risk_Level", color_discrete_map=RC, opacity=0.6,
            title="Risk Score vs Propagation Uncertainty",
        )
        fig_unc.update_traces(marker=dict(size=5))
        fig_unc.add_hline(y=0.15, line_dash="dot", line_color="#EF4444",
                          annotation_text="Uncertainty threshold",
                          annotation_font_color="#EF4444")
        fig_unc.update_layout(**PL, height=270,
                              title_font=dict(color="#94A3B8", size=13),
                              xaxis_title="Enterprise Risk Score",
                              yaxis_title="Belief Uncertainty σ²")
        st.plotly_chart(fig_unc, use_container_width=True)

    with r4c2:
        fig_rbp = px.histogram(
            df_filtered, x="RBP_Risk_Score",
            color="Risk_Level", color_discrete_map=RC,
            nbins=30, opacity=0.85,
            title="Graph Propagation Risk Score",
            category_orders={"Risk_Level":["LOW","MEDIUM","HIGH"]},
        )
        fig_rbp.update_layout(**PL, height=270,
                              title_font=dict(color="#94A3B8", size=13),
                              xaxis_title="RBP Risk Score (0–1)",
                              yaxis_title="Count")
        st.plotly_chart(fig_rbp, use_container_width=True)

# ── CUSTOMER TABLE ─────────────────────────────────────────────
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Customer Risk Intelligence</div>', unsafe_allow_html=True)

tc1, tc2, tc3 = st.columns([2,2,1])
with tc1:
    dec_filter = st.selectbox("Filter by Decision",
        ["ALL","APPROVE","MANUAL REVIEW","REJECT"], label_visibility="collapsed")
with tc2:
    sort_cols = [c for c in ["Enterprise_Risk_Score","Default_Probability",
                              "Credit_Score","RBP_Uncertainty"] if c in df_filtered.columns]
    sort_by = st.selectbox("Sort by", sort_cols, label_visibility="collapsed")
with tc3:
    st.download_button("⬇ Export CSV",
        df_filtered.to_csv(index=False).encode("utf-8"),
        file_name="risk_export.csv", mime="text/csv", use_container_width=True)

tdf = df_filtered if dec_filter == "ALL" else df_filtered[df_filtered["Loan_Decision"]==dec_filter]
tdf = tdf.sort_values(sort_by, ascending=False).reset_index(drop=True)

show_cols = [c for c in ["Customer_ID","Age","Income","Credit_Score",
                          "Debt_to_Income","Default_Probability",
                          "Enterprise_Risk_Score","RBP_Uncertainty",
                          "Risk_Level","Loan_Decision","Risk_Reason"]
             if c in tdf.columns]

col_cfg = {}
if "Default_Probability" in tdf.columns:
    col_cfg["Default_Probability"] = st.column_config.ProgressColumn(
        "Default Prob %", min_value=0, max_value=100, format="%.1f%%")
if "Enterprise_Risk_Score" in tdf.columns:
    col_cfg["Enterprise_Risk_Score"] = st.column_config.ProgressColumn(
        "Risk Score", min_value=0, max_value=100, format="%.1f")
if "Income" in tdf.columns:
    col_cfg["Income"] = st.column_config.NumberColumn("Income", format="$%d")
if "RBP_Uncertainty" in tdf.columns:
    col_cfg["RBP_Uncertainty"] = st.column_config.NumberColumn("Uncertainty σ²", format="%.3f")

st.dataframe(tdf[show_cols].head(300),
             use_container_width=True, height=380,
             column_config=col_cfg)

# ── FOOTER ─────────────────────────────────────────────────────
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#334155;font-size:.72rem;padding:.5rem 0;'>
  CreditIQ Enterprise &nbsp;·&nbsp; PySpark AutoML &nbsp;·&nbsp; Bidirectional LSTM
  &nbsp;·&nbsp; SHAP Explainability &nbsp;·&nbsp; RBP-DCG Graph Propagation
</div>""", unsafe_allow_html=True)