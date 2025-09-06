"""
Professional Streamlit dashboard (single-file) simulating NIS2 control maturity across assembly plants.
How to run:
    1. Create a virtualenv with Python 3.9+.
    2. pip install streamlit pandas numpy plotly openpyxl
    3. streamlit run streamlit_nis2_dashboard_app.py

This app uses synthetic data (12 months) and provides Overview, Plant Drill-Down and Control Drill-Down pages.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime, timedelta

# -----------------------------
# Constants and configuration
# -----------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PLANTS = [
    "Mirafiori","Cassino","Melfi (SATA)","Pomigliano (G.B. Vico)","Modena",
    "Atessa (SEVEL Sud)","Poissy","Mulhouse","Rennes","Sochaux","Hordain (Sevelnord)",
    "Eisenach","Rüsselsheim","Vigo","Zaragoza","Madrid","Mangualde","Tychy",
    "Gliwice","Trnava","Kolín","Szentgotthárd","Kragujevac","Bursa (Tofaş JV)",
    "Luton","Ellesmere Port"
]

CONTROL_DOMAINS = [
    "Governance & Policy","Risk Management","Asset & Configuration Management",
    "Access Control & IAM","Network & Segmentation","Vulnerability & Patch Management",
    "Secure Development & Change","Logging, Monitoring & Detection","Incident Response & Recovery",
    "Business Continuity & DR","Backup & Restore Validation","Supplier/Third-Party Risk",
    "Security Awareness & Phishing","Physical Security","Cloud & SaaS Security"
]

MONTHS = 12
END_DATE = pd.Timestamp.today().normalize()
START_DATE = END_DATE - pd.DateOffset(months=MONTHS-1)
DATES = pd.date_range(start=START_DATE, periods=MONTHS, freq='MS')

# -----------------------------
# Helpers: synthetic data generation
# -----------------------------
@st.cache_data
def generate_controls_taxonomy() -> pd.DataFrame:
    """Create a controls table listing controls per domain with IDs and mandatory flag."""
    rows = []
    ctrl_id = 1
    for domain in CONTROL_DOMAINS:
        for i in range(1, 5):  # 3-5 controls per domain (fixed 4 for compactness)
            rows.append({
                'control_id': f'C{ctrl_id:03d}',
                'domain': domain,
                'control_name': f"{domain} - Control {i}",
                'mandatory': np.random.choice([True, False], p=[0.45, 0.55])
            })
            ctrl_id += 1
    return pd.DataFrame(rows)


def _plant_baseline(plant_name: str) -> float:
    """Assign baseline maturity per plant to create leaders, mid-pack, laggards."""
    # A shortlist of leader and laggard plants to diversify results.
    leaders = {"Mirafiori","Modena","Rennes","Mulhouse","Eisenach","Madrid"}
    laggards = {"Kragujevac","Tychy","Gliwice","Kolín","Bursa (Tofaş JV)","Hordain (Sevelnord)"}
    if plant_name in leaders:
        return np.random.uniform(3.0, 4.2)
    if plant_name in laggards:
        return np.random.uniform(1.2, 2.2)
    return np.random.uniform(2.3, 3.5)

@st.cache_data
def generate_maturity_timeseries(controls_df: pd.DataFrame) -> pd.DataFrame:
    """Generate maturity (CMMI 0-5 float) per plant x control x month with modest trends."""
    records = []
    for plant in PLANTS:
        base = _plant_baseline(plant)
        plant_drift = np.random.normal(0.02, 0.07)  # small monthly drift
        for _, ctrl in controls_df.iterrows():
            # control-specific difficulty
            ctrl_diff = np.random.normal(0, 0.45)
            # generate month series
            for month_idx, date in enumerate(DATES):
                season = 0.08 * np.sin(2 * np.pi * month_idx / 12)
                noise = np.random.normal(0, 0.18)
                # start around base + ctrl_diff with some randomness
                maturity = base + ctrl_diff + plant_drift * month_idx + season + noise
                maturity = float(min(max(maturity, 0.0), 5.0))
                records.append({
                    'plant': plant,
                    'control_id': ctrl['control_id'],
                    'control_name': ctrl['control_name'],
                    'domain': ctrl['domain'],
                    'mandatory': ctrl['mandatory'],
                    'date': date,
                    'cmmi': round(maturity, 2)
                })
    df = pd.DataFrame.from_records(records)
    return df

@st.cache_data
def generate_operational_metrics(maturity_df: pd.DataFrame) -> pd.DataFrame:
    """Generate correlated operational KPIs (MTTD, MTTR, patch SLA, vuln backlog, etc.) per plant x month."""
    rows = []
    # aggregate baseline maturity per plant-month to influence operational metrics
    agg = maturity_df.groupby(['plant','date'])['cmmi'].mean().reset_index().rename(columns={'cmmi':'avg_cmmi'})
    for _, r in agg.iterrows():
        avg = r['avg_cmmi']
        # better maturity -> lower MTTD/MTTR, higher patch SLA, fewer backlog
        mttd = max(0.5, np.random.normal(48 - avg*8, 6))  # hours
        mttr = max(0.2, np.random.normal(10 - avg*1.5, 1.2))  # days
        patch_sla = min(99.5, max(40, np.random.normal(60 + (avg-2.5)*12, 8)))
        vuln_backlog = int(max(0, np.random.normal(400 - avg*70, 80)))
        phishing_fail = max(0.5, np.random.normal(12 - (avg-2.5)*3.2, 2.8))
        backup_pass = min(100, max(40, np.random.normal(78 + (avg-2.5)*8, 6)))
        third_party = min(100, max(20, np.random.normal(55 + (avg-2.5)*10, 10)))
        ot_inventory = min(100, max(10, np.random.normal(60 + (avg-2.5)*12, 12)))
        ics_seg = min(100, max(5, np.random.normal(50 + (avg-2.5)*10, 10)))
        incident_rate = max(0, np.random.normal(2.5 - (avg-2.5)*0.8, 0.9))  # per 1k endpoints
        rows.append({
            'plant': r['plant'],
            'date': r['date'],
            'avg_cmmi': round(avg,2),
            'MTTD_hours': round(float(mttd),2),
            'MTTR_days': round(float(mttr),2),
            'Patch_SLA_pct': round(float(patch_sla),2),
            'Vuln_Backlog_cnt': int(vuln_backlog),
            'Phishing_Fail_pct': round(float(phishing_fail),2),
            'Backup_Restore_Pass_pct': round(float(backup_pass),2),
            'Third_Party_Coverage_pct': round(float(third_party),2),
            'OT_Inventory_Completeness_pct': round(float(ot_inventory),2),
            'ICS_Segmentation_pct': round(float(ics_seg),2),
            'Incident_Rate_per_1k': round(float(incident_rate),2)
        })
    return pd.DataFrame(rows)

# -----------------------------
# KPI calculations
# -----------------------------

def compute_kpis(maturity_df: pd.DataFrame, ops_df: pd.DataFrame, start_date=None, end_date=None, plants_filter=None):
    """Compute the portfolio and plant-level KPIs described in the spec.
    Return a tuple: (kpi_cards_df, portfolio_trends_df, heatmap_df, distributions_df)
    """
    df = maturity_df.copy()
    if plants_filter:
        df = df[df['plant'].isin(plants_filter)]
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    # Weighted compliance: mandatory controls weight 2
    df['mandatory_weight'] = df['mandatory'].apply(lambda x: 2 if x else 1)
    # Define compliance per row (CMMI >= 3.0)
    df['compliant'] = (df['cmmi'] >= 3.0).astype(int)
    df['weighted_compliant'] = df['compliant'] * df['mandatory_weight']
    df['weighted_total'] = df['mandatory_weight']

    # Portfolio-level snapshots using latest month in range
    latest = df['date'].max()
    snapshot = df[df['date'] == latest]

    # NIS2 Compliance Score (%) = weighted compliant / weighted total
    comp_score = snapshot['weighted_compliant'].sum() / snapshot['weighted_total'].sum() * 100

    avg_maturity = snapshot['cmmi'].mean()

    crit_coverage = snapshot[snapshot['mandatory']]['compliant'].sum() / snapshot[snapshot['mandatory']].shape[0] * 100

    # Distribution
    bins = [0,1,2,3,4,5]
    snapshot['maturity_bin'] = pd.cut(snapshot['cmmi'], bins=bins, labels=['0-1','2','3','4','5'], include_lowest=True)
    distribution = snapshot.groupby('maturity_bin').size().reset_index(name='count')
    distribution['pct'] = distribution['count'] / distribution['count'].sum() * 100

    # Operational KPIs from ops_df (latest)
    ops = ops_df.copy()
    if plants_filter:
        ops = ops[ops['plant'].isin(plants_filter)]
    if start_date:
        ops = ops[ops['date'] >= pd.to_datetime(start_date)]
    if end_date:
        ops = ops[ops['date'] <= pd.to_datetime(end_date)]
    ops_latest = ops[ops['date'] == ops['date'].max()]

    mttd = ops_latest['MTTD_hours'].mean()
    mttr = ops_latest['MTTR_days'].mean()
    patch_sla = ops_latest['Patch_SLA_pct'].mean()

    # Heatmap: plants x domain avg cmmi (latest month)
    hm = snapshot.groupby(['plant','domain'])['cmmi'].mean().reset_index()
    heatmap_df = hm.pivot(index='plant', columns='domain', values='cmmi').fillna(0)

    # Top/Bottom controls (portfolio-level average)
    control_avg = snapshot.groupby(['control_id','control_name'])['cmmi'].mean().reset_index()
    top5 = control_avg.sort_values('cmmi', ascending=False).head(5)
    bot5 = control_avg.sort_values('cmmi', ascending=True).head(5)

    # Trends (compliance score per month)
    month_agg = df.copy()
    month_agg['weighted_compliant'] = month_agg['compliant'] * month_agg['mandatory_weight']
    trends = month_agg.groupby('date').agg({'weighted_compliant':'sum','weighted_total':'sum'}).reset_index()
    trends['compliance_pct'] = trends['weighted_compliant'] / trends['weighted_total'] * 100

    # Return dict summarizing
    return {
        'comp_score': round(float(comp_score),2),
        'avg_maturity': round(float(avg_maturity),2),
        'critical_coverage': round(float(crit_coverage),2),
        'mttd': round(float(mttd),2),
        'mttr': round(float(mttr),2),
        'patch_sla': round(float(patch_sla),2),
        'distribution': distribution,
        'heatmap_df': heatmap_df,
        'top5': top5,
        'bot5': bot5,
        'trends': trends
    }

# -----------------------------
# Streamlit UI / App
# -----------------------------

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Return an Excel file in bytes for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='export')
    return output.getvalue()


def main():
    st.set_page_config(page_title="NIS2 Control Maturity - Automotive Portfolio", layout='wide')
    st.title("NIS2 Control Maturity Dashboard — Automotive Plants Portfolio")
    st.markdown("Professional synthetic dataset for demo, KPIs aligned to NIS2 and maturity based on CMMI.")

    # Generate / load data
    controls = generate_controls_taxonomy()
    maturity = generate_maturity_timeseries(controls)
    ops = generate_operational_metrics(maturity)

    # Sidebar filters
    st.sidebar.header("Filters & Controls")
    plant_select = st.sidebar.multiselect("Select Plants (multi)", options=PLANTS, default=None)
    date_range = st.sidebar.date_input("Date Range", value=(DATES.min().date(), DATES.max().date()),
                                      min_value=DATES.min().date(), max_value=DATES.max().date())
    domain_select = st.sidebar.multiselect("Domain (optional)", options=CONTROL_DOMAINS, default=None)
    show_topbottom = st.sidebar.checkbox("Show Top/Bottom 5 Controls", value=True)
    download_raw = st.sidebar.button("Download Current Data (Excel)")

    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    # Apply domain filter to maturity if selected
    mat_filtered = maturity.copy()
    if plant_select and len(plant_select) > 0:
        mat_filtered = mat_filtered[mat_filtered['plant'].isin(plant_select)]
    if domain_select and len(domain_select) > 0:
        mat_filtered = mat_filtered[mat_filtered['domain'].isin(domain_select)]
    mat_filtered = mat_filtered[(mat_filtered['date'] >= start_date) & (mat_filtered['date'] <= end_date)]

    ops_filtered = ops.copy()
    if plant_select and len(plant_select) > 0:
        ops_filtered = ops_filtered[ops_filtered['plant'].isin(plant_select)]
    ops_filtered = ops_filtered[(ops_filtered['date'] >= start_date) & (ops_filtered['date'] <= end_date)]

    # Pages
    page = st.sidebar.radio("Page", ['Overview','Plant Drill-Down','Control Drill-Down'])

    # Compute KPIs
    kpis = compute_kpis(mat_filtered, ops_filtered, start_date=start_date, end_date=end_date, plants_filter=plant_select)

    if page == 'Overview':
        # KPI cards
        col1, col2, col3, col4, col5, col6 = st.columns([1.2,1,1,1,1,1])
        col1.metric("NIS2 Compliance (%)", f"{kpis['comp_score']}%")
        col2.metric("Avg Control Maturity (0–5)", f"{kpis['avg_maturity']}")
        col3.metric("Critical Control Coverage (%)", f"{kpis['critical_coverage']}%")
        col4.metric("Patch SLA (%)", f"{kpis['patch_sla']:.1f}%")
        col5.metric("MTTD (hrs)", f"{kpis['mttd']}")
        col6.metric("MTTR (days)", f"{kpis['mttr']}")

        st.subheader("Portfolio Maturity Heatmap (Plants × Domains)")
        heatmap_df = kpis['heatmap_df'].reset_index().melt(id_vars='plant', var_name='domain', value_name='cmmi')
        fig_hm = px.density_heatmap(heatmap_df, x='domain', y='plant', z='cmmi', histfunc='avg', nbinsx=len(CONTROL_DOMAINS), nbinsy=len(PLANTS), color_continuous_scale='RdYlGn')
        fig_hm.update_layout(height=700, yaxis={'categoryorder':'array', 'categoryarray':PLANTS})
        st.plotly_chart(fig_hm, use_container_width=True)

        st.subheader("Top 5 and Bottom 5 Controls (Portfolio)")
        cols_tb = st.columns(2)
        if show_topbottom:
            cols_tb[0].markdown("**Top 5 Controls (by avg maturity)**")
            cols_tb[0].table(kpis['top5'])
            cols_tb[1].markdown("**Bottom 5 Controls (by avg maturity)**")
            cols_tb[1].table(kpis['bot5'])

        st.subheader("Maturity Distribution (Latest)")
        dist = kpis['distribution']
        fig_dist = px.bar(dist, x='maturity_bin', y='pct', labels={'maturity_bin':'Maturity Bin','pct':'% of controls'})
        st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader("Compliance Trend (12 months)")
        fig_tr = px.line(kpis['trends'], x='date', y='compliance_pct', markers=True)
        fig_tr.update_layout(yaxis_title='Compliance (%)')
        st.plotly_chart(fig_tr, use_container_width=True)

    elif page == 'Plant Drill-Down':
        selected_plant = st.selectbox('Select Plant for Drill-Down', options=PLANTS, index=0)
        st.header(f"Plant — {selected_plant}")
        plant_mat = mat_filtered[mat_filtered['plant'] == selected_plant]
        plant_ops = ops_filtered[ops_filtered['plant'] == selected_plant]

        if plant_mat.empty:
            st.warning('No data available for selected filters.')
        else:
            # Plant KPIs (latest)
            latest_date = plant_mat['date'].max()
            snapshot = plant_mat[plant_mat['date'] == latest_date]
            comp_score = (snapshot.assign(weight=snapshot['mandatory'].apply(lambda x:2 if x else 1)).assign(compliant=(snapshot['cmmi']>=3.0).astype(int)).eval('compliant*weight').sum() / snapshot['mandatory'].apply(lambda x:2 if x else 1).sum())*100
            avg_m = snapshot['cmmi'].mean()
            cols = st.columns(4)
            cols[0].metric('Plant Compliance (%)', f"{comp_score:.1f}%")
            cols[1].metric('Avg Maturity', f"{avg_m:.2f}")
            if not plant_ops.empty:
                ops_latest = plant_ops[plant_ops['date']==plant_ops['date'].max()].iloc[0]
                cols[2].metric('MTTD (hrs)', ops_latest['MTTD_hours'])
                cols[3].metric('MTTR (days)', ops_latest['MTTR_days'])

            # Radar chart (domain-level maturity)
            domain_avg = snapshot.groupby('domain')['cmmi'].mean().reset_index()
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=domain_avg['cmmi'], theta=domain_avg['domain'], fill='toself', name=selected_plant))
            fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0,5])), showlegend=False, height=500)
            st.plotly_chart(fig_radar, use_container_width=True)

            st.subheader('Weakest and Strongest Controls')
            ctrl_avg = snapshot.groupby(['control_id','control_name'])['cmmi'].mean().reset_index()
            weakest = ctrl_avg.sort_values('cmmi').head(6)
            strongest = ctrl_avg.sort_values('cmmi', ascending=False).head(6)
            c1, c2 = st.columns(2)
            c1.table(weakest)
            c2.table(strongest)

            st.subheader('Operational Scatter: MTTD vs MTTR (monthly)')
            if not plant_ops.empty:
                fig_s = px.scatter(plant_ops, x='MTTD_hours', y='MTTR_days', size='Vuln_Backlog_cnt', color='date', hover_data=['Patch_SLA_pct'])
                st.plotly_chart(fig_s, use_container_width=True)

    else:  # Control Drill-Down
        domain = st.selectbox('Choose Domain', options=CONTROL_DOMAINS)
        domain_controls = controls[controls['domain']==domain]
        control_choice = st.selectbox('Choose Control', options=domain_controls['control_id'].tolist())
        control_name = domain_controls[domain_controls['control_id']==control_choice]['control_name'].iloc[0]
        st.header(f"Control — {control_choice} — {control_name}")

        ctrl_df = mat_filtered[mat_filtered['control_id']==control_choice]
        if ctrl_df.empty:
            st.warning('No data for this control with current filters.')
        else:
            # Column chart: maturity by plant (latest month)
            latest_date = ctrl_df['date'].max()
            snapshot = ctrl_df[ctrl_df['date']==latest_date]
            fig_col = px.bar(snapshot.sort_values('cmmi', ascending=False), x='plant', y='cmmi', labels={'cmmi':'CMMI (0-5)'})
            fig_col.update_layout(xaxis_tickangle=45, height=450)
            st.plotly_chart(fig_col, use_container_width=True)

            st.subheader('Time series: Top 3 and Bottom 3 plants for this control')
            plant_avg = snapshot[['plant','cmmi']].sort_values('cmmi', ascending=False)
            top3 = plant_avg['plant'].head(3).tolist()
            bot3 = plant_avg['plant'].tail(3).tolist()
            sel = top3 + bot3
            ts = mat_filtered[(mat_filtered['control_id']==control_choice) & (mat_filtered['plant'].isin(sel))]
            fig_ts = px.line(ts, x='date', y='cmmi', color='plant', markers=True)
            fig_ts.update_layout(yaxis_range=[0,5])
            st.plotly_chart(fig_ts, use_container_width=True)

    # Download button logic (exports the filtered maturity dataframe)
    if download_raw:
        df_export = mat_filtered.copy()
        st.sidebar.success('Preparing download...')
        excel_bytes = to_excel_bytes(df_export)
        st.sidebar.download_button('Download Excel', excel_bytes, file_name='nis2_maturity_export.xlsx')

    # KPI Glossary
    with st.expander('KPI Glossary and Calculation Notes'):
        st.markdown("""
        - **NIS2 Compliance (%)**: percent of controls at CMMI >= 3.0, mandatory controls weighted 2x.
        - **Avg Control Maturity**: arithmetic mean of control CMMI scores (0–5).
        - **Critical Control Coverage**: percent of mandatory controls meeting CMMI >= 3.0.
        - **MTTD / MTTR**: Mean Time to Detect (hours) and Mean Time to Recover (days) — synthetic and correlated with maturity.
        - **Patch SLA**: percent of critical vulnerabilities remediated within policy window.
        - **Vulnerability Backlog**: open vulnerabilities older than 30 days (synthetic count).
        """)

    st.sidebar.markdown("---")
    st.sidebar.caption('Generated with a fixed seed for reproducibility. Synthetic data only; for demo purposes.')

if __name__ == '__main__':
    main()
