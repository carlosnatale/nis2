"""
Refactored Streamlit NIS2 Dashboard (tabs + improved UX)
How to run:
    1. python 3.9+
    2. pip install streamlit pandas numpy plotly openpyxl
    3. streamlit run streamlit_nis2_dashboard_app.py

This version restructures the UI into tabs (Overview, Governance, Supply Chain, Assets, Incident Response, Awareness, Product Security).
It replaces long tables with metric cards, interactive charts (Plotly), progress/gauge visuals, and expandable methodology explainers (accordions).
Synthetic data generation remains; still reproducible via fixed seed.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# -----------------------------
# Config
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
# Data generation (synthetic)
# -----------------------------
@st.cache_data
def generate_controls_taxonomy() -> pd.DataFrame:
    rows = []
    ctrl_id = 1
    for domain in CONTROL_DOMAINS:
        for i in range(1, 5):
            rows.append({
                'control_id': f'C{ctrl_id:03d}',
                'domain': domain,
                'control_name': f"{domain} - Control {i}",
                'mandatory': np.random.choice([True, False], p=[0.45, 0.55])
            })
            ctrl_id += 1
    return pd.DataFrame(rows)

@st.cache_data
def _plant_baseline(plant_name: str) -> float:
    leaders = {"Mirafiori","Modena","Rennes","Mulhouse","Eisenach","Madrid"}
    laggards = {"Kragujevac","Tychy","Gliwice","Kolín","Bursa (Tofaş JV)","Hordain (Sevelnord)"}
    if plant_name in leaders:
        return np.random.uniform(3.0, 4.2)
    if plant_name in laggards:
        return np.random.uniform(1.2, 2.2)
    return np.random.uniform(2.3, 3.5)

@st.cache_data
def generate_maturity_timeseries(controls_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for plant in PLANTS:
        base = _plant_baseline(plant)
        plant_drift = np.random.normal(0.02, 0.07)
        for _, ctrl in controls_df.iterrows():
            ctrl_diff = np.random.normal(0, 0.45)
            for month_idx, date in enumerate(DATES):
                season = 0.08 * np.sin(2 * np.pi * month_idx / 12)
                noise = np.random.normal(0, 0.18)
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
    return pd.DataFrame.from_records(records)

@st.cache_data
def generate_operational_metrics(maturity_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    agg = maturity_df.groupby(['plant','date'])['cmmi'].mean().reset_index().rename(columns={'cmmi':'avg_cmmi'})
    for _, r in agg.iterrows():
        avg = r['avg_cmmi']
        # Governance
        pct_senior_trained = min(100, max(10, np.random.normal(50 + (avg-2.5)*12, 10)))
        time_ack_assign_hours = max(1, np.random.normal(48 - (avg-2.5)*6, 12))
        cyber_budget_pct = min(100, max(1, np.random.normal(7 + (avg-2.5)*4, 2)))
        crisis_plans_tested = max(0, int(np.random.poisson(1 + (avg-2.5)*0.6)))
        # Supply chain
        supplier_contractual_compliance_pct = min(100, max(10, np.random.normal(55 + (avg-2.5)*12, 10)))
        supplier_assessed_pct = min(100, max(5, np.random.normal(48 + (avg-2.5)*14, 12)))
        supplier_mttd_remediate_days = max(1, np.random.normal(20 - (avg-2.5)*3, 4))
        supplier_incidents_cnt = int(max(0, np.random.poisson(0.4 - (avg-2.5)*0.05)))
        # Assets & Vulnerability
        pct_assets_discovered = min(100, max(12, np.random.normal(60 + (avg-2.5)*15, 12)))
        mttd_vuln_scan_hours = max(1, np.random.normal(72 - (avg-2.5)*10, 15))
        mttr_critical_vuln_days = max(0.2, np.random.normal(14 - (avg-2.5)*2.8, 3))
        ot_segmentation_coverage_pct = min(100, max(5, np.random.normal(45 + (avg-2.5)*14, 12)))
        # Incident Response
        mttd_incident_hours = max(0.5, np.random.normal(24 - (avg-2.5)*4.5, 8))
        mttr_contain_hours = max(0.1, np.random.normal(48 - (avg-2.5)*6, 12))
        mttr_recover_hours = max(1, np.random.normal(72 - (avg-2.5)*8, 18))
        pct_reported_within_24h = min(100, max(0, np.random.normal(58 + (avg-2.5)*18, 18)))
        unplanned_downtime_hours = max(0, np.random.normal(40 - (avg-2.5)*9, 30))
        # Awareness
        pct_employees_trained = min(100, max(20, np.random.normal(62 + (avg-2.5)*12, 12)))
        phishing_failure_rate = max(0.1, np.random.normal(12 - (avg-2.5)*2.8, 3))
        incidents_employee_error_cnt = int(max(0, np.random.poisson(1.2 - (avg-2.5)*0.12)))
        # Product Security
        pct_new_models_tara = min(100, max(0, np.random.normal(50 + (avg-2.5)*18, 18)))
        time_vuln_to_patch_days = max(0.5, np.random.normal(30 - (avg-2.5)*6, 8))
        pct_fleet_ota_patchable = min(100, max(0, np.random.normal(48 + (avg-2.5)*15, 15)))
        successful_pentest_count = int(max(0, np.random.poisson(1 + (avg-2.5)*0.6)))
        # legacy ops
        mttd_alarm = max(0.5, np.random.normal(48 - avg*8, 6))
        mttr_ops = max(0.2, np.random.normal(10 - avg*1.5, 1.2))
        patch_sla = min(99.5, max(40, np.random.normal(60 + (avg-2.5)*12, 8)))
        vuln_backlog = int(max(0, np.random.normal(400 - avg*70, 80)))
        backup_pass = min(100, max(40, np.random.normal(78 + (avg-2.5)*8, 6)))
        rows.append({
            'plant': r['plant'],
            'date': r['date'],
            # Governance
            'Pct_Senior_Mgmt_Trained_pct': round(float(pct_senior_trained),2),
            'Time_to_Ack_Assign_New_Risks_hours': round(float(time_ack_assign_hours),2),
            'Cybersecurity_Budget_pct_of_IT_OT': round(float(cyber_budget_pct),2),
            'Num_Cyber_Crisis_Plans_Tested': int(crisis_plans_tested),
            # Supply chain
            'Supplier_Contractual_Compliance_pct': round(float(supplier_contractual_compliance_pct),2),
            'Supplier_Assessed_pct': round(float(supplier_assessed_pct),2),
            'Supplier_MTTD_Remediate_Critical_days': round(float(supplier_mttd_remediate_days),2),
            'Supplier_Incidents_cnt': int(supplier_incidents_cnt),
            # Assets & vuln
            'Pct_Assets_Discovered_Classified_pct': round(float(pct_assets_discovered),2),
            'MTTD_Vuln_Scan_hours': round(float(mttd_vuln_scan_hours),2),
            'MTTR_Critical_Vuln_days': round(float(mttr_critical_vuln_days),2),
            'OT_Segmentation_Coverage_pct': round(float(ot_segmentation_coverage_pct),2),
            # Incident Response
            'MTTD_Incident_hours': round(float(mttd_incident_hours),2),
            'MTTR_Contain_hours': round(float(mttr_contain_hours),2),
            'MTTR_Recovery_hours': round(float(mttr_recover_hours),2),
            'Pct_Reported_within_24h_pct': round(float(pct_reported_within_24h),2),
            'Unplanned_Downtime_hours': round(float(unplanned_downtime_hours),2),
            # Awareness
            'Pct_Employees_Trained_pct': round(float(pct_employees_trained),2),
            'Phishing_Failure_pct': round(float(phishing_failure_rate),2),
            'Incidents_Employee_Error_cnt': int(incidents_employee_error_cnt),
            # Product Security
            'Pct_New_Models_TARA_pct': round(float(pct_new_models_tara),2),
            'Time_Vuln_to_Patch_days': round(float(time_vuln_to_patch_days),2),
            'Pct_Fleet_OTA_Patchable_pct': round(float(pct_fleet_ota_patchable),2),
            'Successful_Pentest_Count': int(successful_pentest_count),
            # legacy ops
            'avg_cmmi': round(avg,2),
            'MTTD_hours': round(float(mttd_alarm),2),
            'MTTR_days': round(float(mttr_ops),2),
            'Patch_SLA_pct': round(float(patch_sla),2),
            'Vuln_Backlog_cnt': int(vuln_backlog),
            'Backup_Restore_Pass_pct': round(float(backup_pass),2)
        })
    return pd.DataFrame(rows)

# -----------------------------
# KPI computations
# -----------------------------

def compute_kpis(maturity_df: pd.DataFrame, ops_df: pd.DataFrame, plants_filter=None, start_date=None, end_date=None):
    df = maturity_df.copy()
    if plants_filter:
        df = df[df['plant'].isin(plants_filter)]
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    df['mandatory_weight'] = df['mandatory'].apply(lambda x: 2 if x else 1)
    df['compliant'] = (df['cmmi'] >= 3.0).astype(int)
    df['weighted_compliant'] = df['compliant'] * df['mandatory_weight']
    df['weighted_total'] = df['mandatory_weight']

    latest = df['date'].max()
    snapshot = df[df['date'] == latest]
    comp_score = snapshot['weighted_compliant'].sum() / snapshot['weighted_total'].sum() * 100
    avg_maturity = snapshot['cmmi'].mean()
    crit_coverage = snapshot[snapshot['mandatory']]['compliant'].sum() / snapshot[snapshot['mandatory']].shape[0] * 100

    # ops aggregation
    ops = ops_df.copy()
    if plants_filter:
        ops = ops[ops['plant'].isin(plants_filter)]
    if start_date:
        ops = ops[ops['date'] >= pd.to_datetime(start_date)]
    if end_date:
        ops = ops[ops['date'] <= pd.to_datetime(end_date)]
    ops_latest = ops[ops['date'] == ops['date'].max()]

    def avg(name):
        return ops_latest[name].mean() if name in ops_latest.columns and not ops_latest.empty else np.nan

    portfolio = {
        'NIS2_Compliance_pct': round(float(comp_score),2),
        'Avg_Maturity': round(float(avg_maturity),2),
        'Critical_Coverage_pct': round(float(crit_coverage),2),
        # governance
        'Pct_Senior_Mgmt_Trained_pct': round(avg('Pct_Senior_Mgmt_Trained_pct'),2),
        'Time_to_Ack_Assign_New_Risks_hours': round(avg('Time_to_Ack_Assign_New_Risks_hours'),2),
        'Cybersecurity_Budget_pct_of_IT_OT': round(avg('Cybersecurity_Budget_pct_of_IT_OT'),2),
        'Num_Cyber_Crisis_Plans_Tested': int(np.nanmean(ops_latest['Num_Cyber_Crisis_Plans_Tested'])) if 'Num_Cyber_Crisis_Plans_Tested' in ops_latest.columns else 0,
        # supply
        'Supplier_Contractual_Compliance_pct': round(avg('Supplier_Contractual_Compliance_pct'),2),
        'Supplier_Assessed_pct': round(avg('Supplier_Assessed_pct'),2),
        'Supplier_MTTD_Remediate_Critical_days': round(avg('Supplier_MTTD_Remediate_Critical_days'),2),
        'Supplier_Incidents_cnt': int(np.nansum(ops_latest['Supplier_Incidents_cnt'])) if 'Supplier_Incidents_cnt' in ops_latest.columns else 0,
        # assets
        'Pct_Assets_Discovered_Classified_pct': round(avg('Pct_Assets_Discovered_Classified_pct'),2),
        'MTTD_Vuln_Scan_hours': round(avg('MTTD_Vuln_Scan_hours'),2),
        'MTTR_Critical_Vuln_days': round(avg('MTTR_Critical_Vuln_days'),2),
        'OT_Segmentation_Coverage_pct': round(avg('OT_Segmentation_Coverage_pct'),2),
        # incident
        'MTTD_Incident_hours': round(avg('MTTD_Incident_hours'),2),
        'MTTR_Contain_hours': round(avg('MTTR_Contain_hours'),2),
        'MTTR_Recovery_hours': round(avg('MTTR_Recovery_hours'),2),
        'Pct_Reported_within_24h_pct': round(avg('Pct_Reported_within_24h_pct'),2),
        'Unplanned_Downtime_hours': round(avg('Unplanned_Downtime_hours'),2),
        # awareness
        'Pct_Employees_Trained_pct': round(avg('Pct_Employees_Trained_pct'),2),
        'Phishing_Failure_pct': round(avg('Phishing_Failure_pct'),2),
        'Incidents_Employee_Error_cnt': int(np.nansum(ops_latest['Incidents_Employee_Error_cnt'])) if 'Incidents_Employee_Error_cnt' in ops_latest.columns else 0,
        # product
        'Pct_New_Models_TARA_pct': round(avg('Pct_New_Models_TARA_pct'),2),
        'Time_Vuln_to_Patch_days': round(avg('Time_Vuln_to_Patch_days'),2),
        'Pct_Fleet_OTA_Patchable_pct': round(avg('Pct_Fleet_OTA_Patchable_pct'),2),
        'Successful_Pentest_Count': int(np.nansum(ops_latest['Successful_Pentest_Count'])) if 'Successful_Pentest_Count' in ops_latest.columns else 0
    }

    # heatmap
    heatmap_df = snapshot.groupby(['plant','domain'])['cmmi'].mean().reset_index().pivot(index='plant', columns='domain', values='cmmi').fillna(0)
    # top/bottom controls
    control_avg = snapshot.groupby(['control_id','control_name'])['cmmi'].mean().reset_index()
    top5 = control_avg.sort_values('cmmi', ascending=False).head(5)
    bot5 = control_avg.sort_values('cmmi', ascending=True).head(5)
    # trend
    month_agg = df.copy()
    month_agg['weighted_compliant'] = month_agg['compliant'] * month_agg['mandatory_weight']
    trends = month_agg.groupby('date').agg({'weighted_compliant':'sum','weighted_total':'sum'}).reset_index()
    trends['compliance_pct'] = trends['weighted_compliant'] / trends['weighted_total'] * 100

    return {
        'portfolio': portfolio,
        'heatmap_df': heatmap_df,
        'top5': top5,
        'bot5': bot5,
        'trends': trends
    }

# -----------------------------
# Utilities
# -----------------------------

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='export')
    return output.getvalue()

# small helper: plotly gauge
def plot_gauge(value: float, title: str, suffix: str = '%') -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': 'darkblue'},
            'steps': [
                {'range': [0, 50], 'color': 'red'},
                {'range': [50, 75], 'color': 'yellow'},
                {'range': [75, 100], 'color': 'green'}
            ]
        },
        number={'suffix': suffix}
    ))
    fig.update_layout(height=240, margin={'t':30,'b':0,'l':0,'r':0})
    return fig

# -----------------------------
# Streamlit app
# -----------------------------

def main():
    st.set_page_config(page_title='NIS2 Dashboard — Improved UX', layout='wide')
    st.title('NIS2 Control Maturity — Usability-Driven Dashboard')
    st.markdown('Tabs for each domain, compact KPI cards, interactive charts, and methodology expanders.')

    controls = generate_controls_taxonomy()
    maturity = generate_maturity_timeseries(controls)
    ops = generate_operational_metrics(maturity)

    # sidebar filters
    st.sidebar.header('Filters')
    selected_plants = st.sidebar.multiselect('Plants (multi-select)', options=PLANTS, default=[PLANTS[0]])
    date_range = st.sidebar.date_input('Date range', value=(DATES.min().date(), DATES.max().date()), min_value=DATES.min().date(), max_value=DATES.max().date())
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    # apply filters
    mat_filtered = maturity[(maturity['date'] >= start_date) & (maturity['date'] <= end_date)]
    ops_filtered = ops[(ops['date'] >= start_date) & (ops['date'] <= end_date)]
    if selected_plants:
        mat_filtered = mat_filtered[mat_filtered['plant'].isin(selected_plants)]
        ops_filtered = ops_filtered[ops_filtered['plant'].isin(selected_plants)]

    # compute KPIs
    kpi_data = compute_kpis(mat_filtered, ops_filtered, plants_filter=selected_plants, start_date=start_date, end_date=end_date)
    portfolio = kpi_data['portfolio']

    tabs = st.tabs(['Overview','Governance & Risk','Supply Chain','Assets & Vulnerabilities','Incident Response','Awareness','Product Security','Control Maturity'])

    # Overview
    with tabs[0]:
        st.subheader('Executive Snapshot')
        c1, c2 = st.columns([2,3])
        with c1:
            # big compliance gauge
            fig_g = plot_gauge(portfolio['NIS2_Compliance_pct'] if portfolio['NIS2_Compliance_pct'] is not None else 0, 'NIS2 Compliance (weighted)')
            st.plotly_chart(fig_g, use_container_width=True)
            st.metric('Avg Control Maturity (0-5)', f"{portfolio['Avg_Maturity']}")
            st.metric('Critical Controls Coverage (%)', f"{portfolio['Critical_Coverage_pct']}%")

        with c2:
            st.markdown('**Portfolio heatmap — avg CMMI by plant & domain**')
            heatmap_df = kpi_data['heatmap_df'].reset_index().melt(id_vars='plant', var_name='domain', value_name='cmmi')
            fig_hm = px.imshow(kpi_data['heatmap_df'], aspect='auto', origin='lower', color_continuous_scale='RdYlGn', labels=dict(x='Domain', y='Plant'))
            fig_hm.update_layout(height=420)
            st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown('**Trends: compliance (12 months)**')
        tr = kpi_data['trends']
        fig_tr = px.line(tr, x='date', y='compliance_pct', markers=True)
        fig_tr.update_layout(yaxis_title='Compliance (%)')
        st.plotly_chart(fig_tr, use_container_width=True)

        cols_tb = st.columns(2)
        cols_tb[0].subheader('Top 5 Controls by maturity')
        cols_tb[0].table(kpi_data['top5'])
        cols_tb[1].subheader('Bottom 5 Controls by maturity')
        cols_tb[1].table(kpi_data['bot5'])

    # Governance & Risk Tab
    with tabs[1]:
        st.subheader('Governance & Risk')
        g = portfolio
        cols = st.columns(4)
        cols[0].metric('% Senior Mgmt Trained', f"{g['Pct_Senior_Mgmt_Trained_pct']}%")
        cols[1].metric('Time to Ack/Assign Risks (hrs)', f"{g['Time_to_Ack_Assign_New_Risks_hours']}")
        cols[2].metric('Cybersecurity Budget % of IT/OT', f"{g['Cybersecurity_Budget_pct_of_IT_OT']}%")
        cols[3].metric('Crisis Plans Tested', f"{g['Num_Cyber_Crisis_Plans_Tested']}")

        # visual: bar showing training and budget
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['Senior Trained %'], y=[g['Pct_Senior_Mgmt_Trained_pct']], name='Senior Trained'))
        fig.add_trace(go.Bar(x=['Cyber Budget %'], y=[g['Cybersecurity_Budget_pct_of_IT_OT']], name='Cyber Budget'))
        fig.update_layout(barmode='group', yaxis=dict(range=[0,100]), height=300)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander('Methodology & Data Requirements'):
            st.write('- % Senior Management Trained: HR/LMS export with role mapping.
- Time to Acknowledge/Assign: timestamps from risk register workflow.
- Cybersecurity budget: financial ledger tagging between cybersecurity and total IT/OT spend.
- Crisis plans: documented plans + test evidence (date, outcome).')

    # Supply Chain Tab
    with tabs[2]:
        st.subheader('Supply Chain & Third-Party Risk')
        s = portfolio
        scols = st.columns(4)
        scols[0].metric('% Suppliers Contractually Compliant', f"{s['Supplier_Contractual_Compliance_pct']}%")
        scols[1].metric('% Suppliers Assessed (12m)', f"{s['Supplier_Assessed_pct']}%")
        scols[2].metric('Supplier MTTD Remediate (days)', f"{s['Supplier_MTTD_Remediate_Critical_days']}")
        scols[3].metric('Supplier-origin incidents', f"{s['Supplier_Incidents_cnt']}")

        # visual: supplier coverage gauge and timeline
        fig_sup = plot_gauge(s['Supplier_Assessed_pct'] if s['Supplier_Assessed_pct'] is not None else 0, 'Suppliers Assessed (12m)')
        st.plotly_chart(fig_sup, use_container_width=True)

        with st.expander('Methodology & Data Requirements'):
            st.write('- Supplier contractual compliance: contract metadata + clause scan.
- Supplier assessment: questionnaires and audit evidence within 12 months.
- MTTD remediation: supplier ticketing and patch validation timestamps.
- Supplier incidents: incident ticket annotated with supplier origin.')

    # Assets & Vulnerabilities Tab
    with tabs[3]:
        st.subheader('Assets & Vulnerabilities')
        a = portfolio
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric('% Assets Discovered & Classified', f"{a['Pct_Assets_Discovered_Classified_pct']}%")
        ac2.metric('MTTD Vuln Scan (hrs)', f"{a['MTTD_Vuln_Scan_hours']}")
        ac3.metric('MTTR Critical Vuln (days)', f"{a['MTTR_Critical_Vuln_days']}")
        ac4.metric('OT Segmentation Coverage', f"{a['OT_Segmentation_Coverage_pct']}%")

        # visualize distribution of asset discovery across plants
        asset_df = ops_filtered[['plant','Pct_Assets_Discovered_Classified_pct']].drop_duplicates()
        fig_assets = px.bar(asset_df.sort_values('Pct_Assets_Discovered_Classified_pct', ascending=False), x='plant', y='Pct_Assets_Discovered_Classified_pct')
        fig_assets.update_layout(xaxis_tickangle=45, height=350)
        st.plotly_chart(fig_assets, use_container_width=True)

        with st.expander('Methodology & Data Requirements'):
            st.write('- Asset discovery: results from CMDB/Discovery tools; classification mapping to critical/non-critical.
- Vulnerability detection: scanner outputs (timestamped findings) and ingestion into tracking system.
- MTTR critical: remediation ticket closure verified by rescans.
- OT segmentation: network diagrams, firewall/zone rules evidence.')

    # Incident Response Tab
    with tabs[4]:
        st.subheader('Incident Response & Resilience')
        ir = portfolio
        i1, i2, i3, i4, i5 = st.columns(5)
        i1.metric('MTTD (hrs)', f"{ir['MTTD_Incident_hours']}")
        i2.metric('MTTR Contain (hrs)', f"{ir['MTTR_Contain_hours']}")
        i3.metric('MTTR Recovery (hrs)', f"{ir['MTTR_Recovery_hours']}")
        i4.metric('% Reported <24h', f"{ir['Pct_Reported_within_24h_pct']}%")
        i5.metric('Unplanned Downtime (hrs)', f"{ir['Unplanned_Downtime_hours']}")

        # SLA chart for reporting
        sla_vals = [ir['Pct_Reported_within_24h_pct'], max(0,100-ir['Pct_Reported_within_24h_pct'])]
        fig_pie = px.pie(names=['Reported <24h','Reported >24h'], values=sla_vals)
        st.plotly_chart(fig_pie, use_container_width=True)

        with st.expander('Methodology & Data Requirements'):
            st.write('- MTTD/MTTR: incident ticket timelines from SOC and IR records.
- % reported in 24h: compare incident detection timestamp with regulator notification timestamp.
- Unplanned downtime: correlate MES/SCADA downtime with incident tickets (start/end times).')

    # Awareness Tab
    with tabs[5]:
        st.subheader('Employee Awareness & Training')
        aw = portfolio
        a1, a2, a3 = st.columns(3)
        a1.metric('% Employees Trained', f"{aw['Pct_Employees_Trained_pct']}%")
        a2.metric('Phishing Failure Rate', f"{aw['Phishing_Failure_pct']}%")
        a3.metric('Incidents due to Employee Error', f"{aw['Incidents_Employee_Error_cnt']}")

        # trend of phishing failures (mock)
        phishing_ts = ops_filtered.groupby('date')['Phishing_Failure_pct'].mean().reset_index()
        if not phishing_ts.empty:
            fig_pf = px.line(phishing_ts, x='date', y='Phishing_Failure_pct', markers=True)
            fig_pf.update_layout(yaxis_title='Phishing Failure (%)')
            st.plotly_chart(fig_pf, use_container_width=True)

        with st.expander('Methodology & Data Requirements'):
            st.write('- Training: LMS completion reports with user and role mapping.
- Phishing failure: simulated campaign platform results; ensure unique user counts and avoid double-counting.
- Incident attribution: incident post-mortems tagging human error as root cause.')

    # Product Security Tab
    with tabs[6]:
        st.subheader('Product Security (Vehicles & Services)')
        ps = portfolio
        p1, p2, p3, p4 = st.columns(4)
        p1.metric('% New Models with TARA', f"{ps['Pct_New_Models_TARA_pct']}%")
        p2.metric('Time vuln -> patch (days)', f"{ps['Time_Vuln_to_Patch_days']}")
        p3.metric('% Fleet OTA patchable', f"{ps['Pct_Fleet_OTA_Patchable_pct']}%")
        p4.metric('Successful Pentests (count)', f"{ps['Successful_Pentest_Count']}")

        fig_pod = px.bar(x=['TARA %','Fleet OTA %'], y=[ps['Pct_New_Models_TARA_pct'], ps['Pct_Fleet_OTA_Patchable_pct']])
        fig_pod.update_layout(yaxis=dict(range=[0,100]), height=300)
        st.plotly_chart(fig_pod, use_container_width=True)

        with st.expander('Methodology & Data Requirements'):
            st.write('- TARA: design-phase artifacts for each new model.
- Vulnerability->patch: track from discovery ticket to OTA release version.
- OTA patchable: device management/telemetry coverage of fleet.
- Penetration tests: validated reports and remediation verification.')

    # Control Maturity tab: radar + plant comparator
    with tabs[7]:
        st.subheader('Control Maturity — Compare Plants')
        cmp_plants = st.multiselect('Select plants to compare (radar)', options=PLANTS, default=[selected_plants[0]])
        if not cmp_plants:
            st.info('Choose one or more plants to compare.')
        else:
            latest = mat_filtered['date'].max()
            cmp_df = mat_filtered[mat_filtered['date']==latest]
            domain_avg = cmp_df.groupby(['plant','domain'])['cmmi'].mean().reset_index()
            fig = go.Figure()
            for plant in cmp_plants:
                row = domain_avg[domain_avg['plant']==plant]
                if not row.empty:
                    fig.add_trace(go.Scatterpolar(r=row['cmmi'], theta=row['domain'], fill='toself', name=plant))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0,5])), height=600)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('**Weakest controls (selected plants)**')
            weak = cmp_df.groupby(['control_id','control_name'])['cmmi'].mean().reset_index().sort_values('cmmi').head(10)
            st.table(weak)

    # download
    with st.sidebar.expander('Export & Debug'):
        if st.button('Download Maturity CSV'):
            st.sidebar.download_button('Download CSV', mat_filtered.to_csv(index=False).encode('utf-8'), file_name='maturity_export.csv')
        if st.button('Download Ops CSV'):
            st.sidebar.download_button('Download CSV', ops_filtered.to_csv(index=False).encode('utf-8'), file_name='ops_export.csv')

    st.sidebar.markdown('---')
    st.sidebar.caption('Refactored UX: tabs, cards, charts, and methodology expanders. Synthetic data only.')

if __name__ == '__main__':
    main()
