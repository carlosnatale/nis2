"""
Professional Streamlit dashboard (single-file) simulating NIS2 control maturity across assembly plants.
How to run:
    1. Create a virtualenv with Python 3.9+.
    2. pip install streamlit pandas numpy plotly openpyxl
    3. streamlit run streamlit_nis2_dashboard_app.py

This app uses synthetic data (12 months) and provides Overview, Plant Drill-Down and Control Drill-Down pages.

Modifications: Added comprehensive KPI generation and visualization for the expanded KPI list requested, including Governance & Risk, Supply Chain, Asset & Vulnerability, Incident Response & Resilience, Employee Awareness, and Product Security.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime

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
    df = pd.DataFrame.from_records(records)
    return df

@st.cache_data
def generate_operational_metrics(maturity_df: pd.DataFrame) -> pd.DataFrame:
    """Generate correlated operational KPIs (expanded list) per plant x month."""
    rows = []
    agg = maturity_df.groupby(['plant','date'])['cmmi'].mean().reset_index().rename(columns={'cmmi':'avg_cmmi'})
    for _, r in agg.iterrows():
        avg = r['avg_cmmi']
        # Governance & Risk
        pct_senior_trained = min(100, max(10, np.random.normal(50 + (avg-2.5)*12, 10)))
        time_ack_assign_hours = max(1, np.random.normal(48 - (avg-2.5)*6, 12))
        cyber_budget_pct = min(100, max(1, np.random.normal(7 + (avg-2.5)*4, 2)))
        crisis_plans_tested = max(0, int(np.random.poisson(1 + (avg-2.5)*0.6)))

        # Supply chain
        supplier_contractual_compliance_pct = min(100, max(10, np.random.normal(55 + (avg-2.5)*12, 10)))
        supplier_assessed_pct = min(100, max(5, np.random.normal(48 + (avg-2.5)*14, 12)))
        supplier_mttd_remediate_days = max(1, np.random.normal(20 - (avg-2.5)*3, 4))
        supplier_incidents_cnt = int(max(0, np.random.poisson(0.4 - (avg-2.5)*0.05)))

        # Asset & Vulnerability
        pct_assets_discovered = min(100, max(12, np.random.normal(60 + (avg-2.5)*15, 12)))
        mttd_vuln_scan_hours = max(1, np.random.normal(72 - (avg-2.5)*10, 15))
        mttr_critical_vuln_days = max(0.2, np.random.normal(14 - (avg-2.5)*2.8, 3))
        ot_segmentation_coverage_pct = min(100, max(5, np.random.normal(45 + (avg-2.5)*14, 12)))

        # Incident Response & Resilience
        mttd_incident_hours = max(0.5, np.random.normal(24 - (avg-2.5)*4.5, 8))
        mttr_contain_hours = max(0.1, np.random.normal(48 - (avg-2.5)*6, 12))
        mttr_recover_hours = max(1, np.random.normal(72 - (avg-2.5)*8, 18))
        pct_reported_within_24h = min(100, max(0, np.random.normal(58 + (avg-2.5)*18, 18)))
        unplanned_downtime_hours = max(0, np.random.normal(40 - (avg-2.5)*9, 30))

        # Employee Awareness & Training
        pct_employees_trained = min(100, max(20, np.random.normal(62 + (avg-2.5)*12, 12)))
        phishing_failure_rate = max(0.1, np.random.normal(12 - (avg-2.5)*2.8, 3))
        incidents_employee_error_cnt = int(max(0, np.random.poisson(1.2 - (avg-2.5)*0.12)))

        # Product Security
        pct_new_models_tara = min(100, max(0, np.random.normal(50 + (avg-2.5)*18, 18)))
        time_vuln_to_patch_days = max(0.5, np.random.normal(30 - (avg-2.5)*6, 8))
        pct_fleet_ota_patchable = min(100, max(0, np.random.normal(48 + (avg-2.5)*15, 15)))
        successful_pentest_count = int(max(0, np.random.poisson(1 + (avg-2.5)*0.6)))

        # existing operational metrics
        mttd_alarm = max(0.5, np.random.normal(48 - avg*8, 6))
        mttr_ops = max(0.2, np.random.normal(10 - avg*1.5, 1.2))
        patch_sla = min(99.5, max(40, np.random.normal(60 + (avg-2.5)*12, 8)))
        vuln_backlog = int(max(0, np.random.normal(400 - avg*70, 80)))
        backup_pass = min(100, max(40, np.random.normal(78 + (avg-2.5)*8, 6)))
        third_party = min(100, max(20, np.random.normal(55 + (avg-2.5)*10, 10)))
        ot_inventory = min(100, max(10, np.random.normal(60 + (avg-2.5)*12, 12)))
        ics_seg = min(100, max(5, np.random.normal(50 + (avg-2.5)*10, 10)))
        incident_rate = max(0, np.random.normal(2.5 - (avg-2.5)*0.8, 0.9))

        rows.append({
            'plant': r['plant'],
            'date': r['date'],
            # Governance & Risk
            'Pct_Senior_Mgmt_Trained_pct': round(float(pct_senior_trained),2),
            'Time_to_Ack_Assign_New_Risks_hours': round(float(time_ack_assign_hours),2),
            'Cybersecurity_Budget_pct_of_IT_OT': round(float(cyber_budget_pct),2),
            'Num_Cyber_Crisis_Plans_Tested': int(crisis_plans_tested),
            # Supply Chain
            'Supplier_Contractual_Compliance_pct': round(float(supplier_contractual_compliance_pct),2),
            'Supplier_Assessed_pct': round(float(supplier_assessed_pct),2),
            'Supplier_MTTD_Remediate_Critical_days': round(float(supplier_mttd_remediate_days),2),
            'Supplier_Incidents_cnt': int(supplier_incidents_cnt),
            # Asset & Vulnerability
            'Pct_Assets_Discovered_Classified_pct': round(float(pct_assets_discovered),2),
            'MTTD_Vuln_Scan_hours': round(float(mttd_vuln_scan_hours),2),
            'MTTR_Critical_Vuln_days': round(float(mttr_critical_vuln_days),2),
            'OT_Segmentation_Coverage_pct': round(float(ot_segmentation_coverage_pct),2),
            # Incident Response & Resilience
            'MTTD_Incident_hours': round(float(mttd_incident_hours),2),
            'MTTR_Contain_hours': round(float(mttr_contain_hours),2),
            'MTTR_Recovery_hours': round(float(mttr_recover_hours),2),
            'Pct_Reported_within_24h_pct': round(float(pct_reported_within_24h),2),
            'Unplanned_Downtime_hours': round(float(unplanned_downtime_hours),2),
            # Employee Awareness
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
    Return a dict with KPI values and dataframes for visuals.
    """
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

    bins = [0,1,2,3,4,5]
    snapshot['maturity_bin'] = pd.cut(snapshot['cmmi'], bins=bins, labels=['0-1','2','3','4','5'], include_lowest=True)
    distribution = snapshot.groupby('maturity_bin').size().reset_index(name='count')
    distribution['pct'] = distribution['count'] / distribution['count'].sum() * 100

    ops = ops_df.copy()
    if plants_filter:
        ops = ops[ops['plant'].isin(plants_filter)]
    if start_date:
        ops = ops[ops['date'] >= pd.to_datetime(start_date)]
    if end_date:
        ops = ops[ops['date'] <= pd.to_datetime(end_date)]
    ops_latest = ops[ops['date'] == ops['date'].max()]

    # operational aggregates for dashboard cards (portfolio-level average)
    def avg_col(name):
        return ops_latest[name].mean() if not ops_latest.empty else np.nan

    governance = {
        'Pct_Senior_Mgmt_Trained_pct': round(avg_col('Pct_Senior_Mgmt_Trained_pct'),2),
        'Time_to_Ack_Assign_New_Risks_hours': round(avg_col('Time_to_Ack_Assign_New_Risks_hours'),2),
        'Cybersecurity_Budget_pct_of_IT_OT': round(avg_col('Cybersecurity_Budget_pct_of_IT_OT'),2),
        'Num_Cyber_Crisis_Plans_Tested': int(np.nanmean(ops_latest['Num_Cyber_Crisis_Plans_Tested'])) if 'Num_Cyber_Crisis_Plans_Tested' in ops_latest.columns else 0
    }

    supply_chain = {
        'Supplier_Contractual_Compliance_pct': round(avg_col('Supplier_Contractual_Compliance_pct'),2),
        'Supplier_Assessed_pct': round(avg_col('Supplier_Assessed_pct'),2),
        'Supplier_MTTD_Remediate_Critical_days': round(avg_col('Supplier_MTTD_Remediate_Critical_days'),2),
        'Supplier_Incidents_cnt': int(np.nansum(ops_latest['Supplier_Incidents_cnt'])) if 'Supplier_Incidents_cnt' in ops_latest.columns else 0
    }

    asset_vuln = {
        'Pct_Assets_Discovered_Classified_pct': round(avg_col('Pct_Assets_Discovered_Classified_pct'),2),
        'MTTD_Vuln_Scan_hours': round(avg_col('MTTD_Vuln_Scan_hours'),2),
        'MTTR_Critical_Vuln_days': round(avg_col('MTTR_Critical_Vuln_days'),2),
        'OT_Segmentation_Coverage_pct': round(avg_col('OT_Segmentation_Coverage_pct'),2)
    }

    incident_resilience = {
        'MTTD_Incident_hours': round(avg_col('MTTD_Incident_hours'),2),
        'MTTR_Contain_hours': round(avg_col('MTTR_Contain_hours'),2),
        'MTTR_Recovery_hours': round(avg_col('MTTR_Recovery_hours'),2),
        'Pct_Reported_within_24h_pct': round(avg_col('Pct_Reported_within_24h_pct'),2),
        'Unplanned_Downtime_hours': round(avg_col('Unplanned_Downtime_hours'),2)
    }

    employee_awareness = {
        'Pct_Employees_Trained_pct': round(avg_col('Pct_Employees_Trained_pct'),2),
        'Phishing_Failure_pct': round(avg_col('Phishing_Failure_pct'),2),
        'Incidents_Employee_Error_cnt': int(np.nansum(ops_latest['Incidents_Employee_Error_cnt'])) if 'Incidents_Employee_Error_cnt' in ops_latest.columns else 0
    }

    product_security = {
        'Pct_New_Models_TARA_pct': round(avg_col('Pct_New_Models_TARA_pct'),2),
        'Time_Vuln_to_Patch_days': round(avg_col('Time_Vuln_to_Patch_days'),2),
        'Pct_Fleet_OTA_Patchable_pct': round(avg_col('Pct_Fleet_OTA_Patchable_pct'),2),
        'Successful_Pentest_Count': int(np.nansum(ops_latest['Successful_Pentest_Count'])) if 'Successful_Pentest_Count' in ops_latest.columns else 0
    }

    heatmap_df = snapshot.groupby(['plant','domain'])['cmmi'].mean().reset_index().pivot(index='plant', columns='domain', values='cmmi').fillna(0)

    control_avg = snapshot.groupby(['control_id','control_name'])['cmmi'].mean().reset_index()
    top5 = control_avg.sort_values('cmmi', ascending=False).head(5)
    bot5 = control_avg.sort_values('cmmi', ascending=True).head(5)

    month_agg = df.copy()
    month_agg['weighted_compliant'] = month_agg['compliant'] * month_agg['mandatory_weight']
    trends = month_agg.groupby('date').agg({'weighted_compliant':'sum','weighted_total':'sum'}).reset_index()
    trends['compliance_pct'] = trends['weighted_compliant'] / trends['weighted_total'] * 100

    return {
        'comp_score': round(float(comp_score),2),
        'avg_maturity': round(float(avg_maturity),2),
        'critical_coverage': round(float(crit_coverage),2),
        'distribution': distribution,
        'heatmap_df': heatmap_df,
        'top5': top5,
        'bot5': bot5,
        'trends': trends,
        'governance': governance,
        'supply_chain': supply_chain,
        'asset_vuln': asset_vuln,
        'incident_resilience': incident_resilience,
        'employee_awareness': employee_awareness,
        'product_security': product_security
    }

# -----------------------------
# Streamlit UI / App
# -----------------------------

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='export')
    return output.getvalue()


def main():
    st.set_page_config(page_title="NIS2 Control Maturity - Automotive Portfolio", layout='wide')
    st.title("NIS2 Control Maturity Dashboard — Automotive Plants Portfolio")
    st.markdown("Professional synthetic dataset for demo, KPIs aligned to NIS2 and maturity based on CMMI.")

    controls = generate_controls_taxonomy()
    maturity = generate_maturity_timeseries(controls)
    ops = generate_operational_metrics(maturity)

    st.sidebar.header("Filters & Controls")
    plant_select = st.sidebar.multiselect("Select Plants (multi)", options=PLANTS, default=None)
    date_range = st.sidebar.date_input("Date Range", value=(DATES.min().date(), DATES.max().date()),
                                      min_value=DATES.min().date(), max_value=DATES.max().date())
    domain_select = st.sidebar.multiselect("Domain (optional)", options=CONTROL_DOMAINS, default=None)
    show_topbottom = st.sidebar.checkbox("Show Top/Bottom 5 Controls", value=True)
    download_raw = st.sidebar.button("Download Current Data (Excel)")

    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

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

    page = st.sidebar.radio("Page", ['Overview','Plant Drill-Down','Control Drill-Down'])

    kpis = compute_kpis(mat_filtered, ops_filtered, start_date=start_date, end_date=end_date, plants_filter=plant_select)

    if page == 'Overview':
        col1, col2, col3, col4, col5, col6 = st.columns([1.2,1,1,1,1,1])
        col1.metric("NIS2 Compliance (%)", f"{kpis['comp_score']}%")
        col2.metric("Avg Control Maturity (0–5)", f"{kpis['avg_maturity']}")
        col3.metric("Critical Control Coverage (%)", f"{kpis['critical_coverage']}%")
        col4.metric("Patch SLA (%)", f"{kpis['asset_vuln']['MTTD_Vuln_Scan_hours'] if 'asset_vuln' in kpis else ''}")
        col5.metric("MTTD (hrs)", f"{kpis['incident_resilience']['MTTD_Incident_hours']}")
        col6.metric("MTTR Recover (hrs)", f"{kpis['incident_resilience']['MTTR_Recovery_hours']}")

        st.subheader("Governance & Risk — portfolio summary")
        g = kpis['governance']
        g1, g2, g3, g4 = st.columns(4)
        g1.metric('% Senior Mgmt Trained', f"{g['Pct_Senior_Mgmt_Trained_pct']}%")
        g2.metric('Time to Acknowledge/Assign Risks (hrs)', f"{g['Time_to_Ack_Assign_New_Risks_hours']}")
        g3.metric('Cybersecurity Budget % of IT/OT', f"{g['Cybersecurity_Budget_pct_of_IT_OT']}%")
        g4.metric('Crisis Plans Tested (count)', f"{g['Num_Cyber_Crisis_Plans_Tested']}")

        st.subheader('Supply Chain & Third-Party — portfolio summary')
        s = kpis['supply_chain']
        s1, s2, s3, s4 = st.columns(4)
        s1.metric('% Suppliers Contractually Compliant', f"{s['Supplier_Contractual_Compliance_pct']}%")
        s2.metric('% Suppliers Assessed (12m)', f"{s['Supplier_Assessed_pct']}%")
        s3.metric('Supplier MTTD Remediate (days)', f"{s['Supplier_MTTD_Remediate_Critical_days']}")
        s4.metric('Supplier-origin incidents (count)', f"{s['Supplier_Incidents_cnt']}")

        st.subheader('Asset & Vulnerability — portfolio summary')
        a = kpis['asset_vuln']
        a1, a2, a3, a4 = st.columns(4)
        a1.metric('% Assets Discovered/Classified', f"{a['Pct_Assets_Discovered_Classified_pct']}%")
        a2.metric('MTTD Vulnerability Scan (hrs)', f"{a['MTTD_Vuln_Scan_hours']}")
        a3.metric('MTTR Critical Vuln (days)', f"{a['MTTR_Critical_Vuln_days']}")
        a4.metric('OT Segmentation Coverage', f"{a['OT_Segmentation_Coverage_pct']}%")

        st.subheader('Incident Response & Resilience — portfolio summary')
        ir = kpis['incident_resilience']
        ir1, ir2, ir3, ir4, ir5 = st.columns(5)
        ir1.metric('MTTD (hrs)', f"{ir['MTTD_Incident_hours']}")
        ir2.metric('MTTR Contain (hrs)', f"{ir['MTTR_Contain_hours']}")
        ir3.metric('MTTR Recovery (hrs)', f"{ir['MTTR_Recovery_hours']}")
        ir4.metric('% Reported <24h', f"{ir['Pct_Reported_within_24h_pct']}%")
        ir5.metric('Unplanned Downtime (hrs)', f"{ir['Unplanned_Downtime_hours']}")

        st.subheader('Employee Awareness & Training — portfolio summary')
        ea = kpis['employee_awareness']
        ea1, ea2, ea3 = st.columns(3)
        ea1.metric('% Employees Trained', f"{ea['Pct_Employees_Trained_pct']}%")
        ea2.metric('Phishing Failure Rate', f"{ea['Phishing_Failure_pct']}%")
        ea3.metric('Incidents due to Employee Error', f"{ea['Incidents_Employee_Error_cnt']}")

        st.subheader('Product Security — portfolio summary')
        ps = kpis['product_security']
        ps1, ps2, ps3, ps4 = st.columns(4)
        ps1.metric('% New Models TARA', f"{ps['Pct_New_Models_TARA_pct']}%")
        ps2.metric('Time vuln -> patch (days)', f"{ps['Time_Vuln_to_Patch_days']}")
        ps3.metric('% Fleet OTA patchable', f"{ps['Pct_Fleet_OTA_Patchable_pct']}%")
        ps4.metric('Successful Pentests (count)', f"{ps['Successful_Pentest_Count']}")

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

            st.subheader('Governance & Risk (plant)')
            if not plant_ops.empty:
                pg = plant_ops[plant_ops['date']==plant_ops['date'].max()].iloc[0]
                p1, p2, p3, p4 = st.columns(4)
                p1.metric('% Senior Mgmt Trained', f"{pg['Pct_Senior_Mgmt_Trained_pct']}%")
                p2.metric('Time to Ack/Assign Risks (hrs)', f"{pg['Time_to_Ack_Assign_New_Risks_hours']}")
                p3.metric('Cybersecurity Budget % of IT/OT', f"{pg['Cybersecurity_Budget_pct_of_IT_OT']}%")
                p4.metric('Crisis Plans Tested', f"{pg['Num_Cyber_Crisis_Plans_Tested']}")

            st.subheader('Asset & Vulnerability (plant)')
            if not plant_ops.empty:
                pa1, pa2, pa3, pa4 = st.columns(4)
                pa1.metric('% Assets Discovered', f"{pg['Pct_Assets_Discovered_Classified_pct']}%")
                pa2.metric('MTTD Vuln Scan (hrs)', f"{pg['MTTD_Vuln_Scan_hours']}")
                pa3.metric('MTTR Critical Vuln (days)', f"{pg['MTTR_Critical_Vuln_days']}")
                pa4.metric('OT Segmentation (%)', f"{pg['OT_Segmentation_Coverage_pct']}%")

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

    if download_raw:
        df_export = mat_filtered.copy()
        st.sidebar.success('Preparing download...')
        excel_bytes = to_excel_bytes(df_export)
        st.sidebar.download_button('Download Excel', excel_bytes, file_name='nis2_maturity_export.xlsx')

    with st.expander('KPI Glossary and Calculation Notes'):
        st.markdown("""
        Governance & Risk KPIs:
        - % Senior Management Trained on Cybersecurity Responsibilities: numerator is senior staff with training certificate; denominator is total senior management headcount per plant.
        - Time to Acknowledge and Assign New Risks: measured in hours from initial risk detection logging to assigned owner in the risk register.
        - Cybersecurity Budget as % of IT/OT Budget: use financial ledger categories tagged for cybersecurity and IT/OT.
        - Number of Defined and Tested Cyber Crisis Management Plans: count of documented crisis plans with successful test evidence.

        Supply Chain KPIs:
        - % Critical Suppliers Contractually Compliant: contractual clauses present and validated.
        - % Critical Suppliers with Completed Security Assessments: questionnaire or audit completed within the last 12 months.
        - Mean Time to Remediate Critical Vulnerabilities in Supplier Components: days between disclosure to supplier and validated remediation.
        - Number of Security Incidents Originating from a Third-Party Supplier: incident records tagged with supplier origin.

        Asset & Vulnerability KPIs:
        - % of IT and OT Assets Discovered and Classified: discovery vs canonical asset registry.
        - MTTD New Vulnerabilities: time from published CVE/scan detection to inclusion in vulnerability tracker.
        - MTTR Critical Vulnerabilities: time from detection to remediation verification.
        - OT Network Segmentation Coverage: percentage of critical OT segments with enforced segmentation controls.

        Incident Response & Resilience KPIs:
        - MTTD / MTTR metrics and definitions aligned to SIEM/SOC and incident response logs.
        - % of major incidents reported within NIS2 timeline: measured against incident report submission timestamps.
        - Unplanned downtime hours: extracted from MES/SCADA downtime records correlated with incident tickets.

        Employee Awareness KPIs:
        - Employee training completion and phishing exercises must be recorded in LMS and phishing platform outputs.

        Product Security KPIs:
        - % new vehicle models with completed TARA: design-phase evidence in product security register.
        - Time from vulnerability discovery to patch availability: measure from internal bugtracker to OTA patch release.
        - % of fleet OTA patchable: device management platform coverage.
        - Successful penetration tests: count of validated penetration test engagements and positive findings addressed.
        """)

    st.sidebar.markdown("---")
    st.sidebar.caption('Expanded KPIs included. Synthetic data generated with fixed seed for reproducibility.')

if __name__ == '__main__':
    main()
