import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="NIS2 Compliance Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 24px;
        color: #1f77b4;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .kpi-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .positive {
        color: #2ecc71;
    }
    .negative {
        color: #e74c3c;
    }
    .metric-value {
        font-size: 20px;
        font-weight: 700;
    }
    .section-header {
        background-color: #1f77b4;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Generate comprehensive fake data
def generate_comprehensive_data():
    # Dates for last 12 months
    dates = pd.date_range(end=datetime.today(), periods=12, freq='M')
    
    # 1. Governance & Risk Management Data
    governance_data = []
    for i, date in enumerate(dates):
        governance_data.append({
            'Date': date,
            'Mgmt_Trained_Pct': min(100, 50 + i*5 + np.random.randint(-5, 10)),
            'Cybersecurity_Budget_Pct': 5 + i*0.5 + np.random.uniform(-0.2, 0.5),
            'Tested_Plans': min(10, 2 + i + np.random.randint(0, 2)),
            'Risk_Ack_Time_Days': max(1, 10 - i*0.8 + np.random.uniform(-1, 1))
        })
    
    # 2. Supply Chain & Third-Party Risk Data
    suppliers = ['Tier 1 Electronics', 'Tier 2 Software', 'Tier 1 Chassis', 'Raw Materials', 'Tier 3 Components']
    supply_chain_data = []
    for supplier in suppliers:
        base_compliance = np.random.randint(40, 70)
        for i, date in enumerate(dates):
            supply_chain_data.append({
                'Date': date,
                'Supplier': supplier,
                'Compliance_Score': min(100, base_compliance + i*5 + np.random.randint(-5, 10)),
                'Assessment_Completed': np.random.choice([0, 1], p=[0.2, 0.8]) if i > 3 else 0
            })
    
    # 3. Asset Management & Vulnerability Data
    asset_data = []
    for i, date in enumerate(dates):
        asset_data.append({
            'Date': date,
            'Assets_Discovered_Pct': min(100, 70 + i*3 + np.random.randint(-5, 8)),
            'Critical_Vulnerabilities': np.random.randint(5, 20),
            'Remediation_Time_Days': max(10, 45 - i*3 + np.random.randint(-5, 5)),
            'OT_Segmentation_Pct': min(100, 60 + i*4 + np.random.randint(-5, 8))
        })
    
    # 4. Incident Response & Resilience Data
    incident_data = []
    for i, date in enumerate(dates):
        incidents = np.random.randint(1, 8)
        incident_data.append({
            'Date': date,
            'Incidents': incidents,
            'Detection_Time_Hours': max(2, 24 - i*2 + np.random.randint(-2, 4)),
            'Containment_Time_Hours': max(1, 12 - i*1.5 + np.random.randint(-1, 3)),
            'Recovery_Time_Hours': max(4, 48 - i*4 + np.random.randint(-3, 6)),
            'Reported_On_Time': np.random.randint(max(0, incidents-2), incidents+1),
            'Downtime_Hours': np.random.randint(0, 10) if incidents > 0 else 0
        })
    
    # 5. Employee Awareness & Training Data
    employee_data = []
    for i, date in enumerate(dates):
        employee_data.append({
            'Date': date,
            'Training_Completed_Pct': min(100, 60 + i*4 + np.random.randint(-5, 10)),
            'Phishing_Failure_Rate': max(5, 25 - i*2 + np.random.randint(-3, 5)),
            'Employee_Error_Incidents': np.random.randint(0, 5)
        })
    
    # 6. Product Security Data (Automotive Specific)
    product_data = []
    for i, date in enumerate(dates):
        product_data.append({
            'Date': date,
            'TARA_Completed_Pct': min(100, 70 + i*3 + np.random.randint(-5, 8)),
            'Vuln_Patch_Time_Days': max(30, 120 - i*8 + np.random.randint(-10, 15)),
            'OTA_Capable_Pct': min(100, 50 + i*5 + np.random.randint(-5, 10)),
            'PenTest_Success_Rate': max(0, 30 - i*2 + np.random.randint(-5, 5))
        })
    
    return (pd.DataFrame(governance_data), 
            pd.DataFrame(supply_chain_data),
            pd.DataFrame(asset_data),
            pd.DataFrame(incident_data),
            pd.DataFrame(employee_data),
            pd.DataFrame(product_data))

# Generate all data
governance_df, supply_chain_df, asset_df, incident_df, employee_df, product_df = generate_comprehensive_data()

# Dashboard title
st.title("🚗 Automotive NIS2 Compliance Dashboard")
st.markdown("Monitoring all 24 key performance indicators for NIS2 implementation and execution")

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(governance_df['Date'].min(), governance_df['Date'].max()),
    min_value=governance_df['Date'].min(),
    max_value=governance_df['Date'].max()
)

selected_suppliers = st.sidebar.multiselect(
    "Select Suppliers",
    options=supply_chain_df['Supplier'].unique(),
    default=supply_chain_df['Supplier'].unique()
)

# Filter data based on selections
filtered_supply_chain_df = supply_chain_df[
    (supply_chain_df['Supplier'].isin(selected_suppliers)) & 
    (supply_chain_df['Date'] >= pd.to_datetime(date_range[0])) & 
    (supply_chain_df['Date'] <= pd.to_datetime(date_range[1]))
]

# Overall Compliance Score
overall_compliance = 78  # This could be calculated from all metrics

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**Overall NIS2 Compliance**")
    st.markdown(f'<div class="metric-value">{overall_compliance}%</div>', unsafe_allow_html=True)
    st.progress(overall_compliance/100)
with col2:
    st.markdown("**Critical Suppliers Compliant**")
    current_val = filtered_supply_chain_df[filtered_supply_chain_df['Date'] == filtered_supply_chain_df['Date'].max()]['Compliance_Score'].mean()
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
with col3:
    st.markdown("**Vulnerability Remediation**")
    current_val = asset_df[asset_df['Date'] == asset_df['Date'].max()]['Remediation_Time_Days'].values[0]
    st.markdown(f'<div class="metric-value">{current_val:.1f} days</div>', unsafe_allow_html=True)
with col4:
    st.markdown("**Incident Detection Time**")
    current_val = incident_df[incident_df['Date'] == incident_df['Date'].max()]['Detection_Time_Hours'].values[0]
    st.markdown(f'<div class="metric-value">{current_val:.1f} hours</div>', unsafe_allow_html=True)

# 1. Governance & Risk Management Section
st.markdown('<div class="section-header">Governance & Risk Management</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**% of Management Trained**')
    current_val = governance_df[governance_df['Date'] == governance_df['Date'].max()]['Mgmt_Trained_Pct'].values[0]
    prev_val = governance_df[governance_df['Date'] == governance_df['Date'].max() - pd.DateOffset(months=1)]['Mgmt_Trained_Pct'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Cybersecurity Budget %**')
    current_val = governance_df[governance_df['Date'] == governance_df['Date'].max()]['Cybersecurity_Budget_Pct'].values[0]
    prev_val = governance_df[governance_df['Date'] == governance_df['Date'].max() - pd.DateOffset(months=1)]['Cybersecurity_Budget_Pct'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Tested Crisis Plans**')
    current_val = governance_df[governance_df['Date'] == governance_df['Date'].max()]['Tested_Plans'].values[0]
    prev_val = governance_df[governance_df['Date'] == governance_df['Date'].max() - pd.DateOffset(months=1)]['Tested_Plans'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.0f}</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.0f}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Risk Acknowledgement Time**')
    current_val = governance_df[governance_df['Date'] == governance_df['Date'].max()]['Risk_Ack_Time_Days'].values[0]
    prev_val = governance_df[governance_df['Date'] == governance_df['Date'].max() - pd.DateOffset(months=1)]['Risk_Ack_Time_Days'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} days</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} days</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 2. Supply Chain & Third-Party Risk Section
st.markdown('<div class="section-header">Supply Chain & Third-Party Risk</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Supplier Compliance Score**')
    current_val = filtered_supply_chain_df[filtered_supply_chain_df['Date'] == filtered_supply_chain_df['Date'].max()]['Compliance_Score'].mean()
    prev_val = filtered_supply_chain_df[filtered_supply_chain_df['Date'] == filtered_supply_chain_df['Date'].max() - pd.DateOffset(months=1)]['Compliance_Score'].mean()
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Suppliers Assessed**')
    current_val = filtered_supply_chain_df[filtered_supply_chain_df['Date'] == filtered_supply_chain_df['Date'].max()]['Assessment_Completed'].mean() * 100
    prev_val = filtered_supply_chain_df[filtered_supply_chain_df['Date'] == filtered_supply_chain_df['Date'].max() - pd.DateOffset(months=1)]['Assessment_Completed'].mean() * 100
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Supplier Vuln. Remediation**')
    # This would typically come from a different data source
    current_val = 45  # Fixed for demonstration
    prev_val = 52     # Fixed for demonstration
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} days</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} days</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Supplier Incidents**')
    # This would typically come from a different data source
    current_val = 3   # Fixed for demonstration
    prev_val = 5      # Fixed for demonstration
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.0f}</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.0f}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 3. Asset Management & Vulnerability Section
st.markdown('<div class="section-header">Asset Management & Vulnerability</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Assets Discovered**')
    current_val = asset_df[asset_df['Date'] == asset_df['Date'].max()]['Assets_Discovered_Pct'].values[0]
    prev_val = asset_df[asset_df['Date'] == asset_df['Date'].max() - pd.DateOffset(months=1)]['Assets_Discovered_Pct'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Critical Vulnerabilities**')
    current_val = asset_df[asset_df['Date'] == asset_df['Date'].max()]['Critical_Vulnerabilities'].values[0]
    prev_val = asset_df[asset_df['Date'] == asset_df['Date'].max() - pd.DateOffset(months=1)]['Critical_Vulnerabilities'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.0f}</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.0f}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Vuln. Remediation Time**')
    current_val = asset_df[asset_df['Date'] == asset_df['Date'].max()]['Remediation_Time_Days'].values[0]
    prev_val = asset_df[asset_df['Date'] == asset_df['Date'].max() - pd.DateOffset(months=1)]['Remediation_Time_Days'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} days</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} days</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**OT Segmentation**')
    current_val = asset_df[asset_df['Date'] == asset_df['Date'].max()]['OT_Segmentation_Pct'].values[0]
    prev_val = asset_df[asset_df['Date'] == asset_df['Date'].max() - pd.DateOffset(months=1)]['OT_Segmentation_Pct'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 4. Incident Response & Resilience Section
st.markdown('<div class="section-header">Incident Response & Resilience</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Incident Detection Time**')
    current_val = incident_df[incident_df['Date'] == incident_df['Date'].max()]['Detection_Time_Hours'].values[0]
    prev_val = incident_df[incident_df['Date'] == incident_df['Date'].max() - pd.DateOffset(months=1)]['Detection_Time_Hours'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} hours</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} hours</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Incident Containment Time**')
    current_val = incident_df[incident_df['Date'] == incident_df['Date'].max()]['Containment_Time_Hours'].values[0]
    prev_val = incident_df[incident_df['Date'] == incident_df['Date'].max() - pd.DateOffset(months=1)]['Containment_Time_Hours'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} hours</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} hours</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Incident Recovery Time**')
    current_val = incident_df[incident_df['Date'] == incident_df['Date'].max()]['Recovery_Time_Hours'].values[0]
    prev_val = incident_df[incident_df['Date'] == incident_df['Date'].max() - pd.DateOffset(months=1)]['Recovery_Time_Hours'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} hours</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} hours</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**NIS2 Reporting Compliance**')
    current_val = incident_df[incident_df['Date'] == incident_df['Date'].max()]['Reported_On_Time'].values[0]
    total_incidents = incident_df[incident_df['Date'] == incident_df['Date'].max()]['Incidents'].values[0]
    compliance_pct = (current_val / total_incidents * 100) if total_incidents > 0 else 100
    prev_val = (incident_df[incident_df['Date'] == incident_df['Date'].max() - pd.DateOffset(months=1)]['Reported_On_Time'].values[0] / 
                incident_df[incident_df['Date'] == incident_df['Date'].max() - pd.DateOffset(months=1)]['Incidents'].values[0] * 100) if incident_df[incident_df['Date'] == incident_df['Date'].max() - pd.DateOffset(months=1)]['Incidents'].values[0] > 0 else 100
    change = compliance_pct - prev_val
    st.markdown(f'<div class="metric-value">{compliance_pct:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Production Downtime Hours**')
    current_val = incident_df[incident_df['Date'] == incident_df['Date'].max()]['Downtime_Hours'].values[0]
    prev_val = incident_df[incident_df['Date'] == incident_df['Date'].max() - pd.DateOffset(months=1)]['Downtime_Hours'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} hours</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} hours</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 5. Employee Awareness & Training Section
st.markdown('<div class="section-header">Employee Awareness & Training</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Training Completed**')
    current_val = employee_df[employee_df['Date'] == employee_df['Date'].max()]['Training_Completed_Pct'].values[0]
    prev_val = employee_df[employee_df['Date'] == employee_df['Date'].max() - pd.DateOffset(months=1)]['Training_Completed_Pct'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Phishing Test Failure Rate**')
    current_val = employee_df[employee_df['Date'] == employee_df['Date'].max()]['Phishing_Failure_Rate'].values[0]
    prev_val = employee_df[employee_df['Date'] == employee_df['Date'].max() - pd.DateOffset(months=1)]['Phishing_Failure_Rate'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Employee Error Incidents**')
    current_val = employee_df[employee_df['Date'] == employee_df['Date'].max()]['Employee_Error_Incidents'].values[0]
    prev_val = employee_df[employee_df['Date'] == employee_df['Date'].max() - pd.DateOffset(months=1)]['Employee_Error_Incidents'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.0f}</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.0f}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 6. Product Security Section
st.markdown('<div class="section-header">Product Security (Automotive Specific)</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**TARA Completion**')
    current_val = product_df[product_df['Date'] == product_df['Date'].max()]['TARA_Completed_Pct'].values[0]
    prev_val = product_df[product_df['Date'] == product_df['Date'].max() - pd.DateOffset(months=1)]['TARA_Completed_Pct'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Vuln. to Patch Time**')
    current_val = product_df[product_df['Date'] == product_df['Date'].max()]['Vuln_Patch_Time_Days'].values[0]
    prev_val = product_df[product_df['Date'] == product_df['Date'].max() - pd.DateOffset(months=1)]['Vuln_Patch_Time_Days'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} days</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} days</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**OTA-Capable Fleet**')
    current_val = product_df[product_df['Date'] == product_df['Date'].max()]['OTA_Capable_Pct'].values[0]
    prev_val = product_df[product_df['Date'] == product_df['Date'].max() - pd.DateOffset(months=1)]['OTA_Capable_Pct'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**PenTest Success Rate**')
    current_val = product_df[product_df['Date'] == product_df['Date'].max()]['PenTest_Success_Rate'].values[0]
    prev_val = product_df[product_df['Date'] == product_df['Date'].max() - pd.DateOffset(months=1)]['PenTest_Success_Rate'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Trend Visualizations
st.markdown('<div class="section-header">Trend Analysis</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Governance", "Supply Chain", "Assets", "Incidents", "Training", "Product Security"])

with tab1:
    st.line_chart(governance_df.set_index('Date'))
with tab2:
    st.line_chart(filtered_supply_chain_df.pivot_table(index='Date', columns='Supplier', values='Compliance_Score'))
with tab3:
    st.line_chart(asset_df.set_index('Date')[['Assets_Discovered_Pct', 'OT_Segmentation_Pct']])
with tab4:
    st.line_chart(incident_df.set_index('Date')[['Detection_Time_Hours', 'Containment_Time_Hours', 'Recovery_Time_Hours']])
with tab5:
    st.line_chart(employee_df.set_index('Date'))
with tab6:
    st.line_chart(product_df.set_index('Date')[['TARA_Completed_Pct', 'OTA_Capable_Pct']])

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This dashboard contains mock data for demonstration purposes only.")
st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
