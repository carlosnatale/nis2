import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="NIS2 Compliance Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 10px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    .section-header {
        background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%);
        color: white;
        padding: 12px 15px;
        border-radius: 5px;
        margin-top: 25px;
        margin-bottom: 15px;
        font-weight: 600;
        font-size: 18px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .kpi-box {
        background-color: white;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        border-left: 4px solid #3498db;
        transition: transform 0.2s;
    }
    .kpi-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .positive {
        color: #27ae60;
        font-weight: 600;
    }
    .negative {
        color: #e74c3c;
        font-weight: 600;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 14px;
        color: #7f8c8d;
        margin-bottom: 5px;
    }
    .info-text {
        background-color: #e8f4fc;
        padding: 12px;
        border-radius: 5px;
        border-left: 4px solid #3498db;
        margin-bottom: 20px;
        font-size: 14px;
    }
    .tab-container {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        margin-bottom: 20px;
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
st.markdown("### Comprehensive Monitoring of Key Performance Indicators for NIS2 Implementation")

# Sidebar
with st.sidebar:
    st.header("Dashboard Controls")
    st.markdown("---")
    
    date_range = st.date_input(
        "Select Date Range",
        value=(governance_df['Date'].min(), governance_df['Date'].max()),
        min_value=governance_df['Date'].min(),
        max_value=governance_df['Date'].max()
    )
    
    selected_suppliers = st.multiselect(
        "Select Suppliers",
        options=supply_chain_df['Supplier'].unique(),
        default=supply_chain_df['Supplier'].unique()
    )
    
    st.markdown("---")
    st.markdown("### NIS2 Overview")
    st.info("""
    The NIS2 Directive enhances cybersecurity across the EU. 
    Automotive companies must implement comprehensive security measures 
    and report significant incidents within strict timelines.
    """)
    
    st.markdown("---")
    st.markdown("**Report Generated:**")
    st.markdown(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Filter data based on selections
filtered_supply_chain_df = supply_chain_df[
    (supply_chain_df['Supplier'].isin(selected_suppliers)) & 
    (supply_chain_df['Date'] >= pd.to_datetime(date_range[0])) & 
    (supply_chain_df['Date'] <= pd.to_datetime(date_range[1]))
]

# Overall Compliance Score
overall_compliance = 78  # This could be calculated from all metrics

# Executive Summary
st.markdown("## Executive Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = overall_compliance,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Compliance"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#3498db"},
            'steps': [
                {'range': [0, 50], 'color': "#e74c3c"},
                {'range': [50, 75], 'color': "#f39c12"},
                {'range': [75, 100], 'color': "#2ecc71"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Critical Suppliers Compliance
    current_val = filtered_supply_chain_df[filtered_supply_chain_df['Date'] == filtered_supply_chain_df['Date'].max()]['Compliance_Score'].mean()
    fig = go.Figure(go.Indicator(
        mode = "number",
        value = current_val,
        number = {'suffix': "%"},
        title = {"text": "Supplier Compliance"},
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

with col3:
    # Vulnerability Remediation
    current_val = asset_df[asset_df['Date'] == asset_df['Date'].max()]['Remediation_Time_Days'].values[0]
    fig = go.Figure(go.Indicator(
        mode = "number",
        value = current_val,
        number = {'suffix': " days"},
        title = {"text": "Avg. Remediation Time"},
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

with col4:
    # Incident Detection Time
    current_val = incident_df[incident_df['Date'] == incident_df['Date'].max()]['Detection_Time_Hours'].values[0]
    fig = go.Figure(go.Indicator(
        mode = "number",
        value = current_val,
        number = {'suffix': " hours"},
        title = {"text": "Incident Detection Time"},
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

# KPI Sections with detailed explanations
# 1. Governance & Risk Management Section
st.markdown('<div class="section-header">Governance & Risk Management</div>', unsafe_allow_html=True)

with st.expander("ℹ️ About these KPIs"):
    st.markdown("""
    These KPIs measure the organization's cybersecurity governance maturity:
    - **Management Training**: Percentage of senior management trained on cybersecurity responsibilities
    - **Budget Allocation**: Cybersecurity budget as a percentage of total IT/OT budget
    - **Crisis Plans**: Number of defined and tested cyber crisis management plans
    - **Risk Acknowledgement**: Time to acknowledge and assign new risks
    """)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Management Trained</div>', unsafe_allow_html=True)
    current_val = governance_df[governance_df['Date'] == governance_df['Date'].max()]['Mgmt_Trained_Pct'].values[0]
    prev_val = governance_df[governance_df['Date'] == governance_df['Date'].max() - pd.DateOffset(months=1)]['Mgmt_Trained_Pct'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Cybersecurity Budget</div>', unsafe_allow_html=True)
    current_val = governance_df[governance_df['Date'] == governance_df['Date'].max()]['Cybersecurity_Budget_Pct'].values[0]
    prev_val = governance_df[governance_df['Date'] == governance_df['Date'].max() - pd.DateOffset(months=1)]['Cybersecurity_Budget_Pct'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Tested Crisis Plans</div>', unsafe_allow_html=True)
    current_val = governance_df[governance_df['Date'] == governance_df['Date'].max()]['Tested_Plans'].values[0]
    prev_val = governance_df[governance_df['Date'] == governance_df['Date'].max() - pd.DateOffset(months=1)]['Tested_Plans'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.0f}</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.0f}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Risk Acknowledgement Time</div>', unsafe_allow_html=True)
    current_val = governance_df[governance_df['Date'] == governance_df['Date'].max()]['Risk_Ack_Time_Days'].values[0]
    prev_val = governance_df[governance_df['Date'] == governance_df['Date'].max() - pd.DateOffset(months=1)]['Risk_Ack_Time_Days'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} days</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} days</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Governance Trend Chart
fig = px.line(governance_df, x='Date', y=['Mgmt_Trained_Pct', 'Cybersecurity_Budget_Pct'],
              title='Governance Metrics Trend',
              labels={'value': 'Percentage', 'variable': 'Metric'})
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

# 2. Supply Chain & Third-Party Risk Section
st.markdown('<div class="section-header">Supply Chain & Third-Party Risk</div>', unsafe_allow_html=True)

with st.expander("ℹ️ About these KPIs"):
    st.markdown("""
    These KPIs measure cybersecurity across the supply chain:
    - **Supplier Compliance**: Percentage of critical suppliers compliant with security requirements
    - **Supplier Assessments**: Percentage of suppliers with completed security assessments
    - **Vulnerability Remediation**: Time to remediate vulnerabilities from suppliers
    - **Supplier Incidents**: Number of security incidents originating from suppliers
    """)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Supplier Compliance Score</div>', unsafe_allow_html=True)
    current_val = filtered_supply_chain_df[filtered_supply_chain_df['Date'] == filtered_supply_chain_df['Date'].max()]['Compliance_Score'].mean()
    prev_val = filtered_supply_chain_df[filtered_supply_chain_df['Date'] == filtered_supply_chain_df['Date'].max() - pd.DateOffset(months=1)]['Compliance_Score'].mean()
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Suppliers Assessed</div>', unsafe_allow_html=True)
    current_val = filtered_supply_chain_df[filtered_supply_chain_df['Date'] == filtered_supply_chain_df['Date'].max()]['Assessment_Completed'].mean() * 100
    prev_val = filtered_supply_chain_df[filtered_supply_chain_df['Date'] == filtered_supply_chain_df['Date'].max() - pd.DateOffset(months=1)]['Assessment_Completed'].mean() * 100
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Supplier Vuln. Remediation</div>', unsafe_allow_html=True)
    current_val = 45  # Fixed for demonstration
    prev_val = 52     # Fixed for demonstration
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} days</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} days</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Supplier Incidents</div>', unsafe_allow_html=True)
    current_val = 3   # Fixed for demonstration
    prev_val = 5      # Fixed for demonstration
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.0f}</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.0f}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Supplier Compliance Trend Chart
supplier_pivot = filtered_supply_chain_df.pivot_table(
    index='Date', columns='Supplier', values='Compliance_Score', aggfunc='mean'
).reset_index()

fig = px.line(supplier_pivot, x='Date', y=supplier_pivot.columns[1:],
              title='Supplier Compliance Trends',
              labels={'value': 'Compliance Score', 'variable': 'Supplier'})
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

# 3. Asset Management & Vulnerability Section
st.markdown('<div class="section-header">Asset Management & Vulnerability</div>', unsafe_allow_html=True)

with st.expander("ℹ️ About these KPIs"):
    st.markdown("""
    These KPIs measure how well the organization manages its assets and vulnerabilities:
    - **Assets Discovered**: Percentage of IT and OT assets discovered and classified
    - **Critical Vulnerabilities**: Number of critical vulnerabilities detected
    - **Remediation Time**: Mean time to remediate critical vulnerabilities
    - **OT Segmentation**: Percentage of critical OT segments properly isolated
    """)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Assets Discovered</div>', unsafe_allow_html=True)
    current_val = asset_df[asset_df['Date'] == asset_df['Date'].max()]['Assets_Discovered_Pct'].values[0]
    prev_val = asset_df[asset_df['Date'] == asset_df['Date'].max() - pd.DateOffset(months=1)]['Assets_Discovered_Pct'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Critical Vulnerabilities</div>', unsafe_allow_html=True)
    current_val = asset_df[asset_df['Date'] == asset_df['Date'].max()]['Critical_Vulnerabilities'].values[0]
    prev_val = asset_df[asset_df['Date'] == asset_df['Date'].max() - pd.DateOffset(months=1)]['Critical_Vulnerabilities'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.0f}</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.0f}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Vuln. Remediation Time</div>', unsafe_allow_html=True)
    current_val = asset_df[asset_df['Date'] == asset_df['Date'].max()]['Remediation_Time_Days'].values[0]
    prev_val = asset_df[asset_df['Date'] == asset_df['Date'].max() - pd.DateOffset(months=1)]['Remediation_Time_Days'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} days</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} days</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">OT Segmentation</div>', unsafe_allow_html=True)
    current_val = asset_df[asset_df['Date'] == asset_df['Date'].max()]['OT_Segmentation_Pct'].values[0]
    prev_val = asset_df[asset_df['Date'] == asset_df['Date'].max() - pd.DateOffset(months=1)]['OT_Segmentation_Pct'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Vulnerability Management Chart
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(x=asset_df['Date'], y=asset_df['Critical_Vulnerabilities'], name="Critical Vulnerabilities"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=asset_df['Date'], y=asset_df['Remediation_Time_Days'], name="Remediation Time (Days)"),
    secondary_y=True,
)
fig.update_layout(
    title_text="Vulnerability Management",
    height=300
)
fig.update_yaxes(title_text="Critical Vulnerabilities", secondary_y=False)
fig.update_yaxes(title_text="Remediation Time (Days)", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)

# 4. Incident Response & Resilience Section
st.markdown('<div class="section-header">Incident Response & Resilience</div>', unsafe_allow_html=True)

with st.expander("ℹ️ About these KPIs"):
    st.markdown("""
    These KPIs measure the organization's ability to respond to and recover from incidents:
    - **Detection Time**: Mean time to detect security incidents
    - **Containment Time**: Mean time to contain incidents
    - **Recovery Time**: Mean time to recover from incidents
    - **Reporting Compliance**: Percentage of incidents reported within NIS2 timelines
    - **Downtime**: Unplanned downtime hours attributed to security incidents
    """)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Incident Detection Time</div>', unsafe_allow_html=True)
    current_val = incident_df[incident_df['Date'] == incident_df['Date'].max()]['Detection_Time_Hours'].values[0]
    prev_val = incident_df[incident_df['Date'] == incident_df['Date'].max() - pd.DateOffset(months=1)]['Detection_Time_Hours'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} hours</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} hours</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Incident Containment Time</div>', unsafe_allow_html=True)
    current_val = incident_df[incident_df['Date'] == incident_df['Date'].max()]['Containment_Time_Hours'].values[0]
    prev_val = incident_df[incident_df['Date'] == incident_df['Date'].max() - pd.DateOffset(months=1)]['Containment_Time_Hours'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} hours</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} hours</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Incident Recovery Time</div>', unsafe_allow_html=True)
    current_val = incident_df[incident_df['Date'] == incident_df['Date'].max()]['Recovery_Time_Hours'].values[0]
    prev_val = incident_df[incident_df['Date'] == incident_df['Date'].max() - pd.DateOffset(months=1)]['Recovery_Time_Hours'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} hours</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} hours</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">NIS2 Reporting Compliance</div>', unsafe_allow_html=True)
    current_val = incident_df[incident_df['Date'] == incident_df['Date'].max()]['Reported_On_Time'].values[0]
    total_incidents = incident_df[incident_df['Date'] == incident_df['Date'].max()]['Incidents'].values[0]
    compliance_pct = (current_val / total_incidents * 100) if total_incidents > 0 else 100
    prev_val = (incident_df[incident_df['Date'] == incident_df['Date'].max() - pd.DateOffset(months=1)]['Reported_On_Time'].values[0] / 
                incident_df[incident_df['Date'] == incident_df['Date'].max() - pd.DateOffset(months=1)]['Incidents'].values[0] * 100) if incident_df[incident_df['Date'] == incident_df['Date'].max() - pd.DateOffset(months=1)]['Incidents'].values[0] > 0 else 100
    change = compliance_pct - prev_val
    st.markdown(f'<div class="metric-value">{compliance_pct:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Additional KPI for this section
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Production Downtime Hours</div>', unsafe_allow_html=True)
    current_val = incident_df[incident_df['Date'] == incident_df['Date'].max()]['Downtime_Hours'].values[0]
    prev_val = incident_df[incident_df['Date'] == incident_df['Date'].max() - pd.DateOffset(months=1)]['Downtime_Hours'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} hours</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} hours</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Incident Response Times Chart
response_df = incident_df[['Date', 'Detection_Time_Hours', 'Containment_Time_Hours', 'Recovery_Time_Hours']].melt(
    id_vars='Date', var_name='Metric', value_name='Hours'
)

fig = px.line(response_df, x='Date', y='Hours', color='Metric',
              title='Incident Response Times',
              labels={'Hours': 'Time (Hours)', 'Metric': 'Response Phase'})
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

# 5. Employee Awareness & Training Section
st.markdown('<div class="section-header">Employee Awareness & Training</div>', unsafe_allow_html=True)

with st.expander("ℹ️ About these KPIs"):
    st.markdown("""
    These KPIs measure the effectiveness of security awareness and training programs:
    - **Training Completion**: Percentage of employees completing mandatory training
    - **Phishing Failure Rate**: Percentage of employees failing phishing tests
    - **Employee Error Incidents**: Number of security incidents linked to employee error
    """)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Training Completed</div>', unsafe_allow_html=True)
    current_val = employee_df[employee_df['Date'] == employee_df['Date'].max()]['Training_Completed_Pct'].values[0]
    prev_val = employee_df[employee_df['Date'] == employee_df['Date'].max() - pd.DateOffset(months=1)]['Training_Completed_Pct'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Phishing Test Failure Rate</div>', unsafe_allow_html=True)
    current_val = employee_df[employee_df['Date'] == employee_df['Date'].max()]['Phishing_Failure_Rate'].values[0]
    prev_val = employee_df[employee_df['Date'] == employee_df['Date'].max() - pd.DateOffset(months=1)]['Phishing_Failure_Rate'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Employee Error Incidents</div>', unsafe_allow_html=True)
    current_val = employee_df[employee_df['Date'] == employee_df['Date'].max()]['Employee_Error_Incidents'].values[0]
    prev_val = employee_df[employee_df['Date'] == employee_df['Date'].max() - pd.DateOffset(months=1)]['Employee_Error_Incidents'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.0f}</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.0f}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Employee Training Chart
fig = px.line(employee_df, x='Date', y=['Training_Completed_Pct', 'Phishing_Failure_Rate'],
              title='Employee Training & Awareness Metrics',
              labels={'value': 'Percentage', 'variable': 'Metric'})
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

# 6. Product Security Section
st.markdown('<div class="section-header">Product Security (Automotive Specific)</div>', unsafe_allow_html=True)

with st.expander("ℹ️ About these KPIs"):
    st.markdown("""
    These KPIs measure the security of automotive products and connected services:
    - **TARA Completion**: Percentage of new vehicle models with completed Threat Analysis and Risk Assessment
    - **Vulnerability to Patch Time**: Time from vulnerability discovery to patch availability
    - **OTA Capable Fleet**: Percentage of connected vehicle fleet patchable Over-The-Air
    - **PenTest Success Rate**: Percentage of successful penetration tests on vehicle systems
    """)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">TARA Completion</div>', unsafe_allow_html=True)
    current_val = product_df[product_df['Date'] == product_df['Date'].max()]['TARA_Completed_Pct'].values[0]
    prev_val = product_df[product_df['Date'] == product_df['Date'].max() - pd.DateOffset(months=1)]['TARA_Completed_Pct'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Vuln. to Patch Time</div>', unsafe_allow_html=True)
    current_val = product_df[product_df['Date'] == product_df['Date'].max()]['Vuln_Patch_Time_Days'].values[0]
    prev_val = product_df[product_df['Date'] == product_df['Date'].max() - pd.DateOffset(months=1)]['Vuln_Patch_Time_Days'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f} days</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} days</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">OTA-Capable Fleet</div>', unsafe_allow_html=True)
    current_val = product_df[product_df['Date'] == product_df['Date'].max()]['OTA_Capable_Pct'].values[0]
    prev_val = product_df[product_df['Date'] == product_df['Date'].max() - pd.DateOffset(months=1)]['OTA_Capable_Pct'].values[0]
    change = current_val - prev_val
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">PenTest Success Rate</div>', unsafe_allow_html=True)
    current_val = product_df[product_df['Date'] == product_df['Date'].max()]['PenTest_Success_Rate'].values[0]
    prev_val = product_df[product_df['Date'] == product_df['Date'].max() - pd.DateOffset(months=1)]['PenTest_Success_Rate'].values[0]
    change = prev_val - current_val  # Lower is better
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Product Security Chart
fig = px.line(product_df, x='Date', y=['TARA_Completed_Pct', 'OTA_Capable_Pct'],
              title='Product Security Metrics',
              labels={'value': 'Percentage', 'variable': 'Metric'})
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

# Summary and Recommendations
st.markdown('<div class="section-header">Summary & Recommendations</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Strengths")
    st.success("""
    - **Strong incident response capabilities**: Detection and containment times are improving
    - **Good progress on governance**: Management training and budget allocation are on track
    - **Excellent security-by-design**: TARA completion for new models is at 100%
    """)

with col2:
    st.markdown("### ⚠️ Areas for Improvement")
    st.warning("""
    - **Supplier security**: Critical supplier compliance needs attention
    - **Vulnerability patching**: Vehicle vulnerability to patch time is too long
    - **OT segmentation**: Not all critical production segments are properly isolated
    """)

st.markdown("### 📋 Recommended Actions")
st.info("""
1. **Implement a supplier security program** with mandatory requirements and regular audits
2. **Streamline the patch management process** for vehicle software to reduce time-to-patch
3. **Accelerate OT network segmentation** for all critical production systems
4. **Enhance employee training** with more frequent phishing simulations
5. **Develop playbooks** for supply chain security incidents
""")

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This dashboard contains mock data for demonstration purposes only. ")
st.markdown(f"*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
