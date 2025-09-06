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
    .plant-comparison {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# List of actual automotive plants
plants = [
    "Mirafiori", "Cassino", "Melfi (SATA)", "Pomigliano (G.B. Vico)", "Modena",
    "Atessa (SEVEL Sud)", "Poissy", "Mulhouse", "Rennes", "Sochaux",
    "Hordain (Sevelnord)", "Eisenach", "Rüsselsheim", "Vigo", "Zaragoza",
    "Madrid", "Mangualde", "Tychy", "Gliwice", "Trnava",
    "Kolín", "Szentgotthárd", "Kragujevac", "Bursa (Tofaş JV)", "Luton",
    "Ellesmere Port"
]

# Generate comprehensive fake data with plant information
def generate_comprehensive_data():
    # Dates for last 12 months
    dates = pd.date_range(end=datetime.today(), periods=12, freq='M')
    
    all_governance_data = []
    all_supply_chain_data = []
    all_asset_data = []
    all_incident_data = []
    all_employee_data = []
    all_product_data = []
    
    for plant in plants:
        # Plant-specific baseline variations
        plant_factor = np.random.uniform(0.8, 1.2)
        
        # 1. Governance & Risk Management Data
        governance_data = []
        for i, date in enumerate(dates):
            governance_data.append({
                'Date': date,
                'Plant': plant,
                'Mgmt_Trained_Pct': min(100, (50 + i*5 + np.random.randint(-5, 10)) * plant_factor),
                'Cybersecurity_Budget_Pct': (5 + i*0.5 + np.random.uniform(-0.2, 0.5)) * plant_factor,
                'Tested_Plans': min(10, (2 + i + np.random.randint(0, 2)) * plant_factor),
                'Risk_Ack_Time_Days': max(1, (10 - i*0.8 + np.random.uniform(-1, 1)) / plant_factor)
            })
        
        # 2. Supply Chain & Third-Party Risk Data
        suppliers = ['Tier 1 Electronics', 'Tier 2 Software', 'Tier 1 Chassis', 'Raw Materials', 'Tier 3 Components']
        supply_chain_data = []
        for supplier in suppliers:
            base_compliance = np.random.randint(40, 70) * plant_factor
            for i, date in enumerate(dates):
                supply_chain_data.append({
                    'Date': date,
                    'Plant': plant,
                    'Supplier': supplier,
                    'Compliance_Score': min(100, (base_compliance + i*5 + np.random.randint(-5, 10)) * plant_factor),
                    'Assessment_Completed': np.random.choice([0, 1], p=[0.2, 0.8]) if i > 3 else 0
                })
        
        # 3. Asset Management & Vulnerability Data
        asset_data = []
        for i, date in enumerate(dates):
            asset_data.append({
                'Date': date,
                'Plant': plant,
                'Assets_Discovered_Pct': min(100, (70 + i*3 + np.random.randint(-5, 8)) * plant_factor),
                'Critical_Vulnerabilities': np.random.randint(5, 20) / plant_factor,
                'Remediation_Time_Days': max(10, (45 - i*3 + np.random.randint(-5, 5)) / plant_factor),
                'OT_Segmentation_Pct': min(100, (60 + i*4 + np.random.randint(-5, 8)) * plant_factor)
            })
        
        # 4. Incident Response & Resilience Data
        incident_data = []
        for i, date in enumerate(dates):
            incidents = np.random.randint(1, 8) / plant_factor
            incident_data.append({
                'Date': date,
                'Plant': plant,
                'Incidents': incidents,
                'Detection_Time_Hours': max(2, (24 - i*2 + np.random.randint(-2, 4)) / plant_factor),
                'Containment_Time_Hours': max(1, (12 - i*1.5 + np.random.randint(-1, 3)) / plant_factor),
                'Recovery_Time_Hours': max(4, (48 - i*4 + np.random.randint(-3, 6)) / plant_factor),
                'Reported_On_Time': np.random.randint(max(0, incidents-2), incidents+1),
                'Downtime_Hours': (np.random.randint(0, 10) if incidents > 0 else 0) / plant_factor
            })
        
        # 5. Employee Awareness & Training Data
        employee_data = []
        for i, date in enumerate(dates):
            employee_data.append({
                'Date': date,
                'Plant': plant,
                'Training_Completed_Pct': min(100, (60 + i*4 + np.random.randint(-5, 10)) * plant_factor),
                'Phishing_Failure_Rate': max(5, (25 - i*2 + np.random.randint(-3, 5)) / plant_factor),
                'Employee_Error_Incidents': np.random.randint(0, 5) / plant_factor
            })
        
        # 6. Product Security Data (Automotive Specific)
        product_data = []
        for i, date in enumerate(dates):
            product_data.append({
                'Date': date,
                'Plant': plant,
                'TARA_Completed_Pct': min(100, (70 + i*3 + np.random.randint(-5, 8)) * plant_factor),
                'Vuln_Patch_Time_Days': max(30, (120 - i*8 + np.random.randint(-10, 15)) / plant_factor),
                'OTA_Capable_Pct': min(100, (50 + i*5 + np.random.randint(-5, 10)) * plant_factor),
                'PenTest_Success_Rate': max(0, (30 - i*2 + np.random.randint(-5, 5)) / plant_factor)
            })
        
        # Append plant data to overall data
        all_governance_data.extend(governance_data)
        all_supply_chain_data.extend(supply_chain_data)
        all_asset_data.extend(asset_data)
        all_incident_data.extend(incident_data)
        all_employee_data.extend(employee_data)
        all_product_data.extend(product_data)
    
    return (pd.DataFrame(all_governance_data), 
            pd.DataFrame(all_supply_chain_data),
            pd.DataFrame(all_asset_data),
            pd.DataFrame(all_incident_data),
            pd.DataFrame(all_employee_data),
            pd.DataFrame(all_product_data))

# Generate all data
governance_df, supply_chain_df, asset_df, incident_df, employee_df, product_df = generate_comprehensive_data()

# Dashboard title
st.title("🚗 Automotive NIS2 Compliance Dashboard")
st.markdown("### Comprehensive Monitoring of Key Performance Indicators for NIS2 Implementation Across Production Plants")

# Sidebar
with st.sidebar:
    st.header("Dashboard Controls")
    st.markdown("---")
    
    # Region filter
    regions = {
        "Italy": ["Mirafiori", "Cassino", "Melfi (SATA)", "Pomigliano (G.B. Vico)", "Modena"],
        "France": ["Poissy", "Mulhouse", "Rennes", "Sochaux", "Hordain (Sevelnord)"],
        "Germany": ["Eisenach", "Rüsselsheim"],
        "Spain": ["Vigo", "Zaragoza", "Madrid"],
        "Portugal": ["Mangualde"],
        "Poland": ["Tychy", "Gliwice"],
        "Slovakia": ["Trnava"],
        "Czech Republic": ["Kolín"],
        "Hungary": ["Szentgotthárd"],
        "Serbia": ["Kragujevac"],
        "Turkey": ["Bursa (Tofaş JV)"],
        "UK": ["Luton", "Ellesmere Port"]
    }
    
    selected_region = st.selectbox(
        "Filter by Region",
        options=["All Regions"] + list(regions.keys())
    )
    
    # Plant selection based on region
    if selected_region == "All Regions":
        available_plants = plants
    else:
        available_plants = regions[selected_region]
    
    selected_plants = st.multiselect(
        "Select Plants",
        options=available_plants,
        default=available_plants[:3] if len(available_plants) > 3 else available_plants
    )
    
    # Select all plants option
    if st.button("Select All Plants in Region"):
        selected_plants = available_plants
    
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
    
    # View option
    view_option = st.radio(
        "View Data As:",
        ["Individual Plants", "Aggregated View"]
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
def filter_data(df, plants, date_range, suppliers=None):
    filtered_df = df[
        (df['Plant'].isin(plants)) & 
        (df['Date'] >= pd.to_datetime(date_range[0])) & 
        (df['Date'] <= pd.to_datetime(date_range[1]))
    ]
    
    if suppliers is not None and 'Supplier' in df.columns:
        filtered_df = filtered_df[filtered_df['Supplier'].isin(suppliers)]
    
    return filtered_df

filtered_governance_df = filter_data(governance_df, selected_plants, date_range)
filtered_supply_chain_df = filter_data(supply_chain_df, selected_plants, date_range, selected_suppliers)
filtered_asset_df = filter_data(asset_df, selected_plants, date_range)
filtered_incident_df = filter_data(incident_df, selected_plants, date_range)
filtered_employee_df = filter_data(employee_df, selected_plants, date_range)
filtered_product_df = filter_data(product_df, selected_plants, date_range)

# Prepare data based on view option
if view_option == "Aggregated View":
    # For aggregated view, we'll group by date and calculate means
    governance_agg = filtered_governance_df.groupby('Date').mean().reset_index()
    supply_chain_agg = filtered_supply_chain_df.groupby(['Date', 'Supplier']).mean().reset_index()
    asset_agg = filtered_asset_df.groupby('Date').mean().reset_index()
    incident_agg = filtered_incident_df.groupby('Date').mean().reset_index()
    employee_agg = filtered_employee_df.groupby('Date').mean().reset_index()
    product_agg = filtered_product_df.groupby('Date').mean().reset_index()
    
    display_governance = governance_agg
    display_supply_chain = supply_chain_agg
    display_asset = asset_agg
    display_incident = incident_agg
    display_employee = employee_agg
    display_product = product_agg
else:
    # For individual view, we'll keep the plant-specific data
    display_governance = filtered_governance_df
    display_supply_chain = filtered_supply_chain_df
    display_asset = filtered_asset_df
    display_incident = filtered_incident_df
    display_employee = filtered_employee_df
    display_product = filtered_product_df

# Calculate overall compliance based on selected plants and time range
def calculate_overall_compliance(governance_df, supply_chain_df, asset_df, incident_df, employee_df, product_df):
    # This is a simplified calculation - in a real scenario, you'd have weighted scores
    governance_score = governance_df['Mgmt_Trained_Pct'].mean() * 0.15
    supply_chain_score = supply_chain_df['Compliance_Score'].mean() * 0.20
    asset_score = (asset_df['Assets_Discovered_Pct'].mean() + (100 - asset_df['Remediation_Time_Days'].mean())) * 0.15
    incident_score = (100 - incident_df['Detection_Time_Hours'].mean()) * 0.20
    employee_score = employee_df['Training_Completed_Pct'].mean() * 0.15
    product_score = product_df['TARA_Completed_Pct'].mean() * 0.15
    
    return min(100, (governance_score + supply_chain_score + asset_score + incident_score + employee_score + product_score))

overall_compliance = calculate_overall_compliance(
    filtered_governance_df, 
    filtered_supply_chain_df, 
    filtered_asset_df, 
    filtered_incident_df, 
    filtered_employee_df, 
    filtered_product_df
)

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
    if view_option == "Aggregated View":
        current_val = display_supply_chain['Compliance_Score'].mean()
    else:
        current_val = display_supply_chain[display_supply_chain['Date'] == display_supply_chain['Date'].max()]['Compliance_Score'].mean()
    
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
    if view_option == "Aggregated View":
        current_val = display_asset['Remediation_Time_Days'].mean()
    else:
        current_val = display_asset[display_asset['Date'] == display_asset['Date'].max()]['Remediation_Time_Days'].mean()
    
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
    if view_option == "Aggregated View":
        current_val = display_incident['Detection_Time_Hours'].mean()
    else:
        current_val = display_incident[display_incident['Date'] == display_incident['Date'].max()]['Detection_Time_Hours'].mean()
    
    fig = go.Figure(go.Indicator(
        mode = "number",
        value = current_val,
        number = {'suffix': " hours"},
        title = {"text": "Incident Detection Time"},
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

# Plant Comparison Section
st.markdown('<div class="section-header">Plant Comparison</div>', unsafe_allow_html=True)

if view_option == "Individual Plants" and len(selected_plants) > 1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Plant compliance comparison
        plant_compliance = filtered_governance_df.groupby('Plant')['Mgmt_Trained_Pct'].mean().reset_index()
        fig = px.bar(plant_compliance, x='Plant', y='Mgmt_Trained_Pct', 
                     title='Management Training by Plant',
                     labels={'Mgmt_Trained_Pct': 'Training Completion (%)', 'Plant': 'Production Plant'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Plant security budget comparison
        plant_budget = filtered_governance_df.groupby('Plant')['Cybersecurity_Budget_Pct'].mean().reset_index()
        fig = px.bar(plant_budget, x='Plant', y='Cybersecurity_Budget_Pct', 
                     title='Cybersecurity Budget by Plant',
                     labels={'Cybersecurity_Budget_Pct': 'Budget (% of IT/OT)', 'Plant': 'Production Plant'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

# Regional Comparison
if selected_region == "All Regions" and view_option == "Aggregated View":
    st.markdown('<div class="section-header">Regional Comparison</div>', unsafe_allow_html=True)
    
    # Create regional data
    regional_data = []
    for region, region_plants in regions.items():
        region_governance = governance_df[governance_df['Plant'].isin(region_plants)]
        region_compliance = calculate_overall_compliance(
            region_governance,
            supply_chain_df[supply_chain_df['Plant'].isin(region_plants)],
            asset_df[asset_df['Plant'].isin(region_plants)],
            incident_df[incident_df['Plant'].isin(region_plants)],
            employee_df[employee_df['Plant'].isin(region_plants)],
            product_df[product_df['Plant'].isin(region_plants)]
        )
        regional_data.append({
            'Region': region,
            'Compliance_Score': region_compliance,
            'Plant_Count': len(region_plants)
        })
    
    regional_df = pd.DataFrame(regional_data)
    
    fig = px.bar(regional_df, x='Region', y='Compliance_Score', 
                 title='NIS2 Compliance by Region',
                 labels={'Compliance_Score': 'Compliance Score', 'Region': 'Geographic Region'})
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
    if view_option == "Aggregated View":
        current_val = display_governance['Mgmt_Trained_Pct'].mean()
    else:
        current_val = display_governance[display_governance['Date'] == display_governance['Date'].max()]['Mgmt_Trained_Pct'].mean()
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Cybersecurity Budget</div>', unsafe_allow_html=True)
    if view_option == "Aggregated View":
        current_val = display_governance['Cybersecurity_Budget_Pct'].mean()
    else:
        current_val = display_governance[display_governance['Date'] == display_governance['Date'].max()]['Cybersecurity_Budget_Pct'].mean()
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Tested Crisis Plans</div>', unsafe_allow_html=True)
    if view_option == "Aggregated View":
        current_val = display_governance['Tested_Plans'].mean()
    else:
        current_val = display_governance[display_governance['Date'] == display_governance['Date'].max()]['Tested_Plans'].mean()
    st.markdown(f'<div class="metric-value">{current_val:.1f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Risk Acknowledgement Time</div>', unsafe_allow_html=True)
    if view_option == "Aggregated View":
        current_val = display_governance['Risk_Ack_Time_Days'].mean()
    else:
        current_val = display_governance[display_governance['Date'] == display_governance['Date'].max()]['Risk_Ack_Time_Days'].mean()
    st.markdown(f'<div class="metric-value">{current_val:.1f} days</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Governance Trend Chart
if view_option == "Aggregated View":
    fig = px.line(display_governance, x='Date', y=['Mgmt_Trained_Pct', 'Cybersecurity_Budget_Pct'],
                  title='Governance Metrics Trend',
                  labels={'value': 'Percentage', 'variable': 'Metric'})
else:
    fig = px.line(display_governance, x='Date', y='Mgmt_Trained_Pct', color='Plant',
                  title='Management Training Trend by Plant',
                  labels={'Mgmt_Trained_Pct': 'Training Completion (%)', 'Plant': 'Production Plant'})
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
    if view_option == "Aggregated View":
        current_val = display_supply_chain['Compliance_Score'].mean()
    else:
        current_val = display_supply_chain[display_supply_chain['Date'] == display_supply_chain['Date'].max()]['Compliance_Score'].mean()
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Suppliers Assessed</div>', unsafe_allow_html=True)
    if view_option == "Aggregated View":
        current_val = display_supply_chain['Assessment_Completed'].mean() * 100
    else:
        current_val = display_supply_chain[display_supply_chain['Date'] == display_supply_chain['Date'].max()]['Assessment_Completed'].mean() * 100
    st.markdown(f'<div class="metric-value">{current_val:.1f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Supplier Vuln. Remediation</div>', unsafe_allow_html=True)
    # This would typically come from a different data source
    current_val = 45  # Fixed for demonstration
    st.markdown(f'<div class="metric-value">{current_val:.1f} days</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Supplier Incidents</div>', unsafe_allow_html=True)
    # This would typically come from a different data source
    current_val = 3   # Fixed for demonstration
    st.markdown(f'<div class="metric-value">{current_val:.0f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Supplier Compliance Trend Chart
if view_option == "Aggregated View":
    supplier_pivot = display_supply_chain.pivot_table(
        index='Date', columns='Supplier', values='Compliance_Score', aggfunc='mean'
    ).reset_index()
    
    fig = px.line(supplier_pivot, x='Date', y=supplier_pivot.columns[1:],
                  title='Supplier Compliance Trends',
                  labels={'value': 'Compliance Score', 'variable': 'Supplier'})
else:
    fig = px.line(display_supply_chain, x='Date', y='Compliance_Score', color='Plant',
                  title='Supplier Compliance by Plant',
                  labels={'Compliance_Score': 'Compliance Score', 'Plant': 'Production Plant'})
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

# Additional sections would follow the same pattern...

# Due to space constraints, I'm showing the pattern for the first two sections.
# The remaining sections (Asset Management, Incident Response, Employee Awareness, Product Security)
# would follow the same pattern with appropriate KPIs and visualizations.

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
