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
    .maturity-1 { background-color: #e74c3c; color: white; }
    .maturity-2 { background-color: #f39c12; color: white; }
    .maturity-3 { background-color: #f1c40f; color: black; }
    .maturity-4 { background-color: #2ecc71; color: white; }
    .maturity-5 { background-color: #27ae60; color: white; }
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

# NIS2 Control Domains with CMMI maturity levels
control_domains = [
    "Risk Management", "Supply Chain Security", "Asset Management", 
    "Incident Response", "Business Continuity", "Employee Awareness",
    "Cryptography", "Access Control", "Network Security", "Physical Security"
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
    all_maturity_data = []
    
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
        
        # 7. Control Maturity Data (NIS2 + CMMI)
        for date in dates:
            for domain in control_domains:
                # Base maturity with some progression over time
                base_maturity = np.random.randint(1, 4)
                maturity_progress = min(5, base_maturity + i//3 + np.random.randint(-1, 2))
                
                all_maturity_data.append({
                    'Date': date,
                    'Plant': plant,
                    'Control_Domain': domain,
                    'Maturity_Level': max(1, min(5, maturity_progress))
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
            pd.DataFrame(all_product_data),
            pd.DataFrame(all_maturity_data))

# Generate all data
governance_df, supply_chain_df, asset_df, incident_df, employee_df, product_df, maturity_df = generate_comprehensive_data()

# Dashboard title
st.title("🚗 Automotive NIS2 Compliance Dashboard")
st.markdown("### Comprehensive Monitoring of Key Performance Indicators and Control Maturity for NIS2 Implementation")

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
    
    selected_control_domains = st.multiselect(
        "Select Control Domains",
        options=control_domains,
        default=control_domains
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
    st.markdown("### CMMI Maturity Levels")
    st.markdown("""
    - **Level 1: Initial** - Processes are unpredictable and reactive
    - **Level 2: Managed** - Processes are characterized for projects and are often reactive
    - **Level 3: Defined** - Processes are characterized for the organization and are proactive
    - **Level 4: Quantitatively Managed** - Processes are measured and controlled
    - **Level 5: Optimizing** - Focus on process improvement
    """)
    
    st.markdown("---")
    st.markdown("**Report Generated:**")
    st.markdown(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Filter data based on selections
def filter_data(df, plants, date_range, suppliers=None, domains=None):
    filtered_df = df[
        (df['Plant'].isin(plants)) & 
        (df['Date'] >= pd.to_datetime(date_range[0])) & 
        (df['Date'] <= pd.to_datetime(date_range[1]))
    ]
    
    if suppliers is not None and 'Supplier' in df.columns:
        filtered_df = filtered_df[filtered_df['Supplier'].isin(suppliers)]
    
    if domains is not None and 'Control_Domain' in df.columns:
        filtered_df = filtered_df[filtered_df['Control_Domain'].isin(domains)]
    
    return filtered_df

filtered_governance_df = filter_data(governance_df, selected_plants, date_range)
filtered_supply_chain_df = filter_data(supply_chain_df, selected_plants, date_range)
filtered_asset_df = filter_data(asset_df, selected_plants, date_range)
filtered_incident_df = filter_data(incident_df, selected_plants, date_range)
filtered_employee_df = filter_data(employee_df, selected_plants, date_range)
filtered_product_df = filter_data(product_df, selected_plants, date_range)
filtered_maturity_df = filter_data(maturity_df, selected_plants, date_range, domains=selected_control_domains)

# Prepare data based on view option
if view_option == "Aggregated View":
    # For aggregated view, we'll group by date and calculate means
    governance_agg = filtered_governance_df.groupby('Date').mean().reset_index()
    supply_chain_agg = filtered_supply_chain_df.groupby(['Date', 'Supplier']).mean().reset_index()
    asset_agg = filtered_asset_df.groupby('Date').mean().reset_index()
    incident_agg = filtered_incident_df.groupby('Date').mean().reset_index()
    employee_agg = filtered_employee_df.groupby('Date').mean().reset_index()
    product_agg = filtered_product_df.groupby('Date').mean().reset_index()
    maturity_agg = filtered_maturity_df.groupby(['Date', 'Control_Domain']).mean().reset_index()
    
    display_governance = governance_agg
    display_supply_chain = supply_chain_agg
    display_asset = asset_agg
    display_incident = incident_agg
    display_employee = employee_agg
    display_product = product_agg
    display_maturity = maturity_agg
else:
    # For individual view, we'll keep the plant-specific data
    display_governance = filtered_governance_df
    display_supply_chain = filtered_supply_chain_df
    display_asset = filtered_asset_df
    display_incident = filtered_incident_df
    display_employee = filtered_employee_df
    display_product = filtered_product_df
    display_maturity = filtered_maturity_df

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

# Control Maturity Section
st.markdown('<div class="section-header">Control Maturity Assessment (NIS2 + CMMI)</div>', unsafe_allow_html=True)

with st.expander("ℹ️ About Control Maturity Assessment"):
    st.markdown("""
    This section assesses the maturity of NIS2 control domains using the CMMI maturity model:
    - **Level 1: Initial** - Processes are unpredictable and reactive
    - **Level 2: Managed** - Processes are characterized for projects and are often reactive
    - **Level 3: Defined** - Processes are characterized for the organization and are proactive
    - **Level 4: Quantitatively Managed** - Processes are measured and controlled
    - **Level 5: Optimizing** - Focus on process improvement
    
    The assessment is based on implementation evidence, process documentation, and performance metrics.
    """)

# Control Maturity Heatmap
st.subheader("Control Maturity Heatmap")

# Prepare data for heatmap
if view_option == "Aggregated View":
    heatmap_data = display_maturity.groupby('Control_Domain')['Maturity_Level'].mean().reset_index()
    heatmap_data['Plant'] = 'Average'
else:
    heatmap_data = display_maturity[display_maturity['Date'] == display_maturity['Date'].max()]
    heatmap_data = heatmap_data.pivot_table(
        index='Plant', 
        columns='Control_Domain', 
        values='Maturity_Level', 
        aggfunc='mean'
    ).reset_index().melt(id_vars='Plant', var_name='Control_Domain', value_name='Maturity_Level')

# Create heatmap
if view_option == "Aggregated View":
    fig = px.imshow(
        heatmap_data.pivot_table(index='Plant', columns='Control_Domain', values='Maturity_Level'),
        title='Average Control Maturity by Domain',
        color_continuous_scale=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60'],
        zmin=1, zmax=5,
        aspect="auto"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
else:
    fig = px.imshow(
        heatmap_data.pivot_table(index='Plant', columns='Control_Domain', values='Maturity_Level'),
        title='Control Maturity by Plant and Domain',
        color_continuous_scale=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60'],
        zmin=1, zmax=5,
        aspect="auto"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Control Maturity Radar Chart
st.subheader("Control Maturity Radar Chart")

# Prepare data for radar chart
if view_option == "Aggregated View":
    radar_data = display_maturity.groupby('Control_Domain')['Maturity_Level'].mean().reset_index()
    radar_data['Plant'] = 'Average'
else:
    radar_data = display_maturity[display_maturity['Date'] == display_maturity['Date'].max()]
    radar_data = radar_data.groupby(['Plant', 'Control_Domain'])['Maturity_Level'].mean().reset_index()

# Create radar chart
if view_option == "Aggregated View":
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_data['Maturity_Level'],
        theta=radar_data['Control_Domain'],
        fill='toself',
        name='Average Maturity'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True,
        title='Average Control Maturity Radar Chart',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    fig = go.Figure()
    for plant in selected_plants:
        plant_data = radar_data[radar_data['Plant'] == plant]
        fig.add_trace(go.Scatterpolar(
            r=plant_data['Maturity_Level'],
            theta=plant_data['Control_Domain'],
            fill='toself',
            name=plant
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True,
        title='Control Maturity Radar Chart by Plant',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Control Maturity Trends
st.subheader("Control Maturity Trends Over Time")

# Prepare data for trend chart
trend_data = display_maturity.groupby(['Date', 'Control_Domain'])['Maturity_Level'].mean().reset_index()

fig = px.line(trend_data, x='Date', y='Maturity_Level', color='Control_Domain',
              title='Control Maturity Trends Over Time',
              labels={'Maturity_Level': 'Maturity Level', 'Control_Domain': 'Control Domain'})
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# Control Maturity Summary
st.subheader("Control Maturity Summary")

col1, col2, col3 = st.columns(3)

with col1:
    # Strongest controls
    strongest_controls = display_maturity.groupby('Control_Domain')['Maturity_Level'].mean().nlargest(3)
    st.markdown("### 🏆 Strongest Controls")
    for control, maturity in strongest_controls.items():
        st.markdown(f"**{control}**: {maturity:.1f}/5.0")

with col2:
    # Weakest controls
    weakest_controls = display_maturity.groupby('Control_Domain')['Maturity_Level'].mean().nsmallest(3)
    st.markdown("### ⚠️ Weakest Controls")
    for control, maturity in weakest_controls.items():
        st.markdown(f"**{control}**: {maturity:.1f}/5.0")

with col3:
    # Maturity distribution
    maturity_dist = display_maturity['Maturity_Level'].value_counts().sort_index()
    st.markdown("### 📊 Maturity Distribution")
    for level, count in maturity_dist.items():
        st.markdown(f"**Level {level}**: {count} assessments")

# Executive Summary (existing code would follow here)
# ... [The rest of your existing dashboard code would go here]

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This dashboard contains mock data for demonstration purposes only. ")
st.markdown(f"*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
