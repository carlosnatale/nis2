import streamlit as st
import pandas as pd
import numpy as np
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
</style>
""", unsafe_allow_html=True)

# Generate fake data
def generate_fake_data():
    # Dates for last 6 months
    dates = pd.date_range(end=datetime.today(), periods=6, freq='M')
    
    # Supplier compliance data
    suppliers = ['Tier 1 Electronics', 'Tier 2 Software', 'Tier 1 Chassis', 'Raw Materials', 'Tier 3 Components']
    compliance_data = []
    for supplier in suppliers:
        base_compliance = np.random.randint(40, 80)
        for i, date in enumerate(dates):
            compliance_data.append({
                'Date': date,
                'Supplier': supplier,
                'Compliance_Score': min(100, base_compliance + i*10 + np.random.randint(-5, 10))
            })
    
    # Vulnerability data
    vuln_data = []
    for i, date in enumerate(dates):
        vuln_data.append({
            'Date': date,
            'Critical_Vulnerabilities': np.random.randint(5, 20),
            'Remediation_Time_Days': max(10, 45 - i*5 + np.random.randint(-5, 5))
        })
    
    # Incident response data
    incident_data = []
    for i, date in enumerate(dates):
        incident_data.append({
            'Date': date,
            'Incidents': np.random.randint(1, 8),
            'Detection_Time_Hours': max(2, 24 - i*3 + np.random.randint(-2, 4)),
            'Containment_Time_Hours': max(1, 12 - i*2 + np.random.randint(-1, 3))
        })
    
    return pd.DataFrame(compliance_data), pd.DataFrame(vuln_data), pd.DataFrame(incident_data)

# Generate data
supplier_df, vuln_df, incident_df = generate_fake_data()

# Dashboard title
st.title("🚗 Automotive NIS2 Compliance Dashboard")
st.markdown("Monitoring key performance indicators for NIS2 implementation and execution")

# Sidebar filters
st.sidebar.header("Filters")
selected_suppliers = st.sidebar.multiselect(
    "Select Suppliers",
    options=supplier_df['Supplier'].unique(),
    default=supplier_df['Supplier'].unique()
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(supplier_df['Date'].min(), supplier_df['Date'].max()),
    min_value=supplier_df['Date'].min(),
    max_value=supplier_df['Date'].max()
)

# Filter data based on selections
filtered_supplier_df = supplier_df[
    (supplier_df['Supplier'].isin(selected_suppliers)) & 
    (supplier_df['Date'] >= pd.to_datetime(date_range[0])) & 
    (supplier_df['Date'] <= pd.to_datetime(date_range[1]))
]

# KPI Overview Section
st.header("📊 KPI Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Supplier Compliance Score**')
    current_avg = filtered_supplier_df[filtered_supplier_df['Date'] == filtered_supplier_df['Date'].max()]['Compliance_Score'].mean()
    prev_avg = filtered_supplier_df[filtered_supplier_df['Date'] == filtered_supplier_df['Date'].max() - pd.DateOffset(months=1)]['Compliance_Score'].mean()
    change = current_avg - prev_avg
    st.markdown(f'<div class="metric-value">{current_avg:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f}%</span> from previous month', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Vulnerability Remediation**')
    current_remediation = vuln_df[vuln_df['Date'] == vuln_df['Date'].max()]['Remediation_Time_Days'].values[0]
    prev_remediation = vuln_df[vuln_df['Date'] == vuln_df['Date'].max() - pd.DateOffset(months=1)]['Remediation_Time_Days'].values[0]
    change = prev_remediation - current_remediation
    st.markdown(f'<div class="metric-value">{current_remediation:.1f} days</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} days</span> from previous month', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**Incident Detection Time**')
    current_detection = incident_df[incident_df['Date'] == incident_df['Date'].max()]['Detection_Time_Hours'].values[0]
    prev_detection = incident_df[incident_df['Date'] == incident_df['Date'].max() - pd.DateOffset(months=1)]['Detection_Time_Hours'].values[0]
    change = prev_detection - current_detection
    st.markdown(f'<div class="metric-value">{current_detection:.1f} hours</div>', unsafe_allow_html=True)
    st.markdown(f'{"↗️" if change >= 0 else "↘️"} <span class={"positive" if change >= 0 else "negative"}>{change:+.1f} hours</span> from previous month', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
    st.markdown('**NIS2 Overall Compliance**')
    compliance_score = 78  # Fixed value for this mockup
    st.markdown(f'<div class="metric-value">{compliance_score}%</div>', unsafe_allow_html=True)
    # Create a progress bar
    st.progress(compliance_score/100)
    st.markdown('</div>', unsafe_allow_html=True)

# Charts Section
st.header("📈 Trend Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Supplier Compliance Over Time")
    # Pivot data for easier charting
    pivot_df = filtered_supplier_df.pivot_table(
        index='Date', 
        columns='Supplier', 
        values='Compliance_Score'
    ).reset_index()
    
    # Create line chart using Streamlit
    st.line_chart(pivot_df.set_index('Date'))

with col2:
    st.subheader("Vulnerability Management")
    
    # Create two columns for the two charts
    col2_1, col2_2 = st.columns(2)
    
    with col2_1:
        st.markdown("**Critical Vulnerabilities**")
        st.bar_chart(vuln_df.set_index('Date')['Critical_Vulnerabilities'])
    
    with col2_2:
        st.markdown("**Remediation Time (Days)**")
        st.line_chart(vuln_df.set_index('Date')['Remediation_Time_Days'])

# Incident Response Section
st.header("🚨 Incident Response Metrics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Incident Trends")
    st.bar_chart(incident_df.set_index('Date')['Incidents'])

with col2:
    st.subheader("Response Times")
    response_df = incident_df[['Date', 'Detection_Time_Hours', 'Containment_Time_Hours']].set_index('Date')
    st.line_chart(response_df)

# Detailed Data Section
st.header("📋 Detailed Data View")

tab1, tab2, tab3 = st.tabs(["Supplier Compliance", "Vulnerability Data", "Incident Data"])

with tab1:
    st.dataframe(filtered_supplier_df.pivot_table(
        index='Date', 
        columns='Supplier', 
        values='Compliance_Score'
    ), use_container_width=True)

with tab2:
    st.dataframe(vuln_df, use_container_width=True)

with tab3:
    st.dataframe(incident_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This dashboard contains mock data for demonstration purposes only.")
st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
