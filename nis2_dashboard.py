import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Set page configuration
st.set_page_config(
    page_title="NIS2 Compliance Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set random seed for reproducibility
np.random.seed(42)

# Constants
PLANTS = [
    "Mirafiori", "Cassino", "Melfi (SATA)", "Pomigliano (G.B. Vico)", "Modena", 
    "Atessa (SEVEL Sud)", "Poissy", "Mulhouse", "Rennes", "Sochaux", 
    "Hordain (Sevelnord)", "Eisenach", "Rüsselsheim", "Vigo", "Zaragoza", 
    "Madrid", "Mangualde", "Tychy", "Gliwice", "Trnava", 
    "Kolín", "Szentgotthárd", "Kragujevac", "Bursa (Tofaş JV)", "Luton", "Ellesmere Port"
]

CONTROL_DOMAINS = {
    "Governance & Policy": ["GP-01: Policy Framework", "GP-02: Roles & Responsibilities", "GP-03: Compliance Monitoring"],
    "Risk Management": ["RM-01: Risk Assessment", "RM-02: Risk Treatment", "RM-03: Risk Reporting"],
    "Asset & Configuration Management": ["ACM-01: Asset Inventory", "ACM-02: Configuration Baseline", "ACM-03: Change Control"],
    "Access Control & IAM": ["AC-01: User Access Management", "AC-02: Privileged Access", "AC-03: Access Reviews"],
    "Network & Segmentation": ["NS-01: Network Architecture", "NS-02: Segmentation", "NS-03: Traffic Monitoring"],
    "Vulnerability & Patch Management": ["VP-01: Vulnerability Scanning", "VP-02: Patch Management", "VP-03: Remediation Tracking"],
    "Secure Development & Change": ["SD-01: Secure Development", "SD-02: Code Review", "SD-03: Deployment Security"],
    "Logging, Monitoring & Detection": ["LM-01: Log Management", "LM-02: Monitoring Coverage", "LM-03: Threat Detection"],
    "Incident Response & Recovery": ["IR-01: Incident Response Plan", "IR-02: Incident Handling", "IR-03: Lessons Learned"],
    "Business Continuity & DR": ["BC-01: BCP Framework", "BC-02: DRP Testing", "BC-03: Recovery Objectives"],
    "Backup & Restore Validation": ["BR-01: Backup Policy", "BR-02: Backup Testing", "BR-03: Recovery Validation"],
    "Supplier/Third-Party Risk": ["SR-01: Third-Party Assessment", "SR-02: Contractual Security", "SR-03: Ongoing Monitoring"],
    "Security Awareness & Phishing": ["SA-01: Security Training", "SA-02: Phishing Simulations", "SA-03: Awareness Metrics"],
    "Physical Security (Plant)": ["PS-01: Physical Access Control", "PS-02: Surveillance", "PS-03: Perimeter Security"],
    "Cloud & SaaS Security": ["CS-01: Cloud Security Policy", "CS-02: Cloud Configuration", "CS-03: SaaS Risk Assessment"]
}

# Define mandatory controls (critical NIS2 requirements)
MANDATORY_CONTROLS = [
    "GP-01: Policy Framework", "RM-01: Risk Assessment", "ACM-01: Asset Inventory",
    "AC-01: User Access Management", "NS-02: Segmentation", "VP-02: Patch Management",
    "IR-01: Incident Response Plan", "BC-01: BCP Framework", "BR-01: Backup Policy",
    "SR-01: Third-Party Assessment", "PS-01: Physical Access Control"
]

MONTHS = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')

@st.cache_data
def generate_data():
    """Generate synthetic but realistic NIS2 compliance data."""
    records = []
    
    # Create plant performance profiles (leaders, mid-pack, laggards)
    plant_profiles = {}
    for i, plant in enumerate(PLANTS):
        if i < 5:  # Leaders
            base_maturity = np.random.uniform(3.5, 4.2)
            improvement_rate = np.random.uniform(0.05, 0.15)
        elif i < 20:  # Mid-pack
            base_maturity = np.random.uniform(2.0, 3.5)
            improvement_rate = np.random.uniform(0.02, 0.08)
        else:  # Laggards
            base_maturity = np.random.uniform(1.0, 2.5)
            improvement_rate = np.random.uniform(0.01, 0.05)
        
        plant_profiles[plant] = {
            'base_maturity': base_maturity,
            'improvement_rate': improvement_rate,
            'variability': np.random.uniform(0.1, 0.3)
        }
    
    # Generate data for each plant, control, and month
    for plant in PLANTS:
        profile = plant_profiles[plant]
        for domain, controls in CONTROL_DOMAINS.items():
            for control in controls:
                is_mandatory = control in MANDATORY_CONTROLS
                # Base maturity with domain-specific adjustments
                domain_factor = np.random.uniform(0.8, 1.2)
                control_base = profile['base_maturity'] * domain_factor
                
                for month in MONTHS:
                    # Calculate maturity with improvement over time and some randomness
                    months_passed = (month - MONTHS[0]).days / 30
                    maturity = control_base + (profile['improvement_rate'] * months_passed)
                    maturity += np.random.normal(0, profile['variability'])
                    
                    # Ensure maturity stays within bounds
                    maturity = max(0, min(5, maturity))
                    
                    records.append({
                        'Plant': plant,
                        'Domain': domain,
                        'Control': control,
                        'Month': month,
                        'CMMI_Maturity': round(maturity, 1),
                        'Mandatory': is_mandatory
                    })
    
    df = pd.DataFrame(records)
    
    # Generate operational metrics
    ops_metrics = []
    for plant in PLANTS:
        profile = plant_profiles[plant]
        for month in MONTHS:
            # MTTD and MTTR correlated with maturity
            maturity_factor = (5 - profile['base_maturity']) / 5
            
            mttd = np.random.uniform(2, 48) * maturity_factor
            mttr = np.random.uniform(0.5, 14) * maturity_factor
            incident_count = max(1, int(np.random.poisson(10 * maturity_factor)))
            
            # Other metrics with realistic ranges
            patch_sla = np.random.uniform(55, 98) - (10 * maturity_factor)
            vuln_backlog = max(0, int(np.random.uniform(20, 800) * maturity_factor))
            phishing_failure = np.random.uniform(2, 18) + (5 * maturity_factor)
            backup_pass = np.random.uniform(70, 100) - (10 * maturity_factor)
            third_party_coverage = np.random.uniform(40, 95) - (15 * maturity_factor)
            ot_inventory = np.random.uniform(50, 98) - (10 * maturity_factor)
            ics_segmentation = np.random.uniform(30, 95) - (15 * maturity_factor)
            incident_rate = np.random.uniform(0.5, 5.0) * maturity_factor
            
            ops_metrics.append({
                'Plant': plant,
                'Month': month,
                'MTTD_hours': round(mttd, 1),
                'MTTR_days': round(mttr, 1),
                'Incident_Count': incident_count,
                'Patch_SLA_Adherence': round(patch_sla, 1),
                'Vulnerability_Backlog': vuln_backlog,
                'Phishing_Failure_Rate': round(phishing_failure, 1),
                'Backup_Restore_Pass_Rate': round(backup_pass, 1),
                'Third_Party_Coverage': round(third_party_coverage, 1),
                'OT_Inventory_Completeness': round(ot_inventory, 1),
                'ICS_Segmentation_Coverage': round(ics_segmentation, 1),
                'Incident_Rate': round(incident_rate, 2)
            })
    
    ops_df = pd.DataFrame(ops_metrics)
    
    return df, ops_df

@st.cache_data
def calculate_kpis(df, ops_df, plants_filter=None, date_range=None):
    """Calculate KPIs based on filtered data."""
    # Filter data if filters are applied
    if plants_filter:
        df = df[df['Plant'].isin(plants_filter)]
        ops_df = ops_df[ops_df['Plant'].isin(plants_filter)]
    
    if date_range:
        start_date, end_date = date_range
        df = df[(df['Month'] >= start_date) & (df['Month'] <= end_date)]
        ops_df = ops_df[(ops_df['Month'] >= start_date) & (ops_df['Month'] <= end_date)]
    
    # Calculate compliance metrics
    if len(df) > 0:
        # NIS2 Compliance Score (% of controls with CMMI ≥ 3.0, weighted 2x for mandatory)
        compliant_controls = df[df['CMMI_Maturity'] >= 3.0].copy()
        compliant_controls['Weight'] = compliant_controls['Mandatory'].apply(lambda x: 2 if x else 1)
        total_weight = df['Mandatory'].apply(lambda x: 2 if x else 1).sum()
        compliance_score = (compliant_controls['Weight'].sum() / total_weight) * 100
        
        # Average Control Maturity
        avg_maturity = df['CMMI_Maturity'].mean()
        
        # Maturity distribution
        maturity_bins = [0, 1.1, 2.1, 3.1, 4.1, 5.1]
        maturity_labels = ['0-1', '2', '3', '4', '5']
        maturity_dist = pd.cut(df['CMMI_Maturity'], bins=maturity_bins, labels=maturity_labels).value_counts(normalize=True) * 100
        
        # Critical Control Coverage (% of mandatory controls at CMMI ≥ 3.0)
        mandatory_df = df[df['Mandatory'] == True]
        if len(mandatory_df) > 0:
            critical_coverage = (mandatory_df['CMMI_Maturity'] >= 3.0).mean() * 100
        else:
            critical_coverage = 0
    else:
        compliance_score = 0
        avg_maturity = 0
        maturity_dist = pd.Series([0, 0, 0, 0, 0], index=maturity_labels)
        critical_coverage = 0
    
    # Calculate operational metrics averages
    if len(ops_df) > 0:
        mttd = ops_df['MTTD_hours'].mean()
        mttr = ops_df['MTTR_days'].mean()
        patch_sla = ops_df['Patch_SLA_Adherence'].mean()
        vuln_backlog = ops_df['Vulnerability_Backlog'].mean()
        phishing_failure = ops_df['Phishing_Failure_Rate'].mean()
        backup_pass = ops_df['Backup_Restore_Pass_Rate'].mean()
        third_party_coverage = ops_df['Third_Party_Coverage'].mean()
        ot_inventory = ops_df['OT_Inventory_Completeness'].mean()
        ics_segmentation = ops_df['ICS_Segmentation_Coverage'].mean()
        incident_rate = ops_df['Incident_Rate'].mean()
    else:
        mttd = mttr = patch_sla = vuln_backlog = phishing_failure = 0
        backup_pass = third_party_coverage = ot_inventory = ics_segmentation = incident_rate = 0
    
    return {
        'compliance_score': compliance_score,
        'avg_maturity': avg_maturity,
        'maturity_dist': maturity_dist,
        'critical_coverage': critical_coverage,
        'mttd': mttd,
        'mttr': mttr,
        'patch_sla': patch_sla,
        'vuln_backlog': vuln_backlog,
        'phishing_failure': phishing_failure,
        'backup_pass': backup_pass,
        'third_party_coverage': third_party_coverage,
        'ot_inventory': ot_inventory,
        'ics_segmentation': ics_segmentation,
        'incident_rate': incident_rate
    }

def create_heatmap(df, plants_filter=None, date_range=None):
    """Create a heatmap of average maturity by plant and domain."""
    # Filter data if filters are applied
    filtered_df = df.copy()
    if plants_filter:
        filtered_df = filtered_df[filtered_df['Plant'].isin(plants_filter)]
    if date_range:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Month'] >= start_date) & (filtered_df['Month'] <= end_date)]
    
    # Calculate average maturity by plant and domain
    heatmap_data = filtered_df.groupby(['Plant', 'Domain'])['CMMI_Maturity'].mean().reset_index()
    pivot_data = heatmap_data.pivot(index='Plant', columns='Domain', values='CMMI_Maturity')
    
    # Create heatmap
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Domain", y="Plant", color="Maturity"),
        color_continuous_scale="RdYlGn",
        range_color=[0, 5],
        aspect="auto"
    )
    
    fig.update_layout(
        title="Average Maturity by Plant and Domain",
        xaxis_title="",
        yaxis_title="",
        height=600
    )
    
    return fig

def create_maturity_trend(df, plants_filter=None):
    """Create a trend line of compliance score over time."""
    # Filter data if filters are applied
    filtered_df = df.copy()
    if plants_filter:
        filtered_df = filtered_df[filtered_df['Plant'].isin(plants_filter)]
    
    # Calculate monthly compliance score
    monthly_data = []
    for month in MONTHS:
        month_data = filtered_df[filtered_df['Month'] == month]
        if len(month_data) > 0:
            compliant_controls = month_data[month_data['CMMI_Maturity'] >= 3.0].copy()
            compliant_controls['Weight'] = compliant_controls['Mandatory'].apply(lambda x: 2 if x else 1)
            total_weight = month_data['Mandatory'].apply(lambda x: 2 if x else 1).sum()
            compliance_score = (compliant_controls['Weight'].sum() / total_weight) * 100
            monthly_data.append({'Month': month, 'Compliance_Score': compliance_score})
    
    trend_df = pd.DataFrame(monthly_data)
    
    # Create trend line
    fig = px.line(
        trend_df, 
        x='Month', 
        y='Compliance_Score',
        title="NIS2 Compliance Score Trend",
        labels={'Compliance_Score': 'Compliance Score (%)', 'Month': 'Month'}
    )
    
    fig.update_layout(
        yaxis_range=[0, 100],
        height=400
    )
    
    return fig

def create_radar_chart(plant_data, overall_avg):
    """Create a radar chart showing domain maturity for a selected plant."""
    domains = list(CONTROL_DOMAINS.keys())
    plant_avgs = []
    overall_avgs = []
    
    for domain in domains:
        domain_data = plant_data[plant_data['Domain'] == domain]
        plant_avgs.append(domain_data['CMMI_Maturity'].mean())
        overall_avgs.append(overall_avg[overall_avg['Domain'] == domain]['CMMI_Maturity'].mean())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=plant_avgs + [plant_avgs[0]],
        theta=domains + [domains[0]],
        fill='toself',
        name='Selected Plant'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=overall_avgs + [overall_avgs[0]],
        theta=domains + [domains[0]],
        fill='toself',
        name='Portfolio Average'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        title="Domain Maturity Comparison",
        height=500
    )
    
    return fig

def create_control_comparison(df, control_id, plants_filter=None, date_range=None):
    """Create a comparison of control maturity across plants."""
    # Filter data if filters are applied
    filtered_df = df.copy()
    if plants_filter:
        filtered_df = filtered_df[filtered_df['Plant'].isin(plants_filter)]
    if date_range:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Month'] >= start_date) & (filtered_df['Month'] <= end_date)]
    
    # Get data for the selected control
    control_data = filtered_df[filtered_df['Control'] == control_id]
    
    # Calculate average maturity by plant
    plant_avgs = control_data.groupby('Plant')['CMMI_Maturity'].mean().reset_index()
    plant_avgs = plant_avgs.sort_values('CMMI_Maturity', ascending=True)
    
    # Create bar chart
    fig = px.bar(
        plant_avgs,
        x='CMMI_Maturity',
        y='Plant',
        orientation='h',
        title=f"Maturity for {control_id} by Plant",
        labels={'CMMI_Maturity': 'Maturity Score', 'Plant': ''}
    )
    
    fig.update_layout(
        xaxis_range=[0, 5],
        height=600
    )
    
    return fig

def main():
    # Generate data
    df, ops_df = generate_data()
    
    # Sidebar
    st.sidebar.title("NIS2 Compliance Dashboard")
    st.sidebar.markdown("### Filters")
    
    # Plant multi-select
    selected_plants = st.sidebar.multiselect(
        "Select Plants",
        options=PLANTS,
        default=PLANTS
    )
    
    # Date range selector
    min_date = df['Month'].min()
    max_date = df['Month'].max()
    selected_dates = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(selected_dates) == 2:
        date_range = (pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1]))
    else:
        date_range = None
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Overview", "Plant Drill-Down", "Control Drill-Down"])
    
    # KPI Glossary
    with st.sidebar.expander("KPI Glossary"):
        st.markdown("""
        - **NIS2 Compliance Score**: % of controls with CMMI ≥ 3.0 (mandatory controls weighted 2x)
        - **Average Control Maturity**: Mean CMMI score across all controls
        - **Critical Control Coverage**: % of mandatory controls at CMMI ≥ 3.0
        - **MTTD**: Mean Time to Detect security incidents (hours)
        - **MTTR**: Mean Time to Respond to security incidents (days)
        - **Patch SLA Adherence**: % of critical patches applied within policy window
        - **Vulnerability Backlog**: Number of vulnerabilities older than 30 days
        - **Phishing Failure Rate**: % of users failing phishing simulations
        - **Backup Restore Pass Rate**: % of successful backup restoration tests
        - **Third-Party Assessment Coverage**: % of third parties with completed security assessments
        - **OT Inventory Completeness**: % of OT assets properly inventoried
        - **ICS Segmentation Coverage**: % of ICS networks properly segmented
        - **Incident Rate**: Security incidents per 1,000 endpoints
        """)
    
    # Download button
    if st.sidebar.button("Download Current Data"):
        filtered_df = df.copy()
        if selected_plants:
            filtered_df = filtered_df[filtered_df['Plant'].isin(selected_plants)]
        if date_range:
            filtered_df = filtered_df[(filtered_df['Month'] >= date_range[0]) & (filtered_df['Month'] <= date_range[1])]
        
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="nis2_compliance_data.csv",
            mime="text/csv"
        )
    
    # Calculate KPIs
    kpis = calculate_kpis(df, ops_df, selected_plants, date_range)
    
    # Main content
    st.title("NIS2 Compliance Dashboard for Automotive Industry")
    
    if page == "Overview":
        st.header("Portfolio Overview")
        
        # KPI cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("NIS2 Compliance Score", f"{kpis['compliance_score']:.1f}%")
            st.metric("Average Maturity", f"{kpis['avg_maturity']:.1f}")
        with col2:
            st.metric("Critical Control Coverage", f"{kpis['critical_coverage']:.1f}%")
            st.metric("Patch SLA Adherence", f"{kpis['patch_sla']:.1f}%")
        with col3:
            st.metric("MTTD (hours)", f"{kpis['mttd']:.1f}")
            st.metric("MTTR (days)", f"{kpis['mttr']:.1f}")
        
        # Heatmap
        st.plotly_chart(create_heatmap(df, selected_plants, date_range), use_container_width=True)
        
        # Maturity distribution
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            maturity_dist_df = pd.DataFrame({
                'Maturity Level': kpis['maturity_dist'].index,
                'Percentage': kpis['maturity_dist'].values
            })
            fig = px.bar(
                maturity_dist_df,
                x='Maturity Level',
                y='Percentage',
                title="Maturity Distribution",
                labels={'Percentage': '% of Controls', 'Maturity Level': 'CMMI Level'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Trend line
        with dist_col2:
            st.plotly_chart(create_maturity_trend(df, selected_plants), use_container_width=True)
        
        # Top and bottom controls
        control_avgs = df.groupby('Control')['CMMI_Maturity'].mean().reset_index()
        top_controls = control_avgs.nlargest(5, 'CMMI_Maturity')
        bottom_controls = control_avgs.nsmallest(5, 'CMMI_Maturity')
        
        controls_col1, controls_col2 = st.columns(2)
        with controls_col1:
            fig = px.bar(
                top_controls,
                x='CMMI_Maturity',
                y='Control',
                orientation='h',
                title="Top 5 Controls by Maturity",
                labels={'CMMI_Maturity': 'Average Maturity', 'Control': ''}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with controls_col2:
            fig = px.bar(
                bottom_controls,
                x='CMMI_Maturity',
                y='Control',
                orientation='h',
                title="Bottom 5 Controls by Maturity",
                labels={'CMMI_Maturity': 'Average Maturity', 'Control': ''}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Plant Drill-Down":
        st.header("Plant Drill-Down")
        
        # Plant selector
        selected_plant = st.selectbox("Select Plant", PLANTS)
        
        # Filter data for selected plant
        plant_data = df[df['Plant'] == selected_plant]
        plant_ops = ops_df[ops_df['Plant'] == selected_plant]
        
        # Calculate plant-specific KPIs
        plant_kpis = calculate_kpis(plant_data, plant_ops)
        
        # Plant KPI cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("NIS2 Compliance Score", f"{plant_kpis['compliance_score']:.1f}%")
            st.metric("MTTD (hours)", f"{plant_kpis['mttd']:.1f}")
        with col2:
            st.metric("Average Maturity", f"{plant_kpis['avg_maturity']:.1f}")
            st.metric("MTTR (days)", f"{plant_kpis['mttr']:.1f}")
        with col3:
            st.metric("Critical Control Coverage", f"{plant_kpis['critical_coverage']:.1f}%")
            st.metric("Patch SLA", f"{plant_kpis['patch_sla']:.1f}%")
        with col4:
            st.metric("Phishing Failure Rate", f"{plant_kpis['phishing_failure']:.1f}%")
            st.metric("Vulnerability Backlog", f"{int(plant_kpis['vuln_backlog'])}")
        
        # Radar chart
        overall_avg = df.groupby('Domain')['CMMI_Maturity'].mean().reset_index()
        st.plotly_chart(create_radar_chart(plant_data, overall_avg), use_container_width=True)
        
        # Strongest and weakest controls
        control_avgs = plant_data.groupby('Control')['CMMI_Maturity'].mean().reset_index()
        top_controls = control_avgs.nlargest(5, 'CMMI_Maturity')
        bottom_controls = control_avgs.nsmallest(5, 'CMMI_Maturity')
        
        controls_col1, controls_col2 = st.columns(2)
        with controls_col1:
            fig = px.bar(
                top_controls,
                x='CMMI_Maturity',
                y='Control',
                orientation='h',
                title=f"Strongest Controls at {selected_plant}",
                labels={'CMMI_Maturity': 'Maturity Score', 'Control': ''}
            )
            fig.update_xaxis(range=[0, 5])
            st.plotly_chart(fig, use_container_width=True)
        
        with controls_col2:
            fig = px.bar(
                bottom_controls,
                x='CMMI_Maturity',
                y='Control',
                orientation='h',
                title=f"Weakest Controls at {selected_plant}",
                labels={'CMMI_Maturity': 'Maturity Score', 'Control': ''}
            )
            fig.update_xaxis(range=[0, 5])
            st.plotly_chart(fig, use_container_width=True)
        
        # MTTD vs MTTR scatter plot
        fig = px.scatter(
            plant_ops,
            x='MTTD_hours',
            y='MTTR_days',
            size='Incident_Count',
            color='Month',
            title="MTTD vs MTTR by Month",
            labels={'MTTD_hours': 'Mean Time to Detect (hours)', 'MTTR_days': 'Mean Time to Respond (days)'},
            hover_data=['Incident_Count']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Control Drill-Down":
        st.header("Control Drill-Down")
        
        # Domain selector
        selected_domain = st.selectbox("Select Domain", list(CONTROL_DOMAINS.keys()))
        
        # Control selector
        domain_controls = CONTROL_DOMAINS[selected_domain]
        selected_control = st.selectbox("Select Control", domain_controls)
        
        # Control comparison across plants
        st.plotly_chart(create_control_comparison(df, selected_control, selected_plants, date_range), use_container_width=True)
        
        # Time series for top and bottom plants
        control_data = df[df['Control'] == selected_control]
        plant_avgs = control_data.groupby('Plant')['CMMI_Maturity'].mean()
        top_plants = plant_avgs.nlargest(3).index.tolist()
        bottom_plants = plant_avgs.nsmallest(3).index.tolist()
        
        # Prepare time series data
        ts_data = []
        for plant in top_plants + bottom_plants:
            plant_ts = control_data[control_data['Plant'] == plant]
            for _, row in plant_ts.iterrows():
                ts_data.append({
                    'Plant': plant,
                    'Month': row['Month'],
                    'CMMI_Maturity': row['CMMI_Maturity'],
                    'Category': 'Top 3' if plant in top_plants else 'Bottom 3'
                })
        
        ts_df = pd.DataFrame(ts_data)
        
        # Create time series plot
        fig = px.line(
            ts_df,
            x='Month',
            y='CMMI_Maturity',
            color='Plant',
            line_dash='Category',
            title=f"Maturity Trend for {selected_control}",
            labels={'CMMI_Maturity': 'Maturity Score', 'Month': 'Month'}
        )
        
        fig.update_layout(
            yaxis_range=[0, 5],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
