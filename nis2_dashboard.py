import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Set a fixed random seed for reproducibility
random.seed(42)
np.random.seed(42)

# --- 1. CONFIGURATION & DATA GENERATION ---

# Define the constants
PLANTS = [
    'Mirafiori', 'Cassino', 'Melfi (SATA)', 'Pomigliano (G.B. Vico)', 'Modena',
    'Atessa (SEVEL Sud)', 'Poissy', 'Mulhouse', 'Rennes', 'Sochaux',
    'Hordain (Sevelnord)', 'Eisenach', 'Rüsselsheim', 'Vigo', 'Zaragoza',
    'Madrid', 'Mangualde', 'Tychy', 'Gliwice', 'Trnava', 'Kolín',
    'Szentgotthárd', 'Kragujevac', 'Bursa (Tofaş JV)', 'Luton', 'Ellesmere Port'
]

CONTROL_DOMAINS = {
    '1. Governance & Policy': ['GvP-01', 'GvP-02', 'GvP-03', 'GvP-04'],
    '2. Risk Management': ['RM-01', 'RM-02', 'RM-03'],
    '3. Asset & Configuration Management (IT/OT)': ['ACM-01', 'ACM-02', 'ACM-03', 'ACM-04'],
    '4. Access Control & IAM': ['AC-01', 'AC-02', 'AC-03'],
    '5. Network & Segmentation (incl. OT/ICS)': ['NS-01', 'NS-02', 'NS-03', 'NS-04'],
    '6. Vulnerability & Patch Management': ['VPM-01', 'VPM-02', 'VPM-03'],
    '7. Secure Development & Change': ['SDC-01', 'SDC-02', 'SDC-03'],
    '8. Logging, Monitoring & Detection (SOC)': ['LMD-01', 'LMD-02', 'LMD-03', 'LMD-04', 'LMD-05'],
    '9. Incident Response & Recovery': ['IRR-01', 'IRR-02', 'IRR-03'],
    '10. Business Continuity & DR': ['BCDR-01', 'BCDR-02', 'BCDR-03'],
    '11. Backup & Restore Validation': ['BRV-01', 'BRV-02'],
    '12. Supplier/Third-Party Risk': ['STP-01', 'STP-02', 'STP-03'],
    '13. Security Awareness & Phishing': ['SAP-01', 'SAP-02'],
    '14. Physical Security (Plant)': ['PS-01', 'PS-02', 'PS-03'],
    '15. Cloud & SaaS Security': ['CSS-01', 'CSS-02', 'CSS-03']
}

CONTROL_DESCRIPTIONS = {
    'GvP-01': 'Cybersecurity strategy defined and approved.',
    'GvP-02': 'Roles & responsibilities for security are clear.',
    'GvP-03': 'CISO reports to executive management.',
    'GvP-04': 'Regular security reviews by leadership.',
    'RM-01': 'Regular cybersecurity risk assessments conducted.',
    'RM-02': 'Risk register is actively maintained.',
    'RM-03': 'Risk treatment plans are documented and tracked.',
    'ACM-01': 'Inventory of IT assets is complete.',
    'ACM-02': 'Inventory of OT assets is complete.',
    'ACM-03': 'Configuration baselines are enforced.',
    'ACM-04': 'Unmanaged devices are detected & controlled.',
    'AC-01': 'User access reviews are performed quarterly.',
    'AC-02': 'Privileged access is strictly controlled.',
    'AC-03': 'Multi-factor authentication is enforced.',
    'NS-01': 'Network architecture is documented.',
    'NS-02': 'Critical OT networks are segmented from IT.',
    'NS-03': 'Firewall rules are regularly audited.',
    'NS-04': 'Wireless networks are secured.',
    'VPM-01': 'Vulnerability scanning is performed regularly.',
    'VPM-02': 'Patch management process is defined and followed.',
    'VPM-03': 'Critical patches are deployed within SLA.',
    'SDC-01': 'Secure coding principles are defined.',
    'SDC-02': 'Application security testing is integrated.',
    'SDC-03': 'Changes are reviewed and approved.',
    'LMD-01': 'Centralized logging is implemented.',
    'LMD-02': 'Security events are monitored 24/7 (SOC).',
    'LMD-03': 'Key security events trigger alerts.',
    'LMD-04': 'Threat intelligence is consumed.',
    'LMD-05': 'Endpoint Detection & Response (EDR) is deployed.',
    'IRR-01': 'Incident response plan is documented & tested.',
    'IRR-02': 'Incidents are classified and reported.',
    'IRR-03': 'Lessons learned from incidents are documented.',
    'BCDR-01': 'Business continuity plan is documented.',
    'BCDR-02': 'Disaster recovery plan is documented.',
    'BCDR-03': 'Plans are tested periodically.',
    'BRV-01': 'Backups are performed routinely.',
    'BRV-02': 'Backup restores are tested monthly.',
    'STP-01': 'Third-party risk assessments are conducted.',
    'STP-02': 'Security clauses are in contracts.',
    'STP-03': 'Supplier performance is monitored.',
    'SAP-01': 'Security awareness training is mandatory.',
    'SAP-02': 'Phishing simulation campaigns are run.',
    'PS-01': 'Physical access to plants is controlled.',
    'PS-02': 'CCTV is deployed and monitored.',
    'PS-03': 'Visitor policies are enforced.',
    'CSS-01': 'Cloud security posture is managed.',
    'CSS-02': 'SaaS applications are vetted.',
    'CSS-03': 'Cloud infrastructure is monitored.'
}

# NIS2 Mandatory Controls (examples)
MANDATORY_CONTROLS = {
    'GvP-01', 'RM-01', 'ACM-02', 'AC-03', 'NS-02', 'VPM-03', 'IRR-01', 'BCDR-01',
    'BRV-02', 'STP-01', 'SAP-02', 'PS-01', 'CSS-01', 'LMD-02', 'LMD-05'
}

# Data generation parameters
START_DATE = datetime.now() - timedelta(days=365)
MONTHS = pd.date_range(start=START_DATE, periods=12, freq='MS')
TOTAL_CONTROLS = sum(len(ids) for ids in CONTROL_DOMAINS.values())

@st.cache_data
def generate_data():
    """
    Generates a synthetic dataset for NIS2 KPIs and CMMI maturity.
    """
    data_points = []
    kpi_data = []

    # Assign plant maturity profiles
    plant_profiles = {
        'leader': ['Modena', 'Rüsselsheim', 'Poissy', 'Trnava'],
        'mid-pack': [p for p in PLANTS if p not in ['Modena', 'Rüsselsheim', 'Poissy', 'Trnava', 'Mulhouse', 'Madrid', 'Kragujevac', 'Bursa (Tofaş JV)']],
        'laggard': ['Mulhouse', 'Madrid', 'Kragujevac', 'Bursa (Tofaş JV)']
    }

    # Generate control maturity data
    for month in MONTHS:
        for plant in PLANTS:
            profile = next(key for key, value in plant_profiles.items() if plant in value)
            
            for domain, control_ids in CONTROL_DOMAINS.items():
                for control_id in control_ids:
                    
                    # Base maturity varies by profile
                    if profile == 'leader':
                        base_maturity = np.random.uniform(3.5, 4.2)
                    elif profile == 'mid-pack':
                        base_maturity = np.random.uniform(2.5, 3.5)
                    else: # laggard
                        base_maturity = np.random.uniform(1.0, 2.5)
                    
                    # Introduce modest upward trend and randomness
                    month_index = (month.year - START_DATE.year) * 12 + (month.month - START_DATE.month)
                    drift = month_index * np.random.uniform(0.01, 0.05)
                    
                    cmmi_maturity = min(5.0, base_maturity + drift + np.random.uniform(-0.3, 0.3))
                    
                    data_points.append({
                        'Month': month,
                        'Plant': plant,
                        'Domain': domain,
                        'Control ID': control_id,
                        'Control Name': CONTROL_DESCRIPTIONS[control_id],
                        'CMMI Maturity': cmmi_maturity,
                        'Mandatory': control_id in MANDATORY_CONTROLS
                    })
    
    # Generate operational KPI data
    for month in MONTHS:
        for plant in PLANTS:
            profile = next(key for key, value in plant_profiles.items() if plant in value)
            
            # Correlate ops KPIs with maturity profile
            if profile == 'leader':
                mttd = max(2, np.random.normal(5, 2))
                mttr = max(0.5, np.random.normal(1.5, 0.5))
                patch_sla = np.random.normal(95, 2)
                vuln_backlog = np.random.normal(50, 20)
                phishing_fail = np.random.normal(3, 1)
                backup_pass = np.random.normal(98, 1)
                third_party_cov = np.random.normal(90, 3)
                ot_inventory = np.random.normal(95, 2)
                ics_segmentation = np.random.normal(90, 3)
                incident_rate = np.random.normal(0.5, 0.2)
            elif profile == 'mid-pack':
                mttd = np.random.normal(12, 5)
                mttr = np.random.normal(3, 1)
                patch_sla = np.random.normal(85, 5)
                vuln_backlog = np.random.normal(200, 50)
                phishing_fail = np.random.normal(8, 3)
                backup_pass = np.random.normal(90, 5)
                third_party_cov = np.random.normal(75, 5)
                ot_inventory = np.random.normal(85, 5)
                ics_segmentation = np.random.normal(70, 8)
                incident_rate = np.random.normal(2, 0.5)
            else: # laggard
                mttd = np.random.normal(30, 10)
                mttr = np.random.normal(8, 3)
                patch_sla = np.random.normal(70, 10)
                vuln_backlog = np.random.normal(500, 100)
                phishing_fail = np.random.normal(15, 5)
                backup_pass = np.random.normal(80, 8)
                third_party_cov = np.random.normal(50, 10)
                ot_inventory = np.random.normal(70, 10)
                ics_segmentation = np.random.normal(50, 10)
                incident_rate = np.random.normal(5, 2)
            
            # Ensure values are within reasonable bounds
            kpi_data.append({
                'Month': month,
                'Plant': plant,
                'MTTD (hours)': max(2, min(mttd, 48)),
                'MTTR (days)': max(0.5, min(mttr, 14)),
                'Patch SLA Adherence (%)': max(55, min(patch_sla, 98)),
                'Vulnerability Backlog (count >30 days)': max(20, min(vuln_backlog, 800)),
                'Phishing Failure Rate (%)': max(2, min(phishing_fail, 18)),
                'Backup Restore Test Pass Rate (%)': max(70, min(backup_pass, 100)),
                'Third-Party Assessment Coverage (%)': max(40, min(third_party_cov, 95)),
                'OT Asset Inventory Completeness (%)': max(50, min(ot_inventory, 98)),
                'ICS/OT Segmentation Coverage (%)': max(30, min(ics_segmentation, 95)),
                'Incident Rate (per 1,000 endpoints)': max(0.1, min(incident_rate, 10))
            })

    controls_df = pd.DataFrame(data_points)
    kpis_df = pd.DataFrame(kpi_data)
    
    return controls_df, kpis_df

# --- 2. KPI CALCULATORS ---

def calculate_kpis(df):
    """
    Calculates portfolio-level NIS2 KPIs from the controls dataframe.
    """
    if df.empty:
        return {
            'nis2_compliance': 0, 'avg_maturity': 0,
            'critical_coverage': 0, 'maturity_dist': [0,0,0,0,0]
        }
    
    # Calculate NIS2 Compliance Score
    df['Weighted CMMI'] = df['CMMI Maturity']
    df.loc[df['Mandatory'], 'Weighted CMMI'] *= 2
    
    compliant_weighted_count = df.loc[df['Weighted CMMI'] >= 3.0].shape[0]
    total_weighted_count = df.shape[0] * 1.0  # Simple non-mandatory weight
    total_weighted_count += df['Mandatory'].sum() * 1.0 # Add extra weight for mandatory
    
    nis2_compliance = (compliant_weighted_count / total_weighted_count) * 100
    
    # Calculate Average Maturity
    avg_maturity = df['CMMI Maturity'].mean()
    
    # Calculate Critical Control Coverage
    critical_df = df[df['Mandatory']]
    if not critical_df.empty:
        critical_coverage = (critical_df[critical_df['CMMI Maturity'] >= 3.0].shape[0] / critical_df.shape[0]) * 100
    else:
        critical_coverage = 0
    
    # Calculate Maturity Distribution
    bins = [0, 1.99, 2.99, 3.99, 4.99, 5.0]
    labels = ['0-1', '2', '3', '4', '5']
    maturity_dist = pd.cut(df['CMMI Maturity'], bins=bins, labels=labels, right=False).value_counts(normalize=True).sort_index() * 100
    
    return {
        'nis2_compliance': nis2_compliance,
        'avg_maturity': avg_maturity,
        'critical_coverage': critical_coverage,
        'maturity_dist': maturity_dist.to_dict()
    }

# --- 3. REUSABLE PLOTTING FUNCTIONS ---

def plot_kpi_trend(df, kpi_col, title):
    """Plots a trend line for a given KPI."""
    fig = px.line(
        df.groupby('Month')[kpi_col].mean().reset_index(),
        x='Month', y=kpi_col,
        title=title, markers=True
    )
    fig.update_layout(xaxis_title='Month', yaxis_title=kpi_col, template='plotly_white')
    return fig

def create_heatmap(df):
    """Creates a heatmap of CMMI maturity by Plant and Domain."""
    heatmap_data = df.groupby(['Plant', 'Domain'])['CMMI Maturity'].mean().reset_index()
    fig = px.density_heatmap(
        heatmap_data, x='Domain', y='Plant', z='CMMI Maturity',
        title='NIS2 Maturity Heatmap (Plants vs. Domains)',
        color_continuous_scale=px.colors.sequential.YlGnBu,
        text_auto=".2f",
    )
    fig.update_layout(
        xaxis={'tickangle': -45, 'title_text': ''},
        yaxis_title='Plant',
        coloraxis_colorbar_title_text='CMMI'
    )
    return fig

def create_radar_chart(df, plant_name):
    """Creates a radar chart for a single plant's domain-level maturity."""
    domain_maturity = df.groupby('Domain')['CMMI Maturity'].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=domain_maturity['CMMI Maturity'],
        theta=domain_maturity['Domain'],
        fill='toself',
        name=plant_name,
        hovertemplate='<b>Domain:</b> %{theta}<br><b>Maturity:</b> %{r:.2f}<extra></extra>'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5])
        ),
        title=f'Maturity Profile for {plant_name}',
        showlegend=False
    )
    return fig

def create_top_bottom_bar_chart(df, title, top_n=5, ascending=True):
    """Creates a bar chart for top or bottom N controls."""
    grouped = df.groupby(['Control ID', 'Control Name'])['CMMI Maturity'].mean().reset_index()
    grouped = grouped.sort_values(by='CMMI Maturity', ascending=ascending).head(top_n)
    
    fig = px.bar(
        grouped,
        x='CMMI Maturity', y='Control Name', orientation='h',
        title=title,
        color='CMMI Maturity',
        color_continuous_scale=px.colors.sequential.YlGnBu
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending' if ascending else 'total descending'})
    return fig

# --- 4. STREAMLIT APP LAYOUT & PAGES ---

st.set_page_config(layout="wide", page_title="NIS2 Dashboard")

# Load data with caching
controls_df, kpis_df = generate_data()

st.title("NIS2 Implementation Dashboard")
st.markdown("### Portfolio & Plant-Level Visibility for Cybersecurity Maturity & Operations")
st.write("This dashboard provides a simulated view of cybersecurity performance aligned with NIS2 controls, offering insights for CISOs, Plant Managers, and GRC stakeholders.")

# --- Sidebar Filters ---
st.sidebar.header("Filter & Navigation")

page = st.sidebar.radio("Go to", ["Overview", "Plant Drill-Down", "Control Drill-Down"])

if page in ["Overview", "Plant Drill-Down"]:
    selected_plant = st.sidebar.selectbox(
        "Select a Plant (for Plant Drill-Down)",
        ['All Plants'] + sorted(PLANTS)
    )
else:
    selected_plant = 'All Plants'

date_range = st.sidebar.slider(
    "Select Date Range (Months)",
    min_value=0, max_value=11, value=(0, 11)
)
start_month, end_month = MONTHS[date_range[0]], MONTHS[date_range[1]]

# Filter dataframes based on user selections
filtered_controls_df = controls_df[
    (controls_df['Month'] >= start_month) &
    (controls_df['Month'] <= end_month)
]
filtered_kpis_df = kpis_df[
    (kpis_df['Month'] >= start_month) &
    (kpis_df['Month'] <= end_month)
]

if selected_plant != 'All Plants':
    filtered_controls_df = filtered_controls_df[filtered_controls_df['Plant'] == selected_plant]
    filtered_kpis_df = filtered_kpis_df[filtered_kpis_df['Plant'] == selected_plant]

# --- Page Logic ---

if page == "Overview":
    st.subheader("Portfolio Overview")
    
    # Calculate portfolio-level KPIs
    portfolio_kpis = calculate_kpis(filtered_controls_df)
    
    # KPI Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            label="NIS2 Compliance Score",
            value=f"{portfolio_kpis['nis2_compliance']:.1f}%",
            help="Percentage of controls with CMMI ≥ 3.0, with mandatory controls weighted 2x."
        )
    with col2:
        st.metric(
            label="Average Control Maturity",
            value=f"{portfolio_kpis['avg_maturity']:.2f}",
            help="Average CMMI maturity score across all controls and plants."
        )
    with col3:
        st.metric(
            label="Critical Control Coverage",
            value=f"{portfolio_kpis['critical_coverage']:.1f}%",
            help="Percentage of mandatory controls with CMMI ≥ 3.0."
        )
    with col4:
        st.metric(
            label="Avg. Patch SLA Adherence",
            value=f"{filtered_kpis_df['Patch SLA Adherence (%)'].mean():.1f}%",
            help="Average percentage of critical vulnerabilities remediated within policy window."
        )
    with col5:
        st.metric(
            label="Avg. MTTD (hours)",
            value=f"{filtered_kpis_df['MTTD (hours)'].mean():.1f}",
            help="Mean Time to Detect an incident."
        )
    with col6:
        st.metric(
            label="Avg. MTTR (days)",
            value=f"{filtered_kpis_df['MTTR (days)'].mean():.1f}",
            help="Mean Time to Respond and Recover from an incident."
        )

    st.markdown("---")
    
    # Visuals
    col_vis1, col_vis2 = st.columns([2,1])
    with col_vis1:
        st.plotly_chart(create_heatmap(filtered_controls_df), use_container_width=True)
    
    with col_vis2:
        maturity_dist_df = pd.DataFrame(
            portfolio_kpis['maturity_dist'].items(),
            columns=['Maturity Level', 'Percentage']
        )
        fig_dist = px.bar(
            maturity_dist_df, x='Maturity Level', y='Percentage',
            title='Portfolio-wide Maturity Distribution',
            color='Maturity Level',
            color_discrete_sequence=px.colors.sequential.YlGnBu
        )
        fig_dist.update_layout(yaxis_title="Percentage (%)")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.plotly_chart(plot_kpi_trend(filtered_controls_df, 'CMMI Maturity', 'Avg. Maturity Trend (Last 12 Months)'), use_container_width=True)

    col_charts = st.columns(2)
    with col_charts[0]:
        st.plotly_chart(create_top_bottom_bar_chart(filtered_controls_df, 'Top 5 Controls by Maturity', ascending=False), use_container_width=True)
    with col_charts[1]:
        st.plotly_chart(create_top_bottom_bar_chart(filtered_controls_df, 'Bottom 5 Controls by Maturity', ascending=True), use_container_width=True)

    # Download button for the overview data
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full Data as CSV",
        data=convert_df_to_csv(controls_df),
        file_name="nis2_dashboard_data.csv",
        mime="text/csv"
    )

elif page == "Plant Drill-Down":
    st.subheader("Plant-Specific KPIs & Insights")
    if selected_plant == 'All Plants':
        st.warning("Please select a specific plant from the sidebar to view this page.")
    else:
        plant_kpis = calculate_kpis(filtered_controls_df)
        
        # KPI Cards for selected plant
        st.write(f"### Performance for {selected_plant}")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("NIS2 Compliance", f"{plant_kpis['nis2_compliance']:.1f}%")
        with col2:
            st.metric("Avg. Maturity", f"{plant_kpis['avg_maturity']:.2f}")
        with col3:
            st.metric("Avg. Patch SLA", f"{filtered_kpis_df['Patch SLA Adherence (%)'].mean():.1f}%")
        with col4:
            st.metric("MTTD (hours)", f"{filtered_kpis_df['MTTD (hours)'].mean():.1f}")
        with col5:
            st.metric("MTTR (days)", f"{filtered_kpis_df['MTTR (days)'].mean():.1f}")

        st.markdown("---")

        # Visuals for a single plant
        col_plant1, col_plant2 = st.columns(2)
        with col_plant1:
            st.plotly_chart(create_radar_chart(filtered_controls_df, selected_plant), use_container_width=True)
        
        with col_plant2:
            st.plotly_chart(create_top_bottom_bar_chart(filtered_controls_df, f"Weakest Controls for {selected_plant}", ascending=True), use_container_width=True)
            st.plotly_chart(create_top_bottom_bar_chart(filtered_controls_df, f"Strongest Controls for {selected_plant}", ascending=False), use_container_width=True)
            
        # Scatter plot for incident metrics
        st.subheader("Operational Metrics Deep-Dive")
        
        fig_scatter = px.scatter(
            filtered_kpis_df,
            x='MTTD (hours)', y='MTTR (days)',
            color='Month',
            size='Incident Rate (per 1,000 endpoints)',
            title=f"MTTD vs. MTTR for {selected_plant}",
            hover_name='Month',
            labels={'MTTD (hours)': 'Mean Time to Detect (hours)', 'MTTR (days)': 'Mean Time to Recover (days)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
elif page == "Control Drill-Down":
    st.subheader("Control-Specific Maturity & Performance")
    
    # Control-specific filters
    domain_list = sorted(list(CONTROL_DOMAINS.keys()))
    selected_domain = st.sidebar.selectbox("Select a Domain", domain_list)
    
    controls_in_domain = CONTROL_DOMAINS[selected_domain]
    control_names_map = {CONTROL_DESCRIPTIONS[c]: c for c in controls_in_domain}
    selected_control_name = st.sidebar.selectbox("Select a Control", sorted(list(control_names_map.keys())))
    selected_control_id = control_names_map[selected_control_name]
    
    filtered_by_control = filtered_controls_df[filtered_controls_df['Control ID'] == selected_control_id]
    
    st.write(f"### Analysis for Control: {selected_control_name} ({selected_control_id})")
    
    # Column chart: Maturity by plant for the chosen control
    fig_bar_plant = px.bar(
        filtered_by_control.groupby('Plant')['CMMI Maturity'].mean().reset_index(),
        x='Plant', y='CMMI Maturity',
        title=f"CMMI Maturity by Plant for '{selected_control_name}'",
        color='CMMI Maturity',
        color_continuous_scale=px.colors.sequential.YlGnBu
    )
    fig_bar_plant.update_layout(xaxis_title="", yaxis_title="Average CMMI Maturity")
    st.plotly_chart(fig_bar_plant, use_container_width=True)
    
    # Line chart: Time series for top/bottom plants
    st.subheader("Maturity Trend over Time")
    
    # Find top 3 and bottom 3 plants for this control
    avg_maturity_by_plant = filtered_by_control.groupby('Plant')['CMMI Maturity'].mean()
    top_3_plants = avg_maturity_by_plant.nlargest(3).index.tolist()
    bottom_3_plants = avg_maturity_by_plant.nsmallest(3).index.tolist()
    
    plants_to_show = top_3_plants + bottom_3_plants
    
    trend_df = filtered_by_control[filtered_by_control['Plant'].isin(plants_to_show)]
    
    fig_trend = px.line(
        trend_df,
        x='Month', y='CMMI Maturity', color='Plant',
        title=f"CMMI Maturity Trend for '{selected_control_name}'",
        markers=True
    )
    fig_trend.update_layout(xaxis_title='Month', yaxis_title='CMMI Maturity')
    st.plotly_chart(fig_trend, use_container_width=True)

# --- KPI Glossary ---
with st.sidebar.expander("KPI Glossary"):
    st.write("""
    - **NIS2 Compliance Score (%):** The percentage of controls with a CMMI maturity score of 3.0 or higher. Mandatory controls are weighted twice.
    - **Average Control Maturity (0-5):** The mean CMMI maturity score across all controls and plants.
    - **Critical Control Coverage (%):** The percentage of mandatory NIS2 controls that have reached a CMMI maturity of 3.0 or higher.
    - **MTTD (hours):** Mean Time to Detect a security incident. Lower is better.
    - **MTTR (days):** Mean Time to Respond and Recover from an incident. Lower is better.
    - **Patch SLA Adherence (%):** Percentage of critical vulnerabilities remediated within the defined policy window. Higher is better.
    - **Vulnerability Backlog (count):** The number of open vulnerabilities that have exceeded their remediation policy window. Lower is better.
    - **Backup Restore Test Pass Rate (%):** The success rate of monthly tests to restore data from backups.
    - **Phishing Failure Rate (%):** Percentage of users who fall for simulated phishing campaigns. Lower is better.
    - **Third-Party Assessment Coverage (%):** Percentage of critical suppliers who have undergone a security risk assessment.
    - **OT Asset Inventory Completeness (%):** The percentage of operational technology (OT) assets that are accurately tracked in the asset inventory.
    - **ICS/OT Segmentation Coverage (%):** The extent to which critical OT networks are isolated from IT and other networks.
    - **Incident Rate (per 1,000 endpoints):** The number of security incidents reported per 1,000 managed devices.
    """)
