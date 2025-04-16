import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Cost of Labor Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    /* Add Font Awesome CDN */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css');

    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A; /* Dark Blue */
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB; /* Medium Blue */
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        border-bottom: 2px solid #DBEAFE; /* Light blue underline */
        padding-bottom: 0.25rem;
    }
    .metric-card {
        background-color: #FFFFFF; /* White background */
        border-radius: 8px; /* Slightly more rounded corners */
        padding: 1.25rem; /* Increased padding */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); /* Lighter initial shadow */
        border: 1px solid #E5E7EB; /* Light grey border */
        border-left: 6px solid #2563EB; /* ACCENT BORDER - Medium Blue */
        margin-bottom: 1rem; /* Space below each card */
        height: 100%; /* Make cards in a row the same height */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.2s ease-in-out; /* Smooth transition for hover */
    }

    /* HOVER EFFECT */
    .metric-card:hover {
        box-shadow: 0 7px 14px rgba(0, 0, 0, 0.1); /* Deeper shadow on hover */
        transform: translateY(-4px); /* Slight lift effect */
        border-left-color: #1E3A8A; /* Darken accent border on hover */
    }

    .metric-value {
        font-size: 2.1rem; /* Slightly larger value */
        font-weight: 700;
        color: #1E3A8A; /* Dark Blue */
        line-height: 1.2;
        margin-bottom: 0.5rem; /* More space before label */
    }

    /* Container for label and icon */
    .metric-label-container {
        display: flex;
        align-items: center; /* Vertically align icon and text */
        gap: 0.6rem;       /* Space between icon and text */
    }

    .metric-label {
        font-size: 0.95rem;
        color: #4B5563; /* Grey text */
        font-weight: 500;
        margin: 0; /* Remove default margins */
    }

    /* Icon Style */
    .metric-icon {
        color: #6B7280; /* Icon color - Medium Grey */
        font-size: 1.2rem; /* Adjust icon size */
        width: 20px; /* Give icon a fixed width for alignment */
        text-align: center;
    }

    .highlight {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2563EB;
        margin-top: 1rem;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    /* Optional: Style default st.metric if used elsewhere */
    div[data-testid="stMetric"] {
       background-color: #FFFFFF; border: 1px solid #E5E7EB;
       box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); padding: 1rem; border-radius: 8px;
    }
    div[data-testid="stMetric"] > label { font-weight: 500; color: #4B5563 !important; }
    div[data-testid="stMetric"] > div { color: #1E3A8A !important; }
</style>
""", unsafe_allow_html=True)

# Helper functions for calculations
def calculate_totals(df):
    """Calculate total cost metrics from employee data"""
    totals = {
        'total_base_salary': df['base_salary'].sum(),
        'total_premiums': df['work_conditions_premium'].sum() + df['overtime_premium'].sum() + df[
            'other_premiums'].sum(),
        'total_bonuses': df['annual_bonus'].sum() + df['profit_sharing'].sum(),
        'total_social_contributions': df['social_security_tax'].sum() + df['medicare'].sum() + df['er_401k'].sum() + df[
            'er_pension'].sum(),
        'total_ltips': df['ltips'].sum(),
        'total_fte': len(df),
        'total_planned_hours': df['planned_hours'].sum(),
        'total_actual_hours': df['actual_hours'].sum(),
        'total_absence_costs': calculate_absence_costs(df),
        'total_overtime_hours': calculate_overtime(df),
        'total_cost': 0  # Will be calculated below
    }

    # Calculate total cost of labor
    totals['total_cost'] = (
            totals['total_base_salary'] +
            totals['total_premiums'] +
            totals['total_bonuses'] +
            totals['total_social_contributions'] +
            totals['total_ltips']
    )

    # Calculate FTE costs
    totals['fte_costs'] = totals['total_cost'] / totals['total_fte'] if totals['total_fte'] > 0 else 0

    return totals


def calculate_absence_costs(df):
    """Calculate absence costs based on base salary and days off"""
    # Assuming 260 working days per year (52 weeks * 5 days)
    daily_rate = df['base_salary'] / 260
    return (daily_rate * (df['sick_days'] + df['holiday_days'] + df['other_absences'])).sum()


def calculate_overtime(df):
    """Calculate overtime hours as actual_hours - planned_hours where positive"""
    return (df['actual_hours'] - df['planned_hours']).apply(lambda x: max(0, x)).sum()


def project_costs_for_new_hires(current_totals, band_avg_df, dept, band, num_new_hires):
    """Project costs for 2025 based on current totals and new hires"""
    # Filter band averages for the selected department and band
    avg_data = band_avg_df[(band_avg_df['department'] == dept) & (band_avg_df['band'] == band)]

    if avg_data.empty:
        st.error(f"No average data available for {dept}, {band}")
        return current_totals

    # Get the first (and should be only) row
    avg = avg_data.iloc[0]

    # Calculate projections
    projections = current_totals.copy()

    # Add costs for new hires
    projections['total_base_salary'] += avg['avg_base_salary'] * num_new_hires
    projections['total_premiums'] += (avg['avg_work_conditions'] + avg['avg_overtime'] + avg[
        'avg_other_premiums']) * num_new_hires
    projections['total_bonuses'] += (avg['avg_annual_bonus'] + avg['avg_profit_sharing']) * num_new_hires
    projections['total_social_contributions'] += (avg['avg_social_security'] + avg['avg_medicare'] + avg['avg_401k'] +
                                                  avg['avg_pension']) * num_new_hires
    projections['total_ltips'] += avg['avg_ltips'] * num_new_hires
    projections['total_fte'] += num_new_hires
    projections['total_planned_hours'] += avg['avg_planned_hours'] * num_new_hires

    # Recalculate total cost
    projections['total_cost'] = (
            projections['total_base_salary'] +
            projections['total_premiums'] +
            projections['total_bonuses'] +
            projections['total_social_contributions'] +
            projections['total_ltips']
    )

    # Calculate new hiring costs
    projections['hiring_costs'] = avg['avg_hiring_cost'] * num_new_hires

    # Update FTE costs
    projections['fte_costs'] = projections['total_cost'] / projections['total_fte'] if projections[
                                                                                           'total_fte'] > 0 else 0

    return projections


def format_currency(value):
    """Format value as currency"""
    return f"${value:,.2f}"


def load_data():
    """Load the CSV data files"""
    try:
        # For a real application, these would be actual file paths
        # For now, let's create sample data

        # Create sample data for 2023 and 2024
        df_2023 = pd.read_csv('employee_data_2023.csv')
        df_2024 = pd.read_csv('employee_data_2024.csv')

        # Load band averages
        band_avg_df = pd.read_csv('band_averages.csv')

        return df_2023, df_2024, band_avg_df
    except Exception as e:
        # If we can't load the files, create dummy data for demonstration
        st.warning(f"Could not load actual data files. Using sample data for demonstration.")

        # Create sample employee data
        sample_data = [
            [1001, "John Doe", "ERP", "BV", 85000, 2500, 3500, 1000, 8500, 2000, 5270, 1232.5, 2550, 1700, 4250, 2080,
             2095, 5, 12, 2, "USD", 1.0],
            [1002, "Jane Smith", "ERA", "BIV", 72000, 1800, 2200, 800, 7200, 1440, 4464, 1044, 2160, 1440, 3600, 2080,
             2065, 8, 12, 3, "USD", 1.0],
            [1003, "Michael Johnson", "ERZ", "BIII", 65000, 1500, 1800, 700, 6500, 1300, 4030, 942.5, 1950, 1300, 3250,
             2080, 2090, 3, 12, 1, "USD", 1.0],
            [1004, "Lisa Brown", "8VC", "BVI", 92000, 3000, 4100, 1200, 9200, 2300, 5704, 1334, 2760, 1840, 4600, 2080,
             2110, 4, 12, 0, "USD", 1.0],
            [1005, "Robert Chen", "MMS", "BV", 84000, 2400, 3200, 900, 8400, 1680, 5208, 1218, 2520, 1680, 4200, 2080,
             2075, 6, 12, 4, "USD", 1.0],
            [1006, "Sarah Wilson", "WC", "BIV", 71000, 1700, 2100, 750, 7100, 1420, 4402, 1029.5, 2130, 1420, 3550,
             2080, 2050, 10, 12, 2, "USD", 1.0],
        ]

        columns = ["employee_id", "name", "department", "band", "base_salary", "work_conditions_premium",
                   "overtime_premium", "other_premiums", "annual_bonus", "profit_sharing",
                   "social_security_tax", "medicare", "er_401k", "er_pension", "ltips",
                   "planned_hours", "actual_hours", "sick_days", "holiday_days", "other_absences",
                   "currency", "fx_rate"]

        df_2023 = pd.DataFrame(sample_data, columns=columns)

        # Create slightly modified data for 2024 (5% increase in most values)
        df_2024 = df_2023.copy()
        for col in ["base_salary", "work_conditions_premium", "overtime_premium", "other_premiums",
                    "annual_bonus", "profit_sharing", "social_security_tax", "medicare",
                    "er_401k", "er_pension", "ltips"]:
            df_2024[col] = df_2024[col] * 1.05

        # Create band averages
        band_data = [
            ["ERP", "BV", 85000, 2500, 3500, 1000, 8500, 2000, 5270, 1232.5, 2550, 1700, 4250, 2080, 5, 12000],
            ["ERA", "BIV", 72000, 1800, 2200, 800, 7200, 1440, 4464, 1044, 2160, 1440, 3600, 2080, 7, 9000],
            ["ERZ", "BIII", 65000, 1500, 1800, 700, 6500, 1300, 4030, 942.5, 1950, 1300, 3250, 2080, 4, 7500],
            ["8VC", "BVI", 92000, 3000, 4100, 1200, 9200, 2300, 5704, 1334, 2760, 1840, 4600, 2080, 4, 15000],
            ["MMS", "BV", 84000, 2400, 3200, 900, 8400, 1680, 5208, 1218, 2520, 1680, 4200, 2080, 6, 12000],
            ["WC", "BIV", 71000, 1700, 2100, 750, 7100, 1420, 4402, 1029.5, 2130, 1420, 3550, 2080, 8, 9000],
            ["BC", "BIII", 66000, 1600, 1900, 700, 6600, 1320, 4092, 957, 1980, 1320, 3300, 2080, 7, 7500],
        ]

        band_columns = ["department", "band", "avg_base_salary", "avg_work_conditions", "avg_overtime",
                        "avg_other_premiums", "avg_annual_bonus", "avg_profit_sharing",
                        "avg_social_security", "avg_medicare", "avg_401k", "avg_pension",
                        "avg_ltips", "avg_planned_hours", "avg_sick_days", "avg_hiring_cost"]

        band_avg_df = pd.DataFrame(band_data, columns=band_columns)

        return df_2023, df_2024, band_avg_df


# Load data
df_2023, df_2024, band_avg_df = load_data()

# Calculate totals for 2023 and 2024
totals_2023 = calculate_totals(df_2023)
totals_2024 = calculate_totals(df_2024)

# Title and description
st.markdown('<div class="main-header">Cost of Labor Dashboard</div>', unsafe_allow_html=True)
st.markdown("""
This dashboard provides a comprehensive view of labor costs for 2023, 2024, and projected costs for 2025.
It allows for simulation of new hires and their impact on the overall cost structure.
""")

# Create sidebar for controls
st.sidebar.markdown("## Dashboard Controls")

# Year selection
selected_year = st.sidebar.selectbox("Select Year", ["2023", "2024", "2025 (Projected)"])

# Department filter for displayed data
selected_dept = st.sidebar.selectbox(
    "Filter by Department",
    ["All Departments"] + sorted(df_2023["department"].unique().tolist())
)

# Initialize projection parameters
num_new_hires = 0
selected_band = "BV"
selected_hire_dept = "ERP"
projected_totals = totals_2024.copy()

# Only show projection inputs when 2025 is selected
if selected_year == "2025 (Projected)":
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 2025 Projection Parameters")

    num_new_hires = st.sidebar.number_input("Number of New Hires", min_value=0, max_value=100, value=0)

    selected_hire_dept = st.sidebar.selectbox(
        "Department for New Hires",
        sorted(df_2023["department"].unique().tolist())
    )

    selected_band = st.sidebar.selectbox(
        "Band for New Hires",
        sorted(df_2023["band"].unique().tolist())
    )

    # Calculate projections based on inputs
    projected_totals = project_costs_for_new_hires(
        totals_2024,
        band_avg_df,
        selected_hire_dept,
        selected_band,
        num_new_hires
    )

# Filter data based on selected department
if selected_dept != "All Departments":
    df_2023_filtered = df_2023[df_2023["department"] == selected_dept]
    df_2024_filtered = df_2024[df_2024["department"] == selected_dept]
    dept_totals_2023 = calculate_totals(df_2023_filtered)
    dept_totals_2024 = calculate_totals(df_2024_filtered)
else:
    df_2023_filtered = df_2023
    df_2024_filtered = df_2024
    dept_totals_2023 = totals_2023
    dept_totals_2024 = totals_2024

# Main dashboard area
col1, col2, col3 = st.columns(3)

# Determine which totals to use based on selected year
if selected_year == "2023":
    current_totals = dept_totals_2023
elif selected_year == "2024":
    current_totals = dept_totals_2024
else:  # 2025 Projected
    # If a department filter is applied, we need to recalculate the projections for just that department
    if selected_dept != "All Departments":
        # Start with the 2024 department totals
        current_totals = dept_totals_2024.copy()

        # Only add new hire costs if they're in the selected department
        if selected_hire_dept == selected_dept:
            current_totals = project_costs_for_new_hires(
                dept_totals_2024,
                band_avg_df,
                selected_hire_dept,
                selected_band,
                num_new_hires
            )
    else:
        current_totals = projected_totals

# Main dashboard area - Use the custom markdown approach with icons
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{format_currency(current_totals["total_cost"])}</div>
            <div class="metric-label-container">
                 <i class="fas fa-dollar-sign metric-icon"></i> {/* Icon */}
                 <span class="metric-label">Total Cost of Labor ({selected_year})</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{current_totals["total_fte"]}</div>
             <div class="metric-label-container">
                 <i class="fas fa-users metric-icon"></i> {/* Icon */}
                 <span class="metric-label">Total FTE ({selected_year})</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{format_currency(current_totals["fte_costs"])}</div>
             <div class="metric-label-container">
                 {/* Using a different icon for variety */}
                 <i class="fas fa-money-check-alt metric-icon"></i> {/* Icon */}
                 <span class="metric-label">Cost per FTE ({selected_year})</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)


# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Cost Breakdown", "📈 Year-over-Year", "🔍 Detailed Analysis"])

# Tab 1: Cost Breakdown
with tab1:
    st.markdown('<div class="sub-header">Cost Breakdown</div>', unsafe_allow_html=True)

    # Create cost breakdown chart
    cost_data = {
        'Category': ['Base Salary', 'Premiums', 'Bonuses', 'Social Contributions', 'LTIPs'],
        'Amount': [
            current_totals['total_base_salary'],
            current_totals['total_premiums'],
            current_totals['total_bonuses'],
            current_totals['total_social_contributions'],
            current_totals['total_ltips']
        ]
    }
    cost_df = pd.DataFrame(cost_data)

    # Calculate percentages
    total = cost_df['Amount'].sum()
    cost_df['Percentage'] = (cost_df['Amount'] / total * 100).round(1)

    # Create the chart
    fig = px.pie(
        cost_df,
        values='Amount',
        names='Category',
        title=f'Cost Distribution for {selected_year}',
        color_discrete_sequence=px.colors.sequential.Blues_r,
        hover_data=['Percentage']
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        height=500,
        margin=dict(t=50, b=0, l=0, r=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display cost breakdown in a table with $ and % columns
    cost_df['Amount'] = cost_df['Amount'].apply(format_currency)
    cost_df['Percentage'] = cost_df['Percentage'].apply(lambda x: f"{x}%")

    st.markdown('<div class="sub-header">Cost Breakdown Details</div>', unsafe_allow_html=True)
    st.table(cost_df)

    # Add special section for 2025 with new hire cost
    if selected_year == "2025 (Projected)" and num_new_hires > 0:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown(f"### New Hire Impact")
        st.markdown(f"**Number of New Hires:** {num_new_hires} ({selected_hire_dept}, {selected_band})")

        if 'hiring_costs' in projected_totals:
            st.markdown(f"**Total Hiring Costs:** {format_currency(projected_totals['hiring_costs'])}")
            st.markdown(f"**Cost per New Hire:** {format_currency(projected_totals['hiring_costs'] / num_new_hires)}")

        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Year-over-Year Comparison
with tab2:
    st.markdown('<div class="sub-header">Year-over-Year Comparison</div>', unsafe_allow_html=True)

    # Create comparison data
    if selected_dept != "All Departments":
        compare_2023 = dept_totals_2023
        compare_2024 = dept_totals_2024
    else:
        compare_2023 = totals_2023
        compare_2024 = totals_2024

    # Calculate growth rates
    growth_base = ((compare_2024['total_base_salary'] - compare_2023['total_base_salary']) / compare_2023[
        'total_base_salary'] * 100).round(1)
    growth_premiums = ((compare_2024['total_premiums'] - compare_2023['total_premiums']) / compare_2023[
        'total_premiums'] * 100).round(1)
    growth_bonuses = ((compare_2024['total_bonuses'] - compare_2023['total_bonuses']) / compare_2023[
        'total_bonuses'] * 100).round(1)
    growth_social = ((compare_2024['total_social_contributions'] - compare_2023['total_social_contributions']) /
                     compare_2023['total_social_contributions'] * 100).round(1)
    growth_ltips = (
                (compare_2024['total_ltips'] - compare_2023['total_ltips']) / compare_2023['total_ltips'] * 100).round(
        1)
    growth_total = ((compare_2024['total_cost'] - compare_2023['total_cost']))
    growth_total = ((compare_2024['total_cost'] - compare_2023['total_cost']) / compare_2023['total_cost'] * 100).round(1)

# Create comparison chart
categories = ['Base Salary', 'Premiums', 'Bonuses', 'Social Contributions', 'LTIPs', 'Total Cost']

year_2023_values = [
    compare_2023['total_base_salary'],
    compare_2023['total_premiums'],
    compare_2023['total_bonuses'],
    compare_2023['total_social_contributions'],
    compare_2023['total_ltips'],
    compare_2023['total_cost']
]

year_2024_values = [
    compare_2024['total_base_salary'],
    compare_2024['total_premiums'],
    compare_2024['total_bonuses'],
    compare_2024['total_social_contributions'],
    compare_2024['total_ltips'],
    compare_2024['total_cost']
]

growth_rates = [
    growth_base,
    growth_premiums,
    growth_bonuses,
    growth_social,
    growth_ltips,
    growth_total
]

# Create grouped bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=categories,
    y=year_2023_values,
    name='2023',
    marker_color='#93C5FD'
))

fig.add_trace(go.Bar(
    x=categories,
    y=year_2024_values,
    name='2024',
    marker_color='#2563EB'
))

# Add 2025 projections if selected
if selected_year == "2025 (Projected)":
    year_2025_values = [
        projected_totals['total_base_salary'],
        projected_totals['total_premiums'],
        projected_totals['total_bonuses'],
        projected_totals['total_social_contributions'],
        projected_totals['total_ltips'],
        projected_totals['total_cost']
    ]

    fig.add_trace(go.Bar(
        x=categories,
        y=year_2025_values,
        name='2025 (Projected)',
        marker_color='#1E3A8A'
    ))

fig.update_layout(
    title="Cost Comparison Between Years",
    xaxis_title="Cost Category",
    yaxis_title="Amount ($)",
    legend_title="Year",
    barmode='group',
    height=500
)

# Add dollar sign formatting to y-axis
fig.update_yaxes(tickprefix="$", tickformat=",")

st.plotly_chart(fig, use_container_width=True)

# Display growth rates in a table
growth_data = {
    'Category': categories,
    '2023 Amount': [format_currency(val) for val in year_2023_values],
    '2024 Amount': [format_currency(val) for val in year_2024_values],
    '2023-2024 Growth': [f"{rate}%" for rate in growth_rates]
}

if selected_year == "2025 (Projected)":
    # Calculate 2024-2025 growth rates
    growth_2025 = [(projected_totals[key] - compare_2024[key]) / compare_2024[key] * 100
                   for key in ['total_base_salary', 'total_premiums', 'total_bonuses',
                               'total_social_contributions', 'total_ltips', 'total_cost']]

    growth_data['2025 Amount (Projected)'] = [format_currency(val) for val in year_2025_values]
    growth_data['2024-2025 Growth (Projected)'] = [f"{rate:.1f}%" for rate in growth_2025]

growth_df = pd.DataFrame(growth_data)
st.table(growth_df)

# Tab 3: Detailed Analysis
with tab3:
    st.markdown('<div class="sub-header">Detailed Analysis</div>', unsafe_allow_html=True)

    # Choose which data to display based on selected year
    if selected_year == "2023":
        analysis_df = df_2023_filtered
        analysis_totals = dept_totals_2023
    elif selected_year == "2024":
        analysis_df = df_2024_filtered
        analysis_totals = dept_totals_2024
    else:  # 2025 projection - base on 2024 data
        analysis_df = df_2024_filtered
        analysis_totals = current_totals

    # Create two sub-tabs for different analysis views
    subtab1, subtab2, subtab3 = st.tabs(["Department Analysis", "Band Distribution", "Hours Analysis"])

    # Department Analysis
    with subtab1:
        if selected_dept == "All Departments":
            # Show department breakdown
            dept_analysis = analysis_df.groupby('department').agg({
                'base_salary': 'sum',
                'employee_id': 'count'
            }).reset_index()

            dept_analysis.columns = ['Department', 'Total Base Salary', 'Employee Count']

            # Calculate average salary by department
            dept_analysis['Average Salary'] = dept_analysis['Total Base Salary'] / dept_analysis['Employee Count']

            # Format as currency
            dept_analysis['Total Base Salary'] = dept_analysis['Total Base Salary'].apply(format_currency)
            dept_analysis['Average Salary'] = dept_analysis['Average Salary'].apply(format_currency)

            # Department distribution chart
            fig = px.bar(
                dept_analysis,
                x='Department',
                y='Employee Count',
                title='Employee Distribution by Department',
                color='Department',
                color_discrete_sequence=px.colors.qualitative.Set3
            )

            st.plotly_chart(fig, use_container_width=True)

            # Department salary breakdown
            st.markdown("### Department Salary Breakdown")
            st.table(dept_analysis)
        else:
            st.markdown(f"### {selected_dept} Department Analysis")

            # Band distribution within department
            band_count = analysis_df['band'].value_counts().reset_index()
            band_count.columns = ['Band', 'Count']

            fig = px.pie(
                band_count,
                values='Count',
                names='Band',
                title=f'Band Distribution in {selected_dept}',
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')

            st.plotly_chart(fig, use_container_width=True)

    # Band Distribution
    with subtab2:
        # Analysis by band
        band_analysis = analysis_df.groupby('band').agg({
            'base_salary': ['sum', 'mean'],
            'employee_id': 'count'
        }).reset_index()

        band_analysis.columns = ['Band', 'Total Base Salary', 'Average Salary', 'Employee Count']

        # Format as currency
        band_analysis['Total Base Salary'] = band_analysis['Total Base Salary'].apply(format_currency)
        band_analysis['Average Salary'] = band_analysis['Average Salary'].apply(format_currency)

        # Create visual band distribution
        fig = px.bar(
            band_analysis,
            x='Band',
            y='Employee Count',
            title='Employee Distribution by Band',
            color='Band',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )

        st.plotly_chart(fig, use_container_width=True)

        # Band salary breakdown
        st.markdown("### Band Salary Breakdown")
        st.table(band_analysis)

    # Hours Analysis
    with subtab3:
        # Planned vs Actual Hours
        if selected_year != "2025 (Projected)":
            st.markdown("### Planned vs. Actual Hours")

            hours_data = {
                'Category': ['Planned Hours', 'Actual Hours'],
                'Hours': [
                    analysis_totals['total_planned_hours'],
                    analysis_totals['total_actual_hours']
                ]
            }
            hours_df = pd.DataFrame(hours_data)

            fig = px.bar(
                hours_df,
                x='Category',
                y='Hours',
                title='Planned vs. Actual Hours',
                color='Category',
                color_discrete_sequence=['#93C5FD', '#2563EB']
            )

            st.plotly_chart(fig, use_container_width=True)

            # Hours difference analysis
            hours_diff = analysis_totals['total_actual_hours'] - analysis_totals['total_planned_hours']
            hours_diff_percent = (hours_diff / analysis_totals['total_planned_hours'] * 100).round(2)

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Total Hours Difference",
                    f"{hours_diff:.0f} hours",
                    f"{hours_diff_percent}%"
                )

            with col2:
                st.metric(
                    "Overtime Hours",
                    f"{analysis_totals['total_overtime_hours']:.0f} hours",
                    help="Total overtime hours calculated as sum of (actual - planned) where positive"
                )

            # Display absence analysis
            st.markdown("### Absence Analysis")

            absence_data = {
                'Category': ['Absence Cost', 'As % of Base Salary'],
                'Value': [
                    format_currency(analysis_totals['total_absence_costs']),
                    f"{(analysis_totals['total_absence_costs'] / analysis_totals['total_base_salary'] * 100).round(2)}%"
                ]
            }
            absence_df = pd.DataFrame(absence_data)

            st.table(absence_df)

# Advanced 2025 Projection Section
if selected_year == "2025 (Projected)":
    st.markdown("---")
    st.markdown('<div class="sub-header">2025 Projection Details</div>', unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### New Hire Cost Breakdown")

        if num_new_hires > 0:
            # Get averages for the selected department and band
            hire_avg = band_avg_df[(band_avg_df['department'] == selected_hire_dept) &
                                   (band_avg_df['band'] == selected_band)].iloc[0]

            # Create breakdown of costs per new hire
            new_hire_data = {
                'Category': [
                    'Base Salary',
                    'Premiums',
                    'Bonuses',
                    'Social Contributions',
                    'LTIPs',
                    'Hiring Cost',
                    'Total Cost per Hire'
                ],
                'Amount': [
                    hire_avg['avg_base_salary'],
                    hire_avg['avg_work_conditions'] + hire_avg['avg_overtime'] + hire_avg['avg_other_premiums'],
                    hire_avg['avg_annual_bonus'] + hire_avg['avg_profit_sharing'],
                    hire_avg['avg_social_security'] + hire_avg['avg_medicare'] + hire_avg['avg_401k'] + hire_avg[
                        'avg_pension'],
                    hire_avg['avg_ltips'],
                    hire_avg['avg_hiring_cost'],
                    0  # Will calculate total below
                ]
            }
            new_hire_df = pd.DataFrame(new_hire_data)

            # Calculate total
            new_hire_df.loc[6, 'Amount'] = new_hire_df['Amount'].sum()

            # Format as currency
            new_hire_df['Amount'] = new_hire_df['Amount'].apply(format_currency)

            # Display breakdown
            st.table(new_hire_df)

            # Total additional cost
            total_add_cost = hire_avg['avg_base_salary'] + \
                             hire_avg['avg_work_conditions'] + hire_avg['avg_overtime'] + hire_avg[
                                 'avg_other_premiums'] + \
                             hire_avg['avg_annual_bonus'] + hire_avg['avg_profit_sharing'] + \
                             hire_avg['avg_social_security'] + hire_avg['avg_medicare'] + hire_avg['avg_401k'] + \
                             hire_avg['avg_pension'] + \
                             hire_avg['avg_ltips'] + hire_avg['avg_hiring_cost']

            st.markdown(
                f"**Total Additional Cost for {num_new_hires} New Hires:** {format_currency(total_add_cost * num_new_hires)}")
        else:
            st.markdown("No new hires selected for 2025 projection.")

    with col2:
        st.markdown("### Cost Growth Analysis")

        # Calculate growth from 2024 to projected 2025
        growth_categories = [
            'Base Salary',
            'Premiums',
            'Bonuses',
            'Social Contributions',
            'LTIPs',
            'Total Cost'
        ]

        growth_2024_values = [
            totals_2024['total_base_salary'],
            totals_2024['total_premiums'],
            totals_2024['total_bonuses'],
            totals_2024['total_social_contributions'],
            totals_2024['total_ltips'],
            totals_2024['total_cost']
        ]

        growth_2025_values = [
            projected_totals['total_base_salary'],
            projected_totals['total_premiums'],
            projected_totals['total_bonuses'],
            projected_totals['total_social_contributions'],
            projected_totals['total_ltips'],
            projected_totals['total_cost']
        ]

        growth_rates = [
            ((g2025 - g2024) / g2024 * 100).round(2)
            for g2025, g2024 in zip(growth_2025_values, growth_2024_values)
        ]

        # Create growth chart
        growth_df = pd.DataFrame({
            'Category': growth_categories,
            'Growth Rate (%)': growth_rates
        })

        fig = px.bar(
            growth_df,
            x='Category',
            y='Growth Rate (%)',
            title='Projected 2024-2025 Growth Rates',
            color='Growth Rate (%)',
            color_continuous_scale='Blues',
            text='Growth Rate (%)'
        )

        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

        st.plotly_chart(fig, use_container_width=True)

    # Add what-if scenario analysis section
    st.markdown("### What-If Scenario Analysis")

    # Create a range of potential new hire counts
    scenario_hires = list(range(0, 21, 5))
    if num_new_hires > 0 and num_new_hires not in scenario_hires:
        scenario_hires.append(num_new_hires)
        scenario_hires.sort()

    # Calculate scenarios
    scenario_data = []

    for hire_count in scenario_hires:
        scenario = project_costs_for_new_hires(
            totals_2024,
            band_avg_df,
            selected_hire_dept,
            selected_band,
            hire_count
        )

        row = {
            'New Hires': hire_count,
            'Total Cost': scenario['total_cost'],
            'Cost Increase': scenario['total_cost'] - totals_2024['total_cost'],
            'Percent Increase': (
                        (scenario['total_cost'] - totals_2024['total_cost']) / totals_2024['total_cost'] * 100).round(2)
        }

        scenario_data.append(row)

    scenario_df = pd.DataFrame(scenario_data)

    # Create scenario chart
    fig = px.line(
        scenario_df,
        x='New Hires',
        y='Total Cost',
        title=f'Cost Projection Scenarios for {selected_hire_dept} {selected_band} Hires',
        markers=True
    )

    # Add second y-axis for percentage increase
    fig.add_trace(
        go.Scatter(
            x=scenario_df['New Hires'],
            y=scenario_df['Percent Increase'],
            name='Percent Increase',
            yaxis='y2',
            line=dict(color='red', dash='dot'),
            mode='lines+markers'
        )
    )

    # Update layout for dual y-axes
    fig.update_layout(
        yaxis=dict(title='Total Cost ($)'),
        yaxis2=dict(title='Percent Increase (%)', overlaying='y', side='right'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # Add dollar sign formatting to primary y-axis
    fig.update_yaxes(tickprefix="$", tickformat=",", secondary_y=False)
    fig.update_yaxes(ticksuffix="%", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # Display scenario table
    scenario_df['Total Cost'] = scenario_df['Total Cost'].apply(format_currency)
    scenario_df['Cost Increase'] = scenario_df['Cost Increase'].apply(format_currency)
    scenario_df['Percent Increase'] = scenario_df['Percent Increase'].apply(lambda x: f"{x}%")

    st.table(scenario_df)

# Footer
st.markdown("---")
st.markdown("**Cost of Labor Dashboard** | Developed for HR Analytics | Last updated: February 2025")

# Add download links for sample data
st.sidebar.markdown("---")
st.sidebar.markdown("### Download Sample Data Templates")

# Create download buttons for sample CSVs
if st.sidebar.button("Download Employee Data Template"):
    csv = df_2023.to_csv(index=False)
    st.sidebar.download_button(
        label="Click to Download",
        data=csv,
        file_name="employee_data_template.csv",
        mime="text/csv"
    )

if st.sidebar.button("Download Band Averages Template"):
    csv = band_avg_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Click to Download",
        data=csv,
        file_name="band_averages_template.csv",
        mime="text/csv"
    )
