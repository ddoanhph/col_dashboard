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
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #4B5563;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #2563EB;
    }
    .stDataFrame {
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


# Helper functions for calculations
def preprocess_data(df):
    """Preprocess and clean the data for analysis"""
    # Create a copy to avoid modifying the original
    data = df.copy()

    # Extract department from 'Business Title' if needed
    if 'department' not in data.columns and 'Business Title' in data.columns:
        # Extract department (this is a placeholder - adjust based on actual data format)
        data['department'] = data['Business Title'].str.split(' - ').str[0]

    # Extract band from 'Business Title' if needed
    if 'band' not in data.columns and 'Business Title' in data.columns:
        # This is a placeholder - adjust based on actual data format
        # For example, if the business title contains band info like "Manager - B4"
        data['band'] = data['Business Title'].str.extract(r'([A-Z][0-9]+)')
        # If no clear pattern, just use employee class as a substitute
        if 'EE Class' in data.columns and data['band'].isna().all():
            data['band'] = data['EE Class']

    # Fill missing departments and bands with placeholder values
    if 'department' in data.columns:
        data['department'].fillna('Uncategorized', inplace=True)
    if 'band' in data.columns:
        data['band'].fillna('Uncategorized', inplace=True)

    # Fix data types
    for col in data.columns:
        if any(substring in col for substring in ['Amount', 'PAY', 'Bonus', 'Premiums', 'Tax', 'PENSION', 'LTIPs']):
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
            except:
                pass

    return data


def calculate_totals(df):
    """Calculate total cost metrics from employee data"""
    # Calculate base salary
    base_salary_cols = ['REG PAY 1', 'Salary OT']
    base_salary = sum([df[col].sum() for col in base_salary_cols if col in df.columns])

    # Calculate premiums
    premium_cols = [
        'Premium Hours', 'Call0InPay', 'Special Project Pay', 'Essential Pay',
        'shift Differential (SHD)', 'Flight Training Earns', 'FAL Inspector Earns',
        'FAL 0 Lead Differntial', '2nd Shift Premium'
    ]
    premiums = sum([df[col].sum() for col in premium_cols if col in df.columns])

    # Calculate overtime
    overtime_cols = ['OT PAY', 'Double Time', 'Double Time and Half']
    overtime = sum([df[col].sum() for col in overtime_cols if col in df.columns])

    # Calculate bonuses
    bonus_cols = [
        'Reg.Bonus', 'Performance Bonus', 'Collective Bonus', 'Comm.Bonus',
        'ForeignBonus', 'Profit Sharing', 'Global Bonus', 'Retention Bonus',
        'Sign on bonus', 'Incentive Pay'
    ]
    bonuses = sum([df[col].sum() for col in bonus_cols if col in df.columns])

    # Calculate social contributions
    social_cols = ['Social Security Tax', 'Medicare', 'ER 401K', 'ER PENSION']
    social_contributions = sum([df[col].sum() for col in social_cols if col in df.columns])

    # Calculate LTIPs
    ltip_cols = ['LTIPs1', 'LTIPs 2', 'LTIPs Foreign LTIP', 'Stocks']
    ltips = sum([df[col].sum() for col in ltip_cols if col in df.columns])

    # Calculate additional benefits
    benefit_cols = [
        'Housing Cost', 'Cost of Living', 'Co. Car', 'Movings', 'Childcare',
        'Expat Premiums', 'Tuitions', 'Other Mobility Pay', 'Relocation'
    ]
    benefits = sum([df[col].sum() for col in benefit_cols if col in df.columns])

    # Calculate hours
    planned_hours = df['CoL Planned Hours'].sum() if 'CoL Planned Hours' in df.columns else 0
    if planned_hours == 0 and 'Standard Hours' in df.columns:
        # Fallback to standard hours x number of employees
        planned_hours = df['Standard Hours'].sum()

    actual_hours_cols = ['Reg Hours', 'OT Hours', 'Retro OT', 'Retro Doubletime']
    actual_hours = sum([df[col].sum() for col in actual_hours_cols if col in df.columns])

    # Calculate absence hours
    absence_hours_cols = ['Sick Hours', 'STD Hours', 'Other Absences', 'Vac & PTO Hours', 'Holiday Hrs']
    absence_hours = sum([df[col].sum() for col in absence_hours_cols if col in df.columns])

    # Calculate FTE count
    fte_count = len(df)
    if 'CoL: FTE/AWF' in df.columns:
        # If we have an FTE column, use the sum of this column
        fte_count = df['CoL: FTE/AWF'].sum()

    # Calculate gross amount
    gross_amount = df['Gross Amount'].sum() if 'Gross Amount' in df.columns else 0

    # Calculate total cost of labor
    total_cost = base_salary + premiums + overtime + bonuses + social_contributions + ltips + benefits

    # If gross amount is available and greater than calculated total, use it
    if gross_amount > total_cost:
        total_cost = gross_amount

    # Store all the calculated values
    totals = {
        'total_base_salary': base_salary,
        'total_premiums': premiums,
        'total_overtime': overtime,
        'total_bonuses': bonuses,
        'total_social_contributions': social_contributions,
        'total_ltips': ltips,
        'total_benefits': benefits,
        'total_planned_hours': planned_hours,
        'total_actual_hours': actual_hours,
        'total_absence_hours': absence_hours,
        'total_fte': fte_count,
        'total_cost': total_cost
    }

    # Calculate FTE costs
    totals['fte_costs'] = totals['total_cost'] / totals['total_fte'] if totals['total_fte'] > 0 else 0

    # Calculate overtime hours
    overtime_hours_cols = ['OT Hours', 'Retro OT', 'Retro Doubletime']
    totals['total_overtime_hours'] = sum([df[col].sum() for col in overtime_hours_cols if col in df.columns])

    return totals


def calculate_absence_costs(df, totals):
    """Calculate absence costs based on hourly rate and absence hours"""
    if totals['total_planned_hours'] > 0:
        # Calculate hourly rate
        hourly_rate = totals['total_base_salary'] / totals['total_planned_hours']

        # Calculate absence cost
        return hourly_rate * totals['total_absence_hours']
    return 0


def calculate_band_averages(df):
    """Calculate average costs by department and band"""
    # If we don't have department or band, return empty DataFrame
    if 'department' not in df.columns or 'band' not in df.columns:
        return pd.DataFrame()

    # Group by department and band
    band_avg = df.groupby(['department', 'band']).agg({
        'REG PAY 1': 'mean',
        'Premium Hours': 'mean',
        'OT PAY': 'mean',
        'Reg.Bonus': 'mean',
        'Performance Bonus': 'mean',
        'Profit Sharing': 'mean',
        'Social Security Tax': 'mean',
        'Medicare': 'mean',
        'ER 401K': 'mean',
        'ER PENSION': 'mean',
        'LTIPs1': 'mean',
        'LTIPs 2': 'mean',
        'CoL Planned Hours': 'mean',
        'Gross Amount': 'mean'
    }).reset_index()

    # Rename columns
    band_avg.columns = [
        'department', 'band', 'avg_base_salary', 'avg_work_conditions', 'avg_overtime',
        'avg_reg_bonus', 'avg_perf_bonus', 'avg_profit_sharing',
        'avg_social_security', 'avg_medicare', 'avg_401k', 'avg_pension',
        'avg_ltips1', 'avg_ltips2', 'avg_planned_hours', 'avg_gross'
    ]

    # Calculate other derived fields
    band_avg['avg_other_premiums'] = 0  # If available
    band_avg['avg_hiring_cost'] = band_avg['avg_gross'] * 0.15  # Estimate hiring cost as 15% of gross

    # Fill NAs with zeros
    band_avg = band_avg.fillna(0)

    return band_avg


def project_costs_for_new_hires(current_totals, band_avg_df, dept, band, num_new_hires):
    """Project costs for next year based on current totals and new hires"""
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
    projections['total_premiums'] += (avg['avg_work_conditions'] + avg['avg_other_premiums']) * num_new_hires
    projections['total_overtime'] += avg['avg_overtime'] * num_new_hires
    projections['total_bonuses'] += (avg['avg_reg_bonus'] + avg['avg_perf_bonus'] + avg[
        'avg_profit_sharing']) * num_new_hires
    projections['total_social_contributions'] += (avg['avg_social_security'] + avg['avg_medicare'] + avg['avg_401k'] +
                                                  avg['avg_pension']) * num_new_hires
    projections['total_ltips'] += (avg['avg_ltips1'] + avg['avg_ltips2']) * num_new_hires
    projections['total_fte'] += num_new_hires
    projections['total_planned_hours'] += avg['avg_planned_hours'] * num_new_hires

    # Recalculate total cost
    projections['total_cost'] = (
            projections['total_base_salary'] +
            projections['total_premiums'] +
            projections['total_overtime'] +
            projections['total_bonuses'] +
            projections['total_social_contributions'] +
            projections['total_ltips'] +
            projections['total_benefits']
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
        # Load the CSV files
        df_2023 = pd.read_csv('Cost_of_Labor_Report_2023.csv')
        df_2024 = pd.read_csv('Cost_of_Labor_Report_2024.csv')

        # Preprocess the data
        df_2023 = preprocess_data(df_2023)
        df_2024 = preprocess_data(df_2024)

        # Generate band averages
        band_avg_df = calculate_band_averages(df_2024)  # Use 2024 data for projections

        return df_2023, df_2024, band_avg_df
    except Exception as e:
        st.warning(f"Could not load actual data files: {e}. Using sample data for demonstration.")

        # Create sample employee data that's aligned with the new dataset structure
        # This is more complex with our new structure, but we'll create representative data

        # Define departments and bands
        departments = ['ERP', 'ERA', 'ERZ', '8VC', 'MMS', 'WC', 'BC']
        bands = ['B3', 'B4', 'B5', 'B6']

        # Create empty DataFrames with the expected columns
        df_2023 = pd.DataFrame()
        df_2024 = pd.DataFrame()

        # Generate 100 sample employees
        num_employees = 100

        # Create basic employee info
        df_2023['Corp ID'] = range(1001, 1001 + num_employees)
        df_2023['Pay Group'] = np.random.choice(['A', 'B', 'C'], size=num_employees)
        df_2023['EE Class'] = np.random.choice(['FT', 'PT'], size=num_employees)
        df_2023['Last Name'] = ['Employee' + str(i) for i in range(num_employees)]
        df_2023['First Name'] = ['Test' + str(i) for i in range(num_employees)]
        df_2023['Business Title'] = [np.random.choice(departments) + ' - Level ' + np.random.choice(bands) for _ in
                                     range(num_employees)]
        df_2023['Standard Hours'] = 2080
        df_2023['CoL: FTE/AWF'] = 1.0
        df_2023['CoL Planned Hours'] = 2080

        # Generate hours
        df_2023['Reg Hours'] = np.random.normal(2000, 50, num_employees)
        df_2023['OT Hours'] = np.random.normal(80, 20, num_employees)
        df_2023['Sick Hours'] = np.random.normal(40, 10, num_employees)
        df_2023['Vac & PTO Hours'] = np.random.normal(80, 20, num_employees)
        df_2023['Holiday Hrs'] = 80

        # Generate pay components for 2023
        base_salary_range = (60000, 120000)
        df_2023['REG PAY 1'] = np.random.uniform(base_salary_range[0], base_salary_range[1], num_employees)
        df_2023['OT PAY'] = df_2023['REG PAY 1'] * 0.05  # 5% of base
        df_2023['Premium Hours'] = df_2023['REG PAY 1'] * 0.02  # 2% of base
        df_2023['Reg.Bonus'] = df_2023['REG PAY 1'] * 0.1  # 10% of base
        df_2023['Performance Bonus'] = df_2023['REG PAY 1'] * 0.05  # 5% of base
        df_2023['Profit Sharing'] = df_2023['REG PAY 1'] * 0.03  # 3% of base
        df_2023['Social Security Tax'] = df_2023['REG PAY 1'] * 0.062  # 6.2% of base
        df_2023['Medicare'] = df_2023['REG PAY 1'] * 0.0145  # 1.45% of base
        df_2023['ER 401K'] = df_2023['REG PAY 1'] * 0.04  # 4% of base
        df_2023['ER PENSION'] = df_2023['REG PAY 1'] * 0.02  # 2% of base
        df_2023['LTIPs1'] = df_2023['REG PAY 1'] * 0.03  # 3% of base
        df_2023['LTIPs 2'] = df_2023['REG PAY 1'] * 0.02  # 2% of base

        # Calculate Gross Amount
        df_2023['Gross Amount'] = (
                df_2023['REG PAY 1'] + df_2023['OT PAY'] + df_2023['Premium Hours'] +
                df_2023['Reg.Bonus'] + df_2023['Performance Bonus'] + df_2023['Profit Sharing']
        )

        # Create 2024 data with 5% increase
        df_2024 = df_2023.copy()
        for col in [
            'REG PAY 1', 'OT PAY', 'Premium Hours', 'Reg.Bonus', 'Performance Bonus',
            'Profit Sharing', 'Social Security Tax', 'Medicare', 'ER 401K', 'ER PENSION',
            'LTIPs1', 'LTIPs 2', 'Gross Amount'
        ]:
            df_2024[col] = df_2024[col] * 1.05

        # Extract department and band from Business Title for easier analysis
        for year_df in [df_2023, df_2024]:
            year_df['department'] = year_df['Business Title'].str.split(' - ').str[0]
            year_df['band'] = year_df['Business Title'].str.split('Level ').str[1]

        # Calculate band averages
        band_avg_df = calculate_band_averages(df_2024)

        return df_2023, df_2024, band_avg_df


# Load data
df_2023, df_2024, band_avg_df = load_data()

# Calculate totals for 2023 and 2024
totals_2023 = calculate_totals(df_2023)
totals_2024 = calculate_totals(df_2024)

# Calculate absence costs
totals_2023['total_absence_costs'] = calculate_absence_costs(df_2023, totals_2023)
totals_2024['total_absence_costs'] = calculate_absence_costs(df_2024, totals_2024)

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
all_departments = sorted(pd.concat([df_2023['department'], df_2024['department']]).unique())
selected_dept = st.sidebar.selectbox(
    "Filter by Department",
    ["All Departments"] + all_departments
)

# Initialize projection parameters
num_new_hires = 0
selected_band = band_avg_df['band'].iloc[0] if not band_avg_df.empty else "B4"
selected_hire_dept = band_avg_df['department'].iloc[0] if not band_avg_df.empty else "ERP"
projected_totals = totals_2024.copy()

# Only show projection inputs when 2025 is selected
if selected_year == "2025 (Projected)":
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 2025 Projection Parameters")

    num_new_hires = st.sidebar.number_input("Number of New Hires", min_value=0, max_value=100, value=0)

    if not band_avg_df.empty:
        # Only show departments and bands that exist in the band_avg_df
        dept_options = sorted(band_avg_df['department'].unique())
        selected_hire_dept = st.sidebar.selectbox(
            "Department for New Hires",
            dept_options
        )

        # Filter bands based on selected department
        band_options = sorted(band_avg_df[band_avg_df['department'] == selected_hire_dept]['band'].unique())
        selected_band = st.sidebar.selectbox(
            "Band for New Hires",
            band_options
        )

        # Calculate projections based on inputs
        if num_new_hires > 0:
            projected_totals = project_costs_for_new_hires(
                totals_2024,
                band_avg_df,
                selected_hire_dept,
                selected_band,
                num_new_hires
            )
    else:
        st.sidebar.warning("No band average data available for projections.")

# Filter data based on selected department
if selected_dept != "All Departments":
    df_2023_filtered = df_2023[df_2023["department"] == selected_dept]
    df_2024_filtered = df_2024[df_2024["department"] == selected_dept]
    dept_totals_2023 = calculate_totals(df_2023_filtered)
    dept_totals_2024 = calculate_totals(df_2024_filtered)

    # Calculate absence costs for filtered data
    dept_totals_2023['total_absence_costs'] = calculate_absence_costs(df_2023_filtered, dept_totals_2023)
    dept_totals_2024['total_absence_costs'] = calculate_absence_costs(df_2024_filtered, dept_totals_2024)
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
        if selected_hire_dept == selected_dept and num_new_hires > 0:
            current_totals = project_costs_for_new_hires(
                dept_totals_2024,
                band_avg_df,
                selected_hire_dept,
                selected_band,
                num_new_hires
            )
    else:
        current_totals = projected_totals

# Display main KPIs
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-label">Total Cost of Labor</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{format_currency(current_totals["total_cost"])}</div>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-label">Total FTE</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{current_totals["total_fte"]:.1f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-label">Cost per FTE</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{format_currency(current_totals["fte_costs"])}</div>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Cost Breakdown", "Year-over-Year Comparison", "Detailed Analysis"])

# Tab 1: Cost Breakdown
with tab1:
    st.markdown('<div class="sub-header">Cost Breakdown</div>', unsafe_allow_html=True)

    # Create cost breakdown chart
    cost_data = {
        'Category': ['Base Salary', 'Premiums', 'Overtime', 'Bonuses', 'Social Contributions', 'LTIPs', 'Benefits'],
        'Amount': [
            current_totals['total_base_salary'],
            current_totals['total_premiums'],
            current_totals['total_overtime'],
            current_totals['total_bonuses'],
            current_totals['total_social_contributions'],
            current_totals['total_ltips'],
            current_totals['total_benefits']
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
        'total_premiums'] * 100).round(1) if compare_2023['total_premiums'] > 0 else 0
    growth_overtime = ((compare_2024['total_overtime'] - compare_2023['total_overtime']) / compare_2023[
        'total_overtime'] * 100).round(1) if compare_2023['total_overtime'] > 0 else 0
    growth_bonuses = ((compare_2024['total_bonuses'] - compare_2023['total_bonuses']) / compare_2023[
        'total_bonuses'] * 100).round(1) if compare_2023['total_bonuses'] > 0 else 0
    growth_social = ((compare_2024['total_social_contributions'] - compare_2023['total_social_contributions']) /
                    compare_2023['total_social_contributions'] * 100).round(1) if compare_2023['total_social_contributions'] > 0 else 0
    growth_ltips = (
                (compare_2024['total_ltips'] - compare_2023['total_ltips']) / compare_2023['total_ltips'] * 100).round(
        1) if compare_2023['total_ltips'] > 0 else 0
    growth_benefits = ((compare_2024['total_benefits'] - compare_2023['total_benefits']) / compare_2023[
        'total_benefits'] * 100).round(1) if compare_2023['total_benefits'] > 0 else 0
    growth_total = ((compare_2024['total_cost'] - compare_2023['total_cost']) / compare_2023['total_cost'] * 100).round(
        1)
    growth_fte = ((compare_2024['total_fte'] - compare_2023['total_fte']) / compare_2023['total_fte'] * 100).round(1) if
    compare_2023['total_fte'] > 0 else 0
    growth_fte_cost = ((compare_2024['fte_costs'] - compare_2023['fte_costs']) / compare_2023['fte_costs'] * 100).round(
        1) if compare_2023['fte_costs'] > 0 else 0

    # Create comparison data for chart
    compare_data = {
        'Category': ['Base Salary', 'Premiums', 'Overtime', 'Bonuses', 'Social Contributions', 'LTIPs', 'Benefits',
                     'Total Cost'],
        '2023': [
            compare_2023['total_base_salary'],
            compare_2023['total_premiums'],
            compare_2023['total_overtime'],
            compare_2023['total_bonuses'],
            compare_2023['total_social_contributions'],
            compare_2023['total_ltips'],
            compare_2023['total_benefits'],
            compare_2023['total_cost']
        ],
        '2024': [
            compare_2024['total_base_salary'],
            compare_2024['total_premiums'],
            compare_2024['total_overtime'],
            compare_2024['total_bonuses'],
            compare_2024['total_social_contributions'],
            compare_2024['total_ltips'],
            compare_2024['total_benefits'],
            compare_2024['total_cost']
        ],
        'Growth (%)': [
            growth_base,
            growth_premiums,
            growth_overtime,
            growth_bonuses,
            growth_social,
            growth_ltips,
            growth_benefits,
            growth_total
        ]
    }
    compare_df = pd.DataFrame(compare_data)

    # Create bar chart for comparison
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=compare_df['Category'],
        y=compare_df['2023'],
        name='2023',
        marker_color='#94C4F5'
    ))

    fig.add_trace(go.Bar(
        x=compare_df['Category'],
        y=compare_df['2024'],
        name='2024',
        marker_color='#1E3A8A'
    ))

    fig.update_layout(
        title='2023 vs 2024 Cost Comparison',
        xaxis_title='Category',
        yaxis_title='Amount ($)',
        barmode='group',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Create table with growth rates
    st.markdown('<div class="sub-header">Year-over-Year Growth</div>', unsafe_allow_html=True)
    growth_df = compare_df.copy()

    # Format columns for display
    growth_df['2023'] = growth_df['2023'].apply(format_currency)
    growth_df['2024'] = growth_df['2024'].apply(format_currency)
    growth_df['Growth (%)'] = growth_df['Growth (%)'].apply(lambda x: f"{x}%")

    st.table(growth_df)

    # Add additional KPIs for comparison
    st.markdown('<div class="sub-header">Additional Metrics Comparison</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-label">FTE Growth</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{growth_fte:.1f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-label">FTE Cost Growth</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{growth_fte_cost:.1f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with col3:
    # Calculate overtime ratio change
        ot_ratio_2023 = compare_2023['total_overtime'] / compare_2023['total_base_salary'] * 100 if compare_2023[
                                                                                                        'total_base_salary'] > 0 else 0
    ot_ratio_2024 = compare_2024['total_overtime'] / compare_2024['total_base_salary'] * 100 if compare_2024[
                                                                                                    'total_base_salary'] > 0 else 0
    ot_ratio_change = ot_ratio_2024 - ot_ratio_2023

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-label">Overtime Ratio Change</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{ot_ratio_change:.1f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Show 2025 projections if selected
    if selected_year == "2025 (Projected)":
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("### 2025 Projections vs 2024 Actuals")

    # Calculate growth rates from 2024 to 2025
    growth_2025 = (
                (current_totals['total_cost'] - compare_2024['total_cost']) / compare_2024['total_cost'] * 100).round(1)
    growth_fte_2025 = (
                (current_totals['total_fte'] - compare_2024['total_fte']) / compare_2024['total_fte'] * 100).round(1)

    st.markdown(f"**Projected Cost Growth:** {growth_2025}%")
    st.markdown(f"**Projected FTE Growth:** {growth_fte_2025}%")
    st.markdown(f"**Projected Total Cost:** {format_currency(current_totals['total_cost'])}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Tab 3: Detailed Analysis
    with tab3:
        st.markdown('<div class="sub-header">Detailed Analysis</div>', unsafe_allow_html=True)

    # Determine which dataset to use
    if selected_year == "2023":
        analysis_df = df_2023_filtered
    elif selected_year == "2024":
        analysis_df = df_2024_filtered
    else:  # 2025 - use 2024 data as base
        analysis_df = df_2024_filtered

    # Create subtabs for detailed analysis
    subtab1, subtab2, subtab3, subtab4 = st.tabs(
        ["Department Analysis", "Band Analysis", "Overtime Analysis", "Absence Analysis"])

    # Subtab 1: Department Analysis
    with subtab1:
        if
    'department' in analysis_df.columns:
    # Group by department
    dept_analysis = analysis_df.groupby('department').agg({
        'Corp ID': 'count',
        'CoL: FTE/AWF': 'sum',
        'REG PAY 1': 'sum',
        'OT PAY': 'sum',
        'Premium Hours': 'sum',
        'Reg.Bonus': 'sum',
        'Performance Bonus': 'sum',
        'Gross Amount': 'sum'
    }).reset_index()

    # Rename columns
    dept_analysis.columns = [
        'Department', 'Employee Count', 'FTE', 'Base Salary', 'Overtime',
        'Premiums', 'Regular Bonus', 'Performance Bonus', 'Gross Amount'
    ]

    # Calculate average cost per FTE
    dept_analysis['Cost per FTE'] = dept_analysis['Gross Amount'] / dept_analysis['FTE']

    # Sort by Gross Amount
    dept_analysis = dept_analysis.sort_values('Gross Amount', ascending=False)

    # Create department bar chart
    fig = px.bar(
        dept_analysis,
        x='Department',
        y='Gross Amount',
        color='Cost per FTE',
        color_continuous_scale='Blues',
        title='Department Cost Analysis',
        labels={'Department': 'Department', 'Gross Amount': 'Total Cost', 'Cost per FTE': 'Cost per FTE ($)'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Format numeric columns for display
    for col in ['Base Salary', 'Overtime', 'Premiums', 'Regular Bonus', 'Performance Bonus', 'Gross Amount',
                'Cost per FTE']:
        dept_analysis[col] = dept_analysis[col].apply(format_currency)

    # Display the detailed table
    st.dataframe(dept_analysis)

    # Add a download button for the data
    @ st.cache_data


    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df_to_csv(dept_analysis)
    st.download_button(
        "Download Department Analysis",
        csv,
        f"department_analysis_{selected_year}.csv",
        "text/csv",
        key='download-dept-csv'
    )
else:
st.warning("Department information is not available in the dataset.")

# Subtab 2: Band Analysis
with subtab2:
    if 'band' in analysis_df.columns:
        # Group by band
        band_analysis = analysis_df.groupby('band').agg({
            'Corp ID': 'count',
            'CoL: FTE/AWF': 'sum',
            'REG PAY 1': ['sum', 'mean'],
            'OT PAY': ['sum', 'mean'],
            'Premium Hours': ['sum', 'mean'],
            'Reg.Bonus': ['sum', 'mean'],
            'Performance Bonus': ['sum', 'mean'],
            'Gross Amount': ['sum', 'mean']
        }).reset_index()

        # Flatten multi-index
        band_analysis.columns = [
            'Band', 'Employee Count', 'FTE',
            'Total Base Salary', 'Avg Base Salary',
            'Total Overtime', 'Avg Overtime',
            'Total Premiums', 'Avg Premiums',
            'Total Regular Bonus', 'Avg Regular Bonus',
            'Total Performance Bonus', 'Avg Performance Bonus',
            'Total Gross Amount', 'Avg Gross Amount'
        ]

        # Create band comparison chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=band_analysis['Band'],
            y=band_analysis['Avg Base Salary'],
            name='Base Salary',
            marker_color='#94C4F5'
        ))

        fig.add_trace(go.Bar(
            x=band_analysis['Band'],
            y=band_analysis['Avg Overtime'],
            name='Overtime',
            marker_color='#3B82F6'
        ))

        fig.add_trace(go.Bar(
            x=band_analysis['Band'],
            y=band_analysis['Avg Regular Bonus'] + band_analysis['Avg Performance Bonus'],
            name='Bonuses',
            marker_color='#1E3A8A'
        ))

        fig.update_layout(
            title='Average Compensation by Band',
            xaxis_title='Band',
            yaxis_title='Amount ($)',
            barmode='stack',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Format numeric columns for display
        for col in band_analysis.columns:
            if any(substr in col for substr in ['Total', 'Avg']):
                band_analysis[col] = band_analysis[col].apply(format_currency)

        # Display the detailed table
        st.dataframe(band_analysis)

        # Add a download button for the data
        csv = convert_df_to_csv(band_analysis)
        st.download_button(
            "Download Band Analysis",
            csv,
            f"band_analysis_{selected_year}.csv",
            "text/csv",
            key='download-band-csv'
        )
    else:
        st.warning("Band information is not available in the dataset.")

    # Subtab 3: Overtime Analysis
with subtab3:
    # Check if overtime columns exist
    ot_cols = [col for col in analysis_df.columns if
               any(substr in col for substr in ['OT Hours', 'OT PAY', 'Double Time'])]

    if ot_cols:
        # Create columns for overtime analysis
        col1, col2 = st.columns(2)

        with col1:
            # Calculate overtime as percentage of base
            ot_ratio = current_totals['total_overtime'] / current_totals['total_base_salary'] * 100 if current_totals[
                                                                                                           'total_base_salary'] > 0 else 0

            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Overtime as % of Base Salary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{ot_ratio:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Calculate overtime hours
            ot_hours = current_totals['total_overtime_hours'] if 'total_overtime_hours' in current_totals else 0

            st.markdown('<div class="metric-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Total Overtime Hours</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{ot_hours:,.0f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # Calculate overtime cost per hour
            ot_cost_per_hour = current_totals['total_overtime'] / ot_hours if ot_hours > 0 else 0

            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Average Overtime Cost per Hour</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(ot_cost_per_hour)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Calculate regular cost per hour for comparison
            reg_cost_per_hour = current_totals['total_base_salary'] / current_totals['total_planned_hours'] if \
            current_totals['total_planned_hours'] > 0 else 0

            # Calculate overtime premium
            ot_premium = (ot_cost_per_hour / reg_cost_per_hour - 1) * 100 if reg_cost_per_hour > 0 else 0

            st.markdown('<div class="metric-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Overtime Premium</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{ot_premium:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # If we have department information, analyze overtime by department
        if 'department' in analysis_df.columns:
            st.markdown('<div class="sub-header">Overtime by Department</div>', unsafe_allow_html=True)

            # Group by department for overtime analysis
            ot_dept = analysis_df.groupby('department').agg({
                'OT Hours': 'sum',
                'OT PAY': 'sum',
                'REG PAY 1': 'sum',
                'Reg Hours': 'sum'
            }).reset_index()

            # Calculate overtime ratios
            ot_dept['OT to Base Ratio (%)'] = (ot_dept['OT PAY'] / ot_dept['REG PAY 1'] * 100).round(1)
            ot_dept['OT Hours to Reg Hours (%)'] = (ot_dept['OT Hours'] / ot_dept['Reg Hours'] * 100).round(1)

            # Sort by overtime ratio
            ot_dept = ot_dept.sort_values('OT to Base Ratio (%)', ascending=False)

            # Create a bar chart for overtime by department
            fig = px.bar(
                ot_dept,
                x='department',
                y='OT to Base Ratio (%)',
                color='OT Hours',
                color_continuous_scale='Blues',
                title='Overtime to Base Salary Ratio by Department',
                labels={'department': 'Department', 'OT to Base Ratio (%)': 'OT as % of Base Salary',
                        'OT Hours': 'Total OT Hours'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Format columns for display
            ot_dept['OT PAY'] = ot_dept['OT PAY'].apply(format_currency)
            ot_dept['REG PAY 1'] = ot_dept['REG PAY 1'].apply(format_currency)
            ot_dept['OT to Base Ratio (%)'] = ot_dept['OT to Base Ratio (%)'].apply(lambda x: f"{x}%")
            ot_dept['OT Hours to Reg Hours (%)'] = ot_dept['OT Hours to Reg Hours (%)'].apply(lambda x: f"{x}%")

            # Rename columns for display
            ot_dept.columns = ['Department', 'OT Hours', 'OT Pay', 'Base Salary', 'Regular Hours', 'OT to Base Ratio',
                               'OT to Reg Hours Ratio']

            # Display the data
            st.dataframe(ot_dept)
    else:
        st.warning("Overtime information is not available in the dataset.")

    # Subtab 4: Absence Analysis
with subtab4:
    # Check if absence columns exist
    absence_cols = [col for col in analysis_df.columns if
                    any(substr in col for substr in ['Sick', 'STD', 'PTO', 'Absence', 'Holiday'])]

    if absence_cols:
        # Calculate total working days per year (approximately)
        working_days_per_year = 260  # 52 weeks * 5 days

        # Create columns for absence analysis
        col1, col2 = st.columns(2)

        with col1:
            # Calculate absence rate
            absence_rate = (current_totals['total_absence_hours'] / (
                        current_totals['total_planned_hours'] + current_totals['total_absence_hours']) * 100) if (
                                                                                                                             current_totals[
                                                                                                                                 'total_planned_hours'] +
                                                                                                                             current_totals[
                                                                                                                                 'total_absence_hours']) > 0 else 0

            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Absence Rate</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{absence_rate:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Calculate absence days per employee (assuming 8-hour workday)
            absence_days_per_employee = (current_totals['total_absence_hours'] / 8 / current_totals['total_fte']) if \
            current_totals['total_fte'] > 0 else 0

            st.markdown('<div class="metric-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Absence Days per Employee</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{absence_days_per_employee:.1f} days</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # Calculate absence cost
            absence_cost = current_totals['total_absence_costs'] if 'total_absence_costs' in current_totals else 0

            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Total Absence Cost</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(absence_cost)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Calculate absence cost per employee
            absence_cost_per_employee = absence_cost / current_totals['total_fte'] if current_totals[
                                                                                          'total_fte'] > 0 else 0

            st.markdown('<div class="metric-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Absence Cost per Employee</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{format_currency(absence_cost_per_employee)}</div>',
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Create breakdown of absence types
        absence_types = {
            'Type': ['Sick Leave', 'Short-Term Disability', 'Vacation & PTO', 'Holiday', 'Other Absences'],
            'Hours': [
                analysis_df['Sick Hours'].sum() if 'Sick Hours' in analysis_df.columns else 0,
                analysis_df['STD Hours'].sum() if 'STD Hours' in analysis_df.columns else 0,
                analysis_df['Vac & PTO Hours'].sum() if 'Vac & PTO Hours' in analysis_df.columns else 0,
                analysis_df['Holiday Hrs'].sum() if 'Holiday Hrs' in analysis_df.columns else 0,
                analysis_df['Other Absences'].sum() if 'Other Absences' in analysis_df.columns else 0
            ]
        }
        absence_df = pd.DataFrame(absence_types)

        # Create pie chart for absence types
        fig = px.pie(
            absence_df,
            values='Hours',
            names='Type',
            title='Breakdown of Absence Hours by Type',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

        # If we have department information, analyze absences by department
        if 'department' in analysis_df.columns:
            st.markdown('<div class="sub-header">Absence by Department</div>', unsafe_allow_html=True)

            # Group by department for absence analysis
            absence_dept = analysis_df.groupby('department').agg({
                'Sick Hours': 'sum',
                'STD Hours': 'sum',
                'Vac & PTO Hours': 'sum',
                'Holiday Hrs': 'sum',
                'Other Absences': 'sum',
                'CoL: FTE/AWF': 'sum'
            }).reset_index()

            # Calculate total absence hours
            absence_dept['Total Absence Hours'] = absence_dept[
                ['Sick Hours', 'STD Hours', 'Vac & PTO Hours', 'Holiday Hrs', 'Other Absences']
            ].sum(axis=1)

            # Calculate absence days per employee
            absence_dept['Absence Days per Employee'] = (
                        absence_dept['Total Absence Hours'] / 8 / absence_dept['CoL: FTE/AWF']).round(1)

            # Sort by absence days per employee
            absence_dept = absence_dept.sort_values('Absence Days per Employee', ascending=False)

            # Create a bar chart for absence by department
            fig = px.bar(
                absence_dept,
                x='department',
                y='Absence Days per Employee',
                color='Total Absence Hours',
                color_continuous_scale='Blues',
                title='Absence Days per Employee by Department',
                labels={'department': 'Department', 'Absence Days per Employee': 'Days per Employee',
                        'Total Absence Hours': 'Total Hours'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Rename columns for display
            absence_dept.columns = [
                'Department', 'Sick Hours', 'STD Hours', 'Vacation & PTO',
                'Holiday Hours', 'Other Absences', 'FTE', 'Total Hours', 'Days per Employee'
            ]

            # Display the data
            st.dataframe(absence_dept)
    else:
        st.warning("Absence information is not available in the dataset.")

# Add a footer with information
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Cost of Labor Dashboard | Created with Streamlit | Last updated: March 2025</p>
    </div>
    """, unsafe_allow_html=True)