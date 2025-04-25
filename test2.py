# Required Imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import matplotlib.pyplot as plt # Not used in current version
import seaborn as sns # Used for table styling
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Cost of Labor Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .airbus-title {
        font-family: 'Poppins', sans-serif; /* MODIFIED: Use Poppins font */
        font-size: 1.8rem; /* Adjust size as needed */
        font-weight: 900; /* Extra bold */
        color: #00205B; /* Airbus dark blue */
        /* text-align: center; */ /* MODIFIED: Removed centering */
        text-align: left; /* Explicitly left-align */
        margin-bottom: 0.1rem; /* Small space below */
        letter-spacing: 1px; /* Optional: slight letter spacing */
        padding-left: 1rem; /* Add some padding to align with content */
    }

    .main-header {
        font-size: 4.0rem;
        font-weight: 700;
        color: #1E3A8A; /* Dark Blue */
        text-align: center;
        margin-bottom: 0.5rem;
    }
    /* Style for the description text below titles */
    .dashboard-description {
        text-align: center; /* Center the description */
        color: #4B5563; /* Grey text */
        margin-bottom: 2rem; /* Add space below description */
    }

    /* --- Keep other existing styles below --- */
    .sub-header {
        font-size: 2.5rem;
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
    .metric-label-container {
        display: flex;
        align-items: center; /* Vertically align icon and text */
        gap: 0.6rem;      /* Space between icon and text */
    }
    .metric-label {
        font-size: 0.95rem;
        color: #4B5563; /* Grey text */
        font-weight: 500;
        margin: 0; /* Remove default margins */
    }
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

st.markdown('<div class="airbus-title">AIRBUS</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Cost of Labor Dashboard</div>', unsafe_allow_html=True)
st.markdown("""
<div class="dashboard-description">
This dashboard provides a comprehensive view of labor costs for 2023, 2024, and projected costs for 2025.
It allows for simulation of new hires and their impact on the overall cost structure.
</div>
""", unsafe_allow_html=True)

# --- Helper Functions ---

# Format value as currency
def format_currency(value):
    """Format value as currency string."""
    if pd.isna(value):
        return "$NaN"
    return f"${value:,.2f}"

# Calculate total costs from employee data
def calculate_totals(df):
    """Calculate total cost metrics from employee data DataFrame."""
    # Ensure numeric types for calculations
    numeric_cols = ['base_salary', 'work_conditions_premium', 'overtime_premium', 'other_premiums',
                    'annual_bonus', 'profit_sharing', 'social_security_tax', 'medicare', 'er_401k',
                    'er_pension', 'ltips', 'planned_hours', 'actual_hours', 'sick_days', 'holiday_days',
                    'other_absences']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    totals = {}
    totals['total_base_salary'] = df['base_salary'].sum()
    totals['total_premiums'] = df['work_conditions_premium'].sum() + df['overtime_premium'].sum() + df['other_premiums'].sum()
    totals['total_bonuses'] = df['annual_bonus'].sum() + df['profit_sharing'].sum()
    totals['total_social_contributions'] = df['social_security_tax'].sum() + df['medicare'].sum() + df['er_401k'].sum() + df['er_pension'].sum()
    totals['total_ltips'] = df['ltips'].sum()
    totals['total_fte'] = len(df) # Count rows for FTE
    totals['total_planned_hours'] = df['planned_hours'].sum()
    totals['total_actual_hours'] = df['actual_hours'].sum()
    totals['total_absence_costs'] = calculate_absence_costs(df)
    totals['total_overtime_hours'] = calculate_overtime(df)

    # Calculate total cost of labor
    totals['total_cost'] = (
        totals['total_base_salary'] +
        totals['total_premiums'] +
        totals['total_bonuses'] +
        totals['total_social_contributions'] +
        totals['total_ltips']
    )

    # Calculate cost per FTE
    totals['fte_costs'] = totals['total_cost'] / totals['total_fte'] if totals['total_fte'] > 0 else 0

    return totals

# Calculate absence costs
def calculate_absence_costs(df):
    """Calculate absence costs based on base salary and days off."""
    # Assuming 260 working days per year (52 weeks * 5 days)
    # Calculate daily rate per employee, handling potential division by zero
    daily_rate = df.apply(lambda row: row['base_salary'] / 260 if row['base_salary'] > 0 else 0, axis=1)
    # Calculate total absence cost
    return (daily_rate * (df['sick_days'] + df['holiday_days'] + df['other_absences'])).sum()

# Calculate overtime hours
def calculate_overtime(df):
    """Calculate overtime hours as actual_hours - planned_hours where positive."""
    overtime = df['actual_hours'] - df['planned_hours']
    return overtime.apply(lambda x: max(0, x)).sum()

# Project costs for new hires
def project_costs_for_new_hires(current_totals, band_avg_df, dept, band, num_new_hires):
    """Project costs for 2025 based on current totals and new hires using band averages."""
    # Filter band averages for the selected department and band
    avg_data = band_avg_df[(band_avg_df['department'] == dept) & (band_avg_df['band'] == band)]

    if avg_data.empty or num_new_hires == 0:
        # If no data or no new hires, return current totals (maybe add hiring_costs=0)
        projections = current_totals.copy()
        projections['hiring_costs'] = 0
        return projections

    # Get the first (and should be only) row of average data
    avg = avg_data.iloc[0]

    # Start projections from current totals
    projections = current_totals.copy()

    # Add costs for new hires based on averages
    projections['total_base_salary'] += avg.get('avg_base_salary', 0) * num_new_hires
    projections['total_premiums'] += (avg.get('avg_work_conditions', 0) + avg.get('avg_overtime', 0) + avg.get('avg_other_premiums', 0)) * num_new_hires
    projections['total_bonuses'] += (avg.get('avg_annual_bonus', 0) + avg.get('avg_profit_sharing', 0)) * num_new_hires
    projections['total_social_contributions'] += (avg.get('avg_social_security', 0) + avg.get('avg_medicare', 0) + avg.get('avg_401k', 0) + avg.get('avg_pension', 0)) * num_new_hires
    projections['total_ltips'] += avg.get('avg_ltips', 0) * num_new_hires
    projections['total_fte'] += num_new_hires
    projections['total_planned_hours'] += avg.get('avg_planned_hours', 0) * num_new_hires
    # Note: Actual hours, absence costs, overtime for new hires are not projected here, could be added if needed

    # Recalculate total cost including new hires
    projections['total_cost'] = (
        projections['total_base_salary'] +
        projections['total_premiums'] +
        projections['total_bonuses'] +
        projections['total_social_contributions'] +
        projections['total_ltips']
    )

    # Calculate estimated hiring costs
    projections['hiring_costs'] = avg.get('avg_hiring_cost', 0) * num_new_hires

    # Update cost per FTE
    projections['fte_costs'] = projections['total_cost'] / projections['total_fte'] if projections['total_fte'] > 0 else 0

    return projections

# --- Data Loading ---
@st.cache_data # Cache the data loading
def load_data():
    """Load the CSV data files or create sample data if files not found."""
    try:
        # Attempt to load actual data files
        df_2023 = pd.read_csv('employee_data_2023.csv')
        df_2024 = pd.read_csv('employee_data_2024.csv')
        band_avg_df = pd.read_csv('band_averages.csv')
        st.success("Loaded data files successfully.")
        return df_2023, df_2024, band_avg_df
    except FileNotFoundError:
        st.warning("Could not find data files (employee_data_2023.csv, employee_data_2024.csv, band_averages.csv). Using sample data for demonstration.")
    except Exception as e:
        st.warning(f"An error occurred loading data files: {e}. Using sample data.")

    # --- Create Sample Data if Loading Fails ---
    sample_data_2023 = [
        [1001, "John Doe", "ERP", "BV", 85000, 2500, 3500, 1000, 8500, 2000, 5270, 1232.5, 2550, 1700, 4250, 2080, 2095, 5, 12, 2, "USD", 1.0],
        [1002, "Jane Smith", "ERA", "BIV", 72000, 1800, 2200, 800, 7200, 1440, 4464, 1044, 2160, 1440, 3600, 2080, 2065, 8, 12, 3, "USD", 1.0],
        [1003, "Michael Johnson", "ERZ", "BIII", 65000, 1500, 1800, 700, 6500, 1300, 4030, 942.5, 1950, 1300, 3250, 2080, 2090, 3, 12, 1, "USD", 1.0],
        [1004, "Lisa Brown", "8VC", "BVI", 92000, 3000, 4100, 1200, 9200, 2300, 5704, 1334, 2760, 1840, 4600, 2080, 2110, 4, 12, 0, "USD", 1.0],
        [1005, "Robert Chen", "MMS", "BV", 84000, 2400, 3200, 900, 8400, 1680, 5208, 1218, 2520, 1680, 4200, 2080, 2075, 6, 12, 4, "USD", 1.0],
        [1006, "Sarah Wilson", "WC", "BIV", 71000, 1700, 2100, 750, 7100, 1420, 4402, 1029.5, 2130, 1420, 3550, 2080, 2050, 10, 12, 2, "USD", 1.0],
        [1007, "David Lee", "BC", "BIII", 66000, 1600, 1900, 700, 6600, 1320, 4092, 957, 1980, 1320, 3300, 2080, 2080, 7, 12, 1, "USD", 1.0] # Added BC dept
    ]
    columns = ["employee_id", "name", "department", "band", "base_salary", "work_conditions_premium",
               "overtime_premium", "other_premiums", "annual_bonus", "profit_sharing",
               "social_security_tax", "medicare", "er_401k", "er_pension", "ltips",
               "planned_hours", "actual_hours", "sick_days", "holiday_days", "other_absences",
               "currency", "fx_rate"]
    df_2023 = pd.DataFrame(sample_data_2023, columns=columns)

    # Create slightly modified data for 2024 (e.g., 5% increase in costs)
    df_2024 = df_2023.copy()
    cost_cols_to_increase = ["base_salary", "work_conditions_premium", "overtime_premium", "other_premiums",
                             "annual_bonus", "profit_sharing", "social_security_tax", "medicare",
                             "er_401k", "er_pension", "ltips"]
    for col in cost_cols_to_increase:
        df_2024[col] = df_2024[col] * 1.05
    # Slightly adjust hours for variation
    df_2024['actual_hours'] = df_2024['actual_hours'] * 1.01
    df_2024['sick_days'] = df_2024['sick_days'] + 1


    # Create sample band averages
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

# Load the data
df_2023, df_2024, band_avg_df = load_data()

# Calculate overall totals for each year
totals_2023 = calculate_totals(df_2023.copy()) # Use copy to avoid modifying cached data
totals_2024 = calculate_totals(df_2024.copy()) # Use copy

# --- Sidebar Controls ---
st.sidebar.markdown("## Dashboard Controls")

# Year selection
selected_year = st.sidebar.selectbox("Select Year", ["2023", "2024", "2025 (Projected)"], key="year_select")

# Department filter for displayed data
# Ensure unique departments are sorted alphabetically
all_departments = sorted(df_2023["department"].unique().tolist())
selected_dept = st.sidebar.selectbox(
    "Filter by Department",
    ["All Departments"] + all_departments,
    key="dept_filter"
)

# --- Projection Parameters (Conditional) ---
num_new_hires = 0
selected_hire_dept = all_departments[0] if all_departments else "N/A" # Default to first dept
all_bands = sorted(df_2023["band"].unique().tolist())
selected_band = all_bands[0] if all_bands else "N/A" # Default to first band

# Initialize projected_totals based on 2024 data
projected_totals = totals_2024.copy()
projected_totals['hiring_costs'] = 0 # Initialize hiring costs

if selected_year == "2025 (Projected)":
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 2025 Projection Parameters")

    num_new_hires = st.sidebar.number_input("Number of New Hires", min_value=0, max_value=100, value=5, key="new_hires") # Default to 5

    selected_hire_dept = st.sidebar.selectbox(
        "Department for New Hires",
        all_departments,
        index=all_departments.index(selected_hire_dept) if selected_hire_dept in all_departments else 0, # Pre-select default
        key="hire_dept"
    )

    selected_band = st.sidebar.selectbox(
        "Band for New Hires",
        all_bands,
        index=all_bands.index(selected_band) if selected_band in all_bands else 0, # Pre-select default
        key="hire_band"
    )

    # Calculate overall projections based on inputs
    # This calculates the *total* projected cost including new hires
    projected_totals = project_costs_for_new_hires(
        totals_2024, # Start from overall 2024 totals
        band_avg_df,
        selected_hire_dept,
        selected_band,
        num_new_hires
    )

# --- Filter Data Based on Department Selection ---
# Apply department filter *after* calculating overall totals and projections
if selected_dept != "All Departments":
    df_2023_filtered = df_2023[df_2023["department"] == selected_dept].copy()
    df_2024_filtered = df_2024[df_2024["department"] == selected_dept].copy()
    # Calculate totals specifically for the filtered department
    dept_totals_2023 = calculate_totals(df_2023_filtered)
    dept_totals_2024 = calculate_totals(df_2024_filtered)
else:
    # If 'All Departments', use the overall data and totals
    df_2023_filtered = df_2023.copy()
    df_2024_filtered = df_2024.copy()
    dept_totals_2023 = totals_2023
    dept_totals_2024 = totals_2024

# --- Determine Current Totals for Display ---
# This dictionary holds the totals to be displayed in the main metrics and tabs,
# considering the selected year and department filter.
if selected_year == "2023":
    current_totals = dept_totals_2023
elif selected_year == "2024":
    current_totals = dept_totals_2024
else: # selected_year == "2025 (Projected)"
    # If a specific department is filtered, calculate projection *for that department only*
    if selected_dept != "All Departments":
        # Start with the 2024 totals for the selected department
        base_proj_dept = dept_totals_2024.copy()
        # Add new hires *only if* they are being hired into the currently filtered department
        if selected_hire_dept == selected_dept:
             current_totals = project_costs_for_new_hires(
                 base_proj_dept,
                 band_avg_df,
                 selected_hire_dept,
                 selected_band,
                 num_new_hires
             )
        else:
            # If hires are in a different dept, the filtered dept's totals don't change from 2024
            current_totals = base_proj_dept
            current_totals['hiring_costs'] = 0 # No hiring costs for this filtered view
    else:
        # If 'All Departments' is selected, use the overall projection calculated earlier
        current_totals = projected_totals


# --- Page Title and Description ---
st.markdown('<div class="airbus-title">AIRBUS</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Cost of Labor Dashboard</div>', unsafe_allow_html=True)
st.markdown("""
<div class="dashboard-description">
This dashboard provides a comprehensive view of labor costs for 2023, 2024, and projected costs for 2025.
It allows for simulation of new hires and their impact on the overall cost structure.
</div>
""", unsafe_allow_html=True)

# --- Main Metric Cards ---
col1, col2, col3 = st.columns(3)

# Use the 'current_totals' dictionary determined above
with col1:
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{format_currency(current_totals.get("total_cost", 0))}</div>
            <div class="metric-label-container">
                 <i class="fas fa-dollar-sign metric-icon"></i>
                 <span class="metric-label">Total Cost of Labor ({selected_year})</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{current_totals.get("total_fte", 0)}</div>
             <div class="metric-label-container">
                 <i class="fas fa-users metric-icon"></i>
                 <span class="metric-label">Total FTE ({selected_year})</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{format_currency(current_totals.get("fte_costs", 0))}</div>
             <div class="metric-label-container">
                 <i class="fas fa-money-check-alt metric-icon"></i>
                 <span class="metric-label">Cost per FTE ({selected_year})</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)


# --- Tabs Definition (Conditional) ---
# Define tab names
tab_names = ["📊 Cost Breakdown", "📈 Year-over-Year", "🔍 Detailed Analysis"]
# Add the projection tab ONLY if 2025 is selected
if selected_year == "2025 (Projected)":
    tab_names.append("🔮 2025 Projection")

# Create the tabs
tabs = st.tabs(tab_names)

# Assign tabs to variables dynamically
tab1 = tabs[0]
tab2 = tabs[1]
tab3 = tabs[2]
if selected_year == "2025 (Projected)":
    tab4 = tabs[3] # Assign the fourth tab if it exists


# --- Tab 1: Cost Breakdown ---
with tab1:
    st.markdown('<div class="sub-header">Cost Breakdown Analysis</div>', unsafe_allow_html=True)

    # Data Preparation for the Chart and Metrics
    cost_data = {
        'Category': ['Base Salary', 'Premiums', 'Bonuses', 'Social Contributions', 'LTIPs'],
        'Amount': [
            current_totals.get('total_base_salary', 0),
            current_totals.get('total_premiums', 0),
            current_totals.get('total_bonuses', 0),
            current_totals.get('total_social_contributions', 0),
            current_totals.get('total_ltips', 0)
        ]
    }
    cost_df = pd.DataFrame(cost_data)
    total_cost_for_breakdown = cost_df['Amount'].sum()
    if total_cost_for_breakdown > 0:
        cost_df['Percentage'] = (cost_df['Amount'] / total_cost_for_breakdown * 100).astype(float)
    else:
        cost_df['Percentage'] = 0.0
    cost_df['Amount'] = pd.to_numeric(cost_df['Amount'])
    cost_df = cost_df.sort_values(by='Amount', ascending=False).reset_index(drop=True)

    # Donut Chart
    st.markdown(f"#### Cost Distribution for {selected_year}")
    fig_donut = go.Figure(data=[go.Pie(
        labels=cost_df['Category'],
        values=cost_df['Amount'],
        hole=.4,
        pull=[0.05 if i == 0 else 0 for i in cost_df.index],
        marker_colors=px.colors.sequential.Blues_r,
        textinfo='label+percent',
        insidetextorientation='auto',
        hovertemplate="<b>%{label}</b><br>Amount: %{value:$,.2f}<br>Percentage: %{percent:.1%}<extra></extra>"
    )])
    fig_donut.add_annotation(
        text=f"Total:<br>{format_currency(total_cost_for_breakdown)}",
        x=0.5, y=0.5, font_size=18, showarrow=False, font_color="#1E3A8A"
    )
    fig_donut.update_layout(
        legend_title_text='Categories', showlegend=True,
        margin=dict(t=20, b=20, l=20, r=20), height=500
    )
    st.plotly_chart(fig_donut, use_container_width=True)

    # Metric Displays for Breakdown Details
    st.markdown("#### Breakdown Details")
    st.markdown("---")
    num_metrics = len(cost_df)
    num_cols = min(num_metrics, 3)
    cols = st.columns(num_cols)
    col_index = 0
    for index, row in cost_df.iterrows():
        with cols[col_index]:
            st.metric(label=row['Category'], value=format_currency(row['Amount']))
            if pd.notna(row['Percentage']):
                 st.markdown(f"({row['Percentage']:.1f}% of Total)")
            else:
                 st.markdown("(N/A)")
            st.markdown("<br>", unsafe_allow_html=True)
        col_index = (col_index + 1) % num_cols

    # REMOVED: Redundant 2025 New Hire Impact section from here. It's now in Tab 4.


# --- Tab 2: Year-over-Year Comparison ---
with tab2:
    st.markdown('<div class="sub-header">Year-over-Year Comparison</div>', unsafe_allow_html=True)

    # Helper Functions for Tab 2 Styling
    def safe_division(numerator, denominator):
        if denominator == 0: return 0.0 if numerator == 0 else float('inf')
        return (numerator - denominator) / denominator * 100

    def color_growth(val):
        if pd.isna(val) or val == float('inf'): color = 'grey'
        elif val > 0: color = 'green'
        elif val < 0: color = 'red'
        else: color = 'grey'
        return f'color: {color}'

    def format_growth_display(val):
        if pd.isna(val): return "N/A"
        if val == float('inf'): return "N/A (from zero)"
        elif val > 0: return f"▲ {val:.1f}%"
        elif val < 0: return f"▼ {abs(val):.1f}%"
        else: return f"▬ {val:.1f}%"

    # Data Preparation for Comparison
    # Use department-specific totals if filtered, otherwise overall totals
    if selected_dept != "All Departments":
        compare_2023 = dept_totals_2023
        compare_2024 = dept_totals_2024
    else:
        compare_2023 = totals_2023
        compare_2024 = totals_2024

    # Calculate 2023-2024 growth rates
    growth_keys = ['total_base_salary', 'total_premiums', 'total_bonuses', 'total_social_contributions', 'total_ltips', 'total_cost']
    growth_rates = {}
    for key in growth_keys:
        rate = safe_division(compare_2024.get(key, 0), compare_2023.get(key, 0))
        growth_rates[key] = round(rate, 1) if rate != float('inf') else float('inf')

    categories = ['Base Salary', 'Premiums', 'Bonuses', 'Social Contributions', 'LTIPs', 'Total Cost']
    year_2023_values = [compare_2023.get(k, 0) for k in growth_keys]
    year_2024_values = [compare_2024.get(k, 0) for k in growth_keys]
    growth_rates_list = [growth_rates.get(k, 0) for k in growth_keys] # For chart annotations

    # Chart Creation with Outlines and Text Annotations
    fig_yoy = go.Figure()
    fig_yoy.add_trace(go.Bar(x=categories, y=year_2023_values, name='2023', marker_color='#93C5FD'))

    bar_outline_colors, bar_outline_widths, annotation_texts, annotation_colors = [], [], [], []
    for i, growth in enumerate(growth_rates_list):
        if growth > 0 and growth != float('inf'):
            bar_outline_colors.append('green'); bar_outline_widths.append(2)
            annotation_texts.append(f"+{growth:.1f}%"); annotation_colors.append('green')
        elif growth < 0:
            bar_outline_colors.append('red'); bar_outline_widths.append(2)
            annotation_texts.append(f"{growth:.1f}%"); annotation_colors.append('red')
        else:
            bar_outline_colors.append('#2563EB'); bar_outline_widths.append(0)
            annotation_texts.append(" "); annotation_colors.append('grey')

    fig_yoy.add_trace(go.Bar(x=categories, y=year_2024_values, name='2024', marker_color='#2563EB',
                           marker_line_color=bar_outline_colors, marker_line_width=bar_outline_widths))

    # Add 2025 projected bars if that year is selected
    if selected_year == "2025 (Projected)":
        # Use 'current_totals' as it holds the correctly filtered/calculated 2025 data
        year_2025_values_chart = [current_totals.get(k, 0) for k in growth_keys]
        fig_yoy.add_trace(go.Bar(x=categories, y=year_2025_values_chart, name='2025 (Projected)', marker_color='#1E3A8A'))

    # Add annotations
    for i, category in enumerate(categories):
        if annotation_texts[i].strip():
            fig_yoy.add_annotation(x=category, y=year_2024_values[i], text=annotation_texts[i],
                                 showarrow=False, font=dict(color=annotation_colors[i], size=11), yshift=15)

    fig_yoy.update_layout(title="Cost Comparison (Outline/Text shows 2023-2024 change)",
                        xaxis_title="Cost Category", yaxis_title="Amount ($)", legend_title="Year",
                        barmode='group', height=550, margin=dict(t=90, b=40, l=40, r=40))
    fig_yoy.update_yaxes(tickprefix="$", tickformat=",")
    st.plotly_chart(fig_yoy, use_container_width=True)

    # Display Growth Rates Table using st.dataframe with Styling
    growth_data_raw = {'Category': categories, '2023 Amount': year_2023_values, '2024 Amount': year_2024_values, '2023-2024 Growth': growth_rates_list}
    if selected_year == "2025 (Projected)":
        year_2025_values_table = [current_totals.get(k, 0) for k in growth_keys]
        growth_2025 = []
        for k in growth_keys:
            rate_2025 = safe_division(current_totals.get(k, 0), compare_2024.get(k, 0))
            growth_2025.append(round(rate_2025, 1) if rate_2025 != float('inf') else float('inf'))
        growth_data_raw['2025 Amount (Projected)'] = year_2025_values_table
        growth_data_raw['2024-2025 Growth (Projected)'] = growth_2025

    growth_df_raw = pd.DataFrame(growth_data_raw)
    amount_cols = [col for col in growth_df_raw.columns if 'Amount' in col]
    growth_cols = [col for col in growth_df_raw.columns if 'Growth' in col]
    styled_df = growth_df_raw.style \
        .format(format_currency, subset=amount_cols, na_rep='$NaN') \
        .format(format_growth_display, subset=growth_cols, na_rep='N/A') \
        .apply(lambda x: x.map(color_growth), subset=growth_cols)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


# --- Tab 3: Detailed Analysis ---
with tab3:
    st.markdown('<div class="sub-header">Detailed Analysis</div>', unsafe_allow_html=True)

    # Data selection based on year remains the same
    if selected_year == "2023":
        analysis_df = df_2023_filtered
        analysis_totals = dept_totals_2023
    elif selected_year == "2024":
        analysis_df = df_2024_filtered
        analysis_totals = dept_totals_2024
    else:  # 2025 projection
        analysis_df = df_2024_filtered # Base analysis on 2024 data for projection year
        analysis_totals = current_totals # Use the calculated totals for the selected view

    subtab1, subtab2 = st.tabs(["Department Analysis", "Band Distribution"])

    # Subtab 1: Department Analysis
    with subtab1:
        if selected_dept == "All Departments":
            st.markdown("##### Employee Distribution by Department")
            dept_analysis = analysis_df.groupby('department').agg(
                total_base_salary = pd.NamedAgg(column='base_salary', aggfunc='sum'),
                employee_count = pd.NamedAgg(column='employee_id', aggfunc='count')
            ).reset_index()
            dept_analysis.rename(columns={'department': 'Department', 'total_base_salary': 'Total Base Salary', 'employee_count': 'Employee Count'}, inplace=True)

            fig_dept_dist = px.bar(dept_analysis, x='Department', y='Employee Count', color='Department',
                                 color_discrete_sequence=px.colors.qualitative.Plotly,
                                 hover_data={'Department': True, 'Employee Count': True})
            fig_dept_dist.update_layout(xaxis_title="Department", yaxis_title="Employee Count", legend_title="Department", height=450)
            fig_dept_dist.update_yaxes(rangemode='tozero')
            st.plotly_chart(fig_dept_dist, use_container_width=True)

            dept_analysis['Average Salary'] = dept_analysis['Total Base Salary'] / dept_analysis['Employee Count']
            try:
                dept_analysis['Total Base Salary Formatted'] = dept_analysis['Total Base Salary'].apply(format_currency)
                dept_analysis['Average Salary Formatted'] = dept_analysis['Average Salary'].apply(format_currency)
            except NameError:
                 dept_analysis['Total Base Salary Formatted'] = dept_analysis['Total Base Salary'].apply(lambda x: f"${x:,.0f}")
                 dept_analysis['Average Salary Formatted'] = dept_analysis['Average Salary'].apply(lambda x: f"${x:,.0f}")
            st.markdown("##### Department Salary Breakdown")
            st.dataframe(dept_analysis[['Department', 'Employee Count', 'Total Base Salary Formatted', 'Average Salary Formatted']],
                         hide_index=True, use_container_width=True)
        else:
            st.markdown(f"##### {selected_dept} Department Analysis")
            band_count = analysis_df['band'].value_counts().reset_index()
            band_count.columns = ['Band', 'Count']
            fig_band_pie = px.pie(band_count, values='Count', names='Band', title=f'Band Distribution in {selected_dept}',
                                color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_band_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_band_pie.update_layout(height=400)
            st.plotly_chart(fig_band_pie, use_container_width=True)
            st.metric(label="Total Employees", value=analysis_df.shape[0])
            try:
                avg_sal = analysis_df['base_salary'].mean()
                st.metric(label="Average Base Salary", value=format_currency(avg_sal))
            except NameError:
                avg_sal = analysis_df['base_salary'].mean()
                st.metric(label="Average Base Salary", value=f"${avg_sal:,.0f}")
            except Exception: st.info("Could not calculate average salary.")

    # Subtab 2: Band Distribution
    with subtab2:
        st.markdown("##### Employee Distribution by Band")
        band_analysis = analysis_df.groupby('band').agg(
             total_base_salary = pd.NamedAgg(column='base_salary', aggfunc='sum'),
             avg_base_salary = pd.NamedAgg(column='base_salary', aggfunc='mean'),
             employee_count = pd.NamedAgg(column='employee_id', aggfunc='count')
        ).reset_index().sort_values(by='employee_count', ascending=False)
        band_analysis.rename(columns={'band': 'Band', 'avg_base_salary': 'Average Salary', 'employee_count': 'Employee Count'}, inplace=True)

        fig_band_bar = px.bar(band_analysis, x='Band', y='Employee Count', color='Band',
                             color_discrete_sequence=px.colors.sequential.Blues_r,
                             hover_data={'Band': True, 'Employee Count': True, 'Average Salary': ':.2f'})
        fig_band_bar.update_layout(xaxis_title="Band", yaxis_title="Employee Count", legend_title="Band", height=450)
        fig_band_bar.update_yaxes(rangemode='tozero')
        st.plotly_chart(fig_band_bar, use_container_width=True)

        st.markdown("##### Band Salary Breakdown")
        try:
            band_analysis['Total Base Salary Formatted'] = band_analysis['total_base_salary'].apply(format_currency)
            band_analysis['Average Salary Formatted'] = band_analysis['Average Salary'].apply(format_currency)
        except NameError:
            band_analysis['Total Base Salary Formatted'] = band_analysis['total_base_salary'].apply(lambda x: f"${x:,.0f}")
            band_analysis['Average Salary Formatted'] = band_analysis['Average Salary'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(band_analysis[['Band', 'Employee Count', 'Total Base Salary Formatted', 'Average Salary Formatted']],
                     hide_index=True, use_container_width=True)

# --- Tab 4: 2025 Projection Details (Conditional) ---
# This block only runs if the fourth tab ('tab4') was created
if selected_year == "2025 (Projected)":
    with tab4:
        st.markdown('<div class="sub-header">2025 Projection Details</div>', unsafe_allow_html=True)

        # Check if new hires were added for projection
        if num_new_hires > 0:
            # --- New Hire Cost Breakdown ---
            st.markdown("##### New Hire Cost Breakdown")
            # Get averages for the selected department and band for new hires
            hire_avg_data = band_avg_df[(band_avg_df['department'] == selected_hire_dept) &
                                        (band_avg_df['band'] == selected_band)]

            if not hire_avg_data.empty:
                hire_avg = hire_avg_data.iloc[0]
                # Create breakdown of costs per new hire
                new_hire_cost_cats = ['Base Salary', 'Premiums', 'Bonuses', 'Social Contributions', 'LTIPs', 'Hiring Cost']
                new_hire_cost_values = [
                    hire_avg.get('avg_base_salary', 0),
                    hire_avg.get('avg_work_conditions', 0) + hire_avg.get('avg_overtime', 0) + hire_avg.get('avg_other_premiums', 0),
                    hire_avg.get('avg_annual_bonus', 0) + hire_avg.get('avg_profit_sharing', 0),
                    hire_avg.get('avg_social_security', 0) + hire_avg.get('avg_medicare', 0) + hire_avg.get('avg_401k', 0) + hire_avg.get('avg_pension', 0),
                    hire_avg.get('avg_ltips', 0),
                    hire_avg.get('avg_hiring_cost', 0)
                ]
                # Calculate total cost per hire (excluding hiring cost itself for this subtotal)
                total_comp_per_hire = sum(new_hire_cost_values[:-1])
                # Add total row
                new_hire_cost_cats.append('Total Comp per Hire')
                new_hire_cost_values.append(total_comp_per_hire)

                new_hire_df = pd.DataFrame({'Category': new_hire_cost_cats, 'Avg Amount per Hire': new_hire_cost_values})
                # Format the dataframe
                styled_nh_df = new_hire_df.style.format({'Avg Amount per Hire': format_currency})
                st.dataframe(styled_nh_df, hide_index=True, use_container_width=True)

                # Display total additional cost for all new hires
                total_add_cost = total_comp_per_hire + hire_avg.get('avg_hiring_cost', 0) # Add hiring cost back for total impact
                st.markdown(f"**Total Estimated Cost per New Hire (incl. Hiring):** {format_currency(total_add_cost)}")
                st.markdown(f"**Total Additional Cost for {num_new_hires} New Hires:** {format_currency(total_add_cost * num_new_hires)}")

            else:
                st.warning(f"No average band data found for {selected_hire_dept} / {selected_band} to break down new hire costs.")

            st.markdown("---")

            # --- Cost Growth Analysis (2024 vs 2025 Projected) ---
            st.markdown("##### Cost Growth Analysis (2024 vs 2025 Projection)")
            # Use overall totals_2024 and overall projected_totals for this comparison
            growth_cats_proj = ['Base Salary', 'Premiums', 'Bonuses', 'Social Contributions', 'LTIPs', 'Total Cost']
            growth_keys_proj = ['total_base_salary', 'total_premiums', 'total_bonuses', 'total_social_contributions', 'total_ltips', 'total_cost']

            growth_2024_vals = [totals_2024.get(k, 0) for k in growth_keys_proj]
            growth_2025_vals = [projected_totals.get(k, 0) for k in growth_keys_proj] # Use overall projection

            growth_rates_proj = []
            for k in growth_keys_proj:
                 rate = safe_division(projected_totals.get(k, 0), totals_2024.get(k, 0))
                 growth_rates_proj.append(round(rate, 2) if rate != float('inf') else float('inf')) # Use 2 decimals for chart

            growth_proj_df = pd.DataFrame({'Category': growth_cats_proj, 'Growth Rate (%)': growth_rates_proj})

            fig_growth = px.bar(growth_proj_df, x='Category', y='Growth Rate (%)',
                                title='Projected 2024-2025 Growth Rates (Overall)',
                                color='Growth Rate (%)', color_continuous_scale='Blues',
                                text='Growth Rate (%)')
            fig_growth.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_growth.update_layout(coloraxis_showscale=False, height=450) # Hide color scale bar
            st.plotly_chart(fig_growth, use_container_width=True)

            st.markdown("---")

            # --- What-If Scenario Analysis ---
            st.markdown("##### What-If Scenario Analysis (Varying New Hires)")
            # Create a range of potential new hire counts around the selected number
            min_h = max(0, num_new_hires - 10)
            max_h = num_new_hires + 11 # Go up to 10 above
            step = max(1, (max_h - min_h) // 5) # Aim for ~6 scenarios
            scenario_hires = list(range(min_h, max_h, step))
            if num_new_hires not in scenario_hires:
                scenario_hires.append(num_new_hires)
            scenario_hires = sorted(list(set(scenario_hires))) # Ensure unique and sorted

            scenario_data = []
            base_cost_2024 = totals_2024.get('total_cost', 0) # Overall 2024 cost

            for hire_count in scenario_hires:
                # Recalculate projection for this specific hire count
                scenario_totals = project_costs_for_new_hires(
                    totals_2024, band_avg_df, selected_hire_dept, selected_band, hire_count
                )
                scenario_cost = scenario_totals.get('total_cost', 0)
                cost_increase = scenario_cost - base_cost_2024
                percent_increase = safe_division(scenario_cost, base_cost_2024)

                scenario_data.append({
                    'New Hires': hire_count,
                    'Projected Total Cost': scenario_cost,
                    'Cost Increase from 2024': cost_increase,
                    'Percent Increase from 2024': round(percent_increase, 2) if percent_increase != float('inf') else float('inf')
                })

            scenario_df = pd.DataFrame(scenario_data)

            # Create scenario line chart
            fig_scenario = go.Figure()
            # Line for Total Cost (Primary Y-axis)
            fig_scenario.add_trace(go.Scatter(
                x=scenario_df['New Hires'], y=scenario_df['Projected Total Cost'],
                mode='lines+markers', name='Projected Total Cost', yaxis='y1'
            ))
            # Line for Percent Increase (Secondary Y-axis)
            fig_scenario.add_trace(go.Scatter(
                x=scenario_df['New Hires'], y=scenario_df['Percent Increase from 2024'],
                mode='lines+markers', name='% Increase from 2024', yaxis='y2',
                line=dict(color='red', dash='dot')
            ))

            fig_scenario.update_layout(
                title=f'Cost Projection Scenarios for {selected_hire_dept} ({selected_band}) Hires',
                xaxis_title='Number of New Hires',
                yaxis=dict(title='Projected Total Cost ($)', side='left', tickformat="$,.0f"),
                yaxis2=dict(title='% Increase from 2024', side='right', overlaying='y', tickformat=".1f", suffix="%"),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=500
            )
            st.plotly_chart(fig_scenario, use_container_width=True)

            # Display scenario table
            styled_scenario_df = scenario_df.style.format({
                'Projected Total Cost': format_currency,
                'Cost Increase from 2024': format_currency,
                'Percent Increase from 2024': "{:.2f}%"
            })
            st.dataframe(styled_scenario_df, hide_index=True, use_container_width=True)

        else:
            st.info("Select a non-zero number of new hires in the sidebar to see projection details.")


# --- Footer ---
st.markdown("---")
st.markdown("**Cost of Labor Dashboard** | Developed for HR Analytics | Last updated: April 2025") # Update date if needed

# --- Download Links in Sidebar ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Download Sample Data Templates")

# Create download buttons for sample CSVs (using placeholder data for demonstration)
# Sample DataFrames (assuming df_2023 and band_avg_df hold the sample data)
try:
    employee_csv = df_2023.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Employee Data Template",
        data=employee_csv,
        file_name="employee_data_template.csv",
        mime="text/csv",
        key="download-employee"
    )

    band_csv = band_avg_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Band Averages Template",
        data=band_csv,
        file_name="band_averages_template.csv",
        mime="text/csv",
        key="download-band"
    )
except Exception as e:
    st.sidebar.error(f"Could not prepare download files: {e}")

