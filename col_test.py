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

# Apply custom CSS for better styling, including updated titles
st.markdown("""
<style>
    # .airbus-title {
    #     font-family: 'Poppins', sans-serif; /* MODIFIED: Use Poppins font */
    #     font-size: 1.8rem; /* Adjust size as needed */
    #     font-weight: 900; /* Extra bold */
    #     color: #00205B; /* Airbus dark blue */
    #     /* text-align: center; */ /* MODIFIED: Removed centering */
    #     text-align: left; /* Explicitly left-align */
    #     margin-bottom: 0.1rem; /* Small space below */
    #     letter-spacing: 1px; /* Optional: slight letter spacing */
    #     padding-left: 1rem; /* Add some padding to align with content */
    # }
    .airbus-title {
        /* Primary font choice with fallbacks */
        font-family: 'Frutiger', 'Univers', 'DIN Condensed', sans-serif;
        font-size: 3.3rem; /* Increased size for better visibility */
        font-weight: 700; /* Bold weight */
        color: #00205B; /* Airbus dark blue */
        text-align: left;
        margin-bottom: 0.1rem;
        letter-spacing: 0.5px; /* Slight letter spacing for distinction */
        padding-left: 1rem;
        text-transform: uppercase; /* Ensure uppercase display */
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
    df_2023 = pd.read_csv('employee_data_2023.csv')
    df_2024 = pd.read_csv('employee_data_2024.csv')
    band_avg_df = pd.read_csv('band_averages.csv')

    return df_2023, df_2024, band_avg_df

# Load data
df_2023, df_2024, band_avg_df = load_data()

# Calculate totals for 2023 and 2024
totals_2023 = calculate_totals(df_2023)
totals_2024 = calculate_totals(df_2024)

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
                 <i class="fas fa-dollar-sign metric-icon"></i>
                 <span class="metric-label">Total Cost of Labor ({selected_year})</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{current_totals["total_fte"]}</div>
             <div class="metric-label-container">
                 <i class="fas fa-users metric-icon"></i>
                 <span class="metric-label">Total FTE ({selected_year})</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{format_currency(current_totals["fte_costs"])}</div>
             <div class="metric-label-container">
                 <i class="fas fa-money-check-alt metric-icon"></i>
                 <span class="metric-label">Cost per FTE ({selected_year})</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Cost Breakdown", "📈 Year-over-Year", "🔍 Detailed Analysis"])


# Tab 1: Cost Breakdown (Good version)
#     # --- Create Styled Dataframe with Data Bars (Displayed below chart) ---
#     st.markdown("#### Breakdown Details") # Add a title above the table

#     # Apply styling: format currency, format percentage, add data bars
#     styled_cost_df = cost_df.style \
#         .format({
#             'Amount': format_currency, # Use your existing helper
#             'Percentage': "{:.1f}%"   # Format percentage with one decimal
#         }) \
#         .bar(subset=['Percentage'], color='#93C5FD', vmin=0, vmax=100, align='left') # Add data bars aligned left

#     # Display the styled dataframe
#     st.dataframe(styled_cost_df, use_container_width=True, hide_index=True)

#     # --- Conditional Section for 2025 New Hire Impact ---
#     # (Placed at the end of the tab)
#     if selected_year == "2025 (Projected)" and num_new_hires > 0:
#         # Add a separator
#         st.markdown("---")
#         st.markdown('<div class="highlight">', unsafe_allow_html=True) # Use existing highlight style
#         st.markdown(f"#### New Hire Impact Analysis ({num_new_hires} Hires)") # More descriptive header

#         # Display details about the new hires
#         st.markdown(f"**Department:** {selected_hire_dept}, **Band:** {selected_band}")

#         # Display calculated hiring costs if available in projected_totals
#         if 'hiring_costs' in projected_totals and projected_totals['hiring_costs'] > 0:
#             st.metric(label="Estimated Total Hiring Costs", value=format_currency(projected_totals['hiring_costs']))
#             # Calculate and display cost per new hire
#             cost_per_hire = projected_totals['hiring_costs'] / num_new_hires
#             st.metric(label="Estimated Cost per Hire", value=format_currency(cost_per_hire))
#         else:
#             st.info("Hiring cost data not available or calculated as zero.")

#         st.markdown('</div>', unsafe_allow_html=True)

# Tab 1: Cost Breakdown
with tab1:
    # Add the sub-header for this tab
    st.markdown('<div class="sub-header">Cost Breakdown Analysis</div>', unsafe_allow_html=True)

    # --- Data Preparation for the Chart and Metrics ---
    # Ensure 'current_totals' dictionary is calculated based on selected year/filters before this tab
    # Example structure of current_totals assumed:
    # current_totals = {
    #     'total_base_salary': ..., 'total_premiums': ..., 'total_bonuses': ...,
    #     'total_social_contributions': ..., 'total_ltips': ..., 'total_cost': ...
    # }

    # Create DataFrame for easier processing
    cost_data = {
        'Category': ['Base Salary', 'Premiums', 'Bonuses', 'Social Contributions', 'LTIPs'],
        'Amount': [
            current_totals.get('total_base_salary', 0), # Use .get for safety
            current_totals.get('total_premiums', 0),
            current_totals.get('total_bonuses', 0),
            current_totals.get('total_social_contributions', 0),
            current_totals.get('total_ltips', 0)
        ]
    }
    cost_df = pd.DataFrame(cost_data)

    # Calculate total cost for percentages and center annotation
    # Use the sum from the DataFrame to ensure consistency
    total_cost_for_breakdown = cost_df['Amount'].sum()

    # Calculate percentages (handle division by zero if total is zero)
    if total_cost_for_breakdown > 0:
        # Ensure Percentage is float
        cost_df['Percentage'] = (cost_df['Amount'] / total_cost_for_breakdown * 100).astype(float)
    else:
        cost_df['Percentage'] = 0.0

    # Ensure Amount is numeric
    cost_df['Amount'] = pd.to_numeric(cost_df['Amount'])

    # Sort by Amount descending for better visual hierarchy in metrics display
    cost_df = cost_df.sort_values(by='Amount', ascending=False).reset_index(drop=True)

    # --- Create Donut Chart (Displayed first) ---
    st.markdown(f"#### Cost Distribution for {selected_year}") # Add a title above the chart

    fig = go.Figure(data=[go.Pie(
        labels=cost_df['Category'],
        values=cost_df['Amount'],
        hole=.4,  # Creates the donut hole
        # Slightly pull out the largest slice (first row after sorting)
        pull=[0.05 if i == 0 else 0 for i in cost_df.index],
        marker_colors=px.colors.sequential.Blues_r, # Use a sequential color scheme
        textinfo='label+percent', # Show label and percentage on slices
        insidetextorientation='auto', # Let Plotly decide best text orientation
        # Define hover text format
        hovertemplate="<b>%{label}</b><br>Amount: %{value:$,.2f}<br>Percentage: %{percent:.1%}<extra></extra>"
    )])

    # Add center annotation for Total Cost
    # Make sure your format_currency function is defined elsewhere or replace with f-string formatting
    try:
        formatted_total = format_currency(total_cost_for_breakdown)
    except NameError:
        # Fallback if format_currency isn't defined in this scope
        formatted_total = f"${total_cost_for_breakdown:,.2f}"

    fig.add_annotation(
        text=f"Total:<br>{formatted_total}",
        x=0.5, y=0.5, # Center position
        font_size=18,
        showarrow=False,
        font_color="#1E3A8A" # Match header color from CSS if possible
    )

    # Update chart layout
    fig.update_layout(
        legend_title_text='Categories', # Add title to legend
        showlegend=True, # Ensure legend is shown
        margin=dict(t=20, b=20, l=20, r=20), # Adjust margins
        height=500 # Adjust height as needed for chart + legend
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # --- Create METRIC DISPLAYS for Breakdown Details (Displayed below chart) ---
    st.markdown("#### Breakdown Details") # Add a title above the metrics
    st.markdown("---") # Add a visual separator

    # Define columns for layout (e.g., 3 columns)
    # Adjust the number of columns based on how many items you have and desired layout
    num_metrics = len(cost_df)
    # Use min() to avoid error if fewer than 3 categories exist
    num_cols = min(num_metrics, 3)
    cols = st.columns(num_cols) # Create the columns

    # Iterate through the sorted cost data and display metrics
    col_index = 0
    for index, row in cost_df.iterrows():
        # Select the current column using the index
        with cols[col_index]:
            # Format the amount value using the helper function or f-string
            try:
                metric_value = format_currency(row['Amount'])
            except NameError:
                metric_value = f"${row['Amount']:,.2f}"

            # Display the metric: Category as label, Amount as value
            st.metric(
                label=row['Category'],
                value=metric_value
            )
            # Display the percentage contribution below the metric
            # Check if percentage is valid before formatting
            if pd.notna(row['Percentage']):
                 st.markdown(f"({row['Percentage']:.1f}% of Total)")
            else:
                 st.markdown("(N/A)") # Handle potential NaN percentages

            # Add some vertical spacing below each metric block for readability
            st.markdown("<br>", unsafe_allow_html=True)

        # Move to the next column index, wrapping around using modulo
        col_index = (col_index + 1) % num_cols


    # --- Conditional Section for 2025 New Hire Impact ---
    # Display this section only if 2025 is selected and new hires are specified
    # Ensure num_new_hires, selected_hire_dept, selected_band, projected_totals are defined earlier
    if selected_year == "2025 (Projected)" and 'num_new_hires' in globals() and num_new_hires > 0:
        st.markdown("---") # Separator before this section
        # Use the custom highlight style defined in your CSS
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown(f"#### New Hire Impact Analysis ({num_new_hires} Hires)")
        st.markdown(f"**Department:** {selected_hire_dept}, **Band:** {selected_band}")

        # Check if hiring cost data is available in the projected totals
        if 'projected_totals' in globals() and 'hiring_costs' in projected_totals and projected_totals['hiring_costs'] > 0:
            # Format hiring cost values
            try:
                hiring_cost_formatted = format_currency(projected_totals['hiring_costs'])
            except NameError:
                hiring_cost_formatted = f"${projected_totals['hiring_costs']:,.2f}"

            st.metric(label="Estimated Total Hiring Costs", value=hiring_cost_formatted)

            # Calculate and display cost per new hire
            cost_per_hire = projected_totals['hiring_costs'] / num_new_hires
            try:
                cost_per_hire_formatted = format_currency(cost_per_hire)
            except NameError:
                 cost_per_hire_formatted = f"${cost_per_hire:,.2f}"

            st.metric(label="Estimated Cost per Hire", value=cost_per_hire_formatted)
        else:
            # Display message if hiring cost data is missing or zero
            st.info("Hiring cost data not available or calculated as zero.")
        # Close the highlight div
        st.markdown('</div>', unsafe_allow_html=True)

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

# Tab 2: Year-over-Year Comparison
with tab2:
    # Add the sub-header for this tab
    st.markdown('<div class="sub-header">Year-over-Year Comparison</div>', unsafe_allow_html=True)

    # --- Helper Functions Defined within Tab 2 Scope ---

    # Helper function for safe division (to handle potential zero denominators)
    def safe_division(numerator, denominator):
        """Performs division, returning 'inf' or 0.0 if denominator is zero."""
        # Check if denominator is zero
        if denominator == 0:
            # If numerator is also 0, result is 0. Otherwise, infinite growth.
            return 0.0 if numerator == 0 else float('inf')
        # Calculate percentage growth: (new - old) / old * 100
        return (numerator - denominator) / denominator * 100

    # Helper function to determine the text color for styling the dataframe
    def color_growth(val):
        """Returns CSS style string for text color based on value."""
        # Check for NaN or infinite values
        if pd.isna(val) or val == float('inf'):
            color = 'grey' # Neutral color for N/A or infinite growth
        elif val > 0:
            color = 'green' # Green for positive growth
        elif val < 0:
            color = 'red'   # Red for negative growth
        else: # val == 0
            color = 'grey'  # Neutral color for zero growth (or 'black')
        # Return the CSS style string
        return f'color: {color}'

    # Helper function to format the growth number into the desired display string for the dataframe
    def format_growth_display(val):
        """Formats number into string with arrow and percentage."""
        # Handle NaN values
        if pd.isna(val):
            return "N/A"
        # Handle infinite growth (from zero base)
        if val == float('inf'):
            return "N/A (from zero)"
        # Format positive growth
        elif val > 0:
            return f"▲ {val:.1f}%" # Up arrow, value, % sign
        # Format negative growth
        elif val < 0:
            # Down arrow, absolute value (to avoid double negative), % sign
            return f"▼ {abs(val):.1f}%"
        # Format zero growth
        else: # val == 0
            return f"▬ {val:.1f}%" # Neutral symbol, value, % sign

    # --- Data Preparation ---

    # Select the correct data source based on the department filter in the sidebar
    if selected_dept != "All Departments":
        # Use totals calculated for the specific selected department
        compare_2023 = dept_totals_2023
        compare_2024 = dept_totals_2024
    else:
        # Use the overall totals calculated from the full dataset
        compare_2023 = totals_2023
        compare_2024 = totals_2024

    # Calculate 2023-2024 growth rates for each category using the safe_division helper
    growth_base = safe_division(compare_2024['total_base_salary'], compare_2023['total_base_salary'])
    growth_premiums = safe_division(compare_2024['total_premiums'], compare_2023['total_premiums'])
    growth_bonuses = safe_division(compare_2024['total_bonuses'], compare_2023['total_bonuses'])
    growth_social = safe_division(compare_2024['total_social_contributions'], compare_2023['total_social_contributions'])
    growth_ltips = safe_division(compare_2024['total_ltips'], compare_2023['total_ltips'])
    growth_total = safe_division(compare_2024['total_cost'], compare_2023['total_cost'])

    # Round valid growth rates to one decimal place (infinite values remain infinite)
    growth_base = round(growth_base, 1) if growth_base != float('inf') else float('inf')
    growth_premiums = round(growth_premiums, 1) if growth_premiums != float('inf') else float('inf')
    growth_bonuses = round(growth_bonuses, 1) if growth_bonuses != float('inf') else float('inf')
    growth_social = round(growth_social, 1) if growth_social != float('inf') else float('inf')
    growth_ltips = round(growth_ltips, 1) if growth_ltips != float('inf') else float('inf')
    growth_total = round(growth_total, 1) if growth_total != float('inf') else float('inf')

    # Prepare lists of categories and values for the chart and table
    categories = ['Base Salary', 'Premiums', 'Bonuses', 'Social Contributions', 'LTIPs', 'Total Cost']
    year_2023_values = [
        compare_2023['total_base_salary'], compare_2023['total_premiums'], compare_2023['total_bonuses'],
        compare_2023['total_social_contributions'], compare_2023['total_ltips'], compare_2023['total_cost']
    ]
    year_2024_values = [
        compare_2024['total_base_salary'], compare_2024['total_premiums'], compare_2024['total_bonuses'],
        compare_2024['total_social_contributions'], compare_2024['total_ltips'], compare_2024['total_cost']
    ]
    # List of calculated (and rounded) growth rates
    growth_rates = [
        growth_base, growth_premiums, growth_bonuses,
        growth_social, growth_ltips, growth_total
    ]

    # --- Chart Creation with Outlines and Text Annotations ---
    fig = go.Figure()

    # Add 2023 bars
    fig.add_trace(go.Bar(
        x=categories,
        y=year_2023_values,
        name='2023',
        marker_color='#93C5FD' # Light Blue
    ))

    # Prepare lists for conditional styling of 2024 bars and annotations
    bar_outline_colors = [] # List to hold outline color for each 2024 bar
    bar_outline_widths = [] # List to hold outline width for each 2024 bar
    annotation_texts = []   # List to hold the text for annotations (e.g., "+2.3%")
    annotation_colors = []  # List to hold the color for annotation text

    # Loop through the 2023-2024 growth rates to determine styling for each category
    for growth in growth_rates:
        if growth > 0 and growth != float('inf'):
            # Positive growth styling
            bar_outline_colors.append('green')
            bar_outline_widths.append(2) # Set outline width
            annotation_texts.append(f"+{growth:.1f}%") # Add '+' sign for clarity
            annotation_colors.append('green')
        elif growth < 0:
            # Negative growth styling
            bar_outline_colors.append('red')
            bar_outline_widths.append(2) # Set outline width
            annotation_texts.append(f"{growth:.1f}%") # Negative sign is inherent
            annotation_colors.append('red')
        else:
            # Zero or infinite growth styling (neutral)
            bar_outline_colors.append('#2563EB') # Use the bar's fill color (or grey)
            bar_outline_widths.append(0) # No distinct outline
            annotation_texts.append(" ") # Empty text for no annotation
            annotation_colors.append('grey') # Neutral color

    # Add 2024 bars with the prepared conditional outlines
    fig.add_trace(go.Bar(
        x=categories,
        y=year_2024_values,
        name='2024',
        marker_color='#2563EB', # Medium Blue fill color
        marker_line_color=bar_outline_colors, # Apply the list of outline colors
        marker_line_width=bar_outline_widths  # Apply the list of outline widths
    ))

    # Add 2025 projected bars if that year is selected in the sidebar
    if selected_year == "2025 (Projected)":
        # 'current_totals' holds the correctly calculated/filtered projected data
        proj_data_source = current_totals
        # Get the projected values for the chart
        year_2025_values_chart = [
            proj_data_source['total_base_salary'], proj_data_source['total_premiums'], proj_data_source['total_bonuses'],
            proj_data_source['total_social_contributions'], proj_data_source['total_ltips'], proj_data_source['total_cost']
        ]
        # Add the 2025 bar trace
        fig.add_trace(go.Bar(
            x=categories,
            y=year_2025_values_chart,
            name='2025 (Projected)',
            marker_color='#1E3A8A' # Dark Blue
            # No special outline needed unless comparing 2024 vs 2025
        ))

    # Add text annotations (percentage change) above the 2024 bars
    for i, category in enumerate(categories):
        val_2024 = year_2024_values[i] # Y position based on 2024 bar height
        text = annotation_texts[i]     # Get the prepared text (e.g., "+2.3%")
        color = annotation_colors[i]   # Get the prepared color (green/red/grey)

        # Only add an annotation if the text is not blank
        if text.strip():
            fig.add_annotation(
                x=category, # X position based on category
                y=val_2024, # Y position based on bar height
                text=text,  # The formatted percentage string
                showarrow=False, # Don't show the default annotation arrow
                font=dict(
                    color=color, # Apply the determined color
                    size=11      # Adjust font size if needed
                ),
                yshift=15 # Shift text vertically above the bar (adjust as needed)
            )

    # Configure the overall chart layout
    fig.update_layout(
        title="Comparative Cost Analysis: 2023 vs. 2024 Trends", # Informative title
        xaxis_title="Cost Category",
        yaxis_title="Amount ($)",
        legend_title="Year",
        barmode='group', # Group bars by category
        height=550,      # Set chart height
        margin=dict(t=90, b=40, l=40, r=40) # Adjust top margin for annotations
    )

    # Format the Y-axis ticks as currency ($)
    fig.update_yaxes(tickprefix="$", tickformat=",")

    # Display the Plotly chart in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)

    # --- Display Growth Rates Table using st.dataframe with Styling ---

    # Create the dictionary for the DataFrame - STORE RAW NUMBERS for styling
    growth_data_raw = {
        'Category': categories,
        '2023 Amount': year_2023_values, # Store raw numbers
        '2024 Amount': year_2024_values, # Store raw numbers
        '2023-2024 Growth': growth_rates # Store raw numerical growth rates
    }

    # Add 2025 columns to the raw data dictionary if applicable
    if selected_year == "2025 (Projected)":
        # Use the same projection data source as the chart
        proj_data_source = current_totals
        # Get raw values for the table
        year_2025_values_table = [
            proj_data_source['total_base_salary'], proj_data_source['total_premiums'], proj_data_source['total_bonuses'],
            proj_data_source['total_social_contributions'], proj_data_source['total_ltips'], proj_data_source['total_cost']
        ]
        # Calculate 2024-2025 growth rates using the correct projected data
        keys_for_growth = ['total_base_salary', 'total_premiums', 'total_bonuses',
                           'total_social_contributions', 'total_ltips', 'total_cost']
        growth_2025 = []
        for key in keys_for_growth:
            # Use compare_2024 as the base for 2024->2025 growth calculation
            rate_2025 = safe_division(proj_data_source[key], compare_2024[key])
            growth_2025.append(round(rate_2025, 1) if rate_2025 != float('inf') else float('inf'))

        # Add raw 2025 data and growth to the dictionary
        growth_data_raw['2025 Amount (Projected)'] = year_2025_values_table
        growth_data_raw['2024-2025 Growth (Projected)'] = growth_2025

    # Create the Pandas DataFrame from the raw data dictionary
    growth_df_raw = pd.DataFrame(growth_data_raw)

    # Define columns to apply specific styling/formatting
    amount_cols = [col for col in growth_df_raw.columns if 'Amount' in col]
    growth_cols = [col for col in growth_df_raw.columns if 'Growth' in col]

    # Apply styling and formatting using Pandas Styler
    styled_df = growth_df_raw.style \
        .format(format_currency, subset=amount_cols) \
        .format(format_growth_display, subset=growth_cols) \
        .apply(lambda x: x.map(color_growth), subset=growth_cols) # Apply color based on original value using apply/map


    # Display the styled DataFrame in Streamlit
    # use_container_width=True makes the table expand to the column width
    st.dataframe(styled_df, use_container_width=True)

# --- End of Tab 2 ---

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
    subtab1, subtab2 = st.tabs(["Department Analysis", "Band Distribution"])

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

#     # Hours Analysis
#     with subtab3:
#         # Planned vs Actual Hours
#         if selected_year != "2025 (Projected)":
#             st.markdown("### Planned vs. Actual Hours")

#             hours_data = {
#                 'Category': ['Planned Hours', 'Actual Hours'],
#                 'Hours': [
#                     analysis_totals['total_planned_hours'],
#                     analysis_totals['total_actual_hours']
#                 ]
#             }
#             hours_df = pd.DataFrame(hours_data)

#             fig = px.bar(
#                 hours_df,
#                 x='Category',
#                 y='Hours',
#                 title='Planned vs. Actual Hours',
#                 color='Category',
#                 color_discrete_sequence=['#93C5FD', '#2563EB']
#             )

#             st.plotly_chart(fig, use_container_width=True)

#             # Hours difference analysis
#             hours_diff = analysis_totals['total_actual_hours'] - analysis_totals['total_planned_hours']
#             hours_diff_percent = (hours_diff / analysis_totals['total_planned_hours'] * 100).round(2)

#             col1, col2 = st.columns(2)

#             with col1:
#                 st.metric(
#                     "Total Hours Difference",
#                     f"{hours_diff:.0f} hours",
#                     f"{hours_diff_percent}%"
#                 )

#             with col2:
#                 st.metric(
#                     "Overtime Hours",
#                     f"{analysis_totals['total_overtime_hours']:.0f} hours",
#                     help="Total overtime hours calculated as sum of (actual - planned) where positive"
#                 )

#             # Display absence analysis
#             st.markdown("### Absence Analysis")

#             absence_data = {
#                 'Category': ['Absence Cost', 'As % of Base Salary'],
#                 'Value': [
#                     format_currency(analysis_totals['total_absence_costs']),
#                     f"{(analysis_totals['total_absence_costs'] / analysis_totals['total_base_salary'] * 100).round(2)}%"
#                 ]
#             }
#             absence_df = pd.DataFrame(absence_data)

#             st.table(absence_df)

# Footer
st.markdown("---")
st.markdown("**Cost of Labor Dashboard** | Developed for HR Analytics")

# # Add download links for sample data
# st.sidebar.markdown("---")
# st.sidebar.markdown("### Download Sample Data Templates")

# # Create download buttons for sample CSVs
# if st.sidebar.button("Download Employee Data Template"):
#     csv = df_2023.to_csv(index=False)
#     st.sidebar.download_button(
#         label="Click to Download",
#         data=csv,
#         file_name="employee_data_template.csv",
#         mime="text/csv"
#     )

# if st.sidebar.button("Download Band Averages Template"):
#     csv = band_avg_df.to_csv(index=False)
#     st.sidebar.download_button(
#         label="Click to Download",
#         data=csv,
#         file_name="band_averages_template.csv",
#         mime="text/csv"
#     )
