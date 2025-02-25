import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
        background-color: #F8F9FA;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
    }
    .risk-table {
        font-size: 0.9rem;
    }
    .action-card {
        background-color: #EBF5FF;
        border: 1px solid #BFDBFE;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #DC2626;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Employee Attrition Risk Dashboard")

# Mock data - in a real scenario, this would come from your model predictions
division_data = pd.DataFrame([
    {"name": "Technology", "attrition": 24, "headcount": 180, "attritionRate": 13.3},
    {"name": "Operations", "attrition": 18, "headcount": 210, "attritionRate": 8.6},
    {"name": "Marketing", "attrition": 12, "headcount": 85, "attritionRate": 14.1},
    {"name": "Finance", "attrition": 8, "headcount": 120, "attritionRate": 6.7},
    {"name": "HR", "attrition": 6, "headcount": 45, "attritionRate": 13.3},
    {"name": "Sales", "attrition": 22, "headcount": 130, "attritionRate": 16.9}
])

band_data = pd.DataFrame([
    {"name": "BZ", "attrition": 32, "headcount": 210, "attritionRate": 15.2},
    {"name": "BII", "attrition": 28, "headcount": 245, "attritionRate": 11.4},
    {"name": "BIII", "attrition": 15, "headcount": 180, "attritionRate": 8.3},
    {"name": "BIV", "attrition": 8, "headcount": 95, "attritionRate": 8.4},
    {"name": "BV", "attrition": 7, "headcount": 60, "attritionRate": 11.7}
])

age_group_data = pd.DataFrame([
    {"name": "20-30", "attrition": 34, "headcount": 190, "attritionRate": 17.9},
    {"name": "31-40", "attrition": 25, "headcount": 280, "attritionRate": 8.9},
    {"name": "41-50", "attrition": 18, "headcount": 210, "attritionRate": 8.6},
    {"name": "51+", "attrition": 13, "headcount": 110, "attritionRate": 11.8}
])

feature_importance_data = pd.DataFrame([
    {"name": "Career Velocity", "value": 0.18},
    {"name": "Employment Complexity", "value": 0.15},
    {"name": "Career Growth Score", "value": 0.12},
    {"name": "Role Stability", "value": 0.11},
    {"name": "Division Transfer Rate", "value": 0.09}
])

high_risk_employees = pd.DataFrame([
    {"id": "E1092", "risk": 0.89, "division": "Sales", "band": "BZ", "key_factors": "Low Role Stability, High Career Velocity"},
    {"id": "E2385", "risk": 0.87, "division": "Technology", "band": "BII", "key_factors": "High Employment Complexity, Low Career Growth Score"},
    {"id": "E4721", "risk": 0.84, "division": "Marketing", "band": "BZ", "key_factors": "Low Division Transfer Rate, Low Role Stability"},
    {"id": "E3056", "risk": 0.82, "division": "Technology", "band": "BIII", "key_factors": "Low Career Growth Score, High Employment Complexity"},
    {"id": "E5193", "risk": 0.79, "division": "Operations", "band": "BII", "key_factors": "Low Role Stability, Low Career Growth Score"}
])

pie_data = pd.DataFrame([
    {"name": "High Risk (>75%)", "value": 28},
    {"name": "Medium Risk (50-75%)", "value": 47},
    {"name": "Low Risk (<50%)", "value": 715}
])

# Main layout - 3 metrics in first row
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<h3>Total Predicted Attrition</h3>', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">90</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">employees predicted to leave</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Out of 790 total employees (11.4% attrition rate)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<h3>Attrition Risk Distribution</h3>', unsafe_allow_html=True)
    
    # Create pie chart using plotly
    fig = px.pie(
        pie_data, 
        values='value', 
        names='name',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.3
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<h3>Top Risk Factors</h3>', unsafe_allow_html=True)
    
    # Create horizontal bar chart for feature importance
    fig = px.bar(
        feature_importance_data,
        x='value',
        y='name',
        orientation='h',
        color_discrete_sequence=['#0088FE'],
        labels={'value': 'Importance', 'name': ''}
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Second row with attrition by category and high risk employees
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<h3>Attrition by Category</h3>', unsafe_allow_html=True)
    
    # Category selection tabs
    category = st.radio(
        "Select Category View:",
        ["Division", "Band", "Age Group"],
        horizontal=True,
        key="category_selector"
    )
    
    # Select data based on category
    if category == "Division":
        plot_data = division_data
    elif category == "Band":
        plot_data = band_data
    else:
        plot_data = age_group_data
    
    # Create a double-axis bar chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=plot_data['name'],
            y=plot_data['attrition'],
            name="Predicted Attrition",
            marker_color='#8884d8'
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=plot_data['name'],
            y=plot_data['attritionRate'],
            name="Attrition Rate (%)",
            marker_color='#82ca9d',
            mode='lines+markers'
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=10, b=0, l=0, r=0),
        height=300
    )
    
    fig.update_yaxes(title_text="Predicted Attrition", secondary_y=False)
    fig.update_yaxes(title_text="Attrition Rate (%)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<h3>Top 5 High-Risk Employees</h3>', unsafe_allow_html=True)
    
    # Format the risk column to show risk bars
    def format_risk_bar(val):
        color = 'red' if val > 0.85 else 'orange'
        width = int(val * 100)
        return f'<div style="display: flex; align-items: center;">\
                <span style="margin-right: 10px;">{val:.2f}</span>\
                <div style="background-color: #eee; width: 100px; height: 10px; border-radius: 5px;">\
                    <div style="background-color: {color}; width: {width}px; height: 10px; border-radius: 5px;"></div>\
                </div>\
                </div>'
    
    # Apply formatting
    formatted_df = high_risk_employees.copy()
    formatted_df['risk_display'] = formatted_df['risk'].apply(format_risk_bar)
    
    # Display the dataframe with formatting for the risk column
    st.write(
        formatted_df[['id', 'risk_display', 'division', 'band', 'key_factors']]
        .rename(columns={
            'id': 'ID', 
            'risk_display': 'Risk Score', 
            'division': 'Division', 
            'band': 'Band', 
            'key_factors': 'Key Factors'
        }),
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Third row with recommended actions
st.markdown('<div class="metric-card">', unsafe_allow_html=True)
st.markdown('<h3>Recommended Actions</h3>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="action-card">', unsafe_allow_html=True)
    st.markdown('<h4 style="color: #1E40AF;">Sales Division</h4>', unsafe_allow_html=True)
    st.markdown('<p>Highest attrition risk. Focus on improving Role Stability and managing Career Velocity expectations.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="action-card">', unsafe_allow_html=True)
    st.markdown('<h4 style="color: #1E40AF;">BZ Band Employees</h4>', unsafe_allow_html=True)
    st.markdown('<p>Develop clear career progression paths and reduce employment complexity for early-career staff.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="action-card">', unsafe_allow_html=True)
    st.markdown('<h4 style="color: #1E40AF;">Technology Team</h4>', unsafe_allow_html=True)
    st.markdown('<p>Review Career Growth Scores and implement targeted retention plans for high-risk, high-value contributors.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Optional: Add a sidebar with filters (for a more complete dashboard)
with st.sidebar:
    st.header("Filters")
    st.markdown("##### Apply filters to refine your view")
    
    # These filters don't actually filter the data in this demo
    # but would be connected in a real application
    divisions = st.multiselect(
        "Division",
        options=division_data['name'].tolist(),
        default=division_data['name'].tolist()
    )
    
    bands = st.multiselect(
        "Band",
        options=band_data['name'].tolist(),
        default=band_data['name'].tolist()
    )
    
    risk_threshold = st.slider(
        "Risk Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        format="%.2f"
    )
    
    st.caption("Note: This is a demo dashboard with static data")
