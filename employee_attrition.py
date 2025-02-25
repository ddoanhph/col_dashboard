import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Employee Attrition Risk Dashboard",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS to make it look more like the React version
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 72rem;
        background-color: #F9FAFB;
        border-radius: 0.5rem;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    h1 {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1.5rem !important;
        color: #1F2937;
    }
    h2 {
        font-size: 1.125rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        color: #1F2937;
    }
    .recommendation {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .recommendation h3 {
        font-weight: 500;
        color: #1E40AF;
        margin-bottom: 0.25rem;
        font-size: 0.95rem;
    }
    .recommendation p {
        font-size: 0.875rem;
        color: #4B5563;
    }
    .stButton button {
        border-radius: 0.25rem;
        padding: 0.25rem 0.75rem;
        font-size: 0.875rem;
    }
    .active-button {
        background-color: #2563EB !important;
        color: white !important;
    }
    .inactive-button {
        background-color: #E5E7EB !important;
        color: #1F2937 !important;
    }
    .risk-table {
        font-size: 0.875rem;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to load and prepare data
@st.cache_data
def load_data():
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 800

    divisions = ['Technology', 'Operations', 'Marketing', 'Finance', 'HR', 'Sales']
    bands = ['BZ', 'BII', 'BIII', 'BIV', 'BV']
    regions = ['EMEA', 'APAC', 'NA', 'LATAM']
    age_groups = ['20-30', '31-40', '41-50', '51+']

    data = {
        'Corp_ID': [f'E{i:04d}' for i in range(1, n_samples + 1)],
        'Attrition': np.random.choice([0, 1], size=n_samples, p=[0.89, 0.11]),
        'Company_Division': np.random.choice(divisions, size=n_samples),
        'Band': np.random.choice(bands, size=n_samples, p=[0.3, 0.3, 0.2, 0.15, 0.05]),
        'Location_Region': np.random.choice(regions, size=n_samples),
        'Age_Group': np.random.choice(age_groups, size=n_samples),
        'Career_Velocity': np.random.uniform(0.1, 2.0, size=n_samples),
        'Role_Stability': np.random.uniform(0, 1, size=n_samples),
        'Career_Growth_Score': np.random.uniform(1, 10, size=n_samples),
        'Employment_Complexity': np.random.randint(0, 5, size=n_samples),
        'Division_Transfer_Rate': np.random.uniform(0, 0.5, size=n_samples)
    }

    df = pd.DataFrame(data)

    # Ensure higher attrition rates for certain conditions to make the data more realistic
    mask = (df['Role_Stability'] < 0.3) & (df['Attrition'] == 0)
    flip_indices = np.random.choice(df[mask].index, size=int(len(df[mask]) * 0.3), replace=False)
    df.loc[flip_indices, 'Attrition'] = 1

    return df


# Function to train model and get predictions
@st.cache_resource
def get_predictions(df):
    # Features to use for prediction
    feature_cols = [
        'Career_Velocity', 'Role_Stability', 'Career_Growth_Score',
        'Employment_Complexity', 'Division_Transfer_Rate'
    ]

    # Prepare data for modeling
    X = df[feature_cols]
    y = df['Attrition']

    # Train a model
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    # Get predictions and probabilities
    df['Predicted_Attrition'] = model.predict(X)
    df['Attrition_Risk'] = model.predict_proba(X)[:, 1]

    # Add risk category
    df['Risk_Category'] = pd.cut(
        df['Attrition_Risk'],
        bins=[0, 0.5, 0.75, 1],
        labels=['Low', 'Medium', 'High']
    )

    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Simulate key factors
    def get_key_factors(row):
        factors = []
        if row['Role_Stability'] < 0.3:
            factors.append('Low Role Stability')
        if row['Career_Velocity'] > 1.5:
            factors.append('High Career Velocity')
        if row['Career_Growth_Score'] < 4:
            factors.append('Low Career Growth Score')
        if row['Employment_Complexity'] > 3:
            factors.append('High Employment Complexity')
        if row['Division_Transfer_Rate'] < 0.1:
            factors.append('Low Division Transfer Rate')

        return ', '.join(factors[:2]) if factors else 'No significant factors'

    df['Key_Factors'] = df.apply(get_key_factors, axis=1)

    return df, model, feature_importance


# Load data and get predictions
df = load_data()
df_with_predictions, model, feature_importance = get_predictions(df)

# Sidebar filters
st.sidebar.header('Filters')

# Division filter
selected_divisions = st.sidebar.multiselect(
    'Select Divisions',
    options=sorted(df['Company_Division'].unique()),
    default=sorted(df['Company_Division'].unique())
)

# Band filter
selected_bands = st.sidebar.multiselect(
    'Select Bands',
    options=sorted(df['Band'].unique()),
    default=sorted(df['Band'].unique())
)

# Region filter
selected_regions = st.sidebar.multiselect(
    'Select Regions',
    options=sorted(df['Location_Region'].unique()),
    default=sorted(df['Location_Region'].unique())
)

# Risk category filter
selected_risk = st.sidebar.multiselect(
    'Risk Category',
    options=['High', 'Medium', 'Low'],
    default=['High', 'Medium', 'Low']
)

# Apply filters
filtered_df = df_with_predictions[
    (df_with_predictions['Company_Division'].isin(selected_divisions)) &
    (df_with_predictions['Band'].isin(selected_bands)) &
    (df_with_predictions['Location_Region'].isin(selected_regions)) &
    (df_with_predictions['Risk_Category'].isin(selected_risk))
]

# Main dashboard layout
st.title('Employee Attrition Risk Dashboard')

# Top metrics row - matching React layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    total_employees = len(filtered_df)
    predicted_attrition = filtered_df['Predicted_Attrition'].sum()
    attrition_rate = (predicted_attrition / total_employees * 100) if total_employees > 0 else 0
    
    st.markdown(f"<h2>Total Predicted Attrition</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display: flex; align-items: center;">
        <span style="font-size: 2rem; font-weight: bold; color: #DC2626;">{int(predicted_attrition)}</span>
        <span style="margin-left: 0.5rem; font-size: 0.875rem; color: #6B7280;">employees predicted to leave</span>
    </div>
    <div style="margin-top: 0.5rem; font-size: 0.875rem; color: #4B5563;">
        Out of {total_employees} total employees ({attrition_rate:.1f}% attrition rate)
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("<h2>Attrition Risk Distribution</h2>", unsafe_allow_html=True)
    
    # Get risk distribution
    risk_counts = filtered_df['Risk_Category'].value_counts().reset_index()
    risk_counts.columns = ['Category', 'Count']
    
    # Prepare pie chart
    risk_colors = {'High': '#FF8042', 'Medium': '#FFBB28', 'Low': '#00C49F'}
    
    fig = px.pie(
        risk_counts, 
        names='Category', 
        values='Count',
        color='Category',
        color_discrete_map=risk_colors,
        hole=0.4
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=5, b=20),
        height=160,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("<h2>Top Risk Factors</h2>", unsafe_allow_html=True)
    
    for idx, row in feature_importance.head(5).iterrows():
        feature_name = row['Feature'].replace('_', ' ')
        importance = row['Importance']
        width_pct = min(importance * 100 * 5, 100)  # Scale for display
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
            <span style="font-size: 0.875rem;">{feature_name}</span>
            <div style="width: 6rem; background-color: #E5E7EB; border-radius: 9999px; height: 0.625rem;">
                <div style="background-color: #2563EB; height: 0.625rem; border-radius: 9999px; width: {width_pct}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Category sections - matching React layout
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Category selector similar to React buttons
    st.markdown("<h2>Attrition by Category</h2>", unsafe_allow_html=True)
    
    category_cols = st.columns(3)
    
    # Button states
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = 'Company_Division'
    
    # Display category buttons
    with category_cols[0]:
        division_btn_class = 'active-button' if st.session_state.selected_category == 'Company_Division' else 'inactive-button'
        if st.button('Division', key='div_btn', help='View attrition by division'):
            st.session_state.selected_category = 'Company_Division'
            st.rerun()
    
    with category_cols[1]:
        band_btn_class = 'active-button' if st.session_state.selected_category == 'Band' else 'inactive-button'
        if st.button('Band', key='band_btn', help='View attrition by band'):
            st.session_state.selected_category = 'Band'
            st.rerun()
    
    with category_cols[2]:
        age_btn_class = 'active-button' if st.session_state.selected_category == 'Age_Group' else 'inactive-button'
        if st.button('Age Group', key='age_btn', help='View attrition by age group'):
            st.session_state.selected_category = 'Age_Group'
            st.rerun()
    
    # Apply the button styling
    st.markdown(f"""
    <style>
        [data-testid="stButton"] > button:first-child {{
            border: none;
        }}
        div[data-testid="element-container"]:nth-child(5) button {{
            {division_btn_class.replace('!', '')}
        }}
        div[data-testid="element-container"]:nth-child(6) button {{
            {band_btn_class.replace('!', '')}
        }}
        div[data-testid="element-container"]:nth-child(7) button {{
            {age_btn_class.replace('!', '')}
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Calculate attrition metrics by selected category
    category_attrition = filtered_df.groupby(st.session_state.selected_category).agg(
        Total=('Corp_ID', 'count'),
        Attrition=('Predicted_Attrition', 'sum')
    )
    category_attrition['Rate'] = (category_attrition['Attrition'] / category_attrition['Total'] * 100).round(1)
    category_attrition = category_attrition.reset_index().sort_values('Rate', ascending=False)
    
    # Create visualization
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=category_attrition[st.session_state.selected_category],
            y=category_attrition['Attrition'],
            name="Predicted Attrition",
            marker_color='#8884d8'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=category_attrition[st.session_state.selected_category],
            y=category_attrition['Rate'],
            name="Attrition Rate (%)",
            mode='lines+markers',
            marker_color='#82ca9d',
            line=dict(width=2)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=240
    )
    
    fig.update_yaxes(title_text="Number of Employees", secondary_y=False)
    fig.update_yaxes(title_text="Attrition Rate (%)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("<h2>Top 5 High-Risk Employees</h2>", unsafe_allow_html=True)
    
    high_risk_employees = filtered_df[filtered_df['Risk_Category'] == 'High'].sort_values('Attrition_Risk', ascending=False).head(5)
    
    if not high_risk_employees.empty:
        # Create a custom table with styling to match React
        table_html = """
        <table style="width:100%; border-collapse: collapse; font-size: 0.875rem;" class="risk-table">
          <thead style="background-color: #F3F4F6;">
            <tr>
              <th style="padding: 0.5rem 0.75rem; text-align: left; font-size: 0.75rem; font-weight: 500; color: #6B7280; text-transform: uppercase;">ID</th>
              <th style="padding: 0.5rem 0.75rem; text-align: left; font-size: 0.75rem; font-weight: 500; color: #6B7280; text-transform: uppercase;">Risk Score</th>
              <th style="padding: 0.5rem 0.75rem; text-align: left; font-size: 0.75rem; font-weight: 500; color: #6B7280; text-transform: uppercase;">Division</th>
              <th style="padding: 0.5rem 0.75rem; text-align: left; font-size: 0.75rem; font-weight: 500; color: #6B7280; text-transform: uppercase;">Band</th>
              <th style="padding: 0.5rem 0.75rem; text-align: left; font-size: 0.75rem; font-weight: 500; color: #6B7280; text-transform: uppercase;">Key Factors</th>
            </tr>
          </thead>
          <tbody>
        """
        
        for i, row in enumerate(high_risk_employees.itertuples()):
            bg_color = '#F9FAFB' if i % 2 == 0 else 'white'
            risk_color = '#DC2626' if row.Attrition_Risk > 0.85 else '#F97316'
            risk_width = min(row.Attrition_Risk * 100, 100)
            
            table_html += f"""
            <tr style="background-color: {bg_color};">
              <td style="padding: 0.5rem 0.75rem; font-weight: 500; color: #111827;">{row.Corp_ID}</td>
              <td style="padding: 0.5rem 0.75rem; color: #111827;">
                <div style="display: flex; align-items: center;">
                  <span style="margin-right: 0.5rem;">{row.Attrition_Risk:.2f}</span>
                  <div style="width: 4rem; background-color: #E5E7EB; border-radius: 9999px; height: 0.5rem;">
                    <div style="background-color: {risk_color}; height: 0.5rem; border-radius: 9999px; width: {risk_width}%;"></div>
                  </div>
                </div>
              </td>
              <td style="padding: 0.5rem 0.75rem; color: #111827;">{row.Company_Division}</td>
              <td style="padding: 0.5rem 0.75rem; color: #111827;">{row.Band}</td>
              <td style="padding: 0.5rem 0.75rem; color: #111827;">{row.Key_Factors}</td>
            </tr>
            """
        
        table_html += """
          </tbody>
        </table>
        """
        
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("No high-risk employees match the current filters.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Add action recommendations section - match React style
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.markdown("<h2>Recommended Actions</h2>", unsafe_allow_html=True)

rec_cols = st.columns(3)

# Get top division with highest attrition
top_attrition_division = "Sales"
if not category_attrition.empty and st.session_state.selected_category == 'Company_Division':
    top_attrition_division = category_attrition.iloc[0][st.session_state.selected_category]

# Get top band with highest attrition
top_attrition_band = "BZ"
band_attrition = filtered_df.groupby('Band').agg(
    Total=('Corp_ID', 'count'),
    Attrition=('Predicted_Attrition', 'sum')
)
if not band_attrition.empty:
    band_attrition['Rate'] = (band_attrition['Attrition'] / band_attrition['Total'] * 100).round(1)
    band_attrition = band_attrition.sort_values('Rate', ascending=False)
    if not band_attrition.empty:
        top_attrition_band = band_attrition.index[0]

with rec_cols[0]:
    st.markdown("""
    <div class="recommendation">
        <h3>Sales Division</h3>
        <p>Highest attrition risk. Focus on improving Role Stability and managing Career Velocity expectations.</p>
    </div>
    """, unsafe_allow_html=True)

with rec_cols[1]:
    st.markdown(f"""
    <div class="recommendation">
        <h3>{top_attrition_band} Band Employees</h3>
        <p>Develop clear career progression paths and reduce employment complexity for early-career staff.</p>
    </div>
    """, unsafe_allow_html=True)

with rec_cols[2]:
    st.markdown("""
    <div class="recommendation">
        <h3>Technology Team</h3>
        <p>Review Career Growth Scores and implement targeted retention plans for high-risk, high-value contributors.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    '<p style="font-size: 0.875rem; color: #6B7280; text-align: center;">'
    'This dashboard uses predictive modeling to identify attrition risks. '
    'All predictions should be verified with qualitative assessment before taking action.'
    '</p>',
    unsafe_allow_html=True
)
