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


# Function to load and prepare data - in a real scenario, load your actual data
@st.cache_data
def load_data():
    # For demo purposes, we'll create a synthetic dataset that mimics your structure

    # In reality, you would use something like:
    # df = pd.read_csv('your_data.csv')
    # or
    # df = your_existing_dataframe

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
    # For example, lower Role_Stability often correlates with higher attrition
    mask = (df['Role_Stability'] < 0.3) & (df['Attrition'] == 0)
    flip_indices = np.random.choice(df[mask].index, size=int(len(df[mask]) * 0.3), replace=False)
    df.loc[flip_indices, 'Attrition'] = 1

    # Apply your feature engineering function - this is a simplified version
    df_engineered = df.copy()

    return df_engineered


# Function to train model and get predictions
@st.cache_resource
def get_predictions(df):
    # In a real scenario, you would use your trained model
    # For demo purposes, we'll train a simple model on our synthetic data

    # Features to use for prediction
    feature_cols = [
        'Career_Velocity', 'Role_Stability', 'Career_Growth_Score',
        'Employment_Complexity', 'Division_Transfer_Rate'
    ]

    # Prepare data for modeling
    X = df[feature_cols]
    y = df['Attrition']

    # Train a model (in practice, you'd load your pre-trained model)
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

    # For demonstration purposes, we'll simulate SHAP values
    # In practice, you would use actual SHAP values
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

# UI Components

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

# Top metrics row
col1, col2, col3 = st.columns(3)

with col1:
    total_employees = len(filtered_df)
    predicted_attrition = filtered_df['Predicted_Attrition'].sum()
    attrition_rate = (predicted_attrition / total_employees * 100) if total_employees > 0 else 0

    st.metric(
        label="Predicted Attrition",
        value=f"{int(predicted_attrition)}",
        delta=f"{attrition_rate:.1f}% of {total_employees} employees"
    )

with col2:
    risk_counts = filtered_df['Risk_Category'].value_counts().to_dict()
    high_risk = risk_counts.get('High', 0)
    medium_risk = risk_counts.get('Medium', 0)
    low_risk = risk_counts.get('Low', 0)

    st.metric(
        label="High Risk Employees",
        value=f"{high_risk}",
        delta=f"{high_risk / total_employees * 100:.1f}% of total" if total_employees > 0 else "0%"
    )

with col3:
    st.metric(
        label="Medium Risk Employees",
        value=f"{medium_risk}",
        delta=f"{medium_risk / total_employees * 100:.1f}% of total" if total_employees > 0 else "0%"
    )

# Attrition by category section
st.subheader('Attrition by Category')

# Category selector
category_options = ['Company_Division', 'Band', 'Location_Region', 'Age_Group']
category_names = ['Division', 'Band', 'Region', 'Age Group']
selected_category_index = st.radio(
    "Select Category",
    options=range(len(category_options)),
    format_func=lambda i: category_names[i],
    horizontal=True
)
selected_category = category_options[selected_category_index]

# Calculate attrition metrics by selected category
category_attrition = filtered_df.groupby(selected_category).agg(
    Total=('Corp_ID', 'count'),
    Attrition=('Predicted_Attrition', 'sum')
)
category_attrition['Rate'] = (category_attrition['Attrition'] / category_attrition['Total'] * 100).round(1)
category_attrition = category_attrition.reset_index().sort_values('Rate', ascending=False)

# Create visualization
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Bar(
        x=category_attrition[selected_category],
        y=category_attrition['Attrition'],
        name="Predicted Attrition",
        marker_color='rgba(55, 83, 109, 0.7)'
    )
)

fig.add_trace(
    go.Scatter(
        x=category_attrition[selected_category],
        y=category_attrition['Rate'],
        name="Attrition Rate (%)",
        mode='lines+markers',
        marker_color='rgba(255, 79, 38, 0.7)',
        line=dict(width=2)
    ),
    secondary_y=True
)

fig.update_layout(
    title_text=f'Attrition by {category_names[selected_category_index]}',
    xaxis_title=category_names[selected_category_index],
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    height=400
)

fig.update_yaxes(title_text="Number of Employees", secondary_y=False)
fig.update_yaxes(title_text="Attrition Rate (%)", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# Feature importance and high risk employees sections
col1, col2 = st.columns(2)

with col1:
    st.subheader('Top Attrition Risk Factors')

    # Create horizontal bar chart for feature importance
    fig = px.bar(
        feature_importance.head(5),
        x='Importance',
        y='Feature',
        orientation='h',
        labels={'Importance': 'Relative Importance', 'Feature': ''},
        color='Importance',
        color_continuous_scale=px.colors.sequential.Blues
    )

    fig.update_layout(
        height=300,
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader('Risk Distribution')

    fig = px.pie(
        filtered_df,
        names='Risk_Category',
        values='Corp_ID',
        color='Risk_Category',
        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
        hole=0.4
    )

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# High risk employees table
st.subheader('Top High-Risk Employees')

high_risk_employees = filtered_df[filtered_df['Risk_Category'] == 'High'].sort_values('Attrition_Risk',
                                                                                      ascending=False).head(10)

if not high_risk_employees.empty:
    high_risk_display = high_risk_employees[['Corp_ID', 'Attrition_Risk', 'Company_Division', 'Band', 'Key_Factors']]
    high_risk_display = high_risk_display.rename(columns={
        'Corp_ID': 'Employee ID',
        'Attrition_Risk': 'Risk Score',
        'Company_Division': 'Division',
        'Key_Factors': 'Key Risk Factors'
    })

    # Format risk score as percentage
    high_risk_display['Risk Score'] = high_risk_display['Risk Score'].apply(lambda x: f"{x:.1%}")

    st.dataframe(high_risk_display, use_container_width=True)
else:
    st.info("No high-risk employees match the current filters.")

# Add action recommendations section
st.subheader('Recommended Actions')

col1, col2, col3 = st.columns(3)

# Get top division with highest attrition
top_attrition_division = category_attrition.iloc[0][selected_category] if not category_attrition.empty else "N/A"
top_attrition_rate = category_attrition.iloc[0]['Rate'] if not category_attrition.empty else 0

# Get top band with highest attrition if we're looking at divisions
if selected_category != 'Band':
    band_attrition = filtered_df.groupby('Band').agg(
        Total=('Corp_ID', 'count'),
        Attrition=('Predicted_Attrition', 'sum')
    )
    band_attrition['Rate'] = (band_attrition['Attrition'] / band_attrition['Total'] * 100).round(1)
    top_attrition_band = band_attrition.sort_values('Rate', ascending=False).index[
        0] if not band_attrition.empty else "N/A"
else:
    top_attrition_band = "See left panel"

# Most common risk factors
most_common_factors = []
for factor in filtered_df['Key_Factors'].str.split(', '):
    if isinstance(factor, list):
        most_common_factors.extend(factor)

if most_common_factors:
    from collections import Counter

    top_factor = Counter(most_common_factors).most_common(1)[0][0]
else:
    top_factor = "No significant factors identified"

with col1:
    st.info(
        f"**{top_attrition_division} {category_names[selected_category_index]}**\n\n"
        f"Highest attrition risk at {top_attrition_rate:.1f}%. "
        f"Consider targeted retention interviews and career development reviews."
    )

with col2:
    st.info(
        f"**{top_attrition_band} Band Employees**\n\n"
        f"Focus on career progression clarity and address growth opportunities. "
        f"Review compensation competitiveness for this band."
    )

with col3:
    st.info(
        f"**Address: {top_factor}**\n\n"
        f"This is the most common risk factor in the current selection. "
        f"Develop organization-wide initiatives to address this specific challenge."
    )

# Footer
st.markdown("---")
st.markdown(
    "**Note:** This dashboard uses predictive modeling to identify attrition risks. "
    "All predictions should be verified with qualitative assessment before taking action."
)
