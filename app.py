import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Set seaborn style and matplotlib aesthetics
sns.set(style="whitegrid", context="talk")
plt.style.use('fivethirtyeight')

# Streamlit page configuration
st.set_page_config(page_title="Market Research Dashboard", layout="wide")

# Title of the dashboard
st.title("📊 Market Research Analysis Dashboard")

# Sidebar configuration for filters
st.sidebar.title("Filter Options")
year_range = st.sidebar.slider("Select Year Range", min_value=2014, max_value=2017, value=(2014, 2017))
selected_segments = st.sidebar.multiselect("Select Segments", options=['Consumer', 'Corporate', 'Home Office'], default=['Consumer'])

# Load trained models
with st.spinner('Loading models...'):
    final_rf_model = joblib.load('./Random_Forest_Model.pkl')
    final_gb_model = joblib.load('./Gradient_Boosting_Model.pkl')

# Load dataset
with st.spinner('Loading dataset...'):
    df = pd.read_excel("./Market Research.xlsx")

# Data Preprocessing
df['Year'] = df['Order Date'].dt.year
df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
df_filtered = df_filtered[df_filtered['Segment'].isin(selected_segments)]

# Encode categorical features
object_cols = ['Category', 'City', 'Region', 'Segment', 'SubCategory']
label_encoder = LabelEncoder()
for col in object_cols:
    if col in df_filtered.columns:
        df_filtered[col] = label_encoder.fit_transform(df_filtered[col])

# Prepare data for scaling and prediction
columns_to_use = ['Category', 'City', 'Region', 'Segment', 'SubCategory']

# Add placeholder columns if necessary
for col in columns_to_use:
    if col not in df_filtered.columns:
        df_filtered[col] = 0

X = df_filtered[columns_to_use]
y = (df_filtered['Profit'] > 0).astype(int)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Display key metrics
total_sales = df_filtered['Sales'].sum()
total_profit = df_filtered['Profit'].sum()
total_orders = df_filtered['Order ID'].nunique()

st.metric(label="Total Sales", value=f"${total_sales:,.2f}")
st.metric(label="Total Profit", value=f"${total_profit:,.2f}")
st.metric(label="Total Orders", value=f"{total_orders:,}")

# Sales and Profit Analysis
st.subheader("Sales and Profit Analysis")

col1, col2 = st.columns(2)

# Sales Over Time
with col1:
    st.write("### Sales Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x='Year', y='Sales', hue='Segment', palette="tab10")
    ax.set_title('Sales Performance per Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Sales')
    st.pyplot(fig)

# Profit Over Time
with col2:
    st.write("### Profit Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x='Year', y='Profit', hue='Segment', palette="tab10")
    ax.set_title('Profit Performance per Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Profit')
    st.pyplot(fig)

# Top 10 Cities by Sales and Profit
st.write("### Top 10 Cities by Sales and Profit")
fig, axes = plt.subplots(2, 1, figsize=(16, 18))

# Top 10 Cities by Sales
top_10_sales = df_filtered.sort_values(by='Sales', ascending=False).head(10)
sns.barplot(ax=axes[0], x='Sales', y='City', hue='Segment', data=top_10_sales, palette="coolwarm")
axes[0].set_title('Top 10 Cities by Sales')
axes[0].set_xlabel('Total Sales')
axes[0].set_ylabel('City')

# Top 10 Cities by Profit
top_10_profit = df_filtered.sort_values(by='Profit', ascending=False).head(10)
sns.barplot(ax=axes[1], x='Profit', y='City', hue='Segment', data=top_10_profit, palette="coolwarm")
axes[1].set_title('Top 10 Cities by Profit')
axes[1].set_xlabel('Total Profit')
axes[1].set_ylabel('City')

plt.tight_layout()
st.pyplot(fig)

# ROC Curve for Model Evaluation
st.write("### Model Evaluation: ROC Curve for Random Forest and Gradient Boosting")
fig, ax = plt.subplots(figsize=(12, 8))

# ROC Curve for Random Forest
rf_fpr, rf_tpr, _ = roc_curve(y, final_rf_model.predict_proba(X_scaled)[:, 1])
ax.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {auc(rf_fpr, rf_tpr):.2f})', color='blue')

# ROC Curve for Gradient Boosting
gb_fpr, gb_tpr, _ = roc_curve(y, final_gb_model.predict_proba(X_scaled)[:, 1])
ax.plot(gb_fpr, gb_tpr, label=f'Gradient Boosting (AUC = {auc(gb_fpr, gb_tpr):.2f})', color='green')

# Plot ROC Curve
ax.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
st.pyplot(fig)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Created by [Your Name] - Powered by Streamlit", unsafe_allow_html=True)
