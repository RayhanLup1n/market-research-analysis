import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Set the style of seaborn and figure aesthetics
sns.set(style="whitegrid", context="talk")
plt.style.use('fivethirtyeight')

# Page configuration
st.set_page_config(page_title="Market Research Dashboard", layout="wide")

# Page title with emoji
st.title("📊 Market Research Analysis Dashboard")

# Sidebar configuration
st.sidebar.title("Filter Options")
year_range = st.sidebar.slider("Select Year Range", min_value=2014, max_value=2017, value=(2014, 2017))
selected_segments = st.sidebar.multiselect("Select Segments", options=['Consumer', 'Corporate', 'Home Office'], default=['Consumer'])

# Load the trained models
with st.spinner('Loading models...'):
    final_rf_model = joblib.load('/mnt/data/Random_Forest_Model.pkl')
    final_gb_model = joblib.load('/mnt/data/Gradient_Boosting_Model.pkl')

# Load the dataset
with st.spinner('Loading dataset...'):
    df = pd.read_excel("/mnt/data/Market Research.xlsx")

# Data Preprocessing
df['Year'] = df['Order Date'].dt.year
df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
df_filtered = df_filtered[df_filtered['Segment'].isin(selected_segments)]

# Encode categorical features including 'Category'
object_coll = ['Category', 'City', 'Region', 'Segment', 'SubCategory']
encode = LabelEncoder()
for col in object_coll:
    if col in df_filtered.columns:
        df_filtered[col] = encode.fit_transform(df_filtered[col])

# Prepare the data for scaling and prediction
columns_to_use = ['Category', 'City', 'Region', 'Segment', 'SubCategory']

# Check if 'SubCategory' is missing and add a placeholder if necessary
for col in columns_to_use:
    if col not in df_filtered.columns:
        df_filtered[col] = 0  # Adding a placeholder column with constant value

X = df_filtered[columns_to_use]
y = (df_filtered['Profit'] > 0).astype(int)

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Metrics
total_sales = df_filtered['Sales'].sum()
total_profit = df_filtered['Profit'].sum()
total_orders = df_filtered['Order ID'].nunique()

st.metric(label="Total Sales", value=f"${total_sales:,.2f}")
st.metric(label="Total Profit", value=f"${total_profit:,.2f}")
st.metric(label="Total Orders", value=f"{total_orders:,}")

# Visualizations
st.subheader("Sales and Profit Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("### Sales Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x='Year', y='Sales', hue='Segment', palette="tab10")
    ax.set_title('Sales Performance per Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Sales')
    st.pyplot(fig)

with col2:
    st.write("### Profit Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x='Year', y='Profit', hue='Segment', palette="tab10")
    ax.set_title('Profit Performance per Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Profit')
    st.pyplot(fig)

st.write("### Top 10 Cities by Sales and Profit")
fig, axes = plt.subplots(2, 1, figsize=(16, 18))
top_10_sales = df_filtered.sort_values(by='Sales', ascending=False).head(10)
top_10_profit = df_filtered.sort_values(by='Profit', ascending=False).head(10)

sns.barplot(ax=axes[0], x='Sales', y='City', hue='Segment', data=top_10_sales, palette="coolwarm")
axes[0].set_title('Top 10 Cities by Sales')
axes[0].set_xlabel('Total Sales')
axes[0].set_ylabel('City')

sns.barplot(ax=axes[1], x='Profit', y='City', hue='Segment', data=top_10_profit, palette="coolwarm")
axes[1].set_title('Top 10 Cities by Profit')
axes[1].set_xlabel('Total Profit')
axes[1].set_ylabel('City')

plt.tight_layout()
st.pyplot(fig)

# ROC Curve
st.write("### Model Evaluation: ROC Curve for Random Forest and Gradient Boosting")
fig, ax = plt.subplots(figsize=(12, 8))
rf_fpr, rf_tpr, _ = roc_curve(y, final_rf_model.predict_proba(X_scaled)[:,1])
gb_fpr, gb_tpr, _ = roc_curve(y, final_gb_model.predict_proba(X_scaled)[:,1])

ax.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {auc(rf_fpr, rf_tpr):.2f})', color='blue')
ax.plot(gb_fpr, gb_tpr, label=f'Gradient Boosting (AUC = {auc(gb_fpr, gb_tpr):.2f})', color='green')
ax.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
st.pyplot(fig)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Created by [Your Name] - Powered by Streamlit", unsafe_allow_html=True)
