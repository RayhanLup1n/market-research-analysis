## 📊 Market Research Analysis Dashboard

This repository contains a Streamlit application for performing market research analysis on a sample superstore dataset. The dashboard provides interactive visualizations and insights into sales and profit performance across different segments and regions.

### Repository Contents

- **`app.py`**: The main Streamlit application file that serves as the entry point for the dashboard. This script includes all the logic for loading the data, preprocessing it, and generating the visualizations and metrics displayed in the dashboard.
  
- **`superstore.ipynb`**: A Jupyter Notebook that contains exploratory data analysis (EDA) on the superstore dataset. This notebook walks through various data preprocessing steps, visualizations, and insights that were used to inform the design of the Streamlit app.
  
- **`superstore.py`**: A Python script version of the Jupyter Notebook (`superstore.ipynb`). This script contains similar content but is structured for use in a standalone Python environment.

### Features

- **Interactive Filters**: Users can filter data by year range and segment (Consumer, Corporate, Home Office) to focus on specific subsets of the dataset.
  
- **Sales and Profit Analysis**: Visualizations include line charts showing sales and profit performance over time, as well as bar charts highlighting the top 10 cities by sales and profit.

- **Model Evaluation**: The app includes an ROC Curve analysis for evaluating the performance of machine learning models (Random Forest and Gradient Boosting) trained on the dataset to predict profitability.

### Installation

To run the Streamlit application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/streamlit-market-research.git
