# 📊 Market Research Analysis Dashboard

This repository contains a Streamlit application for performing market research analysis on a sample superstore dataset. The dashboard provides interactive visualizations and insights into sales and profit performance across different segments and regions.

## Overview

The Market Research Analysis Dashboard allows users to explore sales and profitability data across various segments of a superstore dataset. The dashboard includes features for filtering data by year and segment, providing a detailed analysis of sales performance, profit trends, and top-performing regions.

## Dataset

The dataset used in this project contains detailed information about orders, products, customers, and sales across different segments and regions. This dataset is essential for performing a comprehensive market analysis.

### Features of the dataset include:

- `Order Date`: The date when the order was placed.
- `Segment`: The segment to which the customer belongs (Consumer, Corporate, Home Office).
- `Sales`: The sales amount for the order.
- `Profit`: The profit generated from the order.
- `City`: The city where the order was placed.
- `Region`: The region where the order was placed.
- `Category`: The category of the product ordered.
- `Sub-Category`: The sub-category of the product ordered.

## Features

- **Interactive Filters**: 
  - Filter data by year range and segment (Consumer, Corporate, Home Office) to focus on specific subsets of the dataset.
  
- **Sales and Profit Analysis**: 
  - Visualizations include line charts showing sales and profit performance over time, as well as bar charts highlighting the top 10 cities by sales and profit.

- **Model Evaluation**: 
  - The app includes an ROC Curve analysis for evaluating the performance of machine learning models (Random Forest and Gradient Boosting) trained on the dataset to predict profitability.

## Project Structure

The project is structured as follows:

1. **Data Loading and Preprocessing**:
    - Load the dataset from Excel and CSV files.
    - Filter data based on user inputs (year range, segment).
    - Encode categorical variables for use in machine learning models.

2. **Exploratory Data Analysis (EDA)**:
    - Perform initial exploration to understand sales trends, profit margins, and customer segmentation.

3. **Modeling**:
    - Use pre-trained machine learning models (Random Forest and Gradient Boosting) to predict the profitability of orders.
    - Evaluate model performance using ROC curves.

4. **Visualization and Interaction**:
    - Build an interactive dashboard using Streamlit for users to explore sales and profit data.
    - Include filters and visualizations to make the data exploration intuitive and insightful.

## Installation

To run the Streamlit application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/market-research-analysis.git

2. Navigate to the project directory:
   ```bash
   cd market-research-analysis

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Streamlit app:
   ```bash
   streamlit run app.py

## Deployment

This application is deployed on Streamlit Community Cloud. You can access the live version of the app here.

### Usage

- Exploratory Data Analysis (EDA):
  Use the Jupyter Notebook (superstore.ipynb) to explore the dataset, understand the relationships between different features, and identify trends.

- Dashboard Interaction:
  he Streamlit app (app.py) allows users to interact with the data through various filters and visualizations, providing insights into key business metrics such as sales, profit, and performance across different segments.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please feel free to submit an issue or a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
