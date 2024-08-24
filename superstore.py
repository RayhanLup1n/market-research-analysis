#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


# Load the data
df = pd.read_excel("Market Research.xlsx")


# In[3]:


# Display basic information about the dataset
df.info()
print("Number of null Values: ")
print(df.isna().sum())
print(f"Number of Duplicated Values: {df.duplicated().sum()}")


# In[60]:


df.columns


# In[61]:


df.describe()


# In[62]:


df.describe(include='object')


# In[4]:


# Rename and drop unnecessary columns for clarity
new_df = df.drop(columns=['Row ID', 'Country']).rename(columns={
    'Order ID': 'OrderID',
    'Order Date': 'OrderDate',
    'Ship Date': 'ShipDate',
    'Ship Mode': 'ShipMode',
    'Customer ID': 'CustID',
    'Customer Name': 'CustName',
    'Postal Code': 'Postal',
    'Product ID': 'ProductID',
    'Sub-Category': 'SubCategory',
    'Product Name': 'ProductName'
})


# In[5]:


# Calculate TotalCost and CostPerUnit
new_df['TotalCost'] = new_df['Sales'] - new_df['Profit']
new_df['CostPerUnit'] = new_df['TotalCost'] / new_df['Quantity']


# In[6]:


# Group by Segment to analyze performance
seg_df = new_df.groupby(by='Segment').agg({
    'Profit': 'sum',
    'Sales': 'sum',
    'Quantity': 'sum',
    'TotalCost': 'sum'
}).sort_values(by=['Profit', 'Sales', 'Quantity'], ascending=False)

seg_df


# In[7]:


# Group by City to analyze performance
city_df = new_df.groupby(by='City').agg({
    'Profit': 'sum',
    'Sales': 'sum',
    'Quantity': 'sum',
    'TotalCost': 'sum'
}).sort_values(by=['Profit', 'Sales', 'Quantity'], ascending=False)

city_df


# In[8]:


# Group by ProductName, Category, and SubCategory
cpu_df = new_df.groupby(by=['ProductName', 'Category', 'SubCategory']).agg({
    'CostPerUnit': 'sum',
    'TotalCost': 'sum',
    'Quantity': 'sum'
}).sort_values(by=['CostPerUnit', 'Quantity', 'TotalCost'], ascending=False)

cpu_df


# In[9]:


# Group by Category and City for further analysis
categ_city_df = new_df.groupby(by=['Category', 'City']).agg({
    'Profit': 'sum',
    'Quantity': 'sum',
    'TotalCost': 'sum'
}).sort_values(by=['Profit', 'TotalCost'], ascending=False)

categ_city_df


# In[11]:


# Analyze top 50 profitable products
profit_df = new_df.sort_values(by=['Profit'], ascending=False).head(50)
profit_df


# In[12]:


# Add Year and Month columns based on OrderDate
new_df['Year'] = new_df['OrderDate'].dt.year
new_df['Month'] = new_df['OrderDate'].dt.month


# In[13]:


# Visualize Sales and Profit over time by Category
plt.figure(figsize=(14,8))
sns.lineplot(data=new_df, x='Year', y='Sales', hue='Category')
plt.title('Sales Performance per Year')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.legend(title='Category')
plt.show()


# In[14]:


plt.figure(figsize=(14,8))
sns.lineplot(data=new_df, x='Year', y='Profit', hue='Category')
plt.title('Profit per Year by Categories')
plt.xlabel('Year')
plt.ylabel('Profit')
plt.legend(title='Category')
plt.show()


# In[15]:


# Analyze top 10 cities by Sales and Profit for each Category and Segment
grouped_df = new_df.groupby(['City', 'Category', 'Segment']).agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()

grouped_df['Rank'] = grouped_df.groupby(['Category', 'Segment'])['Sales'].rank(method='first', ascending=False)
top10_cities = grouped_df[grouped_df['Rank'] <= 10].sort_values(by=['Category', 'Segment', 'Rank'])


# In[16]:


# Visualize top 10 cities by Sales
plt.figure(figsize=(14, 10))
sns.barplot(x='Sales', y='City', hue='Segment', data=top10_cities, ci=None)
plt.title('Top 10 Cities by Sales for Each Category and Segment')
plt.xlabel('Total Sales')
plt.ylabel('City')
plt.legend(title='Segment')
plt.show()


# In[17]:


# Visualize top 10 cities by Sales and Profit
fig, axes = plt.subplots(2, 1, figsize=(14, 16))
sns.barplot(ax=axes[0], x='Sales', y='City', hue='Segment', data=top10_cities, ci=None)
axes[0].set_title('Top 10 Cities by Sales for Each Category and Segment')
axes[0].set_xlabel('Total Sales')
axes[0].set_ylabel('City')

sns.barplot(ax=axes[1], x='Profit', y='City', hue='Segment', data=top10_cities, ci=None)
axes[1].set_title('Top 10 Cities by Profit for Each Category and Segment')
axes[1].set_xlabel('Total Profit')
axes[1].set_ylabel('City')

plt.tight_layout()
plt.show()


# In[18]:


# Encode categorical features for modeling
encode = LabelEncoder()
object_coll = new_df.select_dtypes(include='object').columns

for col in object_coll:
    new_df[col] = encode.fit_transform(new_df[col])


# In[19]:


# Analyze correlations with Profit
profit_corr = new_df.corr()['Profit'].sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=profit_corr.index, y=profit_corr.values, palette='coolwarm')
plt.xticks(rotation=45, ha='right')
plt.title('Correlation of Profit with Other Variables')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Features')
plt.show()


# In[20]:


profit_corr_df = new_df.corr()[['Profit']].sort_values(by='Profit', ascending=False)

plt.figure(figsize=(10, 8))
sns.heatmap(data=profit_corr_df, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', 
            annot_kws={"size": 10}, cbar_kws={'shrink': 0.75})
plt.xticks(rotation=45, ha='right')
plt.title('Correlation of Profit with Other Variables')
plt.show()


# In[21]:


# Prepare data for model training
scaler = MinMaxScaler()

X = new_df[['Category', 'City', 'Region', 'Segment', 'SubCategory']]
y = (new_df['Profit'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


# In[22]:


# Scale the features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[23]:


# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[24]:


# Compare actual vs predicted results
comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

comparison_df = pd.concat([pd.DataFrame(X_test_scaled, columns=X_test.columns), comparison_df.reset_index(drop=True)], axis=1)
comparison_df['Difference'] = comparison_df['Actual'] - comparison_df['Predicted']
print(comparison_df.head())


# In[25]:


# Analyze where the model made errors
errors_df = comparison_df[comparison_df['Difference'] != 0]
print(errors_df)


# In[103]:


comparison_df.shape


# In[26]:


# Visualize actual vs predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=comparison_df.index, y='Actual', data=comparison_df, label='Actual', color='blue')
sns.scatterplot(x=comparison_df.index, y='Predicted', data=comparison_df, label='Predicted', color='red')
plt.legend()
plt.title('Actual vs Predicted')
plt.show()


# In[27]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# List of models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}


# In[29]:


# Initialize a dataframe to store the accuracy results
results = pd.DataFrame(columns=['Model', 'Accuracy'])

# Loop through each model
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the result in the dataframe
    new_row = pd.DataFrame({'Model': [name], 'Accuracy': [accuracy]})
    results = pd.concat([results, new_row], ignore_index=True)

# Display the results
print(results)


# In[30]:


# Perform cross-validation to further validate the results
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f'{name} Cross-Validation Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})')


# In[31]:


from sklearn.model_selection import GridSearchCV

# Hyperparameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize GridSearchCV for Random Forest
rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=rf_param_grid, 
                              cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the grid search
rf_grid_search.fit(X_train_scaled, y_train)

# Display the best parameters and best accuracy
print(f'Best Random Forest Parameters: {rf_grid_search.best_params_}')
print(f'Best Random Forest Cross-Validation Accuracy: {rf_grid_search.best_score_:.2f}')


# In[32]:


# Hyperparameter grid for Gradient Boosting
gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.05],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV for Gradient Boosting
gb_grid_search = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42),
                              param_grid=gb_param_grid, 
                              cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the grid search
gb_grid_search.fit(X_train_scaled, y_train)

# Display the best parameters and best accuracy
print(f'Best Gradient Boosting Parameters: {gb_grid_search.best_params_}')
print(f'Best Gradient Boosting Cross-Validation Accuracy: {gb_grid_search.best_score_:.2f}')


# In[34]:


# Train the final Random Forest model with the best hyperparameters
final_rf_model = RandomForestClassifier(
    bootstrap=True, 
    max_depth=20, 
    min_samples_leaf=1, 
    min_samples_split=5, 
    n_estimators=100,
    random_state=42
)
final_rf_model.fit(X_train_scaled, y_train)


# In[36]:


from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Predict with Random Forest
rf_y_pred = final_rf_model.predict(X_test_scaled)


# In[37]:


# Evaluate Random Forest Model
print("Random Forest Classification Report")
print(classification_report(y_test, rf_y_pred))

print("Random Forest Confusion Matrix")
print(confusion_matrix(y_test, rf_y_pred))


# In[35]:


# Train the final Gradient Boosting model with the best hyperparameters
final_gb_model = GradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=5,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
)
final_gb_model.fit(X_train_scaled, y_train)


# In[38]:


# Predict with Gradient Boosting
gb_y_pred = final_gb_model.predict(X_test_scaled)


# In[39]:


# Evaluate Gradient Boosting Model
print("Gradient Boosting Classification Report")
print(classification_report(y_test, gb_y_pred))

print("Gradient Boosting Confusion Matrix")
print(confusion_matrix(y_test, gb_y_pred))


# In[40]:


# Plot ROC Curve for both models
rf_fpr, rf_tpr, _ = roc_curve(y_test, final_rf_model.predict_proba(X_test_scaled)[:,1])
gb_fpr, gb_tpr, _ = roc_curve(y_test, final_gb_model.predict_proba(X_test_scaled)[:,1])

plt.figure(figsize=(10, 6))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {auc(rf_fpr, rf_tpr):.2f})')
plt.plot(gb_fpr, gb_tpr, label=f'Gradient Boosting (AUC = {auc(gb_fpr, gb_tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[41]:


import joblib

# Save the Random Forest model
joblib.dump(final_rf_model, 'Random_Forest_Model.pkl')

# Save the Gradient Boosting model
joblib.dump(final_gb_model, 'Gradient_Boosting_Model.pkl')

