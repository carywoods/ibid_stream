# Import necessary libraries (like tools) to use for our project
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import matplotlib.pyplot as plt

# Load 5% of the data from a file called 'ibid_2020.csv'
df = pd.read_csv('https://www.indybiosystems.com/datasets/ibid_2020.csv').sample(frac=0.01)

# Explore the data to understand what we have
st.write("Data Shape:", df.shape)  # How many rows and columns?
st.write("Data Types:", df.dtypes)  # What type of data is in each column?
st.write("Summary Statistics:", df.describe())  # Summary of the data
st.write("Data Info:", df.info())  # More info about the data

# Data visualization - show histograms to understand the data distribution
st.write("Histograms:")
fig, ax = plt.subplots(figsize=(10, 8))  # Create a figure and axis
df.hist(ax=ax)  # Plot histograms for each column
st.pyplot(fig)  # Display the plot

# Prepare data for training - separate features (X) and target variable (y)
X = df.drop('CHILD_ALIVE', axis=1)  # Features (drop the target variable)
y = df['CHILD_ALIVE']  # Target variable

# Preprocess data - convert categorical variables and scale numerical variables
ohe = OneHotEncoder()  # One-hot encoder for categorical variables
X_ohe = ohe.fit_transform(X.select_dtypes(include=['object']))  # Encode categorical variables
scaler = StandardScaler()  # Standard scaler for numerical variables
X_scaled = scaler.fit_transform(X.select_dtypes(include=['int64', 'float64']))  # Scale numerical variables
X_combined = pd.concat([pd.DataFrame(X_scaled), pd.DataFrame(X_ohe.toarray())], axis=1)  # Combine data

# Train the model - use Logistic Regression to predict the target variable
model = LogisticRegression()  # Logistic regression model
model.fit(X_combined, y)  # Train the model

# Evaluate the model - check how well it performs
y_pred = model.predict(X_combined)  # Predictions
accuracy = accuracy_score(y, y_pred)  # Accuracy - how often is the model correct?
classification_report_ = classification_report(y, y_pred)  # Classification report - more detailed metrics
confusion_matrix_ = confusion_matrix(y, y_pred)  # Confusion matrix - table showing predictions vs actual values

# Display results - show the metrics and reports
st.write("Accuracy:", accuracy)
st.write("Classification Report:")
st.write(classification_report_)
st.write("Confusion Matrix:")
st.write(confusion_matrix_)
