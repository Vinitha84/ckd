# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv('kidney_disease.csv')

# Print the columns to check their names
print("Columns in the dataset:", data.columns)

# Preprocess the data
data = data.dropna()  # Drop rows with missing values

# Replace 'class' with the actual name of your target column
X = data.drop('CKD', axis=1)  # Adjust this line based on the actual column name
y = data['CKD']  # Adjust this line based on the actual column name

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'ckd_model.pkl')