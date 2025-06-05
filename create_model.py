import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load the dataset
data = pd.read_csv('data/kidney_disease.csv')

# Check for missing values and fill them
data.fillna(method='ffill', inplace=True)

# Identify categorical columns (replace with your actual categorical columns)
categorical_columns = ['categorical_column1', 'categorical_column2']  # Replace with your actual categorical columns

# One-hot encode categorical variables
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_features = encoder.fit_transform(data[categorical_columns])

# Create a DataFrame with the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Combine with the rest of the data
data = pd.concat([data.drop(categorical_columns, axis=1), encoded_df], axis=1)

# Split features and target variable
X = data.drop('target_column', axis=1)  # Replace 'target_column' with your actual target column name
y = data['target_column']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'models/ckd_model.pkl')
print("Model saved as 'ckd_model.pkl'")   