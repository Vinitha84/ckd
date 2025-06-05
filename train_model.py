import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle


df = pd.read_csv("kidney_disease.csv")
print("✅ Column names in dataset:", df.columns.tolist())
exit()

# Load the dataset
df = pd.read_csv("kidney_disease.csv")

# Clean the dataset
df = df.dropna()
X = df[["age", "bp", "al", "sc", "hemo", "bu", "sod", "pot", "sg"]]
y = df["classification"].apply(lambda x: 1 if x == "ckd" else 0)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(log_model, f)

# Train Support Vector Machine
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

print("✅ Models trained and saved.")
