
# EXPERIMENT 7
# Reusable End-to-End Machine Learning Pipeline

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Remove unnecessary ID column
if "Loan_ID" in df.columns:
    df = df.drop(columns=["Loan_ID"])

# 3. Remove rows with missing target values
df = df.dropna(subset=["Loan_Status"])

# 4. Separate features and target
X = df.drop(columns=["Loan_Status"]).copy()
y = df["Loan_Status"].astype(str).str.strip()

# 5. Identify numerical and categorical columns
numerical_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# 6. Create numerical preprocessing pipeline
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 7. Create categorical preprocessing pipeline
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 8. Combine preprocessing using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# 9. Create the complete reusable ML pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# 10. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 11. Train the complete pipeline
model_pipeline.fit(X_train, y_train)

print("\nPipeline training completed successfully!")

# 12. Predict the test data
y_pred = model_pipeline.predict(X_test)

# 13. Evaluate the model using accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

# 14. Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 15. Display the complete pipeline
print("\nReusable ML Pipeline:")
print(model_pipeline)

print("\nExperiment 7 completed successfully!")