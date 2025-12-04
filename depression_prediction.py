import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("student_depression_dataset.csv")  
print(df.head())

# Helper function to convert 'Sleep Duration' string to numeric
def convert_sleep_duration(duration):
    if pd.isna(duration):
        return np.nan
    duration = duration.strip().replace("'", "")
    if "Less than" in duration:
        return float(duration.split()[2])
    elif "-" in duration:
        parts = duration.split('-')
        return (float(parts[0]) + float(parts[1].split()[0])) / 2
    else:
        return np.nan

df['Sleep Duration'] = df['Sleep Duration'].apply(convert_sleep_duration)

# Encode binary categorical columns
le = LabelEncoder()
for col in ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']:
    df[col] = le.fit_transform(df[col])

# Try to map dietary habits to a scale (if values exist)
if df['Dietary Habits'].notnull().sum() > 0:
    dietary_map = {'Healthy': 2, 'Moderate': 1, 'Unhealthy': 0}
    df['Dietary Habits'] = df['Dietary Habits'].map(dietary_map)

# Convert Financial Stress to numeric (handle as string sometimes)
df['Financial Stress'] = pd.to_numeric(df['Financial Stress'], errors='coerce')

# Drop high-cardinality or irrelevant columns
df = df.drop(columns=['id', 'City', 'Profession', 'Degree'])

df.info()
df.head()

# Handle missing values
# Show missing value counts
print(df.isnull().sum())

# Impute missing numerical columns with the median
imputer = SimpleImputer(strategy="median")
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
df[num_cols] = imputer.fit_transform(df[num_cols])

# If Dietary Habits is all NaN, drop it
if df['Dietary Habits'].isnull().all():
    df = df.drop(columns=['Dietary Habits'])

# Confirm missing values handled
print(df.isnull().sum())

# Split the dataset into features and target variable
# Separate features and target
X = df.drop('Depression', axis=1)
y = df['Depression']

# Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Training model by using a Decision Tree Classifier
# Initialize the Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Convert predictions: 1 -> 'Yes', 0 -> 'No'
y_pred_label = ['Yes' if pred == 1 else 'No' for pred in y_pred]
y_test_label = ['Yes' if actual == 1 else 'No' for actual in y_test]

# Create a DataFrame for comparison
results_df = pd.DataFrame({
    'Actual': y_test_label,
    'Predicted': y_pred_label
})

# Show first 10 results
print(results_df.head(10))

# Show the first 10 predictions as Yes/No
print("Sample predictions (Yes = Depressed, No = Not Depressed):")
print(y_pred_label[:10])

# After training your model:
y_proba = clf.predict_proba(X_test)[:, 1]  # Probability of class '1' (depressed)

# Set a new threshold (e.g., 0.3 to be more sensitive)
threshold = 0.35
y_pred_new = (y_proba >= threshold).astype(int)

# Evaluate new predictions
print(confusion_matrix(y_test, y_pred_new))
print(classification_report(y_test, y_pred_new))


# Evaluate the model
#print("Accuracy:", accuracy_score(y_test, y_pred_new))
#print("\nClassification Report:\n", classification_report(y_test, y_pred_new))
#print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_new))

# Visualize Confusion Matrix

plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_new), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
