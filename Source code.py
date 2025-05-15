# Upload and Load Dataset
from google.colab import files
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

uploaded = files.upload()
filename = next(iter(uploaded))
df = pd.read_csv(io.BytesIO(uploaded[filename]))

# === Data Cleaning and Preprocessing ===
# Remove duplicates, fix types
df = df.drop_duplicates()
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# Drop customerID as it's not a useful feature
df.drop('customerID', axis=1, inplace=True)

# === Exploratory Data Analysis ===

# 1. Churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# 2. Correlation Heatmap (Numerical)
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=['int64', 'float64']).corr(), annot=True, cmap='coolwarm')
plt.title("Numerical Feature Correlations")
plt.show()

# 3. MonthlyCharges vs Tenure for Churn
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Churn')
plt.title("Tenure vs MonthlyCharges Colored by Churn")
plt.show()

# === Encode Categorical Features ===
from sklearn.preprocessing import LabelEncoder

# Convert target to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical features
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)

# === Train-Test Split ===
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate Model ===
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# === Predict Churn for First 5 Test Customers ===
sample_inputs = X_test.iloc[:5]
predictions = model.predict(sample_inputs)

print("\nPredictions for First 5 Customers:")
for i, pred in enumerate(predictions):
    print(f"  Customer {i+1}: {'Churn' if pred else 'Not Churn'}")
