# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("loan_data.csv")  
print("First 5 rows of dataset:")
print(data.head())

# Data preprocessing
data.fillna(method='ffill', inplace=True)  

# Convert categorical values to numeric
data['Gender'] = data['Gender'].map({'Male':1, 'Female':0})
data['Married'] = data['Married'].map({'Yes':1, 'No':0})
data['Education'] = data['Education'].map({'Graduate':1, 'Not Graduate':0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1, 'No':0})
data['Loan_Status'] = data['Loan_Status'].map({'Y':1, 'N':0})

# Select features and target
X = data[['Gender', 'Married', 'Education', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = data['Loan_Status']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict test set
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Visualize confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Predict for a new applicant
# Format: [Gender, Married, Education, ApplicantIncome, LoanAmount and Credit_History]
new_applicant = [[1, 1, 1, 50000, 200, 1]] 
result = model.predict(new_applicant)

if result[0] == 1:
    print("\nNew Applicant: Loan Approved")
else:
    print("\nNew Applicant: Loan Rejected")
