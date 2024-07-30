import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('loan_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Fill missing values (if any)
data = data.fillna(method='ffill')

# Encode categorical variables with all possible categories
gender_encoder = LabelEncoder().fit(['Male', 'Female'])
married_encoder = LabelEncoder().fit(['Yes', 'No'])
education_encoder = LabelEncoder().fit(['Graduate', 'Not Graduate'])
self_employed_encoder = LabelEncoder().fit(['Yes', 'No'])
property_area_encoder = LabelEncoder().fit(['Urban', 'Rural', 'Semiurban'])

data['Gender'] = gender_encoder.transform(data['Gender'])
data['Married'] = married_encoder.transform(data['Married'])
data['Education'] = education_encoder.transform(data['Education'])
data['Self_Employed'] = self_employed_encoder.transform(data['Self_Employed'])
data['Property_Area'] = property_area_encoder.transform(data['Property_Area'])

# Feature scaling
scaler = StandardScaler()
data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']] = scaler.fit_transform(data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']])

# Compute the correlation matrix
correlation_matrix = data.corr()

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# Split the data into training and testing sets
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Example new applicant data
new_applicant = {
    'Gender': 'Male',
    'Married': 'Yes',
    'Dependents': 1,
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 2000,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360,
    'Credit_History': 1.0,
    'Property_Area': 'Urban'
}

# Preprocess the new applicant data
new_applicant_df = pd.DataFrame([new_applicant])
new_applicant_df['Gender'] = gender_encoder.transform(new_applicant_df['Gender'])
new_applicant_df['Married'] = married_encoder.transform(new_applicant_df['Married'])
new_applicant_df['Education'] = education_encoder.transform(new_applicant_df['Education'])
new_applicant_df['Self_Employed'] = self_employed_encoder.transform(new_applicant_df['Self_Employed'])
new_applicant_df['Property_Area'] = property_area_encoder.transform(new_applicant_df['Property_Area'])
new_applicant_df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']] = scaler.transform(new_applicant_df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']])

# Predict loan approval
recommendation = model.predict(new_applicant_df)
print('Loan Recommendation:', 'Approved' if recommendation[0] == 1 else 'Rejected')
