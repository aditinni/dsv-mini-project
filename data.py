import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
data = {
    'Gender': np.random.choice(['Male', 'Female'], 20),
    'Married': np.random.choice(['Yes', 'No'], 20),
    'Dependents': np.random.choice([0, 1, 2, 3], 20),
    'Education': np.random.choice(['Graduate', 'Not Graduate'], 20),
    'Self_Employed': np.random.choice(['Yes', 'No'], 20),
    'ApplicantIncome': np.random.randint(2000, 20000, 20),
    'CoapplicantIncome': np.random.randint(0, 10000, 20),
    'LoanAmount': np.random.randint(50, 700, 20),
    'Loan_Amount_Term': np.random.choice([360, 120, 240, 180], 20),
    'Credit_History': np.random.choice([1, 0], 20),
    'Property_Area': np.random.choice(['Urban', 'Rural', 'Semiurban'], 20),
    'Loan_Status': np.random.choice([1, 0], 20)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('./loan_data.csv', index=False)

print("CSV file 'loan_data.csv' created with more diverse data values.")
