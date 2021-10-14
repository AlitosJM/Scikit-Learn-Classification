import pandas as pd

LoanData = pd.read_csv("./data/01Exercise1.csv")

LoanPrep = LoanData.copy()

# Idenyify missigin values
print(LoanPrep.isnull().sum(axis=0))

# drop rows with missing values
LoanPrep = LoanPrep.dropna()
