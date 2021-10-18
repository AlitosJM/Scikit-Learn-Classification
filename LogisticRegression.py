# ------------------------------------------------------------------
# Build the Logistic Regression Model
# Predict the loan approval status based on
# Gender, Marital Status, Credit History, Income and Loan Amount
# ------------------------------------------------------------------

# Import Libraries
import pandas as pd

# Read the data and Create a copy
LoanData = pd.read_csv("./data/01Exercise1.csv")
LoanPrep = LoanData.copy()


# Find out columns with missing values
LoanPrep.isnull().sum(axis=0)


# Replace Missing Values. Drop the rows.
LoanPrep = LoanPrep.dropna()

# Drop irrelevant columns based on business sense
LoanPrep = LoanPrep.drop(['gender'], axis=1)
# LoanPrep.columns and LoanPrep.keys() index object
cols1 = list(LoanPrep.columns)
# LoanPrep.columns.to_list()
cols2 = LoanPrep.columns.values.tolist()
cols3 = LoanPrep.keys()
print("m1", cols1)
print("m2", cols2)

resp = (LoanPrep.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()))

print(resp)
truthy0 = (LoanPrep.iloc[:, 2].eq(1) | LoanPrep.iloc[:, 2].eq(0)).eq(True).all()
# searching for 1s and 0s with LoanData
truthy1 = (LoanData.iloc[:, 2].notnull().eq(1) | LoanData.iloc[:, 2].notnull().eq(0)).eq(True).all()

truthy2 = LoanPrep.apply( lambda col: True if (col.eq(1) | col.eq(0)).eq(True).all() else False)
# print(1, pd.to_numeric(LoanPrep[cols1[-2]], errors='coerce').notnull().all())
# print(2, LoanPrep.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()))
# print(3, LoanPrep[cols1[-1]].name in LoanPrep.select_dtypes(include=['O', 'category']).columns)
# print(4, LoanPrep.apply(lambda s: s.name in LoanPrep.select_dtypes(include=['O', 'category']).columns))


# Create Dummy variables
print(LoanPrep.dtypes)
LoanPrep = pd.get_dummies(LoanPrep, drop_first=True)


# Normalize the data (Income and Loan Amount) Using StandardScaler
from sklearn.preprocessing import StandardScaler
scalar_ = StandardScaler()

LoanPrep['income'] = scalar_.fit_transform(LoanPrep[['income']])
# LoanPrep['loanamt'] = scalar_.fit_transform(LoanPrep[['loanamt']])
LoanPrep['loanamt'] = scalar_.transform(LoanPrep[['loanamt']])


# Create the X (Independent) and Y (Dependent) dataframes
# -------------------------------------------------------
Ys = pd.DataFrame(LoanPrep, columns=["status_Y"])
Y = LoanPrep[['status_Y']]
X = LoanPrep.drop(['status_Y'], axis=1)

# X_ = LoanPrep.iloc[:, [0, 1, 2, 3]].values
X_ = LoanPrep.iloc[:, 0:-1].values
# Y_ = LoanPrep.iloc[:, -1].values
Y_ = LoanPrep.iloc[:, 4].values


# Split the X and Y dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, test_size=0.3, random_state=1234, stratify=Y)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Build the Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
# for LoanPrep[['status_Y']]
squezzing_Y_train = Y_train.squeeze()
lr.fit(X_train, squezzing_Y_train)

# lr.fit(X_train, Y_train)


# Predict the outcome using Test data
Y_predict = lr.predict(X_test)

# Build the conufsion matrix and get the accuracy/score
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_predict)

score = lr.score(X_test, Y_test)
print("cm", cm)
print("score: ", score)
