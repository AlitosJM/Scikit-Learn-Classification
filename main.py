import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Different classification metrics
# Accuracy, Reciver Operating Characteristic (ROC curve)/Area under curve (AUC), Confusion matrix, Classification report
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, classification_report, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
# Let's try the Ridge Regression model
from sklearn.linear_model import Ridge
# Import the LinearSVC estimator class
from sklearn.svm import LinearSVC

# Import Boston housing dataset
from sklearn.datasets import load_boston

# Fill missing values with Scikit-Learn
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

selector = 1


def main_fn(string0: str = "fn") -> None:

    # Import dataset
    heart_disease = pd.read_csv("./data/heart-disease.csv")

    # View the data
    print(string0)
    print(heart_disease.head())

    # Create X (all the feature columns)
    X = heart_disease.drop("target", axis=1)

    # Create y (the target column)
    y = heart_disease["target"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # View the data shapes
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Choose the model/estimator
    model = RandomForestClassifier(n_estimators=100)

    # default parameters
    print(model.get_params())

    model.fit(X_train, y_train)

    # Make predictions
    y_preds = model.predict(X_test)

    # This will be in the same format as y_test
    print(y_preds)

    # print(X_test.loc[1])

    print(heart_disease.loc[206])

    # Make a prediction on a single sample (has to be array)
    # model.predict(np.array(X_test.loc[206]).reshape(1, -1))

    # On the training set
    print(model.score(X_train, y_train))

    # On the test set (unseen)
    print(model.score(X_test, y_test))

    print(classification_report(y_test, y_preds))

    print(confusion_matrix(y_test, y_preds))

    print(accuracy_score(y_test, y_preds))

    # Try different numbers of estimators (n_estimators is a hyperparameter you can change)
    np.random.seed(42)
    for i in range(10, 100, 10):
        print(f"Trying model with {i} estimators...")
        model = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
        print(f"Model accruacy on test set: {model.score(X_test, y_test)}")
        print("")

    # Try different numbers of estimators with cross-validation and no cross-validation
    np.random.seed(42)
    for i in range(10, 100, 10):
        print(f"Trying model with {i} estimators...")
        model = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
        print(f"Model accruacy on test set: {model.score(X_test, y_test)}")
        print(f"Cross-validation score: {np.mean(cross_val_score(model, X, y, cv=5)) * 100}%")
        print("")

    # Save trained model to file
    pickle.dump(model, open("random_forest_model_1.pkl", "wb"))

    # Load a saved model and make a prediction on a single example
    loaded_model = pickle.load(open("random_forest_model_1.pkl", "rb"))
    print(loaded_model.score(X_test, y_test))
    # loaded_model.predict(np.array(X_test.loc[206]).reshape(1, -1))

    print("1.-", model.score(X_test, y_test))
    # Compare predictions to truth labels to evaluate the model
    y_preds = model.predict(X_test)
    print("2.-", np.mean(y_preds == y_test))
    print("3.-", accuracy_score(y_test, y_preds))

    # predict_proba() returns probabilities of a classification label
    print(model.predict_proba(X_test[:5]))

    # Let's predict() on the same data...
    print(model.predict(X_test[:5]))

    print(heart_disease["target"].value_counts())

    print(mean_absolute_error(y_test, y_preds))


def second_fn(string1: str = "fn") -> None:
    # data converted to numerical data
    # Turn the categories into numbers
    print(string1)
    categorical_features = ["Make", "Colour", "Doors"]
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer([("one_hot",
                                      one_hot,
                                      categorical_features)],
                                    remainder="passthrough")


    # Import data and drop the rows with missing labels
    data = pd.read_csv("./data/car-sales-extended.csv")

    print(data.head(), "len: " + str(len(data)), sep='\n')

    X = data.drop("Price", axis=1)
    y = data["Price"]

    transformed_X = transformer.fit_transform(X)
    # print(transformed_X)
    print(pd.DataFrame(transformed_X))
    print(X.head())

    # Another way to do it with pd.dummies...
    dummies = pd.get_dummies(data[["Make", "Colour", "Doors"]])
    print(dummies)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)

    model = RandomForestRegressor()

    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))


def third_fn(string2: str = "fn") -> None:
    # Import car sales missing data
    print(string2)
    car_sales_missing = pd.read_csv("./data/car-sales-extended-missing-data.csv")
    print(car_sales_missing.head())

    print("1", car_sales_missing.isna().sum())

    # Fill the "Make" column
    car_sales_missing["Make"].fillna("missing", inplace=True)

    # Fill the "Colour" column
    car_sales_missing["Colour"].fillna("missing", inplace=True)

    # Fill the "Odometer (KM)" column
    car_sales_missing["Odometer (KM)"].fillna(car_sales_missing["Odometer (KM)"].mean(), inplace=True)

    # Fill the "Doors" column
    car_sales_missing["Doors"].fillna(4, inplace=True)

    print("2", car_sales_missing.isna().sum())

    # Remove rows with missing Price value
    car_sales_missing.dropna(inplace=True)

    # Check our dataframe again
    print(3, car_sales_missing.isna().sum())
    print(car_sales_missing.shape, car_sales_missing.shape[0], np.size(car_sales_missing, 0), len(car_sales_missing))

    X = car_sales_missing.drop("Price", axis=1)
    y = car_sales_missing["Price"]

    categorical_features = ["Make", "Colour", "Doors"]
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer([("one_hot",
                                      one_hot,
                                      categorical_features)],
                                    remainder="passthrough")

    transformed_X = transformer.fit_transform(X)


def fourth_fn(string3: str = "fn") -> None:
    car_sales_missing = pd.read_csv("./data/car-sales-extended-missing-data.csv")
    print(car_sales_missing.head())
    print(car_sales_missing.isna().sum())

    # Drop the rows with no labels
    car_sales_missing.dropna(subset=["Price"], inplace=True)
    print(car_sales_missing.isna().sum())

    # Split into X & y
    X = car_sales_missing.drop("Price", axis=1)
    y = car_sales_missing["Price"]

    # Split data into train and test
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fill categorical values with 'missing' & numerical values with mean
    cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    door_imputer = SimpleImputer(strategy="constant", fill_value=4)
    num_imputer = SimpleImputer(strategy="mean")

    # Define columns
    cat_features = ["Make", "Colour"]
    door_feature = ["Doors"]
    num_features = ["Odometer (KM)"]

    # Create an imputer (something that fills missing data)
    imputer = ColumnTransformer([
        ("cat_imputer", cat_imputer, cat_features),
        ("door_imputer", door_imputer, door_feature),
        ("num_imputer", num_imputer, num_features)
    ])

    # Fill train and test values separately
    filled_X_train = imputer.fit_transform(X_train)
    filled_X_test = imputer.transform(X_test)

    print(filled_X_train)

    # Get our transformed data array's back into DataFrame's
    car_sales_filled_train = pd.DataFrame(filled_X_train, columns=["Make", "Colour", "Doors", "Odometer (KM)"])

    car_sales_filled_test = pd.DataFrame(filled_X_test, columns=["Make", "Colour", "Doors", "Odometer (KM)"])

    # Check missing data in training set
    print(car_sales_filled_train.isna().sum())

    car_sales_filled_train.to_csv(r'./data/car-sales-extended-no-missing-data.csv', index=False, header=True)

    # Check to see the original... still missing values
    # car_sales_missing.isna().sum()
    # Now let's one hot encode the features with the same code as before
    categorical_features = ["Make", "Colour", "Doors"]
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer([("one_hot",
                                      one_hot,
                                      categorical_features)],
                                    remainder="passthrough")

    # Fill train and test values separately
    transformed_X_train = transformer.fit_transform(car_sales_filled_train)
    transformed_X_test = transformer.transform(car_sales_filled_test)

    # Check transformed and filled X_train
    transformed_X_train.toarray()

    # Now we've transformed X, let's see if we can fit a model
    np.random.seed(42)

    model = RandomForestRegressor(n_estimators=100)

    # Make sure to use transformed (filled and one-hot encoded X data)
    model.fit(transformed_X_train, y_train)
    print(model.score(transformed_X_test, y_test))


def fiveth_fn(string4: str = "fn") -> None:
    boston = load_boston()

    print(type(boston))

    boston_df = pd.DataFrame(boston["data"], columns=boston["feature_names"])
    boston_df["target"] = pd.Series(boston["target"])

    print(boston_df.head())

    # How many samples?
    print("len", len(boston_df))
    print(np.size(boston_df, 0), boston_df.shape[0], boston_df.shape[1], boston_df.size, sep="\n")

    # Setup random seed
    np.random.seed(42)

    # Create the data
    X = boston_df.drop("target", axis=1)
    y = boston_df["target"]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Instantiate Ridge model
    model = Ridge()
    model.fit(X_train, y_train)

    # Check the score of the Ridge model on test data
    print(model.score(X_test, y_test))


def sixth_fn(string5: str = "fn") -> None:
    # Setup random seed
    np.random.seed(42)

    boston = load_boston()

    boston_df = pd.DataFrame(boston["data"], columns=boston["feature_names"])
    boston_df["target"] = pd.Series(boston["target"])

    # Create the data
    X = boston_df.drop("target", axis=1)
    y = boston_df["target"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Instatiate Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)

    # Evaluate the Random Forest Regressor
    rf.score(X_test, y_test)


def seventh_fn(string6: str = "fn") -> None:
    # Setup random seed
    np.random.seed(42)

    # Import dataset
    heart_disease = pd.read_csv("./data/heart-disease.csv")

    # Make the data
    X = heart_disease.drop("target", axis=1)
    y = heart_disease["target"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Instantiate LinearSVC
    clf = LinearSVC(max_iter=10000)
    clf.fit(X_train, y_train)

    # Evaluate the LinearSVC
    print("1.-", clf.score(X_test, y_test))
    # Compare predictions to truth labels to evaluate the model
    y_preds = clf.predict(X_test)
    print("2.-", np.mean(y_preds == y_test))
    print("3.-", accuracy_score(y_test, y_preds))

    # predict_proba() returns probabilities of a classification label
    clf.predict_proba(X_test[:5])




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if selector == 1:
        main_fn('dummy0')
    elif selector == 2:
        second_fn('dummy1')
    elif selector == 3:
        third_fn('dummy2')
    elif selector == 4:
        fourth_fn('dummy3')
    elif selector == 5:
        fiveth_fn('dummy4')
    elif selector == 6:
        sixth_fn('dummy5')
    elif selector == 7:
        seventh_fn('dummy6')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
