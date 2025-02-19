import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import KFold, cross_val_score


# importing the data
def data_entry(path):
    # importing the data
    df = pd.read_csv(path)
    # visualizing the data
    df.head()
    return df


# analysing the data
def data_analyse(df):
    print("Shape o fthe data : ", df.shape)
    print(df.info())
    print('Statistical Description : ', df.describe())


def null_values(df):
    print(df.isnull().sum())


def boxplot(df):
    numeric_cols = [var for var in df.columns if df[var].dtype == 'float64']
    plt.figure(figsize=(20, 20))
    for var in numeric_cols:
        plt.figure(figsize=(6, 4))  # Set figure size for each boxplot
        df.boxplot(column=var)
        plt.title(var)
        plt.show()


def histogram(df):
    df.hist(bins=50, figsize=(12, 15))
    plt.show()


def One_Hot_encoding(df):
    categorical = [var for var in df.columns if df[var].dtype == 'O']
    print('\nThere are', len(categorical), 'categorical variables.')
    print("Categorical Variables:", categorical)

    if not categorical:
        print("No categorical values found.")
        return df

    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = []  # Store encoded columns
    encoded_column_names = []  # Store column names

    for col in categorical:
        print(f"\nValue counts for {col}:\n", df[col].value_counts())

        # One-hot encode the column
        encoded_array = encoder.fit_transform(df[[col]])
        encoded_columns = encoder.get_feature_names_out([col])

        # Convert to DataFrame
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=df.index)
        encoded_data.append(encoded_df)

    # Drop original categorical columns
    df = df.drop(columns=categorical).reset_index(drop=True)
    # Concatenate encoded data with the remaining DataFrame
    df = pd.concat([df] + encoded_data, axis=1)

    return df


def Label_encoding(df):
    categorical = [var for var in df.columns if df[var].dtype == 'O']
    print('\nThere are', len(categorical), 'categorical variables.')
    print("Categorical Variables:", categorical)

    if not categorical:
        print("No categorical values found.")
        return df

    encoder = LabelEncoder()

    for col in categorical:
        print(f"\nValue counts for {col}:\n", df[col].value_counts())
        df[col] = encoder.fit_transform(df[[col]])

    return df


def Ordinal_encoding(df):
    categorical = [var for var in df.columns if df[var].dtype == 'O']
    print('\nThere are', len(categorical), 'categorical variables.')
    print("Categorical Variables:", categorical)

    if not categorical:
        print("No categorical values found.")
        return df

    encoder = OrdinalEncoder()

    for col in categorical:
        print(f"\nValue counts for {col}:\n", df[col].value_counts())
        df[col] = encoder.fit_transform(df[[col]])

    return df


# defining the x and y
def define_x_y(df, target_var):
    x = df.drop(target_var, axis=1)
    y = df[target_var]
    return x, y


def spliting_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)
    print('x_train - ', x_train.shape, 'x_test - ', x_test.shape)
    print('y_train - ', y_train.shape, 'y_test - ', y_test.shape)
    x_train_val, x_val, y_train_val, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True,
                                                              random_state=42)
    print('x_train_val - ', x_train_val.shape, 'x_val  - ', x_val.shape)
    print('y_train_val - ', y_train_val.shape, 'y_val - ', y_val.shape)
    return x_train_val, x_val, y_train_val, y_val, x_train, x_test, y_train, y_test


def standardizing_data(x_train_val, x_val):
    sk = StandardScaler()
    sk_x_train_val = sk.fit_transform(x_train_val)
    print(sk_x_train_val)
    sk_x_val = sk.transform(x_val)
    print(sk_x_val)
    return sk_x_train_val, sk_x_val


def Model(sk_x_train_val, y_train_val):
    Regressor = LogisticRegression(max_iter=5000)
    Regressor.fit(sk_x_train_val, y_train_val)
    return Regressor


def make_predictions(Regressor, sk_x_val):
    y_pred = Regressor.predict(sk_x_val)
    print("y_pred: ", y_pred)
    return y_pred


def check_performance(y_val, y_pred):
    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy of the model : ", accuracy)


def cross_validation(x, y, num_fold):
    kf = KFold(n_splits=num_fold, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=5000)
    cross_val_results = cross_val_score(model, x, y, cv=kf, scoring='accuracy')
    print("Cross-Validation Results (Accuracy) : ", cross_val_results)
    print("Mean Accuracy : ", cross_val_results.mean())

def cross_val_defined(x_train , y_train):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # initialise the list to store the performance
    all_accuracy = []

    # defining the fold
    for train_index, test_index in kf.split(x_train):
        x_train_val_fold, x_val_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_val_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # scaling
        sk = StandardScaler()
        sk_x_train_val_fold = sk.fit_transform(x_train_val_fold)
        sk_x_val_fold = sk.transform(x_val_fold)

        # fit the model
        regressor = LogisticRegression()
        regressor.fit(sk_x_train_val_fold, y_train_val_fold)

        # make the predictions
        y_pred_fold= regressor.predict(sk_x_val_fold)

        # check performance
        accuracy = accuracy_score(y_val_fold, y_pred_fold)

        # append the performance
        all_accuracy.append(accuracy)

    #printing the list
    print ('all_accuracy : ' , all_accuracy)

    # take the average of performance
    avg_accuracy = np.mean(all_accuracy)
    print('avg_accuracy : ' , avg_accuracy)
    return avg_accuracy , all_accuracy

def plot_accuracy(all_accuracy):
    # Plot the accuracy vs. fold number
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), all_accuracy, marker='o', linestyle='-', color='b', label="Fold Accuracy")
    plt.xlabel("Fold Number")
    plt.ylabel("Accuracy")
    plt.title("Cross-Validation Accuracy per Fold")
    plt.xticks(range(1 , len(all_accuracy)+1))
    plt.ylim(0, 1.2)  # Accuracy is between 0 and 1
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    path = "C:/Users/ankit/Desktop/IBAB learning/Logistic_regression/cancer_classification.csv"
    target_var = "benign_0__mal_1"

    # load data
    df = data_entry(path)

    # analyse data
    data_analyse(df)

    # null values
    null_values(df)

    # encoding
    # df = One_Hot_encoding(df)
    df = Label_encoding(df)
    # df = Ordinal_encoding(df)

    # define x and y
    x, y = define_x_y(df, target_var)

    # splitting data into train , test and validation
    x_train_val, x_val, y_train_val, y_val, x_train, x_test, y_train, y_test = spliting_data(x, y)

    # feature scaling
    sk_x_train_val, sk_x_val = standardizing_data(x_train_val, x_val)

    # defining the model
    Regressor = Model(sk_x_train_val, y_train_val)

    # making the predictions
    y_pred = make_predictions(Regressor, sk_x_val)

    # checking the performance
    check_performance(y_val, y_pred)
    cross_validation(x, y, 10)
    avg_accuracy , all_accuracy = cross_val_defined(x_train, y_train)
    plot_accuracy(all_accuracy)
    histogram(df)
    boxplot(df)


if __name__ == "__main__":
    main()