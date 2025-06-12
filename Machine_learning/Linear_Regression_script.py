import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler , OneHotEncoder , LabelEncoder  , OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
from sklearn.model_selection import  KFold , cross_val_score

#importing the data
def data_entry (path):
    #importing the data
    df = pd.read_csv(path)
    #visualizing the data
    print(df.head())
    return df

#analysing the data
def data_analyse(df):
    print("Shape o fthe data : " , df.shape)
    print(df.info())
    print('Statistical Description : ' , df.describe())

def null_values(df):
    print("Null values : \n" , df.isnull().sum())
    df = df.dropna()
    print("After removing Null values : \n", df.isnull().sum())
    return df


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
    plt.tight_layout()
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
        df[col] = encoder.fit_transform(df[col])

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
        df[col] = encoder.fit_transform(df[col])

    return df


#defining the x and y
def define_x_y (df , target_var):
    x = df.drop(target_var , axis = 1)
    y = df[target_var]
    return x , y

def spliting_data(x,y):
    x_train  , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , shuffle= True  , random_state=42)
    print('x_train - ', x_train.shape , 'x_test - ', x_test.shape)
    print('y_train - ', y_train.shape , 'y_test - ', y_test.shape)
    x_train_val  , x_val , y_train_val , y_val = train_test_split(x_train , y_train, test_size = 0.2 , shuffle= True  , random_state=42)
    print('x_train_val - ', x_train_val.shape , 'x_val  - ' , x_val.shape)
    print('y_train_val - ', y_train_val.shape, 'y_val - ', y_val.shape)
    return x_train_val  , x_val , y_train_val , y_val , x_train  , x_test , y_train , y_test

def standardizing_data(x_train_val , x_val):
    sk = StandardScaler()
    sk_x_train_val = sk.fit_transform(x_train_val)
    print(sk_x_train_val)
    sk_x_val = sk.transform(x_val)
    print(sk_x_val)
    return sk_x_train_val , sk_x_val

def Model (sk_x_train_val , y_train_val):
    Regressor = LinearRegression()
    Regressor.fit(sk_x_train_val, y_train_val)
    return Regressor

def make_predictions(Regressor, sk_x_val):
    y_pred = Regressor.predict(sk_x_val)
    print("y_pred : " , y_pred)
    return y_pred

def visualize_line(y_val , y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred, color='blue')
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.show()

def check_performance (y_val , y_pred):
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R²): {r2}")

def cross_validation(x,y,num_fold):
    kf = KFold(n_splits = num_fold , shuffle=True , random_state=42 )
    # x, y = x.dropna(), y.loc[x.index]
    model = LinearRegression()
    cross_val_results = cross_val_score(model , x , y , cv=kf , scoring = 'r2')
    print("Cross-Validation Results (r2 Scores) : " , cross_val_results)
    print("Mean R2 Score : " , cross_val_results.mean() )

def cross_val_defined(x_train , y_train):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # initialise the list to store the performance
    all_mse = []
    all_mae = []
    all_r2 = []

    # defining the fold
    for train_index, test_index in kf.split(x_train):
        x_train_val_fold, x_val_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_val_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # scaling
        sk = StandardScaler()
        sk_x_train_val_fold = sk.fit_transform(x_train_val_fold)
        sk_x_val_fold = sk.transform(x_val_fold)

        # fit the model
        regressor = LinearRegression()
        regressor.fit(sk_x_train_val_fold, y_train_val_fold)

        # make the predictions
        y_pred_fold= regressor.predict(sk_x_val_fold)

        # check performance
        mse = mean_squared_error(y_val_fold, y_pred_fold)
        mae = mean_absolute_error(y_val_fold, y_pred_fold)
        r2 = r2_score(y_val_fold, y_pred_fold)

        # append the performance
        all_mse.append(mse)
        all_mae.append(mae)
        all_r2.append(r2)

    #printing the lists of the values are folds
    print('all_mse', all_mse)
    print('all_mae', all_mae)
    print('all_r2', all_r2)
    # take the average of performance
    avg_mse = np.mean(all_mse)
    avg_mae = np.mean(all_mae)
    avg_r2 = np.mean(all_r2)
    print('avg_mse' , avg_mse)
    print('avg_mae', avg_mae)
    print('avg_r2', avg_r2)

    return all_mse , all_mae , all_r2


# def plot_cross_val(all_mse , all_mae , all_r2):
#     import matplotlib.pyplot as plt

def plot_cross_val_metrics(all_mse, all_mae, all_r2):
    folds = range(1, len(all_mse) + 1)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot MSE
    axes[0].plot(folds, all_mse, marker='o', linestyle='-', color='r', label="MSE")
    axes[0].set_xlabel("Fold Number")
    axes[0].set_ylabel("Mean Squared Error")
    axes[0].set_title("MSE per Fold")
    axes[0].grid(True)
    axes[0].legend()

    # Plot MAE
    axes[1].plot(folds, all_mae, marker='s', linestyle='-', color='g', label="MAE")
    axes[1].set_xlabel("Fold Number")
    axes[1].set_ylabel("Mean Absolute Error")
    axes[1].set_title("MAE per Fold")
    axes[1].grid(True)
    axes[1].legend()

    # Plot R2 Score
    axes[2].plot(folds, all_r2, marker='^', linestyle='-', color='b', label="R² Score")
    axes[2].set_xlabel("Fold Number")
    axes[2].set_ylabel("R² Score")
    axes[2].set_title("R² Score per Fold")
    axes[2].grid(True)
    axes[2].legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


def main():
    path = "C:/Users/ankit/Desktop/IBAB learning/housing model linear/housing.csv"
    target_var = "median_house_value"

    #load data
    df = data_entry(path)

    #analyse data
    data_analyse(df)

    #Null values
    null_values(df)

    #encoding
    df = One_Hot_encoding(df)
    #df = Label_encoding(df)
    #df = Ordinal_encoding(df)

    #define x and y
    x , y = define_x_y(df, target_var)


    #splitting data into train , test and validation
    x_train_val  , x_val , y_train_val , y_val , x_train  , x_test , y_train , y_test=spliting_data(x, y)

    #feature scaling
    sk_x_train_val, sk_x_val = standardizing_data(x_train_val, x_val)

    #defining the model
    Regressor = Model(sk_x_train_val, y_train_val)

    #making the predictions
    y_pred = make_predictions(Regressor,sk_x_val)

    #visualization
    visualize_line(y_val , y_pred)

    #checking the performance
    cross_val_defined(x_train, y_train)
    check_performance(y_val, y_pred)
    cross_validation(sk_x_train_val, y_train_val, 10)
    all_mse , all_mae , all_r2 = cross_val_defined(x_train, y_train)
    plot_cross_val_metrics(all_mse, all_mae, all_r2)
    histogram(df)
    boxplot(df)
    plt.close()

if __name__ == "__main__":
    main()



















